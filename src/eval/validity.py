from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from OCC.Extend.DataExchange import write_step_file

from src.utils import cad_utils as cad
from src.utils import datasets_utils as dataset_utils


def _to_numpy(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _unflatten_face_z(face_z: np.ndarray) -> np.ndarray:
    return face_z.reshape(6 * 6, 3)


def _unflatten_edge_z(edge_z: np.ndarray) -> np.ndarray:
    return edge_z.reshape(6, 3)


@dataclass(frozen=True)
class SolidClassification:
    closed_solid: bool
    step_written: bool
    step_write_error: bool
    compilable: bool | None
    classification_error: bool


def classify_and_write_solid(
    solid: Any,
    *,
    step_path: str | Path | None,
    save_step_policy: str,
) -> SolidClassification:
    """Classify geometry before applying the independent STEP output policy."""
    if save_step_policy not in {"none", "valid", "all"}:
        raise ValueError("save_step_policy must be one of: none, valid, all.")
    compilable = False if save_step_policy == "all" else None
    if solid is None:
        return SolidClassification(False, False, False, compilable, False)

    classification_error = False
    try:
        closed_solid = bool(
            cad.is_shell_closed(solid) and cad.has_nonzero_volume(solid)
        )
    except Exception:
        closed_solid = False
        classification_error = True
    should_write = step_path is not None and (
        save_step_policy == "all"
        or (save_step_policy == "valid" and closed_solid)
    )
    if not should_write:
        return SolidClassification(
            closed_solid,
            False,
            False,
            compilable,
            classification_error,
        )
    try:
        write_step_file(solid, str(step_path))
        if not Path(step_path).is_file():
            raise RuntimeError("STEP writer returned without creating the output file.")
    except Exception:
        return SolidClassification(
            closed_solid,
            False,
            True,
            compilable,
            classification_error,
        )
    return SolidClassification(
        closed_solid,
        True,
        False,
        True if save_step_policy == "all" else None,
        classification_error,
    )


def compute_validity_batch_latentsortbrep(
    *,
    face_z: torch.Tensor,
    face_pos: torch.Tensor,
    edge_z: torch.Tensor,
    edge_bbox: torch.Tensor,
    edge_corners: torch.Tensor,
    adj_face_bin: torch.Tensor,
    num_face: torch.Tensor,
    num_edge: torch.Tensor,
    save_step_path: str | None = None,
    save_step_policy: str = "all",
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    list[str | None],
    list[bool],
    list[bool | None],
]:
    """Run the frozen single-sample reconstruction oracle for one batch."""
    if save_step_policy not in {"none", "valid", "all"}:
        raise ValueError("save_step_policy must be one of: none, valid, all.")
    if num_face.dim() == 2 and num_face.size(1) == 1:
        num_face = num_face.squeeze(1)
    if num_edge.dim() == 2 and num_edge.size(1) == 1:
        num_edge = num_edge.squeeze(1)

    batch_size = face_pos.shape[0]
    device = face_pos.device
    flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
    failure_reasons: list[str | None] = [None] * batch_size
    step_written = [False] * batch_size
    compilable: list[bool | None] = [
        False if save_step_policy == "all" else None
        for _ in range(batch_size)
    ]
    fail_counts = {
        "empty_after_filter": 0,
        "recon_exception": 0,
        "not_closed_or_zero_volume": 0,
        "not_occ_valid": 0,
        "step_write": 0,
    }

    for batch_index in range(batch_size):
        face_count = int(num_face[batch_index].item())
        edge_count = int(num_edge[batch_index].item())
        if face_count <= 0 or edge_count <= 0:
            fail_counts["empty_after_filter"] += 1
            failure_reasons[batch_index] = "empty_prediction"
            continue

        face_latent = face_z[batch_index, :face_count]
        face_boxes = face_pos[batch_index, :face_count]
        edge_latent = edge_z[batch_index, :edge_count]
        edge_boxes = edge_bbox[batch_index, :edge_count]
        corners = edge_corners[batch_index, :edge_count]
        adjacency = adj_face_bin[batch_index, :edge_count, :face_count]

        face_latent_np = _to_numpy(face_latent).astype(np.float64)
        face_boxes_np = _to_numpy(face_boxes).astype(np.float64)
        edge_latent_np = _to_numpy(edge_latent).astype(np.float64)
        edge_boxes_np = _to_numpy(edge_boxes).astype(np.float64)
        corners_np = _to_numpy(corners).astype(np.float64)
        adjacency_np = _to_numpy(adjacency).astype(np.int32)

        effective_faces = face_latent_np.shape[0]
        effective_edges = edge_latent_np.shape[0]
        if effective_faces == 0 or effective_edges == 0:
            fail_counts["empty_after_filter"] += 1
            failure_reasons[batch_index] = "empty_prediction"
            continue

        try:
            sampled_faces = []
            for flattened_control_points in face_latent_np:
                surface = cad.create_bspline_surface(
                    _unflatten_face_z(flattened_control_points)
                )
                sampled_faces.append(cad.sample_bspline_surface(surface))
            sampled_faces = np.stack(sampled_faces)

            sampled_edges = []
            for flattened_control_points in edge_latent_np:
                curve = cad.create_bspline_curve(
                    _unflatten_edge_z(flattened_control_points)
                )
                sampled_edges.append(cad.sample_bspline_curve(curve))
            sampled_edges = np.stack(sampled_edges)
        except Exception:
            fail_counts["recon_exception"] += 1
            failure_reasons[batch_index] = "reconstruction_exception"
            continue

        try:
            edge_face_list = dataset_utils.adj_matrix_to_edgeFace(adjacency_np)
            edge_mask = dataset_utils.list_to_adj_matrix(
                edge_face_list,
                effective_edges,
                effective_faces,
            )
            edge_mask = np.transpose(edge_mask, (1, 0))
            edge_mask = (1 - edge_mask).astype(bool)
            face_edge_adjacency = dataset_utils.get_faceEdge_adj(
                edge_face_list,
                effective_faces,
            )
        except Exception:
            fail_counts["recon_exception"] += 1
            failure_reasons[batch_index] = "reconstruction_exception"
            continue

        try:
            tiled_edge_boxes = np.expand_dims(edge_boxes_np, axis=0)
            tiled_edge_boxes = np.tile(
                tiled_edge_boxes,
                (face_count, 1, 1),
            )
            sampled_edges = np.expand_dims(sampled_edges, axis=0)
            sampled_edges = np.tile(
                sampled_edges,
                (face_count, 1, 1, 1),
            )
            tiled_corners = np.expand_dims(corners_np, axis=0)
            tiled_corners = np.tile(tiled_corners, (face_count, 1, 1))

            edge_vertex_boxes = []
            for boxes_per_face, edges_per_face in zip(
                tiled_edge_boxes,
                sampled_edges,
            ):
                endpoints = []
                for box, edge in zip(boxes_per_face, edges_per_face):
                    center, size = cad.compute_bbox_center_and_size(
                        box[0:3],
                        box[3:],
                    )
                    world_edge = edge * (size / 2) + center
                    endpoints.append(world_edge[[0, -1]].reshape(1, 2, 3))
                edge_vertex_boxes.append(np.vstack(endpoints))
            sampled_edges = sampled_edges[0]
        except Exception:
            fail_counts["recon_exception"] += 1
            failure_reasons[batch_index] = "reconstruction_exception"
            continue

        try:
            unique_vertices, edge_vertex_adjacency = cad.detect_shared_vertex3(
                tiled_corners,
                edge_mask,
                edge_vertex_boxes,
            )
            face_wcs, edge_wcs = cad.joint_optimize(
                sampled_faces,
                sampled_edges,
                face_boxes_np,
                unique_vertices,
                edge_vertex_adjacency,
                face_edge_adjacency,
                len(sampled_edges),
                len(sampled_faces),
                True,
            )
            solid = cad.construct_brep(
                face_wcs,
                edge_wcs,
                face_edge_adjacency,
                edge_vertex_adjacency,
            )
            step_path = None
            if save_step_path is not None and save_step_path != "":
                step_path = (
                    f"{save_step_path}_idx_{batch_index}_"
                    f"{face_count}f_{edge_count}e.step"
                )
            classification = classify_and_write_solid(
                solid,
                step_path=step_path,
                save_step_policy=save_step_policy,
            )
            step_written[batch_index] = classification.step_written
            compilable[batch_index] = classification.compilable
            fail_counts["step_write"] += int(classification.step_write_error)

            if classification.classification_error:
                fail_counts["recon_exception"] += 1
                failure_reasons[batch_index] = "reconstruction_exception"
                continue
            if not classification.closed_solid:
                fail_counts["not_closed_or_zero_volume"] += 1
                failure_reasons[batch_index] = "not_closed_or_zero_volume"
                continue
            flags[batch_index] = True
        except Exception:
            fail_counts["recon_exception"] += 1
            failure_reasons[batch_index] = "reconstruction_exception"
            continue

    stats = {
        "num_samples": torch.tensor([batch_size], device=device),
        "fails_empty_after_filter": torch.tensor(
            [fail_counts["empty_after_filter"]],
            device=device,
        ),
        "fails_recon_exception": torch.tensor(
            [fail_counts["recon_exception"]],
            device=device,
        ),
        "not_occ_valid": torch.tensor(
            [fail_counts["not_occ_valid"]],
            device=device,
        ),
        "fails_not_closed_or_zero_volume": torch.tensor(
            [fail_counts["not_closed_or_zero_volume"]],
            device=device,
        ),
        "fails_step_write": torch.tensor(
            [fail_counts["step_write"]],
            device=device,
        ),
    }
    return flags, stats, failure_reasons, step_written, compilable
