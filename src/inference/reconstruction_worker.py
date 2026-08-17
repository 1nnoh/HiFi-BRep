from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SampleGeometryInput:
    index: int
    face_z: Any
    face_pos: Any
    edge_z: Any
    edge_bbox: Any
    edge_corners: Any
    adjacency: Any


@dataclass(frozen=True)
class PreparedGeometry:
    index: int
    surface_initial: Any
    edge_wcs: Any
    face_edge_adjacency: tuple[Any, ...]
    edge_vertex_adjacency: Any


@dataclass(frozen=True)
class PreparationResult:
    index: int
    geometry: PreparedGeometry | None
    failure_reason: str | None


@dataclass(frozen=True)
class FinalizationTask:
    index: int
    surface_wcs: Any
    edge_wcs: Any
    face_edge_adjacency: tuple[Any, ...]
    edge_vertex_adjacency: Any
    step_path: str | None
    save_step_policy: str


@dataclass(frozen=True)
class FinalizationResult:
    index: int
    valid: bool
    failure_reason: str | None
    step_written: bool
    compilable: bool | None
    step_write_error: bool


def initialize_occ_worker() -> None:
    """Keep spawned OCC workers CPU-only and prevent nested thread oversubscription."""
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    import torch

    torch.set_num_threads(1)


def _align_edges(
    edge_ncs: Any,
    unique_vertices: Any,
    edge_vertex_adjacency: Any,
) -> Any:
    import numpy as np

    edge_ncs_endpoints = edge_ncs[:, [0, -1]]
    edge_vertex_endpoints = unique_vertices[edge_vertex_adjacency]
    aligned_edges = []
    for points, normalized_endpoints, vertex_endpoints in zip(
        edge_ncs,
        edge_ncs_endpoints,
        edge_vertex_endpoints,
    ):
        target_scale = np.linalg.norm(
            vertex_endpoints[0] - vertex_endpoints[1]
        )
        normalized_scale = np.linalg.norm(
            normalized_endpoints[0] - normalized_endpoints[1]
        )
        edge_scale = target_scale / (normalized_scale + 1e-9)
        updated = points * edge_scale
        scaled_endpoints = normalized_endpoints * edge_scale
        offset = vertex_endpoints - scaled_endpoints
        reverse_offset = vertex_endpoints - scaled_endpoints[::-1]
        if np.abs(reverse_offset[0] - reverse_offset[1]).mean() < np.abs(
            offset[0] - offset[1]
        ).mean():
            updated = updated[::-1]
            offset = reverse_offset
        updated = updated + offset.mean(0)[np.newaxis, np.newaxis, :]
        aligned_edges.append(updated)

    edge_wcs = np.vstack(aligned_edges)
    for index in range(len(edge_wcs)):
        start_vector = edge_vertex_endpoints[index, 0] - edge_wcs[index, 0]
        end_vector = edge_vertex_endpoints[index, 1] - edge_wcs[index, -1]
        weight = np.tile((np.arange(32) / 31)[:, np.newaxis], (1, 3))
        weighted_vector = (
            np.tile(start_vector[np.newaxis, :], (32, 1)) * (1 - weight)
            + np.tile(end_vector, (32, 1)) * weight
        )
        edge_wcs[index] += weighted_vector
    return edge_wcs


def _initialize_surfaces(
    face_ncs: Any,
    face_pos: Any,
    edge_wcs: Any,
    face_edge_adjacency: tuple[Any, ...],
) -> Any:
    import numpy as np

    from src.utils import cad_utils as cad

    initialized = []
    for adjacency, normalized_surface, bbox in zip(
        face_edge_adjacency,
        face_ncs,
        face_pos,
    ):
        edge_points = np.asarray(edge_wcs[adjacency], dtype=np.float32)
        if edge_points.size == 0:
            raise ValueError("A reconstructed face has no incident edge points.")
        surface_center, surface_scale = cad.compute_bbox_center_and_size(
            bbox[0:3], bbox[3:]
        )
        minimum, maximum = cad.get_bbox_minmax(edge_points.reshape(-1, 3))
        _, edge_scale = cad.compute_bbox_center_and_size(minimum, maximum)
        if surface_scale < edge_scale:
            surface_scale = 1.05 * edge_scale
        initialized.append(
            normalized_surface * (surface_scale / 2) + surface_center
        )
    surface_initial = np.asarray(np.stack(initialized), dtype=np.float32)
    if not np.isfinite(surface_initial).all():
        raise ValueError("Surface initialization contains non-finite values.")
    return surface_initial


def prepare_geometry_sample(payload: SampleGeometryInput) -> PreparationResult:
    """Build CPU geometry required by the batched CUDA optimizer."""
    try:
        import numpy as np

        from src.utils import cad_utils as cad
        from src.utils import datasets_utils as datasets

        face_z = np.asarray(payload.face_z, dtype=np.float64)
        face_pos = np.asarray(payload.face_pos, dtype=np.float64)
        edge_z = np.asarray(payload.edge_z, dtype=np.float64)
        edge_bbox = np.asarray(payload.edge_bbox, dtype=np.float64)
        edge_corners = np.asarray(payload.edge_corners, dtype=np.float64)
        adjacency = np.asarray(payload.adjacency, dtype=np.int32)
        if len(face_z) == 0 or len(edge_z) == 0:
            raise ValueError("Empty geometry cannot be prepared.")

        face_ncs = np.stack(
            [
                cad.sample_bspline_surface(
                    cad.create_bspline_surface(control.reshape(36, 3))
                )
                for control in face_z
            ]
        )
        edge_ncs = np.stack(
            [
                cad.sample_bspline_curve(
                    cad.create_bspline_curve(control.reshape(6, 3))
                )
                for control in edge_z
            ]
        )

        edge_face_list = datasets.adj_matrix_to_edgeFace(adjacency)
        edge_mask = datasets.list_to_adj_matrix(
            edge_face_list,
            len(edge_z),
            len(face_z),
        ).T
        edge_mask = (1 - edge_mask).astype(bool)
        face_edge_adjacency = tuple(
            datasets.get_faceEdge_adj(edge_face_list, len(face_z))
        )

        repeated_bbox = np.tile(edge_bbox[np.newaxis, :, :], (len(face_z), 1, 1))
        repeated_edges = np.tile(
            edge_ncs[np.newaxis, :, :, :],
            (len(face_z), 1, 1, 1),
        )
        edge_vertices_cad = np.tile(
            edge_corners[np.newaxis, :, :],
            (len(face_z), 1, 1),
        )
        edge_vertices_bbox = []
        for bboxes, normalized_edges in zip(repeated_bbox, repeated_edges):
            endpoints = []
            for bbox, normalized_edge in zip(bboxes, normalized_edges):
                center, size = cad.compute_bbox_center_and_size(
                    bbox[0:3], bbox[3:]
                )
                world_edge = normalized_edge * (size / 2) + center
                endpoints.append(world_edge[[0, -1]].reshape(1, 2, 3))
            edge_vertices_bbox.append(np.vstack(endpoints))

        unique_vertices, edge_vertex_adjacency = cad.detect_shared_vertex3(
            edge_vertices_cad,
            edge_mask,
            edge_vertices_bbox,
        )
        edge_wcs = _align_edges(
            edge_ncs,
            unique_vertices,
            edge_vertex_adjacency,
        )
        surface_initial = _initialize_surfaces(
            face_ncs,
            face_pos,
            edge_wcs,
            face_edge_adjacency,
        )
        return PreparationResult(
            index=payload.index,
            geometry=PreparedGeometry(
                index=payload.index,
                surface_initial=surface_initial,
                edge_wcs=edge_wcs,
                face_edge_adjacency=face_edge_adjacency,
                edge_vertex_adjacency=edge_vertex_adjacency,
            ),
            failure_reason=None,
        )
    except Exception:
        return PreparationResult(
            index=payload.index,
            geometry=None,
            failure_reason="reconstruction_exception",
        )


def finalize_geometry_sample(task: FinalizationTask) -> FinalizationResult:
    """Construct and classify one OCC shape without exposing exception text."""
    compilable = False if task.save_step_policy == "all" else None
    try:
        from src.eval.validity import classify_and_write_solid
        from src.utils import cad_utils as cad

        solid = cad.construct_brep(
            task.surface_wcs,
            task.edge_wcs,
            task.face_edge_adjacency,
            task.edge_vertex_adjacency,
        )
        classification = classify_and_write_solid(
            solid,
            step_path=task.step_path,
            save_step_policy=task.save_step_policy,
        )
    except Exception:
        return FinalizationResult(
            index=task.index,
            valid=False,
            failure_reason="reconstruction_exception",
            step_written=False,
            compilable=compilable,
            step_write_error=False,
        )

    if classification.classification_error:
        failure_reason = "reconstruction_exception"
        valid = False
    elif not classification.closed_solid:
        failure_reason = "not_closed_or_zero_volume"
        valid = False
    else:
        failure_reason = None
        valid = True
    return FinalizationResult(
        index=task.index,
        valid=valid,
        failure_reason=failure_reason,
        step_written=classification.step_written,
        compilable=classification.compilable,
        step_write_error=classification.step_write_error,
    )
