from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.manifest import load_dataset_manifest, resolve_sample_path
from src.utils.datasets_utils import (
    bbox_corners,
    build_edgepoint_identity_matrix_sorted,
    get_bbox,
    list_to_adj_matrix,
    pad_zero,
    rotate_axis,
)


REQUIRED_FIELDS = (
    "face_6ctrs",
    "edge_6ctrs",
    "corner_wcs",
    "edgeCorner_adj",
    "edgeFace_adj",
    "surf_bbox_wcs",
    "edge_bbox_wcs",
)


@dataclass(frozen=True)
class BrepRecordSummary:
    num_faces: int
    num_edges: int


def _array(record: Mapping[str, object], name: str) -> np.ndarray:
    try:
        return np.asarray(record[name])
    except KeyError as exc:
        raise ValueError(f"Processed B-rep record is missing field '{name}'.") from exc


def _require_shape(name: str, value: np.ndarray, expected: tuple[object, ...]) -> None:
    if value.ndim != len(expected):
        raise ValueError(f"Field '{name}' has shape {value.shape}; expected {expected}.")
    for actual, wanted in zip(value.shape, expected):
        if wanted is not None and actual != wanted:
            raise ValueError(f"Field '{name}' has shape {value.shape}; expected {expected}.")


def validate_brep_record(
    record: Mapping[str, object],
    *,
    max_face: int,
) -> BrepRecordSummary:
    """Validate the processed PKL contract before padding or model execution."""
    if not isinstance(record, Mapping):
        raise ValueError("Processed B-rep sample must be a mapping.")
    missing = [name for name in REQUIRED_FIELDS if name not in record]
    if missing:
        raise ValueError(f"Processed B-rep record is missing fields: {', '.join(missing)}.")

    face_control = _array(record, "face_6ctrs")
    if face_control.ndim == 4 and face_control.shape[1:] == (6, 6, 3):
        face_control = face_control.reshape(face_control.shape[0], 36, 3)
    _require_shape("face_6ctrs", face_control, (None, 36, 3))
    edge_control = _array(record, "edge_6ctrs")
    _require_shape("edge_6ctrs", edge_control, (None, 6, 3))
    num_faces = int(face_control.shape[0])
    num_edges = int(edge_control.shape[0])
    if not 1 <= num_faces <= max_face:
        raise ValueError(
            f"Sample has {num_faces} faces; expected 1..max_face ({max_face})."
        )
    if not 1 <= num_edges <= max_face * 3:
        raise ValueError(
            f"Sample has {num_edges} edges; expected 1..{max_face * 3}."
        )

    corner_world = _array(record, "corner_wcs")
    edge_corner = _array(record, "edgeCorner_adj")
    edge_face = _array(record, "edgeFace_adj")
    surface_bbox = _array(record, "surf_bbox_wcs")
    edge_bbox = _array(record, "edge_bbox_wcs")
    _require_shape("corner_wcs", corner_world, (num_edges, 2, 3))
    _require_shape("edgeCorner_adj", edge_corner, (num_edges, 2))
    _require_shape("edgeFace_adj", edge_face, (num_edges, 2))
    _require_shape("surf_bbox_wcs", surface_bbox, (num_faces, 6))
    _require_shape("edge_bbox_wcs", edge_bbox, (num_edges, 6))

    floating = {
        "face_6ctrs": face_control,
        "edge_6ctrs": edge_control,
        "corner_wcs": corner_world,
        "surf_bbox_wcs": surface_bbox,
        "edge_bbox_wcs": edge_bbox,
    }
    for name, value in floating.items():
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"Field '{name}' must contain only finite numeric values.")
    if not np.issubdtype(edge_corner.dtype, np.integer):
        raise ValueError("Field 'edgeCorner_adj' must use integer indices.")
    if not np.issubdtype(edge_face.dtype, np.integer):
        raise ValueError("Field 'edgeFace_adj' must use integer indices.")
    if edge_corner.size and (edge_corner.min() < 0 or edge_corner.max() >= num_edges * 2):
        raise ValueError("Field 'edgeCorner_adj' contains an out-of-range vertex index.")
    if edge_face.size and (edge_face.min() < 0 or edge_face.max() >= num_faces):
        raise ValueError("Field 'edgeFace_adj' contains an out-of-range face index.")
    if np.any(surface_bbox[:, :3] > surface_bbox[:, 3:]):
        raise ValueError("Field 'surf_bbox_wcs' contains inverted bounds.")
    if np.any(edge_bbox[:, :3] > edge_bbox[:, 3:]):
        raise ValueError("Field 'edge_bbox_wcs' contains inverted bounds.")
    return BrepRecordSummary(num_faces=num_faces, num_edges=num_edges)


def _lexicographic_order(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.empty((0,), dtype=np.int64)
    return np.lexsort(values[:, ::-1].T)


def _rotate_record(
    surface_bbox: np.ndarray,
    edge_bbox: np.ndarray,
    corner_world: np.ndarray,
    face_control: np.ndarray,
    edge_control: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    surface_corners = bbox_corners(surface_bbox)
    edge_corners = bbox_corners(edge_bbox)
    for axis in ("x", "y", "z"):
        angle = random.choice((90, 180, 270))
        surface_corners = rotate_axis(surface_corners, angle, axis, normalized=False)
        edge_corners = rotate_axis(edge_corners, angle, axis, normalized=False)
        corner_world = rotate_axis(corner_world, angle, axis, normalized=False)
        face_control = rotate_axis(face_control, angle, axis, normalized=False)
        edge_control = rotate_axis(edge_control, angle, axis, normalized=False)
    surface_bbox = get_bbox(surface_corners).reshape(len(surface_corners), 6)
    edge_bbox = get_bbox(edge_corners).reshape(len(edge_corners), 6)
    return surface_bbox, edge_bbox, corner_world, face_control, edge_control


class PortableBrepDataset(Dataset[dict[str, object]]):
    """Resolve versioned manifest entries against an explicit processed-data root."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        manifest_path: str | Path,
        split: str,
        max_face: int = 50,
        bbox_scale: float = 3.0,
        augment: bool = False,
        augment_probability: float = 0.5,
        validate_on_load: bool = True,
    ) -> None:
        if max_face <= 0:
            raise ValueError("max_face must be positive.")
        if bbox_scale <= 0:
            raise ValueError("bbox_scale must be positive.")
        if not 0.0 <= augment_probability <= 1.0:
            raise ValueError("augment_probability must be between 0 and 1.")
        self.data_root = Path(data_root).expanduser().resolve()
        self.manifest = load_dataset_manifest(manifest_path)
        self.relative_paths = self.manifest.split(split)
        self.max_face = int(max_face)
        self.bbox_scale = float(bbox_scale)
        self.augment = bool(augment)
        self.augment_probability = float(augment_probability)
        self.validate_on_load = bool(validate_on_load)

    def __len__(self) -> int:
        return len(self.relative_paths)

    def sample_path(self, index: int) -> Path:
        return resolve_sample_path(self.data_root, self.relative_paths[index])

    def __getitem__(self, index: int) -> dict[str, object]:
        relative_path = self.relative_paths[index]
        sample_path = self.sample_path(index)
        if not sample_path.is_file():
            raise FileNotFoundError(f"Processed sample is missing: {relative_path}")
        with sample_path.open("rb") as stream:
            record = pickle.load(stream)
        if not isinstance(record, Mapping):
            raise ValueError(f"Processed sample is not a mapping: {relative_path}")
        summary = validate_brep_record(record, max_face=self.max_face)

        face_control = np.asarray(record["face_6ctrs"], dtype=np.float32).copy()
        if face_control.ndim == 4:
            face_control = face_control.reshape(summary.num_faces, 36, 3)
        edge_control = np.asarray(record["edge_6ctrs"], dtype=np.float32).copy()
        corner_world = np.asarray(record["corner_wcs"], dtype=np.float32).copy()
        edge_corner = np.asarray(record["edgeCorner_adj"], dtype=np.int64)
        edge_face = np.asarray(record["edgeFace_adj"], dtype=np.int64)
        surface_bbox = np.asarray(record["surf_bbox_wcs"], dtype=np.float32).copy()
        edge_bbox = np.asarray(record["edge_bbox_wcs"], dtype=np.float32).copy()

        if self.augment and np.random.random() < self.augment_probability:
            surface_bbox, edge_bbox, corner_world, face_control, edge_control = _rotate_record(
                surface_bbox,
                edge_bbox,
                corner_world,
                face_control,
                edge_control,
            )

        surface_bbox *= self.bbox_scale
        edge_bbox *= self.bbox_scale
        corner_world *= self.bbox_scale
        adjacent_corners, _, flattened_corners = build_edgepoint_identity_matrix_sorted(
            edge_corner,
            corner_world,
        )
        adjacent_faces = list_to_adj_matrix(
            edge_face,
            summary.num_edges,
            summary.num_faces,
        )

        edge_order = _lexicographic_order(edge_bbox)
        edge_bbox = edge_bbox[edge_order]
        edge_control = edge_control[edge_order]
        adjacent_faces = adjacent_faces[edge_order]
        flattened_corners = flattened_corners[edge_order]
        corner_indices = (edge_order[:, None] * 2 + np.asarray([0, 1])).reshape(-1)
        adjacent_corners = adjacent_corners[corner_indices][:, corner_indices]

        face_order = _lexicographic_order(surface_bbox)
        surface_bbox = surface_bbox[face_order]
        face_control = face_control[face_order]
        adjacent_faces = adjacent_faces[:, face_order]
        edge_position = np.concatenate((edge_bbox, flattened_corners), axis=1)

        surface_bbox = pad_zero(surface_bbox, self.max_face)
        face_control = pad_zero(face_control, self.max_face)
        edge_position = pad_zero(edge_position, self.max_face * 3)
        edge_control = pad_zero(edge_control, self.max_face * 3)
        adjacent_faces = pad_zero(adjacent_faces, self.max_face * 3)
        adjacent_faces = pad_zero(adjacent_faces.T, self.max_face).T
        adjacent_corners = pad_zero(adjacent_corners, self.max_face * 6)
        adjacent_corners = pad_zero(adjacent_corners.T, self.max_face * 6).T
        np.fill_diagonal(adjacent_corners, 1)

        return {
            "uid": str(Path(relative_path).with_suffix("")),
            "num_face": torch.tensor([summary.num_faces], dtype=torch.long),
            "num_edge": torch.tensor([summary.num_edges], dtype=torch.long),
            "surf_pos": torch.as_tensor(surface_bbox, dtype=torch.float32),
            "surf_z": torch.as_tensor(face_control, dtype=torch.float32),
            "edge_pos": torch.as_tensor(edge_position, dtype=torch.float32),
            "edge_z": torch.as_tensor(edge_control, dtype=torch.float32),
            "adj_face": torch.as_tensor(adjacent_faces, dtype=torch.long),
            "adj_corner": torch.as_tensor(adjacent_corners, dtype=torch.long),
        }
