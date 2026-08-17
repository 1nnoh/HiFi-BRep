from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def discover_step_files(input_root: str | Path) -> tuple[Path, ...]:
    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"STEP input root does not exist: {root}")
    discovered: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"STEP input tree contains a symbolic link: {path.relative_to(root)}")
        if path.is_file() and path.suffix.lower() in (".step", ".stp"):
            discovered.append(path.relative_to(root))
    return tuple(sorted(discovered, key=lambda value: value.as_posix()))


def _safe_relative(path: Path) -> Path:
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Input path must be relative and traversal-free: {path}")
    return path


def output_relative_path(input_path: str | Path, *, layout: str) -> Path:
    relative = _safe_relative(Path(input_path))
    if relative.suffix.lower() not in (".step", ".stp"):
        raise ValueError(f"Input path is not a STEP file: {relative}")
    if layout == "relative":
        return relative.with_suffix(".pkl")
    if layout == "abc":
        uid = relative.parent.name
        if len(uid) != 8 or not uid.isdigit():
            raise ValueError(
                f"ABC layout requires an eight-digit parent directory; received {relative}."
            )
        return Path(uid[:4]) / f"{uid}.pkl"
    raise ValueError("layout must be 'abc' or 'relative'.")


def _local_normalize(points: np.ndarray, *, epsilon: float) -> np.ndarray:
    flattened = points.reshape(-1, 3)
    minimum = flattened.min(axis=0)
    maximum = flattened.max(axis=0)
    offset = 0.5 * (minimum + maximum)
    scale = float((maximum - minimum).max())
    if not np.isfinite(scale) or scale <= epsilon:
        raise ValueError("Primitive has a degenerate local normalization scale.")
    return (points - offset) / (0.5 * scale)


def normalize_geometry(
    surface_points: np.ndarray,
    edge_points: np.ndarray,
    corner_points: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the historical global WCS and per-primitive NCS normalization."""
    surfaces = np.asarray(surface_points, dtype=np.float64)
    edges = np.asarray(edge_points, dtype=np.float64)
    corners = np.asarray(corner_points, dtype=np.float64)
    if surfaces.ndim != 4 or surfaces.shape[-1] != 3:
        raise ValueError("surface_points must have shape [F,U,V,3].")
    if edges.ndim != 3 or edges.shape[-1] != 3:
        raise ValueError("edge_points must have shape [E,T,3].")
    if corners.shape != (edges.shape[0], 2, 3):
        raise ValueError("corner_points must have shape [E,2,3].")
    if not all(np.isfinite(value).all() for value in (surfaces, edges, corners)):
        raise ValueError("STEP samples must contain only finite coordinates.")

    flattened_surfaces = surfaces.reshape(-1, 3)
    global_minimum = flattened_surfaces.min(axis=0)
    global_maximum = flattened_surfaces.max(axis=0)
    global_offset = 0.5 * (global_minimum + global_maximum)
    global_scale = float((global_maximum - global_minimum).max())
    if not np.isfinite(global_scale) or global_scale <= epsilon:
        raise ValueError("Solid has a degenerate global normalization scale.")
    surfaces_world = (surfaces - global_offset) / (0.5 * global_scale)
    edges_world = (edges - global_offset) / (0.5 * global_scale)
    corners_world = (corners - global_offset) / (0.5 * global_scale)
    surfaces_local = np.stack(
        [_local_normalize(points, epsilon=epsilon) for points in surfaces_world]
    )
    edges_local = np.stack(
        [_local_normalize(points, epsilon=epsilon) for points in edges_world]
    )
    return surfaces_world, edges_world, surfaces_local, edges_local, corners_world


def _bbox(points: np.ndarray) -> np.ndarray:
    flattened = points.reshape(-1, 3)
    return np.concatenate((flattened.min(axis=0), flattened.max(axis=0)))


def _lexicographic_order(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    return np.lexsort(boxes[:, ::-1].T)


def sort_primitives_lexicographically(
    surface_points: np.ndarray,
    edge_points: np.ndarray,
    corner_points: np.ndarray,
    edge_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort face and edge sequences by their bounding boxes and remap topology."""
    surfaces = np.asarray(surface_points)
    edges = np.asarray(edge_points)
    corners = np.asarray(corner_points)
    adjacency = np.asarray(edge_faces, dtype=np.int64)
    if len(corners) != len(edges) or adjacency.shape != (len(edges), 2):
        raise ValueError("Edge geometry and edge-face adjacency have inconsistent lengths.")

    face_boxes = np.stack([_bbox(points) for points in surfaces])
    face_order = _lexicographic_order(face_boxes)
    old_to_new = np.empty(len(surfaces), dtype=np.int64)
    old_to_new[face_order] = np.arange(len(surfaces), dtype=np.int64)

    edge_boxes = np.stack([_bbox(points) for points in edges])
    edge_order = _lexicographic_order(edge_boxes)
    return (
        surfaces[face_order],
        edges[edge_order],
        corners[edge_order],
        old_to_new[adjacency[edge_order]],
    )


def _fit_degree_five(
    surface_local: np.ndarray,
    edge_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_PointsToBSplineSurface
    from OCC.Core.GeomAbs import GeomAbs_C2
    from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
    from OCC.Core.gp import gp_Pnt

    degree = 5
    face_controls = []
    for points in surface_local:
        rows, columns = points.shape[:2]
        point_array = TColgp_Array2OfPnt(1, rows, 1, columns)
        for row in range(rows):
            for column in range(columns):
                point_array.SetValue(
                    row + 1,
                    column + 1,
                    gp_Pnt(*(float(value) for value in points[row, column])),
                )
        surface = GeomAPI_PointsToBSplineSurface(
            point_array, degree, degree, GeomAbs_C2, 5e-2
        ).Surface()
        if (
            surface.UDegree() != degree
            or surface.VDegree() != degree
            or surface.NbUPoles() != degree + 1
            or surface.NbVPoles() != degree + 1
            or surface.IsUPeriodic()
            or surface.IsVPeriodic()
            or surface.IsURational()
            or surface.IsVRational()
        ):
            raise ValueError("Surface fitting did not produce the fixed 6x6 polynomial form.")
        controls = np.empty((36, 3), dtype=np.float64)
        index = 0
        poles = surface.Poles()
        for row in range(1, 7):
            for column in range(1, 7):
                point = poles.Value(row, column)
                controls[index] = (point.X(), point.Y(), point.Z())
                index += 1
        face_controls.append(controls)

    edge_controls = []
    for points in edge_local:
        point_array = TColgp_Array1OfPnt(1, len(points))
        for index, point in enumerate(points, start=1):
            point_array.SetValue(index, gp_Pnt(*(float(value) for value in point)))
        curve = None
        for tolerance in (5e-3, 8e-3, 5e-2):
            try:
                curve = GeomAPI_PointsToBSpline(
                    point_array, degree, degree, GeomAbs_C2, tolerance
                ).Curve()
                break
            except Exception:
                curve = None
        if curve is None:
            raise ValueError("Curve fitting failed at all configured tolerances.")
        if (
            curve.Degree() != degree
            or curve.NbPoles() != degree + 1
            or curve.IsPeriodic()
            or curve.IsRational()
        ):
            raise ValueError("Curve fitting did not produce the fixed six-pole polynomial form.")
        controls = np.empty((6, 3), dtype=np.float64)
        poles = curve.Poles()
        for index in range(1, 7):
            point = poles.Value(index)
            controls[index - 1] = (point.X(), point.Y(), point.Z())
        edge_controls.append(controls)
    return np.stack(face_controls), np.stack(edge_controls)


def _extract_primitives(solid: object) -> tuple[np.ndarray, ...]:
    from occwl.compound import Compound
    from occwl.entity_mapper import EntityMapper
    from occwl.shell import Shell
    from occwl.solid import Solid
    from occwl.uvgrid import ugrid, uvgrid

    if not isinstance(solid, (Solid, Shell, Compound)):
        raise TypeError("Loaded STEP shape is not an occwl solid, shell, or compound.")
    mapper = EntityMapper(solid)
    faces = {mapper.face_index(face): face for face in solid.faces()}
    edge_records: list[tuple[int, object, tuple[int, int]]] = []
    for edge in solid.edges():
        if not edge.has_curve():
            continue
        connected = list(solid.faces_from_edge(edge))
        if len(connected) != 2 or edge.seam(connected[0]) or edge.seam(connected[1]):
            continue
        left, right = edge.find_left_and_right_faces(connected)
        if left is None or right is None:
            continue
        edge_records.append(
            (
                mapper.edge_index(edge),
                edge,
                (mapper.face_index(left), mapper.face_index(right)),
            )
        )
    if not faces or not edge_records:
        raise ValueError("Solid has no usable faces or non-seam manifold edges.")
    face_order = sorted(faces)
    face_mapping = {old: new for new, old in enumerate(face_order)}
    edge_records.sort(key=lambda value: value[0])
    face_points = []
    for index in face_order:
        face_points.append(uvgrid(faces[index], method="point", num_u=32, num_v=32))
    edge_points = []
    corner_points = []
    edge_faces = []
    for _, edge, adjacent in edge_records:
        points = ugrid(edge, method="point", num_u=32)
        edge_points.append(points)
        corner_points.append((points[0], points[-1]))
        edge_faces.append((face_mapping[adjacent[0]], face_mapping[adjacent[1]]))
    return (
        np.asarray(face_points)[..., :3],
        np.asarray(edge_points),
        np.asarray(corner_points),
        np.asarray(edge_faces, dtype=np.int64),
    )


def parse_step_file(
    path: str | Path,
    *,
    uid: str,
    max_face: int,
) -> dict[str, object]:
    """Convert one single-solid STEP file into the public processed PKL contract."""
    from occwl.io import load_step

    loaded = load_step(str(Path(path).expanduser().resolve()))
    if len(loaded) != 1:
        raise ValueError("STEP file must contain exactly one solid.")
    solid = loaded[0].split_all_closed_faces(num_splits=0)
    solid = solid.split_all_closed_edges(num_splits=0)
    if len(list(solid.faces())) > max_face:
        raise ValueError(f"Solid exceeds max_face={max_face} after seam splitting.")
    face_points, edge_points, corner_points, edge_faces = _extract_primitives(solid)
    if len(edge_points) > max_face * 3:
        raise ValueError(f"Solid exceeds max_edge={max_face * 3}.")
    face_points, edge_points, corner_points, edge_faces = (
        sort_primitives_lexicographically(
            face_points,
            edge_points,
            corner_points,
            edge_faces,
        )
    )
    (
        surfaces_world,
        edges_world,
        surfaces_local,
        edges_local,
        corners_world,
    ) = normalize_geometry(face_points, edge_points, corner_points)
    face_controls, edge_controls = _fit_degree_five(surfaces_local, edges_local)

    rounded_corners = np.round(corners_world, 4)
    unique_corners: list[np.ndarray] = []
    corner_indices = np.empty((len(rounded_corners), 2), dtype=np.int64)
    lookup: dict[tuple[float, float, float], int] = {}
    for edge_index, endpoints in enumerate(rounded_corners):
        for endpoint_index, endpoint in enumerate(endpoints):
            key = tuple(float(value) for value in endpoint)
            if key not in lookup:
                lookup[key] = len(unique_corners)
                unique_corners.append(endpoint.copy())
            corner_indices[edge_index, endpoint_index] = lookup[key]
    face_edges = [
        np.flatnonzero((edge_faces == face_index).any(axis=1)).astype(np.int64)
        for face_index in range(len(surfaces_world))
    ]
    record: dict[str, object] = {
        "uid": uid,
        "surf_wcs": surfaces_world.astype(np.float32),
        "edge_wcs": edges_world.astype(np.float32),
        "surf_ncs": surfaces_local.astype(np.float32),
        "edge_ncs": edges_local.astype(np.float32),
        "face_6ctrs": face_controls.astype(np.float32),
        "edge_6ctrs": edge_controls.astype(np.float32),
        "corner_wcs": corners_world.astype(np.float32),
        "edgeFace_adj": edge_faces,
        "edgeCorner_adj": corner_indices,
        "faceEdge_adj": face_edges,
        "surf_bbox_wcs": np.stack([_bbox(points) for points in surfaces_world]).astype(np.float32),
        "edge_bbox_wcs": np.stack([_bbox(points) for points in edges_world]).astype(np.float32),
        "corner_unique": np.stack(unique_corners).astype(np.float32),
    }
    from src.data.brep_dataset import validate_brep_record

    validate_brep_record(record, max_face=max_face)
    return record
