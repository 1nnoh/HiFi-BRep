# datasets_utils.py
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import os


__all__ = [
    "build_edgepoint_identity_matrix_sorted",
    "bbox_corners",
    "rotate_axis",
    "get_bbox",
    "list_to_adj_matrix",
    "pad_zero",
    "adj_matrix_to_edgeFace",
    "adj_matrix_to_edgevert",
    "compute_bbox_center_and_size",
    "get_faceEdge_adj",
    "save_point",
    "lef_dfs_order",
]



def _bbox_center_size(edge_pos: np.ndarray):
    mins = edge_pos[:, 0:3]; maxs = edge_pos[:, 3:6]
    ctr = 0.5 * (mins + maxs)                      # (E,3)
    size_vec = (maxs - mins)
    size = np.max(size_vec, axis=1)                # (E,)
    return ctr.astype(np.float32), size.astype(np.float32)

def _edge_corner_pairs(corner_wcs: np.ndarray, edgeCorner_adj: np.ndarray, E: int):
    if corner_wcs.ndim == 2 and corner_wcs.shape == (2*E, 3):
        idx0 = edgeCorner_adj[:, 0]; idx1 = edgeCorner_adj[:, 1]
        return np.stack([corner_wcs[idx0], corner_wcs[idx1]], axis=1).astype(np.float32)  # (E,2,3)
    elif corner_wcs.ndim == 2 and corner_wcs.shape == (E, 6):
        return corner_wcs.reshape(E, 2, 3).astype(np.float32)
    elif corner_wcs.ndim == 3 and corner_wcs.shape[:2] == (E, 2):
        return corner_wcs.astype(np.float32)
    else:
        raise ValueError(f"Unsupported corner_wcs shape: {corner_wcs.shape}")

def _sim3_align(p0, p1, t0, t1, pts):
    """Map local edge control points with the similarity transform from p0/p1
    to t0/t1.

    Args:
        p0, p1, t0, t1: Endpoint vectors shaped (3,).
        pts: Local edge control points shaped (K, 3).

    Returns:
        Transformed control points shaped (K, 3).
    """
    v = p1 - p0; w = t1 - t0
    nv = np.linalg.norm(v); nw = np.linalg.norm(w)
    if nv < 1e-8 or nw < 1e-12:

        out = pts.copy()
        out -= p0; out += t0
        return out
    s = nw / nv

    v_unit = v / nv; w_unit = w / nw
    axis = np.cross(v_unit, w_unit)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:

        dot = np.dot(v_unit, w_unit)
        if dot > 0.0:
            R = np.eye(3, dtype=np.float32)
        else:

            a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(v_unit[0]) > 0.9:
                a = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            axis = np.cross(v_unit, a); axis /= (np.linalg.norm(axis)+1e-12)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], dtype=np.float32)
            R = -np.eye(3, dtype=np.float32) + 2*np.outer(axis, axis)  # 180°
    else:
        axis /= axis_norm
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float32)
        cos_t = np.dot(v_unit, w_unit)
        sin_t = axis_norm
        R = np.eye(3, dtype=np.float32) + K*sin_t + (K @ K)*(1.0 - cos_t)


    out = (pts - p0[None, :]) @ R.T
    out *= s
    out += t0[None, :]
    return out




def build_edgepoint_identity_matrix_sorted(edgeCorner_adj, corner_wcs):
    """Build a sorted endpoint identity matrix and sorted endpoint coordinates.

    Args:
        edgeCorner_adj: [n, 2] endpoint indices for each edge.
        corner_wcs: Endpoint coordinates shaped [n, 2, 3].

    Returns:
        adj_corner_sorted: [2n, 2n] sorted endpoint identity matrix (uint8).
        corners: [2n, 3] sorted endpoint coordinates (float32).
        corners_flat: [n, 6] flattened endpoint pairs used as features.
    """
    edgeCorner_adj = np.asarray(edgeCorner_adj)
    corner_wcs = np.asarray(corner_wcs, dtype=np.float32)

    n = edgeCorner_adj.shape[0]

    edge_points = edgeCorner_adj.reshape(-1)  # shape = (2n,)
    adj_corner = (edge_points[:, None] == edge_points[None, :]).astype(np.uint8)  # (2n, 2n)

    corners_sorted = []
    index_map = []
    corners = []

    for i, corner in enumerate(corner_wcs):  # corner: (2, 3)

        sorted_indices = np.lexsort((corner[:, 2], corner[:, 1], corner[:, 0]))
        sorted_corner = corner[sorted_indices]
        corners_sorted.append(sorted_corner.flatten())
        corners.append(sorted_corner[0])
        corners.append(sorted_corner[1])


        index_map.append(2 * i + sorted_indices[0])
        index_map.append(2 * i + sorted_indices[1])

    corners_flat = np.stack(corners_sorted, axis=0).astype(np.float32)  # (n, 6)
    corners = np.stack(corners, axis=0).astype(np.float32)              # (2n, 3)
    index_map = np.asarray(index_map, dtype=np.int64)                   # (2n,)


    adj_corner_sorted = adj_corner[index_map][:, index_map]  # (2n, 2n)
    return adj_corner_sorted, corners, corners_flat


def bbox_corners(bboxes):
    """
    Given [N,6] bboxes = [xmin, ymin, zmin, xmax, ymax, zmax],
    return [N,8,3] corners in fixed order.
    """
    bboxes = np.asarray(bboxes, dtype=np.float32)
    out = []
    for bbox in bboxes:
        bottom_left, top_right = bbox[:3], bbox[3:]
        x0, y0, z0 = bottom_left
        x1, y1, z1 = top_right

        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1],
        ], dtype=np.float32)
        out.append(corners)
    return np.stack(out, axis=0)  # [N, 8, 3]

def R_axis(angle_deg, axis):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    if axis == 'x':
        return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)
    if axis == 'y':
        return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)
    if axis == 'z':
        return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
    raise ValueError

def apply_R(arr, R):
    # arr: (..., 3), R: (3,3)
    assert arr.shape[-1] == 3 and R.shape == (3,3)
    return np.tensordot(arr, R.T, axes=([arr.ndim-1],[0]))

import numpy as np


def random_quaternion_uniform():
    """Sample a uniformly distributed unit quaternion from three independent
    uniform values in [0, 1), following Shoemake (1992).

    Returns:
        Quaternion q = [w, x, y, z].
    """
    u1, u2, u3 = np.random.rand(), np.random.rand(), np.random.rand()
    sqrt1_u1 = np.sqrt(1.0 - u1)
    sqrt_u1  = np.sqrt(u1)
    theta1 = 2.0 * np.pi * u2
    theta2 = 2.0 * np.pi * u3
    w = np.cos(theta2) * sqrt_u1
    x = np.sin(theta1) * sqrt1_u1
    y = np.cos(theta1) * sqrt1_u1
    z = np.sin(theta2) * sqrt_u1
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)

def quat_to_R(q):
    """Convert q = [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = map(float, q)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    R = np.array([
        [ww+xx-yy-zz,    2*(x*y - w*z),  2*(x*z + w*y)],
        [2*(x*y + w*z),  ww-xx+yy-zz,    2*(y*z - w*x)],
        [2*(x*z - w*y),  2*(y*z + w*x),  ww-xx-yy+zz  ]
    ], dtype=np.float32)
    return R


def snap_R_to_grid(R, step_deg=45):
    """Find the nearest Z-Y-X Euler-angle grid rotation whose angles are
    integer multiples of step. The small grid is searched exhaustively.
    """
    def R_axis(angle_deg, axis):
        a = np.deg2rad(angle_deg); c,s = np.cos(a), np.sin(a)
        if axis=='x': return np.array([[1,0,0],[0,c,-s],[0,s,c]], np.float32)
        if axis=='y': return np.array([[c,0,s],[0,1,0],[-s,0,c]], np.float32)
        if axis=='z': return np.array([[c,-s,0],[s,c,0],[0,0,1]], np.float32)
        raise ValueError


    angles = np.arange(-180, 180, step_deg, dtype=np.int32)
    best_R, best_err = None, 1e9

    for az in angles:
        Rz = R_axis(az,'z')
        for ay in angles:
            Ry = R_axis(ay,'y')
            for ax in angles:
                Rx = R_axis(ax,'x')
                Rg = Rz @ Ry @ Rx
                err = np.linalg.norm(R - Rg, ord='fro')
                if err < best_err:
                    best_err, best_R = err, Rg
    return best_R


def random_rotation_matrix(mix_snap_prob=0.0, snap_step_deg=45):
    q = random_quaternion_uniform()
    R = quat_to_R(q)
    if mix_snap_prob > 0 and np.random.rand() < mix_snap_prob:
        R = snap_R_to_grid(R, step_deg=snap_step_deg)
    return R



def rotate_axis(pnts, angle_degrees, axis, normalized=False):
    """
    Rotate points around axis by angle (degrees).
    pnts: (..., 3) numpy array
    axis: 'x' | 'y' | 'z'
    normalized: True -> rescale to fit [-1, 1] by max-abs after rotation
    """
    pnts = np.asarray(pnts, dtype=np.float32)
    angle_radians = np.radians(angle_degrees).astype(np.float32)

    # to homogeneous
    shape = list(pnts.shape); shape[-1] = 1
    pnts_h = np.concatenate((pnts, np.ones(shape, dtype=np.float32)), axis=-1)

    c, s = np.cos(angle_radians), np.sin(angle_radians)
    if axis == 'x':
        R = np.array([[1, 0, 0, 0],
                      [0, c, -s, 0],
                      [0, s,  c, 0],
                      [0, 0,  0, 1]], dtype=np.float32)
    elif axis == 'y':
        R = np.array([[ c, 0, s, 0],
                      [ 0, 1, 0, 0],
                      [-s, 0, c, 0],
                      [ 0, 0, 0, 1]], dtype=np.float32)
    elif axis == 'z':
        R = np.array([[ c, -s, 0, 0],
                      [ s,  c, 0, 0],
                      [ 0,  0, 1, 0],
                      [ 0,  0, 0, 1]], dtype=np.float32)
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    rotated_h = np.dot(pnts_h, R.T)
    rotated = rotated_h[..., :3]

    if normalized:
        max_abs = np.max(np.abs(rotated))
        if max_abs > 0:
            rotated = rotated / max_abs
    return rotated


def get_bbox(pnts):
    """Compute axis-aligned bounds for point sets shaped [N, P, 3].

    Returns:
        Bounds shaped [N, 2, 3], containing min_point and max_point.
    """
    pnts = np.asarray(pnts, dtype=np.float32)
    mins = pnts.min(axis=1)
    maxs = pnts.max(axis=1)
    return np.stack([mins, maxs], axis=1)  # [N, 2, 3]


def list_to_adj_matrix(edgeFace_list, num_edge, num_face):
    """Convert per-edge face-index lists into a binary adjacency matrix shaped
    [num_edge, num_face].
    """
    adj_face = np.zeros((num_edge, num_face), dtype=np.int64)
    for i, faces in enumerate(edgeFace_list):
        for f in faces:
            if 0 <= f < num_face:
                adj_face[i, f] = 1
    return adj_face


def pad_zero(x: np.ndarray, max_len: int, return_mask: bool = False):
    """Pad the first dimension with zeros to max_len. When return_mask is true,
    also return a [max_len] boolean mask whose padded entries are true.
    """
    x = np.asarray(x)
    n = x.shape[0]
    assert max_len >= n, f"pad_zero: max_len({max_len}) < len(x)({n})"

    if max_len == n:
        if return_mask:
            mask = np.zeros(n, dtype=bool)
            return x, mask
        return x

    pad_shape = (max_len - n, *x.shape[1:])
    padding = np.zeros(pad_shape, dtype=x.dtype)
    x_padded = np.concatenate([x, padding], axis=0)

    if return_mask:
        mask = np.concatenate([np.zeros(n, dtype=bool), np.ones(max_len - n, dtype=bool)])
        return x_padded, mask
    else:
        return x_padded

def adj_matrix_to_edgeFace(adj_face):
    """Convert a [num_edge, num_face] adjacency matrix into per-edge lists of
    adjacent face indices.
    """
    num_edge = adj_face.shape[0]
    edgeFace_adj = []
    for i in range(num_edge):

        faces = np.where(adj_face[i] == 1)[0].tolist()
        edgeFace_adj.append(faces)
    return edgeFace_adj

def adj_matrix_to_edgevert(adj_vert):
    """Convert a [num_edge, num_vert] adjacency matrix into a [num_edge, 2]
    edge-to-vertex index array.
    """

    assert np.all(adj_vert.sum(axis=1) == 2), 'Every edge must connect exactly two vertices.'
    num_edge = adj_vert.shape[0]
    edgeVert_adj = np.zeros((num_edge, 2), dtype=np.int64)
    for i in range(num_edge):
        verts = np.where(adj_vert[i] == 1)[0]
        if len(verts) >= 2:
            edgeVert_adj[i] = verts[:2]
        elif len(verts) == 1:
            edgeVert_adj[i] = [verts[0], -1]
        else:
            edgeVert_adj[i] = [-1, -1]
    return edgeVert_adj

def compute_bbox_center_and_size(min_corner, max_corner):
    # Calculate the center
    center_x = (min_corner[0] + max_corner[0]) / 2
    center_y = (min_corner[1] + max_corner[1]) / 2
    center_z = (min_corner[2] + max_corner[2]) / 2
    center = np.array([center_x, center_y, center_z])
    # Calculate the size
    size_x = max_corner[0] - min_corner[0]
    size_y = max_corner[1] - min_corner[1]
    size_z = max_corner[2] - min_corner[2]
    size = max(size_x, size_y, size_z)
    return center, size

def get_faceEdge_adj(edgeFace_list, num_faces):

    faceEdge_adj = [[] for _ in range(num_faces)]


    for edge_idx, faces in enumerate(edgeFace_list):
        for face in faces:

            if face != -1 and 0 <= face < num_faces:
                faceEdge_adj[face].append(edge_idx)


    faceEdge_adj = [np.array(edges) for edges in faceEdge_adj]
    return faceEdge_adj

def save_point(face_ncs,surf_pos, edge_ncs,edge_pos, ef_adj=None,file_name='',idx=0):

    if ef_adj is not None:
        wcs_init = []
        for i in range(surf_pos.shape[0]):
            for j in range(edge_ncs.shape[0]):
                if ef_adj[j][i]==1:
                    ncs = edge_ncs[j]
                    bbox = edge_pos[j]
                    face_id = i

                    edge_center, edge_scale = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
                    wcs = ncs * (edge_scale / 2) + edge_center

                    wcs_init.append((wcs, face_id))
    else:
        wcs_init = []
        for i in range(edge_ncs.shape[0]):
            ncs = edge_ncs[i]
            bbox = edge_pos[i]

            edge_center, edge_scale = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
            wcs = ncs * (edge_scale / 2) + edge_center
            wcs_init.append((wcs, i))
    efilename=os.path.join(file_name,str(idx)+'_edge.xyzc')
    os.makedirs(file_name,exist_ok=True)

    with open(efilename, 'w') as file:
        for wcs, i in wcs_init:

            for point in wcs:
                x, y, z = point
                file.write(f'{x:.18e} {y:.18e} {z:.18e} {i}\n')

    surf_wcs_init = []
    for ncs, bbox in zip(face_ncs, surf_pos):
        surf_center, surf_scale = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
        wcs = ncs * (surf_scale/2)*1.05 + surf_center
        # wcs = ncs
        surf_wcs_init.append(wcs)

    ffilename=os.path.join(file_name,str(idx)+'_face.xyzc')
    with open(ffilename, 'w') as file:
        for label, array in enumerate(surf_wcs_init):
            assert array.shape == (32, 32, 3), f"Array at position {label} must have shape (32, 32, 3)"
            for i in range(32):
                for j in range(32):
                    x, y, z = array[i, j]
                    file.write(f'{x:.18e} {y:.18e} {z:.18e} {label}\n')

    return 1




def lef_dfs_order(
    *,
    N_faces: int,
    edge_ctrs: np.ndarray,           # (E, 6, 3)  -- normalized edge control points
    edge_pos: np.ndarray,            # (E, 6)     -- bbox [xmin,ymin,zmin, xmax,ymax,zmax] per edge
    edgeFace_adj: List[List[int]],   # length E, each is [fi, fj] or [fi] / []
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LEF-DFS (Longest-Edge-First DFS) ordering.

    Returns:
        face_order : (N_faces_kept,) np.int64
        edge_order : (E_kept,)      np.int64

    Rotation/translation invariance:
      - Only lengths and counts are compared (and ids as last resort).
      - Edge lengths are computed in global coordinates reconstructed from per-edge bbox.
    """
    # --------- basic checks ---------
    E = int(edge_ctrs.shape[0])
    assert edge_ctrs.ndim == 3 and edge_ctrs.shape[1:] == (6, 3), "edge_ctrs must be (E,6,3)"
    assert edge_pos.shape == (E, 6), "edge_pos must be (E,6)"
    assert len(edgeFace_adj) == E, "edgeFace_adj length must equal E"

    # --------- 1) edge arc lengths (rotation/translation invariant) ---------
    # unnormalize control points by edge bbox: p_glb = center + s * p_norm, s = max side length
    centers = 0.5 * (edge_pos[:, 0:3] + edge_pos[:, 3:6])         # (E,3)
    scales  = np.max(np.abs(edge_pos[:, 3:6] - edge_pos[:, 0:3]), axis=1)  # (E,)
    P = centers[:, None, :] + edge_ctrs * scales[:, None, None]    # (E,6,3)
    seg = P[:, 1:, :] - P[:, :-1, :]                               # (E,5,3)
    edge_len = np.linalg.norm(seg, axis=-1).sum(axis=1).astype(np.float32)  # (E,)

    # --------- 2) face perimeter P_i (sum of incident edge lengths) ---------
    P_face = np.zeros((N_faces,), dtype=np.float64)
    for e in range(E):
        faces = _valid_faces(edgeFace_adj[e], N_faces)
        for fi in faces:
            P_face[fi] += float(edge_len[e])

    # --------- 3) dual graph + shared-length weights w_ij ---------
    # accumulate sum of edge lengths between each face pair
    W: Dict[int, Dict[int, float]] = {i: {} for i in range(N_faces)}
    neighbors: List[List[int]] = [[] for _ in range(N_faces)]
    for e in range(E):
        faces = _valid_faces(edgeFace_adj[e], N_faces)
        if len(faces) < 2:
            continue
        i, j = faces[0], faces[1]
        if i == j:
            continue
        # undirected accumulation
        w = float(edge_len[e])
        W[i][j] = W[i].get(j, 0.0) + w
        W[j][i] = W[j].get(i, 0.0) + w

    for i in range(N_faces):
        neighbors[i] = sorted(W[i].keys())  # stable base order by id

    # --------- 4) connected components (by topology only) ---------
    comps: List[List[int]] = []
    seen = np.zeros((N_faces,), dtype=bool)
    for s in range(N_faces):
        if seen[s]:
            continue
        # BFS/DFS over neighbors list (topology only)
        comp = []
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in neighbors[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(sorted(comp))  # deterministic

    # order components by total perimeter desc, tie -> min face id
    comp_scores = []
    for comp in comps:
        totalP = float(np.sum(P_face[np.array(comp, dtype=np.int64)]))
        comp_scores.append(( -totalP, min(comp) ))
    comps = [c for _, c in sorted(zip(comp_scores, comps), key=lambda x: x[0])]

    # --------- 5) DFS per component (root = max perimeter) ---------
    face_order_list: List[int] = []
    visited = np.zeros((N_faces,), dtype=bool)

    for comp in comps:
        comp_arr = np.array(comp, dtype=np.int64)
        # root: perimeter desc, tie -> face_id asc
        root = int(comp_arr[np.lexsort((comp_arr, -P_face[comp_arr]))][0])

        # iterative DFS (preorder)
        stack: List[Tuple[int, int, List[int]]] = []  # (node, next_idx, sorted_neighbors)
        visited[root] = True
        face_order_list.append(root)
        stack.append((root, 0, _sorted_unvisited_neighbors(root, neighbors, visited, W, P_face)))

        while stack:
            u, idx, nb = stack[-1]
            if idx >= len(nb):
                stack.pop()
                continue
            v = nb[idx]
            stack[-1] = (u, idx + 1, nb)
            if visited[v]:
                continue
            visited[v] = True
            face_order_list.append(v)
            stack.append((v, 0, _sorted_unvisited_neighbors(v, neighbors, visited, W, P_face)))

    face_order = np.array(face_order_list, dtype=np.int64)

    # --------- 6) edge order (unique & clustered by faces) ---------
    INF = 10**9
    rank = np.full((N_faces,), INF, dtype=np.int64)
    rank[face_order] = np.arange(len(face_order), dtype=np.int64)

    edge_keys: List[Tuple[int, int, float, int]] = []
    for e in range(E):
        faces = _valid_faces(edgeFace_adj[e], N_faces)
        if len(faces) == 2:
            r0, r1 = rank[faces[0]], rank[faces[1]]
            lo, hi = (r0, r1) if r0 <= r1 else (r1, r0)
            edge_keys.append((int(lo), int(hi), -float(edge_len[e]), int(e)))
        elif len(faces) == 1:
            r0 = int(rank[faces[0]])
            edge_keys.append((r0, INF, -float(edge_len[e]), int(e)))
        else:
            # Orphan edge: no incident face -> put to the very end deterministically
            edge_keys.append((INF, INF, -float(edge_len[e]), int(e)))

    edge_order = np.array([t[-1] for t in sorted(edge_keys)], dtype=np.int64)
    return face_order, edge_order


# ------------------------ helpers ------------------------

def _valid_faces(faces_raw: List[int], N_faces: int) -> List[int]:
    """Clean a raw faces list coming from edgeFace_adj: drop invalid/duplicate faces."""
    out = []
    for fi in faces_raw:
        if isinstance(fi, (np.integer, int)) and (0 <= int(fi) < N_faces):
            out.append(int(fi))
    if len(out) >= 2 and out[0] == out[1]:
        out = [out[0]]  # degenerate self-pair -> single
    return out


def _sorted_unvisited_neighbors(
    u: int,
    neighbors: List[List[int]],
    visited: np.ndarray,
    W: Dict[int, Dict[int, float]],
    P_face: np.ndarray
) -> List[int]:
    """Sort unvisited neighbors of u by (shared_length desc, perimeter desc, face_id asc)."""
    cand = []
    for v in neighbors[u]:
        if not visited[v]:
            w = float(W[u].get(v, 0.0))
            cand.append(( -w, -float(P_face[v]), int(v) ))
    cand.sort()
    return [v for (_,__,v) in cand]
