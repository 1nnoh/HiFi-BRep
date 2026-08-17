
import numpy as np
import math
import torch
import torch.nn as nn
import random
import string
import argparse
import pickle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Optional, Tuple, Union
from collections import defaultdict
from OCC.Core.gp import gp_Pnt, gp_Pnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pln, gp_Ax3, gp_Vec
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.Geom import (Geom_BSplineSurface, Geom_BSplineCurve, Geom_Plane,
                           Geom_SphericalSurface, Geom_CylindricalSurface, Geom_ConicalSurface)
from concurrent.futures import ProcessPoolExecutor
from OCC.Core.BRepCheck import BRepCheck_Analyzer, BRepCheck_NoError
from multiprocessing import get_context
import torch
import torch.functional as F
from einops import reduce
from einops import rearrange
from typing import Optional
import os


from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SHELL
from OCC.Core.TopoDS import topods

def is_shell_closed(solid):
    explorer = TopExp_Explorer(solid, TopAbs_SHELL)
    while explorer.More():
        shell = topods.Shell(explorer.Current())
        if not shell.Closed():
            return False
        explorer.Next()
    return True

from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties

def has_nonzero_volume(solid):
    props = GProp_GProps()
    brepgprop_VolumeProperties(solid, props)
    return props.Mass() > 1e-6

def sample_bspline_curve(bspline_curve, num_points=32):
    u_start, u_end = bspline_curve.FirstParameter(), bspline_curve.LastParameter()
    u_range = np.linspace(u_start, u_end, num_points)
    points = np.zeros((num_points, 3), dtype=np.float64)

    for i, u in enumerate(u_range):
        pnt = bspline_curve.Value(u)
        points[i] = [pnt.X(), pnt.Y(), pnt.Z()]


    poles = bspline_curve.Poles()
    points[0]  = [poles.Value(1).X(), poles.Value(1).Y(), poles.Value(1).Z()]
    points[-1] = [poles.Value(poles.Length()).X(), poles.Value(poles.Length()).Y(), poles.Value(poles.Length()).Z()]
    return points


def create_bspline_curve(ctrs):
    n_poles = ctrs.shape[0]
    assert n_poles >= 2

    degree = n_poles - 1


    poles = TColgp_Array1OfPnt(1, n_poles)
    for i, ctr in enumerate(ctrs, 1):
        poles.SetValue(i, gp_Pnt(*ctr))


    n_knots = 2
    knots = TColStd_Array1OfReal(1, n_knots)
    knots.SetValue(1, 0.0)
    knots.SetValue(2, 1.0)


    mults = TColStd_Array1OfInteger(1, n_knots)
    mults.SetValue(1, degree + 1)
    mults.SetValue(2, degree + 1)


    bspline_curve = Geom_BSplineCurve(poles, knots, mults, degree)
    return bspline_curve


def sample_bspline_surface(bspline_surface, num_u=32, num_v=32):
    u_start, u_end, v_start, v_end = bspline_surface.Bounds()
    u_range = np.linspace(u_start, u_end, num_u)
    v_range = np.linspace(v_start, v_end, num_v)

    points = np.zeros((num_u, num_v, 3))

    for i, u in enumerate(u_range):
        for j, v in enumerate(v_range):
            pnt = bspline_surface.Value(u, v)
            points[i, j] = [pnt.X(), pnt.Y(), pnt.Z()]

    return points

def create_bspline_surface(ctrs):
    n_pts = ctrs.shape[0]
    side_len = int(math.sqrt(n_pts))
    assert side_len * side_len == n_pts, "ctrs.shape[0] must be a perfect square"

    u_count = side_len
    v_count = side_len

    degree_u = u_count - 1
    degree_v = v_count - 1


    poles = TColgp_Array2OfPnt(1, u_count, 1, v_count)
    for i in range(u_count):
        for j in range(v_count):
            idx = i * v_count + j
            poles.SetValue(i + 1, j + 1, gp_Pnt(*ctrs[idx]))


    u_knots = TColStd_Array1OfReal(1, 2)
    v_knots = TColStd_Array1OfReal(1, 2)
    u_knots.SetValue(1, 0.0)
    u_knots.SetValue(2, 1.0)
    v_knots.SetValue(1, 0.0)
    v_knots.SetValue(2, 1.0)


    u_mults = TColStd_Array1OfInteger(1, 2)
    v_mults = TColStd_Array1OfInteger(1, 2)
    u_mults.SetValue(1, degree_u + 1)
    u_mults.SetValue(2, degree_u + 1)
    v_mults.SetValue(1, degree_v + 1)
    v_mults.SetValue(2, degree_v + 1)

    bspline_surface = Geom_BSplineSurface(
        poles, u_knots, v_knots, u_mults, v_mults, degree_u, degree_v
    )

    return bspline_surface

def detect_shared_vertex3(edgeV_cad, edge_mask_cad, edgeV_bbox):
    """Args:
      edgeV_cad: [num_face, num_edge, 6] array containing the two 3D
        endpoints of each valid edge in CAD coordinates.
      edgeV_bbox: List of num_face arrays shaped [num_edge, 6], containing
        edge endpoints derived from bounding boxes.
      edge_mask_cad: [num_face, num_edge] boolean mask. A false entry means
        that the face contains the corresponding global edge.

    Returns:
      unique_vertices: [num_vertices, 3] merged vertex coordinates. Coordinates
        from multiple candidates for the same physical vertex are averaged.
      EdgeVertexAdj: [num_edge, 2] indices into unique_vertices. Endpoints are
        -1 when a global edge does not occur on any face.

    Each global edge should have two endpoints, but different faces can present
    them in different orders. Per-face edge2loop correspondences and union-find
    merge equivalent candidates before coordinates are averaged and the global
    edge-to-vertex adjacency is assembled.
    """
    num_face, num_global_edges = edgeV_cad.shape[0], edgeV_cad.shape[1]


    global_candidates = []
    candidate_info = []

    edge_endpoint_occurrences = {}


    parent = []
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    n_total = 0


    for i in range(num_face):
        valid_edges = np.where(~edge_mask_cad[i])[0]
        if len(valid_edges) == 0:
            continue


        candidates = edgeV_bbox[i][valid_edges].reshape(-1, 2, 3)
        merge_pairs = edge2loop(candidates)
        if len(merge_pairs) != len(valid_edges):

            candidates = edgeV_cad[i][valid_edges].reshape(-1, 2, 3)
            merge_pairs = edge2loop(candidates)
            if len(merge_pairs) != len(valid_edges):

                assert False, 'Failed to merge vertices for face {}.'.format(i)

        base_index = n_total
        n_candidates_face = candidates.shape[0] * 2  # = 2 * len(valid_edges)
        for j_local, ge in enumerate(valid_edges):
            for ep in range(2):
                v = candidates[j_local, ep, :]
                global_candidates.append(v)
                candidate_info.append( (i, int(ge), ep) )
                parent.append(n_total)
                key = (int(ge), ep)
                if key not in edge_endpoint_occurrences:
                    edge_endpoint_occurrences[key] = []
                edge_endpoint_occurrences[key].append(n_total)
                n_total += 1

        for pair in merge_pairs:
            idx1 = base_index + pair[0]
            idx2 = base_index + pair[1]
            union(idx1, idx2)


    for key, idx_list in edge_endpoint_occurrences.items():
        if len(idx_list) > 1:
            ref = idx_list[0]
            for idx in idx_list[1:]:
                if np.array_equal(global_candidates[ref], global_candidates[idx]):
                    union(ref, idx)


    groups = {}
    for idx in range(n_total):
        r = find(idx)
        if r not in groups:
            groups[r] = []
        groups[r].append(idx)


    unique_vertices_list = []
    mapping = np.zeros(n_total, dtype=int)
    for new_idx, group in enumerate(groups.values()):
        pts = np.array([global_candidates[k] for k in group])
        avg_pt = pts.mean(axis=0)
        unique_vertices_list.append(avg_pt)
        for k in group:
            mapping[k] = new_idx
    unique_vertices = np.array(unique_vertices_list)




    EdgeVertexAdj = np.empty((num_global_edges, 2), dtype=int)
    for ge in range(num_global_edges):
        for ep in [0, 1]:
            key = (ge, ep)
            if key in edge_endpoint_occurrences and len(edge_endpoint_occurrences[key]) > 0:
                rep = find(edge_endpoint_occurrences[key][0])

                for idx in edge_endpoint_occurrences[key]:
                    if find(idx) != rep:

                        union(rep, idx)
                        rep = find(rep)
                EdgeVertexAdj[ge, ep] = mapping[rep]
            else:
                EdgeVertexAdj[ge, ep] = -1
    return unique_vertices, EdgeVertexAdj


def edge2loop(face_edges):
    face_edges_flatten = face_edges.reshape(-1,3)
    # connect end points by closest distance
    merged_vertex_id = []
    for edge_idx, startend in enumerate(face_edges):
        self_id = [2*edge_idx, 2*edge_idx+1]
        # left endpoint
        distance = np.linalg.norm(face_edges_flatten - startend[0], axis=1)
        min_id = list(np.argsort(distance))
        min_id_noself = [x for x in min_id if x not in self_id]
        merged_vertex_id.append(sorted([2*edge_idx, min_id_noself[0]]))
        # right endpoint
        distance = np.linalg.norm(face_edges_flatten - startend[1], axis=1)
        min_id = list(np.argsort(distance))
        min_id_noself = [x for x in min_id if x not in self_id]
        merged_vertex_id.append(sorted([2*edge_idx+1, min_id_noself[0]]))

    merged_vertex_id = np.unique(np.array(merged_vertex_id),axis=0)
    return merged_vertex_id

class STModel(nn.Module):
    def __init__(self, num_edge, num_surf):
        super().__init__()
        self.edge_t = nn.Parameter(torch.zeros((num_edge, 3)))
        self.surf_st = nn.Parameter(torch.FloatTensor([1,0,0,0]).unsqueeze(0).repeat(num_surf,1))

def joint_optimize(surf_ncs, edge_ncs, surfPos, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, num_edge, num_surf, cuda=False,renormalize=True):
    """Jointly optimize face, edge, and vertex geometry under the predicted topology.

    Args:
      surf_ncs: [num_surf, u, v, 3] initial normalized surface coordinates.
      edge_ncs: [num_edge, 32, 3] normalized edge coordinates.
      surfPos: [num_surf, 6] surface bounding boxes.
      unique_vertices: [num_vertices, 3] vertex coordinates.
      EdgeVertexAdj: [num_edge, 2] edge-to-vertex adjacency.
      FaceEdgeAdj: Per-face lists of global edge indices.
      num_edge: Number of global edges.
      num_surf: Number of global surfaces.
      cuda: Prefer GPU execution when true.
    """
    from chamferdist import ChamferDistance


    loss_func = ChamferDistance()
    if cuda:
        loss_func = loss_func.cuda()

    model = STModel(num_edge, num_surf)
    if cuda:
        model = model.cuda()
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-08,
    )


    edge_ncs_se = edge_ncs[:, [0, -1]]
    edge_vertex_se = unique_vertices[EdgeVertexAdj]

    edge_wcs = []

    for wcs, ncs_se, vertex_se in zip(edge_ncs, edge_ncs_se, edge_vertex_se):
        # scale
        scale_target = np.linalg.norm(vertex_se[0] - vertex_se[1])
        scale_ncs = np.linalg.norm(ncs_se[0] - ncs_se[1])
        edge_scale = scale_target / (scale_ncs+1e-9)

        edge_updated = wcs * edge_scale
        edge_se = ncs_se * edge_scale

        # offset
        offset = (vertex_se - edge_se)
        offset_rev = (vertex_se - edge_se[::-1])

        # swap start / end if necessary
        offset_error = np.abs(offset[0] - offset[1]).mean()
        offset_rev_error = np.abs(offset_rev[0] - offset_rev[1]).mean()
        if offset_rev_error < offset_error:
            edge_updated = edge_updated[::-1]
            offset = offset_rev

        edge_updated = edge_updated + offset.mean(0)[np.newaxis, np.newaxis, :]
        edge_wcs.append(edge_updated)

    edge_wcs = np.vstack(edge_wcs)

    # Replace start/end points with corner, and backprop change along curve
    for index in range(len(edge_wcs)):
        start_vec = edge_vertex_se[index, 0] - edge_wcs[index, 0]
        end_vec = edge_vertex_se[index, 1] - edge_wcs[index, -1]
        weight = np.tile((np.arange(32) / 31)[:, np.newaxis], (1, 3))
        weighted_vec = np.tile(start_vec[np.newaxis, :], (32, 1)) * (1 - weight) + np.tile(end_vec, (32, 1)) * weight
        edge_wcs[index] += weighted_vec

    # Optimize surfaces
    face_edges = []
    for adj in FaceEdgeAdj:
        all_pnts = edge_wcs[adj]
        face_edge_tensor = torch.FloatTensor(all_pnts)
        if cuda:
            face_edge_tensor = face_edge_tensor.cuda()
        face_edges.append(face_edge_tensor)

    # Initialize surface in wcs based on surface pos
    if renormalize:
        surf_wcs_init = []
        bbox_threshold_min = []
        bbox_threshold_max = []
        for edges_perface, ncs, bbox in zip(face_edges, surf_ncs, surfPos):
            surf_center, surf_scale = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
            edges_perface_flat = edges_perface.reshape(-1, 3).detach().cpu().numpy()
            min_point, max_point = get_bbox_minmax(edges_perface_flat)
            edge_center, edge_scale = compute_bbox_center_and_size(min_point, max_point)
            bbox_threshold_min.append(min_point)
            bbox_threshold_max.append(max_point)


            if surf_scale < edge_scale:
                surf_scale = 1.05 * edge_scale

            wcs = ncs * (surf_scale / 2) + surf_center
            surf_wcs_init.append(wcs)

        surf_wcs_init = np.stack(surf_wcs_init)
    else:
        surf_wcs_init = surf_ncs


    # optimize the surface offset
    surf = torch.FloatTensor(surf_wcs_init)
    if cuda:
        surf = surf.cuda()
    for iters in range(200):
        surf_scale = model.surf_st[:, 0].reshape(-1, 1, 1, 1)
        surf_offset = model.surf_st[:, 1:].reshape(-1, 1, 1, 3)
        surf_updated = surf + surf_offset

        surf_loss = 0
        for surf_pnt, edge_pnts in zip(surf_updated, face_edges):
            surf_pnt = surf_pnt.reshape(-1, 3)

            edge_pnts = edge_pnts.reshape(-1, 3).detach()
            surf_loss += loss_func(surf_pnt.unsqueeze(0), edge_pnts.unsqueeze(0), bidirectional=False, reverse=True)
        surf_loss /= len(surf_updated)

        optimizer.zero_grad()
        surf_loss.backward()
        optimizer.step()





    surf_wcs = surf_updated.detach().cpu().numpy()

    return (surf_wcs, edge_wcs)

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


def get_bbox_minmax(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return (min_point, max_point)

def fix_wires(face):
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        wire_fixer = ShapeFix_Wire(wire, face, 0.01)

        # wire_fixer.SetClosedWireMode(True)
        # wire_fixer.SetFixConnectedMode(True)
        # wire_fixer.SetFixSeamMode(True)

        assert wire_fixer.IsReady()
        wire_fixer.Perform()

def get_bbox_norm(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return np.linalg.norm(max_point - min_point)


def add_pcurves_to_edges(face):
    edge_fixer = ShapeFix_Edge()
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        wire_exp = WireExplorer(wire)
        for edge in wire_exp.ordered_edges():
            edge_fixer.FixAddPCurve(edge, face, False, 0.001)

def fix_face(face):
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(0.1)
    ok = fixer.Perform()
    # assert ok
    fixer.FixOrientation()
    face = fixer.Face()
    return face

def construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj):
    """
    Fit parametric surfaces / curves and trim into B-rep
    """

    # Fit surface bspline
    recon_faces = []
    for points in surf_wcs:
        num_u_points, num_v_points = 32, 32
        uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
        for u_index in range(1,num_u_points+1):
            for v_index in range(1,num_v_points+1):
                pt = points[u_index-1, v_index-1]
                point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                uv_points_array.SetValue(u_index, v_index, point_3d)
        approx_face =  GeomAPI_PointsToBSplineSurface(uv_points_array, 3, 8, GeomAbs_C2, 5e-2).Surface()
        recon_faces.append(approx_face)

    recon_edges = []
    for points in edge_wcs:
        num_u_points = 32
        u_points_array = TColgp_Array1OfPnt(1, num_u_points)
        for u_index in range(1,num_u_points+1):
            pt = points[u_index-1]
            point_2d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            u_points_array.SetValue(u_index, point_2d)
        try:
            approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 5e-3).Curve()
        except Exception as e:

            try:
                approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 8e-3).Curve()
            except Exception as e:

                approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 5e-2).Curve()
        recon_edges.append(approx_edge)

    # Create edges from the curve list
    edge_list = []
    for curve in recon_edges:
        edge = BRepBuilderAPI_MakeEdge(curve).Edge()
        edge_list.append(edge)

    # Cut surface by wire
    post_faces = []
    post_edges = []
    for idx,(surface, edge_incides) in enumerate(zip(recon_faces, FaceEdgeAdj)):
        corner_indices = EdgeVertexAdj[edge_incides]

        # ordered loop
        loops = []
        ordered = [0]
        seen_corners = [corner_indices[0,0], corner_indices[0,1]]
        next_index = corner_indices[0,1]

        while len(ordered)<len(corner_indices):
            while True:
                next_row = [idx for idx, edge in enumerate(corner_indices) if next_index in edge and idx not in ordered]
                if len(next_row) == 0:
                    break
                ordered += next_row
                next_index = list(set(corner_indices[next_row][0]) - set(seen_corners))
                if len(next_index)==0:break
                else: next_index = next_index[0]
                seen_corners += [corner_indices[next_row][0][0], corner_indices[next_row][0][1]]

            cur_len = int(np.array([len(x) for x in loops]).sum()) # add to inner / outer loops
            loops.append(ordered[cur_len:])

            # Swith to next loop
            next_corner =  list(set(np.arange(len(corner_indices))) - set(ordered))
            if len(next_corner)==0:break
            else: next_corner = next_corner[0]
            next_index = corner_indices[next_corner][0]
            ordered += [next_corner]
            seen_corners += [corner_indices[next_corner][0], corner_indices[next_corner][1]]
            next_index = corner_indices[next_corner][1]

        # Determine the outer loop by bounding box length (?)
        bbox_spans = [get_bbox_norm(edge_wcs[x].reshape(-1,3)) for x in loops]

        # Create wire from ordered edges
        _edge_incides_ = [edge_incides[x] for x in ordered]
        edge_post = [edge_list[x] for x in _edge_incides_]
        post_edges += edge_post

        out_idx = np.argmax(np.array(bbox_spans))
        inner_idx = list(set(np.arange(len(loops))) - set([out_idx]))

        # Outer wire
        wire_builder = BRepBuilderAPI_MakeWire()
        for edge_idx in loops[out_idx]:
            wire_builder.Add(edge_list[edge_incides[edge_idx]])
        outer_wire = wire_builder.Wire()

        # Inner wires
        inner_wires = []
        for idx in inner_idx:
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge_idx in loops[idx]:
                wire_builder.Add(edge_list[edge_incides[edge_idx]])
            inner_wires.append(wire_builder.Wire())

        # Cut by wires
        face_builder = BRepBuilderAPI_MakeFace(surface, outer_wire)
        for wire in inner_wires:
            face_builder.Add(wire)
        face_occ = face_builder.Shape()
        fix_wires(face_occ)
        add_pcurves_to_edges(face_occ)
        fix_wires(face_occ)
        face_occ = fix_face(face_occ)
        post_faces.append(face_occ)

    # Sew faces into solid
    sewing = BRepBuilderAPI_Sewing()
    for face in post_faces:
        sewing.Add(face)

    # Perform the sewing operation
    sewing.Perform()
    sewn_shell = sewing.SewedShape()

    # Make a solid from the shell
    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(sewn_shell)
    maker.Build()
    solid = maker.Solid()
    return solid
