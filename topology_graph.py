"""Face adjacency graph construction.

This module builds an adjacency graph between faces in a shape. Each graph
edge stores the neighboring face id, the shared OpenCascade edge, and the
local relation type ("concave" or "convex").
"""

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods_Edge, topods_Face
from OCC.Core.gp import gp_Pnt, gp_Vec


def _shape_id(shape):
    """Return a stable string id for a TopoDS object within this process."""
    return str(shape.HashCode(1000000))


def _point_to_array(point):
    """Convert an OpenCascade point to a numpy array."""
    return np.array([point.X(), point.Y(), point.Z()])


def _iter_shape_edges(shape):
    """Yield all edges of a shape."""
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        yield topods_Edge(edge_explorer.Current())
        edge_explorer.Next()


def _iter_shape_faces(shape):
    """Yield all faces of a shape."""
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        yield topods_Face(face_explorer.Current())
        face_explorer.Next()


def _iter_face_edges(face):
    """Yield all edges of a face."""
    edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while edge_explorer.More():
        yield topods_Edge(edge_explorer.Current())
        edge_explorer.Next()


def _build_edge_to_faces(shape):
    """Map each edge id to the faces that share that edge."""
    edge_to_faces = {_shape_id(edge): [] for edge in _iter_shape_edges(shape)}

    for face in _iter_shape_faces(shape):
        for edge in _iter_face_edges(face):
            edge_id = _shape_id(edge)
            if edge_id in edge_to_faces:
                edge_to_faces[edge_id].append(face)

    return edge_to_faces


def _has_neighbor(adjacency_list, neighbor_id):
    """Return True if the adjacency list already contains the neighbor."""
    return any(item["neighbor"] == neighbor_id for item in adjacency_list)


def _add_adjacency(adjacency, face_id, neighbor_id, relation_type, edge):
    """Add one directed adjacency record if it does not already exist."""
    if _has_neighbor(adjacency[face_id], neighbor_id):
        return

    adjacency[face_id].append({
        "neighbor": neighbor_id,
        "type": relation_type,
        "edge": edge,
    })


def build_face_adjacency(shape, shape_data):
    """Build a face adjacency graph with concave/convex relation labels."""
    adjacency = {face_id: [] for face_id in shape_data.faces}
    edge_to_faces = _build_edge_to_faces(shape)

    for edge in _iter_shape_edges(shape):
        shared_faces = edge_to_faces.get(_shape_id(edge), [])

        if len(shared_faces) != 2:
            continue

        face1, face2 = shared_faces
        face1_id = _shape_id(face1)
        face2_id = _shape_id(face2)

        if face1_id not in shape_data.faces or face2_id not in shape_data.faces:
            continue

        relation_type = compute_relation_type(face1, face2, edge)

        _add_adjacency(adjacency, face1_id, face2_id, relation_type, edge)
        _add_adjacency(adjacency, face2_id, face1_id, relation_type, edge)

    return adjacency


def compute_relation_type(face1, face2, edge):
    """Classify the local relation between two faces as concave or convex."""
    edge_direction = get_edge_direction(edge)
    if edge_direction is None:
        return None

    edge_midpoint = get_edge_midpoint(edge)
    if edge_midpoint is None:
        return None

    normal1 = get_face_normal(face1, edge_midpoint)
    normal2 = get_face_normal(face2, edge_midpoint)
    if normal1 is None or normal2 is None:
        return None

    tangent1 = np.cross(normal1, edge_direction)
    tangent2 = np.cross(normal2, -edge_direction)

    tangent1_norm = np.linalg.norm(tangent1)
    tangent2_norm = np.linalg.norm(tangent2)
    if tangent1_norm < 1e-8 or tangent2_norm < 1e-8:
        return None

    tangent1 = tangent1 / tangent1_norm
    tangent2 = tangent2 / tangent2_norm

    offset = 0.05
    point1_guess = edge_midpoint + offset * tangent1
    point2_guess = edge_midpoint + offset * tangent2

    point1 = project_point_to_face(point1_guess, face1)
    point2 = project_point_to_face(point2_guess, face2)
    if point1 is None or point2 is None:
        return None

    original_distance = np.linalg.norm(point1 - point2)

    normal_offset = 0.03
    normal1_at_point = get_face_normal(face1, point1)
    normal2_at_point = get_face_normal(face2, point2)
    if normal1_at_point is None or normal2_at_point is None:
        return None

    moved_point1 = point1 + normal_offset * normal1_at_point
    moved_point2 = point2 + normal_offset * normal2_at_point
    moved_distance = np.linalg.norm(moved_point1 - moved_point2)

    if moved_distance < original_distance:
        return "concave"

    return "convex"


def get_edge_midpoint(edge):
    """Return the midpoint of an OpenCascade edge."""
    curve, first, last = BRep_Tool.Curve(edge)
    if curve is None:
        return None

    mid_parameter = (first + last) * 0.5
    point = gp_Pnt()
    curve.D0(mid_parameter, point)

    return _point_to_array(point)


def get_edge_direction(edge):
    """Return the normalized tangent direction at the edge midpoint."""
    curve, first, last = BRep_Tool.Curve(edge)
    if curve is None:
        return None

    mid_parameter = (first + last) * 0.5
    point = gp_Pnt()
    tangent = gp_Vec()
    curve.D1(mid_parameter, point, tangent)

    direction = np.array([tangent.X(), tangent.Y(), tangent.Z()])
    direction_norm = np.linalg.norm(direction)

    if direction_norm < 1e-8:
        return None

    if edge.Orientation() == TopAbs_REVERSED:
        direction = -direction

    return direction / direction_norm


def get_face_normal(face, point):
    """Return the face normal at the location nearest to the input point."""
    query_point = gp_Pnt(float(point[0]), float(point[1]), float(point[2]))
    surface = BRep_Tool.Surface(face)
    surface_analyzer = ShapeAnalysis_Surface(surface)

    uv = surface_analyzer.ValueOfUV(query_point, 1e-6)
    props = GeomLProp_SLProps(surface, uv.X(), uv.Y(), 1, 1e-6)

    if not props.IsNormalDefined():
        return None

    normal = np.array([
        props.Normal().X(),
        props.Normal().Y(),
        props.Normal().Z(),
    ])

    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        return None

    if face.Orientation() == TopAbs_REVERSED:
        normal = -normal

    return normal / normal_norm


def project_point_to_face(point, face):
    """Project a 3D point onto a face and return the nearest point."""
    query_point = gp_Pnt(float(point[0]), float(point[1]), float(point[2]))
    vertex = BRepBuilderAPI_MakeVertex(query_point).Vertex()

    distance_solver = BRepExtrema_DistShapeShape(vertex, face)
    distance_solver.Perform()

    if not distance_solver.IsDone():
        return None

    nearest = distance_solver.PointOnShape2(1)
    return _point_to_array(nearest)
