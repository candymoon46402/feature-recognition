"""Geometry extraction utilities for STEP shapes.

This module converts an OpenCascade shape into lightweight Python data
objects. It extracts face, edge, and vertex information used by topology
analysis and feature recognition.
"""

import math

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods_Edge, topods_Face, topods_Vertex


class VertexData:
    """Minimal vertex representation used by the recognition pipeline."""

    def __init__(self):
        self.id = None
        self.point = None


class EdgeData:
    """Minimal edge representation with references to its vertices."""

    def __init__(self):
        self.id = None
        self.vertices = []


class FaceData:
    """Geometric and topological attributes extracted from a face."""

    def __init__(self):
        self.id = None
        self.surface_type = None
        self.normal = None
        self.axis = None
        self.center = None
        self.radius = None
        self.topods_face = None
        self.edges = []
        self.axis_point = None
        self.is_full = False
        self.depth = None
        self.is_base = False


class ShapeData:
    """Container for all extracted faces, edges, and vertices."""

    def __init__(self):
        self.faces = {}
        self.edges = {}
        self.vertices = {}


def _shape_id(shape):
    """Return a stable string id for a TopoDS object within this process."""
    return str(shape.HashCode(1000000))


def _point_to_array(point):
    """Convert an OpenCascade point to a numpy array."""
    return np.array([point.X(), point.Y(), point.Z()])


def compute_face_geometry(face):
    """Compute a representative center point and outward normal of a face."""
    adaptor = BRepAdaptor_Surface(face)

    u_mid = (adaptor.FirstUParameter() + adaptor.LastUParameter()) * 0.5
    v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) * 0.5

    props = BRepLProp_SLProps(adaptor, u_mid, v_mid, 1, 1e-6)
    if not props.IsNormalDefined():
        return None, None

    center = _point_to_array(props.Value())
    normal = np.array([
        props.Normal().X(),
        props.Normal().Y(),
        props.Normal().Z(),
    ])

    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        return center, None

    normal = normal / normal_norm
    if face.Orientation() == TopAbs_REVERSED:
        normal = -normal

    return center, normal


def get_cylinder_axis(adaptor):
    """Extract the axis direction of a cylindrical surface."""
    direction = adaptor.Cylinder().Axis().Direction()
    return np.array([direction.X(), direction.Y(), direction.Z()])


def get_cylinder_radius(adaptor):
    """Extract the radius of a cylindrical surface."""
    return adaptor.Cylinder().Radius()


def _is_full_cylinder(adaptor):
    """Check whether the cylindrical face covers a full 360-degree angle."""
    angle = abs(adaptor.LastUParameter() - adaptor.FirstUParameter())
    return abs(angle - 2.0 * math.pi) < 1e-3


def _compute_cylinder_depth(face, axis):
    """Estimate cylinder depth from vertex projections along its axis."""
    axis = axis / np.linalg.norm(axis)
    projections = []

    edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while edge_explorer.More():
        edge = topods_Edge(edge_explorer.Current())

        vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        while vertex_explorer.More():
            vertex = topods_Vertex(vertex_explorer.Current())
            point = _point_to_array(BRep_Tool.Pnt(vertex))
            projections.append(np.dot(point, axis))
            vertex_explorer.Next()

        edge_explorer.Next()

    if not projections:
        return 0

    return max(projections) - min(projections)


def _fill_cylinder_attributes(face, adaptor, face_data):
    """Fill cylinder-specific attributes for a face."""
    face_data.surface_type = "cylinder"
    face_data.axis = get_cylinder_axis(adaptor)
    face_data.radius = get_cylinder_radius(adaptor)
    face_data.is_full = _is_full_cylinder(adaptor)

    axis_location = adaptor.Cylinder().Axis().Location()
    face_data.axis_point = _point_to_array(axis_location)
    face_data.depth = _compute_cylinder_depth(face, face_data.axis)


def _extract_or_create_vertex(vertex, shape_data):
    """Return existing vertex data, or create it if it has not been seen."""
    vertex_id = _shape_id(vertex)

    if vertex_id not in shape_data.vertices:
        vertex_data = VertexData()
        vertex_data.id = vertex_id
        vertex_data.point = _point_to_array(BRep_Tool.Pnt(vertex))
        shape_data.vertices[vertex_id] = vertex_data

    return shape_data.vertices[vertex_id]


def _extract_or_create_edge(edge, shape_data):
    """Return existing edge data, or extract its vertices and create it."""
    edge_id = _shape_id(edge)

    if edge_id in shape_data.edges:
        return shape_data.edges[edge_id]

    edge_data = EdgeData()
    edge_data.id = edge_id

    vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    while vertex_explorer.More():
        vertex = topods_Vertex(vertex_explorer.Current())
        edge_data.vertices.append(_extract_or_create_vertex(vertex, shape_data))
        vertex_explorer.Next()

    shape_data.edges[edge_id] = edge_data
    return edge_data


def _attach_edges_to_face(face, face_data, shape_data):
    """Extract all edges of a face and attach them to the face data."""
    edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)

    while edge_explorer.More():
        edge = topods_Edge(edge_explorer.Current())
        face_data.edges.append(_extract_or_create_edge(edge, shape_data))
        edge_explorer.Next()


def extract_face_data(face, shape_data):
    """Extract geometric and topological data from a single face."""
    adaptor = BRepAdaptor_Surface(face)
    surface_type = adaptor.GetType()

    face_data = FaceData()
    face_data.id = _shape_id(face)
    face_data.topods_face = face

    center, normal = compute_face_geometry(face)
    face_data.center = center
    face_data.normal = normal

    if surface_type == GeomAbs_Plane:
        face_data.surface_type = "plane"
    elif surface_type == GeomAbs_Cylinder:
        _fill_cylinder_attributes(face, adaptor, face_data)
    else:
        face_data.surface_type = "other"

    _attach_edges_to_face(face, face_data, shape_data)
    return face_data


def extract_shape_data(shape):
    """Extract face, edge, and vertex data from an OpenCascade shape."""
    shape_data = ShapeData()

    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods_Face(face_explorer.Current())
        face_data = extract_face_data(face, shape_data)
        shape_data.faces[face_data.id] = face_data
        face_explorer.Next()

    return shape_data
