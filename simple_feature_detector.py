"""Detection and parameter extraction for simple hole features.

This module recognizes simple cylindrical hole features from candidate
regions, including through holes and blind holes. It also extracts basic
geometry and topology parameters for the common Feature output structure.
"""

import numpy as np
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add

import data_class as dc


class SimpleFeatureDetector:
    """Detect through-hole and blind-hole features."""

    def __init__(self, shape_data, adjacency, primitive_regions):
        self.shape_data = shape_data
        self.adjacency = adjacency
        self.regions = primitive_regions
        self.features = []

    def detect(self):
        """Detect simple hole features and return extracted feature objects."""
        for region in self.regions:
            if region.visited:
                continue

            self._classify_region(region)

            if region.type in ("through hole", "blind hole"):
                feature = self._build_feature(region)

                if feature is not None:
                    self.features.append(feature)

        return self.features

    def _classify_region(self, region):
        """Classify a candidate region as a through hole or blind hole."""
        cylinder_faces = [
            face
            for face in region.faces
            if face.surface_type == "cylinder"
        ]
        plane_faces = [
            face
            for face in region.faces
            if face.surface_type == "plane"
        ]

        if len(cylinder_faces) != 1:
            return

        cylinder_face = cylinder_faces[0]
        if not (self._is_inner_cylinder(cylinder_face) and cylinder_face.is_full):
            return

        if len(plane_faces) == 0:
            region.type = "through hole"
            region.visited = True

        if len(plane_faces) == 1:
            region.type = "blind hole"
            region.visited = True

    def _is_inner_cylinder(self, face):
        """Return True if a cylindrical face is an inner cylinder."""
        if face.surface_type != "cylinder":
            return False

        if face.axis is None or face.axis_point is None or face.normal is None:
            return False

        point_on_surface = np.array(face.center)
        axis = np.array(face.axis)
        axis = axis / np.linalg.norm(axis)
        axis_point = np.array(face.axis_point)

        vector_to_surface = point_on_surface - axis_point
        projected_axis_point = axis_point + np.dot(vector_to_surface, axis) * axis
        radius_vector = point_on_surface - projected_axis_point

        normal = np.array(face.normal)
        normal = normal / np.linalg.norm(normal)

        return np.dot(radius_vector, normal) < 0

    def _build_feature(self, region):
        """Convert a hole region into the common Feature structure."""
        cylinder_face = self._find_cylinder_face(region)
        if cylinder_face is None:
            return None

        axis = self._get_hole_axis(region, cylinder_face)
        machining_direction = -axis
        center = self._compute_hole_entry_center(cylinder_face, axis)

        if center is None:
            return None

        geometry = dc.Geometry(
            depth=cylinder_face.depth,
            center=center.tolist(),
            radius=cylinder_face.radius,
            axis=axis.tolist(),
            machining_direction=machining_direction,
            spatial_extent=self._compute_spatial_extent(cylinder_face),
        )

        topology = dc.Topology(
            through=len(region.base_faces) == 0,
            baseline_face=[face.id for face in region.faces if face.is_base],
            connected_faces=[face.id for face in region.faces],
        )

        return dc.Feature(
            feature_id=region.id,
            feature_type=region.type,
            geometry=geometry,
            topology=topology,
        )

    def _find_cylinder_face(self, region):
        """Return the first cylindrical face in a region."""
        for face in region.faces:
            if face.surface_type == "cylinder":
                return face

        return None

    def _get_hole_axis(self, region, cylinder_face):
        """Get the machining axis for the hole feature."""
        if len(region.base_faces) != 0:
            return region.base_faces[0].normal

        axis = cylinder_face.axis
        return axis / np.linalg.norm(axis)

    def _compute_hole_entry_center(self, cylinder_face, axis):
        """Compute the hole entry center from maximum axis projection."""
        axis_point = cylinder_face.axis_point
        projections = []

        for edge in cylinder_face.edges:
            for vertex in edge.vertices:
                point = np.array(vertex.point)
                projections.append(np.dot(point - axis_point, axis))

        if len(projections) == 0:
            return None

        return axis_point + max(projections) * axis

    def _compute_spatial_extent(self, face):
        """Compute the XYZ bounding range of a cylindrical hole face."""
        box = Bnd_Box()
        brepbndlib_Add(face.topods_face, box)

        x_min, y_min, z_min, x_max, y_max, z_max = box.Get()

        return {
            "x": [float(x_min), float(x_max)],
            "y": [float(y_min), float(y_max)],
            "z": [float(z_min), float(z_max)],
        }
