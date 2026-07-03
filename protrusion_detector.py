"""Detection and parameter extraction for protrusion features.

The detector searches candidate regions for closed side-face loops whose
normals indicate an outward feature. When such a loop is found, it is split
out as a protrusion region and then converted into the common Feature data
structure.
"""

import numpy as np

import data_class as dc
from primitive_grouping import CandidateRegion


class ProtrusionDetector:
    """Detect protrusion features from candidate face regions."""

    def __init__(self, shape_data, adjacency, primitive_regions):
        self.shape_data = shape_data
        self.adjacency = adjacency
        self.regions = primitive_regions
        self.features = []

    def detect(self):
        """Detect protrusions and return extracted feature objects."""
        for region in self.regions:
            if region.type == "protrusion":
                region.visited = True
                feature = self._build_feature(region)

                if feature is not None:
                    self.features.append(feature)

                continue

            if region.visited:
                continue

            if len(region.faces) < 2:
                continue

            self._analyze_side_loops(region)

        return self.features

    def _analyze_side_loops(self, region):
        """Find outward closed side loops and convert them to protrusions."""
        side_faces = [
            face
            for face in region.faces
            if not getattr(face, "is_base", False)
        ]

        if len(side_faces) == 0:
            return None

        closed_loops = self._find_closed_side_loops(side_faces)

        for chain in closed_loops:
            if self._is_protrusion_loop(chain):
                self._create_protrusion_region(region, chain)

        return None

    def _find_closed_side_loops(self, side_faces):
        """Build side-face chains and keep the chains that form closed loops."""
        side_ids = {face.id for face in side_faces}
        visited = set()
        closed_loops = []

        for face in side_faces:
            if face.id in visited:
                continue

            chain = self._build_side_chain(face, side_ids, visited)

            if self._is_closed_chain(chain, side_ids):
                closed_loops.append(chain)

        return closed_loops

    def _build_side_chain(self, start_face, side_ids, visited):
        """Build one connected side-face chain from a start face."""
        chain = [start_face]
        visited.add(start_face.id)

        current_face = start_face
        while True:
            next_face = self._find_unvisited_side_neighbor(
                current_face,
                side_ids,
                visited,
            )

            if next_face is None:
                break

            chain.append(next_face)
            visited.add(next_face.id)
            current_face = next_face

        current_face = start_face
        while True:
            next_face = self._find_unvisited_side_neighbor(
                current_face,
                side_ids,
                visited,
            )

            if next_face is None:
                break

            chain.insert(0, next_face)
            visited.add(next_face.id)
            current_face = next_face

        return chain

    def _find_unvisited_side_neighbor(self, face, side_ids, visited):
        """Return the first adjacent side face that has not been visited."""
        for neighbor in self._get_side_neighbors(face, side_ids):
            if neighbor.id not in visited:
                return neighbor

        return None

    def _get_side_neighbors(self, face, side_ids):
        """Return neighboring faces that are also part of the side-face set."""
        neighbors = []

        for relation in self.adjacency.get(face.id, []):
            neighbor_id = relation["neighbor"]

            if neighbor_id in side_ids:
                neighbors.append(self.shape_data.faces[neighbor_id])

        return neighbors

    def _is_closed_chain(self, chain, side_ids):
        """Check whether a side-face chain forms a closed loop."""
        if len(chain) == 1:
            face = chain[0]
            return face.surface_type == "cylinder" and face.is_full

        head = chain[0]
        tail = chain[-1]

        return (
            len(self._get_side_neighbors(head, side_ids)) == 2
            and len(self._get_side_neighbors(tail, side_ids)) == 2
        )

    def _is_protrusion_loop(self, chain):
        """Return True if a closed chain represents an outward protrusion."""
        if len(chain) == 1 and chain[0].surface_type == "cylinder":
            return not self._is_inner_cylinder(chain[0])

        if len(chain) > 1:
            return self._is_outer_closed(chain)

        return False

    def _create_protrusion_region(self, source_region, chain):
        """Split a protrusion chain out of its source candidate region."""
        source_region_ids = {face.id for face in source_region.faces}

        for relation in self.adjacency.get(chain[0].id, []):
            neighbor_id = relation["neighbor"]

            if neighbor_id in source_region_ids:
                continue

            base_face = self.shape_data.faces[neighbor_id]
            base_face.is_base = True

            protrusion_region = CandidateRegion("protrusion")

            for face in chain:
                protrusion_region.add_face(face)
                source_region.remove_face(face)

            protrusion_region.add_face(base_face)
            protrusion_region.base_faces.append(base_face)
            protrusion_region.closed_loops.append(chain)
            self.regions.append(protrusion_region)
            break

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

    def _is_outer_closed(self, chain):
        """Check whether a closed side loop has outward-facing normals."""
        centers = [np.array(face.center) for face in chain]
        loop_center = np.mean(centers, axis=0)

        signs = []
        for face in chain:
            face_center = np.array(face.center)
            normal = np.array(face.normal)
            center_to_face = face_center - loop_center
            signs.append(np.dot(center_to_face, normal))

        positive_count = sum(sign > 0 for sign in signs)
        negative_count = sum(sign < 0 for sign in signs)

        return positive_count > negative_count

    def _build_feature(self, region):
        """Convert a protrusion region into the common Feature structure."""
        axis = region.base_faces[0].normal
        machining_direction = -axis
        depth = self._compute_feature_depth(region, axis)
        top_point = np.array(region.base_faces[0].center)

        section_parameters = [
            self._extract_chain_section(chain, region, axis, top_point)
            for chain in region.closed_loops
        ]

        geometry = dc.Geometry(
            depth=depth,
            section_parameters=section_parameters,
            axis=axis.tolist(),
            machining_direction=machining_direction,
            spatial_extent=self._compute_spatial_extent(region, axis),
        )

        topology = dc.Topology(
            through=False,
            baseline_face=[face.id for face in region.faces if face.is_base],
            connected_faces=[face.id for face in region.faces],
        )

        return dc.Feature(
            feature_id=region.id,
            feature_type=region.type,
            geometry=geometry,
            topology=topology,
        )

    def _compute_feature_depth(self, region, axis):
        """Compute protrusion height along the base-face normal."""
        depth = 0.0
        base_face = region.base_faces[0]
        base_point = np.array(base_face.center)

        for face in region.faces:
            if face.is_base:
                continue

            for edge in face.edges:
                for vertex in edge.vertices:
                    point = np.array(vertex.point)
                    projection = np.dot(point - base_point, axis)
                    depth = max(depth, abs(projection))

        return depth

    def _extract_chain_section(self, chain, region, axis, top_point):
        """Extract 2D section parameters for a side-face chain."""
        return {
            "chain_type": "closed" if chain in region.closed_loops else "open",
            "faces": [
                self._extract_face_section(face, chain, top_point, axis)
                for face in chain
            ],
        }

    def _extract_face_section(self, face, chain, top_point, normal):
        """Extract projected section data for one face in a chain."""
        if face.surface_type == "cylinder":
            center = np.array(face.axis_point)
            projected_center = self._project_to_plane(center, top_point, normal)

            return {
                "face_id": face.id,
                "surface": "cylinder",
                "center": projected_center.tolist(),
                "radius": face.radius,
            }

        point1, point2 = self._get_face_section_endpoints(
            face,
            chain,
            top_point,
            normal,
        )

        return {
            "face_id": face.id,
            "surface": "plane",
            "start": point1.tolist(),
            "end": point2.tolist(),
            "length": float(np.linalg.norm(point2 - point1)),
        }

    def _project_to_plane(self, point, plane_point, plane_normal):
        """Project a point onto a plane defined by point and normal."""
        point = np.array(point)
        distance = np.dot(point - plane_point, plane_normal)
        return point - distance * plane_normal

    def _get_face_section_endpoints(self, face, chain, top_point, normal):
        """Find representative section endpoints for a planar side face."""
        shared_points = []
        projected_vertices = []

        for edge in face.edges:
            if self._edge_is_shared_with_chain(edge, face, chain):
                projected_point = self._project_to_plane(
                    edge.vertices[0].point,
                    top_point,
                    normal,
                )
                shared_points.append(projected_point)

            if len(shared_points) == 2:
                return shared_points[0], shared_points[1]

        for edge in face.edges:
            for vertex in edge.vertices:
                projected_vertex = self._project_to_plane(
                    vertex.point,
                    top_point,
                    normal,
                )
                projected_vertices.append(np.array(projected_vertex))

        if len(shared_points) == 1:
            farthest_point = self._find_farthest_point(
                shared_points[0],
                projected_vertices,
            )
            return shared_points[0], farthest_point

        return self._find_farthest_point_pair(projected_vertices)

    def _edge_is_shared_with_chain(self, edge, face, chain):
        """Return True if an edge is shared with another face in the chain."""
        for other_face in chain:
            if other_face.id == face.id:
                continue

            if edge in other_face.edges:
                return True

        return False

    def _find_farthest_point(self, reference_point, points):
        """Return the point farthest from the reference point."""
        max_distance = -1
        best_point = None

        for point in points:
            distance = np.linalg.norm(point - reference_point)

            if distance > max_distance:
                max_distance = distance
                best_point = point

        return best_point

    def _find_farthest_point_pair(self, points):
        """Return the pair of points with the largest mutual distance."""
        max_distance = -1
        best_pair = None

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = np.linalg.norm(points[i] - points[j])

                if distance > max_distance:
                    max_distance = distance
                    best_pair = (points[i], points[j])

        return best_pair

    def _compute_spatial_extent(self, region, axis):
        """Compute axis-filtered XYZ bounding ranges for a protrusion."""
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)

        base_face = region.base_faces[0]
        base_point = np.asarray(base_face.center, dtype=float)

        x_min = y_min = z_min = float("inf")
        x_max = y_max = z_max = float("-inf")
        visited_vertices = set()

        for face in region.faces:
            for edge in face.edges:
                for vertex in edge.vertices:
                    if vertex.id in visited_vertices:
                        continue

                    visited_vertices.add(vertex.id)
                    point = np.asarray(vertex.point, dtype=float)
                    projection = np.dot(point - base_point, axis)

                    if projection > 1e-6:
                        continue

                    x_min = min(x_min, point[0])
                    x_max = max(x_max, point[0])
                    y_min = min(y_min, point[1])
                    y_max = max(y_max, point[1])
                    z_min = min(z_min, point[2])
                    z_max = max(z_max, point[2])

        if x_min == float("inf"):
            return {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
            }

        return {
            "x": [float(x_min), float(x_max)],
            "y": [float(y_min), float(y_max)],
            "z": [float(z_min), float(z_max)],
        }
