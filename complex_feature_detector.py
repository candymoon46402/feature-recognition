"""Detection and parameter extraction for complex machining features.

This module classifies candidate regions into complex subtractive features,
including through pockets, blind pockets, through slots, blind slots, and
steps. Classification is based on side-face closed loops, open chains, base
faces, and local normal relationships.
"""

import numpy as np

import data_class as dc


class ComplexFeatureDetector:
    """Detect complex pocket, slot, and step features."""

    def __init__(self, shape_data, adjacency, primitive_regions):
        self.shape_data = shape_data
        self.adjacency = adjacency
        self.regions = primitive_regions
        self.features = []

    def detect(self):
        """Detect complex features and return extracted feature objects."""
        for region in self.regions:
            if region.visited:
                continue

            if len(region.faces) < 2:
                continue

            self._analyze_side_loops_and_chains(region)

            feature = self._build_feature(region)
            if feature is not None:
                self.features.append(feature)

        return self.features

    def _analyze_side_loops_and_chains(self, region):
        """Build side-face chains and classify the candidate region."""
        side_faces = [
            face
            for face in region.faces
            if not getattr(face, "is_base", False)
        ]

        if len(side_faces) == 0:
            return None

        closed_loops, open_chains = self._build_side_chains(side_faces)
        region.closed_loops = closed_loops
        region.open_chains = open_chains

        self._classify_region_by_chains(region, closed_loops, open_chains)
        return None

    def _build_side_chains(self, side_faces):
        """Build closed loops and open chains from side faces."""
        side_ids = {face.id for face in side_faces}
        visited = set()
        closed_loops = []
        open_chains = []

        for face in side_faces:
            if face.id in visited:
                continue

            chain = self._build_one_side_chain(face, side_ids, visited)

            if self._is_closed_chain(chain, side_ids):
                closed_loops.append(chain)
            else:
                open_chains.append(chain)

        return closed_loops, open_chains

    def _build_one_side_chain(self, start_face, side_ids, visited):
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
        """Return adjacent faces that are also side faces."""
        neighbors = []

        for relation in self.adjacency.get(face.id, []):
            neighbor_id = relation["neighbor"]

            if neighbor_id in side_ids:
                neighbors.append(self.shape_data.faces[neighbor_id])

        return neighbors

    def _is_closed_chain(self, chain, side_ids):
        """Check whether a side-face chain forms a closed loop."""
        head = chain[0]
        tail = chain[-1]

        return (
            len(self._get_side_neighbors(head, side_ids)) == 2
            and len(self._get_side_neighbors(tail, side_ids)) == 2
        )

    def _classify_region_by_chains(self, region, closed_loops, open_chains):
        """Classify a region using closed-loop and open-chain counts."""
        has_base = len(region.base_faces) != 0
        is_coplanar = self._normals_are_coplanar(region)

        if not has_base and len(closed_loops) == 1 and len(open_chains) == 0 and is_coplanar:
            self._mark_region(region, "through pocket")

        elif not has_base and len(closed_loops) == 0 and len(open_chains) == 1 and is_coplanar:
            self._mark_region(region, "through slot")

        elif len(closed_loops) == 1 and len(open_chains) == 0:
            self._mark_region(region, "blind pocket")

        elif len(closed_loops) == 0 and len(open_chains) == 2:
            self._mark_region(region, "through slot")

        elif len(closed_loops) == 0 and len(open_chains) == 1:
            self._classify_blind_slot_or_step(open_chains[0], region)

        else:
            self._mark_region(region, "other")

    def _mark_region(self, region, feature_type):
        """Set the feature type and mark the region as visited."""
        region.type = feature_type
        region.visited = True

    def _classify_blind_slot_or_step(self, chain, region):
        """Distinguish a blind slot from a step for a single open chain."""
        head = chain[0]
        tail = chain[-1]

        if np.dot(head.normal, tail.normal) < -0.9:
            head_center = head.center
            tail_center = tail.center
            original_distance = np.linalg.norm(head_center - tail_center)
            moved_head = head_center + 0.03 * head.normal
            moved_tail = tail_center + 0.03 * tail.normal
            moved_distance = np.linalg.norm(moved_head - moved_tail)

            if original_distance > moved_distance:
                self._mark_region(region, "blind slot")
                return

            self._mark_region(region, "step")
            return

        external_normals = self._collect_external_normals([head, tail], region)
        if len(external_normals) == 0:
            return None

        normal_groups = self._group_parallel_normals(external_normals)
        group_count = len(normal_groups)

        if group_count == 2:
            self._mark_region(region, "blind slot")

        elif group_count >= 3:
            self._mark_region(region, "step")

        return None

    def _collect_external_normals(self, boundary_faces, region):
        """Collect normals of faces adjacent to the chain but outside region."""
        region_ids = {face.id for face in region.faces}
        external_normals = []

        for face in boundary_faces:
            for relation in self.adjacency.get(face.id, []):
                neighbor_id = relation["neighbor"]

                if neighbor_id in region_ids:
                    continue

                neighbor = self.shape_data.faces[neighbor_id]
                if neighbor.normal is None:
                    continue

                normal = np.array(neighbor.normal)
                normal = normal / np.linalg.norm(normal)
                external_normals.append(normal)

        return external_normals

    def _group_parallel_normals(self, normals):
        """Group normals by near-parallel direction."""
        groups = []

        for normal in normals:
            matched = False

            for group in groups:
                if np.dot(normal, group[0]) > 0.95:
                    group.append(normal)
                    matched = True
                    break

            if not matched:
                groups.append([normal])

        return groups

    def _normals_are_coplanar(self, region):
        """Return True if all available face normals lie in the same plane."""
        normals = []

        for face in region.faces:
            if face.normal is None:
                continue

            normal = face.normal / np.linalg.norm(face.normal)
            normals.append(normal)

        if len(normals) < 3:
            return False

        axis = np.cross(normals[0], normals[1])
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-6:
            return False

        axis = axis / axis_norm

        for normal in normals:
            if abs(np.dot(normal, axis)) > 0.1:
                return False

        return True

    def _build_feature(self, region):
        """Convert a classified region into the common Feature structure."""
        axis = self._compute_feature_axis(region)
        machining_direction = -axis
        depth, top_point = self._compute_feature_depth(region, axis)

        section_parameters = [
            self._extract_chain_section(chain, region, axis, top_point)
            for chain in region.closed_loops + region.open_chains
        ]

        geometry = dc.Geometry(
            depth=depth,
            section_parameters=section_parameters,
            axis=axis.tolist(),
            machining_direction=machining_direction,
            spatial_extent=self._compute_spatial_extent(region, axis),
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

    def _compute_feature_axis(self, region):
        """Compute the feature axis from base face or side-face normals."""
        if len(region.base_faces) != 0:
            return region.base_faces[0].normal

        side_faces = [
            face
            for chain in region.closed_loops + region.open_chains
            for face in chain
        ]

        for i in range(len(side_faces)):
            for j in range(i + 1, len(side_faces)):
                normal1 = np.array(side_faces[i].normal)
                normal2 = np.array(side_faces[j].normal)
                cross = np.cross(normal1, normal2)
                cross_norm = np.linalg.norm(cross)

                if cross_norm > 1e-6:
                    return cross / cross_norm

        return np.array(side_faces[0].normal)

    def _compute_feature_depth(self, region, axis):
        """Compute feature depth and a reference top point."""
        if len(region.base_faces) == 0:
            return self._compute_through_feature_depth(region, axis)

        return self._compute_blind_feature_depth(region, axis)

    def _compute_through_feature_depth(self, region, axis):
        """Compute depth for features without a base face."""
        origin = np.array(region.faces[0].edges[0].vertices[0].point)
        projections = []
        max_projection = -1e9
        top_point = None

        for face in region.faces:
            for edge in face.edges:
                for vertex in edge.vertices:
                    point = np.array(vertex.point)
                    projection = np.dot(point - origin, axis)
                    projections.append(projection)

                    if projection > max_projection:
                        max_projection = projection
                        top_point = point

        depth = max(projections) - min(projections)
        return depth, top_point

    def _compute_blind_feature_depth(self, region, axis):
        """Compute depth for features with a base face."""
        depth = 0.0
        top_point = None
        base_face = region.base_faces[0]
        base_point = np.array(base_face.center)

        for face in region.faces:
            if face.is_base:
                continue

            for edge in face.edges:
                for vertex in edge.vertices:
                    point = np.array(vertex.point)
                    projection = np.dot(point - base_point, axis)

                    if projection > depth:
                        depth = projection
                        top_point = point

        return depth, top_point

    def _extract_chain_section(self, chain, region, axis, top_point):
        """Extract 2D section parameters for one side-face chain."""
        return {
            "chain_type": "closed" if chain in region.closed_loops else "open",
            "faces": [
                self._extract_face_section(face, chain, top_point, axis)
                for face in chain
            ],
        }

    def _extract_face_section(self, face, chain, top_point, normal):
        """Extract projected section data for one face."""
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
        """Compute XYZ bounding ranges for a complex feature."""
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)

        has_base = len(region.base_faces) > 0
        if has_base:
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

                    if has_base:
                        projection = np.dot(point - base_point, axis)

                        if projection < -1e-6:
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
