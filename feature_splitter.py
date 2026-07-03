"""Split intersecting candidate regions into independent feature regions.

Primitive grouping may merge intersecting or vertically stacked machining
features into one candidate region. This module uses base-face detection,
projection ranges, and face adjacency to split such regions before the final
feature classifiers are executed.
"""

import numpy as np

from primitive_grouping import CandidateRegion


class FeatureSplitter:
    """Split grouped candidate regions into cleaner feature candidates."""

    def __init__(self, shape_data, adjacency, regions):
        self.shape_data = shape_data
        self.adjacency = adjacency
        self.regions = regions

    def split(self):
        """Split all unvisited candidate regions."""
        new_regions = []

        for region in self.regions:
            if hasattr(region, "visited") and region.visited:
                continue

            new_regions.extend(self._split_intersecting_features(region))

        return new_regions

    def _split_intersecting_features(self, region):
        """Split one candidate region using detected base faces."""
        base_faces = self._find_base_faces(region)
        if len(base_faces) == 0:
            return [region]

        axis = np.array(base_faces[0].normal)
        axis = axis / np.linalg.norm(axis)

        face_ranges = self._compute_face_z_ranges(region, axis)
        sorted_base_faces = sorted(
            base_faces,
            key=lambda face: face_ranges[face.id][0],
        )

        eps = 1e-5
        sub_regions = []

        lower_faces = self._collect_faces_below_lowest_base(
            region,
            sorted_base_faces,
            face_ranges,
            eps,
        )

        for chain in self._split_into_chains(lower_faces):
            sub_regions.append(self._build_subregion(chain, []))

        for base_face in sorted_base_faces:
            feature_faces = self._collect_faces_for_base(
                base_face,
                face_ranges,
                eps,
            )
            side_faces = [
                face
                for face in feature_faces
                if not getattr(face, "is_base", False)
            ]
            sub_regions.append(self._build_subregion(side_faces, [base_face]))

        return self._merge_same_side_features(sub_regions, face_ranges)

    def _collect_faces_below_lowest_base(
            self,
            region,
            sorted_base_faces,
            face_ranges,
            eps):
        """Collect side faces extending below the lowest base face."""
        lowest_base = sorted_base_faces[0]
        lowest_z = face_ranges[lowest_base.id][0]
        base_face_ids = {face.id for face in sorted_base_faces}

        lower_faces = []
        for face in region.faces:
            if face.id in base_face_ids:
                continue

            z_min, _ = face_ranges[face.id]
            if z_min < lowest_z - eps:
                lower_faces.append(face)

        return lower_faces

    def _collect_faces_for_base(self, base_face, face_ranges, eps):
        """Collect faces that belong to the feature grown from one base."""
        base_z = face_ranges[base_face.id][0]
        feature_faces = {base_face}

        for relation in self.adjacency.get(base_face.id, []):
            neighbor = self.shape_data.faces[relation["neighbor"]]
            if neighbor.id not in face_ranges:
                continue

            _, z_max = face_ranges[neighbor.id]
            if z_max > base_z + eps:
                feature_faces.add(neighbor)

        should_expand = not self._forms_closed_loop(feature_faces)
        changed = True

        while should_expand and changed:
            changed = False

            for face in list(feature_faces):
                for relation in self.adjacency.get(face.id, []):
                    neighbor = self.shape_data.faces[relation["neighbor"]]

                    if neighbor in feature_faces:
                        continue

                    if neighbor.id not in face_ranges:
                        continue

                    z_min, z_max = face_ranges[neighbor.id]
                    if z_min < base_z + eps < z_max:
                        feature_faces.add(neighbor)
                        changed = True

        return feature_faces

    def _merge_same_side_features(self, sub_regions, face_ranges):
        """Merge split regions that share identical side faces and base height."""
        if len(sub_regions) <= 1:
            return sub_regions

        eps = 1e-4
        merged = []
        used_indices = set()

        for i, region1 in enumerate(sub_regions):
            if i in used_indices:
                continue

            used_indices.add(i)
            side_face_ids = self._get_side_face_ids(region1)
            base_faces = list(region1.base_faces)
            base_z = self._get_first_base_z(region1, face_ranges)

            for j, region2 in enumerate(sub_regions):
                if j <= i or j in used_indices:
                    continue

                if len(region2.base_faces) == 0:
                    continue

                if side_face_ids != self._get_side_face_ids(region2):
                    continue

                other_base_z = self._get_first_base_z(region2, face_ranges)
                if abs(base_z - other_base_z) > eps:
                    continue

                used_indices.add(j)
                for base_face in region2.base_faces:
                    if base_face not in base_faces:
                        base_faces.append(base_face)

            merged.append(self._rebuild_region_with_base_faces(region1, base_faces))

        return merged

    def _get_side_face_ids(self, region):
        """Return ids of non-base faces in a region."""
        return {
            face.id
            for face in region.faces
            if not getattr(face, "is_base", False)
        }

    def _get_first_base_z(self, region, face_ranges):
        """Return the projection height of the first base face."""
        if len(region.base_faces) == 0:
            return None

        return face_ranges[region.base_faces[0].id][0]

    def _rebuild_region_with_base_faces(self, region, base_faces):
        """Create a new region while preserving side faces and merged bases."""
        new_region = CandidateRegion(region.type)
        new_region.faces = list(region.faces)

        for base_face in base_faces:
            if base_face not in new_region.faces:
                new_region.faces.append(base_face)

        new_region.base_faces = base_faces
        return new_region

    def _compute_face_z_ranges(self, region, axis):
        """Compute projection range of each face along a given axis."""
        face_ranges = {}

        for face in region.faces:
            projections = []

            for edge in face.edges:
                for vertex in edge.vertices:
                    point = np.array(vertex.point)
                    projections.append(np.dot(point, axis))

            if len(projections) == 0:
                continue

            face_ranges[face.id] = (min(projections), max(projections))

        return face_ranges

    def _split_into_chains(self, faces):
        """Split a set of faces into connected chains by adjacency."""
        face_ids = {face.id for face in faces}
        visited = set()
        chains = []

        for face in faces:
            if face.id in visited:
                continue

            chain = [face]
            visited.add(face.id)
            stack = [face]

            while stack:
                current = stack.pop()

                for relation in self.adjacency.get(current.id, []):
                    neighbor_id = relation["neighbor"]

                    if neighbor_id not in face_ids or neighbor_id in visited:
                        continue

                    neighbor = self.shape_data.faces[neighbor_id]
                    chain.append(neighbor)
                    visited.add(neighbor_id)
                    stack.append(neighbor)

            chains.append(chain)

        return chains

    def _build_subregion(self, side_faces, base_faces):
        """Build a candidate region from side faces and optional base faces."""
        region = CandidateRegion("candidate")

        for face in side_faces:
            region.add_face(face)

        for base_face in base_faces:
            region.add_face(base_face)

        region.base_faces = base_faces
        return region

    def _find_base_faces(self, region):
        """Find and mark base faces in a candidate region."""
        faces = [face for face in region.faces if face.normal is not None]

        if len(faces) < 2:
            return []

        if len(faces) == 2 and self._all_faces_are_planes(faces):
            score0 = self._average_right_angle_deviation(faces[0])
            score1 = self._average_right_angle_deviation(faces[1])

            if score0 < score1:
                faces[0].is_base = True
                return [faces[0]]

            faces[1].is_base = True
            return [faces[1]]

        if len(faces) == 3 and self._normals_are_coplanar(region):
            return self._find_middle_base_face(faces)

        if len(faces) > 3 and self._normals_are_coplanar(region):
            if self._is_open_chain_like_region(faces):
                return []

        return self._find_best_base_face_by_angle(faces)

    def _all_faces_are_planes(self, faces):
        """Return True if all faces are planar."""
        return all(face.surface_type == "plane" for face in faces)

    def _average_right_angle_deviation(self, face):
        """Score a face by how close its neighbors are to 90 degrees."""
        angles = []

        for relation in self.adjacency.get(face.id, []):
            neighbor = self.shape_data.faces[relation["neighbor"]]

            if neighbor.normal is None:
                continue

            neighbor_normal = np.array(neighbor.normal)
            cos_angle = np.clip(np.dot(neighbor_normal, face.normal), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            angles.append(abs(angle - 90))

        return np.mean(angles)

    def _find_middle_base_face(self, faces):
        """For a three-face coplanar region, find the middle planar base."""
        face_ids = {face.id for face in faces}
        middle_face = None

        for face in faces:
            neighbor_count = 0

            for relation in self.adjacency.get(face.id, []):
                if relation["neighbor"] in face_ids:
                    neighbor_count += 1

            if neighbor_count == 2:
                middle_face = face
                break

        if middle_face is None or middle_face.surface_type != "plane":
            return []

        side_faces = [face for face in faces if face.id != middle_face.id]
        if len(side_faces) != 2:
            return []

        middle_normal = np.array(middle_face.normal)
        normal1 = np.array(side_faces[0].normal)
        normal2 = np.array(side_faces[1].normal)

        if abs(np.dot(middle_normal, normal1)) < 0.1 and abs(np.dot(middle_normal, normal2)) < 0.1:
            middle_face.is_base = True
            return [middle_face]

        return []

    def _is_open_chain_like_region(self, faces):
        """Return True if each face has at most two neighbors in the region."""
        region_face_ids = {face.id for face in faces}

        for face in faces:
            visited_neighbors = set()
            neighbor_count = 0

            for relation in self.adjacency.get(face.id, []):
                neighbor_id = relation["neighbor"]

                if neighbor_id in visited_neighbors:
                    continue

                if neighbor_id not in region_face_ids:
                    continue

                neighbor_count += 1
                visited_neighbors.add(neighbor_id)

            if neighbor_count > 2:
                return False

        return True

    def _find_best_base_face_by_angle(self, faces):
        """Select the planar face whose normal is most orthogonal to others."""
        best_face = None
        best_score = float("inf")

        for face in faces:
            if face.surface_type == "cylinder":
                continue

            score = self._base_face_angle_score(face, faces)

            if score is None:
                continue

            if score < best_score:
                best_score = score
                best_face = face

        if best_face is None:
            return []

        return self._mark_parallel_base_faces(faces, best_face)

    def _base_face_angle_score(self, face, faces):
        """Compute the base-face score used by the original splitter logic."""
        face_normal = np.array(face.normal)
        angles = []

        for other in faces:
            if face.id == other.id:
                continue

            other_normal = np.array(other.normal)
            cos_angle = np.clip(np.dot(face_normal, other_normal), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            if abs(angle) < 5:
                if other.surface_type == "cylinder":
                    return None
                continue

            angles.append(abs(angle - 90))

        if len(angles) == 0:
            return None

        return np.mean(angles)

    def _mark_parallel_base_faces(self, faces, best_face):
        """Mark faces parallel to the selected best face as base faces."""
        base_faces = []
        base_normal = np.array(best_face.normal)

        for face in faces:
            normal = np.array(face.normal)
            cos_angle = np.clip(np.dot(normal, base_normal), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            if abs(angle) < 5:
                face.is_base = True
                base_faces.append(face)

        return base_faces

    def _forms_closed_loop(self, faces):
        """Check whether the non-base faces form a closed side loop."""
        region_face_ids = {
            face.id
            for face in faces
            if not face.is_base
        }

        for face in faces:
            if face.is_full:
                continue

            visited_neighbors = set()
            neighbor_count = 0

            for relation in self.adjacency.get(face.id, []):
                neighbor_id = relation["neighbor"]

                if neighbor_id in visited_neighbors:
                    continue

                if neighbor_id not in region_face_ids:
                    continue

                neighbor_count += 1
                visited_neighbors.add(neighbor_id)

            if neighbor_count < 2:
                return False

        return True

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
