"""Primitive candidate region grouping.

The grouping stage grows candidate feature regions on the face adjacency
graph. Faces connected through concave relations are collected into the same
region, and isolated cylindrical faces are also kept for hole detection.
"""

import uuid


class CandidateRegion:
    """A candidate feature region composed of related faces."""

    def __init__(self, region_type):
        self.id = str(uuid.uuid4())
        self.type = region_type
        self.faces = []
        self.visited = False
        self.base_faces = []
        self.closed_loops = []
        self.open_chains = []

    def add_face(self, face):
        """Add a face to the region."""
        self.faces.append(face)

    def remove_face(self, face):
        """Remove a face from the region."""
        self.faces.remove(face)


class PrimitiveGrouper:
    """Group primitive faces into candidate feature regions."""

    def __init__(self, shape_data, adjacency):
        self.shape_data = shape_data
        self.adjacency = adjacency
        self.visited = set()
        self.regions = []

    def group_primitives(self):
        """Group faces connected by concave adjacency relations."""
        for face in self.shape_data.faces.values():
            if face.id in self.visited:
                continue

            region = self._grow_region(face)

            if self._should_keep_region(region):
                self.regions.append(region)

        return self.regions

    def _grow_region(self, start_face):
        """Grow one candidate region with depth-first traversal."""
        region = CandidateRegion("candidate")
        stack = [start_face]

        while stack:
            current_face = stack.pop()

            if current_face.id in self.visited:
                continue

            self.visited.add(current_face.id)
            region.add_face(current_face)

            for neighbor in self._iter_concave_neighbors(current_face):
                if neighbor.id not in self.visited:
                    stack.append(neighbor)

        return region

    def _iter_concave_neighbors(self, face):
        """Yield neighboring faces connected through concave edges."""
        for relation in self.adjacency.get(face.id, []):
            if relation["type"] != "concave":
                continue

            neighbor_id = relation["neighbor"]
            yield self.shape_data.faces[neighbor_id]

    def _should_keep_region(self, region):
        """Keep multi-face regions and isolated cylindrical faces."""
        if len(region.faces) > 1:
            return True

        return region.faces[0].surface_type == "cylinder"
