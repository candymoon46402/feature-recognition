"""Data structures for recognized machining features.

The detector modules use these dataclasses as a common output schema. The
objects are later converted to dictionaries and serialized as JSON-compatible
feature recognition results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Geometry:
    """Geometric parameters of a recognized feature."""

    section_parameters: Any = field(default_factory=list)
    depth: Optional[float] = None
    radius: Optional[float] = None
    center: Optional[List[float]] = None
    axis: Optional[List[float]] = None
    machining_direction: Optional[List[float]] = None
    spatial_extent: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class Topology:
    """Topological relationships of a recognized feature."""

    through: Optional[bool] = None
    baseline_face: List[str] = field(default_factory=list)
    connected_faces: List[str] = field(default_factory=list)


@dataclass
class Feature:
    """Complete feature record containing type, geometry, and topology."""

    feature_id: str
    feature_type: str
    geometry: Geometry = field(default_factory=Geometry)
    topology: Topology = field(default_factory=Topology)
