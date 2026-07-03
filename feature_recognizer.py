"""Main pipeline for machining feature recognition.

The recognizer coordinates the complete workflow:
1. read a STEP file;
2. extract geometric data;
3. build the face adjacency graph;
4. group primitive candidate regions;
5. split intersecting feature regions;
6. run feature detectors and export structured results.
"""

import os
import sys

if sys.platform.startswith("win"):
    env = os.path.dirname(sys.executable)

    dll_dirs = [
        os.path.join(env, "Library", "bin"),
        os.path.join(env, "Library", "usr", "bin"),
        os.path.join(env, "Library", "mingw-w64", "bin"),
    ]

    os.environ["PATH"] = ";".join(dll_dirs) + ";" + os.environ["PATH"]

import json
from dataclasses import asdict

import numpy as np

from complex_feature_detector import ComplexFeatureDetector
from feature_splitter import FeatureSplitter
from primitive_grouping import PrimitiveGrouper
from protrusion_detector import ProtrusionDetector
from shape_extractor import extract_shape_data
from simple_feature_detector import SimpleFeatureDetector
from step_reader import read_step_file
from topology_graph import build_face_adjacency


class FeatureRecognizer:
    """Coordinate STEP loading, region processing, and feature detection."""

    def __init__(self):
        self.shape = None
        self.shape_data = None
        self.adjacency = None
        self.regions = None
        self.features = []

    def recognize(self, step_file):
        """Recognize machining features from a STEP file.

        Args:
            step_file (str): Path to the input STEP file.

        Returns:
            dict: Feature recognition result ready for JSON serialization.
        """
        self._reset()

        print("1. Reading STEP...")
        self.shape = read_step_file(step_file)

        print("2. Extracting shape data...")
        self.shape_data = extract_shape_data(self.shape)

        print("3. Building face adjacency...")
        self.adjacency = build_face_adjacency(self.shape, self.shape_data)

        print("4. Primitive grouping...")
        self.regions = self._group_primitives()

        print("5. Splitting features...")
        self.regions = self._split_features()

        print("6. Running feature detectors...")
        self._run_detectors()

        print("Feature recognition completed.")
        return self._build_result()

    def _reset(self):
        """Clear cached state before running a recognition task."""
        self.shape = None
        self.shape_data = None
        self.adjacency = None
        self.regions = None
        self.features = []

    def _group_primitives(self):
        """Group faces into primitive candidate regions."""
        grouper = PrimitiveGrouper(self.shape_data, self.adjacency)
        return grouper.group_primitives()

    def _split_features(self):
        """Split intersecting candidate regions before classification."""
        splitter = FeatureSplitter(self.shape_data, self.adjacency, self.regions)
        return splitter.split()

    def _run_detectors(self):
        """Run feature detectors in the expected recognition order."""
        detectors = [
            ProtrusionDetector(self.shape_data, self.adjacency, self.regions),
            SimpleFeatureDetector(self.shape_data, self.adjacency, self.regions),
            ComplexFeatureDetector(self.shape_data, self.adjacency, self.regions),
        ]

        for detector in detectors:
            detected_features = detector.detect()

            if detected_features:
                self.features.extend(detected_features)

    def _build_result(self):
        """Build the final dictionary result from detected Feature objects."""
        return {
            "feature_count": len(self.features),
            "features": [
                self.remove_empty(asdict(feature))
                for feature in self.features
            ],
        }

    def remove_empty(self, data):
        """Recursively remove None, empty dicts/lists, and serialize arrays."""
        if isinstance(data, dict):
            cleaned_data = {}

            for key, value in data.items():
                cleaned_value = self.remove_empty(value)

                if cleaned_value is None:
                    continue

                if isinstance(cleaned_value, dict) and len(cleaned_value) == 0:
                    continue

                if isinstance(cleaned_value, list) and len(cleaned_value) == 0:
                    continue

                cleaned_data[key] = cleaned_value

            return cleaned_data

        if isinstance(data, list):
            return [
                self.remove_empty(value)
                for value in data
                if value is not None
            ]

        if isinstance(data, np.ndarray):
            if data.size == 0:
                return None

            return data.tolist()

        return data


if __name__ == "__main__":
    recognizer = FeatureRecognizer()
    result = recognizer.recognize(r"test_models\industrial_step_models\a.step")
    print(json.dumps(result, indent=4))
