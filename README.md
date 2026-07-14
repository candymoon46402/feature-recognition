# A Geometry-Topology Framework for Machining Feature Recognition and Parametric Semantics Extraction from STEP Models

This project implements a machining feature recognition pipeline based on STEP
B-Rep geometry and topology using OpenCascade (pythonOCC). It reads STEP
models, extracts geometric and topological information, constructs a face
adjacency graph, and recognizes machining features through rule-based geometric
reasoning.

The current implementation supports feature recognition together with
parametric semantic extraction, providing geometric parameters and topological
relationships suitable for downstream process planning research.

---

## Supported Features

The current implementation recognizes the following machining features:

- Through hole
- Blind hole
- Through pocket
- Blind pocket
- Step
- Blind slot
- Through slot
- Protrusion

For each recognized feature, the program extracts geometric and topological
information such as:

- feature type
- depth
- radius (if applicable)
- center
- feature axis
- machining direction
- section parameters
- spatial extent
- base faces
- connected faces

---

## Project Structure

```text
.
├── test_models/
├── step_reader.py
├── shape_extractor.py
├── topology_graph.py
├── primitive_grouping.py
├── feature_splitter.py
├── protrusion_detector.py
├── simple_feature_detector.py
├── complex_feature_detector.py
├── feature_recognizer.py
├── data_class.py
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Test Models

The `test_models/` directory contains the STEP models used for evaluating the
feature recognition algorithm in the paper.

These models can be directly used to reproduce the experimental results
reported in the paper.

---

## Environment

The project has been developed and tested with:

- Python 3.7
- pythonocc-core 7.5.1
- OCCT 7.5.1
- NumPy 1.21+

The recommended installation method is Conda.

### Create the environment

```bash
conda env create -f environment.yml
conda activate feature-recognition
```

Alternatively, if you already have a compatible `pythonocc-core`
installation, you can install the remaining Python dependency via

```bash
pip install -r requirements.txt
```

Note that `pythonocc-core` is not included in `requirements.txt`
because it is platform-dependent and is recommended to be installed
through Conda.

---

## Usage

Modify the STEP file path in `feature_recognizer.py`:

```python
if __name__ == "__main__":
    recognizer = FeatureRecognizer()

    result = recognizer.recognize(
        r"path/to/model.step"
    )

    print(json.dumps(result, indent=4))
```

Run

```bash
python feature_recognizer.py
```

or call it from another Python script:

```python
from feature_recognizer import FeatureRecognizer

recognizer = FeatureRecognizer()

result = recognizer.recognize(
    "path/to/model.step"
)

print(result)
```

---

## Output

The recognizer returns a JSON-compatible dictionary.

Example:

```json
{
  "feature_count": 1,
  "features": [
    {
      "feature_id": "...",
      "feature_type": "blind hole",
      "geometry": {
        "depth": 10.00,
        "radius": 2.50,
        "center": [0.00, 0.00, 5.00],
        "axis": [0.00, 0.00, 1.00],
        "machining_direction": [0.00, 0.00, -1.00],
        "spatial_extent": {
          "x": [-2.50, 2.50],
          "y": [-2.50, 2.50],
          "z": [0.00, 10.00]
        }
      },
      "topology": {
        "through": false,
        "baseline_face": [
          "face_id"
        ],
        "connected_faces": [
          "face_id"
        ]
      }
    }
  ]
}
```

Different feature types contain different geometric parameters.

For example:

- hole features provide radius and center;
- pockets, slots, steps, and protrusions provide section parameters.

---

## Recognition Pipeline

The recognition workflow consists of the following stages:

1. Read the STEP model.
2. Extract B-Rep faces, edges, and vertices.
3. Construct the face adjacency graph.
4. Classify adjacent faces as concave or convex.
5. Group faces into primitive candidate regions.
6. Split intersecting candidate regions.
7. Detect protrusions.
8. Detect simple hole features.
9. Detect complex machining features.
10. Generate unified feature representations.
