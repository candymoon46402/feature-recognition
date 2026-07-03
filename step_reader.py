"""STEP file reader.

This module provides a lightweight wrapper around pythonOCC's STEP reader.
It loads a STEP file from disk and converts it into an OpenCascade shape for
the subsequent geometry extraction and feature recognition pipeline.
"""

from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader


def read_step_file(step_path):
    """Read a STEP file and return the corresponding OpenCascade shape.

    Args:
        step_path (str): Path to the STEP file.

    Returns:
        TopoDS_Shape: Shape loaded from the STEP file.

    Raises:
        RuntimeError: If the STEP file cannot be read successfully.
    """
    reader = STEPControl_Reader()
    read_status = reader.ReadFile(step_path)

    if read_status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {step_path}")

    reader.TransferRoots()
    return reader.OneShape()
