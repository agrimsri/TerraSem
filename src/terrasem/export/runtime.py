"""ONNX Runtime inference wrapper.

TODO (Phase 8): implement.
"""

from __future__ import annotations

import numpy as np


class ONNXInferenceSession:
    """Thin wrapper around onnxruntime.InferenceSession. TODO (Phase 8): implement."""

    def __init__(self, onnx_path: str) -> None:
        raise NotImplementedError("ONNXInferenceSession not yet implemented.")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
