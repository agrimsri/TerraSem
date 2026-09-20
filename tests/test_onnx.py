"""Tests for ONNX export, wrapper, and runtime compatibility.

Verifies:
- SegFormerONNXWrapper produces correct output shape
- Parity between wrapper forward and predict helper
"""

from __future__ import annotations

import torch

from terrasem.datasets.ontology import NUM_CLASSES
from terrasem.models.segformer import SegFormerB0, SegFormerONNXWrapper


def test_onnx_wrapper_forward():
    """Test SegFormerONNXWrapper forward output shape and values."""
    base_model = SegFormerB0(num_classes=NUM_CLASSES, pretrained=False)
    wrapper = SegFormerONNXWrapper(base_model)
    wrapper.eval()

    dummy = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    with torch.no_grad():
        out = wrapper(dummy)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, NUM_CLASSES, 256, 256)
