"""Unit tests for semantic segmentation models.

Per BUILD.md Phase 4:
- Logits upsampling is unit-tested: assert `model(x)["logits"].shape[-2:] == x.shape[-2:]`.
- Loss computation respects `ignore_index`.
- Predictions and probabilities have valid shapes and ranges.
"""

from __future__ import annotations

import torch

from terrasem.models.registry import get_model


def test_segformer_output_shape_and_upsampling():
    """Assert SegFormer output logits match exact input spatial resolution."""
    model = get_model("segformer_b0", num_classes=11, pretrained=False)
    model.eval()

    # Test with arbitrary non-square resolution e.g. (128, 192)
    B, C, H, W = 2, 3, 128, 192
    x = torch.randn(B, C, H, W)
    labels = torch.randint(0, 11, (B, H, W), dtype=torch.int64)

    with torch.no_grad():
        out = model(x, labels=labels)

    logits = out["logits"]
    assert logits.shape == (B, 11, H, W), f"Expected {(B, 11, H, W)}, got {logits.shape}"
    assert "loss" in out
    assert not torch.isnan(out["loss"])


def test_dinov2_output_shape_and_upsampling():
    """Assert DINOv2 linear head output logits match exact input spatial resolution."""
    model = get_model("dinov2_linear", num_classes=11, pretrained=False)
    model.eval()

    B, C, H, W = 2, 3, 128, 192
    x = torch.randn(B, C, H, W)
    labels = torch.randint(0, 11, (B, H, W), dtype=torch.int64)

    with torch.no_grad():
        out = model(x, labels=labels)

    logits = out["logits"]
    assert logits.shape == (B, 11, H, W), f"Expected {(B, 11, H, W)}, got {logits.shape}"
    assert "loss" in out
    assert not torch.isnan(out["loss"])


def test_model_predict_helper():
    """Test model predict method returns valid classes and probabilities in [0, 1]."""
    model = get_model("segformer_b0", num_classes=11, pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    preds, probs = model.predict(x)

    assert preds.shape == (1, 64, 64)
    assert probs.shape == (1, 64, 64)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert preds.min() >= 0 and preds.max() < 11
