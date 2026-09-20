"""Tests for calibration metrics (ECE, Brier score, reliability diagram, temperature scaling)."""

from __future__ import annotations

import numpy as np

from terrasem.metrics.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram,
    temperature_scale,
)


def test_ece_perfect_calibration():
    """Perfect predictions should have near-zero ECE."""
    # 100 samples of class 1, predicted with 1.0 probability
    probs = np.zeros((100, 3), dtype=np.float32)
    probs[:, 1] = 1.0
    labels = np.ones(100, dtype=np.int64)

    ece = expected_calibration_error(probs, labels, ignore_index=0)
    assert ece < 1e-4


def test_ece_overconfident_mispredictions():
    """Completely overconfident wrong predictions should have high ECE."""
    probs = np.zeros((100, 3), dtype=np.float32)
    probs[:, 1] = 1.0  # predicts class 1 with 100% confidence
    labels = np.full(100, 2, dtype=np.int64)  # true class is 2

    ece = expected_calibration_error(probs, labels, ignore_index=0)
    assert abs(ece - 1.0) < 1e-4


def test_brier_score_bounds():
    """Brier score is 0.0 for perfect prediction and bounded by 2.0."""
    probs = np.zeros((50, 4), dtype=np.float32)
    probs[:, 2] = 1.0
    labels = np.full(50, 2, dtype=np.int64)
    bs = brier_score(probs, labels, num_classes=4, ignore_index=0)
    assert abs(bs - 0.0) < 1e-5

    # Completely wrong
    probs[:, 2] = 0.0
    probs[:, 1] = 1.0
    bs_wrong = brier_score(probs, labels, num_classes=4, ignore_index=0)
    assert abs(bs_wrong - 2.0) < 1e-5


def test_temperature_scaling():
    """Temperature > 1.0 should soften probabilities toward uniform; T < 1 should sharpen."""
    logits = np.array([[2.0, 0.0, -1.0]])

    p_base = temperature_scale(logits, temperature=1.0)
    p_soft = temperature_scale(logits, temperature=2.0)
    p_sharp = temperature_scale(logits, temperature=0.5)

    assert p_base.shape == (1, 3)
    # Highest confidence should be lower with higher temperature
    assert np.max(p_soft) < np.max(p_base)
    # Highest confidence should be higher with lower temperature
    assert np.max(p_sharp) > np.max(p_base)


def test_reliability_diagram():
    """Check reliability diagram dictionary output."""
    probs = np.array([
        [0.1, 0.9, 0.0],
        [0.2, 0.8, 0.0],
        [0.0, 0.3, 0.7],
    ], dtype=np.float32)
    labels = np.array([1, 1, 2], dtype=np.int64)

    diag = reliability_diagram(probs, labels, n_bins=10, ignore_index=0)
    assert "bin_accuracies" in diag
    assert "bin_confidences" in diag
    assert "bin_counts" in diag
    assert "ece" in diag
    assert len(diag["bin_accuracies"]) == 10

