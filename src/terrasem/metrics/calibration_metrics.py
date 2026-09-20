"""Uncertainty calibration metrics (ECE, Brier Score, Reliability Diagrams).

Per BUILD.md Phase 5 Task 4 & §12 (Guo et al., ICML 2017):
- Expected Calibration Error (ECE): binned absolute gap between accuracy and confidence.
- Brier Score: mean squared error between probability distribution and one-hot ground truth.
- Reliability diagram generation.
- Temperature scaling helper.
"""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    ignore_index: int = 0,
) -> float:
    """Compute Expected Calibration Error (ECE) for multi-class predictions.

    Args:
        probs: (N, C) or (H, W, C) predicted class probabilities in [0.0, 1.0].
        labels: (N,) or (H, W) ground truth integer class IDs.
        n_bins: Number of confidence bins (default 15).
        ignore_index: Ground truth class ID to ignore (default 0).

    Returns:
        Scalar ECE value in [0.0, 1.0].
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    if probs.ndim > 2:
        probs = probs.reshape(-1, probs.shape[-1])
    labels = labels.ravel()

    # Filter out ignore_index
    valid = labels != ignore_index
    if not np.any(valid):
        return 0.0

    probs = probs[valid]
    labels = labels[valid]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for b in range(n_bins):
        bin_lower = bin_boundaries[b]
        bin_upper = bin_boundaries[b + 1]

        # Samples in bin
        if b == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 11,
    ignore_index: int = 0,
) -> float:
    """Compute Brier Score: mean squared error of predicted probabilities.

    Args:
        probs: (N, C) predicted class probabilities.
        labels: (N,) ground truth integer class IDs.
        num_classes: Total number of classes.
        ignore_index: Class ID to exclude.

    Returns:
        Scalar Brier score in [0.0, 2.0].
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    if probs.ndim > 2:
        probs = probs.reshape(-1, probs.shape[-1])
    labels = labels.ravel()

    valid = (labels != ignore_index) & (labels >= 0) & (labels < num_classes)
    if not np.any(valid):
        return 0.0

    probs = probs[valid]
    labels = labels[valid]

    # One-hot encode ground truth
    one_hot = np.eye(num_classes, dtype=np.float32)[labels]
    # Sum of squared errors per sample, then mean across samples
    sq_errors = np.sum((probs - one_hot) ** 2, axis=-1)
    return float(np.mean(sq_errors))


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    ignore_index: int = 0,
) -> dict[str, list[float] | float]:
    """Compute reliability diagram data for plotting.

    Returns:
        dict containing 'bin_accuracies', 'bin_confidences', 'bin_counts', 'ece'.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    if probs.ndim > 2:
        probs = probs.reshape(-1, probs.shape[-1])
    labels = labels.ravel()

    valid = labels != ignore_index
    if not np.any(valid):
        return {"bin_accuracies": [], "bin_confidences": [], "bin_counts": [], "ece": 0.0}

    probs = probs[valid]
    labels = labels[valid]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    total = len(confidences)

    for b in range(n_bins):
        bin_lower = bin_boundaries[b]
        bin_upper = bin_boundaries[b + 1]

        if b == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        count = int(np.sum(in_bin))
        bin_counts.append(count)
        if count > 0:
            acc = float(np.mean(accuracies[in_bin]))
            conf = float(np.mean(confidences[in_bin]))
            bin_accs.append(acc)
            bin_confs.append(conf)
            ece += (count / total) * abs(acc - conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)

    return {
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "ece": float(ece),
    }


def temperature_scale(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling to logits and compute calibrated softmax probabilities."""
    scaled = logits / max(1e-5, temperature)
    exp_s = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    return exp_s / np.sum(exp_s, axis=-1, keepdims=True)
