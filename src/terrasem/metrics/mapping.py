"""Voxel-level mapping metrics: semantic accuracy, mIoU, ECE.

See BUILD.md Phase 5 for the full specification.
TODO (Phase 5): implement voxel mIoU and ECE.
"""

from __future__ import annotations

import numpy as np


def voxel_miou(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    num_classes: int = 11,
    ignore_index: int = 0,
) -> float:
    """Compute voxel-level mean Intersection over Union (mIoU).

    Args:
        pred_labels: (N,) predicted integer class labels.
        gt_labels: (N,) ground-truth integer class labels.
        num_classes: Total number of classes.
        ignore_index: Class ID to exclude from evaluation (default 0).

    Returns:
        Scalar mIoU in [0.0, 1.0].
    """
    from terrasem.metrics.segmentation import SegmentationMetrics

    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    metrics.update(pred_labels, gt_labels)
    return float(metrics.miou())


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    ignore_index: int = 0,
) -> float:
    """Compute Expected Calibration Error (ECE) for voxel probability predictions.

    Args:
        probs: (N, C) predicted class probabilities.
        labels: (N,) ground-truth integer class IDs.
        n_bins: Number of confidence bins (default 15).
        ignore_index: Class ID to exclude (default 0).

    Returns:
        Scalar ECE in [0.0, 1.0].
    """
    from terrasem.metrics.calibration_metrics import (
        expected_calibration_error as ece_fn,
    )

    return ece_fn(probs, labels, n_bins=n_bins, ignore_index=ignore_index)

