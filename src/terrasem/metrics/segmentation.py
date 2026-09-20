"""Semantic segmentation metrics: mIoU, per-class IoU, pixel accuracy, FW-IoU.

See BUILD.md Phase 4 for the full specification.
All metrics are computed from a confusion matrix (never from raw predictions
directly) so they can be accumulated over batches without bias.

Unit-tested against a hand-computed 3×3 example in tests/test_metrics.py.
"""

from __future__ import annotations

import numpy as np


class SegmentationMetrics:
    """Accumulate predictions into a confusion matrix and compute metrics.

    Args:
        num_classes: Total number of classes (11 for TerraSem-11).
        ignore_index: Class ID to exclude from all metric calculations.
            Must match ``ontology.IGNORE_INDEX``.

    Example::

        metrics = SegmentationMetrics(num_classes=11, ignore_index=0)
        for pred, gt in dataloader:
            metrics.update(pred.numpy(), gt.numpy())
        print(metrics.miou())
    """

    def __init__(self, num_classes: int = 11, ignore_index: int = 0) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._conf: np.ndarray = np.zeros((num_classes, num_classes), dtype=np.int64)

    @property
    def confusion_matrix(self) -> np.ndarray:
        """Accumulated confusion matrix of shape [num_classes, num_classes]."""
        return self._conf.copy()

    def reset(self) -> None:
        """Reset accumulated confusion matrix."""
        self._conf[:] = 0

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """Accumulate predictions and ground-truth labels.

        Args:
            pred: Predicted class IDs, shape [H, W] or [N].
            gt:   Ground-truth class IDs, same shape as *pred*.
        """
        pred = pred.ravel()
        gt = gt.ravel()
        mask = gt != self.ignore_index
        pred = pred[mask]
        gt = gt[mask]
        valid = (gt >= 0) & (gt < self.num_classes) & (pred >= 0) & (pred < self.num_classes)
        np.add.at(self._conf, (gt[valid], pred[valid]), 1)

    def miou(self) -> float:
        """Mean intersection-over-union, excluding ignore_index."""
        iou = self._per_class_iou()
        valid = ~np.isnan(iou)
        return float(np.mean(iou[valid])) if valid.any() else float("nan")

    def per_class_iou(self) -> dict[int, float]:
        """Per-class IoU dict ``{class_id: iou}``."""
        iou = self._per_class_iou()
        return {i: float(iou[i]) for i in range(self.num_classes) if i != self.ignore_index}

    def pixel_accuracy(self) -> float:
        """Overall pixel accuracy excluding ignore_index."""
        conf = self._valid_conf()
        total = conf.sum()
        return float(conf.diagonal().sum() / total) if total > 0 else float("nan")

    def frequency_weighted_iou(self) -> float:
        """Frequency-weighted IoU."""
        conf = self._valid_conf()
        freq = conf.sum(axis=1)
        total = freq.sum()
        if total == 0:
            return float("nan")
        iou = self._per_class_iou()
        valid = ~np.isnan(iou)
        return float((freq[valid] * iou[valid]).sum() / total)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _valid_conf(self) -> np.ndarray:
        """Confusion matrix with ignore_index row/col zeroed out."""
        conf = self._conf.copy()
        conf[self.ignore_index, :] = 0
        conf[:, self.ignore_index] = 0
        return conf

    def _per_class_iou(self) -> np.ndarray:
        """Per-class IoU array (NaN for absent classes)."""
        conf = self._valid_conf()
        tp = conf.diagonal()
        fn = conf.sum(axis=1) - tp
        fp = conf.sum(axis=0) - tp
        denom = tp + fn + fp
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.where(denom > 0, tp / denom, np.nan)
        iou[self.ignore_index] = np.nan
        return iou
