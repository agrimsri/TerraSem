"""Unit tests for segmentation metrics.

Tests mIoU against a hand-computed 3×3 confusion matrix example,
as required by BUILD.md §6.
"""

from __future__ import annotations

import numpy as np

from terrasem.metrics.segmentation import SegmentationMetrics


def _make_metrics(**kwargs) -> SegmentationMetrics:
    return SegmentationMetrics(num_classes=3, ignore_index=0, **kwargs)


class TestSegmentationMetrics:
    """Hand-computed reference on a 3-class problem (classes 0=ignore, 1, 2)."""

    def _build_from_conf(self, conf: np.ndarray) -> SegmentationMetrics:
        m = _make_metrics()
        m._conf = conf.copy()
        return m

    def test_miou_hand_computed(self):
        """
        Confusion matrix (rows=GT, cols=pred), ignore_index=0:

              pred0  pred1  pred2
        gt0  [  5,    0,    0 ]   <- ignored
        gt1  [  0,    3,    1 ]
        gt2  [  0,    2,    4 ]

        Class 1: tp=3, fn=1, fp=2 → IoU = 3/(3+1+2) = 0.5
        Class 2: tp=4, fn=2, fp=1 → IoU = 4/(4+2+1) ≈ 0.571
        mIoU = (0.5 + 0.571) / 2 ≈ 0.536
        """
        conf = np.array([
            [5, 0, 0],
            [0, 3, 1],
            [0, 2, 4],
        ], dtype=np.int64)
        m = self._build_from_conf(conf)
        iou1 = 3 / (3 + 1 + 2)
        iou2 = 4 / (4 + 2 + 1)
        expected = (iou1 + iou2) / 2
        assert abs(m.miou() - expected) < 1e-6

    def test_per_class_iou_excludes_ignore(self):
        conf = np.array([
            [5, 0, 0],
            [0, 3, 1],
            [0, 2, 4],
        ], dtype=np.int64)
        m = self._build_from_conf(conf)
        iou = m.per_class_iou()
        assert 0 not in iou, "ignore_index should not appear in per_class_iou"
        assert abs(iou[1] - 3 / 6) < 1e-6
        assert abs(iou[2] - 4 / 7) < 1e-6

    def test_pixel_accuracy(self):
        conf = np.array([
            [5, 0, 0],
            [0, 3, 1],
            [0, 2, 4],
        ], dtype=np.int64)
        m = self._build_from_conf(conf)
        # correct = 3 + 4 = 7; total non-ignored = 3+1+2+4 = 10
        assert abs(m.pixel_accuracy() - 7 / 10) < 1e-6

    def test_perfect_predictions(self):
        conf = np.diag([0, 10, 10]).astype(np.int64)
        m = self._build_from_conf(conf)
        assert abs(m.miou() - 1.0) < 1e-6
        assert abs(m.pixel_accuracy() - 1.0) < 1e-6

    def test_update_accumulates_correctly(self):
        m = _make_metrics()
        pred = np.array([1, 2, 1, 2])
        gt = np.array([1, 2, 2, 1])
        m.update(pred, gt)
        # gt=1,pred=1 → conf[1,1] += 1
        # gt=2,pred=2 → conf[2,2] += 1
        # gt=1,pred=2 → conf[1,2] += 1
        # gt=2,pred=1 → conf[2,1] += 1
        assert m._conf[1, 1] == 1
        assert m._conf[2, 2] == 1
        assert m._conf[1, 2] == 1
        assert m._conf[2, 1] == 1

    def test_ignore_index_excluded_from_update(self):
        m = _make_metrics()
        pred = np.array([1, 2])
        gt = np.array([0, 1])  # first pixel is ignored
        m.update(pred, gt)
        # Only gt=1,pred=2 is valid (gt=0 ignored)
        assert m._conf[1, 2] == 1
        total_valid = m._conf[1:, 1:].sum()
        assert total_valid == 1

    def test_reset(self):
        m = _make_metrics()
        m.update(np.array([1]), np.array([1]))
        m.reset()
        assert m._conf.sum() == 0

    def test_empty_metrics_return_nan(self):
        m = _make_metrics()
        assert np.isnan(m.miou())
        assert np.isnan(m.pixel_accuracy())
