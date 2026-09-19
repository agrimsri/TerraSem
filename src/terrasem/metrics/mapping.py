"""Voxel-level mapping metrics: semantic accuracy, mIoU, ECE.

See BUILD.md Phase 5 for the full specification.
TODO (Phase 5): implement voxel mIoU and ECE.
"""

from __future__ import annotations

import numpy as np


def voxel_miou(pred_labels: np.ndarray, gt_labels: np.ndarray, num_classes: int = 11) -> float:
    """Voxel-level mIoU. TODO (Phase 5): implement."""
    raise NotImplementedError("voxel_miou not yet implemented (Phase 5).")


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error (ECE). TODO (Phase 5): implement."""
    raise NotImplementedError("ECE not yet implemented (Phase 5).")
