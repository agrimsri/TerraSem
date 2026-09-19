"""Camera intrinsics parsing and undistortion.

See BUILD.md Phase 3 for the full specification.
TODO (Phase 3): implement after calibration files are inspected in Phase 2.
"""

from __future__ import annotations

import numpy as np


class CameraIntrinsics:
    """Camera intrinsics K and distortion coefficients.

    TODO (Phase 3): parse from RELLIS-3D calibration YAML/txt files.
    Verify undistortion visually before any downstream use.
    """

    def __init__(self, K: np.ndarray, dist: np.ndarray | None = None) -> None:
        self.K = K
        self.dist = dist if dist is not None else np.zeros(5)
