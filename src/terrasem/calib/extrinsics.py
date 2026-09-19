"""LiDAR→camera extrinsic transform parsing.

See BUILD.md Phase 3 for the full specification.
TODO (Phase 3): implement + document convention in docs/calibration_notes.md.

CRITICAL: must record in calibration_notes.md —
  - quaternion order (xyzw vs wxyz)
  - whether transform is T_cam_lidar or its inverse
  - axis convention
Justify with the visual projection test.
"""

from __future__ import annotations

import numpy as np


class Extrinsics:
    """LiDAR→camera extrinsic transform T_cam_lidar.

    TODO (Phase 3): parse from RELLIS-3D calibration files.
    """

    def __init__(self, R: np.ndarray, t: np.ndarray) -> None:
        self.R = R   # (3, 3)
        self.t = t   # (3,)
