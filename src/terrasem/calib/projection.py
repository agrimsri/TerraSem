"""Vectorised LiDAR→image projection.

See BUILD.md Phase 3 for the full specification.
TODO (Phase 3): implement fully-vectorised NumPy projection:
    P_cam = R @ P_lidar + t
    keep P_cam.z > min_depth
    uv_h = K @ P_cam
    u = uv_h[0] / uv_h[2]
    v = uv_h[1] / uv_h[2]
    keep 0 <= u < W and 0 <= v < H

Return (u, v, depth, valid_mask, point_indices). No Python loops over points.
"""

from __future__ import annotations

import numpy as np


def project_lidar_to_image(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    image_wh: tuple[int, int],
    min_depth: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project LiDAR points into image plane.

    TODO (Phase 3): implement.

    Returns:
        u, v, depth, valid_mask, point_indices
    """
    raise NotImplementedError("project_lidar_to_image not yet implemented (Phase 3).")
