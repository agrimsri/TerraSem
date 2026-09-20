"""LiDAR fog corruption.

Per BUILD.md Phase 7.3:
Drop points with probability increasing with range + add spurious near returns.
Severity levels: 0 (no-op), 1 to 5.
Deterministic under a fixed seed.
"""

from __future__ import annotations

import numpy as np


def apply(
    points: np.ndarray,
    severity: int,
    seed: int = 42,
) -> np.ndarray:
    """Apply fog attenuation and backscatter noise to a LiDAR point cloud.

    Args:
        points: (N, 3) or (N, 4) numpy array.
        severity: Int from 0 to 5 (0 = no-op).
        seed: Random seed for reproducibility.

    Returns:
        Corrupted point cloud with attenuated distant points and spurious near returns.
    """
    if severity <= 0 or len(points) == 0:
        return points.copy()

    severity = min(severity, 5)
    rng = np.random.RandomState(seed)

    pts = points[:, :3]
    ranges = np.linalg.norm(pts, axis=1)

    # 1. Attenuation: probability of signal penetrating fog decreases with distance
    # Extinction coefficients
    alphas = [0.0, 0.02, 0.04, 0.07, 0.11, 0.16]
    alpha = alphas[severity]

    # Penetration probability p = exp(-alpha * range)
    p_detect = np.exp(-alpha * ranges)
    keep_mask = rng.rand(len(points)) < p_detect
    attenuated_points = points[keep_mask]

    # 2. Spurious backscatter returns (particles reflecting laser close to sensor, e.g. 0.5m - 5.0m)
    num_spurious = [0, 100, 300, 600, 1000, 1600][severity]
    if num_spurious > 0:
        spurious_dirs = rng.normal(size=(num_spurious, 3))
        norms = np.linalg.norm(spurious_dirs, axis=1, keepdims=True)
        spurious_dirs = spurious_dirs / np.maximum(norms, 1e-6)

        spurious_ranges = rng.uniform(0.5, 4.0, size=(num_spurious, 1))
        spurious_xyz = spurious_dirs * spurious_ranges

        if points.shape[1] == 4:
            spurious_intensity = rng.uniform(5.0, 20.0, size=(num_spurious, 1))
            spurious_full = np.hstack([spurious_xyz, spurious_intensity]).astype(points.dtype)
        else:
            spurious_full = spurious_xyz.astype(points.dtype)

        combined = np.vstack([attenuated_points, spurious_full])
        return combined

    return attenuated_points.copy()

