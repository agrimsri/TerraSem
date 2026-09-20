"""LiDAR range noise corruption.

Per BUILD.md Phase 7.3:
Additive Gaussian noise on range, sigma = 0.02 - 0.10 m.
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
    """Apply Gaussian range noise to LiDAR point cloud.

    Args:
        points: (N, 3) or (N, 4) numpy array.
        severity: Int from 0 to 5 (0 = no-op).
        seed: Random seed for reproducibility.

    Returns:
        Points with noisy ranges, preserving original shape and dtype.
    """
    if severity <= 0 or len(points) == 0:
        return points.copy()

    severity = min(severity, 5)
    rng = np.random.RandomState(seed)

    # Sigmas from 0.02m to 0.10m
    sigmas = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    sigma = sigmas[severity]

    pts = points[:, :3].copy()
    ranges = np.linalg.norm(pts, axis=1)
    valid = ranges > 1e-4

    noise = rng.normal(0.0, sigma, size=len(points)).astype(pts.dtype)
    perturbed_ranges = np.maximum(0.1, ranges + noise)

    scale = np.ones(len(points), dtype=pts.dtype)
    scale[valid] = perturbed_ranges[valid] / ranges[valid]

    out = points.copy()
    out[:, :3] = pts * scale[:, None]
    return out
