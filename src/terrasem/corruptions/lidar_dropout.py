"""LiDAR beam and point dropout corruptions.

Per BUILD.md Phase 7.3:
Randomly drop whole rings/beams at 10, 25, 40, 50, 75%.
Severity levels: 0 (no-op), 1 to 5.
Deterministic under a fixed seed.
"""

from __future__ import annotations

import numpy as np


def apply(
    points: np.ndarray,
    severity: int,
    seed: int = 42,
    num_rings: int = 64,
) -> np.ndarray:
    """Apply beam / ring dropout to a LiDAR point cloud.

    Args:
        points: (N, 3) or (N, 4) numpy array of LiDAR points.
        severity: Int from 0 to 5 (0 = no-op).
        seed: Random seed for reproducibility.
        num_rings: Estimated number of vertical beam rings in the sensor (default 64 for OS1-64).

    Returns:
        Subsampled point cloud with dropped rings removed.
    """
    if severity <= 0 or len(points) == 0:
        return points.copy()

    severity = min(severity, 5)
    rng = np.random.RandomState(seed)

    # Drop fraction per severity
    drop_fractions = [0.0, 0.10, 0.25, 0.40, 0.50, 0.75]
    drop_fraction = drop_fractions[severity]

    pts = points[:, :3]
    ranges = np.linalg.norm(pts, axis=1)
    valid_range = ranges > 1e-3

    # Estimate ring index based on pitch / elevation angle
    # pitch = arcsin(z / range)
    pitch = np.zeros(len(points), dtype=np.float32)
    pitch[valid_range] = np.arcsin(np.clip(pts[valid_range, 2] / ranges[valid_range], -1.0, 1.0))

    # Discretize into ring bins
    min_pitch, max_pitch = float(np.min(pitch)), float(np.max(pitch))
    if max_pitch == min_pitch:
        # Fallback to random point dropout if pitch is degenerate
        keep_mask = rng.rand(len(points)) >= drop_fraction
        return points[keep_mask].copy()

    ring_ids = np.clip(
        np.floor((pitch - min_pitch) / (max_pitch - min_pitch + 1e-6) * num_rings).astype(int),
        0,
        num_rings - 1,
    )

    # Randomly select rings to drop
    num_dropped_rings = int(round(num_rings * drop_fraction))
    dropped_rings = set(rng.choice(num_rings, size=num_dropped_rings, replace=False))

    keep_mask = np.array([rid not in dropped_rings for rid in ring_ids], dtype=bool)
    return points[keep_mask].copy()
