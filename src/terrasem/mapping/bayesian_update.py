"""Bayesian occupancy and Dirichlet semantic voxel updates.

See BUILD.md Phase 5 for the full specification.
Includes 3D DDA (Amanatides-Woo) ray traversal.
TODO (Phase 5): implement.
"""

from __future__ import annotations

import numpy as np

# Log-odds parameters (BUILD.md Phase 5)
L_OCC: float = 0.8473  # log(0.7 / 0.3)
L_FREE: float = -0.4055  # log(0.4 / 0.6)
L_CLAMP_MIN: float = -2.0
L_CLAMP_MAX: float = 3.5

# Semantic update parameters
ALPHA: float = 0.1       # Dirichlet prior
TAU: float = 30.0        # range weight decay (m)
NUM_CLASSES: int = 11


def dda_voxel_traversal(
    origin: np.ndarray,
    endpoint: np.ndarray,
    voxel_size: float,
) -> list[tuple[int, int, int]]:
    """Return the ordered list of voxel indices along the ray origin→endpoint.

    TODO (Phase 5): implement 3D Amanatides-Woo DDA.
    Unit-tested in tests/test_bayesian_update.py.
    """
    raise NotImplementedError("dda_voxel_traversal not yet implemented (Phase 5).")
