"""Rain corruption for sensor degradation robustness evaluation.

See BUILD.md Phase 7.3 for the full specification.
TODO (Phase 7): implement with severity levels 1-5.
Severity 0 must be a no-op. Must be deterministic under fixed seed.
"""

from __future__ import annotations

import numpy as np


def apply(data: np.ndarray, severity: int, seed: int = 42) -> np.ndarray:
    """Apply rain corruption at given severity (0=no-op, 1-5 increasing).

    TODO (Phase 7): implement.
    """
    if severity == 0:
        return data
    raise NotImplementedError("rain corruption not yet implemented (Phase 7).")
