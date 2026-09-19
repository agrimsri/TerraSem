"""2.5D traversability cost map from voxel grid.

See BUILD.md Phase 6 for the full specification.
TODO (Phase 6): implement.
"""

from __future__ import annotations

import numpy as np


class TraversabilityCostMap:
    """2.5D traversability cost map compatible with nav_msgs/OccupancyGrid.

    TODO (Phase 6): implement projection + cost fusion logic.
    """

    def __init__(self, resolution: float = 0.2) -> None:
        self.resolution = resolution

    def build(self, voxel_grid) -> np.ndarray:
        """Build cost map from voxel_grid. Returns int8 array (0-100, -1=unknown)."""
        raise NotImplementedError("TraversabilityCostMap.build not yet implemented.")
