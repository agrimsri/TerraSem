"""Sparse voxel grid keyed by integer (ix, iy, iz) coordinates.

See BUILD.md Phase 5 for the full specification.
TODO (Phase 5): implement.
"""

from __future__ import annotations

import numpy as np

NUM_CLASSES = 11  # TerraSem-11


class Voxel:
    """Per-voxel data structure."""
    __slots__ = ("log_odds", "class_counts", "hit_count", "miss_count", "z_min", "z_max")

    def __init__(self) -> None:
        self.log_odds: float = 0.0
        self.class_counts: np.ndarray = np.zeros(NUM_CLASSES, dtype=np.float32)
        self.hit_count: int = 0
        self.miss_count: int = 0
        self.z_min: float = float("inf")
        self.z_max: float = float("-inf")


class VoxelGrid:
    """Sparse voxel grid backed by a Python dict.

    TODO (Phase 5): implement world↔voxel transforms, insertion, lookup,
    and the Bayesian occupancy/semantic update logic.
    """

    def __init__(self, voxel_size: float = 0.2) -> None:
        self.voxel_size = voxel_size
        self._grid: dict[tuple[int, int, int], Voxel] = {}

    def world_to_index(self, xyz: np.ndarray) -> np.ndarray:
        """Convert Nx3 world coords to integer voxel indices."""
        raise NotImplementedError("VoxelGrid.world_to_index not yet implemented.")

    def index_to_world(self, idx: np.ndarray) -> np.ndarray:
        """Convert Nx3 integer indices to voxel centroid world coords."""
        raise NotImplementedError("VoxelGrid.index_to_world not yet implemented.")

    def __len__(self) -> int:
        return len(self._grid)
