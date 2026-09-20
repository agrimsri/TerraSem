"""2.5D Traversability Cost Map projection from 3D VoxelGrid.

Per BUILD.md Phase 5 Task 3 & Phase 6:
- Flattens 3D voxel grid to 2.5D BEV grid in a height band.
- Column-wise dominant semantic class aggregation.
- Base semantic cost from configs/traversability/costs.yaml.
- Uncertainty-aware cost inflation:
    cost_final = base_cost + lambda_unc * uncertainty
- Converts to standard nav_msgs/OccupancyGrid format:
    int8 in [0, 100], with -1 for unknown/unobserved cells.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from terrasem.datasets.ontology import CLASS_NAMES

if TYPE_CHECKING:
    from terrasem.mapping.voxel_grid import VoxelGrid

# Default semantic costs in [0.0, 1.0]
DEFAULT_SEMANTIC_COSTS: dict[str, float] = {
    "void_unlabeled": 0.5,
    "smooth_traversable": 0.0,
    "rough_traversable": 0.2,
    "high_cost_terrain": 0.6,
    "non_traversable_veg": 1.0,
    "water": 1.0,
    "obstacle_static": 1.0,
    "obstacle_dynamic": 1.0,
    "sky": 0.0,
    "barrier": 1.0,
    "unknown_other": 0.5,
}

LETHAL_CLASSES: set[str] = {"water", "obstacle_static", "obstacle_dynamic", "barrier", "non_traversable_veg"}


class TraversabilityCostMap:
    """2.5D BEV Traversability Cost Map generator.

    Args:
        resolution: Grid cell size in metres (default 0.2m).
        height_band: (z_min, z_max) elevation range to evaluate relative to vehicle.
        lambda_unc: Uncertainty inflation penalty weight (default 0.3).
        semantic_costs: Dict mapping class names to costs in [0.0, 1.0].
    """

    def __init__(
        self,
        resolution: float = 0.2,
        height_band: tuple[float, float] = (-0.5, 2.0),
        lambda_unc: float = 0.3,
        semantic_costs: dict[str, float] | None = None,
    ) -> None:
        self.resolution = float(resolution)
        self.height_band = height_band
        self.lambda_unc = float(lambda_unc)
        self.semantic_costs = semantic_costs or DEFAULT_SEMANTIC_COSTS

        # Array of costs indexed by class ID
        self.cost_by_id = np.array([
            self.semantic_costs.get(name, 0.5) for name in CLASS_NAMES
        ], dtype=np.float32)

    def build_costmap(
        self,
        voxel_grid: VoxelGrid,
        bounds_xy: tuple[float, float, float, float] = (-20.0, 20.0, -20.0, 20.0),
    ) -> dict[str, np.ndarray]:
        """Generate 2.5D BEV costmap arrays from 3D voxel grid.

        Args:
            voxel_grid: Input VoxelGrid instance.
            bounds_xy: (min_x, max_x, min_y, max_y) in metres.

        Returns:
            dict containing:
                "cost": (H, W) float32 in [0.0, 1.0] (nan for unobserved).
                "occupancy_grid": (H, W) int8 in [0, 100] (-1 for unknown).
                "uncertainty": (H, W) float32 normalized entropy [0.0, 1.0].
                "dominant_class": (H, W) int64 class IDs.
                "elevation": (H, W) float32 max height in column.
        """
        min_x, max_x, min_y, max_y = bounds_xy
        z_min_b, z_max_b = self.height_band

        w_cells = int(math.ceil((max_x - min_x) / self.resolution))
        h_cells = int(math.ceil((max_y - min_y) / self.resolution))

        cost_map = np.full((h_cells, w_cells), np.nan, dtype=np.float32)
        unc_map = np.zeros((h_cells, w_cells), dtype=np.float32)
        class_map = np.zeros((h_cells, w_cells), dtype=np.int64)
        elev_map = np.full((h_cells, w_cells), np.nan, dtype=np.float32)

        # Iterate over voxels in grid
        for (ix, iy, iz), vox in voxel_grid.items():
            world_pos = voxel_grid.index_to_world(np.array([ix, iy, iz]))
            wx, wy, wz = float(world_pos[0]), float(world_pos[1]), float(world_pos[2])

            if not (min_x <= wx < max_x and min_y <= wy < max_y):
                continue
            if not (z_min_b <= wz <= z_max_b):
                continue

            cx = int((wx - min_x) / self.resolution)
            cy = int((wy - min_y) / self.resolution)

            if not (0 <= cx < w_cells and 0 <= cy < h_cells):
                continue

            # Update elevation
            if np.isnan(elev_map[cy, cx]) or wz > elev_map[cy, cx]:
                elev_map[cy, cx] = wz

            # Evaluate voxel occupancy & semantics
            occ_prob = vox.occupancy_prob
            if occ_prob > 0.4:  # Occupied or ground hit
                dom_class = vox.dominant_class
                entropy = vox.normalized_entropy

                base_cost = self.cost_by_id[dom_class]
                inflated_cost = base_cost + self.lambda_unc * entropy
                total_cost = float(np.clip(inflated_cost, 0.0, 1.0))

                # Keep worst-case (maximum) cost in vertical column
                if np.isnan(cost_map[cy, cx]) or total_cost > cost_map[cy, cx]:
                    cost_map[cy, cx] = total_cost
                    unc_map[cy, cx] = entropy
                    class_map[cy, cx] = dom_class

        # Convert to ROS OccupancyGrid int8 format
        occ_grid = np.full((h_cells, w_cells), -1, dtype=np.int8)
        observed = ~np.isnan(cost_map)
        occ_grid[observed] = np.clip(cost_map[observed] * 100.0, 0, 100).astype(np.int8)

        return {
            "cost": cost_map,
            "occupancy_grid": occ_grid,
            "uncertainty": unc_map,
            "dominant_class": class_map,
            "elevation": elev_map,
        }
