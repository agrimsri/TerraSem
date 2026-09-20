"""2.5D Traversability Cost Map projection from 3D VoxelGrid.

Per BUILD.md Phase 6:
- Flattens 3D voxel grid to 2.5D BEV grid in a height band.
- Column-wise geometry:
    - step height: h_max - h_ground
    - slope: least-squares plane fit over 3x3 neighbourhood
    - roughness: standard deviation of ground elevations
- Semantic cost: c_sem = cost[dominant_class]
- Geometric cost:
    c_geom = w_s * clip(slope/slope_max, 0, 1)
           + w_h * clip(step/step_max,  0, 1)
           + w_r * clip(rough/rough_max,0, 1)
- Raw fusion:
    c_raw = clip(w_g * c_geom + w_m * c_sem, 0, 1)
- Uncertainty inflation:
    c_final = c_raw + lambda_u * H * (1 - c_raw)
- Lethal cell (100 / 1.0) if:
    p_occ > 0.7 and class in {water, obstacle_static, obstacle_dynamic, barrier, non_traversable_veg}
    or slope > slope_max_hard
    or step  > step_max_hard
- Unknown cells: -1 (ROS OccupancyGrid format)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from terrasem.datasets.ontology import CLASS_NAMES

if TYPE_CHECKING:
    from terrasem.mapping.voxel_grid import VoxelGrid

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

LETHAL_CLASSES: set[str] = {
    "water",
    "obstacle_static",
    "obstacle_dynamic",
    "barrier",
    "non_traversable_veg",
}


class TraversabilityCostMap:
    """2.5D BEV Traversability Cost Map generator with geometric and semantic fusion.

    Args:
        resolution: Grid cell size in metres (default 0.2m).
        height_band: (z_min, z_max) elevation range to evaluate.
        lambda_unc: Uncertainty inflation penalty weight (default 0.3).
        slope_max: Soft maximum slope in degrees (default 25.0°).
        slope_max_hard: Hard maximum lethal slope in degrees (default 35.0°).
        step_max: Soft maximum step height in metres (default 0.25m).
        step_max_hard: Hard maximum lethal step height in metres (default 0.50m).
        rough_max: Maximum terrain roughness in metres (default 0.15m).
        w_slope: Weight for slope in geometric cost (default 0.4).
        w_step: Weight for step height in geometric cost (default 0.4).
        w_rough: Weight for roughness in geometric cost (default 0.2).
        w_geom: Weight for geometric cost in raw fusion (default 0.5).
        w_sem: Weight for semantic cost in raw fusion (default 0.5).
        semantic_costs: Optional custom mapping of class names to base costs.
    """

    def __init__(
        self,
        resolution: float = 0.2,
        height_band: tuple[float, float] = (-1.5, 2.5),
        lambda_unc: float = 0.3,
        slope_max: float = 25.0,
        slope_max_hard: float = 35.0,
        step_max: float = 0.25,
        step_max_hard: float = 0.50,
        rough_max: float = 0.15,
        w_slope: float = 0.4,
        w_step: float = 0.4,
        w_rough: float = 0.2,
        w_geom: float = 0.5,
        w_sem: float = 0.5,
        semantic_costs: dict[str, float] | None = None,
    ) -> None:
        self.resolution = float(resolution)
        self.height_band = height_band
        self.lambda_unc = float(lambda_unc)
        self.slope_max = float(slope_max)
        self.slope_max_hard = float(slope_max_hard)
        self.step_max = float(step_max)
        self.step_max_hard = float(step_max_hard)
        self.rough_max = float(rough_max)
        self.w_slope = float(w_slope)
        self.w_step = float(w_step)
        self.w_rough = float(w_rough)
        self.w_geom = float(w_geom)
        self.w_sem = float(w_sem)
        self.semantic_costs = semantic_costs or DEFAULT_SEMANTIC_COSTS

        self.cost_by_id = np.array([
            self.semantic_costs.get(name, 0.5) for name in CLASS_NAMES
        ], dtype=np.float32)

        self.lethal_ids = {
            i for i, name in enumerate(CLASS_NAMES) if name in LETHAL_CLASSES
        }

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
                "slope": (H, W) float32 slope angle in degrees.
                "step_height": (H, W) float32 step height in metres.
                "roughness": (H, W) float32 roughness in metres.
        """
        min_x, max_x, min_y, max_y = bounds_xy
        z_min_b, z_max_b = self.height_band

        w_cells = int(math.ceil((max_x - min_x) / self.resolution))
        h_cells = int(math.ceil((max_y - min_y) / self.resolution))

        z_min_map = np.full((h_cells, w_cells), np.nan, dtype=np.float32)
        z_max_map = np.full((h_cells, w_cells), np.nan, dtype=np.float32)
        z_samples: list[list[list[float]]] = [[[] for _ in range(w_cells)] for _ in range(h_cells)]

        max_occ_map = np.zeros((h_cells, w_cells), dtype=np.float32)
        class_map = np.zeros((h_cells, w_cells), dtype=np.int64)
        unc_map = np.zeros((h_cells, w_cells), dtype=np.float32)
        has_obs = np.zeros((h_cells, w_cells), dtype=bool)

        # 1. First pass: aggregate voxel data into vertical columns
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

            occ_prob = vox.occupancy_prob
            if occ_prob > 0.4:
                has_obs[cy, cx] = True
                z_samples[cy][cx].append(wz)

                if np.isnan(z_min_map[cy, cx]) or wz < z_min_map[cy, cx]:
                    z_min_map[cy, cx] = wz
                if np.isnan(z_max_map[cy, cx]) or wz > z_max_map[cy, cx]:
                    z_max_map[cy, cx] = wz

                if occ_prob > max_occ_map[cy, cx]:
                    max_occ_map[cy, cx] = occ_prob
                    class_map[cy, cx] = vox.dominant_class
                    unc_map[cy, cx] = vox.normalized_entropy

        # 2. Geometric feature calculation: step height, roughness, slope
        step_map = np.zeros((h_cells, w_cells), dtype=np.float32)
        rough_map = np.zeros((h_cells, w_cells), dtype=np.float32)
        slope_map = np.zeros((h_cells, w_cells), dtype=np.float32)

        for cy in range(h_cells):
            for cx in range(w_cells):
                if not has_obs[cy, cx]:
                    continue

                # Step height
                step_map[cy, cx] = z_max_map[cy, cx] - z_min_map[cy, cx]

                # Roughness
                pts = z_samples[cy][cx]
                if len(pts) > 1:
                    rough_map[cy, cx] = float(np.std(pts))

                # Slope: 3x3 plane fit over ground elevations z_min_map
                pts_plane_x = []
                pts_plane_y = []
                pts_plane_z = []

                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h_cells and 0 <= nx < w_cells and has_obs[ny, nx]:
                            pts_plane_x.append(dx * self.resolution)
                            pts_plane_y.append(dy * self.resolution)
                            pts_plane_z.append(z_min_map[ny, nx])

                if len(pts_plane_z) >= 4:
                    A = np.column_stack([pts_plane_x, pts_plane_y, np.ones(len(pts_plane_z))])
                    b = np.array(pts_plane_z)
                    try:
                        plane_fit, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        norm_z = 1.0
                        norm_x = -plane_fit[0]
                        norm_y = -plane_fit[1]
                        norm_mag = math.sqrt(norm_x**2 + norm_y**2 + norm_z**2)
                        cos_theta = norm_z / max(1e-6, norm_mag)
                        slope_deg = math.degrees(math.acos(np.clip(cos_theta, 0.0, 1.0)))
                        slope_map[cy, cx] = slope_deg
                    except Exception:
                        slope_map[cy, cx] = 0.0

        # 3. Cost fusion
        cost_map = np.full((h_cells, w_cells), np.nan, dtype=np.float32)
        occ_grid = np.full((h_cells, w_cells), -1, dtype=np.int8)

        for cy in range(h_cells):
            for cx in range(w_cells):
                if not has_obs[cy, cx]:
                    continue

                dom_class = class_map[cy, cx]
                entropy = unc_map[cy, cx]
                step = step_map[cy, cx]
                slope = slope_map[cy, cx]
                rough = rough_map[cy, cx]

                # Semantic cost
                c_sem = self.cost_by_id[dom_class]

                # Geometric cost
                c_geom = (
                    self.w_slope * np.clip(slope / self.slope_max, 0.0, 1.0)
                    + self.w_step * np.clip(step / self.step_max, 0.0, 1.0)
                    + self.w_rough * np.clip(rough / self.rough_max, 0.0, 1.0)
                )

                # Raw fusion
                c_raw = np.clip(self.w_geom * c_geom + self.w_sem * c_sem, 0.0, 1.0)

                # Uncertainty inflation: c_final = c_raw + lambda_u * H * (1 - c_raw)
                c_final = float(np.clip(c_raw + self.lambda_unc * entropy * (1.0 - c_raw), 0.0, 1.0))

                # Check lethal conditions
                is_lethal = False
                if max_occ_map[cy, cx] > 0.7 and dom_class in self.lethal_ids or slope > self.slope_max_hard or step > self.step_max_hard:
                    is_lethal = True

                if is_lethal:
                    c_final = 1.0
                    grid_val = 100
                else:
                    grid_val = int(np.clip(round(c_final * 100.0), 0, 100))

                cost_map[cy, cx] = c_final
                occ_grid[cy, cx] = grid_val

        return {
            "cost": cost_map,
            "occupancy_grid": occ_grid,
            "uncertainty": unc_map,
            "dominant_class": class_map,
            "elevation": z_max_map,
            "slope": slope_map,
            "step_height": step_map,
            "roughness": rough_map,
        }


def evaluate_trajectory_agreement(
    cost_map: np.ndarray,
    bounds_xy: tuple[float, float, float, float],
    resolution: float,
    trajectory_xy: np.ndarray,
    low_cost_threshold: float = 0.3,
    lethal_cost_threshold: float = 0.9,
) -> dict[str, Any]:
    """Evaluate proxy traversability agreement against driven robot trajectory.

    Per BUILD.md Phase 6 Task 4:
    - Robot future trajectory is traversable ground truth by definition.
    - trajectory_agreement: fraction of driven cells marked low-cost (< 0.3).
    - false_lethal_rate: fraction of driven cells marked lethal (>= 0.9).

    Args:
        cost_map: (H, W) float32 cost array.
        bounds_xy: (min_x, max_x, min_y, max_y) in metres.
        resolution: Grid resolution in metres.
        trajectory_xy: (N, 2) array of (x, y) coordinates along vehicle path.
        low_cost_threshold: Threshold for drivable cells (default 0.3).
        lethal_cost_threshold: Threshold for lethal cells (default 0.9).

    Returns:
        dict with metrics: 'trajectory_agreement', 'false_lethal_rate', 'mean_cost', 'total_evaluated_cells'.
    """
    min_x, max_x, min_y, max_y = bounds_xy
    h_cells, w_cells = cost_map.shape

    sampled_costs: list[float] = []
    r_cells = max(1, int(math.ceil(1.2 / resolution)))

    for x, y in trajectory_xy:
        if not (min_x <= x < max_x and min_y <= y < max_y):
            continue
        cx = int((x - min_x) / resolution)
        cy = int((y - min_y) / resolution)
        if 0 <= cx < w_cells and 0 <= cy < h_cells:
            val = float(cost_map[cy, cx])
            if not np.isnan(val):
                sampled_costs.append(val)
            else:
                # Search within robot chassis footprint radius (1.2m)
                y0 = max(0, cy - r_cells)
                y1 = min(h_cells, cy + r_cells + 1)
                x0 = max(0, cx - r_cells)
                x1 = min(w_cells, cx + r_cells + 1)
                window = cost_map[y0:y1, x0:x1]
                valid_vals = window[~np.isnan(window)]
                if len(valid_vals) > 0:
                    sampled_costs.append(float(np.min(valid_vals)))

    if not sampled_costs:
        return {
            "trajectory_agreement": 0.0,
            "false_lethal_rate": 0.0,
            "mean_cost": 0.0,
            "total_evaluated_cells": 0,
        }

    costs_arr = np.array(sampled_costs, dtype=np.float32)
    agreement = float(np.mean(costs_arr < low_cost_threshold))
    false_lethal = float(np.mean(costs_arr >= lethal_cost_threshold))
    mean_cost = float(np.mean(costs_arr))

    return {
        "trajectory_agreement": agreement,
        "false_lethal_rate": false_lethal,
        "mean_cost": mean_cost,
        "total_evaluated_cells": len(costs_arr),
    }
