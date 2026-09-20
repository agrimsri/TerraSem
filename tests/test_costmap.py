"""Tests for 2.5D traversability cost map generation.

See BUILD.md §6 testing requirements:
- flat plane -> traversable
- wall -> lethal
- unknown cells -> -1
"""

from __future__ import annotations

import numpy as np

from terrasem.datasets.ontology import CLASS_TO_ID
from terrasem.mapping.costmap import TraversabilityCostMap
from terrasem.mapping.voxel_grid import VoxelGrid


def test_flat_plane_is_traversable():
    """A synthetic flat plane of smooth_traversable ground must yield low cost and non-lethal cells."""
    grid = VoxelGrid(voxel_size=0.2)
    costmap_gen = TraversabilityCostMap(resolution=0.2, height_band=(-0.5, 1.0), lambda_unc=0.0)

    # Populate a 2m x 2m flat plane at z=0.0 with smooth_traversable (class 1)
    for x in np.arange(-1.0, 1.0, 0.2):
        for y in np.arange(-1.0, 1.0, 0.2):
            idx = grid.world_to_index(np.array([x, y, 0.0]))
            vox = grid.get_or_create(tuple(idx))
            vox.log_odds = 2.0  # occupied
            vox.alpha[CLASS_TO_ID["smooth_traversable"]] += 10.0

    res = costmap_gen.build_costmap(grid, bounds_xy=(-2.0, 2.0, -2.0, 2.0))
    occ_grid = res["occupancy_grid"]
    cost = res["cost"]

    # Observed cells on the plane should have cost == 0.0 (smooth_traversable)
    observed = occ_grid != -1
    assert np.any(observed)
    assert np.all(cost[observed] == 0.0)
    assert np.all(occ_grid[observed] == 0)


def test_wall_is_lethal():
    """A synthetic vertical wall of obstacle_static must produce lethal cells (cost=1.0, grid=100)."""
    grid = VoxelGrid(voxel_size=0.2)
    costmap_gen = TraversabilityCostMap(resolution=0.2, height_band=(-0.5, 2.0), lambda_unc=0.0)

    # Populate a wall along y at x=0.0 from z=0.0 to z=1.5
    for y in np.arange(-1.0, 1.0, 0.2):
        for z in np.arange(0.0, 1.6, 0.2):
            idx = grid.world_to_index(np.array([0.0, y, z]))
            vox = grid.get_or_create(tuple(idx))
            vox.log_odds = 3.0
            vox.alpha[CLASS_TO_ID["obstacle_static"]] += 20.0

    res = costmap_gen.build_costmap(grid, bounds_xy=(-2.0, 2.0, -2.0, 2.0))
    occ_grid = res["occupancy_grid"]
    cost = res["cost"]

    # Wall cells must be 100 (lethal)
    wall_cells = occ_grid == 100
    assert np.any(wall_cells)
    assert np.all(cost[wall_cells] >= 1.0)


def test_unknown_cells_are_minus_one():
    """Unobserved cells outside the populated region must be marked -1 (ROS unknown convention)."""
    grid = VoxelGrid(voxel_size=0.2)
    costmap_gen = TraversabilityCostMap(resolution=0.2)

    # Empty grid
    res = costmap_gen.build_costmap(grid, bounds_xy=(-2.0, 2.0, -2.0, 2.0))
    occ_grid = res["occupancy_grid"]
    cost = res["cost"]

    assert np.all(occ_grid == -1)
    assert np.all(np.isnan(cost))


def test_uncertainty_cost_inflation():
    """Higher entropy uncertainty must increase cost when lambda_unc > 0."""
    grid = VoxelGrid(voxel_size=0.2)
    costmap_with_unc = TraversabilityCostMap(resolution=0.2, lambda_unc=0.5)

    # Cell A: confident smooth_traversable (class 1)
    idx_a = grid.world_to_index(np.array([0.0, 0.0, 0.0]))
    vox_a = grid.get_or_create(tuple(idx_a))
    vox_a.log_odds = 2.0
    vox_a.alpha[1] += 50.0

    # Cell B: uncertain smooth_traversable / rough_traversable / void
    idx_b = grid.world_to_index(np.array([1.0, 0.0, 0.0]))
    vox_b = grid.get_or_create(tuple(idx_b))
    vox_b.log_odds = 2.0
    vox_b.alpha[1] += 1.0
    vox_b.alpha[2] += 1.0
    vox_b.alpha[3] += 1.0

    res = costmap_with_unc.build_costmap(grid, bounds_xy=(-2.0, 2.0, -2.0, 2.0))
    cost = res["cost"]

    ca = cost[int((0.0 - (-2.0)) / 0.2), int((0.0 - (-2.0)) / 0.2)]
    cb = cost[int((0.0 - (-2.0)) / 0.2), int((1.0 - (-2.0)) / 0.2)]

    assert cb > ca
