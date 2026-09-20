"""Tests for VoxelGrid: world<->index round-trip, insertion, lookup, memory scaling.

See BUILD.md §6 testing requirements.
"""

from __future__ import annotations

import numpy as np

from terrasem.mapping.voxel_grid import Voxel, VoxelGrid


def test_world_index_roundtrip():
    """Converting integer index -> world centroid -> integer index must be an exact identity."""
    grid = VoxelGrid(voxel_size=0.2)
    indices = np.array([
        [0, 0, 0],
        [10, -5, 3],
        [-25, -100, 42],
        [1, 2, -3],
    ], dtype=np.int64)

    world_coords = grid.index_to_world(indices)
    roundtrip_indices = grid.world_to_index(world_coords)

    np.testing.assert_array_equal(roundtrip_indices, indices)


def test_insertion_lookup():
    """Voxel creation, field modification, and lookup by tuple key."""
    grid = VoxelGrid(voxel_size=0.1)
    key = (3, -2, 7)

    assert key not in grid
    assert grid.get(key) is None

    voxel = grid.get_or_create(key)
    assert key in grid
    assert grid.get(key) is voxel

    voxel.log_odds = 1.5
    voxel.hit_count = 5
    voxel.z_min = 0.5
    voxel.z_max = 1.2

    retrieved = grid.get(key)
    assert retrieved is not None
    assert abs(retrieved.log_odds - 1.5) < 1e-6
    assert retrieved.hit_count == 5
    assert abs(retrieved.z_min - 0.5) < 1e-6
    assert abs(retrieved.z_max - 1.2) < 1e-6


def test_memory_grows_sublinearly_on_duplicates():
    """Inserting multiple duplicate points falling into the same voxel must not increase grid size."""
    grid = VoxelGrid(voxel_size=0.2)

    # 100 points, all inside voxel (0, 0, 0) whose bounds are [0, 0.2)
    pts = np.random.uniform(0.01, 0.19, size=(100, 3))
    indices = grid.world_to_index(pts)
    for idx in indices:
        grid.get_or_create(tuple(idx))

    assert len(grid) == 1


def test_voxel_properties():
    """Test occupancy probability, Dirichlet probabilities, and entropy."""
    vox = Voxel(prior_alpha=0.1)

    # Initial prior log-odds = 0 -> p = 0.5
    assert abs(vox.occupancy_prob - 0.5) < 1e-6

    # Log-odds = 3.5 -> p ~ 0.97
    vox.log_odds = 3.5
    assert vox.occupancy_prob > 0.95

    # Log-odds = -2.0 -> p ~ 0.119
    vox.log_odds = -2.0
    assert vox.occupancy_prob < 0.2

    # Dirichlet class probabilities
    probs = vox.class_probabilities
    assert abs(np.sum(probs) - 1.0) < 1e-5

    # Uniform prior entropy should be ~ 1.0
    assert abs(vox.normalized_entropy - 1.0) < 1e-3

    # Add strong evidence to class 2 (rough_traversable)
    vox.alpha[2] += 50.0
    assert vox.dominant_class == 2
    assert vox.class_probabilities[2] > 0.9
    # Entropy should drop significantly
    assert vox.normalized_entropy < 0.3
