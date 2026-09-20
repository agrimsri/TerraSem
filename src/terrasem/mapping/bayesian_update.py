"""Bayesian occupancy and Dirichlet semantic voxel updates with 3D DDA ray traversal.

Per BUILD.md Phase 5:
- 3D Amanatides-Woo DDA ray traversal for freespace updates.
- Occupancy log-odds update:
    - Free voxels along ray: log-odds += l_free (p_miss=0.4 => -0.4055)
    - Hit voxels: log-odds += l_hit (p_hit=0.7 => +0.8473)
    - Clamped to [l_clamp_min, l_clamp_max] ([-2.0, 3.5])
- Semantic Dirichlet update on hit voxels:
    - alpha[c] += confidence * exp(-range / tau)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from terrasem.mapping.voxel_grid import VoxelGrid

# Log-odds update parameters
P_HIT: float = 0.7
P_MISS: float = 0.4
L_OCC: float = float(np.log(P_HIT / (1.0 - P_HIT)))       # +0.8473
L_FREE: float = float(np.log(P_MISS / (1.0 - P_MISS)))    # -0.4055
L_CLAMP_MIN: float = -2.0
L_CLAMP_MAX: float = 3.5

# Semantic update parameters
TAU: float = 30.0  # Range decay scale in metres


def dda_voxel_traversal(
    origin: np.ndarray,
    endpoint: np.ndarray,
    voxel_size: float,
) -> list[tuple[int, int, int]]:
    """Amanatides-Woo 3D Fast Voxel Traversal algorithm.

    Yields all integer voxel indices (ix, iy, iz) intersected by the ray from origin to endpoint,
    in order, including the starting voxel and the terminating endpoint voxel.

    Args:
        origin: (3,) ray start coordinates in metres.
        endpoint: (3,) ray end coordinates in metres.
        voxel_size: Edge length of cubic voxels in metres.

    Returns:
        List of integer (ix, iy, iz) voxel tuples from origin to endpoint.
    """
    origin = np.asarray(origin, dtype=np.float64)
    endpoint = np.asarray(endpoint, dtype=np.float64)
    vs = float(voxel_size)

    # Starting and ending voxel coordinates
    start_voxel = np.floor(origin / vs).astype(np.int64)
    end_voxel = np.floor(endpoint / vs).astype(np.int64)

    x, y, z = int(start_voxel[0]), int(start_voxel[1]), int(start_voxel[2])
    end_x, end_y, end_z = int(end_voxel[0]), int(end_voxel[1]), int(end_voxel[2])

    direction = endpoint - origin
    dx, dy, dz = direction[0], direction[1], direction[2]

    # Step directions (+1 or -1)
    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
    step_z = 1 if dz > 0 else (-1 if dz < 0 else 0)

    # t_max: distance along ray to next voxel boundary
    # t_delta: distance along ray to traverse one full voxel
    def init_axis(pos: float, step: int, delta_pos: float) -> tuple[float, float]:
        if step == 0 or delta_pos == 0.0:
            return float("inf"), float("inf")
        next_boundary = (math.floor(pos / vs) + 1.0) * vs if step > 0 else math.floor(pos / vs) * vs
        t_max = (next_boundary - pos) / delta_pos

        t_delta = (vs * step) / delta_pos
        return t_max, abs(t_delta)

    t_max_x, t_delta_x = init_axis(origin[0], step_x, dx)
    t_max_y, t_delta_y = init_axis(origin[1], step_y, dy)
    t_max_z, t_delta_z = init_axis(origin[2], step_z, dz)

    traversed: list[tuple[int, int, int]] = [(x, y, z)]

    # Max iteration limit to prevent infinite loops from float precision
    max_steps = int(abs(end_x - x) + abs(end_y - y) + abs(end_z - z) + 4)
    for _ in range(max_steps):
        if x == end_x and y == end_y and z == end_z:
            break

        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                x += step_x
                t_max_x += t_delta_x
            else:
                z += step_z
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                y += step_y
                t_max_y += t_delta_y
            else:
                z += step_z
                t_max_z += t_delta_z

        traversed.append((x, y, z))

    return traversed


def update_voxel_grid(
    voxel_grid: VoxelGrid,
    sensor_origin: np.ndarray,
    hit_points: np.ndarray,
    point_labels: np.ndarray | None = None,
    point_confs: np.ndarray | None = None,
    enable_freespace: bool = True,
    raycast_stride: int = 4,
    l_occ: float = L_OCC,
    l_free: float = L_FREE,
    l_clamp_min: float = L_CLAMP_MIN,
    l_clamp_max: float = L_CLAMP_MAX,
    tau: float = TAU,
) -> None:
    """Perform Bayesian occupancy and Dirichlet semantic updates on a VoxelGrid.

    Args:
        voxel_grid: Target VoxelGrid instance.
        sensor_origin: (3,) sensor origin position in world coordinates.
        hit_points: (N, 3) endpoint hit coordinates in world coordinates.
        point_labels: Optional (N,) integer semantic class IDs.
        point_confs: Optional (N,) float confidence values in [0, 1].
        enable_freespace: Whether to raycast and update free voxels along rays.
        raycast_stride: Raycast every Nth point to optimize freespace runtime.
    """
    if len(hit_points) == 0:
        return

    pts_3d = hit_points[:, :3]
    num_pts = len(pts_3d)

    # 1. Freespace update along rays
    if enable_freespace:
        for idx in range(0, num_pts, max(1, raycast_stride)):
            ray_voxels = dda_voxel_traversal(sensor_origin, pts_3d[idx], voxel_grid.voxel_size)
            # All voxels except the last (hit) are free
            for key in ray_voxels[:-1]:
                vox = voxel_grid.get_or_create(key)
                vox.log_odds = float(np.clip(vox.log_odds + l_free, l_clamp_min, l_clamp_max))
                vox.miss_count += 1

    # 2. Hit voxel occupancy and semantic update
    hit_indices = voxel_grid.world_to_index(pts_3d)

    for i in range(num_pts):
        key = (int(hit_indices[i, 0]), int(hit_indices[i, 1]), int(hit_indices[i, 2]))
        vox = voxel_grid.get_or_create(key)

        # Occupancy update
        vox.log_odds = float(np.clip(vox.log_odds + l_occ, l_clamp_min, l_clamp_max))
        vox.hit_count += 1

        z_val = float(pts_3d[i, 2])
        if z_val < vox.z_min:
            vox.z_min = z_val
        if z_val > vox.z_max:
            vox.z_max = z_val

        # Semantic Dirichlet update
        if point_labels is not None:
            cid = int(point_labels[i])
            if 0 <= cid < len(vox.alpha):
                conf = float(point_confs[i]) if point_confs is not None else 1.0
                dist = float(np.linalg.norm(pts_3d[i] - sensor_origin))
                weight = conf * math.exp(-dist / tau)
                vox.alpha[cid] += weight
