"""Vectorised LiDAR→image projection and semantic point labelling.

See BUILD.md Phase 3.
Fully vectorised NumPy implementation (no loops over points).
Transforms:
    P_cam = R @ P_lidar + t
    keep P_cam.z > min_depth (default 0.5 m)
    uv_h = K @ P_cam
    u = uv_h[0] / uv_h[2]
    v = uv_h[1] / uv_h[2]
    keep 0 <= u < W and 0 <= v < H
"""

from __future__ import annotations

import numpy as np


def project_lidar_to_image(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    image_wh: tuple[int, int],
    min_depth: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project LiDAR 3D points into camera 2D image plane.

    Args:
        points: (N, 3) or (N, 4) float32/float64 LiDAR points [X, Y, Z, (Intensity)].
        R: (3, 3) rotation matrix from LiDAR to camera frame.
        t: (3,) translation vector from LiDAR to camera frame.
        K: (3, 3) camera intrinsic projection matrix.
        image_wh: (W, H) image resolution in pixels.
        min_depth: Minimum forward depth (Z in camera frame) to keep (default 0.5m).

    Returns:
        u: (M,) horizontal pixel coordinates of valid projected points.
        v: (M,) vertical pixel coordinates of valid projected points.
        depth: (M,) depth in metres (Z_cam) of valid projected points.
        valid_mask: (N,) boolean mask indicating which original points projected inside image frustum.
        point_indices: (M,) integer indices into original points array for each valid point.
    """
    points = np.asarray(points)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    K = np.asarray(K, dtype=np.float64)
    W, H = image_wh

    num_points = len(points)
    if num_points == 0:
        empty_f = np.zeros(0, dtype=np.float64)
        empty_b = np.zeros(0, dtype=bool)
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_f, empty_f, empty_f, empty_b, empty_i

    # 1. Transform points from LiDAR to camera coordinate frame: P_cam = P_lidar @ R.T + t
    pts_3d = points[:, :3].astype(np.float64)
    P_cam = (pts_3d @ R.T) + t

    depth_all = P_cam[:, 2]

    # 2. Keep points in front of the camera (Z > min_depth)
    front_mask = depth_all > min_depth
    front_indices = np.nonzero(front_mask)[0]

    if len(front_indices) == 0:
        empty_f = np.zeros(0, dtype=np.float64)
        empty_b = np.zeros(num_points, dtype=bool)
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_f, empty_f, empty_f, empty_b, empty_i

    P_front = P_cam[front_mask]

    # 3. Perspective projection: uv_h = P_front @ K.T
    uv_h = P_front @ K.T
    u_front = uv_h[:, 0] / uv_h[:, 2]
    v_front = uv_h[:, 1] / uv_h[:, 2]

    # 4. Frustum bounds check: 0 <= u < W and 0 <= v < H
    in_frustum = (u_front >= 0.0) & (u_front < float(W)) & (v_front >= 0.0) & (v_front < float(H))

    # 5. Build valid outputs
    point_indices = front_indices[in_frustum]
    u = u_front[in_frustum]
    v = v_front[in_frustum]
    depth = depth_all[point_indices]

    valid_mask = np.zeros(num_points, dtype=bool)
    valid_mask[point_indices] = True

    return u, v, depth, valid_mask, point_indices


def sample_point_semantics(
    u: np.ndarray,
    v: np.ndarray,
    label_map: np.ndarray,
    prob_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample semantic class labels and confidence for projected points.

    Args:
        u: (M,) horizontal pixel coordinates.
        v: (M,) vertical pixel coordinates.
        label_map: (H, W) integer semantic label map.
        prob_map: Optional (H, W) float confidence map (e.g. max softmax probability).

    Returns:
        labels: (M,) integer class labels sampled at (round(v), round(u)).
        confidences: (M,) float confidences in [0.0, 1.0].
    """
    H, W = label_map.shape[:2]
    u_idx = np.clip(np.round(u).astype(np.int64), 0, W - 1)
    v_idx = np.clip(np.round(v).astype(np.int64), 0, H - 1)

    labels = label_map[v_idx, u_idx]

    if prob_map is not None:
        confidences = prob_map[v_idx, u_idx].astype(np.float32)
    else:
        # Default full confidence if no model probabilities available
        confidences = np.ones(len(labels), dtype=np.float32)

    return labels, confidences
