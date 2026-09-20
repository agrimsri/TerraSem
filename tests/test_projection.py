"""Unit tests for LiDAR→image projection.

Per BUILD.md Phase 3 & §6:
- Synthetic point at known 3D location with identity extrinsics projects
  to analytically computed pixel within 1e-4.
- Out-of-frustum and behind-camera points are filtered.
- CPU latency of 65k-point cloud projection is measured and validated (<10 ms).
"""

from __future__ import annotations

import numpy as np

from terrasem.calib.extrinsics import Extrinsics
from terrasem.calib.intrinsics import CameraIntrinsics
from terrasem.calib.projection import project_lidar_to_image, sample_point_semantics
from terrasem.utils.timing import benchmark


class TestProjection:
    def test_projection_identity_extrinsics_analytic(self):
        """Construct synthetic point at known 3D location; assert projected pixel within 1e-4."""
        fx, fy, cx, cy = 1000.0, 1000.0, 500.0, 400.0
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        image_wh = (1000, 800)  # W=1000, H=800

        R = np.eye(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)

        # 3D Point: X=2.5, Y=1.5, Z=10.0
        # Analytic projection:
        # u = fx * (X/Z) + cx = 1000 * 0.25 + 500 = 750.0
        # v = fy * (Y/Z) + cy = 1000 * 0.15 + 400 = 550.0
        # depth = 10.0
        points = np.array([[2.5, 1.5, 10.0]], dtype=np.float64)

        u, v, depth, valid_mask, indices = project_lidar_to_image(
            points, R, t, K, image_wh, min_depth=0.5
        )

        assert len(u) == 1
        assert valid_mask[0] is np.True_ or valid_mask[0] == True  # noqa: E712
        assert abs(u[0] - 750.0) < 1e-4, f"Expected u=750.0, got {u[0]}"
        assert abs(v[0] - 550.0) < 1e-4, f"Expected v=550.0, got {v[0]}"
        assert abs(depth[0] - 10.0) < 1e-4, f"Expected depth=10.0, got {depth[0]}"
        assert indices[0] == 0

    def test_out_of_frustum_and_behind_camera_filtered(self):
        """Points behind camera or outside image bounds must be filtered."""
        fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        image_wh = (640, 480)
        R = np.eye(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)

        points = np.array([
            [0.0, 0.0, 5.0],     # Point 0: exactly on optical axis at 5m -> inside (u=320, v=240)
            [0.0, 0.0, -2.0],    # Point 1: behind camera (Z = -2.0) -> filtered
            [0.0, 0.0, 0.2],     # Point 2: closer than min_depth=0.5 -> filtered
            [500.0, 0.0, 2.0],   # Point 3: far to the right -> out of frustum (u >> 640)
            [0.0, -500.0, 2.0],  # Point 4: far above -> out of frustum (v < 0)
        ], dtype=np.float64)

        u, v, depth, valid_mask, indices = project_lidar_to_image(
            points, R, t, K, image_wh, min_depth=0.5
        )

        assert len(u) == 1, f"Expected exactly 1 valid point, got {len(u)}"
        assert indices[0] == 0
        assert valid_mask[0] == True  # noqa: E712
        assert not np.any(valid_mask[1:])
        assert abs(u[0] - 320.0) < 1e-4
        assert abs(v[0] - 240.0) < 1e-4

    def test_intrinsics_rescaling(self):
        """Rescaling intrinsics correctly scales fx, fy, cx, cy."""
        intr = CameraIntrinsics.from_params(2000.0, 2000.0, 960.0, 600.0, image_wh=(1920, 1200))
        scaled = intr.rescale(0.5, 0.5)

        assert abs(scaled.fx - 1000.0) < 1e-6
        assert abs(scaled.fy - 1000.0) < 1e-6
        assert abs(scaled.cx - 480.0) < 1e-6
        assert abs(scaled.cy - 300.0) < 1e-6
        assert scaled.image_wh == (960, 600)

    def test_extrinsics_quaternion_and_transform(self):
        """Test extrinsics from wxyz quaternion."""
        # 90 deg rotation around Y axis: q = [cos(45°), 0, sin(45°), 0] in wxyz
        angle = np.pi / 2
        qw = np.cos(angle / 2)
        qy = np.sin(angle / 2)
        ext = Extrinsics.from_quat_translation([qw, 0.0, qy, 0.0], [1.0, 2.0, 3.0], quat_order="wxyz")

        # Rotating [1, 0, 0] by 90 deg around Y gives [0, 0, -1]
        # Then + t [1, 2, 3] gives [1, 2, 2]
        pts = np.array([[1.0, 0.0, 0.0]])
        transformed = ext.transform_points(pts)
        expected = np.array([[1.0, 2.0, 2.0]])
        np.testing.assert_allclose(transformed, expected, atol=1e-6)

    def test_semantic_point_sampling(self):
        """Test sampling pixel semantics at projected point coordinates."""
        label_map = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=np.int64)

        u = np.array([0.1, 1.9, 0.8])
        v = np.array([0.2, 0.1, 1.7])

        labels, confs = sample_point_semantics(u, v, label_map)
        # (0, 0) -> 1; (2, 0) -> 3; (1, 2) -> 8
        expected_labels = np.array([1, 3, 8])
        np.testing.assert_array_equal(labels, expected_labels)
        assert len(confs) == 3

    def test_projection_cpu_latency_65k_points(self):
        """Phase 3 exit criterion: Projection of a 65k-point cloud takes <10 ms on CPU."""
        np.random.seed(42)
        n_points = 65536
        points = np.random.uniform(-30.0, 30.0, size=(n_points, 4)).astype(np.float32)
        points[:, 2] = np.random.uniform(0.5, 40.0, size=n_points)  # mostly in front

        K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 600.0], [0, 0, 1.0]], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        t = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        image_wh = (1920, 1200)

        median_ms, p95_ms = benchmark(
            lambda: project_lidar_to_image(points, R, t, K, image_wh),
            warmup=20,
            iters=100,
        )

        print(f"\n[BENCHMARK] 65k points CPU projection latency: median={median_ms:.2f} ms, p95={p95_ms:.2f} ms")
        assert median_ms < 10.0, f"Projection too slow: {median_ms:.2f} ms >= 10.0 ms"
