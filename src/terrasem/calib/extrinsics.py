"""LiDAR→camera extrinsic transform parsing and point transformation.

See BUILD.md Phase 3.
Transforms 3D points from LiDAR frame to camera frame:
    P_cam = R @ P_lidar + t

Handles YAML quaternion convention:
RELLIS-3D stores q as {w, x, y, z} in transforms.yaml, while
scipy.spatial.transform.Rotation.from_quat expects [x, y, z, w].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


class Extrinsics:
    """Rigid 3D transformation T_cam_lidar [R | t].

    Attributes:
        R: (3, 3) rotation matrix float64.
        t: (3,) translation vector float64.
        T: (4, 4) homogeneous transformation matrix.
    """

    def __init__(self, R: np.ndarray, t: np.ndarray) -> None:
        self.R = np.asarray(R, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64).reshape(3)
        assert self.R.shape == (3, 3), f"R must be (3, 3), got {self.R.shape}"
        assert self.t.shape == (3,), f"t must be (3,), got {self.t.shape}"

        self._T = np.eye(4, dtype=np.float64)
        self._T[:3, :3] = self.R
        self._T[:3, 3] = self.t

    @property
    def T(self) -> np.ndarray:
        """4x4 homogeneous transformation matrix."""
        return self._T

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> Extrinsics:
        """Create from 4x4 or 3x4 transformation matrix."""
        T = np.asarray(T, dtype=np.float64)
        return cls(R=T[:3, :3], t=T[:3, 3])

    @classmethod
    def from_quat_translation(
        cls,
        quat: list[float] | np.ndarray,
        translation: list[float] | np.ndarray,
        quat_order: str = "wxyz",
    ) -> Extrinsics:
        """Create Extrinsics from quaternion and translation.

        Args:
            quat: 4 elements.
            translation: 3 elements [x, y, z].
            quat_order: 'wxyz' (RELLIS-3D YAML) or 'xyzw' (scipy/ROS).
        """
        quat = np.asarray(quat, dtype=np.float64)
        if quat_order == "wxyz":
            # Convert [w, x, y, z] -> [x, y, z, w] for scipy
            xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
        elif quat_order == "xyzw":
            xyzw = quat
        else:
            raise ValueError(f"Unknown quat_order: {quat_order}")

        rot = Rotation.from_quat(xyzw)
        R = rot.as_matrix()
        return cls(R=R, t=np.asarray(translation, dtype=np.float64))

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        key: str = "os1_cloud_node-pylon_camera_node",
    ) -> Extrinsics:
        """Parse transforms.yaml from RELLIS-3D.

        The YAML contains:
            key:
              q: {w: ..., x: ..., y: ..., z: ...}
              t: {x: ..., y: ..., z: ...}
        """
        path = Path(path)
        with path.open() as fh:
            data = yaml.safe_load(fh)

        if key not in data:
            # Try to find first key with matching pattern
            matches = [k for k in data if "camera" in k and ("os1" in k or "lidar" in k)]
            if matches:
                key = matches[0]
            else:
                raise KeyError(f"Transform key '{key}' not found in {path}. Keys: {list(data.keys())}")

        tf_data = data[key]
        q_dict = tf_data["q"]
        t_dict = tf_data["t"]

        # RELLIS-3D stores w, x, y, z
        quat_wxyz = [float(q_dict["w"]), float(q_dict["x"]), float(q_dict["y"]), float(q_dict["z"])]
        t_xyz = [float(t_dict["x"]), float(t_dict["y"]), float(t_dict["z"])]

        return cls.from_quat_translation(quat_wxyz, t_xyz, quat_order="wxyz")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform Nx3 (or Nx4) 3D points from LiDAR frame to camera frame.

        Args:
            points: (N, 3) or (N, 4) array of points.

        Returns:
            (N, 3) transformed points in camera coordinate frame.
        """
        pts_3d = points[:, :3]
        # P_cam = P_lidar @ R.T + t
        return (pts_3d @ self.R.T) + self.t

    def inverse(self) -> Extrinsics:
        """Return inverse transform T_lidar_cam."""
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return Extrinsics(R=R_inv, t=t_inv)
