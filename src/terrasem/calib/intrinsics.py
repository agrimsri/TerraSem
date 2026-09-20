"""Camera intrinsics parsing, scaling, and undistortion.

See BUILD.md Phase 3.
CRITICAL: When resizing images by scale factor s, fx, fy, cx, cy must be scaled by s.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class CameraIntrinsics:
    """Camera intrinsics 3x3 matrix K and distortion coefficients.

    Attributes:
        K: (3, 3) intrinsic camera projection matrix float64/float32.
        dist: (5,) or (4,) distortion coefficients [k1, k2, p1, p2, k3].
        image_wh: tuple of (width, height) in pixels.
    """

    def __init__(
        self,
        K: np.ndarray,
        dist: np.ndarray | None = None,
        image_wh: tuple[int, int] = (1920, 1200),
    ) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        assert self.K.shape == (3, 3), f"K must be (3, 3), got {self.K.shape}"
        self.dist = np.asarray(dist, dtype=np.float64) if dist is not None else np.zeros(5, dtype=np.float64)
        self.image_wh = image_wh

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @classmethod
    def from_params(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist: np.ndarray | None = None,
        image_wh: tuple[int, int] = (1920, 1200),
    ) -> CameraIntrinsics:
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return cls(K, dist=dist, image_wh=image_wh)

    @classmethod
    def from_file(cls, path: str | Path, image_wh: tuple[int, int] = (1920, 1200)) -> CameraIntrinsics:
        """Parse 4 floats (fx fy cx cy) from RELLIS-3D camera_info.txt."""
        path = Path(path)
        content = path.read_text().strip()
        vals = [float(x) for x in content.split()]
        if len(vals) == 4:
            fx, fy, cx, cy = vals
            return cls.from_params(fx, fy, cx, cy, image_wh=image_wh)
        elif len(vals) >= 9:
            # 3x3 matrix format
            K = np.array(vals[:9], dtype=np.float64).reshape(3, 3)
            return cls(K, image_wh=image_wh)
        else:
            raise ValueError(f"Unrecognized camera info format in {path}: '{content}'")

    def rescale(self, scale_x: float, scale_y: float) -> CameraIntrinsics:
        """Return a new CameraIntrinsics scaled for image resolution change.

        Per BUILD.md §8: If image is resized by scale (sx, sy),
        fx, fy, cx, cy must be scaled accordingly.
        """
        K_new = self.K.copy()
        K_new[0, 0] *= scale_x  # fx
        K_new[0, 2] *= scale_x  # cx
        K_new[1, 1] *= scale_y  # fy
        K_new[1, 2] *= scale_y  # cy

        new_w = int(round(self.image_wh[0] * scale_x))
        new_h = int(round(self.image_wh[1] * scale_y))
        return CameraIntrinsics(K_new, dist=self.dist.copy(), image_wh=(new_w, new_h))

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistort image using OpenCV if distortion coefficients are non-zero."""
        if np.all(self.dist == 0):
            return image
        return cv2.undistort(image, self.K, self.dist)
