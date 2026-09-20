"""Rain / motion blur corruption for camera images.

Per BUILD.md Phase 7.3:
Directional rain streaks + Gaussian blur.
Severity levels: 0 (no-op), 1 to 5.
Deterministic under a fixed seed.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def apply(image: np.ndarray, severity: int, seed: int = 42) -> np.ndarray:
    """Apply rain and blur corruption to an image.

    Args:
        image: (H, W, 3) or (3, H, W) numpy array, float in [0, 1] or uint8 in [0, 255].
        severity: Int from 0 to 5 (0 = no-op).
        seed: Random seed for reproducibility.

    Returns:
        Corrupted image with same shape and dtype as input.
    """
    if severity <= 0:
        return image.copy()

    severity = min(severity, 5)
    rng = np.random.RandomState(seed)

    is_channels_first = image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[2] > 3
    img_hwc = np.transpose(image, (1, 2, 0)) if is_channels_first else image.copy()

    is_uint8 = img_hwc.dtype == np.uint8
    img_float = img_hwc.astype(np.float32) / 255.0 if is_uint8 else img_hwc.astype(np.float32)

    h, w = img_float.shape[:2]

    # 1. Subtle motion / rain blur
    blur_ksize = [1, 3, 3, 5, 5, 7][severity]
    if blur_ksize > 1:
        # Directional motion blur kernel simulating falling raindrops
        kernel = np.zeros((blur_ksize, blur_ksize), dtype=np.float32)
        np.fill_diagonal(kernel, 1.0 / blur_ksize)
        img_blurred = cv2.filter2D(img_float, -1, kernel)
    else:
        img_blurred = img_float

    # 2. Additive rain streaks
    streak_counts = [0, 300, 700, 1500, 3000, 5000][severity]
    streak_lengths = [0, 8, 14, 20, 28, 38][severity]

    rain_layer = np.zeros((h, w), dtype=np.float32)
    x_coords = rng.randint(0, w, size=streak_counts)
    y_coords = rng.randint(0, h, size=streak_counts)
    angles = rng.uniform(65.0, 75.0, size=streak_counts)  # slight angle from vertical

    for x, y, ang in zip(x_coords, y_coords, angles, strict=False):
        rad = math.radians(ang)
        x_end = int(x + streak_lengths * math.cos(rad))
        y_end = int(y + streak_lengths * math.sin(rad))
        cv2.line(rain_layer, (x, y), (x_end, y_end), color=1.0, thickness=1)

    # Blur the rain streaks
    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0.5)
    rain_layer = np.expand_dims(rain_layer, axis=-1)

    rain_intensity = [0.0, 0.08, 0.15, 0.22, 0.30, 0.40][severity]
    corrupted = img_blurred + rain_intensity * rain_layer
    corrupted = np.clip(corrupted, 0.0, 1.0)

    if is_uint8:
        out = (corrupted * 255.0).round().astype(np.uint8)
    else:
        out = corrupted.astype(image.dtype)

    if is_channels_first:
        out = np.transpose(out, (2, 0, 1))

    return out
