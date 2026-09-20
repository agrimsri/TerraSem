"""Fog corruption for camera images.

Per BUILD.md Phase 7.3:
Atmospheric scattering model: I' = I * t + A * (1 - t)
where t = exp(-beta * d), A is airlight (typically [0.8, 0.8, 0.8]).
If no depth map is provided, a vertical depth proxy is generated.
Severity levels: 0 (no-op), 1 to 5.
Deterministic under a fixed seed.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply(
    image: np.ndarray,
    severity: int,
    seed: int = 42,
    depth: np.ndarray | None = None,
) -> np.ndarray:
    """Apply fog corruption to an RGB image.

    Args:
        image: (H, W, 3) or (3, H, W) numpy array, float in [0, 1] or uint8 in [0, 255].
        severity: Int from 0 to 5 (0 = no-op).
        seed: Random seed for reproducibility.
        depth: Optional (H, W) depth map in metres.

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

    # Atmospheric extinction coefficient beta per severity
    betas = [0.0, 0.04, 0.08, 0.14, 0.22, 0.32]
    beta = betas[severity]

    # Generate depth proxy if not provided
    if depth is None:
        # Distance increases from bottom (ground near) to top (horizon/distance)
        y_coords = np.linspace(1.0, 0.05, h, dtype=np.float32).reshape(h, 1)
        depth_proxy = 30.0 / (y_coords + 0.1)  # range ~ 25m to 270m
        depth_proxy = np.tile(depth_proxy, (1, w))
    else:
        depth_proxy = np.asarray(depth, dtype=np.float32)
        if depth_proxy.shape != (h, w):
            depth_proxy = cv2.resize(depth_proxy, (w, h))

    # Transmission map t = exp(-beta * d)
    transmission = np.exp(-beta * (depth_proxy / 10.0))
    # Add slight random spatial variation to fog thickness
    noise = rng.normal(0.0, 0.03 * severity, size=(h, w)).astype(np.float32)
    transmission = np.clip(transmission + noise, 0.05, 1.0)
    transmission = np.expand_dims(transmission, axis=-1)

    # Atmospheric light A
    airlight = np.array([0.85, 0.88, 0.90], dtype=np.float32)

    foggy = img_float * transmission + airlight * (1.0 - transmission)
    foggy = np.clip(foggy, 0.0, 1.0)

    out = (foggy * 255.0).round().astype(np.uint8) if is_uint8 else foggy.astype(image.dtype)

    if is_channels_first:
        out = np.transpose(out, (2, 0, 1))

    return out
