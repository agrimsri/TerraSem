"""Low light and sensor noise corruption.

Per BUILD.md Phase 7.3:
Gamma attenuation + Poisson-Gaussian sensor noise.
Severity levels: 0 (no-op), 1 to 5.
Deterministic under a fixed seed.
"""

from __future__ import annotations

import numpy as np


def apply(image: np.ndarray, severity: int, seed: int = 42) -> np.ndarray:
    """Apply low-light and sensor noise corruption to an image.

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

    # 1. Low light attenuation via gamma
    gammas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.8]
    dim_scales = [1.0, 0.85, 0.70, 0.55, 0.40, 0.25]
    gamma = gammas[severity]
    scale = dim_scales[severity]

    dimmed = (img_float ** gamma) * scale

    # 2. Shot (Poisson) noise + Read (Gaussian) noise
    gaussian_sigmas = [0.0, 0.02, 0.04, 0.06, 0.08, 0.12]
    sigma = gaussian_sigmas[severity]

    gaussian_noise = rng.normal(0.0, sigma, size=dimmed.shape).astype(np.float32)

    # Scaled Poisson noise
    peak = 30.0 / severity
    noisy_poisson = rng.poisson(np.clip(dimmed, 0.0, 1.0) * peak).astype(np.float32) / peak
    poisson_contribution = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8][severity]

    blended = (1.0 - poisson_contribution) * dimmed + poisson_contribution * noisy_poisson + gaussian_noise
    corrupted = np.clip(blended, 0.0, 1.0)

    if is_uint8:
        out = (corrupted * 255.0).round().astype(np.uint8)
    else:
        out = corrupted.astype(image.dtype)

    if is_channels_first:
        out = np.transpose(out, (2, 0, 1))

    return out
