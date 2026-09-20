"""Visualisation utilities for TerraSem.

Provides colour mapping, alpha blending, depth colourmaps, and overlay plots
for 2D semantics, 3D LiDAR projections, and traversability cost maps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from terrasem.datasets.ontology import CLASSES, NUM_CLASSES


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))


# Color palette array of shape (NUM_CLASSES, 3) in uint8
CLASS_PALETTE = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
for cid, _, hex_col in CLASSES:
    if 0 <= cid < NUM_CLASSES:
        CLASS_PALETTE[cid] = hex_to_rgb(hex_col)


def colorize_label(label: np.ndarray) -> np.ndarray:
    """Colorize a 2D integer label array [H, W] into an RGB image [H, W, 3] uint8."""
    label_clipped = np.clip(label, 0, NUM_CLASSES - 1).astype(np.int64)
    return CLASS_PALETTE[label_clipped]


def overlay_mask(image: np.ndarray, label: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend an RGB image [H, W, 3] with a colored semantic label [H, W].

    Void pixels (label == 0) remain unblended (100% original image).
    """
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    colored = colorize_label(label)
    blended = image.copy().astype(np.float32)

    non_void = label > 0
    blended[non_void] = (1 - alpha) * image[non_void] + alpha * colored[non_void]
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_sample_grid(
    image: np.ndarray,
    label: np.ndarray,
    out_path: str | Path,
    title: str = "",
) -> None:
    """Save side-by-side [Image | Ground Truth | Overlay] figure."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    colored = colorize_label(label)
    overlaid = overlay_mask(image, label, alpha=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(image)
    axes[0].set_title("Camera RGB")
    axes[0].axis("off")

    axes[1].imshow(colored)
    axes[1].set_title("TerraSem-11 Semantic Map")
    axes[1].axis("off")

    axes[2].imshow(overlaid)
    axes[2].set_title("Semantic Overlay")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
