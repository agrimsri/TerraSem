"""Frozen DINOv2 ViT-S/14 + light conv head for label-efficiency experiments.

See BUILD.md Phase 4 for the full specification.
TODO (Phase 4): implement.
"""

from __future__ import annotations


class DINOv2Linear:
    """Frozen DINOv2 ViT-S/14 backbone + trainable conv segmentation head.

    TODO (Phase 4): implement using facebook/dinov2-small from HuggingFace.
    """

    def __init__(self, num_labels: int = 11) -> None:
        raise NotImplementedError("DINOv2Linear not yet implemented (Phase 4).")
