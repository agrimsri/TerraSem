"""Data augmentation transforms for TerraSem datasets.

Augmentations applied during training (per BUILD.md Phase 2):
- Random scale 0.5–2.0
- Random crop (512×512)
- Horizontal flip
- Photometric jitter (brightness, contrast, saturation, hue)

TODO (Phase 2): implement with torchvision.transforms.v2 or albumentations.
"""

from __future__ import annotations


class TrainTransform:
    """Training-time augmentation pipeline.

    TODO (Phase 2): implement after dataset loader is working.
    """

    def __call__(self, sample: dict) -> dict:
        raise NotImplementedError("TrainTransform not yet implemented.")


class ValTransform:
    """Validation/test-time transform (resize + normalise only).

    TODO (Phase 2): implement after dataset loader is working.
    """

    def __call__(self, sample: dict) -> dict:
        raise NotImplementedError("ValTransform not yet implemented.")
