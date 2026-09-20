"""Data augmentation transforms for TerraSem datasets.

Augmentations applied during training (per BUILD.md Phase 2):
- Random scale 0.5–2.0
- Random crop (default 512×512)
- Horizontal flip
- Photometric jitter (brightness, contrast, saturation, hue)
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TrainTransform:
    """Training augmentation pipeline for synchronized image + semantic label."""

    def __init__(
        self,
        crop_size: tuple[int, int] = (512, 512),
        scale_range: tuple[float, float] = (0.5, 2.0),
        hflip_prob: float = 0.5,
        ignore_index: int = 0,
    ) -> None:
        self.crop_size = crop_size  # (H, W)
        self.scale_range = scale_range
        self.hflip_prob = hflip_prob
        self.ignore_index = ignore_index

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image: Image.Image = sample["image"]
        label: Image.Image = sample["label"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if not isinstance(label, Image.Image):
            label = Image.fromarray(label)

        # 1. Random scaling (0.5 to 2.0)
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_w = max(1, int(image.width * scale))
        target_h = max(1, int(image.height * scale))
        image = TF.resize(image, (target_h, target_w), interpolation=TF.InterpolationMode.BILINEAR)
        label = TF.resize(label, (target_h, target_w), interpolation=TF.InterpolationMode.NEAREST)

        # 2. Random crop (with padding if scaled image is smaller than crop_size)
        crop_h, crop_w = self.crop_size
        pad_h = max(0, crop_h - image.height)
        pad_w = max(0, crop_w - image.width)
        if pad_h > 0 or pad_w > 0:
            # pad left, top, right, bottom
            padding = [0, 0, pad_w, pad_h]
            image = TF.pad(image, padding, fill=0)
            label = TF.pad(label, padding, fill=self.ignore_index)

        max_i = image.height - crop_h
        max_j = image.width - crop_w
        i = random.randint(0, max_i) if max_i > 0 else 0
        j = random.randint(0, max_j) if max_j > 0 else 0
        image = TF.crop(image, i, j, crop_h, crop_w)
        label = TF.crop(label, i, j, crop_h, crop_w)

        # 3. Random horizontal flip
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # 4. Photometric jitter (image only)
        if random.random() < 0.8:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
            image = TF.adjust_saturation(image, saturation)

        # 5. Convert to tensor and normalize
        image_tensor = TF.to_tensor(image)  # [3, H, W] in [0.0, 1.0]
        image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        label_tensor = torch.from_numpy(np.array(label, dtype=np.int64))

        sample_out = dict(sample)
        sample_out["image"] = image_tensor
        sample_out["label"] = label_tensor
        return sample_out


class ValTransform:
    """Validation/test-time transform: resize/crop to fixed size and normalize."""

    def __init__(
        self,
        target_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.target_size = target_size

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image: Image.Image = sample["image"]
        label: Image.Image = sample["label"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if not isinstance(label, Image.Image):
            label = Image.fromarray(label)

        # Resize to target size
        image = TF.resize(image, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        label = TF.resize(label, self.target_size, interpolation=TF.InterpolationMode.NEAREST)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        label_tensor = torch.from_numpy(np.array(label, dtype=np.int64))

        sample_out = dict(sample)
        sample_out["image"] = image_tensor
        sample_out["label"] = label_tensor
        return sample_out
