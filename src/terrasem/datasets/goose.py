"""GOOSE dataset loader for cross-domain evaluation.

Per BUILD.md Phase 7.1:
Loads GOOSE images and semantic annotations mapped into TerraSem-11.
Provides synthetic fallback if external dataset directory is empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from terrasem.datasets.ontology import GOOSE_TO_TERRASEM

_GOOSE_REMAP_LUT = np.full(256, 10, dtype=np.int64)
for _src_id, _tgt_id in GOOSE_TO_TERRASEM.items():
    if 0 <= _src_id < 256:
        _GOOSE_REMAP_LUT[_src_id] = _tgt_id


class GOOSEDataset(Dataset):
    """PyTorch Dataset for GOOSE off-road semantic segmentation.

    Args:
        root: Path to GOOSE dataset root.
        split: 'train', 'val', or 'test'.
        transform: Optional torchvision or custom transform.
    """

    def __init__(
        self,
        root: str | Path = "data/goose",
        split: str = "test",
        transform: Any = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.image_paths: list[Path] = []
        self.label_paths: list[Path] = []

        if self.root.exists():
            img_dir = self.root / "images" / split
            lbl_dir = self.root / "labels" / split
            if img_dir.exists() and lbl_dir.exists():
                self.image_paths = sorted(img_dir.glob("**/*.png"))
                self.label_paths = sorted(lbl_dir.glob("**/*.png"))

        self.num_synthetic = 50 if len(self.image_paths) == 0 else 0

    def __len__(self) -> int:
        if self.image_paths:
            return len(self.image_paths)
        return self.num_synthetic

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.image_paths:
            img_p = self.image_paths[idx]
            lbl_p = self.label_paths[idx]
            img = Image.open(img_p).convert("RGB")
            raw_lbl = np.array(Image.open(lbl_p), dtype=np.uint8)
            if raw_lbl.ndim == 3:
                raw_lbl = raw_lbl[:, :, 0]
            label_mapped = _GOOSE_REMAP_LUT[raw_lbl]
            label = Image.fromarray(label_mapped.astype(np.uint8))
            frame_id = str(img_p.stem)
        else:
            # Generate deterministic synthetic cross-domain GOOSE test sample (German terrain)
            rng = np.random.RandomState(idx + 2000)
            H, W = 512, 512
            # Characteristic European forest / gravel road palette
            img_arr = rng.randint(30, 180, size=(H, W, 3), dtype=np.uint8)
            # GOOSE classes: 4 (grass), 2 (gravel), 6 (tree trunk), 13 (sky)
            raw_lbl = np.full((H, W), 4, dtype=np.uint8)
            raw_lbl[: int(H * 0.3), :] = 13   # sky
            raw_lbl[int(H * 0.65) :, :] = 2   # gravel path
            label_mapped = _GOOSE_REMAP_LUT[raw_lbl]
            img = Image.fromarray(img_arr)
            label = Image.fromarray(label_mapped.astype(np.uint8))
            frame_id = f"goose_synth_{idx:04d}"

        sample = {
            "image": img,
            "label": label,
            "frame_id": frame_id,
        }

        if self.transform is not None:
            sample = self.transform(sample)
        else:
            sample["image"] = torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            sample["label"] = torch.from_numpy(np.array(label, dtype=np.int64))

        return sample
