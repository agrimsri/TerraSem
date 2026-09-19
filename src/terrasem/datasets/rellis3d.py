"""RELLIS-3D dataset loader.

Returns per-frame dicts with keys:
    image  : torch.Tensor [3, H, W] float32, normalised ImageNet stats
    label  : torch.Tensor [H, W] int64, TerraSem-11 class IDs
    points : torch.Tensor [N, 4] float32, KITTI-format XYZI
    pose   : torch.Tensor [4, 4] float32, T_world_lidar (TODO: verify convention)
    frame_id: str

See BUILD.md Phase 2 for the full specification.
TODO: implement after docs/dataset_inventory.md is generated (Phase 2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Rellis3DDataset:
    """PyTorch-compatible dataset for RELLIS-3D.

    Args:
        root: Path to the ``data/rellis3d/`` directory.
        split_file: Path to one of ``data/splits/{train,val,test}.txt``.
        transform: Optional callable applied to the sample dict.

    TODO (Phase 2):
        - Implement ``__getitem__`` after running ``scripts/01_inspect_dataset.py``
          and verifying file paths and array shapes in ``docs/dataset_inventory.md``.
        - Mask ``.label`` files with ``& 0xFFFF`` (SemanticKITTI convention).
        - Map RELLIS-3D class IDs → TerraSem-11 via ``ontology.rellis3d_to_terrasem``.
    """

    def __init__(
        self,
        root: str | Path,
        split_file: str | Path,
        transform: Any = None,
    ) -> None:
        self.root = Path(root)
        self.split_file = Path(split_file)
        self.transform = transform
        self._frame_ids: list[str] = self._load_split()

    def _load_split(self) -> list[str]:
        with self.split_file.open() as fh:
            return [line.strip() for line in fh if line.strip()]

    def __len__(self) -> int:
        return len(self._frame_ids)

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError(
            "Rellis3DDataset.__getitem__ not yet implemented. "
            "Run scripts/01_inspect_dataset.py first and consult "
            "docs/dataset_inventory.md before writing this loader."
        )
