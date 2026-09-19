"""RUGD dataset loader stub.

TODO (Phase 7): implement after RUGD is downloaded and inspected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class RUGDDataset:
    """PyTorch-compatible dataset for RUGD (cross-domain evaluation).

    TODO (Phase 7): Implement after running dataset inspection on RUGD.
    See docs/dataset_inventory.md for the auto-generated layout.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        transform: Any = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        raise NotImplementedError("RUGDDataset not yet implemented.")

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError("RUGDDataset not yet implemented.")
