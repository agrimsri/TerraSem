#!/usr/bin/env python3
"""Build train/val/test split files by sequence to prevent data leakage.

Per BUILD.md Phase 2 Task 4 & §8 Known Trap #3:
- Split strictly by sequence so adjacent frames never leak into val/test.
- Default sequence split across available sequences (00000..00004):
    Train: 00000, 00001, 00003
    Val:   00004
    Test:  00002
- Emits data/splits/{train,val,test}.txt containing relative frame paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_sequence_splits(
    data_root: str | Path,
    splits_dir: str | Path,
    train_seqs: list[str] | None = None,
    val_seqs: list[str] | None = None,
    test_seqs: list[str] | None = None,
) -> dict[str, list[str]]:
    data_root = Path(data_root)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    if train_seqs is None:
        train_seqs = ["00000", "00001", "00003"]
    if val_seqs is None:
        val_seqs = ["00004"]
    if test_seqs is None:
        test_seqs = ["00002"]

    # Assert zero sequence overlap
    set_train = set(train_seqs)
    set_val = set(val_seqs)
    set_test = set(test_seqs)

    assert not (set_train & set_val), f"Overlap between train and val: {set_train & set_val}"
    assert not (set_train & set_test), f"Overlap between train and test: {set_train & set_test}"
    assert not (set_val & set_test), f"Overlap between val and test: {set_val & set_test}"

    # Search for frames under Rellis-3D or data_root
    rellis_dir = data_root / "Rellis-3D" if (data_root / "Rellis-3D").exists() else data_root

    splits = {"train": train_seqs, "val": val_seqs, "test": test_seqs}
    frame_splits: dict[str, list[str]] = {}

    for split_name, seq_list in splits.items():
        frames = []
        for seq in seq_list:
            seq_dir = rellis_dir / seq
            label_dir = seq_dir / "pylon_camera_node_label_id"
            if label_dir.exists():
                for label_file in sorted(label_dir.glob("*.png")):
                    frame_id = label_file.stem
                    frames.append(f"{seq}/{frame_id}")
            else:
                # Also check point cloud labels
                pt_label_dir = seq_dir / "os1_cloud_node_semantickitti_label_id"
                if pt_label_dir.exists():
                    for pt_file in sorted(pt_label_dir.glob("*.label")):
                        frame_id = pt_file.stem
                        frames.append(f"{seq}/{frame_id}")

        out_file = splits_dir / f"{split_name}.txt"
        with out_file.open("w") as f:
            for frame in frames:
                f.write(f"{frame}\n")

        frame_splits[split_name] = frames
        print(f"[{split_name.upper()}] {len(frames)} frames from sequences {seq_list} -> {out_file}")

    return frame_splits


def main():
    parser = argparse.ArgumentParser(description="Build sequence-based train/val/test splits")
    parser.add_argument("--data-root", default="data/rellis3d")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--train-seqs", nargs="+", default=["00000", "00001", "00003"])
    parser.add_argument("--val-seqs", nargs="+", default=["00004"])
    parser.add_argument("--test-seqs", nargs="+", default=["00002"])
    args = parser.parse_args()

    build_sequence_splits(
        args.data_root,
        args.splits_dir,
        args.train_seqs,
        args.val_seqs,
        args.test_seqs,
    )


if __name__ == "__main__":
    main()
