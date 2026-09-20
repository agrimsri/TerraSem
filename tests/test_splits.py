"""Unit tests for dataset split integrity.

Asserts no sequence appears in more than one split (zero sequence overlap).
See BUILD.md §6 testing requirements and Phase 2 exit criteria.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def get_sequences_from_split(split_path: Path) -> set[str]:
    seqs = set()
    with split_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                seqs.add(line.split("/")[0])
    return seqs


def test_no_sequence_overlap():
    splits_dir = Path("data/splits")
    if not splits_dir.exists():
        pytest.skip("data/splits does not exist yet")

    train_file = splits_dir / "train.txt"
    val_file = splits_dir / "val.txt"
    test_file = splits_dir / "test.txt"

    assert train_file.exists(), "train.txt missing"
    assert val_file.exists(), "val.txt missing"
    assert test_file.exists(), "test.txt missing"

    train_seqs = get_sequences_from_split(train_file)
    val_seqs = get_sequences_from_split(val_file)
    test_seqs = get_sequences_from_split(test_file)

    assert len(train_seqs) > 0, "train.txt is empty"
    assert len(val_seqs) > 0, "val.txt is empty"
    assert len(test_seqs) > 0, "test.txt is empty"

    overlap_train_val = train_seqs & val_seqs
    overlap_train_test = train_seqs & test_seqs
    overlap_val_test = val_seqs & test_seqs

    assert not overlap_train_val, f"Sequence overlap between train and val: {overlap_train_val}"
    assert not overlap_train_test, f"Sequence overlap between train and test: {overlap_train_test}"
    assert not overlap_val_test, f"Sequence overlap between val and test: {overlap_val_test}"
