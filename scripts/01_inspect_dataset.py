#!/usr/bin/env python3
"""Inspect RELLIS-3D data layout and write docs/dataset_inventory.md.

See BUILD.md Phase 2 for the full specification.
TODO (Phase 2): implement after data is downloaded.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Inspect RELLIS-3D dataset")
    parser.add_argument("--data-root", default="data/rellis3d")
    parser.add_argument("--out", default="docs/dataset_inventory.md")
    parser.parse_args()
    raise NotImplementedError(
        "01_inspect_dataset.py not yet implemented. "
        "Run after downloading RELLIS-3D (scripts/00_download_rellis.sh)."
    )

if __name__ == "__main__":
    main()
