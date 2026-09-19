# TerraSem Build Log

Project: TerraSem — Uncertainty-aware semantic occupancy and traversability mapping  
Repo: https://github.com/agrimsri/TerraSem

Each phase entry records: what was done, what was measured, what broke, what assumptions were made.

---

## Phase 0 — Scaffold

**Date:** 2026-09-19  
**Status:** ✅ Complete

### What was done
- Initialised git repo and created public GitHub repo at https://github.com/agrimsri/TerraSem
- Created complete directory tree from BUILD.md §2
- Implemented:
  - `pyproject.toml` with ruff config (line-length 100) and `[project] name="terrasem"`
  - `requirements.txt` + `environment.yml`
  - `.gitignore` (ignores data/, checkpoints/, *.onnx, *.pt, *.ckpt, __pycache__/, .venv/, wandb/, outputs/)
  - `src/terrasem/utils/seed.py` → `set_seed(seed: int)` seeding python/numpy/torch/cuda + cudnn deterministic
  - `src/terrasem/utils/config.py` → YAML loader with `--override key=value` CLI merging
  - `src/terrasem/utils/timing.py` → `Timer` context manager + `benchmark(fn, warmup, iters)` returning median/p95
  - All subpackage stubs with docstrings (datasets, models, calib, mapping, corruptions, metrics, export)
  - `src/terrasem/datasets/ontology.py` → TerraSem-11 definition + RELLIS-3D/RUGD/GOOSE mapping dicts
  - `src/terrasem/metrics/segmentation.py` → confusion-matrix mIoU, per-class IoU, pixel acc, FW-IoU
  - All configs/ YAML files
  - All scripts/ stubs
  - All tests/ stubs + real tests for metrics and ontology
  - `.github/workflows/python-ci.yml` + `.github/workflows/ros2-docker.yml`

### What was measured
- `pip install -e .` → TODO: not yet measured (no conda/pip env set up on dev machine yet)
- `ruff check .` → TODO: not yet measured
- `pytest -q` → TODO: not yet measured (record after environment setup)

### What broke
- Nothing at scaffold stage.

### Assumptions made
- RELLIS-3D class IDs in `ontology.py` are from the official paper/GitHub; they **must** be verified against `docs/dataset_inventory.md` in Phase 2 before the loader is written.
- GOOSE mapping is a placeholder — only void→0 is set. Full mapping requires Phase 2 inspection.

### Next steps
- Phase 1: Literature grounding → `docs/related_work.md`
- Phase 2: Dataset download + inspection → `docs/dataset_inventory.md`

---

<!-- Add a new ## Phase N — ... section after each phase is complete. -->
