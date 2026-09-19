# BUILD.md — TerraSem

**Uncertainty-aware semantic occupancy and traversability mapping for off-road autonomous navigation (Camera + LiDAR, ROS 2).**

This document is the single source of truth for any AI agent working on this repository. Read it fully before writing code. Follow the phases in order. Do not skip ahead.

---

## 0. Rules for agents (read first, non-negotiable)

These rules exist because the most common failure mode for agents on this project is confidently inventing dataset paths, calibration formats, API signatures, and benchmark numbers.

### 0.1 Never fabricate

1. **Never invent a number.** Every metric (mIoU, latency, memory, FPS) that appears in any README, report, plot, or commit message MUST come from a script in this repo that was actually executed, and MUST be traceable to a JSON file under `results/`. If you have not run it, write `TODO: not yet measured`. Never write a plausible placeholder that looks like a real measurement.
2. **Never invent a file path or dataset field layout.** Before writing any dataset loader, you must first run an inspection script that prints the real directory tree and the real shapes/dtypes of the arrays, and save that output to `docs/dataset_inventory.md`. Write the loader against that file, not against memory.
3. **Never invent an API.** If you are unsure whether a function exists (e.g. an Open3D, ROS 2, or `transformers` API), verify it in the installed package before use:
   ```bash
   python -c "import open3d; print(open3d.__version__); print([a for a in dir(open3d.geometry) if 'Voxel' in a])"
   ```
   If you cannot verify it, use a documented alternative or implement it yourself with NumPy.
4. **Never invent calibration conventions.** LiDAR→camera extrinsics are the #1 source of silent bugs. You must validate the projection visually (Phase 3) before any downstream work is allowed to start.
5. **Never invent a citation.** Only cite papers whose arXiv ID is listed in §12 of this document, or that you have actually fetched.

### 0.2 Working discipline

- Work in **one phase at a time**. Each phase has an **Exit criteria** block. Do not open the next phase until every box is satisfiable and you have produced the named artifacts.
- After each phase, write a short entry in `docs/BUILD_LOG.md`: what was done, what was measured, what broke, what assumptions were made. Timestamp it.
- Commit at the end of every phase with the message prefix `phase-N: <summary>`.
- **Determinism:** every training/eval script must accept `--seed` and set seeds for `random`, `numpy`, and `torch` (and `torch.cuda`). Default seed `42`. Log the seed into the results JSON.
- **Config over constants.** No magic numbers inside functions. All tunables live in YAML under `configs/`. Scripts read configs; configs are committed.
- **Every script must run end-to-end on a tiny subset** via a `--smoke` flag (e.g. 8 frames, 1 epoch, 20 steps) before any long run is launched. If `--smoke` fails, do not launch the long run.

### 0.3 When to stop and ask the human

Stop and ask (do not guess) if:
- The dataset download fails or requires credentials you do not have.
- Calibration validation in Phase 3 fails after two genuine debugging attempts.
- A phase would require a GPU/compute budget beyond free tiers (Colab/Kaggle) — report the estimate instead.
- A dataset licence appears to forbid redistributing sample frames in the public demo.
- Measured results contradict the project's core claim (e.g. fusion is *worse* than camera-only everywhere). Report it honestly; do not tune until the story looks nice. A negative result, clearly explained, is a strong thesis-internship signal.

### 0.4 Scientific integrity

- The **test split is touched exactly once per experiment**, at the end. All tuning happens on validation.
- Report **mean ± std over 3 seeds** for the headline segmentation numbers. If compute does not allow 3 seeds, state `single seed` explicitly in the results table.
- Latency = **median over ≥200 warm iterations after ≥20 warm-up iterations**, and you must record CPU model / GPU model / batch size / input resolution alongside it.
- Never compare numbers measured on different hardware or different input resolutions without labelling them as such.

---

## 1. What we are building (target system)

A perception stack that consumes synchronised RGB + LiDAR + pose from an off-road dataset and emits:

1. **2D semantic segmentation** of the camera image (off-road classes: grass, dirt, mud, water, bush, tree, rubble, sky, vehicle, person, …).
2. **Semantically-labelled point clouds** by projecting LiDAR into the segmented image.
3. A **3D semantic occupancy voxel map**, accumulated over time using poses, where each voxel stores:
   - occupancy log-odds,
   - a Dirichlet-style class count vector,
   - derived class entropy (epistemic-ish uncertainty proxy).
4. A **2.5D traversability cost map** (`nav_msgs/OccupancyGrid`) fusing geometry (slope, step height, roughness) with semantics and uncertainty.
5. A **ROS 2 Humble package** with a Python inference node and a C++ voxel-fusion node, runnable on dataset rosbags.
6. A **free public demo** (Hugging Face Spaces, CPU) + a **Dockerised, CI-tested** ROS 2 package.

The research contribution of the repo is not the model — it is the **evaluation**: cross-dataset generalisation, label efficiency, sensor-degradation robustness, and the resolution/memory/latency trade-off of the voxel map.

---

## 2. Repository layout (create exactly this)

```
terrasem/
├── BUILD.md                     # this file
├── README.md                    # written LAST (Phase 9)
├── LICENSE                      # MIT
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── .gitignore                   # ignores data/, checkpoints/, *.onnx, wandb/
├── .github/workflows/
│   ├── python-ci.yml            # lint + unit tests
│   └── ros2-docker.yml          # build ROS 2 image, push to GHCR
├── configs/
│   ├── dataset/rellis3d.yaml
│   ├── dataset/goose.yaml
│   ├── dataset/rugd.yaml
│   ├── model/segformer_b0.yaml
│   ├── model/dinov2_linear.yaml
│   ├── mapping/voxel_default.yaml
│   ├── traversability/costs.yaml
│   └── experiments/*.yaml
├── docs/
│   ├── BUILD_LOG.md
│   ├── dataset_inventory.md     # AUTO-GENERATED in Phase 2, never hand-written
│   ├── calibration_notes.md
│   └── report.md                # 2-page technical write-up (Phase 9)
├── data/                        # gitignored. Symlink to real storage.
├── checkpoints/                 # gitignored
├── results/                     # COMMITTED. JSON metrics + generated plots.
├── scripts/                     # thin CLI entry points only
│   ├── 00_download_rellis.sh
│   ├── 01_inspect_dataset.py
│   ├── 02_build_splits.py
│   ├── 10_train_seg.py
│   ├── 11_eval_seg.py
│   ├── 20_project_lidar.py
│   ├── 30_build_map.py
│   ├── 40_eval_generalization.py
│   ├── 41_eval_label_efficiency.py
│   ├── 42_eval_corruptions.py
│   ├── 43_eval_voxel_sweep.py
│   ├── 50_export_onnx.py
│   ├── 51_quantize_int8.py
│   ├── 52_benchmark_latency.py
│   └── 90_make_plots.py
├── src/terrasem/
│   ├── __init__.py
│   ├── datasets/  {rellis3d.py, goose.py, rugd.py, ontology.py, transforms.py}
│   ├── models/    {segformer.py, dinov2_linear.py, registry.py}
│   ├── calib/     {intrinsics.py, extrinsics.py, projection.py}
│   ├── mapping/   {voxel_grid.py, bayesian_update.py, costmap.py}
│   ├── corruptions/ {fog.py, rain.py, noise.py, lidar_dropout.py}
│   ├── metrics/   {segmentation.py, mapping.py, calibration_metrics.py}
│   ├── export/    {onnx_export.py, quantize.py, runtime.py}
│   └── utils/     {seed.py, config.py, logging.py, viz.py, timing.py}
├── ros2_ws/src/terrasem_ros/
│   ├── package.xml
│   ├── CMakeLists.txt
│   ├── terrasem_ros/            # python nodes
│   │   ├── seg_node.py
│   │   └── costmap_node.py
│   ├── src/                     # C++ nodes
│   │   └── voxel_fusion_node.cpp
│   ├── include/terrasem_ros/
│   │   └── voxel_grid.hpp
│   ├── launch/terrasem_bringup.launch.py
│   ├── config/params.yaml
│   └── rviz/terrasem.rviz
├── docker/
│   ├── Dockerfile.ros2
│   └── Dockerfile.inference
├── demo/                        # Hugging Face Space (separate HF repo, mirrored here)
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md                # with HF YAML front-matter
│   └── samples/
└── tests/
    ├── test_projection.py
    ├── test_voxel_grid.py
    ├── test_bayesian_update.py
    ├── test_costmap.py
    └── test_corruptions.py
```

---

## 3. Environment and pinned versions

Two separate environments. Do not try to merge them.

### 3.1 `terrasem-dl` (training / export / demo) — Python 3.10

```yaml
# environment.yml
name: terrasem-dl
channels: [conda-forge]
dependencies:
  - python=3.10
  - pip
  - pip:
      - torch                 # install per pytorch.org selector for your CUDA
      - torchvision
      - transformers>=4.40
      - timm
      - numpy<2.0             # avoid ABI breakage with older wheels
      - opencv-python-headless
      - open3d
      - pyyaml
      - tqdm
      - matplotlib
      - pandas
      - scikit-learn
      - onnx
      - onnxruntime
      - gradio
      - plotly
      - pytest
      - ruff
```

> **Agent note:** do **not** pin a torch version in this file. Print the resolved versions after install and record them in `docs/BUILD_LOG.md`:
> ```bash
> python -c "import torch, transformers, numpy, onnxruntime as ort; print(torch.__version__, torch.version.cuda, transformers.__version__, numpy.__version__, ort.__version__)"
> ```

### 3.2 ROS 2 environment — **ROS 2 Humble on Ubuntu 22.04, inside Docker only**

Do not attempt a native ROS 2 install on the developer machine. Use `docker/Dockerfile.ros2`, base image `ros:humble-ros-base`. Docs: https://docs.ros.org/en/humble/

---

## 4. Datasets — authoritative links

**Verify licences before redistributing anything.** Record each dataset's licence in `docs/dataset_inventory.md`.

| Dataset | Role | Link |
|---|---|---|
| **RELLIS-3D** | Primary. Camera + LiDAR + semantic labels + poses + rosbags. | https://github.com/unmannedlab/RELLIS-3D |
| RELLIS-3D paper | Class ontology, benchmarks, sensor setup | https://arxiv.org/abs/2011.12954 |
| **GOOSE** | Cross-domain test set (multimodal, German off-road) | https://goose-dataset.de/ |
| GOOSE paper | Ontology + baselines | https://arxiv.org/abs/2310.16788 |
| GOOSE-Ex paper | Extra platforms/domains | https://arxiv.org/abs/2409.18788 |
| **RUGD** | Second cross-domain test set (image only) | http://rugd.vision/ — if unreachable, get the canonical link from https://paperswithcode.com/dataset/rugd |

RELLIS-3D ships Ouster 64-channel and Velodyne 32-channel point clouds in KITTI `.bin` format, with SemanticKITTI-style `.label` files, plus `pt_train.lst` / `pt_val.lst` / `pt_test.lst` split files. It contains annotations for 13,556 LiDAR scans and 6,235 images, collected on the Rellis Campus of Texas A&M University. It also provides full-stack sensor data in ROS bag format including RGB images, LiDAR point clouds, stereo pairs, high-precision GPS, and IMU.

GOOSE provides 10,000 labeled image + point cloud pairs with an ontology for unstructured terrain, plus pre-trained models. RUGD provides dense pixel-wise annotations for every fifth frame of robot video sequences, with 24 semantic categories including eight terrain types.

### 4.1 Storage budget

The full RELLIS-3D is large (tens of GB). **Start with 1–2 sequences** (e.g. `00000`, `00001`) for all of Phases 2–5. Only expand once the pipeline works. Put `data/` on external storage and symlink.

### 4.2 Download rules

- Follow the download instructions **on the RELLIS-3D GitHub README itself** (it links Google Drive / mirrors). Do not hardcode a Drive file ID from memory — read it from the repo.
- `scripts/00_download_rellis.sh` must: create `data/rellis3d/`, download, verify with checksums if published, extract, and print the resulting tree depth-3.
- If a download link is dead, stop and report to the human (rule 0.3).

---

## 5. Phase plan

Each phase: **Goal → Tasks → Exit criteria → Artifacts.**

---

### Phase 0 — Scaffold

**Goal:** an empty but complete, installable, linted, CI-green repository.

**Tasks**
1. Create the tree in §2 (empty modules with docstrings are fine).
2. `pyproject.toml` with `[project] name = "terrasem"`, and ruff config (line-length 100).
3. `.gitignore`: `data/`, `checkpoints/`, `*.onnx`, `*.pt`, `*.ckpt`, `__pycache__/`, `.venv/`, `wandb/`, `outputs/`.
4. `src/terrasem/utils/seed.py` → `set_seed(seed: int)` seeding python/numpy/torch/cuda + `torch.backends.cudnn.deterministic`.
5. `src/terrasem/utils/config.py` → load YAML, support `--override key=value` CLI merging.
6. `src/terrasem/utils/timing.py` → a `Timer` context manager and a `benchmark(fn, warmup, iters)` returning median/p95.
7. `.github/workflows/python-ci.yml`: ruff check + pytest on push.
8. Write `docs/BUILD_LOG.md` header.

**Exit criteria**
- [ ] `pip install -e .` succeeds.
- [ ] `ruff check .` clean.
- [ ] `pytest -q` passes (even with 0 real tests).
- [ ] CI badge green on GitHub.

---

### Phase 1 — Literature grounding (short, but do it)

**Goal:** agents and the human share vocabulary; the report has real citations.

**Tasks**
1. Read abstracts + method sections of the papers in §12.
2. Write `docs/related_work.md`: ≤2 pages, four subsections — off-road perception & traversability, semantic segmentation, 3D semantic occupancy mapping, camera-LiDAR fusion. For each, 3–5 papers, one sentence each on what it does and how TerraSem differs.
3. Record in the same file: the exact class ontologies of RELLIS-3D, RUGD, and GOOSE, copied from their official repos/papers (not memory).

**Exit criteria**
- [ ] `docs/related_work.md` exists with ≥12 real, verifiable citations (arXiv IDs or DOIs).
- [ ] All three ontologies transcribed with class IDs.

---

### Phase 2 — Dataset inventory and unified ontology

**Goal:** know exactly what is on disk; define one label space.

**Tasks**
1. `scripts/00_download_rellis.sh` — download sequences `00000` and `00001`.
2. `scripts/01_inspect_dataset.py`:
   - walk `data/rellis3d`, print the tree to depth 4;
   - for one frame: load the image (print shape/dtype), load the `.bin` point cloud with `np.fromfile(path, dtype=np.float32).reshape(-1, 4)` **and verify the reshape divides evenly — assert it**;
   - load the `.label` file (`np.fromfile(path, dtype=np.uint32)`) and print unique values and counts;
   - locate and print every calibration/pose file found (`*.yaml`, `*.txt`, `poses*`), dumping their contents verbatim;
   - write everything to `docs/dataset_inventory.md`.
3. `src/terrasem/datasets/ontology.py` — define **TerraSem-11**, the unified label space, and explicit mapping dicts from RELLIS-3D, RUGD, and GOOSE class IDs into it:

   | ID | Name | Notes |
   |---|---|---|
   | 0 | void/unlabeled | ignored in loss (`ignore_index=0` or 255 — pick one and be consistent) |
   | 1 | smooth_traversable | asphalt, concrete, hard dirt path |
   | 2 | rough_traversable | gravel, grass, sand |
   | 3 | high_cost_terrain | mud, tall grass, bush |
   | 4 | non_traversable_veg | tree trunk, dense vegetation |
   | 5 | water | puddle, stream |
   | 6 | obstacle_static | rock, rubble, pole, fence, building, log |
   | 7 | obstacle_dynamic | vehicle, person, animal |
   | 8 | sky | |
   | 9 | barrier | fence, barrier, wall |
   | 10 | unknown_other | |

   > Every source class must map to exactly one TerraSem-11 class. Any source class you cannot confidently map goes to `10`, and must be listed in a `UNMAPPED` comment block. Do not silently drop classes.
4. `scripts/02_build_splits.py` — produce `data/splits/{train,val,test}.txt` of frame IDs. Use the **official** RELLIS-3D `.lst` splits if present; if you create your own, split **by sequence**, never by random frame (adjacent frames are near-duplicates → leakage).
5. `src/terrasem/datasets/rellis3d.py` — a `torch.utils.data.Dataset` returning `{image, label, points, pose, frame_id}`; augmentations in `transforms.py` (random scale 0.5–2.0, random crop, hflip, photometric jitter).

**Exit criteria**
- [ ] `docs/dataset_inventory.md` is auto-generated and committed.
- [ ] Ontology mapping unit-tested: every source ID maps into `[0,10]`.
- [ ] Dataset loader returns correct shapes; a script saves 8 sample image/label overlays to `results/figs/samples/`.
- [ ] Splits have **zero sequence overlap**; a test asserts this.

---

### Phase 3 — Calibration and LiDAR→image projection ⚠️ critical gate

**Goal:** points land on the right pixels. Nothing downstream is valid until this is visually confirmed.

**Tasks**
1. `src/terrasem/calib/intrinsics.py` — parse camera intrinsics `K` and distortion from the dataset's calibration files (as recorded in Phase 2). Support undistortion via `cv2.undistort`.
2. `src/terrasem/calib/extrinsics.py` — parse the LiDAR→camera transform. **Write down explicitly in `docs/calibration_notes.md`:** quaternion order (`xyzw` vs `wxyz`), whether the transform is `T_cam_lidar` or its inverse, and the axis convention. Then justify the choice with the visual test below.
3. `src/terrasem/calib/projection.py`:
   ```
   P_cam = R @ P_lidar + t
   keep points with P_cam.z > min_depth (default 0.5 m)
   uv_h = K @ P_cam;  u = uv_h[0]/uv_h[2];  v = uv_h[1]/uv_h[2]
   keep 0 <= u < W and 0 <= v < H
   ```
   Return `(u, v, depth, valid_mask, point_indices)`. Fully vectorised NumPy; no Python loops over points.
4. **Validation (mandatory):** `scripts/20_project_lidar.py --visualize` overlays projected points, coloured by depth, on 10 images → `results/figs/projection/`. A human (or the agent, honestly) confirms: ground points fall on ground, tree points on trees, the horizon line is coherent, no mirrored/rotated-90° layout.
5. **Sanity test:** `tests/test_projection.py` — construct a synthetic point at a known 3D location with identity extrinsics and a known `K`; assert the projected pixel equals the analytically computed one within 1e-4.
6. Semantic point labelling: for each valid point, sample the segmentation map (or GT label map) at `(round(v), round(u))`. Keep a `label_confidence` = the softmax probability at that pixel (needed in Phase 5).

**Exit criteria**
- [ ] 10 overlay images saved and visually correct.
- [ ] Synthetic projection unit test passes.
- [ ] `docs/calibration_notes.md` states the exact convention used.
- [ ] Projection of a 65k-point cloud takes <10 ms on CPU (measure it).

> **If projection looks wrong:** check, in this order — (a) inverse vs forward transform, (b) quaternion order, (c) undistortion applied twice or not at all, (d) image resized but `K` not rescaled (if you resize by `s`, scale `fx, fy, cx, cy` by `s`), (e) wrong LiDAR (Ouster vs Velodyne) paired with the transform.

---

### Phase 4 — Semantic segmentation baseline

**Goal:** a trained, evaluated image segmentation model on TerraSem-11.

**Tasks**
1. `src/terrasem/models/segformer.py` — SegFormer-B0 via `transformers`:
   ```python
   from transformers import SegformerForSemanticSegmentation
   model = SegformerForSemanticSegmentation.from_pretrained(
       "nvidia/mit-b0", num_labels=11, ignore_mismatched_sizes=True)
   ```
   SegFormer outputs logits at 1/4 input resolution → upsample with `F.interpolate(..., mode="bilinear", align_corners=False)` before the loss.
2. `src/terrasem/models/dinov2_linear.py` — frozen DINOv2 ViT-S/14 backbone (`facebook/dinov2-small`) + a light conv head. Used for the label-efficiency comparison in Phase 7.
3. `scripts/10_train_seg.py`:
   - AdamW, lr `6e-5` for SegFormer, poly schedule `power=1.0`, weight decay `0.01`;
   - loss: cross-entropy with `ignore_index` for void **plus** optional class-weighted or focal variant (off-road data is heavily long-tailed — grass/sky dominate);
   - mixed precision (`torch.amp`), grad clip 1.0;
   - crop `512×512`, batch size to fit free-tier GPU (start at 8; halve on OOM);
   - `--smoke` flag: 20 steps, 8 images;
   - save best-on-val checkpoint + a `results/train_<exp>.json` with per-epoch metrics.
4. `src/terrasem/metrics/segmentation.py` — confusion-matrix based mIoU, per-class IoU, pixel accuracy, frequency-weighted IoU. **Unit-test against a hand-computed 3×3 example.**
5. `scripts/11_eval_seg.py` — evaluate on val/test, dump `results/seg_<exp>.json` with per-class IoU, plus a confusion-matrix figure.

**Exit criteria**
- [ ] `--smoke` run completes in <3 minutes.
- [ ] Full training run completes; val mIoU is recorded in `results/`.
- [ ] Per-class IoU table exists — expect low IoU on rare classes; **report it, do not hide it**.
- [ ] Qualitative grid of 12 predictions vs GT saved to `results/figs/seg/`.

> **Reference point:** published RELLIS-3D image-segmentation benchmarks report around 48.8 mIoU for HRNet+OCR, and 3D point-cloud baselines around 20–43 mIoU depending on method. Those are on the *original* ontology, not TerraSem-11, so they are **not directly comparable** — say so explicitly wherever you mention them.

---

### Phase 5 — Semantic occupancy voxel map (core contribution)

**Goal:** a multi-frame 3D voxel map with occupancy, class distribution, and uncertainty.

**Tasks**

1. `src/terrasem/mapping/voxel_grid.py` — a sparse voxel grid keyed by integer coords, backed by a dict or a hash of linearised indices. Each voxel stores:
   - `log_odds: float32`
   - `class_counts: float32[11]`
   - `hit_count: int32`, `miss_count: int32`
   - `z_min, z_max: float32` (for step-height in Phase 6)

2. **Occupancy update (binary Bayes filter, log-odds):**
   ```
   l_t = l_{t-1} + l_meas - l_0
   l_0 = 0                      (prior 0.5)
   l_occ  = log(p_hit  / (1 - p_hit)),   p_hit  = 0.7
   l_free = log(p_miss / (1 - p_miss)),  p_miss = 0.4
   clamp l_t to [-2.0, 3.5]      (prevents over-confidence, allows dynamic updates)
   occupancy p = 1 / (1 + exp(-l_t))
   ```
   Endpoints get `l_occ`; voxels traversed along the ray from sensor origin to endpoint get `l_free`. Use a **3D amanatides-woo / DDA voxel traversal**; implement it in `bayesian_update.py` and unit-test it (a ray along +x from origin must return exactly the expected voxel sequence).

3. **Semantic update (Dirichlet counts):**
   ```
   class_counts[c] += w
   w = label_confidence * range_weight
   range_weight = exp(-range / tau),  tau = 30.0 m    # distant points are less reliable
   p(c) = (class_counts[c] + alpha) / (sum(class_counts) + 11*alpha),  alpha = 0.1
   entropy H = -sum_c p(c) * log(p(c))            # normalised by log(11) -> [0,1]
   ```

4. **Pose handling:** transform each labelled cloud into a fixed world/map frame using the dataset poses (Phase 2 inventory tells you the format — likely a 4×4 or KITTI-style 12-value row per frame). **Sanity check:** accumulate 50 frames with no semantics and render the occupancy cloud — the ground plane must be flat and walls must not smear. If it smears, the pose convention is wrong (check frame ordering and whether poses are `T_world_lidar` or `T_world_base`).

5. **Ray-casting cost control:** free-space raycasting is the expensive part. Add config options: `max_range` (default 40 m), `raycast_stride` (only raycast every Nth point, default 4), `enable_freespace` (bool).

6. `scripts/30_build_map.py` — builds a map for a sequence window, saves:
   - `results/maps/<seq>_voxels.npz` (coords + log_odds + counts),
   - a PLY of occupied voxel centroids coloured by class,
   - a PLY coloured by entropy,
   - `results/maps/<seq>_stats.json`: #voxels, RAM (measure with `tracemalloc` or `psutil` RSS delta), wall time, voxel size.

7. `src/terrasem/metrics/mapping.py` — voxel-level semantic accuracy/mIoU against GT-labelled points aggregated into the same grid, plus **ECE (expected calibration error)** of the voxel class probabilities. ECE is what makes the "uncertainty-aware" claim real rather than decorative.

**Exit criteria**
- [ ] DDA raycast unit test passes.
- [ ] Bayesian update unit test: 10 consecutive hits → `p > 0.9`; 10 misses after → `p < 0.3` (clamping respected).
- [ ] A 100-frame map builds without OOM at 0.2 m voxels.
- [ ] Both PLYs render correctly (screenshot into `results/figs/maps/`).
- [ ] Voxel mIoU and ECE recorded in JSON.

---

### Phase 6 — Traversability cost map

**Goal:** a 2D `OccupancyGrid`-compatible cost map a planner could consume.

**Tasks**
1. Project the voxel map to a 2.5D grid. Per (x, y) cell compute:
   - `h_max`, `h_min` from occupied voxels → **step height** `h_max - h_ground`;
   - **slope** from a plane fit (least-squares) over a 3×3 cell neighbourhood of ground heights → angle from vertical;
   - **roughness** = std-dev of ground-point heights within the cell;
   - **semantic cost** `c_sem` = `sum_c p(c) * cost[c]` using `configs/traversability/costs.yaml`;
   - **uncertainty** `H` = normalised class entropy of the column.
2. Cost fusion:
   ```
   c_geom = w_s * clip(slope/slope_max, 0, 1)
          + w_h * clip(step/step_max,  0, 1)
          + w_r * clip(rough/rough_max,0, 1)

   c_raw  = clip(w_g * c_geom + w_m * c_sem, 0, 1)

   # uncertainty inflates cost toward "unsafe" — optimism is dangerous off-road
   c_final = c_raw + lambda_u * H * (1 - c_raw)

   cell is LETHAL (100) if: p_occ > 0.7 and class in {obstacle_static, obstacle_dynamic, water, barrier}
                            or slope > slope_max_hard
                            or step  > step_max_hard
   unknown cells -> -1 (ROS "unknown"), never 0
   ```
   Defaults: `w_s=0.4, w_h=0.4, w_r=0.2, w_g=0.5, w_m=0.5, lambda_u=0.3, slope_max=25°, slope_max_hard=35°, step_max=0.25 m, step_max_hard=0.5 m`. All in YAML.
3. Default semantic costs (`costs.yaml`, tune later, document the tuning):
   `smooth_traversable 0.0 · rough_traversable 0.2 · high_cost_terrain 0.6 · water 1.0 · non_traversable_veg 1.0 · obstacle_static 1.0 · obstacle_dynamic 1.0 · barrier 1.0 · sky ignored · unknown_other 0.5`
4. **Evaluation:** RELLIS-3D has no traversability GT. Build a **proxy ground truth**: the robot's own future trajectory (from poses) is by definition traversable. Compute, over held-out sequences, the fraction of cells actually driven through that TerraSem marks as low-cost (`<0.3`) — call it **trajectory agreement**. Also report the **false-lethal rate**: driven cells marked lethal. State clearly in the report that this is a proxy metric with a positive-only bias.
5. Outputs: PNG heatmaps + a `results/traversability_<seq>.json`.

**Exit criteria**
- [ ] Cost map renders with plausible structure (path low-cost, trees/water lethal).
- [ ] Trajectory agreement and false-lethal rate measured on ≥2 held-out sequences.
- [ ] Ablation: with vs without the uncertainty term (`lambda_u = 0` vs `0.3`) — report both.
- [ ] `tests/test_costmap.py` covers a synthetic flat plane (→ all traversable) and a synthetic wall (→ lethal).

---

### Phase 7 — The experiments that make this a research project

**Do not skip this phase. It is the differentiator.** Every experiment writes a JSON into `results/` and a plot into `results/figs/`.

**7.1 Cross-dataset generalisation** — `scripts/40_eval_generalization.py`
- Train on RELLIS-3D → evaluate zero-shot on GOOSE and RUGD (mapped into TerraSem-11).
- Report source mIoU, target mIoU, and absolute drop.
- Then one adaptation run: fine-tune on 100 target images and re-measure. Plot the recovery.

**7.2 Label efficiency** — `scripts/41_eval_label_efficiency.py`
- Subsample the training set at **1%, 5%, 10%, 25%, 50%, 100%** (stratified by sequence; fixed seeds).
- Two models: fully fine-tuned SegFormer-B0 vs frozen-DINOv2 + linear/conv head.
- Plot mIoU vs label fraction, 2 curves, error bars over 3 seeds where feasible.
- Expected narrative: the frozen foundation backbone wins in the low-label regime. **Report whatever actually happens.**

**7.3 Sensor degradation robustness** — `scripts/42_eval_corruptions.py`
Implement in `src/terrasem/corruptions/`, each with severity levels 1–5:
- **Fog** (image): atmospheric scattering, `I' = I·t + A·(1-t)`, `t = exp(-beta·d)`; with no depth map, use a depth proxy from the row index or the LiDAR depth if available.
- **Rain/motion blur:** directional streaks + Gaussian blur.
- **Low light:** gamma + Poisson–Gaussian sensor noise.
- **LiDAR beam dropout:** randomly drop whole rings/beams at 10/25/50/75%.
- **LiDAR range noise:** additive Gaussian on range, σ = 0.02–0.10 m.
- **LiDAR fog:** drop points with probability rising with range + add spurious near returns.

Compare three configurations across all corruptions: **camera-only**, **LiDAR-only** (geometry + projected labels unavailable → use geometric traversability only), and **fused**. Report voxel mIoU and trajectory agreement. Produce a degradation-curve figure. Unit-test that each corruption is deterministic given a seed and is a no-op at severity 0.

**7.4 Voxel resolution trade-off** — `scripts/43_eval_voxel_sweep.py`
- Voxel sizes `0.05, 0.1, 0.2, 0.4, 0.8` m.
- Measure: voxel mIoU, trajectory agreement, peak RAM, map-update latency per frame, #voxels.
- Plot a 3-panel figure (accuracy / memory / latency vs resolution) and state the chosen operating point with justification.

**Exit criteria**
- [ ] Four result JSONs + four figures exist and are committed.
- [ ] `docs/report.md` has a paragraph per experiment stating the finding in one sentence.
- [ ] Every number traceable to a script invocation logged in `BUILD_LOG.md`.

---

### Phase 8 — Edge optimisation and ROS 2 integration

**8.1 ONNX export** — `scripts/50_export_onnx.py`
```python
torch.onnx.export(model, dummy, "checkpoints/segformer_b0.onnx",
                  opset_version=17,
                  input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
```
Then **verify numerically**: max abs difference between PyTorch and ONNXRuntime logits must be `< 1e-3` on 20 real images. If it is not, stop and fix before quantising.

**8.2 INT8 quantisation** — `scripts/51_quantize_int8.py`
- Use ONNXRuntime **static** quantisation with a calibration reader over ~200 training images (`onnxruntime.quantization.quantize_static`, `QuantFormat.QDQ`, `QuantType.QInt8`).
- Also produce a dynamic-quantised variant for comparison.
- Re-run `11_eval_seg.py` against each `.onnx` and record the mIoU delta. **A real INT8 mIoU drop must be reported, not assumed to be zero.**

**8.3 Latency benchmark** — `scripts/52_benchmark_latency.py`
- Configs: PyTorch FP32 GPU, PyTorch FP32 CPU, ONNX FP32 CPU, ONNX INT8 CPU (+ Jetson/TensorRT only if hardware is actually available).
- 20 warm-up + 200 timed iterations, median and p95, at fixed resolution.
- Record hardware strings: `platform.processor()`, `torch.cuda.get_device_name()` when relevant, thread count.
- Output `results/latency.json` + a bar chart.

**8.4 ROS 2 package** (`ros2_ws/src/terrasem_ros`)
- **`seg_node.py` (Python):** subscribes `/camera/.../image_raw` (`sensor_msgs/Image`), runs ONNX INT8 inference, publishes `/terrasem/semantic_image` (`sensor_msgs/Image`, mono8 class IDs) and `/terrasem/semantic_confidence` (`sensor_msgs/Image`, 32FC1).
- **`voxel_fusion_node.cpp` (C++):** subscribes the LiDAR `sensor_msgs/PointCloud2` + the semantic image + confidence, time-synchronised with `message_filters::sync_policies::ApproximateTime`; looks up the transform via `tf2_ros::Buffer`; projects, updates the voxel grid (port `voxel_grid.py` logic into `include/terrasem_ros/voxel_grid.hpp` using `std::unordered_map<uint64_t, Voxel>`); publishes `/terrasem/semantic_cloud` (`PointCloud2` with an `rgb` + `label` field) and `visualization_msgs/MarkerArray` for voxels.
- **`costmap_node.py`:** consumes the semantic cloud, publishes `/terrasem/traversability` as `nav_msgs/OccupancyGrid` (values 0–100, `-1` unknown).
- **Launch file** brings up all three + RViz2 with `rviz/terrasem.rviz` preconfigured (displays for image, point cloud, markers, map).
- **Verify on a real rosbag** from RELLIS-3D: `ros2 bag play <bag> --clock` with `use_sim_time:=true` on all nodes. Record a screen capture.

**8.5 Docker + CI**
- `docker/Dockerfile.ros2`: from `ros:humble-ros-base`, install deps, `colcon build`, entrypoint sourcing the overlay.
- `.github/workflows/ros2-docker.yml`: build the image on push, run `colcon test`, push to `ghcr.io/<user>/terrasem:latest` on main.

**Exit criteria**
- [ ] ONNX parity `< 1e-3` verified and logged.
- [ ] INT8 speedup and mIoU delta measured and recorded.
- [ ] `colcon build` succeeds in Docker; CI green.
- [ ] A recorded video/GIF of RViz2 showing live semantic cloud + cost map on a real bag.
- [ ] `docker run ... ros2 launch terrasem_ros terrasem_bringup.launch.py` works from a clean pull.

---

### Phase 9 — Free deployment, README, report

**9.1 Hugging Face Space (free CPU tier)** — https://huggingface.co/spaces

`demo/app.py` (Gradio) must:
1. Load the **INT8 ONNX** model (CPU-only; do not import torch in the Space — keep the image small and cold-start fast).
2. Offer 4–6 bundled sample frames **plus** user image upload.
3. Show three tabs: **Segmentation overlay**, **Traversability heatmap**, **3D semantic voxel view** (precomputed voxel map rendered with `plotly.graph_objects.Scatter3d`, with a dropdown for colouring = semantic / occupancy / entropy — precompute the voxels offline and ship a small `.npz`, do not build maps live on free CPU).
4. Display measured inference latency per request.
5. Include a clear caption: this demo shows the perception output; the full ROS 2 stack runs via the Docker image (link it).

`demo/README.md` needs HF front-matter:
```yaml
---
title: TerraSem Off-Road Semantic Occupancy
emoji: 🛻
colorFrom: green
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
---
```

**Constraints:** free Spaces CPU is small — keep the model `< 50 MB`, total repo `< 1 GB`, and per-request compute under a few seconds. Use HF LFS for the `.onnx` and `.npz`. **Check the sample-frame licence before committing dataset images** (rule 0.3).

**9.2 README.md** (the recruiter-facing artifact) — in this order:
1. One-line description + **demo link** + Docker pull command, at the very top.
2. A 60–90 s demo GIF (RViz2 + web demo side by side).
3. Architecture diagram (Mermaid is fine): sensors → segmentation → projection → voxel fusion → cost map → planner interface.
4. **Results tables** (real numbers only, with hardware and resolution labelled).
5. The four experiment figures.
6. Quickstart: pip install, download data, train, build map, run ROS 2.
7. Limitations & future work — be candid (proxy traversability GT, single primary dataset, no real vehicle test, CPU-only demo).
8. Citations + dataset licences + acknowledgements.

**9.3 `docs/report.md`** — 2 pages: problem, method, experiments, results, limitations. This is what you send with the internship application.

**Exit criteria**
- [ ] Space is live and loads in <30 s cold.
- [ ] README has zero `TODO` and zero unmeasured numbers.
- [ ] `docs/report.md` complete.
- [ ] Repo is public, CI green, license present.

---

## 6. Testing requirements

Minimum tests before Phase 9 is considered done:

| Test | Asserts |
|---|---|
| `test_projection.py` | synthetic point projects to analytically known pixel; out-of-frustum points filtered |
| `test_voxel_grid.py` | world↔voxel index round-trip; insertion/lookup; memory grows sub-linearly with duplicate inserts |
| `test_bayesian_update.py` | log-odds converge and clamp; DDA ray returns the expected voxel sequence; Dirichlet posterior sums to 1 |
| `test_costmap.py` | flat plane → traversable; wall → lethal; unknown cells → `-1` |
| `test_corruptions.py` | severity 0 is identity; deterministic under a fixed seed; output dtype/range preserved |
| `test_ontology.py` | every source class maps into `[0,10]`; no source class unmapped silently |
| `test_metrics.py` | mIoU matches a hand-computed 3×3 confusion matrix |
| `test_splits.py` | no sequence appears in two splits |

---

## 7. Compute plan (free tiers)

- **Kaggle Notebooks:** ~30 GPU-hours/week (T4/P100). Best for the Phase 7 sweeps. Save checkpoints to Kaggle Datasets between sessions.
- **Google Colab free:** good for smoke runs; session limits make long runs risky — always checkpoint every epoch and support `--resume`.
- Keep SegFormer-B0 at `512×512` crops; that fits comfortably in 16 GB.
- If a sweep is estimated at >20 GPU-hours, reduce epochs and say so in the report rather than silently shrinking the experiment.

---

## 8. Known traps (each has cost people days)

1. **Resizing images without rescaling `K`.** Always scale `fx, fy, cx, cy`.
2. **Quaternion order.** `scipy.spatial.transform.Rotation.from_quat` expects `[x, y, z, w]`; many YAML files store `[w, x, y, z]`.
3. **Random frame splits.** Adjacent frames are near-duplicates → inflated mIoU. Split by sequence.
4. **`ignore_index` mismatch.** Void is `0` in the ontology table above but many losses default to `255`. Pick one, define it once in `ontology.py`, and use it everywhere.
5. **SegFormer output stride.** Logits come out at 1/4 resolution. Upsample before the loss, not after `argmax`.
6. **`.label` files carry semantics in the lower 16 bits** and instance IDs in the upper 16 in SemanticKITTI format — mask with `& 0xFFFF`. Verify against the real unique values you printed in Phase 2.
7. **Free-space raycasting is the bottleneck**, not the network. Profile before optimising the model.
8. **Unknown vs free in `OccupancyGrid`.** `0` means free (confidently drivable). Unobserved must be `-1`.
9. **Docker + GPU.** The ROS 2 image is CPU-only by design; run ONNX INT8 there. Do not fight CUDA-in-Docker for this project.
10. **HF Spaces cold start.** Lazy-load the model inside the first request handler, not at import time, or the Space times out on build.

---

## 9. Definition of done (whole project)

- [ ] Public GitHub repo, MIT licensed, CI green.
- [ ] Live Hugging Face Space.
- [ ] Public GHCR Docker image running the ROS 2 stack on a dataset bag.
- [ ] `results/` contains ≥8 committed JSON result files and ≥8 figures, all reproducible from committed scripts.
- [ ] README with demo GIF, architecture diagram, and real results tables.
- [ ] `docs/report.md` (2 pages) and `docs/related_work.md`.
- [ ] `docs/BUILD_LOG.md` with an entry per phase.

---

## 10. Suggested schedule (6 weeks, part-time)

| Week | Phases |
|---|---|
| 1 | 0, 1, 2 |
| 2 | 3, 4 |
| 3 | 5 |
| 4 | 6, 7.1–7.2 |
| 5 | 7.3–7.4, 8.1–8.3 |
| 6 | 8.4–8.5, 9 |

If time runs short, cut in this order: 7.1 adaptation run → 7.4 extra voxel sizes → the DINOv2 arm of 7.2. **Never cut Phase 3 validation, Phase 5, or Phase 9 README/demo.**

---

## 11. Resume bullets (fill the `X/Y/Z` from `results/` only)

- Built a camera+LiDAR **semantic occupancy and traversability mapping** pipeline for off-road autonomy on RELLIS-3D, with a C++ voxel-fusion node and Python inference node in **ROS 2 Humble**.
- Added Bayesian log-odds occupancy with Dirichlet semantic voxels and entropy-based uncertainty; measured voxel mIoU **X** and ECE **Y**.
- Evaluated cross-dataset generalisation (RELLIS-3D → GOOSE/RUGD, **Z** mIoU drop) and robustness to fog and **50% LiDAR beam dropout**; fusion recovered **W** mIoU over camera-only.
- Quantised to **INT8 via ONNX Runtime** — **A ms/frame (B× speedup, C mIoU drop)**; shipped a live Hugging Face demo and a CI-tested Docker image.

---

## 12. Reference reading (verified IDs only)

| Topic | Paper | Link |
|---|---|---|
| Primary dataset | RELLIS-3D: Data, Benchmarks and Analysis | https://arxiv.org/abs/2011.12954 |
| Off-road dataset | GOOSE: Perception in Unstructured Environments | https://arxiv.org/abs/2310.16788 |
| Off-road dataset | GOOSE-Ex | https://arxiv.org/abs/2409.18788 |
| Dataset | RUGD (IROS 2019) | https://paperswithcode.com/dataset/rugd |
| Segmentation | SegFormer | https://arxiv.org/abs/2105.15203 |
| Backbone | DINOv2 | https://arxiv.org/abs/2304.07193 |
| Occupancy mapping | OctoMap | https://octomap.github.io/ |
| Semantic occupancy | MonoScene | https://arxiv.org/abs/2112.00726 |
| Semantic occupancy | TPVFormer | https://arxiv.org/abs/2302.07817 |
| Occupancy benchmark | Occ3D | https://arxiv.org/abs/2304.14365 |
| Fusion | BEVFusion | https://arxiv.org/abs/2205.13542 |
| Robustness protocol | Common Corruptions (ImageNet-C) | https://arxiv.org/abs/1903.12261 |
| Uncertainty | Deep Ensembles | https://arxiv.org/abs/1612.01474 |
| Calibration | On Calibration of Modern Neural Networks | https://arxiv.org/abs/1706.04599 |

Survey of off-road/unstructured datasets and methods (useful index): https://github.com/chaytonmin/Survey-Autonomous-Driving-in-Unstructured-Environments

Tooling docs: ROS 2 Humble https://docs.ros.org/en/humble/ · Open3D https://www.open3d.org/docs/release/ · ONNX Runtime quantisation https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html · Gradio https://www.gradio.app/docs · HF Spaces https://huggingface.co/docs/hub/spaces

---

**Final reminder to agents:** a smaller system with honest, reproducible measurements beats a larger system with invented ones. When in doubt, measure it or mark it `TODO: not yet measured`.