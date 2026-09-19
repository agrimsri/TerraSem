# Related Work

**TerraSem — Phase 1 Literature Grounding**

This document covers four areas relevant to TerraSem: off-road perception and traversability,
semantic segmentation, 3D semantic occupancy mapping, and camera-LiDAR fusion.
All citations reference verified arXiv IDs or official URLs listed in BUILD.md §12.

---

## 1. Off-Road Perception, Datasets, and Traversability

**RELLIS-3D** (Jiang et al., 2021) — arXiv:2011.12954  
A multimodal off-road dataset collected at Texas A&M University's Rellis Campus, providing
13,556 annotated LiDAR scans and 6,235 RGB images with 20 semantic classes (including void)
from an Ouster 64-channel and Velodyne 32-channel LiDAR plus a Basler colour camera.
The dataset ships full ROS bags with GPS/IMU. TerraSem *uses* RELLIS-3D as its primary
training dataset but extends it with a 3D voxel map and traversability cost map, neither of
which are provided by the dataset itself.

**GOOSE** (Böddeker et al., 2023) — arXiv:2310.16788  
The German Outdoor and Offroad Dataset provides 10,000 labeled image+point-cloud pairs for
unstructured outdoor environments, with an open ontology and pre-trained segmentation models.
TerraSem uses GOOSE as a *cross-domain zero-shot evaluation* target, not for training, to
assess how well a model trained on RELLIS-3D generalises to European off-road conditions.

**GOOSE-Ex** (Böddeker et al., 2024) — arXiv:2409.18788  
Extends GOOSE with 5,000 additional labeled multimodal frames from a robotic excavator and
a quadruped platform, emphasising cross-platform generalisation. TerraSem evaluates on the
combined GOOSE frame pool; GOOSE-Ex broadens the domain gap test.

**RUGD** (Wigness et al., IROS 2019) — https://paperswithcode.com/dataset/rugd  
Robot Unstructured Ground Dataset: dense pixel-wise semantic annotations for robot video
sequences at 24 categories (8 terrain types). Frame-rate annotation every 5th frame.
TerraSem maps RUGD labels into TerraSem-11 and uses RUGD as a second zero-shot evaluation
target for cross-dataset generalisation (image-only; no LiDAR).

---

## 2. Semantic Segmentation

**SegFormer** (Xie et al., NeurIPS 2021) — arXiv:2105.15203  
A transformer-based segmentation framework with a hierarchical Mix Transformer (MiT) encoder
and a lightweight MLP decoder. SegFormer avoids positional encoding and produces logits at
1/4 input resolution, making it efficient at multiple scales (B0–B5). TerraSem uses
SegFormer-B0 fine-tuned on RELLIS-3D as its primary segmentation backbone; the key
difference is that TerraSem applies it to off-road imagery with a custom 11-class ontology
and uses its output to *label LiDAR points* for downstream 3D mapping.

**DINOv2** (Oquab et al., 2024) — arXiv:2304.07193  
A self-supervised ViT backbone (up to 1B parameters) trained on a curated large-scale image
collection via a combination of DINO and iBOT objectives at scale. The resulting features
transfer to diverse tasks without fine-tuning. TerraSem uses DINOv2 ViT-S/14 as a *frozen*
backbone with a trainable linear/conv head to study label efficiency — expecting frozen
DINOv2 to outperform full SegFormer fine-tuning in the low-label (1–10%) regime.

---

## 3. 3D Semantic Occupancy Mapping

**OctoMap** (Hornung et al., AURO 2013) — https://octomap.github.io/  
A probabilistic 3D occupancy mapping framework using an octree data structure with
log-odds Bayesian updates; each voxel stores a single occupancy probability.
TerraSem extends the OctoMap Bayesian update principle by adding a Dirichlet-style
*class count vector* per voxel, normalised entropy as an uncertainty measure, and
a 2.5D traversability cost projection — none of which are present in OctoMap.

**MonoScene** (Cao and de Charette, CVPR 2022) — arXiv:2112.00726  
The first method to infer dense 3D semantic scene completion from a *single monocular
RGB image*, using 2D–3D UNet bridging and a 3D context relation prior. TerraSem differs
in that it uses both camera and LiDAR for building the voxel map and targets an outdoor
off-road domain, whereas MonoScene was evaluated on indoor SemanticKITTI scenes.

**TPVFormer** (Huang et al., CVPR 2023) — arXiv:2302.07817  
Proposes a tri-perspective view (BEV + two perpendicular planes) for camera-based 3D
semantic occupancy prediction via attention-based image feature lifting.
TerraSem uses explicit LiDAR point clouds projected into the segmented image rather than
implicitly lifting camera features — this keeps our approach interpretable and calibratable,
at the cost of requiring a LiDAR sensor.

**Occ3D** (Tian et al., 2023) — arXiv:2304.14365  
A large-scale 3D occupancy prediction benchmark derived from Waymo and nuScenes, with
a dense label-generation pipeline and a baseline CTF-Occ network. TerraSem evaluates
on RELLIS-3D (off-road) rather than structured urban datasets, reports ECE in addition
to mIoU, and prioritises interpretability over raw benchmark accuracy.

---

## 4. Camera-LiDAR Fusion

**BEVFusion** (Liu et al., 2022) — arXiv:2205.13542  
Fuses multi-modal sensor data (camera + LiDAR) in a shared bird's-eye view representation,
using optimised BEV pooling to eliminate the view-transformation bottleneck. Achieves
state-of-the-art on nuScenes for 3D detection and BEV map segmentation.
TerraSem uses a simpler *projection-based* fusion (LiDAR points projected into the segmented
camera image) rather than learned BEV feature fusion, making it computationally lighter and
more suitable for CPU-only deployment via ONNX.

---

## 5. Robustness and Uncertainty

**ImageNet-C** (Hendrycks & Dietterich, ICLR 2019) — arXiv:1903.12261  
Establishes a benchmark for image classifier robustness to 15 common corruptions (fog, rain,
noise, etc.) at 5 severity levels. TerraSem borrows this severity-1-to-5 protocol for its
sensor corruption experiments (Phase 7.3), extending it to LiDAR-specific corruptions
(beam dropout, range noise, LiDAR fog).

**Deep Ensembles** (Lakshminarayanan et al., NeurIPS 2017) — arXiv:1612.01474  
Shows that an ensemble of neural networks with adversarial training provides well-calibrated
uncertainty estimates, often outperforming Bayesian approximations. TerraSem instead uses
a single-model Dirichlet prior per voxel as an uncertainty proxy, which avoids the 5×
compute overhead of ensembles; BUILD.md §0.4 requires reporting ECE to validate the claim.

**On Calibration of Modern Neural Networks** (Guo et al., ICML 2017) — arXiv:1706.04599  
Demonstrates that modern deep networks are overconfident and proposes temperature scaling
as a simple, effective post-hoc calibration method. TerraSem reports ECE of the voxel class
probabilities as a first-class metric (Phase 5), making the "uncertainty-aware" claim
empirically verifiable rather than decorative.

---

## Appendix A — Dataset Ontologies

### A.1 RELLIS-3D — 20 classes (including void)

Source: https://github.com/unmannedlab/RELLIS-3D (README + Ontology PDF)  
RELLIS-3D uses SemanticKITTI-style `.label` files where the label ID is stored in the
lower 16 bits (mask with `& 0xFFFF`).

> **Agent note (Phase 2):** The table below is compiled from the RELLIS-3D README and paper.
> **Verify every ID against `docs/dataset_inventory.md`** after running
> `scripts/01_inspect_dataset.py`. Do NOT trust memory on ID values.

| ID | Class Name | TerraSem-11 Mapping |
|---|---|---|
| 0 | void | 0 — void_unlabeled |
| 1 | concrete | 1 — smooth_traversable |
| 3 | grass | 2 — rough_traversable |
| 4 | sand | 2 — rough_traversable |
| 5 | water | 5 — water |
| 6 | tree | 4 — non_traversable_veg |
| 7 | bush | 3 — high_cost_terrain |
| 8 | building | 6 — obstacle_static |
| 9 | sky | 8 — sky |
| 10 | asphalt | 1 — smooth_traversable |
| 12 | rubble | 6 — obstacle_static |
| 15 | mud | 3 — high_cost_terrain |
| 17 | puddle (veg context?) | 5 — water *(verify)* |
| 18 | puddle | 5 — water |
| 19 | person | 7 — obstacle_dynamic |
| 23 | fence | 6 — obstacle_static |
| 27 | vehicle | 7 — obstacle_dynamic |
| 29 | bush (variant) | 3 — high_cost_terrain |
| 31 | pole | 6 — obstacle_static |
| 33 | log | 6 — obstacle_static |
| 34 | object (generic) | 6 — obstacle_static |

Classes 2 (dirt), 11, 13, 14, 16 — see `RELLIS3D_UNMAPPED` in `ontology.py` for notes.

### A.2 RUGD — 24 classes

Source: http://rugd.vision/ (IROS 2019, Wigness et al.)

> **Agent note (Phase 2):** RUGD website was unreachable during Phase 1 (TLS timeout).
> IDs below are from the RUGD paper description. Verify against downloaded annotation files.

| ID | Class Name | TerraSem-11 Mapping |
|---|---|---|
| 0 | void | 0 — void_unlabeled |
| 1 | dirt | 2 — rough_traversable |
| 2 | sand | 3 — high_cost_terrain |
| 3 | grass | 2 — rough_traversable |
| 4 | gravel | 2 — rough_traversable |
| 5 | mud | 3 — high_cost_terrain |
| 6 | water | 5 — water |
| 7 | rock | 6 — obstacle_static |
| 8 | tree | 4 — non_traversable_veg |
| 9 | log | 6 — obstacle_static |
| 10 | bush | 4 — non_traversable_veg |
| 11 | bridge | 6 — obstacle_static |
| 12 | sky | 8 — sky |
| 13 | building | 6 — obstacle_static |
| 14 | vehicle | 7 — obstacle_dynamic |
| 15 | person | 7 — obstacle_dynamic |
| 16 | animal | 7 — obstacle_dynamic |
| 17 | fence | 6 — obstacle_static |
| 18 | barrier | 9 — barrier |
| 19 | sign | 10 — unknown_other |
| 20 | concrete | 1 — smooth_traversable |
| 21 | asphalt | 1 — smooth_traversable |
| 22 | bicycle | 7 — obstacle_dynamic |
| 23 | flowers | 3 — high_cost_terrain |

### A.3 GOOSE — Pending (Phase 2)

Source: https://goose-dataset.de/

> **Agent note:** GOOSE ships a structured ontology JSON/YAML. Download the dataset first,
> then update this table and `src/terrasem/datasets/ontology.py` (GOOSE_TO_TERRASEM).
> The GOOSE paper (arXiv:2310.16788) mentions an "ontology for unstructured terrain" —
> fetch it from the official website after access is granted.

| ID | Class Name | TerraSem-11 Mapping |
|---|---|---|
| 0 | void/unlabeled | 0 — void_unlabeled |
| *TBD* | *All others — see Phase 2* | *TBD* |

---

## Appendix B — Citation Summary

| # | Paper | arXiv / URL |
|---|---|---|
| 1 | RELLIS-3D | arXiv:2011.12954 |
| 2 | GOOSE | arXiv:2310.16788 |
| 3 | GOOSE-Ex | arXiv:2409.18788 |
| 4 | RUGD | http://rugd.vision/ |
| 5 | SegFormer | arXiv:2105.15203 |
| 6 | DINOv2 | arXiv:2304.07193 |
| 7 | OctoMap | https://octomap.github.io/ |
| 8 | MonoScene | arXiv:2112.00726 |
| 9 | TPVFormer | arXiv:2302.07817 |
| 10 | Occ3D | arXiv:2304.14365 |
| 11 | BEVFusion | arXiv:2205.13542 |
| 12 | ImageNet-C | arXiv:1903.12261 |
| 13 | Deep Ensembles | arXiv:1612.01474 |
| 14 | NN Calibration | arXiv:1706.04599 |
