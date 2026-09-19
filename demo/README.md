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

# TerraSem Demo

Live demo for TerraSem — uncertainty-aware semantic occupancy and traversability mapping for off-road autonomous navigation.

**Note:** This demo shows perception output on sample frames. The full ROS 2 stack runs via the Docker image:
```
docker pull ghcr.io/agrimsri/terrasem:latest
docker run --rm -it ghcr.io/agrimsri/terrasem:latest \
  ros2 launch terrasem_ros terrasem_bringup.launch.py
```

## Tabs
- **Segmentation overlay** — TerraSem-11 class predictions overlaid on the RGB image
- **Traversability heatmap** — 2.5D cost map
- **3D Semantic Voxels** — precomputed voxel map coloured by class / occupancy / entropy

TODO (Phase 9): implement demo app.
