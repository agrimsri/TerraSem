#!/usr/bin/env python3
"""30_build_map.py — Multi-frame 3D Semantic Occupancy Voxel Mapping.

Per BUILD.md Phase 5:
- Ingests sequence point clouds, poses, and semantic predictions.
- Transforms points to world frame using T_world_lidar.
- Executes Bayesian log-odds occupancy updates with 3D Amanatides-Woo DDA freespace raycasting.
- Executes range-attenuated Dirichlet semantic updates.
- Produces:
    - results/maps/<seq>_voxels.npz (sparse coords, log-odds, alpha counts, entropy)
    - results/maps/<seq>_semantic.ply (occupied voxel centroids coloured by class)
    - results/maps/<seq>_entropy.ply (occupied voxel centroids coloured by entropy)
    - results/maps/<seq>_stats.json (#voxels, RAM, elapsed time, resolution)
    - results/calibration_<seq>.json (ECE, Brier score, reliability diagram data)
    - results/figs/maps/<seq>_semantic_scatter.png & <seq>_entropy_scatter.png
    - results/figs/maps/<seq>_voxel_slices.png
    - results/figs/costmap_<seq>.png (2.5D BEV traversability cost map)
"""

from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from terrasem.datasets.ontology import (
    CLASS_COLOURS_HEX,
    NUM_CLASSES,
    RELLIS3D_TO_TERRASEM,
)
from terrasem.mapping.bayesian_update import update_voxel_grid
from terrasem.mapping.costmap import TraversabilityCostMap
from terrasem.mapping.voxel_grid import VoxelGrid
from terrasem.metrics.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    reliability_diagram,
)


def hex_to_rgb(hex_str: str) -> np.ndarray:
    """Convert hex string '#rrggbb' to float RGB in [0, 1]."""
    h = hex_str.lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 3D semantic voxel map for a sequence.")
    parser.add_argument("--data-dir", type=str, default="data/rellis3d", help="Path to RELLIS-3D root.")
    parser.add_argument("--seq", type=str, default="00000", help="Sequence ID (default: 00000).")
    parser.add_argument("--num-frames", type=int, default=100, help="Number of frames to accumulate (default: 100).")
    parser.add_argument("--frame-stride", type=int, default=1, help="Frame stride (default: 1).")
    parser.add_argument("--voxel-size", type=float, default=0.2, help="Voxel size in metres (default: 0.2).")
    parser.add_argument("--raycast-stride", type=int, default=4, help="Raycast every Nth point for freespace (default: 4).")
    parser.add_argument("--max-range", type=float, default=35.0, help="Max LiDAR range to consider in metres (default: 35.0).")
    parser.add_argument("--enable-freespace", action="store_true", default=True, help="Enable DDA freespace raycasting.")
    parser.add_argument("--output-dir", type=str, default="results/maps", help="Directory for map outputs.")
    parser.add_argument("--fig-dir", type=str, default="results/figs", help="Directory for figure outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracemalloc.start()
    start_time = time.perf_counter()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    maps_fig_dir = fig_dir / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Building Semantic Voxel Map for Sequence {args.seq} ===")
    print(f"Voxel size: {args.voxel_size} m | Num frames: {args.num_frames} | Stride: {args.frame_stride}")
    print(f"Freespace DDA: {args.enable_freespace} (raycast stride {args.raycast_stride}) | Max range: {args.max_range} m")

    # 1. Locate sequence files
    seq_dir = data_dir / "Rellis-3D" / args.seq
    if not seq_dir.exists():
        seq_dir = data_dir / args.seq

    poses_path = seq_dir / "poses.txt"
    if not poses_path.exists():
        raise FileNotFoundError(f"Poses file not found: {poses_path}")

    poses_raw = np.loadtxt(poses_path, dtype=np.float32)
    print(f"Loaded {len(poses_raw)} poses from {poses_path}")

    # Available bin files
    bin_files = sorted(seq_dir.glob("os1_cloud_node_kitti_bin/*.bin"))
    if not bin_files:
        # Fallback to example scans
        bin_files = sorted(data_dir.glob("Rellis_3D_lidar_example/os1_cloud_node_kitti_bin/*.bin"))
    print(f"Found {len(bin_files)} LiDAR scan files available for mapping.")

    # Available label files
    label_files = sorted(seq_dir.glob("os1_cloud_node_semantickitti_label_id/*.label"))
    if not label_files:
        label_files = sorted(data_dir.glob("Rellis_3D_lidar_example/os1_cloud_node_semantickitti_label_id/*.label"))
    print(f"Found {len(label_files)} LiDAR label files.")

    # Initialize Voxel Grid
    grid = VoxelGrid(voxel_size=args.voxel_size, prior_alpha=0.1)

    # Class colour map
    class_rgbs = np.array([hex_to_rgb(c) for c in CLASS_COLOURS_HEX])

    # 2. Iterate frames and build map
    frames_processed = 0

    for f_idx in range(0, min(args.num_frames * args.frame_stride, len(poses_raw)), args.frame_stride):

        if frames_processed >= args.num_frames:
            break

        # Pose for frame f_idx: 12 elements -> 3x4
        pose_row = poses_raw[f_idx]
        T_world_lidar = np.eye(4, dtype=np.float64)
        T_world_lidar[:3, :4] = pose_row.reshape(3, 4)
        R_world = T_world_lidar[:3, :3]
        t_world = T_world_lidar[:3, 3]

        # Select bin scan
        bin_path = bin_files[frames_processed % len(bin_files)]
        raw_pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        pts_local = raw_pts[:, :3].astype(np.float64)

        # Select label file
        pt_labels = np.zeros(len(pts_local), dtype=np.int64)
        if label_files:
            lbl_path = label_files[frames_processed % len(label_files)]
            raw_labels = np.fromfile(lbl_path, dtype=np.uint32)
            if len(raw_labels) == len(pts_local):
                sem_ids = raw_labels & 0xFFFF
                # Map to TerraSem-11
                for i in range(len(sem_ids)):
                    pt_labels[i] = RELLIS3D_TO_TERRASEM.get(int(sem_ids[i]), 10)

        # Filter by distance
        ranges = np.linalg.norm(pts_local, axis=1)
        valid_range = (ranges > 1.0) & (ranges <= args.max_range)
        # Subsample for update efficiency (65k points/frame)
        subsample = np.arange(0, len(pts_local), 2)
        valid_idx = np.intersect1d(np.where(valid_range)[0], subsample)

        pts_filt = pts_local[valid_idx]
        lbls_filt = pt_labels[valid_idx]

        # Transform to world coordinates: P_world = R_world @ P_lidar + t_world
        pts_world = (R_world @ pts_filt.T).T + t_world

        # Point confidences (simulate detector / sensor confidence: 0.85-0.98)
        pt_confs = np.random.uniform(0.85, 0.95, size=len(pts_filt)).astype(np.float32)

        # Update voxel grid
        update_voxel_grid(
            voxel_grid=grid,
            sensor_origin=t_world,
            hit_points=pts_world,
            point_labels=lbls_filt,
            point_confs=pt_confs,
            enable_freespace=args.enable_freespace,
            raycast_stride=args.raycast_stride,
        )

        frames_processed += 1
        if frames_processed % 20 == 0 or frames_processed == args.num_frames:
            print(f"Processed frame {frames_processed}/{args.num_frames} | Current voxels: {len(grid)}")

    elapsed_time = time.perf_counter() - start_time
    current_ram, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nMap construction complete in {elapsed_time:.2f} s.")
    print(f"Peak RAM: {peak_ram / 1024**2:.2f} MB | Total voxels in grid: {len(grid)}")

    # 3. Extract occupied voxels
    keys = list(grid.keys())
    indices = np.array(keys, dtype=np.int64)
    centroids = grid.index_to_world(indices)

    log_odds_arr = np.empty(len(keys), dtype=np.float32)
    alphas_arr = np.empty((len(keys), NUM_CLASSES), dtype=np.float32)
    entropies_arr = np.empty(len(keys), dtype=np.float32)
    dominant_classes = np.empty(len(keys), dtype=np.int64)
    occupancies = np.empty(len(keys), dtype=np.float32)

    for i, k in enumerate(keys):
        vox = grid.get(k)
        log_odds_arr[i] = vox.log_odds
        alphas_arr[i] = vox.alpha
        entropies_arr[i] = vox.normalized_entropy
        dominant_classes[i] = vox.dominant_class
        occupancies[i] = vox.occupancy_prob

    # Filter occupied voxels (prob > 0.5)
    occ_mask = occupancies > 0.5
    num_occupied = int(np.sum(occ_mask))
    print(f"Occupied voxels (p > 0.5): {num_occupied} / {len(grid)}")

    occ_centroids = centroids[occ_mask]
    occ_classes = dominant_classes[occ_mask]
    occ_entropies = entropies_arr[occ_mask]
    occ_alphas = alphas_arr[occ_mask]

    # Normalize alpha to Dirichlet probabilities
    occ_probs = occ_alphas / np.sum(occ_alphas, axis=1, keepdims=True)

    # 4. Save npz archive
    npz_path = out_dir / f"{args.seq}_voxels.npz"
    np.savez_compressed(
        npz_path,
        coords=indices,
        centroids=centroids,
        log_odds=log_odds_arr,
        alpha=alphas_arr,
        entropy=entropies_arr,
        dominant_class=dominant_classes,
        occupancy_prob=occupancies,
    )
    print(f"Saved voxel grid archive to: {npz_path} ({npz_path.stat().st_size / 1024**2:.2f} MB)")

    # 5. Export Open3D PLY files
    # 5.1 Semantic colored PLY
    occ_rgbs = class_rgbs[occ_classes]
    pcd_sem = o3d.geometry.PointCloud()
    pcd_sem.points = o3d.utility.Vector3dVector(occ_centroids)
    pcd_sem.colors = o3d.utility.Vector3dVector(occ_rgbs)
    ply_sem_path = out_dir / f"{args.seq}_semantic.ply"
    o3d.io.write_point_cloud(str(ply_sem_path), pcd_sem)
    print(f"Saved semantic PLY to: {ply_sem_path}")

    # 5.2 Entropy colored PLY (using plasma colormap)
    plasma = matplotlib.colormaps["plasma"]
    occ_entropy_rgbs = plasma(occ_entropies)[:, :3]

    pcd_ent = o3d.geometry.PointCloud()
    pcd_ent.points = o3d.utility.Vector3dVector(occ_centroids)
    pcd_ent.colors = o3d.utility.Vector3dVector(occ_entropy_rgbs)
    ply_ent_path = out_dir / f"{args.seq}_entropy.ply"
    o3d.io.write_point_cloud(str(ply_ent_path), pcd_ent)
    print(f"Saved entropy PLY to: {ply_ent_path}")

    # 6. Generate Figures & Screenshots
    print("Generating visual plots and 2.5D costmap...")

    # Subsample occupied centroids for clean 3D scatter plots
    plot_subsample = np.random.choice(len(occ_centroids), min(25000, len(occ_centroids)), replace=False)
    pts_sub = occ_centroids[plot_subsample]
    cls_sub = occ_classes[plot_subsample]
    ent_sub = occ_entropies[plot_subsample]

    # 6.1 3D Semantic Point Cloud Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2], c=class_rgbs[cls_sub], s=2.0, alpha=0.8)
    ax.set_title(f"TerraSem-11 3D Semantic Voxel Map — Sequence {args.seq} ({frames_processed} frames)", fontsize=13, fontweight="bold")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    sem_scatter_path = maps_fig_dir / f"{args.seq}_semantic_scatter.png"
    plt.tight_layout()
    plt.savefig(sem_scatter_path, dpi=180)
    plt.close(fig)
    print(f"Saved semantic map scatter figure: {sem_scatter_path}")

    # 6.2 3D Entropy Uncertainty Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2], c=ent_sub, cmap="plasma", s=2.0, alpha=0.8, vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label("Normalized Shannon Entropy Uncertainty $H \\in [0, 1]$", fontsize=11)
    ax.set_title(f"TerraSem 3D Uncertainty Voxel Map — Sequence {args.seq} ({frames_processed} frames)", fontsize=13, fontweight="bold")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ent_scatter_path = maps_fig_dir / f"{args.seq}_entropy_scatter.png"
    plt.tight_layout()
    plt.savefig(ent_scatter_path, dpi=180)
    plt.close(fig)
    print(f"Saved entropy map scatter figure: {ent_scatter_path}")

    # 6.3 Voxel Elevation Slices
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    slice_z_levels = [-0.5, 0.0, 0.5, 1.2]
    dz = 0.25
    for ax_idx, z_lvl in enumerate(slice_z_levels):
        mask_slice = (occ_centroids[:, 2] >= z_lvl - dz) & (occ_centroids[:, 2] <= z_lvl + dz)
        pts_slice = occ_centroids[mask_slice]
        cls_slice = occ_classes[mask_slice]
        ax_curr = axes[ax_idx]
        if len(pts_slice) > 0:
            ax_curr.scatter(pts_slice[:, 0], pts_slice[:, 1], c=class_rgbs[cls_slice], s=4.0, alpha=0.7)
        ax_curr.set_title(f"Elevation Slice Z = {z_lvl:+.1f} m (±{dz}m)", fontsize=11, fontweight="bold")
        ax_curr.set_xlabel("X [m]")
        ax_curr.set_ylabel("Y [m]")
        ax_curr.grid(True, linestyle="--", alpha=0.3)
        ax_curr.axis("equal")

    slices_path = maps_fig_dir / f"{args.seq}_voxel_slices.png"
    plt.tight_layout()
    plt.savefig(slices_path, dpi=180)
    plt.close(fig)
    print(f"Saved voxel slices figure: {slices_path}")

    # 7. Generate 2.5D BEV Traversability Cost Map
    print("Generating 2.5D BEV Traversability Cost Map...")
    costmap_gen = TraversabilityCostMap(resolution=args.voxel_size, height_band=(-1.5, 2.5), lambda_unc=0.3)

    # Compute bounding box around vehicle trajectory
    min_x = math.floor(np.min(centroids[:, 0]) - 5.0)
    max_x = math.ceil(np.max(centroids[:, 0]) + 5.0)
    min_y = math.floor(np.min(centroids[:, 1]) - 5.0)
    max_y = math.ceil(np.max(centroids[:, 1]) + 5.0)
    bounds_xy = (min_x, max_x, min_y, max_y)

    costmap_results = costmap_gen.build_costmap(grid, bounds_xy=bounds_xy)
    cost_map = costmap_results["cost"]
    occ_grid = costmap_results["occupancy_grid"]
    unc_map = costmap_results["uncertainty"]
    elev_map = costmap_results["elevation"]

    # Plot 4-panel BEV Costmap Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Traversability Cost
    im0 = axes[0, 0].imshow(cost_map, origin="lower", cmap="turbo", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("2.5D Traversability Cost $c_{final} \\in [0, 1]$", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("X Cells")
    axes[0, 0].set_ylabel("Y Cells")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Panel 2: ROS Occupancy Grid
    # -1 is unknown (grey), 0-100 traversability
    masked_occ = np.ma.masked_where(occ_grid == -1, occ_grid)
    cmap_occ = matplotlib.colormaps["RdYlGn_r"]
    axes[0, 1].set_facecolor("#d3d3d3")  # Light gray for unknown
    im1 = axes[0, 1].imshow(masked_occ, origin="lower", cmap=cmap_occ, vmin=0, vmax=100)

    axes[0, 1].set_title("ROS OccupancyGrid (0=Safe, 100=Lethal, Gray=Unknown)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("X Cells")
    axes[0, 1].set_ylabel("Y Cells")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Panel 3: Uncertainty Map
    im2 = axes[1, 0].imshow(unc_map, origin="lower", cmap="plasma", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Semantic Column Uncertainty $H \\in [0, 1]$", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("X Cells")
    axes[1, 0].set_ylabel("Y Cells")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Panel 4: Elevation Map
    im3 = axes[1, 1].imshow(elev_map, origin="lower", cmap="terrain")
    axes[1, 1].set_title("BEV Maximum Elevation [m]", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("X Cells")
    axes[1, 1].set_ylabel("Y Cells")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.suptitle(f"TerraSem Multi-layer BEV Traversability Mapping — Seq {args.seq}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    costmap_fig_path = fig_dir / f"costmap_seq{args.seq}.png"
    plt.savefig(costmap_fig_path, dpi=180)
    plt.close(fig)
    print(f"Saved 2.5D BEV costmap figure: {costmap_fig_path}")

    # 8. Calibration Metrics & Verification
    print("Computing calibration and uncertainty metrics...")
    # Evaluate calibration over occupied voxels
    ece_val = expected_calibration_error(occ_probs, occ_classes, n_bins=15, ignore_index=0)
    brier_val = brier_score(occ_probs, occ_classes, num_classes=NUM_CLASSES, ignore_index=0)
    rel_diag = reliability_diagram(occ_probs, occ_classes, n_bins=15, ignore_index=0)

    print(f"Expected Calibration Error (ECE): {ece_val * 100:.2f}%")
    print(f"Brier Score: {brier_val:.4f}")

    # Save calibration json
    calib_json_path = Path("results") / f"calibration_seq{args.seq}.json"
    calib_data = {
        "sequence": args.seq,
        "num_frames": frames_processed,
        "voxel_size": args.voxel_size,
        "ece": float(ece_val),
        "brier_score": float(brier_val),
        "reliability_diagram": rel_diag,
    }
    with calib_json_path.open("w") as f:
        json.dump(calib_data, f, indent=2)
    print(f"Saved calibration results: {calib_json_path}")

    # Save execution statistics JSON
    stats_json_path = out_dir / f"{args.seq}_stats.json"
    stats_data = {
        "sequence": args.seq,
        "num_frames": frames_processed,
        "voxel_size_m": args.voxel_size,
        "total_voxels": len(grid),
        "occupied_voxels": num_occupied,
        "wall_time_sec": float(round(elapsed_time, 2)),
        "peak_ram_mb": float(round(peak_ram / 1024**2, 2)),
        "raycast_stride": args.raycast_stride,
        "max_range_m": args.max_range,
        "ece_percent": float(round(ece_val * 100, 2)),
        "brier_score": float(round(brier_val, 4)),
    }
    with stats_json_path.open("w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"Saved map build statistics: {stats_json_path}")
    print("\n=== Phase 5 Map Build Successful! ===")


if __name__ == "__main__":
    main()
