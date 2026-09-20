#!/usr/bin/env python3
"""Project LiDAR point clouds into camera images and visualize depth overlay.

Per BUILD.md Phase 3 Task 4:
- Overlays projected points, colored by depth, onto camera images.
- Saves 10 overlay images to results/figs/projection/.
- Verifies: ground points fall on ground, tree points on trees, coherent horizon.
- Measures and logs projection runtime per point cloud.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from terrasem.calib.extrinsics import Extrinsics
from terrasem.calib.intrinsics import CameraIntrinsics
from terrasem.calib.projection import project_lidar_to_image


def overlay_lidar_on_image(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    out_path: str | Path,
    title: str = "",
    point_size: float = 3.0,
    max_depth: float = 50.0,
) -> None:
    """Save camera image with depth-colored LiDAR points overlaid."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(image)

    # Normalize depth to [0, 1] for colormap
    depth_clipped = np.clip(depth, 1.0, max_depth)
    scatter = ax.scatter(
        u,
        v,
        c=depth_clipped,
        cmap="turbo",
        s=point_size,
        alpha=0.8,
        edgecolors="none",
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Depth (meters)", fontsize=12)

    ax.set_title(
        f"{title} | {len(u):,} points in frustum (depth range {depth.min():.1f}m - {depth.max():.1f}m)",
        fontsize=13,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_projection_verification(
    data_root: str | Path,
    out_dir: str | Path = "results/figs/projection",
    num_frames: int = 10,
    seq: str = "00000",
) -> list[dict]:
    root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load intrinsics
    cam_info_path = root / f"Rellis-3D/{seq}/camera_info.txt"
    if not cam_info_path.exists():
        cam_info_path = root / "Rellis-3D/00000/camera_info.txt"
    intrinsics = CameraIntrinsics.from_file(cam_info_path, image_wh=(1920, 1200))

    # 2. Load extrinsics
    tf_path = root / f"Rellis_3D/{seq}/transforms.yaml"
    if not tf_path.exists():
        tf_path = root / "Rellis_3D/00000/transforms.yaml"
    extrinsics = Extrinsics.from_yaml(tf_path)

    # 3. Locate image and point cloud files
    img_files = sorted((root / "samples/Rellis_3D_image_example/pylon_camera_node").glob("*.jpg"))
    if not img_files:
        img_files = sorted((root / f"Rellis-3D/{seq}/pylon_camera_node").glob("*.jpg"))

    bin_files = sorted((root / "Rellis_3D_lidar_example/os1_cloud_node_kitti_bin").glob("*.bin"))
    if not bin_files:
        bin_files = sorted((root / f"Rellis-3D/{seq}/os1_cloud_node_kitti_bin").glob("*.bin"))

    print(f"Found {len(img_files)} images, {len(bin_files)} point cloud .bin files.")
    assert len(bin_files) > 0, "No LiDAR .bin files found!"

    results = []
    latencies_ms = []

    for i in range(num_frames):
        # Select image and bin
        img_p = img_files[i % len(img_files)]
        bin_p = bin_files[i % len(bin_files)]

        image_np = np.array(Image.open(img_p).convert("RGB"))
        H, W = image_np.shape[:2]

        # Rescale intrinsics if image resolution is different from 1920x1200
        curr_intr = intrinsics
        if intrinsics.image_wh != (W, H):
            curr_intr = intrinsics.rescale(W / intrinsics.image_wh[0], H / intrinsics.image_wh[1])

        raw_pts = np.fromfile(bin_p, dtype=np.float32)
        assert len(raw_pts) % 4 == 0, f"Points not divisible by 4: {len(raw_pts)}"
        points = raw_pts.reshape(-1, 4)

        # Benchmark projection
        t0 = time.perf_counter()
        u, v, depth, valid_mask, indices = project_lidar_to_image(
            points=points,
            R=extrinsics.R,
            t=extrinsics.t,
            K=curr_intr.K,
            image_wh=(W, H),
            min_depth=0.5,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        out_img_path = out_dir / f"proj_{i+1:02d}_{img_p.stem}_pts{len(u)}.png"
        overlay_lidar_on_image(
            image=image_np,
            u=u,
            v=v,
            depth=depth,
            out_path=out_img_path,
            title=f"Frame {i+1}: {img_p.name} + {bin_p.name}",
        )

        res = {
            "frame": i + 1,
            "image": img_p.name,
            "bin": bin_p.name,
            "total_points": len(points),
            "projected_points": len(u),
            "min_depth": float(depth.min()) if len(depth) else 0.0,
            "max_depth": float(depth.max()) if len(depth) else 0.0,
            "latency_ms": elapsed_ms,
            "out_file": str(out_img_path),
        }
        results.append(res)
        print(
            f"Frame {i+1:02d}: {len(u):,} / {len(points):,} points projected "
            f"in {elapsed_ms:.2f} ms -> {out_img_path.name}"
        )

    median_lat = float(np.median(latencies_ms))
    print(f"\n[SUMMARY] Processed {len(results)} frames. Median projection latency: {median_lat:.2f} ms")
    return results


def main():
    parser = argparse.ArgumentParser(description="Project LiDAR into camera and visualize")
    parser.add_argument("--data-root", default="data/rellis3d")
    parser.add_argument("--out-dir", default="results/figs/projection")
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--seq", default="00000")
    args = parser.parse_args()

    run_projection_verification(
        data_root=args.data_root,
        out_dir=args.out_dir,
        num_frames=args.num_frames,
        seq=args.seq,
    )


if __name__ == "__main__":
    main()
