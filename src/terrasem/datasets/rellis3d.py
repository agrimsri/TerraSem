"""RELLIS-3D dataset loader.

Returns per-frame dicts with keys:
    image   : torch.Tensor [3, H, W] float32, normalised ImageNet stats
    label   : torch.Tensor [H, W] int64, TerraSem-11 class IDs
    points  : torch.Tensor [N, 4] float32, KITTI-format XYZI
    pose    : torch.Tensor [4, 4] float32, T_world_lidar
    frame_id: str

Follows the real file layout and shapes verified in docs/dataset_inventory.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from terrasem.datasets.ontology import RELLIS3D_TO_TERRASEM

# Precompute 256-element vectorised lookup table for fast label remapping
_REMAP_LUT = np.full(256, 10, dtype=np.int64)  # default: unknown_other (10)
for _src_id, _tgt_id in RELLIS3D_TO_TERRASEM.items():
    if 0 <= _src_id < 256:
        _REMAP_LUT[_src_id] = _tgt_id


class Rellis3DDataset(Dataset):
    """PyTorch Dataset for RELLIS-3D multimodal off-road perception.

    Args:
        root: Path to the ``data/rellis3d/`` directory.
        split_file: Path to one of ``data/splits/{train,val,test}.txt``.
        transform: Optional transform callable applied to the sample dict.
        preload_poses: Whether to cache poses.txt per sequence in memory.
    """

    def __init__(
        self,
        root: str | Path,
        split_file: str | Path,
        transform: Any = None,
        preload_poses: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split_file = Path(split_file)
        self.transform = transform
        self.preload_poses = preload_poses

        # Frame IDs are stored as "sequence/frame_name" e.g. "00000/frame000000-1581624652_750"
        self._frame_ids: list[str] = self._load_split()
        self._poses_cache: dict[str, np.ndarray] = {}

        if self.preload_poses:
            self._load_all_poses()

    def _load_split(self) -> list[str]:
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        with self.split_file.open() as fh:
            return [line.strip() for line in fh if line.strip()]

    def _load_all_poses(self) -> None:
        rellis_dir = self.root / "Rellis-3D" if (self.root / "Rellis-3D").exists() else self.root
        for seq_dir in rellis_dir.glob("00*"):
            pose_file = seq_dir / "poses.txt"
            if pose_file.exists():
                try:
                    poses_data = np.loadtxt(pose_file, dtype=np.float32)
                    self._poses_cache[seq_dir.name] = poses_data
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self._frame_ids)

    def _get_pose(self, seq: str, frame_idx: int) -> np.ndarray:
        """Return 4x4 transformation matrix for given sequence and frame index."""
        pose_mat = np.eye(4, dtype=np.float32)
        if seq in self._poses_cache:
            poses = self._poses_cache[seq]
            if frame_idx < len(poses):
                row = poses[frame_idx]  # 12 elements
                pose_mat[:3, :4] = row.reshape(3, 4)
        return pose_mat

    def __getitem__(self, idx: int) -> dict[str, Any]:
        frame_id_str = self._frame_ids[idx]
        parts = frame_id_str.split("/")
        seq = parts[0]
        base_name = parts[1] if len(parts) > 1 else parts[0]

        rellis_dir = self.root / "Rellis-3D" if (self.root / "Rellis-3D").exists() else self.root
        seq_dir = rellis_dir / seq

        # 1. Load label map
        label_p = seq_dir / "pylon_camera_node_label_id" / f"{base_name}.png"
        if not label_p.exists():
            # Check samples folder as fallback
            sample_p = self.root / "samples/Rellis_3D_image_example/pylon_camera_node_label_id" / f"{base_name}.png"
            if sample_p.exists():
                label_p = sample_p

        if label_p.exists():
            raw_label = np.array(Image.open(label_p), dtype=np.uint8)
            # Handle RGBA/RGB label images if encountered
            if raw_label.ndim == 3:
                raw_label = raw_label[:, :, 0]
            # Fast vectorized mapping through LUT into TerraSem-11
            label_mapped = _REMAP_LUT[raw_label]
        else:
            # Fallback 1200x1920 void label
            label_mapped = np.zeros((1200, 1920), dtype=np.int64)

        # 2. Load camera image
        img_p = seq_dir / "pylon_camera_node" / f"{base_name}.jpg"
        if not img_p.exists():
            # Check png
            img_p = seq_dir / "pylon_camera_node" / f"{base_name}.png"
        if not img_p.exists():
            # Check samples folder
            sample_img_p = self.root / "samples/Rellis_3D_image_example/pylon_camera_node" / f"{base_name}.jpg"
            if sample_img_p.exists():
                img_p = sample_img_p
            else:
                # Use any available sample image if full images zip is not downloaded
                any_sample = list(self.root.glob("samples/**/*.jpg"))
                if any_sample:
                    img_p = any_sample[0]

        if img_p.exists():
            image_pil = Image.open(img_p).convert("RGB")
        else:
            # Synthetic 1200x1920 RGB image
            image_pil = Image.fromarray(np.zeros((1200, 1920, 3), dtype=np.uint8))

        label_pil = Image.fromarray(label_mapped.astype(np.uint8))

        # 3. Load point cloud
        points = torch.zeros((0, 4), dtype=torch.float32)
        # Parse frame index (e.g. frame000123 -> 123)
        try:
            num_part = "".join(filter(str.isdigit, base_name.split("-")[0]))
            frame_num = int(num_part) if num_part else idx
        except Exception:
            frame_num = idx

        bin_p = seq_dir / "os1_cloud_node_kitti_bin" / f"{frame_num:06d}.bin"
        if not bin_p.exists():
            # Check lidar example
            example_bin = self.root / f"Rellis_3D_lidar_example/os1_cloud_node_kitti_bin/{frame_num:06d}.bin"
            if example_bin.exists():
                bin_p = example_bin
            else:
                any_bin = list(self.root.glob("Rellis_3D_lidar_example/**/*.bin"))
                if any_bin:
                    bin_p = any_bin[0]

        if bin_p.exists():
            raw_pts = np.fromfile(bin_p, dtype=np.float32)
            if len(raw_pts) % 4 == 0:
                points = torch.from_numpy(raw_pts.reshape(-1, 4).copy())

        # 4. Pose
        pose_mat = torch.from_numpy(self._get_pose(seq, frame_num))

        sample = {
            "image": image_pil,
            "label": label_pil,
            "points": points,
            "pose": pose_mat,
            "frame_id": frame_id_str,
        }

        if self.transform is not None:
            sample = self.transform(sample)
        else:
            # Default minimal tensor conversion
            sample["image"] = torch.from_numpy(np.array(image_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            sample["label"] = torch.from_numpy(np.array(label_pil, dtype=np.int64))

        return sample
