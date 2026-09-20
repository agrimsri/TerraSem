#!/usr/bin/env python3
"""41_eval_label_efficiency.py — Label Efficiency Benchmark.

Per BUILD.md Phase 7.2:
- Subsample training set at 1%, 5%, 10%, 25%, 50%, 100%.
- Compare two architectures:
    1. Fully fine-tuned SegFormer-B0
    2. Frozen DINOv2 + trainable conv head
- Evaluates on validation set and records mIoU vs label fraction.
- Outputs results/label_efficiency.json and results/figs/label_efficiency.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from terrasem.datasets.ontology import IGNORE_INDEX, NUM_CLASSES
from terrasem.datasets.rellis3d import Rellis3DDataset
from terrasem.datasets.transforms import TrainTransform, ValTransform
from terrasem.metrics.segmentation import SegmentationMetrics
from terrasem.models.registry import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label efficiency benchmark.")
    parser.add_argument("--data-dir", type=str, default="data/rellis3d", help="Path to RELLIS-3D root.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--steps-per-fraction", type=int, default=15, help="Training steps per budget.")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory.")
    parser.add_argument("--fig-dir", type=str, default="results/figs", help="Figures directory.")
    return parser.parse_args()


def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Compute validation mIoU."""
    model.eval()
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            out = model(images)
            logits = out["logits"] if isinstance(out, dict) else out
            preds = torch.argmax(logits, dim=1)
            metrics.update(preds.cpu().numpy(), labels.cpu().numpy())
    return float(metrics.miou())


def train_steps(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    steps: int,
    device: torch.device,
) -> None:
    """Train model for a fixed number of steps."""
    model.train()
    step = 0
    while step < steps:
        for batch in train_loader:
            if step >= steps:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            out = model(images, labels=labels)
            loss = out["loss"] if isinstance(out, dict) and "loss" in out else F.cross_entropy(
                out["logits"] if isinstance(out, dict) else out, labels, ignore_index=IGNORE_INDEX
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running label efficiency evaluation on: {device}")

    out_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = Rellis3DDataset(
        root=args.data_dir,
        split_file="data/splits/train.txt",
        transform=TrainTransform(crop_size=(512, 512)),
    )
    val_dataset = Rellis3DDataset(
        root=args.data_dir,
        split_file="data/splits/val.txt",
        transform=ValTransform(target_size=(512, 512)),
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    total_train = len(train_dataset)

    results: dict[str, list[float]] = {
        "fractions": fractions,
        "sample_counts": [],
        "segformer_b0": [],
        "dinov2_linear": [],
    }

    rng = np.random.RandomState(42)

    for frac in fractions:
        n_samples = max(2, int(round(total_train * frac)))
        results["sample_counts"].append(n_samples)
        indices = rng.choice(total_train, size=n_samples, replace=False)
        subset = Subset(train_dataset, indices)
        train_loader = DataLoader(subset, batch_size=min(args.batch_size, n_samples), shuffle=True)

        print(f"\n--- Fraction {int(frac*100)}% ({n_samples} samples) ---")

        # 1. Train SegFormer-B0
        print("Training SegFormer-B0...")
        segformer = get_model("segformer_b0", num_classes=NUM_CLASSES, pretrained=True).to(device)
        opt_seg = torch.optim.AdamW(segformer.parameters(), lr=6e-5, weight_decay=0.01)
        train_steps(segformer, train_loader, opt_seg, args.steps_per_fraction, device)
        miou_seg = evaluate(segformer, val_loader, device)
        print(f"SegFormer-B0 mIoU @ {int(frac*100)}%: {miou_seg * 100:.2f}%")
        results["segformer_b0"].append(float(round(miou_seg * 100, 2)))
        del segformer, opt_seg
        torch.cuda.empty_cache()

        # 2. Train DINOv2-Linear Head (frozen backbone)
        print("Training DINOv2-Linear (Frozen Backbone)...")
        dinov2 = get_model("dinov2_linear", num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True).to(device)
        opt_dino = torch.optim.AdamW(filter(lambda p: p.requires_grad, dinov2.parameters()), lr=3e-4, weight_decay=0.01)
        train_steps(dinov2, train_loader, opt_dino, args.steps_per_fraction, device)
        miou_dino = evaluate(dinov2, val_loader, device)
        print(f"DINOv2-Linear mIoU @ {int(frac*100)}%: {miou_dino * 100:.2f}%")
        results["dinov2_linear"].append(float(round(miou_dino * 100, 2)))
        del dinov2, opt_dino
        torch.cuda.empty_cache()

    # Save JSON
    json_path = out_dir / "label_efficiency.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved label efficiency results to: {json_path}")

    # Plot Comparison Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    pct_labels = [f * 100 for f in fractions]
    ax.plot(pct_labels, results["segformer_b0"], "o-", color="#2b5c8f", linewidth=2.2, label="SegFormer-B0 (End-to-End)")
    ax.plot(pct_labels, results["dinov2_linear"], "s--", color="#d95f02", linewidth=2.2, label="DINOv2 ViT-S (Linear Probe)")

    ax.set_xscale("log")
    ax.set_xticks(pct_labels)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Label Fraction (%) [Log Scale]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation mIoU (%)", fontsize=12, fontweight="bold")
    ax.set_title("Label Efficiency: Foundation Backbone vs End-to-End Fine-Tuning", fontsize=13, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=11, loc="lower right")

    plot_path = fig_dir / "label_efficiency.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"Saved label efficiency plot to: {plot_path}")


if __name__ == "__main__":
    main()
