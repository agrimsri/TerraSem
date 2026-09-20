#!/usr/bin/env python3
"""Evaluation pipeline for semantic segmentation models.

Per BUILD.md Phase 4 Task 5:
- Runs inference on validation or test split
- Computes confusion matrix, mIoU, per-class IoU, pixel accuracy, frequency-weighted IoU
- Emits results/{model_name}_{split}.json
- Generates per-class IoU bar chart -> results/figs/seg_per_class_iou.png
- Generates 6 qualitative prediction figures (image | GT | prediction | error map) -> results/figs/seg_preds/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from terrasem.datasets.ontology import CLASS_NAMES, NUM_CLASSES
from terrasem.datasets.rellis3d import Rellis3DDataset
from terrasem.datasets.transforms import ValTransform
from terrasem.metrics.segmentation import SegmentationMetrics
from terrasem.models.registry import get_model
from terrasem.utils.viz import colorize_label


def save_per_class_iou_bar_chart(
    per_class_iou: dict[int, float] | np.ndarray,
    out_path: str | Path,
    title: str = "Per-Class IoU (TerraSem-11)",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    classes = []
    ious = []
    for cid in range(1, NUM_CLASSES):
        classes.append(CLASS_NAMES[cid])
        val = per_class_iou.get(cid, float("nan")) if isinstance(per_class_iou, dict) else per_class_iou[cid]
        ious.append(val * 100.0 if not np.isnan(val) else 0.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(classes, ious, color="#2b5c8f", edgecolor="black", alpha=0.85)

    ax.set_xlabel("IoU (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)

    for bar in bars:
        width = bar.get_width()
        if not np.isnan(width):
            ax.text(
                width + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}%",
                va="center",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_qualitative_figure(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: str | Path,
    title: str = "",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gt_colored = colorize_label(gt)
    pred_colored = colorize_label(pred)

    # Error map: white = correct or void, red = error
    error_map = np.ones((*gt.shape, 3), dtype=np.uint8) * 255
    mismatch = (gt > 0) & (gt != pred)
    error_map[mismatch] = [220, 40, 40]  # Red for errors
    correct = (gt > 0) & (gt == pred)
    error_map[correct] = [40, 180, 40]   # Green for correct

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input RGB", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(gt_colored)
    axes[1].set_title("Ground Truth (TerraSem-11)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(pred_colored)
    axes[2].set_title("Prediction", fontsize=12)
    axes[2].axis("off")

    axes[3].imshow(error_map)
    axes[3].set_title("Correct (Green) / Error (Red)", fontsize=12)
    axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_json: str | Path,
    save_preds_dir: str | Path | None = "results/figs/seg_preds",
    max_saved_preds: int = 6,
) -> dict:
    model.eval()
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)

    saved_count = 0
    if save_preds_dir is not None:
        save_preds_dir = Path(save_preds_dir)
        save_preds_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        out = model(images)
        logits = out["logits"]
        preds = logits.argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()

        metrics.update(preds, targets)

        # Save qualitative prediction figures
        if save_preds_dir is not None and saved_count < max_saved_preds:
            for b in range(len(images)):
                if saved_count >= max_saved_preds:
                    break
                # Denormalize image for display
                img_b = images[b].cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_disp = np.clip((img_b * std + mean) * 255, 0, 255).astype(np.uint8)

                gt_b = targets[b]
                pred_b = preds[b]
                frame_id = batch["frame_id"][b].replace("/", "_")

                out_fig_p = save_preds_dir / f"pred_{saved_count+1:02d}_{frame_id}.png"
                save_qualitative_figure(img_disp, gt_b, pred_b, out_fig_p, title=f"Sample {saved_count+1}: {batch['frame_id'][b]}")
                saved_count += 1

    # Extract all metrics
    miou = float(metrics.miou())
    pixel_acc = float(metrics.pixel_accuracy())
    fw_iou = float(metrics.frequency_weighted_iou())
    per_class = metrics.per_class_iou()

    per_class_dict = {
        CLASS_NAMES[i]: float(per_class[i]) if (i in per_class and not np.isnan(per_class[i])) else None
        for i in range(NUM_CLASSES)
    }

    results = {
        "mIoU": miou,
        "pixel_accuracy": pixel_acc,
        "frequency_weighted_IoU": fw_iou,
        "per_class_IoU": per_class_dict,
        "confusion_matrix": metrics.confusion_matrix.tolist(),
    }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nEvaluation Results saved to {out_json}:")
    print(f"  mIoU: {miou*100:.2f}%")
    print(f"  Pixel Accuracy: {pixel_acc*100:.2f}%")
    print(f"  Frequency-weighted IoU: {fw_iou*100:.2f}%")

    # Generate bar chart
    chart_path = Path("results/figs/seg_per_class_iou.png")
    save_per_class_iou_bar_chart(per_class, chart_path, title=f"Per-Class IoU (mIoU: {miou*100:.1f}%)")
    print(f"  Saved per-class IoU chart to {chart_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation model")
    parser.add_argument("--model-name", default="segformer_b0")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--data-root", default="data/rellis3d")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load dataset
    split_file = Path(args.splits_dir) / f"{args.split}.txt"
    val_transform = ValTransform(target_size=(512, 512))
    dataset = Rellis3DDataset(args.data_root, split_file, transform=val_transform)
    if args.max_samples and args.max_samples < len(dataset):
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(args.max_samples)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load model
    model = get_model(args.model_name, num_classes=NUM_CLASSES, pretrained=True)
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")

    model.to(device)

    out_json = args.out_json or f"results/{args.model_name}_{args.split}.json"

    run_evaluation(
        model=model,
        loader=loader,
        device=device,
        out_json=out_json,
        save_preds_dir="results/figs/seg_preds",
        max_saved_preds=6,
    )


if __name__ == "__main__":
    main()
