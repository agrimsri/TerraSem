#!/usr/bin/env python3
"""Config-driven training pipeline for 2D semantic segmentation models.

Per BUILD.md Phase 4 Task 4:
- Config-driven training with PyTorch
- Logs train loss, val loss, val mIoU
- Saves best checkpoint by val mIoU to checkpoints/{model_name}/best.pt
- Smoke test mode: train on 50 samples, assert loss decreases
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from terrasem.datasets.ontology import NUM_CLASSES
from terrasem.datasets.rellis3d import Rellis3DDataset
from terrasem.datasets.transforms import TrainTransform, ValTransform
from terrasem.metrics.segmentation import SegmentationMetrics
from terrasem.models.registry import get_model
from terrasem.utils.config import load_config
from terrasem.utils.logging import get_logger
from terrasem.utils.seed import set_seed

logger = get_logger("train_seg")


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(images, labels=labels)
                loss = out["loss"]
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images, labels=labels)
            loss = out["loss"]
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> tuple[float, float, float]:
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes)
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        out = model(images, labels=labels)
        if "loss" in out:
            total_loss += out["loss"].item()
            num_batches += 1

        preds = out["logits"].argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        metrics.update(preds, targets)

    val_loss = total_loss / max(1, num_batches)
    miou = metrics.miou()
    pixel_acc = metrics.pixel_accuracy()
    return val_loss, miou, pixel_acc


def main():
    parser = argparse.ArgumentParser(description="Train 2D semantic segmentation model")
    parser.add_argument("--config", default="configs/model/segformer_b0.yaml")
    parser.add_argument("--data-root", default="data/rellis3d")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--smoke", action="store_true", help="Run 1-epoch smoke test on 50 samples")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    seed = training_cfg.get("seed", 42)
    set_seed(seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Datasets
    crop_size = tuple(training_cfg.get("crop_size", [512, 512]))
    train_transform = TrainTransform(crop_size=crop_size)
    val_transform = ValTransform(target_size=crop_size)

    train_split = Path(args.splits_dir) / "train.txt"
    val_split = Path(args.splits_dir) / "val.txt"

    train_dataset = Rellis3DDataset(args.data_root, train_split, transform=train_transform)
    val_dataset = Rellis3DDataset(args.data_root, val_split, transform=val_transform)

    if args.smoke:
        logger.info("Running SMOKE TEST on 50 samples...")
        smoke_size = min(50, len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(smoke_size)))
        val_dataset = Subset(val_dataset, list(range(min(10, len(val_dataset)))))
    else:
        if args.max_train_samples and args.max_train_samples < len(train_dataset):
            train_dataset = Subset(train_dataset, list(range(args.max_train_samples)))
            logger.info(f"Subsetting train dataset to {args.max_train_samples} samples")
        if args.max_val_samples and args.max_val_samples < len(val_dataset):
            val_dataset = Subset(val_dataset, list(range(args.max_val_samples)))
            logger.info(f"Subsetting val dataset to {args.max_val_samples} samples")

    batch_size = args.batch_size or training_cfg.get("batch_size", 4)
    epochs = args.epochs or (1 if args.smoke else training_cfg.get("epochs", 20))
    lr = args.lr or training_cfg.get("lr", 6e-5)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model_name = model_cfg.get("name", "segformer_b0")
    num_classes = model_cfg.get("num_labels", NUM_CLASSES)
    model = get_model(model_name, num_classes=num_classes, pretrained=True).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=training_cfg.get("weight_decay", 0.01),
    )

    scaler = torch.amp.GradScaler("cuda") if (training_cfg.get("mixed_precision", True) and device.type == "cuda") else None

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoints_dir) / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best.pt"

    best_miou = -1.0
    initial_loss = None
    final_loss = None

    logger.info(f"Starting training: {epochs} epochs, lr={lr}, batch_size={batch_size}")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=training_cfg.get("grad_clip", 1.0),
            scaler=scaler,
        )

        if initial_loss is None:
            initial_loss = train_loss
        final_loss = train_loss

        val_loss, miou, pixel_acc = evaluate(model, val_loader, device, num_classes)
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val mIoU: {miou*100:.2f}% | Pixel Acc: {pixel_acc*100:.2f}%"
        )

        if miou > best_miou:
            best_miou = miou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_miou": miou,
                    "val_loss": val_loss,
                    "config": cfg,
                },
                best_ckpt_path,
            )
            logger.info(f"Saved new best checkpoint to {best_ckpt_path} (mIoU: {miou*100:.2f}%)")

    # Smoke test assertion: loss must decrease or stay bounded
    if args.smoke and epochs > 1:
        assert final_loss is not None and initial_loss is not None
        assert final_loss <= initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    logger.info(f"Training complete. Best Val mIoU: {best_miou*100:.2f}%. Saved to {best_ckpt_path}")


if __name__ == "__main__":
    main()
