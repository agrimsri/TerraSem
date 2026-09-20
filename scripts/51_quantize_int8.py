#!/usr/bin/env python3
"""51_quantize_int8.py — Static and Dynamic INT8 Quantization with ONNX Runtime.

Per BUILD.md Phase 8.2:
- Static INT8 quantization with a CalibrationDataReader over ~50-200 images.
- Dynamic INT8 quantization variant for comparison.
- Evaluates mIoU on validation set for FP32 vs Static INT8 vs Dynamic INT8.
- Records real mIoU delta and model file sizes.
- Outputs results/quantization.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from PIL import Image
from torchvision import transforms

from terrasem.datasets.ontology import NUM_CLASSES
from terrasem.metrics.segmentation import SegmentationMetrics


class ImageCalibrationDataReader(CalibrationDataReader):
    """Feeds preprocessed real images to ONNX Runtime static quantizer."""

    def __init__(self, image_paths: list[Path], input_name: str = "image", batch_size: int = 1) -> None:
        self.image_paths = image_paths
        self.input_name = input_name
        self.batch_size = batch_size
        self.enum_data = iter(self.image_paths)
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            img_path = next(self.enum_data)
        except StopIteration:
            return None

        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).numpy()
        return {self.input_name: tensor}

    def rewind(self) -> None:
        self.enum_data = iter(self.image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize SegFormer-B0 ONNX to INT8.")
    parser.add_argument("--onnx-model", type=str, default="checkpoints/segformer_b0.onnx", help="Input FP32 ONNX.")
    parser.add_argument("--static-out", type=str, default="checkpoints/segformer_b0_int8_static.onnx")
    parser.add_argument("--dynamic-out", type=str, default="checkpoints/segformer_b0_int8_dynamic.onnx")
    parser.add_argument("--data-dir", type=str, default="data/rellis3d", help="RELLIS-3D root.")
    parser.add_argument("--num-calibration", type=int, default=50, help="Calibration images count.")
    parser.add_argument("--num-eval", type=int, default=30, help="Validation images count for mIoU delta.")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory.")
    return parser.parse_args()


def evaluate_onnx_miou(
    model_path: Path,
    image_paths: list[Path],
    label_paths: list[Path],
    mapping_dict: dict[int, int],
) -> tuple[float, float]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preds = []
    targets = []
    latencies = []

    for img_p, lbl_p in zip(image_paths, label_paths, strict=False):
        img = Image.open(img_p).convert("RGB")
        inp = transform(img).unsqueeze(0).numpy()

        t0 = time.perf_counter()
        out = session.run([output_name], {input_name: inp})[0]
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        pred_mask = np.argmax(out[0], axis=0)  # (512, 512)

        # Load GT
        lbl_img = Image.open(lbl_p).resize((512, 512), Image.NEAREST)
        raw_lbl = np.array(lbl_img)
        remapped_lbl = np.zeros_like(raw_lbl, dtype=np.int64)
        for k, v in mapping_dict.items():
            remapped_lbl[raw_lbl == k] = v

        preds.append(pred_mask)
        targets.append(remapped_lbl)

    metrics = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=0)
    for p, g in zip(preds, targets, strict=False):
        metrics.update(p, g)
    miou_val = metrics.miou() if len(preds) > 0 else 0.0

    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    return float(round(miou_val * 100, 2)), float(round(avg_lat, 2))


def main() -> None:
    args = parse_args()
    fp32_path = Path(args.onnx_model)
    static_path = Path(args.static_out)
    dynamic_path = Path(args.dynamic_out)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    if not fp32_path.exists():
        raise FileNotFoundError(f"Base ONNX model not found: {fp32_path}. Run scripts/50_export_onnx.py first.")

    # Collect calibration images
    all_imgs = sorted(data_dir.glob("**/pylon_camera_node/*.jpg"))
    if not all_imgs:
        all_imgs = sorted(data_dir.glob("**/*.jpg"))

    calib_imgs = all_imgs[: args.num_calibration]
    print(f"Loaded {len(calib_imgs)} images for INT8 calibration.")

    # 1. Dynamic Quantization
    print(f"\nPerforming dynamic INT8 quantization -> {dynamic_path}...")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(dynamic_path),
        weight_type=QuantType.QInt8,
    )
    print(f"Dynamic INT8 exported. Size: {dynamic_path.stat().st_size / (1024*1024):.2f} MB")

    # 2. Static Quantization (QDQ)
    print(f"\nPerforming static INT8 quantization (QDQ) -> {static_path}...")
    calib_reader = ImageCalibrationDataReader(calib_imgs, input_name="image")
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(static_path),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        extra_options={"DisableShapeInference": True},
    )
    print(f"Static INT8 exported. Size: {static_path.stat().st_size / (1024*1024):.2f} MB")

    # 3. Validation Evaluation & mIoU Delta
    from terrasem.datasets.ontology import RELLIS3D_TO_TERRASEM

    val_imgs = all_imgs[-args.num_eval :]
    val_lbls = []
    matched_imgs = []
    for img_p in val_imgs:
        lbl_p = Path(str(img_p).replace("pylon_camera_node", "pylon_camera_node_label_id").replace(".jpg", ".png"))
        if lbl_p.exists():
            val_lbls.append(lbl_p)
            matched_imgs.append(img_p)

    print(f"\nEvaluating mIoU on {len(matched_imgs)} validation images...")
    fp32_miou, fp32_lat = evaluate_onnx_miou(fp32_path, matched_imgs, val_lbls, RELLIS3D_TO_TERRASEM)
    dyn_miou, dyn_lat = evaluate_onnx_miou(dynamic_path, matched_imgs, val_lbls, RELLIS3D_TO_TERRASEM)
    stat_miou, stat_lat = evaluate_onnx_miou(static_path, matched_imgs, val_lbls, RELLIS3D_TO_TERRASEM)

    fp32_size = fp32_path.stat().st_size / (1024 * 1024)
    dyn_size = dynamic_path.stat().st_size / (1024 * 1024)
    stat_size = static_path.stat().st_size / (1024 * 1024)

    results: dict[str, Any] = {
        "models": {
            "fp32": {
                "path": str(fp32_path),
                "size_mb": round(fp32_size, 2),
                "miou_pct": fp32_miou,
                "latency_ms": fp32_lat,
            },
            "int8_dynamic": {
                "path": str(dynamic_path),
                "size_mb": round(dyn_size, 2),
                "miou_pct": dyn_miou,
                "miou_delta": round(dyn_miou - fp32_miou, 2),
                "latency_ms": dyn_lat,
                "speedup_vs_fp32": round(fp32_lat / dyn_lat, 2) if dyn_lat > 0 else 1.0,
            },
            "int8_static": {
                "path": str(static_path),
                "size_mb": round(stat_size, 2),
                "miou_pct": stat_miou,
                "miou_delta": round(stat_miou - fp32_miou, 2),
                "latency_ms": stat_lat,
                "speedup_vs_fp32": round(fp32_lat / stat_lat, 2) if stat_lat > 0 else 1.0,
            },
        },
        "compression_ratio": round(fp32_size / stat_size, 2),
    }

    out_file = results_dir / "quantization.json"
    with out_file.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved quantization results to: {out_file}")
    print(
        f"Summary: FP32 ({fp32_size:.1f} MB, {fp32_miou:.2f}%) -> Static INT8 ({stat_size:.1f} MB, "
        f"{stat_miou:.2f}%, delta {results['models']['int8_static']['miou_delta']:+.2f}%)"
    )


if __name__ == "__main__":
    main()
