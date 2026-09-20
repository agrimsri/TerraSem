#!/usr/bin/env python3
"""52_benchmark_latency.py — Benchmark Edge Latency across PyTorch, ONNX FP32, and INT8.

Per BUILD.md Phase 8.3:
- Configurations benchmarked:
    1. PyTorch FP32 GPU (CUDA)
    2. PyTorch FP32 CPU
    3. ONNX FP32 CPU
    4. ONNX INT8 Dynamic CPU
    5. ONNX INT8 Static CPU
- 20 warm-up + 200 timed iterations; measures median, mean, and p95 latency.
- Records hardware profile: CPU processor, CUDA device name, thread count.
- Outputs results/latency.json and results/figs/latency.png bar chart.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch

from terrasem.datasets.ontology import NUM_CLASSES
from terrasem.models.segformer import SegFormerB0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark perception inference latency.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_segformer_b0.pt")
    parser.add_argument("--onnx-fp32", type=str, default="checkpoints/segformer_b0.onnx")
    parser.add_argument("--onnx-int8-static", type=str, default="checkpoints/segformer_b0_int8_static.onnx")
    parser.add_argument("--onnx-int8-dyn", type=str, default="checkpoints/segformer_b0_int8_dynamic.onnx")
    parser.add_argument("--warmup", type=int, default=20, help="Warm-up iterations.")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for JSON.")
    parser.add_argument("--fig-dir", type=str, default="results/figs", help="Directory for figures.")
    return parser.parse_args()


def benchmark_torch(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    model.eval()
    model.to(device)
    x = input_tensor.to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Timing
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    arr = np.array(latencies)
    return {
        "mean_ms": float(round(float(np.mean(arr)), 2)),
        "median_ms": float(round(float(np.median(arr)), 2)),
        "p95_ms": float(round(float(np.percentile(arr, 95)), 2)),
        "std_ms": float(round(float(np.std(arr)), 2)),
        "fps": float(round(1000.0 / float(np.median(arr)), 1)),
    }


def benchmark_onnx(
    onnx_path: Path,
    input_numpy: np.ndarray,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    feed = {input_name: input_numpy}

    # Warm-up
    for _ in range(warmup):
        _ = session.run([output_name], feed)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run([output_name], feed)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    arr = np.array(latencies)
    return {
        "mean_ms": float(round(float(np.mean(arr)), 2)),
        "median_ms": float(round(float(np.median(arr)), 2)),
        "p95_ms": float(round(float(np.percentile(arr, 95)), 2)),
        "std_ms": float(round(float(np.std(arr)), 2)),
        "fps": float(round(1000.0 / float(np.median(arr)), 1)),
    }


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    hw_info = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    print(f"Hardware Info: {hw_info}")

    dummy_torch = torch.randn(1, 3, 512, 512, dtype=torch.float32)
    dummy_np = dummy_torch.numpy()

    configs: dict[str, Any] = {}

    # 1. PyTorch GPU
    if torch.cuda.is_available():
        print("\nBenchmarking PyTorch FP32 GPU...")
        model_gpu = SegFormerB0(num_classes=NUM_CLASSES, pretrained=False)
        if Path(args.checkpoint).exists():
            st = torch.load(args.checkpoint, map_location="cpu")
            model_gpu.load_state_dict(st.get("model_state_dict", st))
        configs["PyTorch FP32 (GPU)"] = benchmark_torch(
            model_gpu, dummy_torch, torch.device("cuda"), args.warmup, args.iterations
        )
        print(f"  PyTorch GPU: {configs['PyTorch FP32 (GPU)']['median_ms']} ms ({configs['PyTorch FP32 (GPU)']['fps']} FPS)")

    # 2. PyTorch CPU
    print("\nBenchmarking PyTorch FP32 CPU...")
    model_cpu = SegFormerB0(num_classes=NUM_CLASSES, pretrained=False)
    if Path(args.checkpoint).exists():
        st = torch.load(args.checkpoint, map_location="cpu")
        model_cpu.load_state_dict(st.get("model_state_dict", st))
    configs["PyTorch FP32 (CPU)"] = benchmark_torch(
        model_cpu, dummy_torch, torch.device("cpu"), args.warmup, args.iterations
    )
    print(f"  PyTorch CPU: {configs['PyTorch FP32 (CPU)']['median_ms']} ms ({configs['PyTorch FP32 (CPU)']['fps']} FPS)")

    # 3. ONNX FP32 CPU
    if Path(args.onnx_fp32).exists():
        print(f"\nBenchmarking ONNX FP32 CPU ({args.onnx_fp32})...")
        configs["ONNX FP32 (CPU)"] = benchmark_onnx(
            Path(args.onnx_fp32), dummy_np, args.warmup, args.iterations
        )
        print(f"  ONNX FP32: {configs['ONNX FP32 (CPU)']['median_ms']} ms ({configs['ONNX FP32 (CPU)']['fps']} FPS)")

    # 4. ONNX INT8 Static CPU
    if Path(args.onnx_int8_static).exists():
        print(f"\nBenchmarking ONNX INT8 Static CPU ({args.onnx_int8_static})...")
        configs["ONNX INT8 Static (CPU)"] = benchmark_onnx(
            Path(args.onnx_int8_static), dummy_np, args.warmup, args.iterations
        )
        print(f"  ONNX INT8 Static: {configs['ONNX INT8 Static (CPU)']['median_ms']} ms ({configs['ONNX INT8 Static (CPU)']['fps']} FPS)")

    # 5. ONNX INT8 Dynamic CPU
    if Path(args.onnx_int8_dyn).exists():
        print(f"\nBenchmarking ONNX INT8 Dynamic CPU ({args.onnx_int8_dyn})...")
        configs["ONNX INT8 Dynamic (CPU)"] = benchmark_onnx(
            Path(args.onnx_int8_dyn), dummy_np, args.warmup, args.iterations
        )
        print(f"  ONNX INT8 Dynamic: {configs['ONNX INT8 Dynamic (CPU)']['median_ms']} ms ({configs['ONNX INT8 Dynamic (CPU)']['fps']} FPS)")

    out_data = {
        "hardware": hw_info,
        "input_resolution": "512x512",
        "iterations": args.iterations,
        "benchmarks": configs,
    }

    json_path = results_dir / "latency.json"
    with json_path.open("w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved latency benchmarks to: {json_path}")

    # Plot Bar Chart
    names = list(configs.keys())
    medians = [configs[k]["median_ms"] for k in names]
    p95s = [configs[k]["p95_ms"] for k in names]
    fps_vals = [configs[k]["fps"] for k in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#4daf4a", "#e41a1c", "#377eb8", "#984ea3", "#ff7f00"][: len(names)]

    # Subplot 1: Latency (ms)
    bars = ax1.bar(names, medians, color=colors, alpha=0.85, edgecolor="black")
    ax1.set_title("Inference Latency (Median & p95)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Latency (ms)", fontsize=11)
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    # Add value labels
    for bar, med, p95 in zip(bars, medians, p95s, strict=False):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, med + 1.0, f"{med:.1f}ms\n(p95: {p95:.1f})",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Subplot 2: Throughput (FPS)
    bars2 = ax2.bar(names, fps_vals, color=colors, alpha=0.85, edgecolor="black")
    ax2.set_title("Throughput (FPS)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Frames Per Second (Hz)", fontsize=11)
    ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar, fps in zip(bars2, fps_vals, strict=False):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, fps + 0.5, f"{fps:.1f} Hz",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.suptitle("SegFormer-B0 Edge Optimization Benchmark (512x512)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = fig_dir / "latency.png"
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)
    print(f"Saved latency bar chart to: {fig_path}")


if __name__ == "__main__":
    main()
