#!/usr/bin/env python3
"""Benchmark PyTorch (ultralytics) vs ONNX inference latency.

Usage::

    python scripts/benchmark_onnx.py
    python scripts/benchmark_onnx.py --frames 100 --model-dir data/models

Prints p50 / p95 / p99 latency in milliseconds for both backends.
"""

import argparse
import time
from pathlib import Path

import numpy as np


def _synthetic_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Generate a random BGR frame for benchmarking."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def benchmark_onnx(model_path: str, n_frames: int) -> list[float]:
    from sopilot.perception.onnx_detector import OnnxYOLOWorldDetector

    detector = OnnxYOLOWorldDetector(model_path)
    frames = [_synthetic_frame() for _ in range(n_frames)]
    latencies = []
    for frame in frames:
        t0 = time.perf_counter()
        detector.detect(frame)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def benchmark_ultralytics(n_frames: int) -> list[float]:
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError:
        print("ultralytics not installed — skipping PyTorch benchmark")
        return []

    model = YOLO("yolov8s-worldv2.pt")
    frames = [_synthetic_frame() for _ in range(n_frames)]
    latencies = []
    for frame in frames:
        t0 = time.perf_counter()
        model.predict(frame, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def _percentile(data: list[float], p: float) -> float:
    arr = np.array(data)
    return float(np.percentile(arr, p))


def print_stats(label: str, latencies: list[float]) -> None:
    if not latencies:
        print(f"{label}: no data")
        return
    print(
        f"{label:30s}  "
        f"p50={_percentile(latencies, 50):7.1f}ms  "
        f"p95={_percentile(latencies, 95):7.1f}ms  "
        f"p99={_percentile(latencies, 99):7.1f}ms  "
        f"(n={len(latencies)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--model-dir", default="data/models")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    onnx_path = model_dir / "yolov8s-worldv2.onnx"

    print(f"Benchmarking {args.frames} frames ...")

    if onnx_path.exists():
        onnx_latencies = benchmark_onnx(str(onnx_path), args.frames)
        print_stats("ONNX (onnxruntime)", onnx_latencies)
    else:
        print(f"ONNX model not found: {onnx_path} — run scripts/export_onnx.py first")

    pt_latencies = benchmark_ultralytics(args.frames)
    print_stats("PyTorch (ultralytics)", pt_latencies)


if __name__ == "__main__":
    main()
