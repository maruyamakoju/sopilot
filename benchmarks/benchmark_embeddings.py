"""
Video Embedding Generation Benchmark

Measures throughput and latency of different embedders with various batch sizes.

Usage:
    python benchmarks/benchmark_embeddings.py

Tests:
    - Heuristic embedder (CPU, always available)
    - V-JEPA2 embedder (GPU if available)
    - Different batch sizes (1, 2, 4, 8, 16, 32)
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from datetime import datetime

import cv2
import numpy as np


class DummyClipWindow:
    """Mock ClipWindow for benchmark."""
    def __init__(self, frames: np.ndarray):
        self.frames = frames


def generate_dummy_clips(n: int, frames_per_clip: int = 64, h: int = 256, w: int = 256) -> list:
    """Generate random video clip data."""
    clips = []
    for _ in range(n):
        frames = np.random.randint(0, 256, size=(frames_per_clip, h, w, 3), dtype=np.uint8)
        clips.append(DummyClipWindow(frames))
    return clips


def benchmark_heuristic_embedder(num_clips: int, runs: int = 3) -> dict:
    """Benchmark heuristic embedder."""
    from sopilot.embeddings import HeuristicClipEmbedder

    embedder = HeuristicClipEmbedder()
    clips = generate_dummy_clips(num_clips)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        embeddings = embedder.embed_clips(clips)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_time = np.mean(times)
    throughput = num_clips / mean_time

    return {
        "embedder": "heuristic-v1",
        "num_clips": num_clips,
        "mean_sec": mean_time,
        "std_sec": np.std(times),
        "throughput_clips_per_sec": throughput,
        "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
    }


def benchmark_vjepa2_embedder(
    num_clips: int,
    batch_size: int,
    device: str = "auto",
    runs: int = 3,
) -> dict:
    """Benchmark V-JEPA2 embedder."""
    try:
        from sopilot.embeddings import VJepa2Embedder

        embedder = VJepa2Embedder(
            repo="facebookresearch/vjepa2",
            source="hub",
            local_repo="",
            local_checkpoint="",
            variant="vjepa2_vit_large",
            pretrained=True,
            device=device,
            num_frames=64,
            image_size=256,
            batch_size=batch_size,
        )

        clips = generate_dummy_clips(num_clips)

        # Warmup (first run loads model)
        print(f"    Warmup (loading model)...")
        _ = embedder.embed_clips(clips[:min(2, num_clips)])

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            embeddings = embedder.embed_clips(clips)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        throughput = num_clips / mean_time

        return {
            "embedder": "vjepa2:vit_large:pt",
            "num_clips": num_clips,
            "batch_size": batch_size,
            "device": str(embedder._device) if embedder._device else "unknown",
            "mean_sec": mean_time,
            "std_sec": np.std(times),
            "throughput_clips_per_sec": throughput,
            "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
            "compiled": getattr(embedder, "_compile_enabled", False),
        }
    except Exception as e:
        return {
            "embedder": "vjepa2",
            "error": str(e),
            "available": False,
        }


def run_benchmark_suite():
    """Run full embedding benchmark suite."""
    print("=" * 80)
    print("Video Embedding Generation Benchmark")
    print("=" * 80)

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "test_cases": [],
    }

    # Test 1: Heuristic embedder baseline
    print("\n[1] Heuristic Embedder (CPU)")
    print("-" * 80)
    for num_clips in [10, 50, 100]:
        print(f"  Testing {num_clips} clips...")
        result = benchmark_heuristic_embedder(num_clips, runs=3)
        print(f"    Throughput: {result['throughput_clips_per_sec']:.1f} clips/sec")
        results["test_cases"].append(result)

    # Test 2: V-JEPA2 with different batch sizes
    print("\n[2] V-JEPA2 Embedder (GPU if available)")
    print("-" * 80)

    num_clips = 100  # Fixed clip count, vary batch size
    for batch_size in [2, 4, 8, 16, 32]:
        print(f"  Testing batch_size={batch_size}...")
        result = benchmark_vjepa2_embedder(num_clips, batch_size, device="auto", runs=3)
        if "error" in result:
            print(f"    Error: {result['error']}")
        else:
            print(f"    Throughput: {result['throughput_clips_per_sec']:.1f} clips/sec")
            print(f"    Device: {result['device']}, Compiled: {result.get('compiled', False)}")
        results["test_cases"].append(result)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"embedding_benchmark_{timestamp}.json"

    with output_file.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)

    # Summary table
    print("\nSummary:")
    print(f"{'Embedder':<25} {'Clips':<10} {'Batch':<10} {'Throughput (clips/s)':<20}")
    print("-" * 65)
    for case in results["test_cases"]:
        if "error" in case:
            continue
        embedder = case.get("embedder", "unknown")
        num_clips = case.get("num_clips", 0)
        batch_size = case.get("batch_size", "-")
        throughput = case.get("throughput_clips_per_sec", 0)
        print(f"{embedder:<25} {num_clips:<10} {batch_size!s:<10} {throughput:<20.1f}")

    return results


if __name__ == "__main__":
    run_benchmark_suite()
