"""
DTW Performance Benchmark

Compares CPU vs GPU DTW performance across different matrix sizes.

Usage:
    python benchmarks/benchmark_dtw.py

Output:
    - Console table with results
    - JSON file: benchmarks/results/dtw_benchmark_{timestamp}.json
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from datetime import datetime

import numpy as np


def generate_random_embeddings(n: int, d: int = 768) -> np.ndarray:
    """Generate random normalized embeddings."""
    x = np.random.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def benchmark_cpu_dtw(gold: np.ndarray, trainee: np.ndarray, runs: int = 3) -> dict:
    """Benchmark CPU DTW implementation."""
    from sopilot.step_engine import dtw_align

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = dtw_align(gold, trainee)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "mean_sec": np.mean(times),
        "std_sec": np.std(times),
        "min_sec": np.min(times),
        "max_sec": np.max(times),
        "path_length": len(result.path),
        "mean_cost": result.mean_cost,
    }


def benchmark_gpu_dtw(gold: np.ndarray, trainee: np.ndarray, runs: int = 3) -> dict:
    """Benchmark GPU DTW implementation."""
    try:
        from sopilot.dtw_gpu import _dtw_align_gpu

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            result = _dtw_align_gpu(gold, trainee)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return {
            "mean_sec": np.mean(times),
            "std_sec": np.std(times),
            "min_sec": np.min(times),
            "max_sec": np.max(times),
            "path_length": len(result.path),
            "mean_cost": result.mean_cost,
        }
    except Exception as e:
        return {
            "error": str(e),
            "available": False,
        }


def run_benchmark_suite():
    """Run full DTW benchmark suite."""
    print("=" * 80)
    print("DTW Performance Benchmark")
    print("=" * 80)

    # Test matrix sizes (m, n, embedding_dim)
    test_cases = [
        (100, 100, 768, "Small (100x100)"),
        (500, 500, 768, "Medium (500x500)"),
        (1000, 1000, 768, "Large (1000x1000)"),
        (2000, 2000, 768, "MAX (2000x2000)"),
    ]

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "test_cases": [],
    }

    for m, n, d, label in test_cases:
        print(f"\n{label} - Dimension: {d}")
        print("-" * 80)

        # Generate test data
        gold = generate_random_embeddings(m, d)
        trainee = generate_random_embeddings(n, d)

        # CPU benchmark
        print("  Running CPU DTW (3 runs)...")
        cpu_result = benchmark_cpu_dtw(gold, trainee, runs=3)
        print(f"    Mean: {cpu_result['mean_sec']:.3f}s ± {cpu_result['std_sec']:.3f}s")

        # GPU benchmark
        print("  Running GPU DTW (3 runs)...")
        gpu_result = benchmark_gpu_dtw(gold, trainee, runs=3)
        if "error" in gpu_result:
            print(f"    GPU not available: {gpu_result['error']}")
            speedup = None
        else:
            print(f"    Mean: {gpu_result['mean_sec']:.3f}s ± {gpu_result['std_sec']:.3f}s")
            speedup = cpu_result['mean_sec'] / gpu_result['mean_sec']
            print(f"    Speedup: {speedup:.1f}x")

        results["test_cases"].append({
            "label": label,
            "m": m,
            "n": n,
            "d": d,
            "cpu": cpu_result,
            "gpu": gpu_result,
            "speedup": speedup,
        })

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"dtw_benchmark_{timestamp}.json"

    with output_file.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)

    # Summary table
    print("\nSummary:")
    print(f"{'Size':<20} {'CPU (s)':<15} {'GPU (s)':<15} {'Speedup':<10}")
    print("-" * 60)
    for case in results["test_cases"]:
        cpu_time = case["cpu"].get("mean_sec", float("nan"))
        gpu_time = case["gpu"].get("mean_sec", "N/A")
        speedup = case["speedup"]

        cpu_str = f"{cpu_time:.3f}" if not np.isnan(cpu_time) else "N/A"
        gpu_str = f"{gpu_time:.3f}" if isinstance(gpu_time, float) else "N/A"
        speedup_str = f"{speedup:.1f}x" if speedup else "N/A"

        print(f"{case['label']:<20} {cpu_str:<15} {gpu_str:<15} {speedup_str:<10}")

    return results


if __name__ == "__main__":
    run_benchmark_suite()
