#!/usr/bin/env python3
"""VIGIL-RAG Benchmark Evaluation CLI.

Runs retrieval benchmarks comparing visual-only vs hybrid (visual + audio text)
search modes using synthetic controlled embeddings.

Usage:
    # Default: full benchmark, hybrid alpha=0.7
    python scripts/evaluate_vigil_benchmark.py

    # Smoke test (fast, for CI)
    python scripts/evaluate_vigil_benchmark.py --benchmark benchmarks/smoke_benchmark.jsonl

    # Alpha sweep
    python scripts/evaluate_vigil_benchmark.py --alpha-sweep 0.0,0.3,0.5,0.7,1.0

    # Custom top-k and signal strength
    python scripts/evaluate_vigil_benchmark.py --top-k 10 --signal-strength 0.8

    # Save results to JSON
    python scripts/evaluate_vigil_benchmark.py --output results/benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from sopilot.evaluation.vigil_benchmark import (  # noqa: E402
    VIGILBenchmarkRunner,
    format_report,
    report_to_dict,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VIGIL-RAG Benchmark: visual-only vs hybrid retrieval comparison",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(project_root / "benchmarks" / "vigil_benchmark_v1.jsonl"),
        help="Path to benchmark JSONL file (default: benchmarks/vigil_benchmark_v1.jsonl)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Hybrid search alpha weight for audio text score (default: 0.7)",
    )
    parser.add_argument(
        "--alpha-sweep",
        type=str,
        default=None,
        help="Comma-separated alpha values to sweep (e.g., 0.0,0.3,0.5,0.7,1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results per query (default: 10)",
    )
    parser.add_argument(
        "--signal-strength",
        type=float,
        default=0.85,
        help="Cosine similarity for relevant clips (0-1, default: 0.85)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save JSON results to this path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Determine alpha(s)
    if args.alpha_sweep:
        alphas = [float(a.strip()) for a in args.alpha_sweep.split(",")]
    else:
        alphas = [args.alpha]

    # Load benchmark
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(f"ERROR: Benchmark file not found: {benchmark_path}", file=sys.stderr)
        return 1

    runner = VIGILBenchmarkRunner(
        seed=args.seed,
        signal_strength=args.signal_strength,
    )

    queries = runner.load_benchmark(benchmark_path)
    if not queries:
        print("ERROR: No queries loaded from benchmark file", file=sys.stderr)
        return 1

    # Run evaluation
    report = runner.run(queries, alphas=alphas, top_k=args.top_k)
    report.benchmark_file = str(benchmark_path)

    # Print report
    print(format_report(report))

    # Save JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_to_dict(report), f, indent=2)
        print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
