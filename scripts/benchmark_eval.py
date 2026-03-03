#!/usr/bin/env python3
"""Benchmark evaluation CLI for the SOPilot Perception Engine.

Usage:
    python scripts/benchmark_eval.py --anomaly
    python scripts/benchmark_eval.py --anomaly --n-normal 100 --n-anomaly 20
    python scripts/benchmark_eval.py --anomaly --threshold 3.0

Outputs JSON results to stdout.
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is importable regardless of where the script is invoked.
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_anomaly_benchmark(n_normal: int, n_anomaly: int, threshold: float) -> dict:
    """Run the anomaly detection synthetic benchmark and return result dict."""
    from sopilot.perception.benchmark import AnomalyBenchmarkEvaluator

    evaluator = AnomalyBenchmarkEvaluator()
    result = evaluator.run_synthetic_benchmark(
        n_normal=n_normal,
        n_anomaly=n_anomaly,
        threshold=threshold,
    )
    return result.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SOPilot Perception Engine Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_eval.py --anomaly
  python scripts/benchmark_eval.py --anomaly --n-normal 200 --n-anomaly 50
  python scripts/benchmark_eval.py --anomaly --threshold 3.0
""",
    )
    parser.add_argument(
        "--anomaly",
        action="store_true",
        help="Run synthetic anomaly detection benchmark",
    )
    parser.add_argument(
        "--n-normal",
        type=int,
        default=100,
        metavar="N",
        help="Number of normal frames to evaluate (default: 100)",
    )
    parser.add_argument(
        "--n-anomaly",
        type=int,
        default=20,
        metavar="N",
        help="Number of anomaly frames to evaluate (default: 20)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.5,
        metavar="SIGMA",
        help="Sigma threshold for anomaly detection (default: 2.5)",
    )

    args = parser.parse_args()

    if args.anomaly:
        result = run_anomaly_benchmark(args.n_normal, args.n_anomaly, args.threshold)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
        print(
            "\nNo benchmark type selected. Use --anomaly to run the anomaly benchmark.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
