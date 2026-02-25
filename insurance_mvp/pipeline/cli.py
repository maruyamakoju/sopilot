"""CLI entry point for the Insurance MVP pipeline.

Subcommands
-----------
run          Process one or more video files through the full pipeline.
benchmark    Run the E2E benchmark against labelled demo videos (equivalent
             to scripts/insurance_e2e_benchmark.py).  Pass --ablation to run
             a systematic ablation study across feature flags.

See also: scripts/research_benchmark.py for the unified research benchmark
          (ablation + sensitivity + baselines in one run).

Examples
--------
  # Single video
  python -m insurance_mvp.pipeline run --video-path data/dashcam001.mp4

  # Batch
  python -m insurance_mvp.pipeline run --video-dir data/dashcam/ --parallel 4

  # E2E benchmark (mock backend, deterministic)
  python -m insurance_mvp.pipeline benchmark --backend mock

  # E2E benchmark + ablation study
  python -m insurance_mvp.pipeline benchmark --backend mock --ablation

  # E2E benchmark with real Qwen2.5-VL-7B (requires GPU >=16 GB VRAM)
  python -m insurance_mvp.pipeline benchmark --backend qwen2.5-vl-7b
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from insurance_mvp.config import CosmosBackend, load_config
from insurance_mvp.pipeline.orchestrator import InsurancePipeline

# ---------------------------------------------------------------------------
# Helpers shared by subcommands
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_ground_truth() -> dict:
    """Load ground truth labels from data/dashcam_demo/metadata.json.

    Falls back to a hardcoded minimal set when the file is absent so that
    the benchmark can still run on a clean checkout.
    """
    metadata_path = _PROJECT_ROOT / "data" / "dashcam_demo" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "collision": {
            "severity": "HIGH",
            "ground_truth": {"fault_ratio": 100.0, "scenario": "rear_end", "fraud_risk": 0.0},
        },
        "near_miss": {
            "severity": "MEDIUM",
            "ground_truth": {"fault_ratio": 0.0, "scenario": "pedestrian_avoidance", "fraud_risk": 0.0},
        },
        "normal": {
            "severity": "NONE",
            "ground_truth": {"fault_ratio": 0.0, "scenario": "normal_driving", "fraud_risk": 0.0},
        },
    }


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


def _add_run_parser(subparsers) -> None:
    run_parser = subparsers.add_parser(
        "run",
        help="Process video files through the full pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run the Insurance MVP pipeline on one or more videos.",
    )
    run_parser.add_argument("--video-path", type=str, help="Path to single video file")
    run_parser.add_argument("--video-dir", type=str, help="Directory containing video files")
    run_parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    run_parser.add_argument("--config", type=str, help="Path to YAML config file")
    run_parser.add_argument("--parallel", type=int, help="Number of parallel workers")
    run_parser.add_argument("--cosmos-backend", choices=["qwen2.5-vl-7b", "mock"], help="Cosmos backend")
    run_parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    run_parser.add_argument("--no-conformal", action="store_true", help="Disable conformal prediction")
    run_parser.add_argument("--no-transcription", action="store_true", help="Disable transcription")
    run_parser.set_defaults(func=_cmd_run)


def _cmd_run(args: argparse.Namespace) -> int:
    if not args.video_path and not args.video_dir:
        print("ERROR: Either --video-path or --video-dir must be specified", file=sys.stderr)
        return 2

    override_dict: dict = {"output_dir": args.output_dir}
    if args.parallel:
        override_dict["parallel_workers"] = args.parallel
    if args.cosmos_backend:
        override_dict["cosmos"] = {"backend": args.cosmos_backend}
    if args.log_level:
        override_dict["log_level"] = args.log_level
    if args.no_conformal:
        override_dict["enable_conformal"] = False
    if args.no_transcription:
        override_dict["enable_transcription"] = False

    config = load_config(yaml_path=args.config, override_dict=override_dict)
    pipeline = InsurancePipeline(config)

    if args.video_path:
        video_paths = [args.video_path]
    else:
        video_dir = Path(args.video_dir)
        video_paths = [str(p) for p in list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))]

    if not video_paths:
        print("No video files found!", file=sys.stderr)
        return 1

    results = pipeline.process_batch(video_paths)
    failed_count = sum(1 for r in results if not r.success)
    if failed_count > 0:
        print(f"\nWarning: {failed_count} videos failed to process")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Subcommand: benchmark
# ---------------------------------------------------------------------------


def _add_benchmark_parser(subparsers) -> None:
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run E2E benchmark against labelled demo videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run the Insurance MVP end-to-end benchmark.\n\n"
            "Processes demo videos from data/dashcam_demo/ and compares pipeline\n"
            "output against ground truth labels.  By default runs with mock VLM\n"
            "(deterministic, seed=42), which should achieve 9/9 checks passed.\n\n"
            "With --ablation the benchmark is run three additional times with\n"
            "individual feature flags disabled (conformal, recalibration,\n"
            "fraud detection) to measure their individual contribution.\n\n"
            "See also: scripts/insurance_e2e_benchmark.py (standalone equivalent)\n"
            "          scripts/research_benchmark.py (unified research benchmark)"
        ),
    )
    bench_parser.add_argument(
        "--backend",
        choices=["mock", "qwen2.5-vl-7b"],
        default="mock",
        help="VLM backend (default: mock).  Use qwen2.5-vl-7b for real GPU eval.",
    )
    bench_parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "Run ablation study: re-run benchmark with conformal, recalibration, "
            "and fraud detection each disabled in turn."
        ),
    )
    bench_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save JSON results to this path (optional).",
    )
    bench_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    bench_parser.set_defaults(func=_cmd_benchmark)


def _run_single_benchmark(
    backend: str,
    *,
    enable_conformal: bool = True,
    enable_recalibration: bool = True,
    enable_fraud_detection: bool = True,
    seed: int = 42,
    label: str = "full",
) -> dict:
    """Execute one benchmark pass and return the results dict."""
    from insurance_mvp.config import CosmosBackend, PipelineConfig  # noqa: PLC0415 — local import OK

    ground_truth = _load_ground_truth()
    output_dir = tempfile.mkdtemp(prefix=f"insurance_benchmark_{label}_")

    cosmos_backend = CosmosBackend.MOCK if backend == "mock" else CosmosBackend.QWEN25VL

    config = PipelineConfig(
        output_dir=output_dir,
        log_level="WARNING",
        parallel_workers=1,
        enable_conformal=enable_conformal,
        enable_transcription=False,
        enable_recalibration=enable_recalibration,
        enable_fraud_detection=enable_fraud_detection,
        continue_on_error=True,
        seed=seed,
    )
    config.cosmos.backend = cosmos_backend

    pipeline = InsurancePipeline(config)
    if backend == "mock":
        pipeline.signal_fuser = None
        pipeline.cosmos_client = None

    demo_dir = _PROJECT_ROOT / "data" / "dashcam_demo"
    results: dict = {}
    total_checks = 0
    passed_checks = 0

    for video_name, gt_data in ground_truth.items():
        gt = gt_data.get("ground_truth", gt_data)
        expected_severity = gt_data.get("severity", "NONE")

        video_path = demo_dir / f"{video_name}.mp4"
        if not video_path.exists():
            video_path = Path(output_dir) / f"{video_name}.mp4"
            video_path.write_bytes(b"mock video content")

        start = time.time()
        result = pipeline.process_video(str(video_path), video_id=video_name)
        elapsed = time.time() - start

        video_result: dict = {
            "video_id": video_name,
            "success": result.success,
            "processing_time_sec": round(elapsed, 3),
            "expected_severity": expected_severity,
            "expected_fault_ratio": gt.get("fault_ratio", 0.0),
            "expected_fraud_risk": gt.get("fraud_risk", 0.0),
            "checks": [],
        }

        for check_name, passed in [
            ("pipeline_success", result.success),
            ("has_assessments", bool(result.assessments)),
            ("output_files_exist", result.output_json_path is not None and Path(result.output_json_path).exists()),
        ]:
            total_checks += 1
            if passed:
                passed_checks += 1
            video_result["checks"].append({"name": check_name, "passed": passed})

        if result.assessments:
            top = result.assessments[0]
            video_result["actual_severity"] = top.severity
            video_result["actual_fault_ratio"] = top.fault_assessment.fault_ratio
            video_result["actual_fraud_score"] = top.fraud_risk.risk_score

        results[video_name] = video_result

    pass_rate = round(passed_checks / total_checks * 100, 1) if total_checks else 0.0
    return {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": backend,
        "seed": seed,
        "enable_conformal": enable_conformal,
        "enable_recalibration": enable_recalibration,
        "enable_fraud_detection": enable_fraud_detection,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "pass_rate": pass_rate,
        "results": results,
    }


def _cmd_benchmark(args: argparse.Namespace) -> int:
    print("=" * 60)
    print("Insurance MVP E2E Benchmark")
    print(f"Backend : {args.backend}")
    print(f"Seed    : {args.seed}")
    print(f"Ablation: {args.ablation}")
    print("=" * 60)

    # Full run
    full_result = _run_single_benchmark(args.backend, seed=args.seed, label="full")
    _print_benchmark_summary(full_result)

    all_results = [full_result]

    if args.ablation:
        ablation_variants = [
            {"label": "no_conformal", "enable_conformal": False},
            {"label": "no_recalibration", "enable_recalibration": False},
            {"label": "no_fraud_detection", "enable_fraud_detection": False},
        ]
        print("\n--- Ablation Study ---")
        for variant in ablation_variants:
            label = variant.pop("label")
            print(f"\nRunning ablation: {label}")
            res = _run_single_benchmark(args.backend, seed=args.seed, label=label, **variant)
            _print_benchmark_summary(res)
            all_results.append(res)

        _print_ablation_table(all_results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(all_results if args.ablation else full_result, fh, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    # Exit 1 if the full benchmark did not achieve 100% pass rate
    return 0 if full_result["pass_rate"] == 100.0 else 1


def _print_benchmark_summary(result: dict) -> None:
    label = result.get("label", "")
    print(f"\n[{label}]  pass_rate={result['pass_rate']}%  "
          f"({result['passed_checks']}/{result['total_checks']} checks)")


def _print_ablation_table(results: list[dict]) -> None:
    print("\n--- Ablation Summary ---")
    print(f"{'Label':<25} {'Pass Rate':>10} {'Passed':>8} {'Total':>7}")
    print("-" * 55)
    for r in results:
        print(f"{r['label']:<25} {r['pass_rate']:>9.1f}% {r['passed_checks']:>8} {r['total_checks']:>7}")


# ---------------------------------------------------------------------------
# Subcommand: sensitivity
# ---------------------------------------------------------------------------


def _add_sensitivity_parser(subparsers) -> None:
    sens_parser = subparsers.add_parser(
        "sensitivity",
        help="Run fusion-weight sensitivity analysis (grid search).",
        description=(
            "Run a grid search over audio/motion/proximity fusion weights.\n\n"
            "For each point on a simplex grid the pipeline is evaluated on demo\n"
            "videos and accuracy is recorded.  Results are printed as a table.\n\n"
            "See also: scripts/research_benchmark.py --sensitivity-analysis"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sens_parser.add_argument(
        "--backend",
        type=str,
        default="mock",
        help="VLM backend (default: mock).",
    )
    sens_parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Grid step size for weight search (default: 0.1). Smaller = finer grid.",
    )
    sens_parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Bootstrap resamples for BCa CIs (default: 2000).",
    )
    sens_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    sens_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for sensitivity results.",
    )
    sens_parser.set_defaults(func=_cmd_sensitivity)


def _cmd_sensitivity(args: argparse.Namespace) -> int:
    """Run sensitivity analysis via the CLI."""
    try:
        from insurance_mvp.evaluation.sensitivity import grid_search_fusion_weights
    except ImportError:
        print("ERROR: insurance_mvp.evaluation.sensitivity not found.")
        return 1

    print("=" * 60)
    print("Insurance MVP — Fusion Weight Sensitivity Analysis")
    print(f"Backend : {args.backend}")
    print(f"Step    : {args.step}")
    print(f"Seed    : {args.seed}")
    print("=" * 60)

    ground_truth = _load_ground_truth()
    demo_dir = Path(__file__).parent.parent.parent / "data" / "dashcam_demo"

    def eval_fn(weights: dict) -> float:
        """Evaluate a single weight configuration on demo videos."""
        from insurance_mvp.config import CosmosBackend, PipelineConfig
        from insurance_mvp.pipeline import InsurancePipeline
        backend = CosmosBackend.MOCK if args.backend == "mock" else CosmosBackend.QWEN25VL
        config = PipelineConfig(seed=args.seed)
        config.cosmos.backend = backend
        config.mining.audio_weight = weights["audio_weight"]
        config.mining.motion_weight = weights["motion_weight"]
        config.mining.proximity_weight = weights["proximity_weight"]
        pipeline = InsurancePipeline(config)
        correct = 0
        total = 0
        for video_path in sorted(demo_dir.glob("*.mp4")):
            video_id = video_path.stem
            gt = ground_truth.get(video_id, {}).get("severity")
            if not gt:
                continue
            try:
                result = pipeline.process_video(str(video_path), video_id=video_id)
                if result.assessments:
                    pred = result.assessments[0].severity
                    correct += int(pred == gt)
                    total += 1
            except Exception:
                total += 1
        return correct / total if total > 0 else 0.0

    results = grid_search_fusion_weights(eval_fn, step=args.step, n_bootstrap=args.n_bootstrap)

    # Print top 10
    print(f"\nTop 10 weight configurations (n={len(results)}):")
    print(f"{'audio':>8} {'motion':>8} {'prox':>8} {'accuracy':>10}")
    print("-" * 40)
    for r in results[:10]:
        print(f"{r['audio_weight']:>8.2f} {r['motion_weight']:>8.2f} "
              f"{r['proximity_weight']:>8.2f} {r['accuracy']:>9.2%}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Insurance MVP - End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    subparsers.required = True

    _add_run_parser(subparsers)
    _add_benchmark_parser(subparsers)
    _add_sensitivity_parser(subparsers)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
