#!/usr/bin/env python3
"""Unified research benchmark with experiment tracking and ablation studies.

Runs the insurance pipeline on demo videos, records results via
ExperimentTracker, and reports accuracy with BCa confidence intervals.

Usage:
    python scripts/research_benchmark.py --backend mock
    python scripts/research_benchmark.py --backend mock --ablation
    python scripts/research_benchmark.py --backend mock --ablation-flag recalibration
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from insurance_mvp.config import CosmosBackend, PipelineConfig, load_config
from insurance_mvp.evaluation import statistical
from insurance_mvp.evaluation.sensitivity import format_sensitivity_table, grid_search_fusion_weights
from insurance_mvp.evaluation.experiment import (
    AblationRunner,
    ExperimentConfig,
    ExperimentResult,
    ExperimentTracker,
    format_ablation_report,
)
from insurance_mvp.pipeline.orchestrator import InsurancePipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]

# All supported ablation feature flags
ALL_ABLATION_FLAGS = [
    "recalibration",
    "conformal",
    "audio_signal",
    "motion_signal",
    "proximity_signal",
]

DEFAULT_VIDEOS_DIR = Path(__file__).resolve().parent.parent / "data" / "dashcam_demo"
DEFAULT_EXPERIMENTS_PATH = (
    Path(__file__).resolve().parent.parent / "experiments" / "insurance_experiments.json"
)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def load_metadata(videos_dir: Path) -> dict:
    """Load ground-truth metadata from videos_dir/metadata.json."""
    meta_path = videos_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {videos_dir}")
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _build_pipeline(config: ExperimentConfig, backend: str) -> InsurancePipeline:
    """Construct an InsurancePipeline reflecting the given experiment config flags."""
    # Use PipelineConfig constructor (same pattern as expanded_video_eval.py)
    pipeline_cfg = PipelineConfig()
    pipeline_cfg.continue_on_error = True

    # Set VLM backend
    pipeline_cfg.cosmos.backend = (
        CosmosBackend.QWEN25VL if backend == "real" else CosmosBackend.MOCK
    )

    ablation_flags = config.ablation_flags

    # Apply ablation flags to pipeline config
    if "recalibration" in ablation_flags:
        pipeline_cfg.enable_recalibration = ablation_flags["recalibration"]

    if "conformal" in ablation_flags:
        pipeline_cfg.enable_conformal = ablation_flags["conformal"]

    # Signal flags: disable individual mining analyzers via their weights.
    # MiningConfig exposes audio_weight, motion_weight, proximity_weight.
    if not ablation_flags.get("audio_signal", True):
        pipeline_cfg.mining.audio_weight = 0.0

    if not ablation_flags.get("motion_signal", True):
        pipeline_cfg.mining.motion_weight = 0.0

    if not ablation_flags.get("proximity_signal", True):
        pipeline_cfg.mining.proximity_weight = 0.0

    return InsurancePipeline(pipeline_cfg)


def run_evaluation(
    config: ExperimentConfig,
    backend: str,
    videos_dir: Path,
) -> tuple[dict[str, float], float]:
    """Run the full pipeline on demo videos and return (metrics, timing_sec).

    This is the eval_fn signature required by AblationRunner:
        eval_fn(config) -> (metrics_dict, timing_seconds)

    The closure captures ``backend`` and ``videos_dir`` from the outer scope.
    """
    ExperimentTracker.set_seed(config.seed)
    t0 = time.time()

    pipeline = _build_pipeline(config, backend)
    metadata = load_metadata(videos_dir)

    y_true: list[str] = []
    y_pred: list[str] = []

    for video_name, meta in sorted(metadata.items()):
        expected = meta["severity"]
        video_path = videos_dir / f"{video_name}.mp4"

        if not video_path.exists():
            continue

        result = pipeline.process_video(str(video_path), video_id=video_name)

        if not result.success or not result.assessments:
            predicted = "NONE"
        else:
            predicted = result.assessments[0].severity

        y_true.append(expected)
        y_pred.append(predicted)

    timing_sec = time.time() - t0

    # Map string labels to integer indices for compute_standard_metrics
    label_to_idx = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
    yt_int = [label_to_idx.get(s, 0) for s in y_true]
    yp_int = [label_to_idx.get(s, 0) for s in y_pred]

    metrics = ExperimentTracker.compute_metrics(yt_int, yp_int, n_classes=len(SEVERITY_LEVELS))

    # Attach n_samples for downstream BCa reporting
    metrics["n_samples"] = float(len(y_true))

    return metrics, timing_sec


# ---------------------------------------------------------------------------
# BCa CI reporting
# ---------------------------------------------------------------------------

def _print_bca_footer(y_true: list[str], y_pred: list[str], alpha: float = 0.05) -> None:
    """Print BCa confidence interval footer for accuracy."""
    n = len(y_true)
    if n == 0:
        print("  No samples — cannot compute CI.")
        return

    report = statistical.evaluate(y_true, y_pred, labels=SEVERITY_LEVELS, alpha=alpha)
    ci = report.accuracy
    pct = int((1 - alpha) * 100)
    print(
        f"\nAccuracy: {ci.point:.2f} [{ci.lower:.2f}, {ci.upper:.2f}] "
        f"({pct}% CI, BCa, n={n})"
    )


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------

def run_single(
    backend: str,
    videos_dir: Path,
    tracker: ExperimentTracker,
    ablation_flags: dict[str, bool] | None = None,
    seed: int = 42,
    experiment_name: str = "insurance_pipeline_full",
) -> tuple[ExperimentConfig, list[str], list[str]]:
    """Run a single experiment run (no ablation) and record via tracker.

    Returns (config, y_true, y_pred) for downstream CI computation.
    """
    flags = ablation_flags if ablation_flags is not None else {
        "recalibration": True,
        "conformal": True,
        "audio_signal": True,
        "motion_signal": True,
        "proximity_signal": True,
    }

    config = ExperimentConfig(
        name=experiment_name,
        seed=seed,
        description="Full-feature insurance pipeline on dashcam_demo",
        hyperparams={
            "backend": backend,
            "videos_dir": str(videos_dir),
            "n_bootstrap": 10000,
            "alpha": 0.05,
        },
        ablation_flags=flags,
    )

    # Register (idempotent if same fingerprint)
    try:
        tracker.register(config)
    except ValueError as exc:
        # Config name exists with different fingerprint — use unique timestamp name
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config = config.derive(name=f"{experiment_name}_{ts}")
        tracker.register(config)

    print(f"\nRunning experiment: {config.name}")
    print(f"  Backend: {backend}")
    print(f"  Videos dir: {videos_dir}")
    print(f"  Ablation flags: {flags}")
    print(f"  Seed: {seed}\n")

    # Collect per-video labels for CI computation
    ExperimentTracker.set_seed(seed)
    t0 = time.time()

    pipeline = _build_pipeline(config, backend)
    metadata = load_metadata(videos_dir)

    y_true: list[str] = []
    y_pred: list[str] = []

    for video_name, meta in sorted(metadata.items()):
        expected = meta["severity"]
        video_path = videos_dir / f"{video_name}.mp4"
        if not video_path.exists():
            print(f"  SKIP {video_name}: file not found")
            continue

        result = pipeline.process_video(str(video_path), video_id=video_name)

        if not result.success or not result.assessments:
            predicted = "NONE"
        else:
            predicted = result.assessments[0].severity

        is_correct = predicted == expected
        status = "OK" if is_correct else "FAIL"
        print(f"  [{status}] {video_name}: expected={expected}, got={predicted}")

        y_true.append(expected)
        y_pred.append(predicted)

    timing_sec = time.time() - t0
    n = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = correct / n if n > 0 else 0.0

    print(f"\n  Accuracy: {correct}/{n} = {accuracy:.1%} in {timing_sec:.1f}s")

    # Compute full metrics for tracker
    label_to_idx = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
    yt_int = [label_to_idx.get(s, 0) for s in y_true]
    yp_int = [label_to_idx.get(s, 0) for s in y_pred]
    metrics = ExperimentTracker.compute_metrics(yt_int, yp_int, n_classes=len(SEVERITY_LEVELS))
    metrics["n_samples"] = float(n)

    tracker.record(
        config.name,
        metrics,
        timing_sec=timing_sec,
        metadata={
            "backend": backend,
            "videos_dir": str(videos_dir),
            "correct": correct,
            "total": n,
        },
    )

    return config, y_true, y_pred


def run_sensitivity_analysis(
    backend: str,
    videos_dir: Path,
    seed: int = 42,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    step: float = 0.1,
) -> None:
    """Run fusion-weight sensitivity analysis and print a ranked table.

    Evaluates all (audio_w, motion_w, proximity_w) combinations on the
    3-simplex with resolution *step* (66 points at step=0.1).  Each grid
    point runs the full pipeline on the demo video set and computes a BCa
    bootstrap CI on accuracy.
    """
    metadata = load_metadata(videos_dir)

    def _eval_fn(audio_w: float, motion_w: float, proximity_w: float):
        """Pipeline eval for a single weight triple."""
        pipeline_cfg = PipelineConfig()
        pipeline_cfg.continue_on_error = True
        pipeline_cfg.cosmos.backend = (
            CosmosBackend.QWEN25VL if backend == "real" else CosmosBackend.MOCK
        )
        pipeline_cfg.mining.audio_weight = audio_w
        pipeline_cfg.mining.motion_weight = motion_w
        pipeline_cfg.mining.proximity_weight = proximity_w
        pipeline_cfg.seed = seed

        pipeline = InsurancePipeline(pipeline_cfg)

        y_true_s: list[str] = []
        y_pred_s: list[str] = []
        for video_name, meta in sorted(metadata.items()):
            expected = meta["severity"]
            video_path = videos_dir / f"{video_name}.mp4"
            if not video_path.exists():
                continue
            result = pipeline.process_video(str(video_path), video_id=video_name)
            predicted = result.assessments[0].severity if (result.success and result.assessments) else "NONE"
            y_true_s.append(expected)
            y_pred_s.append(predicted)
        return y_true_s, y_pred_s

    n_points = len(list(__import__("itertools").product(
        range(round(1 / step) + 1), repeat=2
    )))  # rough estimate for progress display

    print(f"\nRunning fusion-weight sensitivity analysis")
    print(f"  Backend  : {backend}")
    print(f"  Grid step: {step} (~66 points for step=0.1)")
    print(f"  Bootstrap: n={n_bootstrap}, alpha={alpha}")
    print(f"  Seed     : {seed}\n")

    results = grid_search_fusion_weights(
        _eval_fn,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        seed=seed,
        step=step,
    )

    print("\nFUSION WEIGHT SENSITIVITY (top 15, sorted by accuracy)")
    print(format_sensitivity_table(results, top_k=15))

    # Also print bottom 5 to show contrast
    print("\nFUSION WEIGHT SENSITIVITY (bottom 5, worst configurations)")
    worst = sorted(results, key=lambda r: r.accuracy_ci.point)
    print(format_sensitivity_table(worst, top_k=5))

    best = results[0]
    print(
        f"\nOptimal weights: audio={best.audio_w:.2f}  motion={best.motion_w:.2f}  "
        f"proximity={best.proximity_w:.2f}\n"
        f"  Accuracy: {best.accuracy_ci.point:.4f} "
        f"[{best.accuracy_ci.lower:.4f}, {best.accuracy_ci.upper:.4f}] "
        f"(95% CI, BCa, n_bootstrap={n_bootstrap})"
    )


def run_ablation_study(
    backend: str,
    videos_dir: Path,
    tracker: ExperimentTracker,
    active_flags: list[str],
    seed: int = 42,
) -> None:
    """Run a full leave-one-out ablation study over active_flags."""
    ablation_flags = {flag: True for flag in active_flags}

    base_config = ExperimentConfig(
        name="insurance_ablation_base",
        seed=seed,
        description="Ablation baseline: all selected features enabled",
        hyperparams={
            "backend": backend,
            "videos_dir": str(videos_dir),
        },
        ablation_flags=ablation_flags,
    )

    # Capture backend/videos_dir in closure
    def eval_fn(cfg: ExperimentConfig) -> tuple[dict[str, float], float]:
        return run_evaluation(cfg, backend=backend, videos_dir=videos_dir)

    runner = AblationRunner(
        base_config=base_config,
        eval_fn=eval_fn,
        primary_metric="accuracy",
        tracker=tracker,
    )

    print(f"\nRunning ablation study over flags: {active_flags}")
    print(f"  This will run {len(active_flags) + 1} evaluation(s).\n")

    report = runner.run()
    print(format_ablation_report(report, primary_metric="accuracy"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified research benchmark with experiment tracking and ablation studies."
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "real"],
        default="mock",
        help="VLM backend (default: mock)",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run a full ablation study over all feature flags",
    )
    parser.add_argument(
        "--ablation-flag",
        dest="ablation_flag",
        choices=ALL_ABLATION_FLAGS,
        default=None,
        help="Run ablation study over a single flag only",
    )
    parser.add_argument(
        "--sensitivity-analysis",
        dest="sensitivity",
        action="store_true",
        help="Run fusion-weight sensitivity analysis (66 grid points on simplex)",
    )
    parser.add_argument(
        "--sensitivity-step",
        dest="sensitivity_step",
        type=float,
        default=0.1,
        help="Grid resolution for sensitivity analysis (default: 0.1 → 66 points)",
    )
    parser.add_argument(
        "--n-bootstrap",
        dest="n_bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples per grid point for sensitivity CI (default: 1000)",
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=str(DEFAULT_VIDEOS_DIR),
        help="Directory containing videos + metadata.json",
    )
    parser.add_argument(
        "--experiments-path",
        type=str,
        default=str(DEFAULT_EXPERIMENTS_PATH),
        help="Path to experiments JSON storage file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed (default: 42)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for BCa CI (default: 0.05)",
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        print(f"ERROR: videos directory not found: {videos_dir}", file=sys.stderr)
        sys.exit(1)

    experiments_path = Path(args.experiments_path)
    experiments_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = ExperimentTracker(storage_path=experiments_path)

    # Determine ablation flags to study
    if args.ablation:
        active_flags = ALL_ABLATION_FLAGS
    elif args.ablation_flag:
        active_flags = [args.ablation_flag]
    else:
        active_flags = None

    if args.sensitivity:
        # Sensitivity analysis mode
        run_sensitivity_analysis(
            backend=args.backend,
            videos_dir=videos_dir,
            seed=args.seed,
            n_bootstrap=args.n_bootstrap,
            alpha=args.alpha,
            step=args.sensitivity_step,
        )
    elif active_flags is not None:
        # Ablation mode
        run_ablation_study(
            backend=args.backend,
            videos_dir=videos_dir,
            tracker=tracker,
            active_flags=active_flags,
            seed=args.seed,
        )
    else:
        # Standard single-run mode
        config, y_true, y_pred = run_single(
            backend=args.backend,
            videos_dir=videos_dir,
            tracker=tracker,
            seed=args.seed,
        )

        # Print BCa confidence interval footer
        _print_bca_footer(y_true, y_pred, alpha=args.alpha)

    print(f"\nExperiments stored at: {experiments_path}")

    # Print best result across all recorded runs
    best = tracker.best_result(metric="accuracy")
    if best:
        print(
            f"Best recorded accuracy: {best.metrics.get('accuracy', 0.0):.4f} "
            f"({best.config.name})"
        )


if __name__ == "__main__":
    main()
