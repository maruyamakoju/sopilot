"""VLM Severity Accuracy Benchmark.

Runs VLM inference (mock or real) on all labeled demo videos and
compares predicted severity against ground truth from metadata.json.

Usage:
    python scripts/vlm_accuracy_benchmark.py --backend mock
    python scripts/vlm_accuracy_benchmark.py --backend mock --videos-dir data/dashcam_demo
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from insurance_mvp.config import PipelineConfig, load_config
from insurance_mvp.pipeline.orchestrator import InsurancePipeline

SEVERITY_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]


def load_metadata(videos_dir: Path) -> dict:
    """Load ground truth metadata."""
    meta_path = videos_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {videos_dir}")
    with open(meta_path) as f:
        return json.load(f)


def run_benchmark(videos_dir: Path, backend: str = "mock", output_path: Path | None = None) -> dict:
    """Run VLM accuracy benchmark on all labeled videos.

    Args:
        videos_dir: Directory containing videos and metadata.json.
        backend: VLM backend ("mock" or "real").
        output_path: Optional path to save JSON report.

    Returns:
        Benchmark results dict.
    """
    metadata = load_metadata(videos_dir)

    # Create pipeline with mock backend
    config = load_config()
    config.continue_on_error = True

    pipeline = InsurancePipeline(config)

    results = []
    correct = 0
    total = 0

    # Confusion matrix: confusion[actual][predicted] = count
    confusion = defaultdict(lambda: defaultdict(int))

    for video_name, meta in sorted(metadata.items()):
        expected_severity = meta["severity"]
        video_path = videos_dir / f"{video_name}.mp4"

        if not video_path.exists():
            print(f"  SKIP {video_name}: video file not found")
            continue

        # Process video through pipeline
        result = pipeline.process_video(str(video_path), video_id=video_name)

        if not result.success or not result.assessments:
            predicted_severity = "ERROR"
            is_correct = False
        else:
            # Use highest-severity assessment (first after ranking)
            predicted_severity = result.assessments[0].severity
            is_correct = predicted_severity == expected_severity

        total += 1
        if is_correct:
            correct += 1

        confusion[expected_severity][predicted_severity] += 1

        status = "OK" if is_correct else "FAIL"
        print(f"  [{status}] {video_name}: expected={expected_severity}, got={predicted_severity}")

        results.append({
            "video": video_name,
            "expected": expected_severity,
            "predicted": predicted_severity,
            "correct": is_correct,
            "confidence": result.assessments[0].confidence if result.assessments else 0.0,
        })

    # Compute metrics
    accuracy = correct / total if total > 0 else 0.0

    # Per-class precision and recall
    per_class = {}
    for sev in SEVERITY_LEVELS:
        tp = confusion[sev][sev]
        fp = sum(confusion[other][sev] for other in SEVERITY_LEVELS if other != sev)
        fn = sum(confusion[sev][other] for other in SEVERITY_LEVELS if other != sev)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[sev] = {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}

    # Build confusion matrix as 2D list
    matrix = [[confusion[actual][predicted] for predicted in SEVERITY_LEVELS] for actual in SEVERITY_LEVELS]

    report = {
        "backend": backend,
        "videos_dir": str(videos_dir),
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "target_accuracy": 0.80,
        "target_met": accuracy >= 0.80,
        "results": results,
        "confusion_matrix": {
            "labels": SEVERITY_LEVELS,
            "matrix": matrix,
        },
        "per_class": per_class,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {output_path}")

    return report


def print_report(report: dict):
    """Print formatted benchmark report."""
    print("\n" + "=" * 60)
    print("VLM SEVERITY ACCURACY BENCHMARK")
    print("=" * 60)
    print(f"Backend:  {report['backend']}")
    print(f"Videos:   {report['total']}")
    print(f"Correct:  {report['correct']}/{report['total']}")
    print(f"Accuracy: {report['accuracy']:.1%}")
    print(f"Target:   {report['target_accuracy']:.0%} {'MET' if report['target_met'] else 'NOT MET'}")

    print("\n--- Per-Class Metrics ---")
    for sev in SEVERITY_LEVELS:
        cls = report["per_class"].get(sev, {})
        print(f"  {sev:6s}: precision={cls.get('precision', 0):.2f}  recall={cls.get('recall', 0):.2f}")

    print("\n--- Confusion Matrix (rows=actual, cols=predicted) ---")
    labels = report["confusion_matrix"]["labels"]
    matrix = report["confusion_matrix"]["matrix"]
    header = "        " + "  ".join(f"{l:>6s}" for l in labels)
    print(header)
    for i, row in enumerate(matrix):
        row_str = "  ".join(f"{v:>6d}" for v in row)
        print(f"  {labels[i]:>6s} {row_str}")

    print("\n--- Per-Video Results ---")
    for r in report["results"]:
        status = "OK" if r["correct"] else "FAIL"
        print(f"  [{status}] {r['video']}: expected={r['expected']}, got={r['predicted']} (conf={r['confidence']:.2f})")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="VLM Severity Accuracy Benchmark")
    parser.add_argument("--backend", choices=["mock", "real"], default="mock", help="VLM backend")
    parser.add_argument("--videos-dir", type=str, default="data/dashcam_demo", help="Directory with videos + metadata")
    parser.add_argument("--output", type=str, default="reports/vlm_accuracy_benchmark.json", help="Output JSON path")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    output_path = Path(args.output)

    print(f"Running VLM accuracy benchmark ({args.backend} backend)...")
    print(f"Videos dir: {videos_dir}")

    report = run_benchmark(videos_dir, backend=args.backend, output_path=output_path)
    print_report(report)

    sys.exit(0 if report["target_met"] else 1)


if __name__ == "__main__":
    main()
