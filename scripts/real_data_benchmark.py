#!/usr/bin/env python3
"""Run Insurance MVP pipeline on real dashcam videos and measure accuracy.

Supports multiple input sources:
- Japanese YouTube dashcam videos (data/jp_dashcam/)
- Nexar dataset (data/real_dashcam/nexar/)
- Custom videos with metadata.json

Usage:
    python scripts/real_data_benchmark.py --input data/jp_dashcam --output reports/jp_benchmark.json
"""

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from insurance_mvp.config import CosmosBackend, PipelineConfig
from insurance_mvp.cosmos import create_client
from insurance_mvp.pipeline import InsurancePipeline

try:
    from insurance_mvp.evaluation.statistical import evaluate as bca_evaluate
    _HAS_STATISTICAL = True
except ImportError:
    _HAS_STATISTICAL = False


def load_ground_truth(input_dir: Path):
    """Load ground truth annotations if available."""
    gt_path = input_dir / "ground_truth.json"
    metadata_path = input_dir / "metadata.json"

    if gt_path.exists():
        with open(gt_path) as f:
            return json.load(f)
    elif metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    else:
        return None


def scan_videos(input_dir: Path):
    """Scan directory for video files."""
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = []

    for video_path in input_dir.glob("*"):
        if video_path.suffix.lower() in video_exts:
            videos.append(video_path)

    return sorted(videos)


def infer_ground_truth_from_filename(video_path: Path):
    """Infer ground truth from video filename patterns."""
    name_lower = video_path.stem.lower()

    # Common patterns
    if any(kw in name_lower for kw in ["collision", "crash", "accident", "衝突", "事故"]):
        return {"gt_severity": "HIGH", "gt_fault_ratio": 100.0, "source": "filename"}
    elif any(kw in name_lower for kw in ["near", "ニアミス", "危険"]):
        return {"gt_severity": "MEDIUM", "gt_fault_ratio": 50.0, "source": "filename"}
    elif any(kw in name_lower for kw in ["normal", "通常", "安全"]):
        return {"gt_severity": "NONE", "gt_fault_ratio": 0.0, "source": "filename"}
    else:
        return {"gt_severity": "UNKNOWN", "gt_fault_ratio": None, "source": "unknown"}


def main():
    parser = argparse.ArgumentParser(description="Run real data benchmark")
    parser.add_argument("--input", type=str, default="data/jp_dashcam", help="Input video directory")
    parser.add_argument("--output", type=str, default="reports/real_data_benchmark.json", help="Output JSON path")
    parser.add_argument("--backend", type=str, default="real", choices=["mock", "real"], help="VLM backend")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos")
    parser.add_argument("--extract-clips", action="store_true", help="Extract danger clips as separate files")
    parser.add_argument("--fps", type=float, default=None, help="Frame sampling rate for VLM (default: VlmConfig.fps=2.0)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per clip for VLM (default: VlmConfig.max_frames=48)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Real Data Benchmark ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_path}")
    print(f"Backend: {args.backend}")

    # Load ground truth
    ground_truth = load_ground_truth(input_dir)
    if ground_truth:
        print(f"Ground truth loaded: {len(ground_truth)} entries")
    else:
        print("No ground truth found, will infer from filenames")

    # Scan videos
    videos = scan_videos(input_dir)
    if not videos:
        print(f"ERROR: No videos found in {input_dir}")
        return

    if args.max_videos:
        videos = videos[:args.max_videos]

    print(f"Found {len(videos)} videos")

    # Create pipeline with backend configuration
    from insurance_mvp.config import CosmosConfig

    backend = CosmosBackend.QWEN25VL if args.backend == "real" else CosmosBackend.MOCK

    cosmos_config = CosmosConfig(backend=backend)
    config = PipelineConfig()
    config.cosmos = cosmos_config
    if args.fps is not None:
        config.vlm.fps = args.fps
    if args.max_frames is not None:
        config.vlm.max_frames = args.max_frames
    if args.extract_clips:
        config.mining.extract_clips = True

    pipeline = InsurancePipeline(config)

    print(f"\nVLM Backend: {cosmos_config.backend}")
    print(f"Model: {cosmos_config.model_name}")
    print(f"FPS: {config.vlm.fps}  Max frames: {config.vlm.max_frames}")

    # Run inference
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": args.backend,
        "input_dir": str(input_dir),
        "total_videos": len(videos),
        "videos": {},
    }

    start_time = time.time()

    for video_path in tqdm(videos, desc="Processing videos"):
        video_id = video_path.stem

        # Get ground truth
        if ground_truth:
            if isinstance(ground_truth, dict):
                gt = ground_truth.get(video_id, ground_truth.get(video_path.name, {}))
            else:
                gt = {}
        else:
            gt = infer_ground_truth_from_filename(video_path)

        # Run pipeline
        try:
            result = pipeline.process_video(str(video_path), video_id=video_id)
            # VideoResult has .assessments list of ClaimAssessment
            if result.assessments:
                assessment = result.assessments[0]
                results["videos"][video_id] = {
                    "video_path": str(video_path),
                    "success": True,
                    "ground_truth": gt,
                    "predicted": {
                        "severity": assessment.severity,
                        "confidence": assessment.confidence,
                        "fault_ratio": assessment.fault_assessment.fault_ratio,
                        "fraud_score": assessment.fraud_risk.risk_score,
                        "prediction_set": list(assessment.prediction_set),
                        "review_priority": assessment.review_priority,
                    },
                    "processing_time_sec": assessment.processing_time_sec,
                }
            else:
                results["videos"][video_id] = {
                    "video_path": str(video_path),
                    "success": True,
                    "ground_truth": gt,
                    "predicted": None,
                    "processing_time_sec": result.processing_time_sec,
                }
        except Exception as e:
            results["videos"][video_id] = {
                "video_path": str(video_path),
                "success": False,
                "error": str(e),
                "ground_truth": gt,
            }

    total_time = time.time() - start_time

    # Calculate metrics
    metrics = calculate_metrics(results["videos"])
    results["metrics"] = metrics
    results["total_time_sec"] = total_time
    results["mean_time_per_video_sec"] = total_time / len(videos)

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n=== Benchmark Complete ===")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Mean per video: {total_time/len(videos):.1f}s")
    print(f"\nResults saved to: {output_path}")

    if metrics:
        print(f"\n=== Accuracy Metrics ===")
        n = metrics.get("total_with_gt", 0)
        matches = metrics.get("severity_exact_matches", 0)
        acc = metrics.get("severity_accuracy", 0)
        ci = metrics.get("bca_ci_95")
        if ci:
            lo, hi = ci
            print(f"Severity accuracy: {acc:.1f}% [{lo*100:.1f}%, {hi*100:.1f}%] (95% CI, BCa, n={n})")
        else:
            print(f"Severity accuracy: {acc:.1f}%")
        print(f"Exact matches: {matches}/{n}")
        print(f"Mean distance: {metrics.get('mean_severity_distance', 0):.2f}")

        if metrics.get("confusion_matrix"):
            print(f"\nConfusion Matrix:")
            cm = metrics["confusion_matrix"]
            for true_label, preds in cm.items():
                print(f"  {true_label}: {preds}")


def calculate_metrics(videos):
    """Calculate accuracy metrics from results."""
    severity_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

    total_with_gt = 0
    exact_matches = 0
    severity_distances = []
    confusion = {}

    for video_id, result in videos.items():
        if not result.get("success"):
            continue

        gt = result.get("ground_truth") or {}
        pred = result.get("predicted") or {}

        gt_severity = gt.get("gt_severity")
        pred_severity = pred.get("severity") if pred else None

        if not gt_severity or gt_severity == "UNKNOWN":
            continue

        total_with_gt += 1

        # Exact match
        if pred_severity == gt_severity:
            exact_matches += 1

        # Severity distance
        if pred_severity in severity_order and gt_severity in severity_order:
            dist = abs(severity_order[pred_severity] - severity_order[gt_severity])
            severity_distances.append(dist)

        # Confusion matrix
        if gt_severity not in confusion:
            confusion[gt_severity] = {}
        confusion[gt_severity][pred_severity] = confusion[gt_severity].get(pred_severity, 0) + 1

    if total_with_gt == 0:
        return None

    accuracy = exact_matches / total_with_gt

    # BCa bootstrap confidence interval
    bca_ci = None
    if _HAS_STATISTICAL:
        try:
            labels_true = []
            labels_pred = []
            for video_id, result in videos.items():
                if not result.get("success"):
                    continue
                gt = result.get("ground_truth") or {}
                pred = result.get("predicted") or {}
                gt_sev = gt.get("gt_severity")
                pred_sev = pred.get("severity") if pred else None
                if gt_sev and gt_sev != "UNKNOWN":
                    labels_true.append(gt_sev)
                    labels_pred.append(pred_sev or "NONE")
            report = bca_evaluate(labels_true, labels_pred, n_bootstrap=2000)
            bca_ci = (report.accuracy.lower, report.accuracy.upper)
        except Exception:
            bca_ci = None

    return {
        "total_with_gt": total_with_gt,
        "severity_exact_matches": exact_matches,
        "severity_accuracy": 100.0 * accuracy,
        "bca_ci_95": bca_ci,
        "mean_severity_distance": sum(severity_distances) / len(severity_distances) if severity_distances else 0,
        "confusion_matrix": confusion,
    }


if __name__ == "__main__":
    main()
