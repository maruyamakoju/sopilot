#!/usr/bin/env python3
"""Run VLM inference directly on dashcam videos (bypassing InsurancePipeline).

Direct Video-LLM assessment without mining/ranking stages.
"""

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from insurance_mvp.cosmos import create_client

# Optional GPU memory tracking
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def scan_videos(input_dir: Path):
    """Scan directory for video files."""
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = []
    for video_path in input_dir.glob("*"):
        if video_path.suffix.lower() in video_exts:
            videos.append(video_path)
    return sorted(videos)


def infer_ground_truth(video_path: Path):
    """Infer ground truth from filename."""
    name_lower = video_path.stem.lower()
    if any(kw in name_lower for kw in ["collision", "crash", "accident", "衝突", "事故"]):
        return {"gt_severity": "HIGH", "gt_fault_ratio": 100.0}
    elif any(kw in name_lower for kw in ["near", "ニアミス", "危険"]):
        return {"gt_severity": "MEDIUM", "gt_fault_ratio": 50.0}
    elif any(kw in name_lower for kw in ["normal", "通常", "安全"]):
        return {"gt_severity": "NONE", "gt_fault_ratio": 0.0}
    else:
        return {"gt_severity": "UNKNOWN", "gt_fault_ratio": None}


def main():
    parser = argparse.ArgumentParser(description="Direct VLM benchmark")
    parser.add_argument("--input", type=str, default="data/jp_dashcam")
    parser.add_argument("--output", type=str, default="reports/jp_direct_benchmark.json")
    parser.add_argument("--backend", type=str, default="real", choices=["mock", "real"])
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Direct VLM Benchmark ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_path}")
    print(f"Backend: {args.backend}")

    # Scan videos
    videos = scan_videos(input_dir)
    if not videos:
        print(f"ERROR: No videos found in {input_dir}")
        return

    if args.max_videos:
        videos = videos[:args.max_videos]

    print(f"Found {len(videos)} videos")

    # Create VLM client
    model_name = "qwen2.5-vl-7b" if args.backend == "real" else "mock"
    device = "cuda" if args.backend == "real" else "cpu"

    print(f"\nInitializing VLM client...")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")

    client = create_client(model_name=model_name, device=device)

    print(f"  Max frames: {client.config.max_frames}")
    print(f"  FPS: {client.config.fps}")

    # Run inference
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": args.backend,
        "model": model_name,
        "input_dir": str(input_dir),
        "total_videos": len(videos),
        "videos": {},
        "gpu_memory_tracking": [],
    }

    start_time = time.time()
    gpu_memory_baseline = None

    # Track initial GPU memory
    if TORCH_AVAILABLE and device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_memory_baseline = torch.cuda.memory_allocated() / 1024**3  # GB
        results["gpu_memory_baseline_gb"] = round(gpu_memory_baseline, 3)
        print(f"  GPU memory baseline: {gpu_memory_baseline:.3f} GB")

    for video_path in tqdm(videos, desc="Processing videos"):
        video_id = video_path.stem
        gt = infer_ground_truth(video_path)

        # Track GPU memory before processing
        gpu_mem_before = None
        if TORCH_AVAILABLE and device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_before = torch.cuda.memory_allocated() / 1024**3

        try:
            # Run VLM inference (full video)
            assessment = client.assess_claim(video_path=video_path, video_id=video_id)

            # Track GPU memory after processing
            gpu_mem_after = None
            gpu_mem_delta = None
            if TORCH_AVAILABLE and device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_mem_after = torch.cuda.memory_allocated() / 1024**3
                gpu_mem_delta = gpu_mem_after - gpu_mem_before if gpu_mem_before else None

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
                    "causal_reasoning": assessment.causal_reasoning,
                },
                "processing_time_sec": assessment.processing_time_sec,
                "gpu_memory_gb": {
                    "before": round(gpu_mem_before, 3) if gpu_mem_before else None,
                    "after": round(gpu_mem_after, 3) if gpu_mem_after else None,
                    "delta": round(gpu_mem_delta, 3) if gpu_mem_delta else None,
                },
            }

            if gpu_mem_delta is not None:
                results["gpu_memory_tracking"].append({
                    "video_id": video_id,
                    "delta_gb": round(gpu_mem_delta, 3),
                })

        except Exception as e:
            results["videos"][video_id] = {
                "video_path": str(video_path),
                "success": False,
                "error": str(e),
                "ground_truth": gt,
            }
            print(f"\nERROR processing {video_id}: {e}")

    total_time = time.time() - start_time

    # Calculate metrics
    severity_order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

    total_with_gt = 0
    exact_matches = 0
    severity_distances = []
    confusion = {}

    for video_id, result in results["videos"].items():
        if not result.get("success"):
            continue

        gt = result.get("ground_truth", {})
        pred = result.get("predicted", {})

        gt_severity = gt.get("gt_severity")
        pred_severity = pred.get("severity")

        if not gt_severity or gt_severity == "UNKNOWN":
            continue

        total_with_gt += 1

        if pred_severity == gt_severity:
            exact_matches += 1

        if pred_severity in severity_order and gt_severity in severity_order:
            dist = abs(severity_order[pred_severity] - severity_order[gt_severity])
            severity_distances.append(dist)

        if gt_severity not in confusion:
            confusion[gt_severity] = {}
        confusion[gt_severity][pred_severity] = confusion[gt_severity].get(pred_severity, 0) + 1

    metrics = {
        "total_with_gt": total_with_gt,
        "severity_exact_matches": exact_matches,
        "severity_accuracy": 100.0 * exact_matches / total_with_gt if total_with_gt > 0 else 0,
        "mean_severity_distance": sum(severity_distances) / len(severity_distances) if severity_distances else 0,
        "confusion_matrix": confusion,
    }

    results["metrics"] = metrics
    results["total_time_sec"] = total_time
    results["mean_time_per_video_sec"] = total_time / len(videos)

    # Calculate GPU memory metrics
    if results["gpu_memory_tracking"]:
        deltas = [m["delta_gb"] for m in results["gpu_memory_tracking"]]
        results["gpu_memory_metrics"] = {
            "mean_delta_gb": round(sum(deltas) / len(deltas), 3),
            "max_delta_gb": round(max(deltas), 3),
            "min_delta_gb": round(min(deltas), 3),
        }

    # Calculate success rate
    successful = sum(1 for v in results["videos"].values() if v.get("success"))
    cuda_failures = sum(1 for v in results["videos"].values() if not v.get("success") and "CUDA" in v.get("error", "").upper())
    results["success_rate"] = 100.0 * successful / len(videos)
    results["cuda_failures"] = cuda_failures

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n=== Benchmark Complete ===")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Mean per video: {total_time/len(videos):.1f}s")
    print(f"Success rate: {results['success_rate']:.1f}% ({successful}/{len(videos)})")
    if cuda_failures > 0:
        print(f"CUDA failures: {cuda_failures}")
    print(f"\nResults saved to: {output_path}")

    if metrics:
        print(f"\n=== Accuracy Metrics ===")
        print(f"Severity accuracy: {metrics.get('severity_accuracy', 0):.1f}%")
        print(f"Exact matches: {metrics.get('severity_exact_matches', 0)}/{metrics.get('total_with_gt', 0)}")
        print(f"Mean distance: {metrics.get('mean_severity_distance', 0):.2f}")

        if metrics.get("confusion_matrix"):
            print(f"\nConfusion Matrix:")
            for true_label, preds in metrics["confusion_matrix"].items():
                print(f"  {true_label}: {preds}")

    # Print GPU memory metrics
    if results.get("gpu_memory_metrics"):
        print(f"\n=== GPU Memory Metrics ===")
        gm = results["gpu_memory_metrics"]
        print(f"Baseline: {results.get('gpu_memory_baseline_gb', 0):.3f} GB")
        print(f"Mean delta: {gm['mean_delta_gb']:.3f} GB")
        print(f"Max delta: {gm['max_delta_gb']:.3f} GB")
        print(f"Min delta: {gm['min_delta_gb']:.3f} GB")
        if gm["max_delta_gb"] < 1.0:
            print("[OK] GPU memory cleanup working (max delta < 1GB)")


if __name__ == "__main__":
    main()
