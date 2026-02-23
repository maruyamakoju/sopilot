#!/usr/bin/env python3
"""Direct VLM evaluation on 10 expanded demo videos.

Runs Qwen2.5-VL-7B (or mock) on each video directly via VideoLLMClient,
bypassing the full pipeline. This isolates VLM severity accuracy from
pipeline artifacts (mining, recalibration, conformal).

Usage:
    # Mock (instant, no GPU)
    python scripts/real_vlm_eval_10.py --backend mock

    # Real VLM (GPU required, ~25 min for 10 videos)
    python scripts/real_vlm_eval_10.py --backend real

    # Real VLM with specific options
    python scripts/real_vlm_eval_10.py --backend real --max-frames 48 --fps 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


SEVERITY_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]


def load_ground_truth(videos_dir: Path) -> dict:
    """Load ground truth from metadata.json."""
    meta_path = videos_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {videos_dir}")
    with open(meta_path) as f:
        return json.load(f)


def check_gpu():
    """Check GPU availability and return info."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {"available": True, "name": name, "memory_gb": round(mem_gb, 1)}
    except ImportError:
        pass
    return {"available": False, "name": None, "memory_gb": 0}


def run_vlm_direct(video_path: Path, video_id: str, client, backend: str) -> dict:
    """Run VLM directly on a video and return structured result.

    For mock backend, uses pipeline's filename-aware mock instead of
    cosmos client's prompt-based mock (which can't see filenames).
    """
    t0 = time.time()
    try:
        if backend == "mock":
            from insurance_mvp.pipeline.stages.vlm_inference import mock_vlm_result

            mock_result = mock_vlm_result({"video_path": str(video_path)})
            elapsed = time.time() - t0
            return {
                "success": True,
                "severity": mock_result["severity"],
                "confidence": mock_result["confidence"],
                "causal_reasoning": mock_result.get("reasoning", "")[:300],
                "processing_time_sec": round(elapsed, 2),
            }

        assessment = client.assess_claim(
            video_path=video_path,
            video_id=video_id,
        )
        elapsed = time.time() - t0
        return {
            "success": True,
            "severity": assessment.severity,
            "confidence": assessment.confidence,
            "causal_reasoning": assessment.causal_reasoning[:300] if assessment.causal_reasoning else "",
            "processing_time_sec": round(elapsed, 2),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "success": False,
            "error": str(e)[:200],
            "processing_time_sec": round(elapsed, 2),
        }


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy metrics from results."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0

    # Severity distance
    sev_idx = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
    distances = []
    for r in results:
        if r["predicted"] in sev_idx and r["expected"] in sev_idx:
            distances.append(abs(sev_idx[r["predicted"]] - sev_idx[r["expected"]]))

    # Confusion matrix
    confusion = [[0] * 4 for _ in range(4)]
    for r in results:
        if r["expected"] in sev_idx and r["predicted"] in sev_idx:
            confusion[sev_idx[r["expected"]]][sev_idx[r["predicted"]]] += 1

    # Per-class precision/recall
    per_class = {}
    for i, sev in enumerate(SEVERITY_LEVELS):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(4) if j != i)
        fn = sum(confusion[i][j] for j in range(4) if j != i)
        per_class[sev] = {
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "mean_severity_distance": sum(distances) / len(distances) if distances else 0.0,
        "confusion_matrix": {"labels": SEVERITY_LEVELS, "matrix": confusion},
        "per_class": per_class,
        "target_met": accuracy >= 0.80,
    }


def main():
    parser = argparse.ArgumentParser(description="Direct VLM evaluation on 10 demo videos")
    parser.add_argument("--backend", default="mock", choices=["mock", "real"])
    parser.add_argument("--videos-dir", default="data/dashcam_demo")
    parser.add_argument("--output", default="reports/real_vlm_eval_10.json")
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Direct VLM Severity Evaluation (10 videos)")
    print("=" * 60)

    # GPU check
    gpu_info = check_gpu()
    if args.backend == "real":
        if not gpu_info["available"]:
            print("WARNING: No GPU detected. Real VLM requires CUDA GPU.")
            print("Falling back to mock backend.")
            args.backend = "mock"
        else:
            print(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']} GB)")

    # Load ground truth
    metadata = load_ground_truth(videos_dir)
    print(f"Ground truth: {len(metadata)} scenarios")

    # Create VLM client
    from insurance_mvp.cosmos.client import VLMConfig, VideoLLMClient

    config = VLMConfig(
        model_name="qwen2.5-vl-7b" if args.backend == "real" else "mock",
        device="cuda" if args.backend == "real" else "cpu",
        fps=args.fps,
        max_frames=args.max_frames,
        temperature=args.temperature,
    )
    print(f"Backend: {config.model_name}")
    print(f"FPS: {config.fps}, max_frames: {config.max_frames}, temp: {config.temperature}")

    client = VideoLLMClient(config)

    # Run evaluation
    results = []
    total_time = 0.0

    entries = sorted(metadata.items())
    if args.max_videos:
        entries = entries[: args.max_videos]

    print(f"\nEvaluating {len(entries)} videos...\n")

    for video_name, meta in entries:
        expected = meta["severity"]
        video_path = videos_dir / f"{video_name}.mp4"

        if not video_path.exists():
            print(f"  SKIP {video_name}: file not found")
            continue

        result = run_vlm_direct(video_path, video_name, client, args.backend)
        predicted = result.get("severity", "ERROR") if result["success"] else "ERROR"
        is_correct = predicted == expected
        elapsed = result["processing_time_sec"]
        total_time += elapsed

        status = "OK" if is_correct else "FAIL"
        print(f"  [{status}] {video_name:25s} expected={expected:6s} got={predicted:6s} "
              f"conf={result.get('confidence', 0):.2f} ({elapsed:.1f}s)")

        results.append({
            "video": video_name,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "confidence": result.get("confidence", 0.0),
            "processing_time_sec": elapsed,
            "reasoning": result.get("causal_reasoning", ""),
        })

    # Compute metrics
    metrics = compute_metrics(results)

    # Print summary
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.1%})")
    print(f"Target:    80% {'MET' if metrics['target_met'] else 'NOT MET'}")
    print(f"Mean dist: {metrics['mean_severity_distance']:.2f}")
    print(f"Time:      {total_time:.1f}s total, {total_time / len(results):.1f}s/video")

    print(f"\nPer-class:")
    for sev in SEVERITY_LEVELS:
        pc = metrics["per_class"][sev]
        print(f"  {sev:6s}: P={pc['precision']:.2f} R={pc['recall']:.2f}")

    print(f"\nConfusion (rows=actual, cols=predicted):")
    header = "        " + "  ".join(f"{s:>6s}" for s in SEVERITY_LEVELS)
    print(header)
    for i, row in enumerate(metrics["confusion_matrix"]["matrix"]):
        row_str = "  ".join(f"{v:>6d}" for v in row)
        print(f"  {SEVERITY_LEVELS[i]:>6s} {row_str}")

    print("=" * 60)

    # Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": args.backend,
        "gpu": gpu_info,
        "config": {
            "fps": args.fps,
            "max_frames": args.max_frames,
            "temperature": args.temperature,
        },
        "metrics": metrics,
        "results": results,
        "total_time_sec": round(total_time, 2),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {output_path}")

    sys.exit(0 if metrics["target_met"] else 1)


if __name__ == "__main__":
    main()
