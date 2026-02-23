#!/usr/bin/env python3
"""Expanded Real Video Evaluation — runs mock+real pipeline on all available videos.

Processes videos from multiple directories, generates comprehensive analysis:
- JP dashcam (20 YouTube compilations)
- Demo synthetic videos (3 ground-truth labeled)
- Raw labeled SOP videos (65 files)

Usage:
    # Mock backend (fast, all videos)
    python scripts/expanded_video_eval.py --backend mock --output reports/expanded_mock.json

    # Real VLM backend (GPU required, limited videos)
    python scripts/expanded_video_eval.py --backend real --max-videos 5 --output reports/expanded_real.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from insurance_mvp.config import CosmosBackend, PipelineConfig


# Optional GPU tracking
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


VIDEO_DIRS = [
    ("jp_dashcam", Path("data/jp_dashcam")),
    ("dashcam_demo", Path("data/dashcam_demo")),
    ("raw_sop", Path("data/raw")),
]

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def scan_all_videos(dirs: list[tuple[str, Path]], max_per_dir: int | None = None) -> list[dict]:
    """Scan all configured directories for video files."""
    videos = []
    for source_name, dir_path in dirs:
        if not dir_path.exists():
            continue
        found = sorted(p for p in dir_path.glob("*") if p.suffix.lower() in VIDEO_EXTS)
        if max_per_dir:
            found = found[:max_per_dir]
        for p in found:
            videos.append({
                "path": p,
                "source": source_name,
                "video_id": p.stem,
                "ground_truth": infer_ground_truth(p, source_name),
            })
    return videos


def _load_demo_metadata(dir_path: Path) -> dict | None:
    """Load metadata.json from demo directory if available."""
    meta_path = dir_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


# Cache for demo metadata
_demo_metadata_cache: dict[str, dict | None] = {}


def infer_ground_truth(video_path: Path, source: str) -> dict:
    """Infer ground truth from filename and source directory."""
    name = video_path.stem.lower()

    # Demo videos: use metadata.json for all 10 scenarios
    if source == "dashcam_demo":
        if source not in _demo_metadata_cache:
            _demo_metadata_cache[source] = _load_demo_metadata(video_path.parent)
        meta = _demo_metadata_cache.get(source)
        if meta and video_path.stem in meta:
            entry = meta[video_path.stem]
            return {
                "severity": entry["severity"],
                "fault_ratio": entry.get("ground_truth", {}).get("fault_ratio", 50.0),
                "confidence": 1.0,
                "label_source": "ground_truth",
            }

    # Raw SOP videos: suffix indicates label
    if source == "raw_sop":
        if "_gold" in name:
            return {"severity": "UNKNOWN", "label_source": "gold_reference", "quality": "gold"}
        elif "_missing" in name:
            return {"severity": "UNKNOWN", "label_source": "missing_step", "quality": "missing"}
        elif "_swap" in name or "_deviation" in name:
            return {"severity": "UNKNOWN", "label_source": "deviation", "quality": "deviation"}
        elif "_mixed" in name:
            return {"severity": "UNKNOWN", "label_source": "mixed", "quality": "mixed"}

    # JP dashcam: no ground truth (YouTube compilations)
    return {"severity": "UNKNOWN", "label_source": "none"}


def run_pipeline_on_video(pipeline, video: dict) -> dict:
    """Run pipeline on a single video, return structured result."""
    video_path = video["path"]
    video_id = video["video_id"]

    gpu_mem_before = None
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_mem_before = torch.cuda.memory_allocated() / 1024**3

    t0 = time.time()
    try:
        result = pipeline.process_video(str(video_path), video_id=video_id)
        elapsed = time.time() - t0

        gpu_mem_after = None
        gpu_delta = None
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_after = torch.cuda.memory_allocated() / 1024**3
            gpu_delta = gpu_mem_after - gpu_mem_before if gpu_mem_before is not None else None

        if result.assessments:
            a = result.assessments[0]
            return {
                "success": True,
                "severity": a.severity,
                "confidence": a.confidence,
                "fault_ratio": a.fault_assessment.fault_ratio,
                "fault_scenario": a.fault_assessment.scenario_type,
                "fraud_score": a.fraud_risk.risk_score,
                "prediction_set": list(a.prediction_set),
                "review_priority": a.review_priority,
                "causal_reasoning": a.causal_reasoning[:200] if a.causal_reasoning else "",
                "n_assessments": len(result.assessments),
                "processing_time_sec": round(elapsed, 2),
                "gpu_mem_delta_gb": round(gpu_delta, 3) if gpu_delta is not None else None,
            }
        else:
            return {
                "success": True,
                "severity": None,
                "n_assessments": 0,
                "processing_time_sec": round(elapsed, 2),
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "processing_time_sec": round(time.time() - t0, 2),
        }


def compute_aggregate_stats(results: list[dict]) -> dict:
    """Compute aggregate statistics across all results."""
    successful = [r for r in results if r["result"]["success"] and r["result"].get("severity")]
    failed = [r for r in results if not r["result"]["success"]]

    stats = {
        "total_videos": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate_pct": round(100.0 * len(successful) / len(results), 1) if results else 0,
    }

    if successful:
        # Severity distribution
        sev_dist = {}
        for r in successful:
            sev = r["result"]["severity"]
            sev_dist[sev] = sev_dist.get(sev, 0) + 1
        stats["severity_distribution"] = sev_dist

        # Review priority distribution
        pri_dist = {}
        for r in successful:
            pri = r["result"].get("review_priority", "UNKNOWN")
            pri_dist[pri] = pri_dist.get(pri, 0) + 1
        stats["review_priority_distribution"] = pri_dist

        # Confidence stats
        confidences = [r["result"]["confidence"] for r in successful]
        stats["confidence"] = {
            "mean": round(float(sum(confidences) / len(confidences)), 3),
            "min": round(float(min(confidences)), 3),
            "max": round(float(max(confidences)), 3),
        }

        # Fraud score stats
        fraud_scores = [r["result"]["fraud_score"] for r in successful if r["result"].get("fraud_score") is not None]
        if fraud_scores:
            stats["fraud_score"] = {
                "mean": round(float(sum(fraud_scores) / len(fraud_scores)), 3),
                "min": round(float(min(fraud_scores)), 3),
                "max": round(float(max(fraud_scores)), 3),
            }

        # Processing time
        times = [r["result"]["processing_time_sec"] for r in successful]
        stats["processing_time_sec"] = {
            "mean": round(float(sum(times) / len(times)), 2),
            "min": round(float(min(times)), 2),
            "max": round(float(max(times)), 2),
            "total": round(float(sum(times)), 2),
        }

        # GPU memory (if tracked)
        gpu_deltas = [r["result"]["gpu_mem_delta_gb"] for r in successful if r["result"].get("gpu_mem_delta_gb") is not None]
        if gpu_deltas:
            stats["gpu_memory_delta_gb"] = {
                "mean": round(float(sum(gpu_deltas) / len(gpu_deltas)), 3),
                "max": round(float(max(gpu_deltas)), 3),
            }

    # Per-source breakdown
    by_source = {}
    for r in results:
        src = r["source"]
        if src not in by_source:
            by_source[src] = {"total": 0, "successful": 0}
        by_source[src]["total"] += 1
        if r["result"]["success"] and r["result"].get("severity"):
            by_source[src]["successful"] += 1
    stats["by_source"] = by_source

    # Ground truth accuracy (for labeled videos)
    gt_results = []
    for r in successful:
        gt = r.get("ground_truth", {})
        gt_sev = gt.get("severity")
        if gt_sev and gt_sev != "UNKNOWN":
            pred_sev = r["result"]["severity"]
            gt_results.append({
                "video_id": r["video_id"],
                "expected": gt_sev,
                "predicted": pred_sev,
                "match": pred_sev == gt_sev,
            })
    if gt_results:
        correct = sum(1 for g in gt_results if g["match"])
        stats["ground_truth_accuracy"] = {
            "total": len(gt_results),
            "correct": correct,
            "accuracy_pct": round(100.0 * correct / len(gt_results), 1),
            "details": gt_results,
        }

    return stats


def generate_markdown_summary(stats: dict, backend: str) -> str:
    """Generate markdown summary report."""
    lines = [
        "# Expanded Video Evaluation Report",
        "",
        f"**Backend**: {backend}",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Overview",
        "",
        f"- Total videos: {stats['total_videos']}",
        f"- Successful: {stats['successful']}",
        f"- Failed: {stats['failed']}",
        f"- Success rate: {stats['success_rate_pct']}%",
        "",
    ]

    if "severity_distribution" in stats:
        lines.append("## Severity Distribution\n")
        for sev, count in sorted(stats["severity_distribution"].items()):
            lines.append(f"- **{sev}**: {count}")
        lines.append("")

    if "review_priority_distribution" in stats:
        lines.append("## Review Priority Distribution\n")
        for pri, count in sorted(stats["review_priority_distribution"].items()):
            lines.append(f"- **{pri}**: {count}")
        lines.append("")

    if "confidence" in stats:
        c = stats["confidence"]
        lines.append(f"## Confidence: mean={c['mean']}, range=[{c['min']}, {c['max']}]\n")

    if "processing_time_sec" in stats:
        t = stats["processing_time_sec"]
        lines.append(f"## Processing Time: mean={t['mean']}s, total={t['total']}s\n")

    if "ground_truth_accuracy" in stats:
        gt = stats["ground_truth_accuracy"]
        lines.extend([
            "## Ground Truth Accuracy\n",
            f"- Total labeled: {gt['total']}",
            f"- Correct: {gt['correct']}",
            f"- Accuracy: {gt['accuracy_pct']}%",
            "",
            "| Video | Expected | Predicted | Match |",
            "|-------|----------|-----------|-------|",
        ])
        for d in gt["details"]:
            m = "PASS" if d["match"] else "FAIL"
            lines.append(f"| {d['video_id']} | {d['expected']} | {d['predicted']} | {m} |")
        lines.append("")

    if "by_source" in stats:
        lines.extend(["## By Source\n"])
        for src, data in stats["by_source"].items():
            lines.append(f"- **{src}**: {data['successful']}/{data['total']} successful")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Expanded real video evaluation")
    parser.add_argument("--backend", default="mock", choices=["mock", "real"])
    parser.add_argument("--output", default="reports/expanded_eval.json")
    parser.add_argument("--max-videos", type=int, default=None, help="Max videos total")
    parser.add_argument("--max-per-dir", type=int, default=None, help="Max videos per directory")
    parser.add_argument("--sources", nargs="+", default=None, help="Limit to specific sources")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Expanded Real Video Evaluation")
    print(f"Backend: {args.backend}")
    print("=" * 60)

    # Filter directories
    dirs = VIDEO_DIRS
    if args.sources:
        dirs = [(name, path) for name, path in dirs if name in args.sources]

    # Scan videos
    videos = scan_all_videos(dirs, max_per_dir=args.max_per_dir)
    if args.max_videos:
        videos = videos[:args.max_videos]

    print(f"\nFound {len(videos)} videos:")
    by_source = {}
    for v in videos:
        by_source.setdefault(v["source"], 0)
        by_source[v["source"]] += 1
    for src, count in by_source.items():
        print(f"  {src}: {count}")

    if not videos:
        print("ERROR: No videos found")
        return

    # Initialize pipeline
    backend = CosmosBackend.QWEN25VL if args.backend == "real" else CosmosBackend.MOCK
    config = PipelineConfig(output_dir=str(output_path.parent / "expanded_eval"))
    config.cosmos.backend = backend

    from insurance_mvp.pipeline import InsurancePipeline
    pipeline = InsurancePipeline(config)

    # GPU info
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Process all videos
    print(f"\nProcessing {len(videos)} videos...")
    all_results = []
    start = time.time()

    for video in tqdm(videos, desc="Evaluating"):
        result = run_pipeline_on_video(pipeline, video)
        all_results.append({
            "video_id": video["video_id"],
            "source": video["source"],
            "path": str(video["path"]),
            "ground_truth": video["ground_truth"],
            "result": result,
        })

    total_time = time.time() - start

    # Compute stats
    stats = compute_aggregate_stats(all_results)
    stats["total_time_sec"] = round(total_time, 2)
    stats["mean_time_per_video_sec"] = round(total_time / len(videos), 2) if videos else 0

    # Save JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": args.backend,
        "stats": stats,
        "results": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nJSON report: {output_path}")

    # Save Markdown
    md_path = output_path.with_suffix(".md")
    md = generate_markdown_summary(stats, args.backend)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown report: {md_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total: {stats['total_videos']} videos, {stats['successful']} successful")
    print(f"Time: {total_time:.1f}s total, {total_time/len(videos):.1f}s/video")
    if "severity_distribution" in stats:
        print(f"Severity: {stats['severity_distribution']}")
    if "ground_truth_accuracy" in stats:
        gt = stats["ground_truth_accuracy"]
        print(f"GT accuracy: {gt['accuracy_pct']}% ({gt['correct']}/{gt['total']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
