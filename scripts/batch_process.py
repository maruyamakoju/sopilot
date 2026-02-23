#!/usr/bin/env python3
"""Batch process all videos in a folder through the insurance pipeline.

Scans for video files, runs each through InsurancePipeline, and produces
a JSON summary report with per-video and aggregate statistics.

Usage:
    python scripts/batch_process.py --input-dir data/videos --output-dir results/batch
    python scripts/batch_process.py --input-dir data/videos --backend mock --parallel 4
    python scripts/batch_process.py --input-dir data/videos --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from insurance_mvp.config import CosmosBackend, PipelineConfig, load_config  # noqa: E402
from insurance_mvp.pipeline import InsurancePipeline  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

_EMPTY_FIELDS = {"severity": None, "confidence": None, "fault_ratio": None,
                 "fraud_score": None, "num_assessments": 0}


def get_gpu_memory() -> dict | None:
    """Return GPU memory stats if torch + CUDA are available."""
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda
            return {
                "allocated_mb": round(mem.memory_allocated() / 1048576, 1),
                "reserved_mb": round(mem.memory_reserved() / 1048576, 1),
                "max_allocated_mb": round(mem.max_memory_allocated() / 1048576, 1),
                "device": mem.get_device_name(0),
            }
    except ImportError:
        pass
    return None


def scan_videos(input_dir: Path) -> list[Path]:
    """Find all video files in the input directory (non-recursive)."""
    return [p for p in sorted(input_dir.iterdir())
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]


def process_one(pipeline: InsurancePipeline, video_path: Path) -> dict:
    """Process a single video and return a result dict."""
    video_id = video_path.stem
    t0 = time.time()
    try:
        result = pipeline.process_video(str(video_path), video_id=video_id)
        elapsed = time.time() - t0
        entry: dict = {"video_id": video_id, "file": video_path.name,
                        "success": result.success,
                        "processing_time_sec": round(elapsed, 3),
                        "error": result.error_message}
        if result.success and result.assessments:
            a = result.assessments[0]
            entry.update(severity=a.severity,
                         confidence=round(a.confidence, 3),
                         fault_ratio=round(a.fault_assessment.fault_ratio, 1),
                         fraud_score=round(a.fraud_risk.risk_score, 3),
                         num_assessments=len(result.assessments))
        else:
            entry.update(_EMPTY_FIELDS)
        return entry
    except Exception as e:
        return {"video_id": video_id, "file": video_path.name, "success": False,
                "processing_time_sec": round(time.time() - t0, 3),
                "error": str(e), **_EMPTY_FIELDS}


def build_summary(per_video: list[dict], total_time: float) -> dict:
    """Compute aggregate statistics from per-video results."""
    total = len(per_video)
    successful = [v for v in per_video if v["success"]]
    failed = [v for v in per_video if not v["success"]]
    severity_counts = Counter(v["severity"] for v in successful if v["severity"])
    confidences = [v["confidence"] for v in successful if v["confidence"] is not None]
    return {
        "total_videos": total,
        "successful": len(successful),
        "failed": len(failed),
        "severity_distribution": dict(severity_counts),
        "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "total_processing_time_sec": round(total_time, 2),
        "avg_time_per_video_sec": round(total_time / total, 2) if total else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch process videos through the insurance pipeline")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder with video files")
    parser.add_argument("--output-dir", type=str, default="results/batch", help="Output directory")
    parser.add_argument("--backend", type=str, default="mock", choices=["mock", "real"],
                        help="VLM backend: mock (fast) or real (Qwen2.5-VL)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")
    parser.add_argument("--max-videos", type=int, default=None, help="Process at most N videos")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = scan_videos(input_dir)
    if not videos:
        print(f"ERROR: No video files ({', '.join(VIDEO_EXTENSIONS)}) found in {input_dir}")
        sys.exit(1)
    if args.max_videos:
        videos = videos[: args.max_videos]

    print("=" * 60)
    print("Insurance MVP  --  Batch Processing")
    print("=" * 60)
    print(f"Input dir:   {input_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Backend:     {args.backend}")
    print(f"Workers:     {args.parallel}")
    print(f"Videos:      {len(videos)}")

    # Build config
    config = load_config(yaml_path=args.config) if args.config else PipelineConfig()
    config.output_dir = str(output_dir)
    config.parallel_workers = args.parallel
    config.continue_on_error = True
    config.cosmos.backend = CosmosBackend.MOCK if args.backend == "mock" else CosmosBackend.QWEN25VL

    gpu_before = get_gpu_memory()
    if gpu_before:
        print(f"GPU:         {gpu_before['device']}  (allocated {gpu_before['allocated_mb']} MB)")

    pipeline = InsurancePipeline(config)

    print(f"\nProcessing {len(videos)} videos...\n")
    per_video: list[dict] = []
    wall_start = time.time()

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **_kw):  # type: ignore[misc]
            return iterable

    def _log(entry: dict) -> None:
        status = "OK" if entry["success"] else "FAIL"
        sev = entry.get("severity") or "-"
        print(f"  [{status}] {entry['file']}  severity={sev}  time={entry['processing_time_sec']}s")

    if args.parallel <= 1:
        for vp in tqdm(videos, desc="Processing", unit="video"):
            entry = process_one(pipeline, vp)
            per_video.append(entry)
            _log(entry)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(process_one, pipeline, vp): vp for vp in videos}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="video"):
                entry = future.result()
                per_video.append(entry)
                _log(entry)

    wall_time = time.time() - wall_start
    gpu_after = get_gpu_memory()
    summary = build_summary(per_video, wall_time)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"input_dir": str(input_dir), "output_dir": str(output_dir),
                    "backend": args.backend, "parallel_workers": args.parallel},
        "summary": summary,
        "gpu_memory": {"before": gpu_before, "after": gpu_after},
        "videos": per_video,
    }
    report_path = output_dir / "batch_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # Console summary
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Total:       {summary['total_videos']}")
    print(f"Successful:  {summary['successful']}")
    print(f"Failed:      {summary['failed']}")
    print(f"Severity:    {summary['severity_distribution']}")
    print(f"Avg conf:    {summary['avg_confidence']}")
    print(f"Wall time:   {summary['total_processing_time_sec']}s")
    print(f"Avg/video:   {summary['avg_time_per_video_sec']}s")
    if gpu_after:
        print(f"GPU peak:    {gpu_after['max_allocated_mb']} MB")
    print(f"\nReport saved: {report_path}")
    print("=" * 60)

    if summary["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
