"""
End-to-End Pipeline Benchmark

Measures full ingest→score→training pipeline performance.

Usage:
    python benchmarks/benchmark_end_to_end.py --temp-dir /path/to/temp

This creates:
- Synthetic video files
- Runs ingest jobs
- Runs scoring jobs
- Measures total latency and throughput

Warning: This is a heavy benchmark that exercises the full system.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np


def create_synthetic_video(
    output_path: Path,
    duration_sec: float = 10.0,
    fps: int = 30,
    resolution: tuple[int, int] = (640, 480),
) -> None:
    """Create a synthetic video file for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)

    num_frames = int(duration_sec * fps)
    for i in range(num_frames):
        # Generate a simple gradient pattern that changes over time
        frame = np.zeros((*resolution[::-1], 3), dtype=np.uint8)
        intensity = int((i / num_frames) * 255)
        frame[:, :] = [intensity, 128, 255 - intensity]

        # Add some text
        cv2.putText(
            frame,
            f"Frame {i}/{num_frames}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

    writer.release()


def benchmark_ingest_pipeline(
    video_path: Path,
    data_dir: Path,
    embedder_backend: str = "heuristic",
) -> dict:
    """Benchmark video ingest (processing + embedding generation)."""
    from sopilot.config import Settings
    from sopilot.db import Database
    from sopilot.service import SopilotService

    # Create settings
    settings = Settings(
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        embeddings_dir=data_dir / "embeddings",
        reports_dir=data_dir / "reports",
        index_dir=data_dir / "index",
        models_dir=data_dir / "models",
        db_path=data_dir / "sopilot.db",
        target_fps=4,
        clip_seconds=4.0,
        max_side=320,
        min_clip_coverage=0.6,
        ingest_embed_batch_size=8,
        upload_max_mb=1024,
        min_scoring_clips=1,
        change_threshold_factor=1.0,
        min_step_clips=2,
        low_similarity_threshold=0.75,
        w_miss=12.0,
        w_swap=8.0,
        w_dev=30.0,
        w_time=15.0,
        w_warp=12.0,
        embedder_backend=embedder_backend,
        embedder_fallback_enabled=True,
        embedding_device="auto",
        vjepa2_repo="facebookresearch/vjepa2",
        vjepa2_variant="vjepa2_vit_large",
        vjepa2_pretrained=True,
        vjepa2_source="hub",
        vjepa2_local_repo="",
        vjepa2_local_checkpoint="",
        vjepa2_num_frames=64,
        vjepa2_image_size=256,
        vjepa2_batch_size=8,
        queue_backend="inline",
        redis_url="redis://127.0.0.1:6379/0",
        rq_queue_prefix="bench",
        rq_job_timeout_sec=21600,
        rq_result_ttl_sec=0,
        rq_failure_ttl_sec=604800,
        rq_retry_max=2,
        score_worker_threads=1,
        training_worker_threads=1,
        nightly_enabled=False,
        nightly_hour_local=2,
        nightly_min_new_videos=10,
        nightly_check_interval_sec=30,
        adapt_command="",
        adapt_timeout_sec=14400,
        enable_feature_adapter=False,  # Disable for benchmark simplicity
        report_title="Benchmark",
        auth_required=False,
        api_token="",
        api_token_role="admin",
        api_role_tokens="",
        basic_user="",
        basic_password="",
        basic_role="admin",
        auth_default_role="admin",
        audit_signing_key="",
        audit_signing_key_id="bench",
        privacy_mask_enabled=False,
        privacy_mask_mode="black",
        privacy_mask_rects="",
        privacy_face_blur=False,
        watch_enabled=False,
        watch_dir=data_dir / "watch",
        watch_poll_sec=5,
        watch_task_id="",
        watch_role="trainee",
        dtw_use_gpu=True,
    )

    # Ensure directories
    for d in [settings.raw_dir, settings.embeddings_dir, settings.reports_dir,
              settings.index_dir, settings.models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    db = Database(settings.db_path)
    service = SopilotService(settings, db, runtime_mode="api")

    try:
        # Enqueue ingest
        start = time.perf_counter()
        result = service.enqueue_ingest_from_path(
            file_name="benchmark_video.mp4",
            staged_path=video_path,
            task_id="bench_task",
            role="gold",
            requested_by="benchmark",
        )
        enqueue_time = time.perf_counter() - start

        job_id = result["ingest_job_id"]

        # Wait for completion (inline queue executes immediately)
        start = time.perf_counter()
        final_result = service.get_ingest_job(job_id)
        processing_time = time.perf_counter() - start

        return {
            "embedder": embedder_backend,
            "enqueue_time_sec": enqueue_time,
            "processing_time_sec": processing_time,
            "total_time_sec": enqueue_time + processing_time,
            "status": final_result["status"] if final_result else "unknown",
            "num_clips": final_result.get("num_clips", 0) if final_result else 0,
            "video_id": final_result.get("video_id", 0) if final_result else 0,
        }
    finally:
        service.shutdown()


def run_benchmark_suite(temp_dir: Path | None = None):
    """Run full end-to-end benchmark."""
    print("=" * 80)
    print("End-to-End Pipeline Benchmark")
    print("=" * 80)

    if temp_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = Path(temp_dir_obj.name)
    else:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTemp directory: {temp_dir}")

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "test_cases": [],
    }

    # Create synthetic videos
    print("\nCreating synthetic videos...")
    videos = []
    for duration in [5, 10, 20]:
        video_path = temp_dir / f"video_{duration}s.mp4"
        print(f"  Creating {duration}s video...")
        create_synthetic_video(video_path, duration_sec=duration)
        videos.append((video_path, duration))

    # Benchmark ingest with different embedders
    for embedder in ["heuristic"]:  # Add "vjepa2" if GPU available
        print(f"\n[{embedder}] Testing ingest pipeline...")
        print("-" * 80)

        for video_path, duration in videos:
            data_dir = temp_dir / f"data_{embedder}_{duration}s"
            print(f"  Processing {duration}s video...")

            try:
                result = benchmark_ingest_pipeline(video_path, data_dir, embedder)
                print(f"    Total time: {result['total_time_sec']:.2f}s")
                print(f"    Clips generated: {result['num_clips']}")
                results["test_cases"].append({
                    "pipeline": "ingest",
                    "video_duration_sec": duration,
                    **result,
                })
            except Exception as e:
                print(f"    ERROR: {e}")
                results["test_cases"].append({
                    "pipeline": "ingest",
                    "embedder": embedder,
                    "video_duration_sec": duration,
                    "error": str(e),
                })

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"end_to_end_benchmark_{timestamp}.json"

    with output_file.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)

    # Summary
    print("\nSummary:")
    print(f"{'Embedder':<15} {'Video (s)':<12} {'Total (s)':<12} {'Clips':<10}")
    print("-" * 50)
    for case in results["test_cases"]:
        if "error" in case:
            continue
        embedder = case.get("embedder", "?")
        duration = case.get("video_duration_sec", 0)
        total = case.get("total_time_sec", 0)
        clips = case.get("num_clips", 0)
        print(f"{embedder:<15} {duration:<12} {total:<12.2f} {clips:<10}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline benchmark")
    parser.add_argument("--temp-dir", type=str, help="Temporary directory for test data")
    args = parser.parse_args()

    temp_dir = Path(args.temp_dir) if args.temp_dir else None
    run_benchmark_suite(temp_dir)
