"""
Prometheus metrics for SOPilot monitoring.

Metrics exposed:
- sopilot_ingest_jobs_total: Counter of ingest jobs by status
- sopilot_score_jobs_total: Counter of score jobs by status
- sopilot_training_jobs_total: Counter of training jobs by status
- sopilot_job_duration_seconds: Histogram of job execution times
- sopilot_video_clips_total: Gauge of total clips indexed
- sopilot_queue_depth: Gauge of current queue depths
- sopilot_gpu_memory_bytes: Gauge of GPU memory usage (if available)
- sopilot_dtw_execution_seconds: Histogram of DTW computation time

Usage:
    from sopilot.metrics import (
        track_job_duration,
        increment_job_counter,
        update_queue_depth,
    )

    with track_job_duration("ingest", job_id="j123"):
        # Job execution
        pass

    increment_job_counter("score", status="completed")
"""
from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Any

try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None  # type: ignore
    Histogram = None  # type: ignore
    Gauge = None  # type: ignore
    Info = None  # type: ignore


# Counters
if PROMETHEUS_AVAILABLE:
    ingest_jobs_total = Counter(
        "sopilot_ingest_jobs_total",
        "Total number of ingest jobs",
        ["status"],  # queued, running, completed, failed
    )

    score_jobs_total = Counter(
        "sopilot_score_jobs_total",
        "Total number of score jobs",
        ["status"],
    )

    training_jobs_total = Counter(
        "sopilot_training_jobs_total",
        "Total number of training jobs",
        ["status", "trigger"],  # trigger: manual, nightly
    )

    # Histograms (for latency percentiles)
    job_duration_seconds = Histogram(
        "sopilot_job_duration_seconds",
        "Job execution duration in seconds",
        ["job_type"],  # ingest, score, training
        buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
    )

    dtw_execution_seconds = Histogram(
        "sopilot_dtw_execution_seconds",
        "DTW alignment computation time",
        ["use_gpu"],  # true, false
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    )

    embedding_generation_seconds = Histogram(
        "sopilot_embedding_generation_seconds",
        "Video embedding generation time",
        ["embedder_type"],  # heuristic, vjepa2
        buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
    )

    # Gauges (current state)
    queue_depth = Gauge(
        "sopilot_queue_depth",
        "Current depth of job queues",
        ["queue_name"],  # ingest, score, training
    )

    video_clips_total = Gauge(
        "sopilot_video_clips_total",
        "Total number of clips indexed",
        ["task_id"],
    )

    gpu_memory_bytes = Gauge(
        "sopilot_gpu_memory_bytes",
        "GPU memory usage in bytes",
        ["device_id", "memory_type"],  # memory_type: allocated, reserved, total
    )

    active_workers = Gauge(
        "sopilot_active_workers",
        "Number of active RQ workers",
        ["queue_name"],
    )

    # Info (metadata)
    build_info = Info(
        "sopilot_build",
        "Build and version information",
    )
else:
    # Dummy placeholders if prometheus not available
    ingest_jobs_total = None
    score_jobs_total = None
    training_jobs_total = None
    job_duration_seconds = None
    dtw_execution_seconds = None
    embedding_generation_seconds = None
    queue_depth = None
    video_clips_total = None
    gpu_memory_bytes = None
    active_workers = None
    build_info = None


def increment_job_counter(job_type: str, status: str, trigger: str | None = None) -> None:
    """
    Increment job counter for given type and status.

    Args:
        job_type: "ingest", "score", or "training"
        status: "queued", "running", "completed", "failed", "skipped"
        trigger: (training only) "manual" or "nightly"
    """
    if not PROMETHEUS_AVAILABLE:
        return

    if job_type == "ingest" and ingest_jobs_total is not None:
        ingest_jobs_total.labels(status=status).inc()
    elif job_type == "score" and score_jobs_total is not None:
        score_jobs_total.labels(status=status).inc()
    elif job_type == "training" and training_jobs_total is not None and trigger is not None:
        training_jobs_total.labels(status=status, trigger=trigger).inc()


@contextmanager
def track_job_duration(job_type: str, **labels):
    """
    Context manager to track job execution duration.

    Usage:
        with track_job_duration("ingest"):
            # Job execution
            pass
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        if PROMETHEUS_AVAILABLE and job_duration_seconds is not None:
            job_duration_seconds.labels(job_type=job_type).observe(duration)


@contextmanager
def track_dtw_duration(use_gpu: bool):
    """Track DTW computation time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        if PROMETHEUS_AVAILABLE and dtw_execution_seconds is not None:
            dtw_execution_seconds.labels(use_gpu=str(use_gpu).lower()).observe(duration)


@contextmanager
def track_embedding_duration(embedder_type: str):
    """Track embedding generation time."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        if PROMETHEUS_AVAILABLE and embedding_generation_seconds is not None:
            embedding_generation_seconds.labels(embedder_type=embedder_type).observe(duration)


def update_queue_depth(queue_name: str, depth: int) -> None:
    """Update queue depth gauge."""
    if PROMETHEUS_AVAILABLE and queue_depth is not None:
        queue_depth.labels(queue_name=queue_name).set(depth)


def update_gpu_memory(device_id: int, allocated: int, reserved: int, total: int) -> None:
    """Update GPU memory gauges."""
    if PROMETHEUS_AVAILABLE and gpu_memory_bytes is not None:
        gpu_memory_bytes.labels(device_id=str(device_id), memory_type="allocated").set(allocated)
        gpu_memory_bytes.labels(device_id=str(device_id), memory_type="reserved").set(reserved)
        gpu_memory_bytes.labels(device_id=str(device_id), memory_type="total").set(total)


def update_active_workers(queue_name: str, count: int) -> None:
    """Update active worker count."""
    if PROMETHEUS_AVAILABLE and active_workers is not None:
        active_workers.labels(queue_name=queue_name).set(count)


def set_build_info(version: str, commit: str | None = None, build_date: str | None = None) -> None:
    """Set build metadata."""
    if PROMETHEUS_AVAILABLE and build_info is not None:
        info = {"version": version}
        if commit:
            info["commit"] = commit
        if build_date:
            info["build_date"] = build_date
        build_info.info(info)


def collect_gpu_metrics() -> None:
    """Collect current GPU memory metrics (if CUDA available)."""
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        import torch
        if not torch.cuda.is_available():
            return

        for device_id in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            total = torch.cuda.get_device_properties(device_id).total_memory
            update_gpu_memory(device_id, allocated, reserved, total)
    except Exception:
        pass  # Ignore GPU metrics collection errors
