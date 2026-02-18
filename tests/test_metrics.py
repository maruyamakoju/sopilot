"""Tests for metrics module (P2-4).

Covers Prometheus metric helpers: counters, histograms, gauges, context managers.
"""

from sopilot.metrics import (
    PROMETHEUS_AVAILABLE,
    collect_gpu_metrics,
    increment_job_counter,
    set_build_info,
    track_dtw_duration,
    track_embedding_duration,
    track_job_duration,
    update_active_workers,
    update_gpu_memory,
    update_queue_depth,
)


class TestIncrementJobCounter:
    """Test increment_job_counter."""

    def test_ingest_completed(self):
        """Increment ingest completed counter."""
        increment_job_counter("ingest", "completed")

    def test_ingest_failed(self):
        """Increment ingest failed counter."""
        increment_job_counter("ingest", "failed")

    def test_score_completed(self):
        """Increment score completed counter."""
        increment_job_counter("score", "completed")

    def test_training_with_trigger(self):
        """Increment training counter with trigger."""
        increment_job_counter("training", "completed", trigger="manual")

    def test_training_nightly(self):
        """Increment training nightly trigger."""
        increment_job_counter("training", "completed", trigger="nightly")

    def test_unknown_job_type(self):
        """Unknown job type doesn't crash."""
        increment_job_counter("unknown", "completed")


class TestTrackJobDuration:
    """Test track_job_duration context manager."""

    def test_basic_tracking(self):
        """Context manager tracks duration."""
        with track_job_duration("ingest"):
            pass

    def test_tracking_with_exception(self):
        """Duration tracked even on exception."""
        import pytest

        with pytest.raises(ValueError):
            with track_job_duration("score"):
                raise ValueError("test")

    def test_different_job_types(self):
        """Multiple job types can be tracked."""
        with track_job_duration("ingest"):
            pass
        with track_job_duration("score"):
            pass
        with track_job_duration("training"):
            pass


class TestTrackDtwDuration:
    """Test track_dtw_duration context manager."""

    def test_cpu_tracking(self):
        """Track DTW on CPU."""
        with track_dtw_duration(use_gpu=False):
            pass

    def test_gpu_tracking(self):
        """Track DTW on GPU."""
        with track_dtw_duration(use_gpu=True):
            pass


class TestTrackEmbeddingDuration:
    """Test track_embedding_duration context manager."""

    def test_heuristic(self):
        """Track heuristic embedder."""
        with track_embedding_duration("heuristic"):
            pass

    def test_vjepa2(self):
        """Track vjepa2 embedder."""
        with track_embedding_duration("vjepa2"):
            pass


class TestUpdateFunctions:
    """Test gauge update functions."""

    def test_update_queue_depth(self):
        """Update queue depth gauge."""
        update_queue_depth("ingest", 5)
        update_queue_depth("score", 0)

    def test_update_gpu_memory(self):
        """Update GPU memory gauges."""
        update_gpu_memory(0, allocated=1024, reserved=2048, total=8192)

    def test_update_active_workers(self):
        """Update active worker count."""
        update_active_workers("ingest", 3)
        update_active_workers("score", 1)


class TestSetBuildInfo:
    """Test set_build_info function."""

    def test_version_only(self):
        """Set build info with version only."""
        set_build_info("1.0.0")

    def test_full_info(self):
        """Set build info with all fields."""
        set_build_info("1.0.0", commit="abc123", build_date="2026-02-18")

    def test_partial_info(self):
        """Set build info with commit only."""
        set_build_info("1.0.0", commit="def456")


class TestCollectGpuMetrics:
    """Test collect_gpu_metrics function."""

    def test_collect_no_crash(self):
        """Collecting GPU metrics doesn't crash (even without GPU)."""
        collect_gpu_metrics()
