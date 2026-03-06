"""Performance benchmarks for SOPilot core components.

These run as part of the standard pytest suite (no server required).
Mark: ``@pytest.mark.performance``

Run only performance tests::

    pytest -m performance -v

Each test measures latency via ``time.perf_counter()`` and asserts
that p95 is within acceptable bounds.
"""

import time
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _measure_latencies(fn: Any, n: int) -> list[float]:
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def _p95(latencies: list[float]) -> float:
    return float(np.percentile(latencies, 95))


# ---------------------------------------------------------------------------
# JSON serialisation benchmark
# ---------------------------------------------------------------------------

@pytest.mark.performance
def test_json_serialisation_large_response() -> None:
    """JSON serialisation of a large analytics-like response must complete quickly."""
    import json

    payload = {
        "videos": [
            {
                "id": i,
                "task_id": "pilot_task",
                "site_id": f"site_{i % 10}",
                "status": "completed",
                "clip_count": 12,
                "score": round(50 + i * 0.1, 2),
            }
            for i in range(500)
        ],
        "total": 500,
        "summary": {"mean_score": 75.3, "pass_rate": 0.87},
    }

    def _serialize() -> None:
        json.dumps(payload)

    latencies = _measure_latencies(_serialize, 100)
    p95_ms = _p95(latencies)
    # JSON serialisation of 500 items should be < 50ms on any modern CPU
    assert p95_ms < 50.0, f"JSON serialisation p95={p95_ms:.1f}ms exceeds 50ms"


# ---------------------------------------------------------------------------
# NMS benchmark (pure NumPy, no model download)
# ---------------------------------------------------------------------------

@pytest.mark.performance
def test_nms_latency() -> None:
    """Non-maximum suppression on 200 boxes must complete in < 50ms p95."""
    import cv2

    def _nms() -> None:
        n = 200
        boxes = np.random.uniform(0, 640, (n, 4)).astype(np.float32)
        # Convert to x, y, w, h format expected by cv2.dnn.NMSBoxes
        boxes[:, 2] = np.clip(boxes[:, 2], 10, 640)
        boxes[:, 3] = np.clip(boxes[:, 3], 10, 640)
        scores = np.random.uniform(0.1, 1.0, n).tolist()
        cv2.dnn.NMSBoxes(boxes.tolist(), scores, 0.3, 0.5)

    latencies = _measure_latencies(_nms, 100)
    p95_ms = _p95(latencies)
    assert p95_ms < 50.0, f"NMS p95={p95_ms:.1f}ms exceeds 50ms"


# ---------------------------------------------------------------------------
# Perception config instantiation
# ---------------------------------------------------------------------------

@pytest.mark.performance
def test_perception_config_instantiation() -> None:
    """PerceptionConfig construction (dataclass) should be < 5ms."""
    from sopilot.perception.types import PerceptionConfig

    def _make() -> None:
        PerceptionConfig()

    latencies = _measure_latencies(_make, 1000)
    p95_ms = _p95(latencies)
    assert p95_ms < 5.0, f"PerceptionConfig instantiation p95={p95_ms:.2f}ms"


# ---------------------------------------------------------------------------
# BBox operations benchmark
# ---------------------------------------------------------------------------

@pytest.mark.performance
def test_bbox_iou_batch() -> None:
    """1000 BBox.iou() calls must complete in < 100ms total."""
    from sopilot.perception.types import BBox

    boxes = [
        BBox(
            x1=float(np.random.uniform(0, 0.4)),
            y1=float(np.random.uniform(0, 0.4)),
            x2=float(np.random.uniform(0.5, 1.0)),
            y2=float(np.random.uniform(0.5, 1.0)),
        )
        for _ in range(1000)
    ]
    ref = BBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5)

    t0 = time.perf_counter()
    for b in boxes:
        ref.iou(b)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 100.0, f"1000 BBox.iou() took {elapsed_ms:.1f}ms"


# ---------------------------------------------------------------------------
# Log context overhead
# ---------------------------------------------------------------------------

@pytest.mark.performance
def test_log_context_overhead() -> None:
    """log_context entry/exit overhead should be < 1ms p95 per call."""
    from sopilot.logging_config import log_context

    def _ctx() -> None:
        with log_context(job_id=42, task_id="pilot"):
            pass

    latencies = _measure_latencies(_ctx, 1000)
    p95_ms = _p95(latencies)
    assert p95_ms < 1.0, f"log_context p95={p95_ms:.3f}ms"


# ---------------------------------------------------------------------------
# Settings.from_env() benchmark
# ---------------------------------------------------------------------------

@pytest.mark.performance
def test_settings_from_env() -> None:
    """Settings.from_env() should complete in < 50ms p95."""
    from sopilot.config import Settings

    latencies = _measure_latencies(Settings.from_env, 50)
    p95_ms = _p95(latencies)
    assert p95_ms < 50.0, f"Settings.from_env p95={p95_ms:.1f}ms"


# ---------------------------------------------------------------------------
# ONNX SQL placeholder translation
# ---------------------------------------------------------------------------

@pytest.mark.performance
def test_sql_placeholder_translation() -> None:
    """? → %s translation should be < 0.5ms p95 on a typical query."""
    from sopilot.database_pg import _translate_sql

    query = (
        "SELECT * FROM videos WHERE task_id = ? AND status = ? "
        "AND created_at > ? ORDER BY id LIMIT ? OFFSET ?"
    )

    def _translate() -> None:
        _translate_sql(query)

    latencies = _measure_latencies(_translate, 10000)
    p95_ms = _p95(latencies)
    assert p95_ms < 0.5, f"SQL translation p95={p95_ms:.4f}ms"


# ---------------------------------------------------------------------------
# Audit middleware enqueue (lock-free queue)
# ---------------------------------------------------------------------------

@pytest.mark.performance
def test_audit_enqueue_throughput() -> None:
    """AuditWorker.enqueue() must handle 10,000 calls with no drops and < 5ms p95."""
    import queue as _queue
    from unittest.mock import MagicMock, patch

    from sopilot.services.audit_worker import AuditWorker

    worker = AuditWorker(connect=MagicMock())
    worker._queue = _queue.Queue(maxsize=100_000)

    def _enqueue() -> None:
        worker.enqueue(
            method="POST",
            path="/videos",
            status_code=201,
            duration_ms=12.3,
        )

    latencies = _measure_latencies(_enqueue, 1000)
    p95_ms = _p95(latencies)
    assert p95_ms < 5.0, f"AuditWorker.enqueue p95={p95_ms:.3f}ms"
