from __future__ import annotations

from sopilot.queueing import _build_retry_intervals, InlineQueueManager


def test_build_retry_intervals_with_zero() -> None:
    assert _build_retry_intervals(0) == []
    assert _build_retry_intervals(-3) == []


def test_build_retry_intervals_within_default_window() -> None:
    assert _build_retry_intervals(1) == [10]
    assert _build_retry_intervals(2) == [10, 60]
    assert _build_retry_intervals(3) == [10, 60, 300]


def test_build_retry_intervals_extend_tail() -> None:
    assert _build_retry_intervals(4) == [10, 60, 300, 300]
    assert _build_retry_intervals(6) == [10, 60, 300, 300, 300, 300]


def test_inline_queue_metrics_shape() -> None:
    q = InlineQueueManager(
        ingest_handler=lambda _: None,
        score_handler=lambda _: None,
        training_handler=lambda _: None,
    )
    payload = q.metrics()
    assert payload["backend"] == "inline"
    assert isinstance(payload["queues"], list)
    assert len(payload["queues"]) == 3
