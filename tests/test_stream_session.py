"""Tests for StreamSession (vigil/stream_session.py)."""

from __future__ import annotations

import asyncio
import time

import pytest

from sopilot.vigil.stream_session import StreamSession, create_session


# ── Factory ───────────────────────────────────────────────────────────────────


def test_create_session_generates_uuid():
    s = create_session("test", [])
    assert len(s.session_id) == 36  # UUID4
    assert "-" in s.session_id


def test_create_session_stores_name_and_rules():
    s = create_session("cam1", ["ヘルメット未着用", "立入禁止エリア侵入"])
    assert s.name == "cam1"
    assert s.rules == ["ヘルメット未着用", "立入禁止エリア侵入"]


def test_create_session_unique_ids():
    ids = {create_session("x", []).session_id for _ in range(100)}
    assert len(ids) == 100


def test_create_session_initial_state():
    s = create_session("cam", [])
    assert s.frame_count == 0
    assert s.last_result is None
    assert s.last_frame_jpeg is None
    assert s.is_active is True


def test_created_at_is_recent():
    before = time.time()
    s = create_session("cam", [])
    after = time.time()
    assert before <= s.created_at <= after


# ── Active / Stop ─────────────────────────────────────────────────────────────


def test_session_is_active_initially():
    s = create_session("cam", [])
    assert s.is_active is True


def test_session_stop_sets_inactive():
    s = create_session("cam", [])
    s.stop()
    assert s.is_active is False


def test_session_stop_is_idempotent():
    s = create_session("cam", [])
    s.stop()
    s.stop()
    assert s.is_active is False


# ── Subscribe / Unsubscribe ───────────────────────────────────────────────────


def test_subscribe_returns_queue():
    s = create_session("cam", [])
    q = s.subscribe()
    assert q is not None
    assert s.to_dict()["subscribers"] == 1


def test_subscribe_multiple():
    s = create_session("cam", [])
    q1 = s.subscribe()
    q2 = s.subscribe()
    assert s.to_dict()["subscribers"] == 2


def test_unsubscribe_removes_queue():
    s = create_session("cam", [])
    q = s.subscribe()
    s.unsubscribe(q)
    assert s.to_dict()["subscribers"] == 0


def test_unsubscribe_unknown_queue_no_error():
    s = create_session("cam", [])
    import asyncio

    q = asyncio.Queue()
    s.unsubscribe(q)  # should not raise


# ── push_result ───────────────────────────────────────────────────────────────


def test_push_result_delivers_to_subscriber():
    s = create_session("cam", [])
    q = s.subscribe()
    payload = {"detections": [], "violations": [], "frame_ms": 5.0}
    s.push_result(payload)
    result = q.get_nowait()
    assert result == payload


def test_push_result_to_multiple_subscribers():
    s = create_session("cam", [])
    q1 = s.subscribe()
    q2 = s.subscribe()
    payload = {"x": 1}
    s.push_result(payload)
    assert q1.get_nowait() == payload
    assert q2.get_nowait() == payload


def test_push_result_drops_on_full_queue():
    s = create_session("cam", [])
    q = s.subscribe()
    # Fill the queue
    for i in range(200):
        try:
            q.put_nowait({"i": i})
        except asyncio.QueueFull:
            break
    # push_result should not raise even when queue is full
    s.push_result({"overflow": True})


# ── Stop sends sentinel ───────────────────────────────────────────────────────


def test_stop_sends_none_sentinel_to_subscribers():
    s = create_session("cam", [])
    q = s.subscribe()
    s.stop()
    sentinel = q.get_nowait()
    assert sentinel is None


# ── to_dict ───────────────────────────────────────────────────────────────────


def test_to_dict_fields():
    s = create_session("cam", ["rule1"])
    d = s.to_dict()
    assert "session_id" in d
    assert "name" in d
    assert "rules" in d
    assert "created_at" in d
    assert "frame_count" in d
    assert "is_active" in d
    assert "subscribers" in d
    assert "last_result" in d


def test_to_dict_last_result_initially_none():
    s = create_session("cam", [])
    assert s.to_dict()["last_result"] is None


def test_to_dict_reflects_frame_count():
    s = create_session("cam", [])
    s.frame_count = 42
    assert s.to_dict()["frame_count"] == 42
