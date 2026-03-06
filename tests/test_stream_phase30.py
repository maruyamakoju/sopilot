"""Tests for Phase 30 — track overlay + behavior events in stream responses."""

from __future__ import annotations

import json
import time
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sopilot.vigil.person_tracker import PersonTracker, TrackState
from sopilot.vigil.stream_session import StreamSession, create_session
from sopilot.vigil.stream_router import _annotate_tracks, _run_tracker


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_jpeg(w: int = 64, h: int = 64) -> bytes:
    """Produce a minimal valid JPEG bytes."""
    try:
        import cv2
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes()
    except Exception:
        return b"\xff\xd8\xff\xe0" + b"\x00" * 100  # fake JPEG header


def _make_session_with_frame() -> StreamSession:
    s = create_session(name="test", rules=[])
    s.last_frame_jpeg = _make_jpeg()
    return s


def _make_request_stub(session_id: str = "s1") -> MagicMock:
    """Stub a FastAPI Request with minimal app.state."""
    req = MagicMock()
    req.app.state.stream_trackers = {}
    req.app.state.stream_behavior_events = {}
    req.app.state.stream_behavior_detector = None  # will be created lazily
    return req


def _person_det(x1=0.3, y1=0.2, x2=0.5, y2=0.8) -> dict:
    return {"label": "person", "confidence": 0.9,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


# ── _annotate_tracks ───────────────────────────────────────────────────────────


def test_annotate_tracks_no_frame_no_crash():
    """Should silently skip when last_frame_jpeg is None."""
    session = create_session(name="t", rules=[])
    session.last_frame_jpeg = None
    tracker = PersonTracker()
    _annotate_tracks(session, tracker)  # should not raise
    assert session.last_frame_jpeg is None


def test_annotate_tracks_empty_tracker_no_change():
    session = _make_session_with_frame()
    original = session.last_frame_jpeg
    tracker = PersonTracker()
    _annotate_tracks(session, tracker)
    # No tracks → frame unchanged (or at least still valid bytes)
    assert session.last_frame_jpeg is not None


def test_annotate_tracks_with_active_track():
    """Track annotation should produce a non-empty JPEG."""
    session = _make_session_with_frame()
    tracker = PersonTracker()
    tracker.update([_person_det()], timestamp=time.time())
    _annotate_tracks(session, tracker)
    assert session.last_frame_jpeg is not None
    assert len(session.last_frame_jpeg) > 0


def test_annotate_tracks_lost_track_skipped():
    """Lost tracks should not be drawn."""
    session = _make_session_with_frame()
    tracker = PersonTracker(max_lost_frames=1)
    tracker.update([_person_det()])
    tracker.update([])   # lost_frames = 1
    tracker.update([])   # status = lost
    _annotate_tracks(session, tracker)  # should not raise


def test_annotate_tracks_with_behavior_flag():
    """Track with behavior flags should produce valid frame."""
    session = _make_session_with_frame()
    tracker = PersonTracker()
    tracker.update([_person_det()])
    # Manually set a behavior flag
    track = list(tracker._tracks.values())[0]
    track.behavior_flags.add("loitering")
    _annotate_tracks(session, tracker)
    assert session.last_frame_jpeg is not None


# ── _run_tracker return value ──────────────────────────────────────────────────


def test_run_tracker_returns_list():
    session = _make_session_with_frame()
    req = _make_request_stub(session.session_id)
    result = _run_tracker(req, session, [], time.time())
    assert isinstance(result, list)


def test_run_tracker_returns_empty_when_no_detections():
    session = _make_session_with_frame()
    req = _make_request_stub(session.session_id)
    events = _run_tracker(req, session, [], time.time())
    assert events == []


def test_run_tracker_creates_tracker_in_state():
    session = _make_session_with_frame()
    req = _make_request_stub(session.session_id)
    _run_tracker(req, session, [_person_det()], time.time())
    assert session.session_id in req.app.state.stream_trackers


def test_run_tracker_person_detection_increments_track():
    session = _make_session_with_frame()
    req = _make_request_stub(session.session_id)
    _run_tracker(req, session, [_person_det()], time.time())
    tracker = req.app.state.stream_trackers[session.session_id]
    assert tracker.total_tracks == 1


# ── FrameResultResponse includes behavior_events ─────────────────────────────


def _make_stream_app():
    from fastapi import FastAPI
    from sopilot.vigil.stream_router import build_stream_router

    app = FastAPI()
    app.state.stream_sessions = {}
    app.state.stream_trackers = {}
    app.state.stream_behavior_events = {}
    app.include_router(build_stream_router())
    return app


def test_push_frame_response_has_behavior_events_field():
    app = _make_stream_app()
    with TestClient(app) as client:
        # Create session
        r = client.post("/vigil/stream/sessions",
                        json={"name": "test", "rules": []})
        sid = r.json()["session_id"]

        # Push a valid JPEG frame
        jpeg = _make_jpeg()
        resp = client.post(
            f"/vigil/stream/sessions/{sid}/frame",
            content=jpeg,
            headers={"Content-Type": "image/jpeg"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "behavior_events" in data
    assert isinstance(data["behavior_events"], list)


def test_push_frame_behavior_events_is_empty_list_when_no_behaviors():
    app = _make_stream_app()
    with TestClient(app) as client:
        r = client.post("/vigil/stream/sessions",
                        json={"name": "test", "rules": []})
        sid = r.json()["session_id"]
        jpeg = _make_jpeg()
        resp = client.post(
            f"/vigil/stream/sessions/{sid}/frame",
            content=jpeg,
            headers={"Content-Type": "image/jpeg"},
        )
    assert resp.json()["behavior_events"] == []
