"""Tests for tracker router (vigil/tracker_router.py) — Phase 29."""

from __future__ import annotations

import time
from collections import deque

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sopilot.vigil.behavior_detector import BehaviorEvent
from sopilot.vigil.person_tracker import PersonTracker, TrackState
from sopilot.vigil.tracker_router import build_tracker_router


# ── App fixture ────────────────────────────────────────────────────────────────


def _make_app() -> FastAPI:
    app = FastAPI()
    app.state.stream_trackers = {}
    app.state.stream_behavior_events = {}
    app.include_router(build_tracker_router())
    return app


@pytest.fixture
def client():
    app = _make_app()
    with TestClient(app) as c:
        yield c


# ── Helpers ────────────────────────────────────────────────────────────────────


def _add_track(app: FastAPI, session_id: str) -> str:
    """Inject one person detection into the tracker for a session."""
    tracker = PersonTracker()
    det = {"label": "person", "confidence": 0.9,
           "bbox": {"x1": 0.4, "y1": 0.3, "x2": 0.6, "y2": 0.9}}
    tracker.update([det], timestamp=time.time())
    app.state.stream_trackers[session_id] = tracker
    return list(tracker._tracks.keys())[0]


def _add_event(app: FastAPI, session_id: str, btype: str = "loitering") -> None:
    ev = BehaviorEvent(
        event_id="ev-1", track_id="t1", session_id=session_id,
        behavior_type=btype, severity="warning",
        confidence=0.8, timestamp=time.time(),
        description="テスト滞留",
    )
    events = app.state.stream_behavior_events.setdefault(session_id, [])
    events.append(ev.to_dict())


# ── GET /{session_id}/tracks ───────────────────────────────────────────────────


def test_get_tracks_empty_session(client):
    resp = client.get("/vigil/tracker/no-session/tracks")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tracks"] == []
    assert data["total"] == 0


def test_get_tracks_with_data():
    app = _make_app()
    _add_track(app, "cam1")
    with TestClient(app) as c:
        resp = c.get("/vigil/tracker/cam1/tracks")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["session_id"] == "cam1"


def test_get_tracks_response_fields():
    app = _make_app()
    _add_track(app, "cam1")
    with TestClient(app) as c:
        resp = c.get("/vigil/tracker/cam1/tracks")
    track = resp.json()["tracks"][0]
    for key in ("track_id", "label", "status", "age_seconds", "sightings",
                "behavior_flags"):
        assert key in track


def test_get_tracks_include_lost():
    app = _make_app()
    tracker = PersonTracker(max_lost_frames=1)
    det = {"label": "person", "confidence": 0.9,
           "bbox": {"x1": 0.4, "y1": 0.3, "x2": 0.6, "y2": 0.9}}
    tracker.update([det])
    tracker.update([])   # lost_frames = 1
    tracker.update([])   # lost_frames = 2 → lost
    app.state.stream_trackers["cam1"] = tracker
    with TestClient(app) as c:
        resp_active = c.get("/vigil/tracker/cam1/tracks")
        resp_all = c.get("/vigil/tracker/cam1/tracks?include_lost=true")
    assert resp_active.json()["total"] == 0
    assert resp_all.json()["total"] == 1


# ── GET /{session_id}/events ───────────────────────────────────────────────────


def test_get_events_empty(client):
    resp = client.get("/vigil/tracker/no-session/events")
    assert resp.status_code == 200
    data = resp.json()
    assert data["events"] == []
    assert data["total"] == 0


def test_get_events_populated():
    app = _make_app()
    _add_event(app, "cam1", "loitering")
    with TestClient(app) as c:
        resp = c.get("/vigil/tracker/cam1/events")
    data = resp.json()
    assert data["total"] == 1
    assert data["events"][0]["behavior_type"] == "loitering"


def test_get_events_filter_by_type():
    app = _make_app()
    _add_event(app, "cam1", "loitering")
    _add_event(app, "cam1", "running")
    with TestClient(app) as c:
        resp = c.get("/vigil/tracker/cam1/events?behavior_type=running")
    evs = resp.json()["events"]
    assert all(e["behavior_type"] == "running" for e in evs)


def test_get_events_limit():
    app = _make_app()
    for _ in range(10):
        _add_event(app, "cam1", "loitering")
    with TestClient(app) as c:
        resp = c.get("/vigil/tracker/cam1/events?limit=3")
    assert len(resp.json()["events"]) <= 3


# ── GET /{session_id}/stats ────────────────────────────────────────────────────


def test_get_stats_empty(client):
    resp = client.get("/vigil/tracker/no-session/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_tracks"] == 0
    assert data["total_tracks"] == 0
    assert data["total_behavior_events"] == 0


def test_get_stats_with_data():
    app = _make_app()
    _add_track(app, "cam1")
    _add_event(app, "cam1", "loitering")
    _add_event(app, "cam1", "fall")
    with TestClient(app) as c:
        resp = c.get("/vigil/tracker/cam1/stats")
    data = resp.json()
    assert data["active_tracks"] == 1
    assert data["total_behavior_events"] == 2
    assert data["events_by_type"].get("loitering") == 1
    assert data["events_by_type"].get("fall") == 1


def test_get_stats_session_id_in_response():
    app = _make_app()
    with TestClient(app) as c:
        resp = c.get("/vigil/tracker/mySession/stats")
    assert resp.json()["session_id"] == "mySession"


# ── DELETE /{session_id} ───────────────────────────────────────────────────────


def test_delete_tracker_clears_tracks():
    app = _make_app()
    _add_track(app, "cam1")
    _add_event(app, "cam1")
    with TestClient(app) as c:
        resp = c.delete("/vigil/tracker/cam1")
    data = resp.json()
    assert resp.status_code == 200
    assert data["status"] == "cleared"
    assert data["tracks_removed"] >= 1
    assert data["events_removed"] >= 1


def test_delete_nonexistent_session(client):
    resp = client.delete("/vigil/tracker/no-session")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tracks_removed"] == 0
    assert data["events_removed"] == 0


def test_delete_clears_state():
    app = _make_app()
    _add_track(app, "cam1")
    with TestClient(app) as c:
        c.delete("/vigil/tracker/cam1")
        resp = c.get("/vigil/tracker/cam1/tracks")
    assert resp.json()["total"] == 0
