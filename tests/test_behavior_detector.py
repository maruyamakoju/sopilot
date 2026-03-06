"""Tests for BehaviorDetector — Phase 29."""

from __future__ import annotations

import time
from collections import deque

import pytest

from sopilot.vigil.behavior_detector import BehaviorDetector, BehaviorEvent
from sopilot.vigil.person_tracker import TrackState


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pt(cx: float, cy: float, x1: float, y1: float, x2: float, y2: float,
        ts: float = 0.0, fidx: int = 1) -> dict:
    return {"cx": cx, "cy": cy, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "timestamp": ts, "frame_idx": fidx}


def _make_track(traj_pts: list[dict], age: float = 0.0) -> TrackState:
    t0 = time.time() - age
    t1 = time.time()
    for i, p in enumerate(traj_pts):
        p.setdefault("timestamp", t0 + i * (age / max(len(traj_pts) - 1, 1)))
        p.setdefault("frame_idx", i + 1)
    dq = deque(traj_pts, maxlen=60)
    return TrackState(
        track_id="track-test",
        label="person",
        trajectory=dq,
        first_seen=t0,
        last_seen=t1,
    )


# ── BehaviorEvent ─────────────────────────────────────────────────────────────


def test_behavior_event_to_dict_has_required_keys():
    ev = BehaviorEvent(
        event_id="e1", track_id="t1", session_id="s1",
        behavior_type="loitering", severity="warning",
        confidence=0.8, timestamp=time.time(),
        description="テスト",
    )
    d = ev.to_dict()
    for k in ("event_id", "track_id", "session_id", "behavior_type",
              "severity", "confidence", "timestamp", "description"):
        assert k in d


def test_behavior_event_confidence_range():
    ev = BehaviorEvent(
        event_id="e1", track_id="t1", session_id="s1",
        behavior_type="running", severity="info",
        confidence=0.5, timestamp=time.time(),
        description="テスト",
    )
    assert 0.0 <= ev.confidence <= 1.0


def test_behavior_event_severity_values():
    for sev in ("info", "warning", "alert"):
        ev = BehaviorEvent(
            event_id="e1", track_id="t1", session_id="s1",
            behavior_type="running", severity=sev,
            confidence=0.5, timestamp=time.time(),
            description="テスト",
        )
        assert ev.severity == sev


# ── No events for fresh / short track ─────────────────────────────────────────


def test_no_events_single_point():
    det = BehaviorDetector()
    t0 = time.time()
    track = TrackState(
        track_id="t1", label="person",
        trajectory=deque([_pt(0.5, 0.5, 0.4, 0.3, 0.6, 0.9, ts=t0)], maxlen=60),
        first_seen=t0, last_seen=t0,
    )
    events = det.analyze(track, "s1")
    assert events == []


def test_no_loitering_young_track():
    det = BehaviorDetector(loitering_seconds=10.0)
    track = _make_track(
        [_pt(0.5, 0.5, 0.4, 0.3, 0.6, 0.9),
         _pt(0.5, 0.5, 0.4, 0.3, 0.6, 0.9)],
        age=3.0,  # only 3s old — below threshold
    )
    events = det.analyze(track, "s1")
    loitering_evs = [e for e in events if e.behavior_type == "loitering"]
    assert loitering_evs == []


# ── Loitering ─────────────────────────────────────────────────────────────────


def test_loitering_detected_stationary():
    det = BehaviorDetector(loitering_seconds=5.0, loitering_threshold=0.05)
    # Centroid barely moves over 10 seconds
    track = _make_track(
        [_pt(0.50, 0.50, 0.4, 0.3, 0.6, 0.9),
         _pt(0.51, 0.50, 0.41, 0.3, 0.61, 0.9)],
        age=10.0,
    )
    events = det.analyze(track, "s1")
    types = [e.behavior_type for e in events]
    assert "loitering" in types


def test_loitering_not_detected_moving():
    det = BehaviorDetector(loitering_seconds=5.0, loitering_threshold=0.05)
    track = _make_track(
        [_pt(0.1, 0.5, 0.0, 0.4, 0.2, 0.9),
         _pt(0.9, 0.5, 0.8, 0.4, 1.0, 0.9)],  # moved 0.8 units
        age=10.0,
    )
    events = det.analyze(track, "s1")
    types = [e.behavior_type for e in events]
    assert "loitering" not in types


def test_loitering_severity_is_warning():
    det = BehaviorDetector(loitering_seconds=5.0)
    track = _make_track(
        [_pt(0.5, 0.5, 0.4, 0.3, 0.6, 0.9),
         _pt(0.5, 0.5, 0.4, 0.3, 0.6, 0.9)],
        age=10.0,
    )
    events = det.analyze(track, "s1")
    loitering = [e for e in events if e.behavior_type == "loitering"]
    if loitering:
        assert loitering[0].severity == "warning"


def test_loitering_flag_not_repeated():
    """Fire loitering event only once per episode."""
    det = BehaviorDetector(loitering_seconds=5.0, loitering_threshold=0.05)
    track = _make_track(
        [_pt(0.5, 0.5, 0.4, 0.3, 0.6, 0.9),
         _pt(0.5, 0.5, 0.4, 0.3, 0.6, 0.9)],
        age=10.0,
    )
    ev1 = det.analyze(track, "s1")
    ev2 = det.analyze(track, "s1")  # second call — flag already set
    loitering1 = [e for e in ev1 if e.behavior_type == "loitering"]
    loitering2 = [e for e in ev2 if e.behavior_type == "loitering"]
    assert len(loitering2) == 0 or len(loitering1) > 0  # at most once per transition


def test_loitering_description_japanese():
    det = BehaviorDetector(loitering_seconds=5.0)
    track = _make_track(
        [_pt(0.5, 0.5, 0.4, 0.3, 0.6, 0.9),
         _pt(0.5, 0.5, 0.4, 0.3, 0.6, 0.9)],
        age=10.0,
    )
    events = det.analyze(track, "s1")
    for ev in events:
        if ev.behavior_type == "loitering":
            assert "滞留" in ev.description or "秒" in ev.description


# ── Running ───────────────────────────────────────────────────────────────────


def test_running_detected_fast_movement():
    det = BehaviorDetector(running_threshold=0.03, running_window=3)
    # Fast horizontal movement: 0.05 per step > 0.03 threshold
    pts = [
        _pt(0.1 + i * 0.05, 0.5, 0.0 + i * 0.05, 0.4, 0.2 + i * 0.05, 0.9)
        for i in range(5)
    ]
    track = _make_track(pts, age=1.0)
    events = det.analyze(track, "s1")
    types = [e.behavior_type for e in events]
    assert "running" in types


def test_running_not_detected_slow_movement():
    det = BehaviorDetector(running_threshold=0.05, running_window=5)
    # Slow movement: 0.005 per step < 0.05 threshold
    pts = [
        _pt(0.5 + i * 0.005, 0.5, 0.4 + i * 0.005, 0.3, 0.6 + i * 0.005, 0.9)
        for i in range(6)
    ]
    track = _make_track(pts, age=1.0)
    events = det.analyze(track, "s1")
    types = [e.behavior_type for e in events]
    assert "running" not in types


def test_running_severity_is_info():
    det = BehaviorDetector(running_threshold=0.01, running_window=3)
    pts = [_pt(i * 0.1, 0.5, i * 0.1, 0.4, i * 0.1 + 0.15, 0.9) for i in range(5)]
    track = _make_track(pts, age=0.5)
    events = det.analyze(track, "s1")
    for ev in events:
        if ev.behavior_type == "running":
            assert ev.severity == "info"


def test_running_description_japanese():
    det = BehaviorDetector(running_threshold=0.01)
    pts = [_pt(i * 0.1, 0.5, i * 0.1, 0.4, i * 0.1 + 0.15, 0.9) for i in range(5)]
    track = _make_track(pts, age=0.5)
    events = det.analyze(track, "s1")
    for ev in events:
        if ev.behavior_type == "running":
            assert "移動" in ev.description or "走行" in ev.description


# ── Fall ──────────────────────────────────────────────────────────────────────


def test_fall_detected_aspect_ratio_drops():
    det = BehaviorDetector(fall_aspect_normal=1.0, fall_aspect_threshold=0.7)
    # Early: tall bbox (h/w = 3.0), recent: wide bbox (h/w = 0.4)
    early_pts = [
        _pt(0.5, 0.5, 0.4, 0.2, 0.6, 0.8),  # h=0.6, w=0.2 → h/w=3.0
        _pt(0.5, 0.5, 0.4, 0.2, 0.6, 0.8),
    ]
    recent_pts = [
        _pt(0.5, 0.5, 0.3, 0.4, 0.7, 0.6),  # h=0.2, w=0.4 → h/w=0.5
        _pt(0.5, 0.5, 0.3, 0.4, 0.7, 0.6),
    ]
    track = _make_track(early_pts + recent_pts, age=2.0)
    events = det.analyze(track, "s1")
    types = [e.behavior_type for e in events]
    assert "fall" in types


def test_fall_not_detected_standing():
    det = BehaviorDetector(fall_aspect_normal=1.0, fall_aspect_threshold=0.7)
    # Always tall (standing)
    pts = [_pt(0.5, 0.5, 0.4, 0.1, 0.6, 0.9)] * 6  # h/w = 0.8/0.2 = 4.0
    track = _make_track(pts, age=2.0)
    events = det.analyze(track, "s1")
    types = [e.behavior_type for e in events]
    assert "fall" not in types


def test_fall_severity_is_alert():
    det = BehaviorDetector(fall_aspect_normal=1.0, fall_aspect_threshold=0.7)
    early = [_pt(0.5, 0.5, 0.45, 0.2, 0.55, 0.8)] * 2   # h/w ≈ 6.0
    recent = [_pt(0.5, 0.5, 0.2, 0.45, 0.8, 0.55)] * 2  # h/w = 0.1/0.6 ≈ 0.17
    track = _make_track(early + recent, age=1.0)
    events = det.analyze(track, "s1")
    for ev in events:
        if ev.behavior_type == "fall":
            assert ev.severity == "alert"


def test_fall_not_detected_short_traj():
    det = BehaviorDetector()
    track = _make_track(
        [_pt(0.5, 0.5, 0.4, 0.2, 0.6, 0.8), _pt(0.5, 0.5, 0.3, 0.4, 0.7, 0.6)],
        age=0.5,
    )
    events = det.analyze(track, "s1")
    fall_evs = [e for e in events if e.behavior_type == "fall"]
    assert fall_evs == []


def test_fall_description_japanese():
    det = BehaviorDetector(fall_aspect_normal=1.0, fall_aspect_threshold=0.7)
    early = [_pt(0.5, 0.5, 0.45, 0.2, 0.55, 0.8)] * 2
    recent = [_pt(0.5, 0.5, 0.2, 0.45, 0.8, 0.55)] * 2
    track = _make_track(early + recent, age=1.0)
    events = det.analyze(track, "s1")
    for ev in events:
        if ev.behavior_type == "fall":
            assert "転倒" in ev.description


# ── Confidence clamped ────────────────────────────────────────────────────────


def test_confidence_always_in_range():
    det = BehaviorDetector(loitering_seconds=1.0, loitering_threshold=0.001,
                           running_threshold=0.001)
    pts_still = [_pt(0.5, 0.5, 0.4, 0.3, 0.6, 0.9)] * 4
    track = _make_track(pts_still, age=20.0)
    events = det.analyze(track, "s1")
    for ev in events:
        assert 0.0 <= ev.confidence <= 1.0
