"""Tests for PersonTracker — Phase 29."""

from __future__ import annotations

import time
from collections import deque

import pytest

from sopilot.vigil.person_tracker import PersonTracker, TrackState, _iou


# ── _iou ──────────────────────────────────────────────────────────────────────


def test_iou_identical():
    b = {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9}
    assert abs(_iou(b, b) - 1.0) < 1e-6


def test_iou_no_overlap():
    b1 = {"x1": 0.0, "y1": 0.0, "x2": 0.4, "y2": 0.4}
    b2 = {"x1": 0.6, "y1": 0.6, "x2": 1.0, "y2": 1.0}
    assert _iou(b1, b2) == 0.0


def test_iou_partial_overlap():
    b1 = {"x1": 0.0, "y1": 0.0, "x2": 0.6, "y2": 0.6}
    b2 = {"x1": 0.4, "y1": 0.4, "x2": 1.0, "y2": 1.0}
    val = _iou(b1, b2)
    assert 0.0 < val < 1.0


def test_iou_symmetric():
    b1 = {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5}
    b2 = {"x1": 0.3, "y1": 0.3, "x2": 0.7, "y2": 0.7}
    assert abs(_iou(b1, b2) - _iou(b2, b1)) < 1e-9


def test_iou_zero_area_box():
    b1 = {"x1": 0.3, "y1": 0.3, "x2": 0.3, "y2": 0.3}  # zero area
    b2 = {"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8}
    assert _iou(b1, b2) == 0.0


# ── TrackState ─────────────────────────────────────────────────────────────────


def _make_track(age_seconds: float = 5.0) -> TrackState:
    t0 = time.time() - age_seconds
    t1 = time.time()
    traj = deque(
        [
            {
                "cx": 0.5, "cy": 0.5,
                "x1": 0.4, "y1": 0.3, "x2": 0.6, "y2": 0.9,
                "timestamp": t0, "frame_idx": 1,
            },
            {
                "cx": 0.51, "cy": 0.51,
                "x1": 0.41, "y1": 0.31, "x2": 0.61, "y2": 0.91,
                "timestamp": t1, "frame_idx": 2,
            },
        ],
        maxlen=60,
    )
    return TrackState(
        track_id="test-track",
        label="person",
        trajectory=traj,
        first_seen=t0,
        last_seen=t1,
    )


def test_track_state_age():
    track = _make_track(age_seconds=7.0)
    assert abs(track.age - 7.0) < 0.5


def test_track_state_to_dict_has_required_keys():
    track = _make_track()
    d = track.to_dict()
    for key in ("track_id", "label", "status", "age_seconds", "sightings",
                "first_seen", "last_seen", "current_bbox", "behavior_flags"):
        assert key in d


def test_track_state_current_bbox_populated():
    track = _make_track()
    bbox = track.to_dict()["current_bbox"]
    assert bbox is not None
    assert all(k in bbox for k in ("x1", "y1", "x2", "y2"))


def test_track_state_behavior_flags_empty_by_default():
    track = _make_track()
    assert track.to_dict()["behavior_flags"] == []


def test_track_state_to_dict_empty_trajectory():
    t = time.time()
    track = TrackState(
        track_id="t1", label="person",
        trajectory=deque(maxlen=60),
        first_seen=t, last_seen=t,
    )
    d = track.to_dict()
    assert d["current_bbox"] is None
    assert d["sightings"] == 0


# ── PersonTracker ──────────────────────────────────────────────────────────────


def _det(x1=0.4, y1=0.3, x2=0.6, y2=0.9, label="person", conf=0.9) -> dict:
    return {"label": label, "confidence": conf, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def test_tracker_starts_empty():
    t = PersonTracker()
    assert t.total_tracks == 0
    assert t.active_count == 0
    assert t.get_active_tracks() == []


def test_tracker_adds_person_detection():
    t = PersonTracker()
    t.update([_det()])
    assert t.total_tracks == 1
    assert t.active_count == 1


def test_tracker_ignores_non_person():
    t = PersonTracker()
    t.update([_det(label="helmet")])
    assert t.total_tracks == 0


def test_tracker_matches_same_person_second_frame():
    t = PersonTracker(iou_threshold=0.3)
    t.update([_det()])
    t.update([_det()])  # same bbox → should match
    assert t.total_tracks == 1
    assert t.active_count == 1


def test_tracker_trajectory_grows():
    t = PersonTracker()
    t.update([_det()])
    t.update([_det()])
    t.update([_det()])
    track_id = list(t._tracks.keys())[0]
    assert len(t._tracks[track_id].trajectory) == 3


def test_tracker_creates_new_track_no_overlap():
    t = PersonTracker(iou_threshold=0.3)
    t.update([_det(x1=0.0, y1=0.0, x2=0.2, y2=0.2)])
    t.update([_det(x1=0.8, y1=0.8, x2=1.0, y2=1.0)])  # no overlap
    assert t.total_tracks == 2


def test_tracker_lost_track_after_max_frames():
    t = PersonTracker(iou_threshold=0.3, max_lost_frames=2)
    t.update([_det()])
    # 3 frames with no detection → should be lost
    for _ in range(3):
        t.update([])
    assert t.active_count == 0
    lost = [tr for tr in t._tracks.values() if tr.status == "lost"]
    assert len(lost) == 1


def test_tracker_lost_track_not_in_active():
    t = PersonTracker(max_lost_frames=1)
    t.update([_det()])
    t.update([])
    t.update([])
    assert t.get_active_tracks() == []


def test_tracker_clear_resets_all():
    t = PersonTracker()
    t.update([_det()])
    t.update([_det()])
    t.clear()
    assert t.total_tracks == 0
    assert t.active_count == 0


def test_tracker_multiple_persons_tracked():
    t = PersonTracker(iou_threshold=0.3)
    # Two distinct persons, no overlap
    t.update([
        _det(x1=0.0, y1=0.0, x2=0.2, y2=0.4),
        _det(x1=0.8, y1=0.6, x2=1.0, y2=1.0),
    ])
    assert t.total_tracks == 2
    assert t.active_count == 2


def test_tracker_max_history_respected():
    t = PersonTracker(max_history=5)
    for _ in range(10):
        t.update([_det()])
    track_id = list(t._tracks.keys())[0]
    assert len(t._tracks[track_id].trajectory) <= 5


def test_tracker_get_all_tracks_includes_lost():
    t = PersonTracker(max_lost_frames=1)
    t.update([_det()])
    t.update([])
    t.update([])
    all_t = t.get_all_tracks()
    assert len(all_t) == 1
    assert all_t[0]["status"] == "lost"


def test_tracker_get_track_state():
    t = PersonTracker()
    t.update([_det()])
    track_id = list(t._tracks.keys())[0]
    state = t.get_track_state(track_id)
    assert state is not None
    assert state.track_id == track_id


def test_tracker_get_track_state_missing():
    t = PersonTracker()
    assert t.get_track_state("no-such") is None


def test_tracker_japanese_person_label():
    t = PersonTracker()
    t.update([_det(label="人")])
    assert t.total_tracks == 1


def test_tracker_frame_idx_increments():
    t = PersonTracker()
    t.update([_det()])
    t.update([_det()])
    assert t._frame_idx == 2


def test_tracker_matching_highest_iou_first():
    """Two tracks, two detections — greedy matching should pair correctly."""
    t = PersonTracker(iou_threshold=0.01)
    # First frame: two persons
    t.update([
        _det(x1=0.0, y1=0.0, x2=0.3, y2=0.5),
        _det(x1=0.7, y1=0.5, x2=1.0, y2=1.0),
    ])
    assert t.total_tracks == 2
    # Second frame: same two persons (slight shift)
    t.update([
        _det(x1=0.01, y1=0.0, x2=0.31, y2=0.5),
        _det(x1=0.71, y1=0.5, x2=1.0, y2=1.0),
    ])
    # Should still be 2 tracks (matched), not 4 (new tracks)
    assert t.total_tracks == 2


def test_tracker_empty_detections_increments_lost():
    t = PersonTracker(max_lost_frames=5)
    t.update([_det()])
    for _ in range(3):
        t.update([])
    track = list(t._tracks.values())[0]
    assert track.lost_frames == 3
    assert track.status == "active"  # not yet exceeded max_lost_frames=5
