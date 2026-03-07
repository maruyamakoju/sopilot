"""Tests for Phase 45: ActiveLearningLoop (active_loop.py).

Tests cover:
    - LabelSubmission dataclass + to_dict
    - ActiveLearningLoop init (with/without queue/learner)
    - submit_label (returns dict, learner.observe called, confidence filter)
    - get_pending_frames (no queue → empty, mock queue integration)
    - get_stats (structure, counters)
    - get_recent_labels (limit, ordering)
    - Persistence (save/load round-trip)
    - Thread safety (concurrent submit_label)
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from sopilot.perception.active_loop import (
    ActiveLearningLoop,
    LabelSubmission,
    _to_numpy,
)


_DIM = 768


def _rand_feat(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(_DIM).astype(np.float32)


# ---------------------------------------------------------------------------
# LabelSubmission
# ---------------------------------------------------------------------------


class TestLabelSubmission:
    def test_fields(self):
        s = LabelSubmission(
            submission_id="sub1",
            frame_id="frm1",
            label="helmet_violation",
            confidence=0.95,
            operator_id="op01",
            submitted_at=time.time(),
        )
        assert s.label == "helmet_violation"
        assert s.confidence == 0.95

    def test_to_dict_keys(self):
        s = LabelSubmission(
            submission_id="s1", frame_id="f1", label="normal",
            confidence=1.0, operator_id="op", submitted_at=time.time(),
        )
        d = s.to_dict()
        for key in ("submission_id", "frame_id", "label", "confidence",
                    "operator_id", "submitted_at", "feature_dim",
                    "previous_prediction"):
            assert key in d

    def test_to_dict_confidence_rounded(self):
        s = LabelSubmission(
            submission_id="s1", frame_id="f1", label="normal",
            confidence=0.123456, operator_id="op", submitted_at=time.time(),
        )
        d = s.to_dict()
        assert abs(d["confidence"] - 0.123) < 0.001


# ---------------------------------------------------------------------------
# _to_numpy helper
# ---------------------------------------------------------------------------


class TestToNumpy:
    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_numpy(arr)
        assert result is not None
        assert len(result) == 3

    def test_list_input(self):
        result = _to_numpy([1.0, 2.0, 3.0])
        assert result is not None

    def test_none_input(self):
        assert _to_numpy(None) is None

    def test_empty_array_returns_none(self):
        assert _to_numpy(np.array([])) is None

    def test_2d_array_returns_none(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert _to_numpy(arr) is None


# ---------------------------------------------------------------------------
# ActiveLearningLoop init
# ---------------------------------------------------------------------------


class TestActiveLearningLoopInit:
    def test_init_defaults(self):
        loop = ActiveLearningLoop()
        stats = loop.get_stats()
        assert stats["labels_submitted"] == 0
        assert stats["learner_updates"] == 0

    def test_no_queue_no_learner(self):
        loop = ActiveLearningLoop(review_queue=None, online_learner=None)
        stats = loop.get_stats()
        assert stats["queue_available"] is False
        assert stats["learner_available"] is False

    def test_with_queue_and_learner(self):
        mock_q = MagicMock()
        mock_l = MagicMock()
        loop = ActiveLearningLoop(review_queue=mock_q, online_learner=mock_l)
        stats = loop.get_stats()
        assert stats["queue_available"] is True
        assert stats["learner_available"] is True

    def test_min_confidence_stored(self):
        loop = ActiveLearningLoop(min_confidence=0.75)
        assert loop.get_stats()["min_confidence"] == 0.75


# ---------------------------------------------------------------------------
# submit_label
# ---------------------------------------------------------------------------


class TestSubmitLabel:
    def test_returns_dict(self):
        loop = ActiveLearningLoop()
        result = loop.submit_label("frame1", "normal")
        assert isinstance(result, dict)

    def test_result_keys(self):
        loop = ActiveLearningLoop()
        result = loop.submit_label("frame1", "helmet_violation")
        for key in ("submission_id", "frame_id", "label", "learner_updated", "confidence"):
            assert key in result

    def test_result_frame_id(self):
        loop = ActiveLearningLoop()
        result = loop.submit_label("my_frame", "normal")
        assert result["frame_id"] == "my_frame"

    def test_result_label(self):
        loop = ActiveLearningLoop()
        result = loop.submit_label("f1", "ppe_violation")
        assert result["label"] == "ppe_violation"

    def test_label_count_increments(self):
        loop = ActiveLearningLoop()
        loop.submit_label("f1", "normal")
        loop.submit_label("f2", "helmet_violation")
        stats = loop.get_stats()
        assert stats["labels_submitted"] == 2

    def test_learner_observe_called_with_feature(self):
        mock_learner = MagicMock()
        loop = ActiveLearningLoop(online_learner=mock_learner)
        feat = _rand_feat(seed=1)
        loop.submit_label("f1", "normal", feature_vector=feat, confidence=0.9)
        mock_learner.observe.assert_called_once()

    def test_learner_observe_not_called_without_feature(self):
        mock_learner = MagicMock()
        loop = ActiveLearningLoop(online_learner=mock_learner)
        loop.submit_label("f1", "normal", feature_vector=None)
        mock_learner.observe.assert_not_called()

    def test_learner_not_called_below_min_confidence(self):
        mock_learner = MagicMock()
        loop = ActiveLearningLoop(online_learner=mock_learner, min_confidence=0.8)
        feat = _rand_feat()
        loop.submit_label("f1", "normal", feature_vector=feat, confidence=0.5)
        # confidence 0.5 < min_confidence 0.8
        mock_learner.observe.assert_not_called()

    def test_learner_updated_flag(self):
        mock_learner = MagicMock()
        loop = ActiveLearningLoop(online_learner=mock_learner)
        feat = _rand_feat()
        result = loop.submit_label("f1", "normal", feature_vector=feat, confidence=0.9)
        assert result["learner_updated"] is True

    def test_learner_not_updated_flag(self):
        loop = ActiveLearningLoop()  # no learner
        result = loop.submit_label("f1", "normal", feature_vector=_rand_feat())
        assert result["learner_updated"] is False

    def test_learner_updates_count(self):
        mock_learner = MagicMock()
        loop = ActiveLearningLoop(online_learner=mock_learner)
        feat = _rand_feat()
        loop.submit_label("f1", "normal", feature_vector=feat)
        loop.submit_label("f2", "violation", feature_vector=feat)
        assert loop.get_stats()["learner_updates"] == 2

    def test_label_distribution_tracked(self):
        loop = ActiveLearningLoop()
        loop.submit_label("f1", "normal")
        loop.submit_label("f2", "normal")
        loop.submit_label("f3", "violation")
        dist = loop.get_stats()["label_distribution"]
        assert dist["normal"] == 2
        assert dist["violation"] == 1

    def test_learner_failure_handled(self):
        mock_learner = MagicMock()
        mock_learner.observe.side_effect = RuntimeError("fail")
        loop = ActiveLearningLoop(online_learner=mock_learner)
        # Should not raise
        result = loop.submit_label("f1", "normal", feature_vector=_rand_feat())
        assert isinstance(result, dict)

    def test_submission_id_is_uuid(self):
        loop = ActiveLearningLoop()
        result = loop.submit_label("f1", "normal")
        sid = result["submission_id"]
        assert isinstance(sid, str)
        assert len(sid) == 36


# ---------------------------------------------------------------------------
# get_pending_frames
# ---------------------------------------------------------------------------


class TestGetPendingFrames:
    def test_no_queue_returns_empty(self):
        loop = ActiveLearningLoop(review_queue=None)
        assert loop.get_pending_frames() == []

    def test_mock_queue_with_dict_items(self):
        mock_queue = MagicMock()
        mock_queue._queue = [
            {"frame_id": "f1", "score": 0.9, "timestamp": time.time()},
            {"frame_id": "f2", "score": 0.7, "timestamp": time.time()},
        ]
        loop = ActiveLearningLoop(review_queue=mock_queue)
        frames = loop.get_pending_frames(limit=10)
        assert len(frames) == 2
        assert all("frame_id" in f for f in frames)
        assert all("already_labeled" in f for f in frames)

    def test_already_labeled_flag(self):
        mock_queue = MagicMock()
        mock_queue._queue = [
            {"frame_id": "f1", "score": 0.9, "timestamp": time.time()},
        ]
        loop = ActiveLearningLoop(review_queue=mock_queue)
        loop.submit_label("f1", "normal")
        frames = loop.get_pending_frames()
        labeled = [f for f in frames if f["frame_id"] == "f1"]
        if labeled:
            assert labeled[0]["already_labeled"] is True

    def test_limit_applied(self):
        mock_queue = MagicMock()
        mock_queue._queue = [
            {"frame_id": f"f{i}", "score": 0.5, "timestamp": time.time()}
            for i in range(10)
        ]
        loop = ActiveLearningLoop(review_queue=mock_queue)
        frames = loop.get_pending_frames(limit=3)
        assert len(frames) <= 3


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_stats_structure(self):
        loop = ActiveLearningLoop()
        stats = loop.get_stats()
        for key in ("labels_submitted", "unique_frames_labeled", "learner_updates",
                    "label_distribution", "unique_labels", "min_confidence",
                    "queue_available", "learner_available", "uptime_seconds"):
            assert key in stats

    def test_unique_frames_labeled(self):
        loop = ActiveLearningLoop()
        loop.submit_label("f1", "normal")
        loop.submit_label("f1", "normal")  # same frame twice
        loop.submit_label("f2", "violation")
        # unique frames: f1 + f2 = 2
        assert loop.get_stats()["unique_frames_labeled"] == 2

    def test_uptime_positive(self):
        loop = ActiveLearningLoop()
        time.sleep(0.01)
        assert loop.get_stats()["uptime_seconds"] >= 0


# ---------------------------------------------------------------------------
# get_recent_labels
# ---------------------------------------------------------------------------


class TestGetRecentLabels:
    def test_returns_list(self):
        loop = ActiveLearningLoop()
        loop.submit_label("f1", "normal")
        labels = loop.get_recent_labels()
        assert isinstance(labels, list)

    def test_labels_are_dicts(self):
        loop = ActiveLearningLoop()
        loop.submit_label("f1", "normal")
        labels = loop.get_recent_labels()
        for l in labels:
            assert isinstance(l, dict)
            assert "label" in l

    def test_most_recent_first(self):
        loop = ActiveLearningLoop()
        loop.submit_label("f1", "label_a")
        loop.submit_label("f2", "label_b")
        labels = loop.get_recent_labels(limit=2)
        assert labels[0]["label"] == "label_b"

    def test_limit_applied(self):
        loop = ActiveLearningLoop()
        for i in range(5):
            loop.submit_label(f"f{i}", "normal")
        labels = loop.get_recent_labels(limit=3)
        assert len(labels) <= 3


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "al.json"
            loop = ActiveLearningLoop(persist_path=path)
            loop.submit_label("f1", "normal")
            loop._save()
            assert path.exists()

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "al.json"
            loop = ActiveLearningLoop(persist_path=path)
            for i in range(3):
                loop.submit_label(f"f{i}", f"label_{i}")
            loop._save()

            loop2 = ActiveLearningLoop(persist_path=path)
            assert loop2.get_stats()["labels_submitted"] == 3

    def test_load_restores_label_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "al.json"
            loop = ActiveLearningLoop(persist_path=path)
            loop.submit_label("f1", "normal")
            loop.submit_label("f2", "normal")
            loop._save()

            loop2 = ActiveLearningLoop(persist_path=path)
            assert loop2.get_stats()["label_distribution"].get("normal", 0) == 2

    def test_no_persist_path_save_silent(self):
        loop = ActiveLearningLoop()
        loop.submit_label("f1", "normal")
        loop._save()  # should not raise

    def test_load_missing_file_silent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            loop = ActiveLearningLoop(persist_path=path)
            assert loop.get_stats()["labels_submitted"] == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_submit(self):
        loop = ActiveLearningLoop()
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                loop.submit_label(f"frame_{i}", "normal", feature_vector=_rand_feat(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert loop.get_stats()["labels_submitted"] == 30

    def test_concurrent_submit_and_stats(self):
        loop = ActiveLearningLoop()
        errors: list[Exception] = []

        def submit_worker() -> None:
            try:
                loop.submit_label("f1", "normal")
            except Exception as e:
                errors.append(e)

        def stats_worker() -> None:
            try:
                loop.get_stats()
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=submit_worker) for _ in range(10)]
            + [threading.Thread(target=stats_worker) for _ in range(10)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
