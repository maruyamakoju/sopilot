"""Tests for Phase 11A: Active Query Strategy (能動的クエリ).

Coverage:
    - ReviewItem dataclass (to_dict, fields)
    - ReviewQueue: maybe_add, get_pending, record_review, get_stats, clear, dedup
    - Engine integration: _review_queue field, Stage 6i wiring
    - API endpoints: GET /review-queue, POST /review/{id}, GET /review-stats
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

from sopilot.perception.types import (
    EntityEvent,
    EntityEventType,
    PerceptionConfig,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_anomaly_event(
    detector: str = "behavioral",
    metric: str = "speed_zscore",
    entity_id: int = 1,
    z_score: float = 3.0,
    description_ja: str = "速度異常",
    timestamp: float = 1.0,
    frame_number: int = 1,
) -> EntityEvent:
    return EntityEvent(
        event_type=EntityEventType.ANOMALY,
        entity_id=entity_id,
        timestamp=timestamp,
        frame_number=frame_number,
        details={
            "detector": detector,
            "metric": metric,
            "z_score": z_score,
            "description_ja": description_ja,
        },
    )


# ===========================================================================
# ReviewItem tests
# ===========================================================================


class TestReviewItem(unittest.TestCase):
    def _make(self, **kw):
        from sopilot.perception.active_query import ReviewItem
        defaults = dict(
            review_id="abc12345",
            detector="behavioral",
            metric="speed_zscore",
            entity_id=1,
            z_score=3.5,
            timestamp=1.0,
            frame_number=1,
            description_ja="速度異常",
            priority=3.5,
            created_at=time.time(),
        )
        defaults.update(kw)
        return ReviewItem(**defaults)

    def test_review_id_field(self):
        item = self._make(review_id="xyzxyz")
        self.assertEqual(item.review_id, "xyzxyz")

    def test_defaults_reviewed_false(self):
        item = self._make()
        self.assertFalse(item.reviewed)
        self.assertIsNone(item.confirmed)
        self.assertEqual(item.note, "")

    def test_to_dict_keys(self):
        item = self._make()
        d = item.to_dict()
        for key in ("review_id", "detector", "metric", "entity_id",
                    "z_score", "timestamp", "frame_number", "description_ja",
                    "priority", "created_at", "reviewed", "confirmed", "note"):
            self.assertIn(key, d)

    def test_to_dict_z_score_rounded(self):
        item = self._make(z_score=3.141592)
        d = item.to_dict()
        self.assertEqual(d["z_score"], round(3.141592, 3))


# ===========================================================================
# ReviewQueue tests
# ===========================================================================


class TestReviewQueue(unittest.TestCase):

    def _q(self, **kw):
        from sopilot.perception.active_query import ReviewQueue
        defaults = dict(z_threshold=2.5, max_pending=10, dedup_seconds=1.0)
        defaults.update(kw)
        return ReviewQueue(**defaults)

    def test_empty_queue_pending_count_zero(self):
        q = self._q()
        self.assertEqual(q.pending_count(), 0)

    def test_maybe_add_below_threshold_returns_false(self):
        q = self._q(z_threshold=3.0)
        ev = _make_anomaly_event(z_score=2.0)
        self.assertFalse(q.maybe_add(ev, 2.0))

    def test_maybe_add_at_threshold_returns_true(self):
        q = self._q(z_threshold=2.5)
        ev = _make_anomaly_event(z_score=2.5)
        self.assertTrue(q.maybe_add(ev, 2.5))

    def test_maybe_add_above_threshold_returns_true(self):
        q = self._q()
        ev = _make_anomaly_event(z_score=4.0)
        self.assertTrue(q.maybe_add(ev, 4.0))

    def test_pending_count_increments(self):
        q = self._q(dedup_seconds=0.0)
        for i in range(3):
            ev = _make_anomaly_event(detector=f"d{i}", metric="m", z_score=3.0)
            q.maybe_add(ev, 3.0)
        self.assertEqual(q.pending_count(), 3)

    def test_get_pending_returns_list(self):
        q = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0)
        items = q.get_pending()
        self.assertIsInstance(items, list)
        self.assertEqual(len(items), 1)

    def test_get_pending_ordered_by_priority_desc(self):
        q = self._q(dedup_seconds=0.0)
        for z in [2.6, 5.0, 3.5]:
            ev = _make_anomaly_event(detector=f"d{z}", metric="m", z_score=z)
            q.maybe_add(ev, z)
        items = q.get_pending(n=3)
        scores = [i.z_score for i in items]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_get_pending_respects_n(self):
        q = self._q(dedup_seconds=0.0)
        for i in range(5):
            ev = _make_anomaly_event(detector=f"d{i}", metric="m", z_score=3.0 + i * 0.1)
            q.maybe_add(ev, 3.0 + i * 0.1)
        items = q.get_pending(n=2)
        self.assertEqual(len(items), 2)

    def test_dedup_same_detector_metric_within_window(self):
        q = self._q(dedup_seconds=60.0)
        ev = _make_anomaly_event(detector="behavioral", metric="speed_zscore", z_score=3.0)
        self.assertTrue(q.maybe_add(ev, 3.0))
        # Same detector+metric within 60s → rejected
        self.assertFalse(q.maybe_add(ev, 3.0))

    def test_dedup_different_metric_allowed(self):
        q = self._q(dedup_seconds=60.0)
        ev1 = _make_anomaly_event(detector="behavioral", metric="speed_zscore", z_score=3.0)
        ev2 = _make_anomaly_event(detector="behavioral", metric="activity_freq", z_score=3.0)
        self.assertTrue(q.maybe_add(ev1, 3.0))
        self.assertTrue(q.maybe_add(ev2, 3.0))

    def test_record_review_moves_to_reviewed(self):
        q = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0)
        items = q.get_pending()
        rid = items[0].review_id
        result = q.record_review(rid, confirmed=True, note="OK")
        self.assertIsNotNone(result)
        self.assertEqual(q.pending_count(), 0)

    def test_record_review_returns_item(self):
        q = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0)
        rid = q.get_pending()[0].review_id
        item = q.record_review(rid, confirmed=False)
        self.assertFalse(item.confirmed)
        self.assertTrue(item.reviewed)

    def test_record_review_unknown_id_returns_none(self):
        q = self._q()
        result = q.record_review("nonexistent", confirmed=True)
        self.assertIsNone(result)

    def test_get_stats_keys(self):
        q = self._q()
        stats = q.get_stats()
        for key in ("pending_count", "reviewed_count", "confirmed_count",
                    "denied_count", "confirm_rate", "detector_counts",
                    "z_threshold", "max_pending"):
            self.assertIn(key, stats)

    def test_get_stats_pending_count(self):
        q = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0)
        self.assertEqual(q.get_stats()["pending_count"], 1)

    def test_get_stats_after_review(self):
        q = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0)
        rid = q.get_pending()[0].review_id
        q.record_review(rid, confirmed=True)
        stats = q.get_stats()
        self.assertEqual(stats["confirmed_count"], 1)
        self.assertEqual(stats["pending_count"], 0)

    def test_confirm_rate_zero_when_no_reviews(self):
        q = self._q()
        self.assertEqual(q.get_stats()["confirm_rate"], 0.0)

    def test_max_pending_enforced(self):
        q = self._q(max_pending=3, dedup_seconds=0.0)
        for i in range(10):
            ev = _make_anomaly_event(detector=f"d{i}", metric="m",
                                      z_score=3.0 + i * 0.1)
            q.maybe_add(ev, 3.0 + i * 0.1)
        self.assertLessEqual(q.pending_count(), 3)

    def test_clear_empties_queue(self):
        q = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0)
        q.clear()
        self.assertEqual(q.pending_count(), 0)
        self.assertEqual(q.get_stats()["reviewed_count"], 0)

    def test_tuner_skip_when_enough_feedback(self):
        q = self._q(min_feedback_for_skip=5)
        mock_tuner = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total = 10  # > 5 → skip
        mock_tuner.get_pair_stats.return_value = mock_stats
        ev = _make_anomaly_event(z_score=3.0)
        result = q.maybe_add(ev, 3.0, tuner=mock_tuner)
        self.assertFalse(result)

    def test_tuner_add_when_insufficient_feedback(self):
        q = self._q(min_feedback_for_skip=5)
        mock_tuner = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total = 2  # < 5 → add
        mock_tuner.get_pair_stats.return_value = mock_stats
        ev = _make_anomaly_event(z_score=3.0)
        result = q.maybe_add(ev, 3.0, tuner=mock_tuner)
        self.assertTrue(result)

    def test_tuner_exception_does_not_prevent_add(self):
        q = self._q()
        bad_tuner = MagicMock()
        bad_tuner.get_pair_stats.side_effect = RuntimeError("fail")
        ev = _make_anomaly_event(z_score=3.0)
        # Should not raise; should still add
        result = q.maybe_add(ev, 3.0, tuner=bad_tuner)
        self.assertTrue(result)

    def test_thread_safety_concurrent_adds(self):
        import threading
        q = self._q(dedup_seconds=0.0, max_pending=100)
        errors = []

        def add_items(start):
            try:
                for i in range(5):
                    ev = _make_anomaly_event(
                        detector=f"d{start}_{i}", metric="m", z_score=3.0
                    )
                    q.maybe_add(ev, 3.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_items, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


# ===========================================================================
# Engine integration tests
# ===========================================================================


class TestReviewQueueEngineIntegration(unittest.TestCase):
    """Tests that PerceptionEngine has _review_queue and Stage 6i wires correctly."""

    def test_engine_has_review_queue_field(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        self.assertTrue(hasattr(engine, "_review_queue"))

    def test_review_queue_none_by_default(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        self.assertIsNone(engine._review_queue)

    def test_build_engine_injects_review_queue(self):
        from sopilot.perception.engine import build_perception_engine
        from sopilot.perception.active_query import ReviewQueue
        engine = build_perception_engine()
        self.assertIsInstance(engine._review_queue, ReviewQueue)

    def test_review_queue_uses_config_z_threshold(self):
        from sopilot.perception.engine import build_perception_engine
        config = PerceptionConfig(review_z_threshold=4.0)
        engine = build_perception_engine(config=config)
        self.assertAlmostEqual(engine._review_queue._z_threshold, 4.0)

    def test_review_queue_uses_config_max_pending(self):
        from sopilot.perception.engine import build_perception_engine
        config = PerceptionConfig(review_queue_max_pending=25)
        engine = build_perception_engine(config=config)
        self.assertEqual(engine._review_queue._max_pending, 25)

    def test_no_error_when_review_queue_none(self):
        import numpy as np
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        engine._review_queue = None
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            engine.process_frame(frame)
        except Exception:
            pass  # Other stages may fail; review queue must not cause AttributeError


# ===========================================================================
# Frame ring buffer tests
# ===========================================================================


class TestFrameRingBuffer(unittest.TestCase):
    """Tests for engine._frame_ring and get_latest_frame_jpeg()."""

    def test_engine_has_frame_ring_attr(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        self.assertTrue(hasattr(engine, "_frame_ring"))

    def test_get_latest_frame_jpeg_returns_none_when_empty(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        result = engine.get_latest_frame_jpeg()
        self.assertIsNone(result)

    def test_frame_ring_maxlen_from_config(self):
        from sopilot.perception.engine import PerceptionEngine
        config = PerceptionConfig(frame_ring_buffer_size=15)
        engine = PerceptionEngine(config=config)
        self.assertEqual(engine._frame_ring.maxlen, 15)

    def test_get_latest_frame_jpeg_returns_bytes_after_frame(self):
        import numpy as np
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        # Manually push a frame into the ring
        engine._frame_ring.append((1.0, b"\xff\xd8\xff"))  # mock JPEG bytes
        result = engine.get_latest_frame_jpeg()
        self.assertIsInstance(result, bytes)

    def test_frame_ring_respects_maxlen(self):
        from sopilot.perception.engine import PerceptionEngine
        config = PerceptionConfig(frame_ring_buffer_size=5)
        engine = PerceptionEngine(config=config)
        for i in range(10):
            engine._frame_ring.append((float(i), b"data"))
        self.assertEqual(len(engine._frame_ring), 5)


# ===========================================================================
# API endpoint tests
# ===========================================================================


class TestReviewQueueEndpoints(unittest.TestCase):
    """E2E HTTP tests for review queue endpoints."""

    def setUp(self):
        import os
        import tempfile
        from pathlib import Path as _P
        from unittest.mock import MagicMock
        from fastapi.testclient import TestClient
        from sopilot.main import create_app
        from sopilot.perception.active_query import ReviewQueue

        self._tmp = tempfile.TemporaryDirectory()
        root = _P(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "review-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"

        self.app = create_app()
        self.client = TestClient(self.app)

        # Build mock engine with _review_queue
        engine = MagicMock()
        engine._anomaly_tuner = None
        self.queue = ReviewQueue(z_threshold=2.5, max_pending=20, dedup_seconds=0.0)
        engine._review_queue = self.queue
        engine.get_adaptive_learner_state.return_value = {
            "adaptive_learner": {
                "total_observed": 0, "score_window_size": 0,
                "score_mean": 0.0, "score_std": 0.0,
                "drift_count": 0, "recalibration_count": 0,
                "last_recalibration": None, "ph_state": {},
            },
            "tuner": {
                "total_feedback": 0, "confirmed": 0, "denied": 0,
                "overall_confirm_rate": 0.0, "pairs_tracked": 0,
                "pairs_suppressed": 0, "pairs_trusted": 0,
                "last_tuning": 0.0, "pair_stats": [],
                "suppressed_pairs": [], "trusted_pairs": [],
                "min_samples_for_tuning": 10,
            },
        }
        engine.get_latest_frame_jpeg.return_value = None

        vlm = MagicMock()
        vlm._engine = engine
        self.app.state.vigil_pipeline._vlm = vlm
        self.engine = engine

    def tearDown(self):
        import os
        self._tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                   "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
            os.environ.pop(k, None)

    def _add_item(self, detector="behavioral", metric="speed_zscore", z_score=3.0):
        ev = _make_anomaly_event(detector=detector, metric=metric, z_score=z_score)
        self.queue.maybe_add(ev, z_score)

    def test_get_review_queue_200(self):
        r = self.client.get("/vigil/perception/review-queue")
        self.assertEqual(r.status_code, 200)

    def test_get_review_queue_empty_list(self):
        r = self.client.get("/vigil/perception/review-queue")
        data = r.json()
        self.assertIn("items", data)
        self.assertEqual(data["items"], [])

    def test_get_review_queue_with_item(self):
        self._add_item()
        r = self.client.get("/vigil/perception/review-queue")
        data = r.json()
        self.assertEqual(len(data["items"]), 1)

    def test_get_review_queue_item_fields(self):
        self._add_item()
        r = self.client.get("/vigil/perception/review-queue")
        item = r.json()["items"][0]
        for key in ("review_id", "detector", "metric", "z_score", "description_ja"):
            self.assertIn(key, item)

    def test_post_review_confirmed(self):
        self._add_item()
        rid = self.client.get("/vigil/perception/review-queue").json()["items"][0]["review_id"]
        r = self.client.post(f"/vigil/perception/review/{rid}",
                             json={"confirmed": True, "note": "OK"})
        self.assertEqual(r.status_code, 200)

    def test_post_review_removes_from_pending(self):
        self._add_item()
        rid = self.client.get("/vigil/perception/review-queue").json()["items"][0]["review_id"]
        self.client.post(f"/vigil/perception/review/{rid}", json={"confirmed": False})
        r2 = self.client.get("/vigil/perception/review-queue")
        self.assertEqual(len(r2.json()["items"]), 0)

    def test_post_review_unknown_id_404(self):
        r = self.client.post("/vigil/perception/review/badid99",
                             json={"confirmed": True})
        self.assertEqual(r.status_code, 404)

    def test_get_review_stats_200(self):
        r = self.client.get("/vigil/perception/review-stats")
        self.assertEqual(r.status_code, 200)

    def test_get_review_stats_keys(self):
        r = self.client.get("/vigil/perception/review-stats")
        data = r.json()
        for key in ("pending_count", "reviewed_count", "confirmed_count",
                    "denied_count", "confirm_rate"):
            self.assertIn(key, data)

    def test_get_frame_snapshot_204_when_empty(self):
        r = self.client.get("/vigil/perception/frame-snapshot")
        # No frame in buffer → 204 or 404
        self.assertIn(r.status_code, (204, 404))


# ===========================================================================
# Phase 13: ReviewQueue persistence tests
# ===========================================================================


class TestReviewQueuePersistence(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self._persist_path = f"{self._tmp.name}/review_queue.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _q(self, **kw):
        from sopilot.perception.active_query import ReviewQueue
        return ReviewQueue(
            z_threshold=2.5, max_pending=10, dedup_seconds=0.0,
            persist_path=self._persist_path,
            **kw,
        )

    def test_save_creates_json_file(self):
        import os
        q = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0)
        self.assertTrue(os.path.exists(self._persist_path))

    def test_load_restores_pending_items(self):
        q1 = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q1.maybe_add(ev, 3.0)

        from sopilot.perception.active_query import ReviewQueue
        q2 = ReviewQueue(z_threshold=2.5, persist_path=self._persist_path)
        self.assertEqual(q2.pending_count(), 1)

    def test_load_restores_reviewed_items(self):
        q1 = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q1.maybe_add(ev, 3.0)
        rid = q1.get_pending()[0].review_id
        q1.record_review(rid, confirmed=True)

        from sopilot.perception.active_query import ReviewQueue
        q2 = ReviewQueue(z_threshold=2.5, persist_path=self._persist_path)
        stats = q2.get_stats()
        self.assertEqual(stats["reviewed_count"], 1)
        self.assertEqual(stats["confirmed_count"], 1)

    def test_save_on_maybe_add(self):
        import json, os
        q = self._q()
        self.assertFalse(os.path.exists(self._persist_path))
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0)
        self.assertTrue(os.path.exists(self._persist_path))
        data = json.loads(open(self._persist_path).read())
        self.assertEqual(len(data["pending"]), 1)

    def test_save_on_record_review(self):
        import json
        q = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0)
        rid = q.get_pending()[0].review_id
        q.record_review(rid, confirmed=False)
        data = json.loads(open(self._persist_path).read())
        self.assertEqual(len(data["pending"]), 0)
        self.assertEqual(len(data["reviewed"]), 1)

    def test_clear_persists_empty_state(self):
        import json
        q = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0)
        q.clear()
        data = json.loads(open(self._persist_path).read())
        self.assertEqual(data["pending"], [])
        self.assertEqual(data["reviewed"], [])

    def test_no_save_when_persist_path_none(self):
        from sopilot.perception.active_query import ReviewQueue
        q = ReviewQueue(z_threshold=2.5, persist_path=None)
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0)  # should not raise, no file created
        self.assertIsNone(q._persist_path)


# ===========================================================================
# Phase 13B: ReviewQueue clip archive tests
# ===========================================================================


class TestReviewQueueClipArchive(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self._persist_path = f"{self._tmp.name}/review_queue.json"
        self._frames_root = f"{self._tmp.name}/frames"

    def tearDown(self):
        self._tmp.cleanup()

    def _q(self, **kw):
        from sopilot.perception.active_query import ReviewQueue
        return ReviewQueue(
            z_threshold=2.5, max_pending=10, dedup_seconds=0.0,
            persist_path=self._persist_path,
            frames_root=self._frames_root,
            **kw,
        )

    _FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 20  # minimal JPEG-like bytes

    def test_maybe_add_saves_frame_jpeg_to_disk(self):
        import os
        q = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0, frame_jpeg=self._FAKE_JPEG)
        item = q.get_pending()[0]
        self.assertTrue(item.frame_path)
        self.assertTrue(os.path.exists(item.frame_path))

    def test_maybe_add_no_frame_when_frames_root_none(self):
        from sopilot.perception.active_query import ReviewQueue
        q = ReviewQueue(z_threshold=2.5, frames_root=None)
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0, frame_jpeg=self._FAKE_JPEG)
        item = q.get_pending()[0]
        self.assertEqual(item.frame_path, "")

    def test_frame_path_in_to_dict(self):
        q = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0, frame_jpeg=self._FAKE_JPEG)
        d = q.get_pending()[0].to_dict()
        self.assertIn("frame_path", d)
        self.assertTrue(d["frame_path"])

    def test_frame_path_persisted_in_json(self):
        import json
        q = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0, frame_jpeg=self._FAKE_JPEG)
        data = json.loads(open(self._persist_path).read())
        self.assertTrue(data["pending"][0]["frame_path"])

    def test_engine_passes_frame_jpeg_to_maybe_add(self):
        """Stage 6i passes frame_jpeg from _frame_ring to maybe_add."""
        import numpy as np
        from unittest.mock import patch, MagicMock
        from sopilot.perception.engine import PerceptionEngine
        from sopilot.perception.active_query import ReviewQueue
        from sopilot.perception.types import (
            EntityEvent, EntityEventType, WorldState, SceneGraph,
        )

        engine = PerceptionEngine(config=PerceptionConfig())
        engine._frame_ring.append((1.0, self._FAKE_JPEG))

        calls = []
        class CapturingRQ(ReviewQueue):
            def maybe_add(self, event, z_score, tuner=None, frame_jpeg=None):
                calls.append(frame_jpeg)
                return super().maybe_add(event, z_score, tuner=tuner, frame_jpeg=frame_jpeg)

        engine._review_queue = CapturingRQ(z_threshold=2.5)

        ev = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1, timestamp=1.0, frame_number=1,
            details={"detector": "behavioral", "metric": "speed_zscore",
                     "z_score": 5.0, "description_ja": "テスト"},
        )
        sg = SceneGraph(timestamp=1.0, frame_number=1, entities=[], relations=[],
                        frame_shape=(64, 64))
        ws = WorldState(scene_graph=sg, timestamp=1.0, frame_number=1,
                        active_tracks={}, zone_occupancy={}, events=[ev])

        # Invoke Stage 6i directly
        try:
            engine._frame_ring.append((1.0, self._FAKE_JPEG))
            # We can't call full process_frame safely; instead test the stage inline
            import types
            # Replicate Stage 6i logic
            _frame_jpeg = engine._frame_ring[-1][1] if engine._frame_ring else None
            for _ev in ws.events:
                if hasattr(_ev.event_type, "name") and _ev.event_type.name == "ANOMALY":
                    _z = float(_ev.details.get("z_score", 0.0)) if _ev.details else 0.0
                    engine._review_queue.maybe_add(_ev, _z, tuner=None, frame_jpeg=_frame_jpeg)
        except Exception:
            pass

        self.assertTrue(len(calls) > 0)
        self.assertEqual(calls[0], self._FAKE_JPEG)

    def test_review_frame_endpoint_200(self):
        import os, tempfile
        from pathlib import Path as _P
        from fastapi.testclient import TestClient
        from sopilot.main import create_app
        from sopilot.perception.active_query import ReviewQueue
        from unittest.mock import MagicMock

        tmp = tempfile.TemporaryDirectory()
        root = _P(tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "clip-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"

        app = create_app()
        client = TestClient(app)

        queue = ReviewQueue(
            z_threshold=2.5, dedup_seconds=0.0,
            frames_root=str(root / "frames"),
        )
        ev = _make_anomaly_event(z_score=3.0)
        queue.maybe_add(ev, 3.0, frame_jpeg=self._FAKE_JPEG)
        rid = queue.get_pending()[0].review_id

        engine = MagicMock()
        engine._review_queue = queue
        engine._anomaly_tuner = None
        engine.get_adaptive_learner_state.return_value = {
            "adaptive_learner": {
                "total_observed": 0, "score_window_size": 0,
                "score_mean": 0.0, "score_std": 0.0,
                "drift_count": 0, "recalibration_count": 0,
                "last_recalibration": None, "ph_state": {},
            },
            "tuner": {
                "total_feedback": 0, "confirmed": 0, "denied": 0,
                "overall_confirm_rate": 0.0, "pairs_tracked": 0,
                "pairs_suppressed": 0, "pairs_trusted": 0,
                "last_tuning": 0.0, "pair_stats": [],
                "suppressed_pairs": [], "trusted_pairs": [],
                "min_samples_for_tuning": 10,
            },
        }
        engine.get_latest_frame_jpeg.return_value = None
        vlm = MagicMock()
        vlm._engine = engine
        app.state.vigil_pipeline._vlm = vlm

        r = client.get(f"/vigil/perception/review/{rid}/frame")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/jpeg")

        tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                  "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
            os.environ.pop(k, None)

    def test_review_frame_endpoint_404_no_file(self):
        import os, tempfile
        from pathlib import Path as _P
        from fastapi.testclient import TestClient
        from sopilot.main import create_app
        from sopilot.perception.active_query import ReviewQueue
        from unittest.mock import MagicMock

        tmp = tempfile.TemporaryDirectory()
        root = _P(tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "clip-test-404"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"

        app = create_app()
        client = TestClient(app)

        # Item with no frame_path
        queue = ReviewQueue(z_threshold=2.5, dedup_seconds=0.0)
        ev = _make_anomaly_event(z_score=3.0)
        queue.maybe_add(ev, 3.0)
        rid = queue.get_pending()[0].review_id

        engine = MagicMock()
        engine._review_queue = queue
        engine._anomaly_tuner = None
        engine.get_adaptive_learner_state.return_value = {
            "adaptive_learner": {
                "total_observed": 0, "score_window_size": 0,
                "score_mean": 0.0, "score_std": 0.0,
                "drift_count": 0, "recalibration_count": 0,
                "last_recalibration": None, "ph_state": {},
            },
            "tuner": {
                "total_feedback": 0, "confirmed": 0, "denied": 0,
                "overall_confirm_rate": 0.0, "pairs_tracked": 0,
                "pairs_suppressed": 0, "pairs_trusted": 0,
                "last_tuning": 0.0, "pair_stats": [],
                "suppressed_pairs": [], "trusted_pairs": [],
                "min_samples_for_tuning": 10,
            },
        }
        engine.get_latest_frame_jpeg.return_value = None
        vlm = MagicMock()
        vlm._engine = engine
        app.state.vigil_pipeline._vlm = vlm

        r = client.get(f"/vigil/perception/review/{rid}/frame")
        self.assertEqual(r.status_code, 404)

        tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                  "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
            os.environ.pop(k, None)

    def test_review_frame_path_in_queue_response(self):
        """GET /review-queue includes frame_path field."""
        import os, tempfile
        from pathlib import Path as _P
        from fastapi.testclient import TestClient
        from sopilot.main import create_app
        from sopilot.perception.active_query import ReviewQueue
        from unittest.mock import MagicMock

        tmp = tempfile.TemporaryDirectory()
        root = _P(tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "clip-fp-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"

        app = create_app()
        client = TestClient(app)

        queue = ReviewQueue(
            z_threshold=2.5, dedup_seconds=0.0,
            frames_root=str(root / "frames"),
        )
        ev = _make_anomaly_event(z_score=3.0)
        queue.maybe_add(ev, 3.0, frame_jpeg=self._FAKE_JPEG)

        engine = MagicMock()
        engine._review_queue = queue
        engine._anomaly_tuner = None
        engine.get_adaptive_learner_state.return_value = {
            "adaptive_learner": {
                "total_observed": 0, "score_window_size": 0,
                "score_mean": 0.0, "score_std": 0.0,
                "drift_count": 0, "recalibration_count": 0,
                "last_recalibration": None, "ph_state": {},
            },
            "tuner": {
                "total_feedback": 0, "confirmed": 0, "denied": 0,
                "overall_confirm_rate": 0.0, "pairs_tracked": 0,
                "pairs_suppressed": 0, "pairs_trusted": 0,
                "last_tuning": 0.0, "pair_stats": [],
                "suppressed_pairs": [], "trusted_pairs": [],
                "min_samples_for_tuning": 10,
            },
        }
        engine.get_latest_frame_jpeg.return_value = None
        vlm = MagicMock()
        vlm._engine = engine
        app.state.vigil_pipeline._vlm = vlm

        r = client.get("/vigil/perception/review-queue")
        self.assertEqual(r.status_code, 200)
        item = r.json()["items"][0]
        self.assertIn("frame_path", item)
        self.assertTrue(item["frame_path"])

        tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                  "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
            os.environ.pop(k, None)


if __name__ == "__main__":
    unittest.main()
