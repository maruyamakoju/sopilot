"""Tests for Phase 14A: 自己学習モニタリングダッシュボード.

Coverage:
    - SigmaTuner.get_history()
    - ReviewQueue.get_review_history()
    - GET /vigil/perception/learning-dashboard endpoint
    - LearningHealthStatus computation (ok / warning / degraded)
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

from sopilot.perception.types import PerceptionConfig


# ---------------------------------------------------------------------------
# Helpers (reused from test_sigma_tuner.py / test_active_query.py patterns)
# ---------------------------------------------------------------------------


def _make_pair_stats(
    detector="behavioral", metric="speed_zscore",
    total=10, confirmed=2, denied=8,
) -> dict:
    return {
        "detector": detector, "metric": metric,
        "total": total, "confirmed": confirmed, "denied": denied,
        "confirmation_rate": confirmed / total if total else 0.0,
        "fp_rate": denied / total if total else 0.0,
    }


def _make_tuner_stats(pair_stats: list) -> dict:
    total = sum(p["total"] for p in pair_stats)
    confirmed = sum(p["confirmed"] for p in pair_stats)
    denied = sum(p["denied"] for p in pair_stats)
    return {
        "total_feedback": total, "confirmed": confirmed, "denied": denied,
        "overall_confirm_rate": confirmed / total if total else 0.0,
        "pairs_tracked": len(pair_stats), "pairs_suppressed": 0,
        "pairs_trusted": 0, "last_tuning": time.time(),
        "pair_stats": pair_stats, "suppressed_pairs": [], "trusted_pairs": [],
        "min_samples_for_tuning": 10,
    }


def _make_anomaly_event(
    detector="behavioral", metric="speed_zscore",
    entity_id=1, z_score=3.0, timestamp=1.0, frame_number=1,
):
    from sopilot.perception.types import EntityEvent, EntityEventType
    return EntityEvent(
        event_type=EntityEventType.ANOMALY,
        entity_id=entity_id, timestamp=timestamp, frame_number=frame_number,
        details={"detector": detector, "metric": metric, "z_score": z_score,
                 "description_ja": "テスト"},
    )


# ===========================================================================
# SigmaTuner.get_history() tests
# ===========================================================================


class TestSigmaTunerHistory(unittest.TestCase):
    def _make(self, **kw):
        from sopilot.perception.sigma_tuner import SigmaTuner
        return SigmaTuner(**kw)

    def test_get_history_empty_when_no_adjustments(self):
        st = self._make()
        self.assertEqual(st.get_history(), [])

    def test_get_history_returns_entries_after_adjustment(self):
        st = self._make(min_samples=5)
        ps = _make_pair_stats(total=10, confirmed=0, denied=10)
        st.compute_and_apply(_make_tuner_stats([ps]))
        h = st.get_history()
        self.assertGreater(len(h), 0)
        self.assertIn("detector", h[0])
        self.assertIn("new_sigma", h[0])
        self.assertIn("timestamp", h[0])

    def test_get_history_respects_limit(self):
        st = self._make(min_samples=1)
        # Apply multiple times by tweaking stats
        for i in range(5):
            ps = _make_pair_stats(total=10, confirmed=0, denied=10)
            st.compute_and_apply(_make_tuner_stats([ps]))
        h = st.get_history(limit=2)
        self.assertLessEqual(len(h), 2)

    def test_get_history_limit_zero_returns_all(self):
        st = self._make(min_samples=1)
        ps = _make_pair_stats(total=10, confirmed=0, denied=10)
        st.compute_and_apply(_make_tuner_stats([ps]))
        st.compute_and_apply(_make_tuner_stats([ps]))
        all_hist = st.get_history(limit=0)
        limited = st.get_history(limit=1)
        self.assertGreaterEqual(len(all_hist), len(limited))


# ===========================================================================
# ReviewQueue.get_review_history() tests
# ===========================================================================


class TestReviewQueueHistory(unittest.TestCase):
    def _q(self, **kw):
        from sopilot.perception.active_query import ReviewQueue
        return ReviewQueue(z_threshold=2.5, max_pending=20,
                           dedup_seconds=0.0, **kw)

    def test_get_review_history_empty_initially(self):
        q = self._q()
        self.assertEqual(q.get_review_history(), [])

    def test_get_review_history_returns_reviewed_items(self):
        q = self._q()
        ev = _make_anomaly_event(z_score=3.0)
        q.maybe_add(ev, 3.0)
        rid = q.get_pending()[0].review_id
        q.record_review(rid, confirmed=True)
        history = q.get_review_history()
        self.assertEqual(len(history), 1)
        self.assertIn("confirmed", history[0])
        self.assertIn("timestamp", history[0])
        self.assertIn("detector", history[0])
        self.assertTrue(history[0]["confirmed"])

    def test_get_review_history_respects_limit(self):
        q = self._q()
        for i in range(5):
            ev = _make_anomaly_event(detector=f"d{i}", metric="m", z_score=3.0 + i)
            q.maybe_add(ev, 3.0 + i)
            rid = q.get_pending()[-1].review_id
            q.record_review(rid, confirmed=(i % 2 == 0))
        h = q.get_review_history(limit=2)
        self.assertEqual(len(h), 2)


# ===========================================================================
# GET /vigil/perception/learning-dashboard endpoint tests
# ===========================================================================


class TestLearningDashboardEndpoint(unittest.TestCase):
    def setUp(self):
        import os
        import tempfile
        from pathlib import Path as _P
        from fastapi.testclient import TestClient
        from sopilot.main import create_app
        from sopilot.perception.active_query import ReviewQueue
        from sopilot.perception.sigma_tuner import SigmaTuner

        self._tmp = tempfile.TemporaryDirectory()
        root = _P(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "ld-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"

        self.app = create_app()
        self.client = TestClient(self.app)

        self.sigma_tuner = SigmaTuner(base_sigma=2.0)
        self.review_queue = ReviewQueue(z_threshold=2.5, dedup_seconds=0.0)

        engine = MagicMock()
        engine._sigma_tuner = self.sigma_tuner
        engine._review_queue = self.review_queue
        engine._anomaly_tuner = None
        engine.get_sigma_state.return_value = self.sigma_tuner.get_state()
        engine.get_adaptive_learner_state.return_value = {
            "adaptive_learner": {
                "total_observed": 0, "score_window_size": 0,
                "score_mean": 0.0, "score_std": 0.0,
                "drift_count": 0, "recalibration_count": 0,
                "last_recalibration": None, "ph_state": {},
            },
            "tuner": _make_tuner_stats([]),
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

    def test_learning_dashboard_200(self):
        r = self.client.get("/vigil/perception/learning-dashboard")
        self.assertEqual(r.status_code, 200)

    def test_learning_dashboard_has_health(self):
        r = self.client.get("/vigil/perception/learning-dashboard")
        self.assertIn("health", r.json())

    def test_learning_dashboard_has_sigma_history(self):
        r = self.client.get("/vigil/perception/learning-dashboard")
        self.assertIn("sigma_history", r.json())

    def test_learning_dashboard_has_confirm_timeline(self):
        r = self.client.get("/vigil/perception/learning-dashboard")
        self.assertIn("confirm_timeline", r.json())

    def test_learning_dashboard_has_queue_stats(self):
        r = self.client.get("/vigil/perception/learning-dashboard")
        self.assertIn("queue_stats", r.json())

    def test_learning_dashboard_health_ok_when_clean(self):
        r = self.client.get("/vigil/perception/learning-dashboard")
        self.assertEqual(r.json()["health"]["status"], "ok")

    def test_learning_dashboard_sigma_history_empty_initially(self):
        r = self.client.get("/vigil/perception/learning-dashboard")
        self.assertEqual(r.json()["sigma_history"], [])

    def test_learning_dashboard_confirm_timeline_reflects_reviews(self):
        ev = _make_anomaly_event(z_score=3.0)
        self.review_queue.maybe_add(ev, 3.0)
        rid = self.review_queue.get_pending()[0].review_id
        self.review_queue.record_review(rid, confirmed=True)

        r = self.client.get("/vigil/perception/learning-dashboard")
        timeline = r.json()["confirm_timeline"]
        self.assertEqual(len(timeline), 1)
        self.assertTrue(timeline[0]["confirmed"])

    def test_learning_dashboard_sigma_history_after_adjustment(self):
        ps = _make_pair_stats(total=10, confirmed=0, denied=10)
        self.sigma_tuner.compute_and_apply(_make_tuner_stats([ps]))

        r = self.client.get("/vigil/perception/learning-dashboard")
        sigma_h = r.json()["sigma_history"]
        self.assertGreater(len(sigma_h), 0)
        self.assertEqual(sigma_h[0]["detector"], "behavioral")


# ===========================================================================
# LearningHealthStatus computation tests
# ===========================================================================


class TestLearningDashboardHealth(unittest.TestCase):
    def setUp(self):
        import os
        import tempfile
        from pathlib import Path as _P
        from fastapi.testclient import TestClient
        from sopilot.main import create_app
        from sopilot.perception.active_query import ReviewQueue
        from sopilot.perception.sigma_tuner import SigmaTuner

        self._tmp = tempfile.TemporaryDirectory()
        root = _P(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "ld-health-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"

        self.app = create_app()
        self.client = TestClient(self.app)

        self.sigma_tuner = SigmaTuner(base_sigma=2.0)
        self.review_queue = ReviewQueue(z_threshold=2.5, dedup_seconds=0.0)

        self.mock_tuner = MagicMock()
        self.engine = MagicMock()
        self.engine._sigma_tuner = self.sigma_tuner
        self.engine._review_queue = self.review_queue
        self.engine._anomaly_tuner = self.mock_tuner
        self.engine.get_sigma_state.return_value = self.sigma_tuner.get_state()
        self.engine.get_adaptive_learner_state.return_value = {
            "adaptive_learner": {
                "total_observed": 0, "score_window_size": 0,
                "score_mean": 0.0, "score_std": 0.0,
                "drift_count": 0, "recalibration_count": 0,
                "last_recalibration": None, "ph_state": {},
            },
            "tuner": _make_tuner_stats([]),
        }
        self.engine.get_latest_frame_jpeg.return_value = None

        vlm = MagicMock()
        vlm._engine = self.engine
        self.app.state.vigil_pipeline._vlm = vlm

    def tearDown(self):
        import os
        self._tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                  "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
            os.environ.pop(k, None)

    def test_health_warning_when_sigma_near_max(self):
        # Set behavioral sigma near SIGMA_MAX (5.8+)
        self.sigma_tuner._detector_sigmas["behavioral"] = 5.9
        self.mock_tuner.get_stats.return_value = _make_tuner_stats([])

        r = self.client.get("/vigil/perception/learning-dashboard")
        data = r.json()
        self.assertNotEqual(data["health"]["status"], "ok")
        self.assertIn("behavioral", data["health"]["sigma_clamped_detectors"])

    def test_health_warning_when_high_fp_rate(self):
        # 90% FP rate with enough samples
        ps = _make_pair_stats(total=20, confirmed=2, denied=18)
        tuner_stats = _make_tuner_stats([ps])
        self.mock_tuner.get_stats.return_value = tuner_stats

        r = self.client.get("/vigil/perception/learning-dashboard")
        data = r.json()
        # fp_rate = 18/20 = 0.9 > HIGH_FP_THRESHOLD(0.6) and total=20 >= 10
        self.assertNotEqual(data["health"]["status"], "ok")
        issues = data["health"]["issues"]
        self.assertTrue(any("FP率" in i for i in issues))


if __name__ == "__main__":
    unittest.main()
