"""Tests for Phase 18: PerceptionHealthScorer.

Coverage:
    - PerceptionHealthScorer.compute(): score, grade, factors
    - Grade thresholds: A/B/C/D/F
    - Score floor at 0
    - Each factor contributes correctly
    - Missing engine components handled gracefully
    - engine.get_health_score()
    - API: GET /vigil/perception/health-score
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_scorer():
    from sopilot.perception.perception_health import PerceptionHealthScorer
    return PerceptionHealthScorer()


def _make_empty_engine():
    """Engine with no phase 12-16 components."""
    e = MagicMock()
    e._early_warning = None
    e._anomaly_tuner = None
    e._sigma_tuner = None
    e._review_queue = None
    e._early_warning_responder = None
    return e


def _make_clean_engine():
    """Real engine via build_perception_engine() — all components present."""
    from sopilot.perception.engine import build_perception_engine
    return build_perception_engine()


# ── TestPerceptionHealthScorer ────────────────────────────────────────────────

class TestPerceptionHealthScorer(unittest.TestCase):

    def test_compute_returns_dict(self):
        scorer = _make_scorer()
        result = scorer.compute(_make_empty_engine())
        self.assertIsInstance(result, dict)

    def test_compute_has_required_keys(self):
        scorer = _make_scorer()
        result = scorer.compute(_make_empty_engine())
        for key in ("score", "grade", "factors", "total_penalty", "computed_at"):
            self.assertIn(key, result)

    def test_perfect_score_when_no_signals(self):
        scorer = _make_scorer()
        result = scorer.compute(_make_empty_engine())
        self.assertEqual(result["score"], 100)

    def test_grade_A_for_score_90_plus(self):
        from sopilot.perception.perception_health import _grade
        self.assertEqual(_grade(100), "A")
        self.assertEqual(_grade(90), "A")

    def test_grade_B_for_75_to_89(self):
        from sopilot.perception.perception_health import _grade
        self.assertEqual(_grade(89), "B")
        self.assertEqual(_grade(75), "B")

    def test_grade_C_for_60_to_74(self):
        from sopilot.perception.perception_health import _grade
        self.assertEqual(_grade(74), "C")
        self.assertEqual(_grade(60), "C")

    def test_grade_D_for_40_to_59(self):
        from sopilot.perception.perception_health import _grade
        self.assertEqual(_grade(59), "D")
        self.assertEqual(_grade(40), "D")

    def test_grade_F_for_below_40(self):
        from sopilot.perception.perception_health import _grade
        self.assertEqual(_grade(39), "F")
        self.assertEqual(_grade(0), "F")

    def test_score_is_int_in_0_to_100(self):
        scorer = _make_scorer()
        result = scorer.compute(_make_clean_engine())
        self.assertIsInstance(result["score"], int)
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)

    def test_factors_has_all_five_keys(self):
        scorer = _make_scorer()
        result = scorer.compute(_make_empty_engine())
        for key in ("early_warning", "fp_rate", "sigma_drift", "anomaly_burst", "review_backlog"):
            self.assertIn(key, result["factors"])

    def test_each_factor_has_penalty_key(self):
        scorer = _make_scorer()
        result = scorer.compute(_make_empty_engine())
        for fkey, fval in result["factors"].items():
            self.assertIn("penalty", fval, f"Factor {fkey} missing 'penalty'")

    def test_computed_at_is_recent(self):
        scorer = _make_scorer()
        before = time.time()
        result = scorer.compute(_make_empty_engine())
        self.assertGreaterEqual(result["computed_at"], before)

    def test_score_degraded_by_high_early_warning_risk(self):
        """High overall_risk from EarlyWarningEngine should reduce score."""
        scorer = _make_scorer()
        engine = _make_empty_engine()

        mock_ew = MagicMock()
        mock_ew.get_state.return_value = {"overall_risk": 1.0, "overall_level": "high", "detectors": {}}
        mock_ew._lock = __import__("threading").Lock()
        mock_ew._drift_velocity = {}
        mock_ew._anomaly_timestamps = {}
        mock_ew._burst_window_seconds = 300
        engine._early_warning = mock_ew

        result = scorer.compute(engine)
        # W_EARLY_WARNING = 30 → max 30-pt penalty → score ≤ 70
        self.assertLessEqual(result["score"], 70)
        self.assertGreater(result["factors"]["early_warning"]["penalty"], 0)

    def test_score_degraded_by_high_fp_rate(self):
        """High FP rate should increase fp_rate penalty."""
        scorer = _make_scorer()
        engine = _make_empty_engine()

        mock_tuner = MagicMock()
        mock_tuner.get_stats.return_value = {
            "pair_stats": [{"detector": "behavioral", "total": 100, "denied": 70}]
        }
        engine._anomaly_tuner = mock_tuner

        result = scorer.compute(engine)
        # 70/100=0.70, fp_norm=1.0, penalty=W_FP_RATE=25
        self.assertAlmostEqual(result["factors"]["fp_rate"]["penalty"], 25.0, places=0)

    def test_score_degraded_by_review_backlog(self):
        """Large review queue backlog should increase backlog penalty."""
        scorer = _make_scorer()
        engine = _make_empty_engine()

        mock_rq = MagicMock()
        mock_rq.get_stats.return_value = {"pending_count": 50}  # = BACKLOG_CAP
        engine._review_queue = mock_rq

        result = scorer.compute(engine)
        # 50/50=1.0 → penalty=W_REVIEW_BACKLOG=10
        self.assertAlmostEqual(result["factors"]["review_backlog"]["penalty"], 10.0, places=0)

    def test_score_floor_at_zero(self):
        """Even with maximum penalties on all factors, score cannot go below 0."""
        scorer = _make_scorer()
        engine = _make_empty_engine()

        # Mock all factors at maximum penalty
        mock_ew = MagicMock()
        mock_ew.get_state.return_value = {"overall_risk": 1.0, "overall_level": "high", "detectors": {}}
        import threading
        mock_ew._lock = threading.Lock()
        mock_ew._drift_velocity = {"behavioral": 1.0, "spatial": 1.0}
        mock_ew._anomaly_timestamps = {"behavioral": [time.time()] * 25, "spatial": [time.time()] * 25}
        mock_ew._burst_window_seconds = 300
        engine._early_warning = mock_ew

        mock_tuner = MagicMock()
        mock_tuner.get_stats.return_value = {
            "pair_stats": [{"detector": "behavioral", "total": 100, "denied": 100}]
        }
        engine._anomaly_tuner = mock_tuner

        mock_rq = MagicMock()
        mock_rq.get_stats.return_value = {"pending_count": 100}
        engine._review_queue = mock_rq

        result = scorer.compute(engine)
        self.assertGreaterEqual(result["score"], 0)

    def test_compute_graceful_when_ew_raises(self):
        """Exceptions in early_warning query should not propagate."""
        scorer = _make_scorer()
        engine = _make_empty_engine()

        mock_ew = MagicMock()
        mock_ew.get_state.side_effect = RuntimeError("boom")
        import threading
        mock_ew._lock = threading.Lock()
        mock_ew._drift_velocity = {}
        mock_ew._anomaly_timestamps = {}
        mock_ew._burst_window_seconds = 300
        engine._early_warning = mock_ew

        try:
            result = scorer.compute(engine)
            self.assertIsInstance(result["score"], int)
        except Exception as exc:
            self.fail(f"compute() raised unexpectedly: {exc}")

    def test_compute_graceful_when_review_queue_raises(self):
        scorer = _make_scorer()
        engine = _make_empty_engine()

        mock_rq = MagicMock()
        mock_rq.get_stats.side_effect = RuntimeError("rq boom")
        engine._review_queue = mock_rq

        try:
            result = scorer.compute(engine)
            self.assertIsInstance(result["score"], int)
        except Exception as exc:
            self.fail(f"compute() raised unexpectedly: {exc}")


# ── TestEngineGetHealthScore ──────────────────────────────────────────────────

class TestEngineGetHealthScore(unittest.TestCase):

    def test_engine_has_get_health_score_method(self):
        engine = _make_clean_engine()
        self.assertTrue(hasattr(engine, "get_health_score"))

    def test_get_health_score_returns_dict(self):
        engine = _make_clean_engine()
        result = engine.get_health_score()
        self.assertIsInstance(result, dict)

    def test_get_health_score_has_score_and_grade(self):
        engine = _make_clean_engine()
        result = engine.get_health_score()
        self.assertIn("score", result)
        self.assertIn("grade", result)

    def test_get_health_score_in_range(self):
        engine = _make_clean_engine()
        result = engine.get_health_score()
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)


# ── TestHealthScoreAPI ────────────────────────────────────────────────────────

class TestHealthScoreAPI(unittest.TestCase):

    def _make_client(self):
        import os, tempfile
        from fastapi.testclient import TestClient
        from sopilot.main import create_app

        tmp = tempfile.mkdtemp()
        os.environ.update({
            "SOPILOT_DATA_DIR": tmp,
            "SOPILOT_EMBEDDER_BACKEND": "color-motion",
            "SOPILOT_PRIMARY_TASK_ID": "health-test",
            "SOPILOT_RATE_LIMIT_RPM": "0",
            "VIGIL_VLM_BACKEND": "perception",
        })
        return TestClient(create_app()), tmp

    def _cleanup(self):
        import os
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                  "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM",
                  "VIGIL_VLM_BACKEND"):
            os.environ.pop(k, None)

    def test_health_score_endpoint_200(self):
        try:
            client, _ = self._make_client()
            r = client.get("/vigil/perception/health-score")
            self.assertEqual(r.status_code, 200)
        finally:
            self._cleanup()

    def test_health_score_response_has_score(self):
        try:
            client, _ = self._make_client()
            r = client.get("/vigil/perception/health-score")
            body = r.json()
            self.assertIn("score", body)
        finally:
            self._cleanup()

    def test_health_score_response_has_grade(self):
        try:
            client, _ = self._make_client()
            r = client.get("/vigil/perception/health-score")
            body = r.json()
            self.assertIn("grade", body)
            self.assertIn(body["grade"], ("A", "B", "C", "D", "F"))
        finally:
            self._cleanup()

    def test_health_score_response_has_factors(self):
        try:
            client, _ = self._make_client()
            r = client.get("/vigil/perception/health-score")
            body = r.json()
            self.assertIn("factors", body)
            self.assertIsInstance(body["factors"], dict)
        finally:
            self._cleanup()

    def test_health_score_in_range_0_to_100(self):
        try:
            client, _ = self._make_client()
            r = client.get("/vigil/perception/health-score")
            body = r.json()
            score = body["score"]
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)
        finally:
            self._cleanup()

    def test_health_score_fresh_engine_is_acceptable(self):
        """Fresh engine should have a reasonable health score (grade B or above, ≥75)."""
        try:
            client, _ = self._make_client()
            r = client.get("/vigil/perception/health-score")
            body = r.json()
            # Engine may load existing FP/tuner data so we allow grade B (≥75)
            self.assertGreaterEqual(body["score"], 75)
        finally:
            self._cleanup()


if __name__ == "__main__":
    unittest.main()
