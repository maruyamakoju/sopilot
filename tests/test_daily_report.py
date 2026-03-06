"""Tests for Phase 20: DailyReportGenerator + API.

Coverage:
    - DailyReportGenerator.generate(): all 6 sections
    - summary: health_score, grade, assessment
    - anomalies: fp_rate_by_detector
    - early_warning: overall_risk, peak_detector
    - responses: total_responses, recent_responses
    - recommendations: top-3 list
    - metadata: generated_at, window fields
    - Graceful when components are None
    - engine.get_daily_report()
    - API: GET /vigil/perception/daily-report
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_generator():
    from sopilot.perception.daily_report import DailyReportGenerator
    return DailyReportGenerator()


def _make_empty_engine():
    e = MagicMock()
    e.get_health_score.return_value = {"score": 100, "grade": "A", "factors": {}, "total_penalty": 0.0}
    e.get_health_history.return_value = {"history": [], "trend": {}, "sparkline": []}
    e.get_early_warning_responder_state.return_value = {
        "total_responses": 0, "cooldowns_remaining": {}, "recent_responses": []
    }
    e._early_warning = None
    e._anomaly_tuner = None
    e._review_queue = None
    return e


def _make_real_engine():
    from sopilot.perception.engine import build_perception_engine
    return build_perception_engine()


# ── TestDailyReportGenerator ──────────────────────────────────────────────────

class TestDailyReportGenerator(unittest.TestCase):

    def test_generate_returns_dict(self):
        gen = _make_generator()
        result = gen.generate(_make_empty_engine())
        self.assertIsInstance(result, dict)

    def test_generate_has_all_sections(self):
        gen = _make_generator()
        result = gen.generate(_make_empty_engine())
        for key in ("summary", "anomalies", "early_warning", "responses",
                    "recommendations", "metadata"):
            self.assertIn(key, result)

    # ── metadata ─────────────────────────────────────────────────────────────

    def test_metadata_has_generated_at(self):
        gen = _make_generator()
        before = time.time()
        result = gen.generate(_make_empty_engine())
        self.assertGreaterEqual(result["metadata"]["generated_at"], before)

    def test_metadata_window_is_24h(self):
        gen = _make_generator()
        result = gen.generate(_make_empty_engine())
        meta = result["metadata"]
        self.assertEqual(meta["window_hours"], 24)
        self.assertAlmostEqual(
            meta["window_end"] - meta["window_start"], 86400.0, delta=1.0
        )

    # ── summary ──────────────────────────────────────────────────────────────

    def test_summary_has_health_score(self):
        gen = _make_generator()
        result = gen.generate(_make_empty_engine())
        self.assertIn("health_score", result["summary"])
        self.assertEqual(result["summary"]["health_score"], 100)

    def test_summary_has_health_grade(self):
        gen = _make_generator()
        result = gen.generate(_make_empty_engine())
        self.assertIn("health_grade", result["summary"])
        self.assertEqual(result["summary"]["health_grade"], "A")

    def test_summary_overall_assessment_nonempty(self):
        gen = _make_generator()
        result = gen.generate(_make_empty_engine())
        assessment = result["summary"]["overall_assessment"]
        self.assertIsNotNone(assessment)
        self.assertGreater(len(assessment), 5)

    def test_summary_assessment_varies_by_health(self):
        from sopilot.perception.daily_report import _overall_assessment
        good = _overall_assessment(95, 0.1, 0)
        bad = _overall_assessment(30, 0.9, 5)
        self.assertNotEqual(good, bad)

    # ── anomalies ────────────────────────────────────────────────────────────

    def test_anomalies_section_is_dict(self):
        gen = _make_generator()
        result = gen.generate(_make_empty_engine())
        self.assertIsInstance(result["anomalies"], dict)

    def test_anomalies_fp_rate_from_tuner(self):
        gen = _make_generator()
        engine = _make_empty_engine()
        mock_tuner = MagicMock()
        mock_tuner.get_stats.return_value = {
            "pair_stats": [{"detector": "behavioral", "total": 100, "denied": 62}]
        }
        engine._anomaly_tuner = mock_tuner
        result = gen.generate(engine)
        fp_map = result["anomalies"]["fp_rate_by_detector"]
        self.assertIsNotNone(fp_map)
        self.assertIn("behavioral", fp_map)
        self.assertAlmostEqual(fp_map["behavioral"], 0.62, places=2)

    # ── early_warning ─────────────────────────────────────────────────────────

    def test_early_warning_section_is_dict(self):
        gen = _make_generator()
        result = gen.generate(_make_empty_engine())
        self.assertIsInstance(result["early_warning"], dict)

    def test_early_warning_fields(self):
        gen = _make_generator()
        result = gen.generate(_make_empty_engine())
        ew = result["early_warning"]
        for key in ("overall_risk", "risk_level", "peak_detector", "peak_risk"):
            self.assertIn(key, ew)

    def test_early_warning_populated_from_engine(self):
        gen = _make_generator()
        engine = _make_empty_engine()
        import threading
        mock_ew = MagicMock()
        mock_ew._lock = threading.Lock()
        mock_ew.get_state.return_value = {
            "overall_risk": 0.75,
            "overall_level": "high",
            "detectors": {
                "behavioral": {"risk_score": 0.75, "risk_level": "high"},
                "spatial": {"risk_score": 0.30, "risk_level": "medium"},
            }
        }
        engine._early_warning = mock_ew
        result = gen.generate(engine)
        ew = result["early_warning"]
        self.assertAlmostEqual(ew["overall_risk"], 0.75, places=2)
        self.assertEqual(ew["peak_detector"], "behavioral")

    # ── responses ────────────────────────────────────────────────────────────

    def test_responses_section_has_total(self):
        gen = _make_generator()
        engine = _make_empty_engine()
        engine.get_early_warning_responder_state.return_value = {
            "total_responses": 3, "cooldowns_remaining": {"behavioral": 120.0},
            "recent_responses": []
        }
        result = gen.generate(engine)
        self.assertEqual(result["responses"]["total_responses"], 3)
        self.assertEqual(result["responses"]["cooldowns_active"], 1)

    def test_responses_recent_capped_at_5(self):
        gen = _make_generator()
        engine = _make_empty_engine()
        recent = [{"detector": f"d{i}", "explanation_ja": "test",
                   "recommendations": [], "triggered_at": time.time()} for i in range(10)]
        engine.get_early_warning_responder_state.return_value = {
            "total_responses": 10, "cooldowns_remaining": {},
            "recent_responses": recent,
        }
        result = gen.generate(engine)
        self.assertLessEqual(len(result["responses"]["recent_responses"]), 5)

    # ── recommendations ───────────────────────────────────────────────────────

    def test_recommendations_is_list(self):
        gen = _make_generator()
        result = gen.generate(_make_empty_engine())
        self.assertIsInstance(result["recommendations"], list)

    def test_recommendations_max_3(self):
        gen = _make_generator()
        engine = _make_empty_engine()
        engine.get_early_warning_responder_state.return_value = {
            "total_responses": 1,
            "cooldowns_remaining": {},
            "recent_responses": [
                {"detector": "behavioral",
                 "recommendations": ["rec1", "rec2", "rec3", "rec4"],
                 "triggered_at": time.time()}
            ]
        }
        result = gen.generate(engine)
        self.assertLessEqual(len(result["recommendations"]), 3)

    def test_recommendations_have_action_field(self):
        gen = _make_generator()
        result = gen.generate(_make_empty_engine())
        for rec in result["recommendations"]:
            self.assertIn("action", rec)
            self.assertIn("detector", rec)

    # ── Graceful failures ─────────────────────────────────────────────────────

    def test_graceful_when_health_score_raises(self):
        gen = _make_generator()
        engine = _make_empty_engine()
        engine.get_health_score.side_effect = RuntimeError("boom")
        try:
            result = gen.generate(engine)
            self.assertIsInstance(result, dict)
        except Exception as exc:
            self.fail(f"generate() should not raise: {exc}")

    def test_graceful_when_all_components_none(self):
        """Engine with no perception components should still generate a report."""
        gen = _make_generator()
        engine = MagicMock()
        engine.get_health_score.side_effect = RuntimeError("no health")
        engine.get_health_history.side_effect = RuntimeError("no history")
        engine.get_early_warning_responder_state.side_effect = RuntimeError("no rs")
        engine._early_warning = None
        engine._anomaly_tuner = None
        engine._review_queue = None
        try:
            result = gen.generate(engine)
            self.assertIn("metadata", result)
        except Exception as exc:
            self.fail(f"generate() should not raise: {exc}")


# ── TestEngineGetDailyReport ──────────────────────────────────────────────────

class TestEngineGetDailyReport(unittest.TestCase):

    def test_engine_has_get_daily_report(self):
        engine = _make_real_engine()
        self.assertTrue(hasattr(engine, "get_daily_report"))

    def test_get_daily_report_returns_dict(self):
        engine = _make_real_engine()
        result = engine.get_daily_report()
        self.assertIsInstance(result, dict)

    def test_get_daily_report_has_all_sections(self):
        engine = _make_real_engine()
        result = engine.get_daily_report()
        for key in ("summary", "anomalies", "early_warning", "responses",
                    "recommendations", "metadata"):
            self.assertIn(key, result)


# ── TestDailyReportAPI ────────────────────────────────────────────────────────

class TestDailyReportAPI(unittest.TestCase):

    def _make_client(self):
        import os, tempfile
        from fastapi.testclient import TestClient
        from sopilot.main import create_app

        tmp = tempfile.mkdtemp()
        os.environ.update({
            "SOPILOT_DATA_DIR": tmp,
            "SOPILOT_EMBEDDER_BACKEND": "color-motion",
            "SOPILOT_PRIMARY_TASK_ID": "dr-api-test",
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

    def test_daily_report_endpoint_200(self):
        try:
            client, _ = self._make_client()
            r = client.get("/vigil/perception/daily-report")
            self.assertEqual(r.status_code, 200)
        finally:
            self._cleanup()

    def test_daily_report_has_summary(self):
        try:
            client, _ = self._make_client()
            r = client.get("/vigil/perception/daily-report")
            self.assertIn("summary", r.json())
        finally:
            self._cleanup()

    def test_daily_report_has_recommendations(self):
        try:
            client, _ = self._make_client()
            r = client.get("/vigil/perception/daily-report")
            body = r.json()
            self.assertIn("recommendations", body)
            self.assertIsInstance(body["recommendations"], list)
        finally:
            self._cleanup()

    def test_daily_report_metadata_present(self):
        try:
            client, _ = self._make_client()
            r = client.get("/vigil/perception/daily-report")
            meta = r.json().get("metadata", {})
            self.assertIn("generated_at", meta)
            self.assertIn("window_hours", meta)
        finally:
            self._cleanup()

    def test_daily_report_health_score_in_range(self):
        try:
            client, _ = self._make_client()
            r = client.get("/vigil/perception/daily-report")
            score = r.json()["summary"].get("health_score")
            if score is not None:
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 100)
        finally:
            self._cleanup()


if __name__ == "__main__":
    unittest.main()
