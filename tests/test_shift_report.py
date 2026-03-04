"""Tests for Phase 12B: ShiftReport (自動シフトレポート).

Coverage:
    - DetectorSummary: fp_rate, to_dict
    - ShiftReport: to_dict keys
    - ShiftReportGenerator: generate with empty/populated sessions,
      learning state integration, recommendations
    - _build_recommendations: various scenarios
    - API endpoint: GET /vigil/perception/shift-report/{session_id}
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helper: mock VigilRepository
# ---------------------------------------------------------------------------


def _make_mock_repo(
    session: dict | None = None,
    events: list[dict] | None = None,
) -> MagicMock:
    repo = MagicMock()
    if session is None:
        session = {
            "id": 1,
            "name": "テストセッション",
            "status": "completed",
            "created_at": "2026-03-05T09:00:00.000+00:00",
            "updated_at": "2026-03-05T10:00:00.000+00:00",
        }
    repo.get_session.return_value = session
    repo.list_events.return_value = events or []
    return repo


def _make_event(
    severity: str = "warning",
    violations: list[dict] | None = None,
) -> dict:
    return {
        "id": 1,
        "session_id": 1,
        "severity": severity,
        "timestamp_sec": 10.0,
        "violations": violations or [],
        "frame_jpeg": None,
    }


# ===========================================================================
# DetectorSummary tests
# ===========================================================================


class TestDetectorSummary(unittest.TestCase):
    def _make(self, **kw):
        from sopilot.vigil.shift_report import DetectorSummary
        defaults = dict(detector="behavioral", event_count=10)
        defaults.update(kw)
        return DetectorSummary(**defaults)

    def test_fp_rate_zero_when_no_feedback(self):
        ds = self._make(fp_count=0, tp_count=0)
        self.assertAlmostEqual(ds.fp_rate, 0.0)

    def test_fp_rate_computed_correctly(self):
        ds = self._make(fp_count=3, tp_count=7)
        self.assertAlmostEqual(ds.fp_rate, 0.3)

    def test_to_dict_keys(self):
        ds = self._make(fp_count=2, tp_count=8)
        d = ds.to_dict()
        for key in ("detector", "event_count", "fp_count", "tp_count", "fp_rate"):
            self.assertIn(key, d)

    def test_fp_rate_100_percent(self):
        ds = self._make(fp_count=5, tp_count=0)
        self.assertAlmostEqual(ds.fp_rate, 1.0)


# ===========================================================================
# ShiftReport dataclass tests
# ===========================================================================


class TestShiftReport(unittest.TestCase):
    def _make(self, **kw):
        from sopilot.vigil.shift_report import ShiftReport
        defaults = dict(
            session_id=1, session_name="test",
            generated_at=time.time(),
            duration_seconds=3600.0, status="completed",
            total_events=5,
        )
        defaults.update(kw)
        return ShiftReport(**defaults)

    def test_to_dict_has_required_keys(self):
        report = self._make()
        d = report.to_dict()
        for key in ("session_id", "session_name", "generated_at", "duration_seconds",
                    "status", "total_events", "events_by_severity", "top_rules",
                    "top_detectors", "tuner_feedback_total", "tuner_confirm_rate",
                    "sigma_adjustments_total", "detector_sigmas", "drift_events",
                    "recalibrations", "review_pending", "review_confirmed",
                    "review_denied", "recommendations"):
            self.assertIn(key, d)

    def test_to_dict_recommendations_is_list(self):
        report = self._make(recommendations=["test rec"])
        d = report.to_dict()
        self.assertIsInstance(d["recommendations"], list)

    def test_to_dict_duration_none_allowed(self):
        report = self._make(duration_seconds=None)
        d = report.to_dict()
        self.assertIsNone(d["duration_seconds"])


# ===========================================================================
# ShiftReportGenerator tests
# ===========================================================================


class TestShiftReportGenerator(unittest.TestCase):
    def _gen(self):
        from sopilot.vigil.shift_report import ShiftReportGenerator
        return ShiftReportGenerator()

    def test_generate_basic_report(self):
        gen = self._gen()
        repo = _make_mock_repo()
        report = gen.generate(session_id=1, vigil_repo=repo)
        self.assertEqual(report.session_id, 1)

    def test_generate_session_name(self):
        gen = self._gen()
        repo = _make_mock_repo(session={
            "id": 1, "name": "夜間シフト", "status": "completed",
            "created_at": None, "updated_at": None,
        })
        report = gen.generate(1, repo)
        self.assertEqual(report.session_name, "夜間シフト")

    def test_generate_session_not_found_raises(self):
        gen = self._gen()
        repo = _make_mock_repo()
        repo.get_session.return_value = None
        with self.assertRaises(ValueError):
            gen.generate(999, repo)

    def test_generate_total_events_zero_when_empty(self):
        gen = self._gen()
        repo = _make_mock_repo(events=[])
        report = gen.generate(1, repo)
        self.assertEqual(report.total_events, 0)

    def test_generate_total_events_counted(self):
        gen = self._gen()
        events = [_make_event("critical"), _make_event("warning"), _make_event("info")]
        repo = _make_mock_repo(events=events)
        report = gen.generate(1, repo)
        self.assertEqual(report.total_events, 3)

    def test_generate_events_by_severity(self):
        gen = self._gen()
        events = [_make_event("critical"), _make_event("critical"), _make_event("warning")]
        repo = _make_mock_repo(events=events)
        report = gen.generate(1, repo)
        self.assertEqual(report.events_by_severity.get("critical", 0), 2)
        self.assertEqual(report.events_by_severity.get("warning", 0), 1)

    def test_generate_top_rules_counted(self):
        gen = self._gen()
        events = [
            _make_event(violations=[{"rule": "rule_A", "source": "world_model"}]),
            _make_event(violations=[{"rule": "rule_A", "source": "world_model"}]),
            _make_event(violations=[{"rule": "rule_B", "source": "world_model"}]),
        ]
        repo = _make_mock_repo(events=events)
        report = gen.generate(1, repo)
        # rule_A should appear in top_rules
        rules = {r["rule"]: r["count"] for r in report.top_rules}
        self.assertIn("rule_A", rules)
        self.assertEqual(rules["rule_A"], 2)

    def test_generate_recommendations_not_empty(self):
        gen = self._gen()
        repo = _make_mock_repo()
        report = gen.generate(1, repo)
        self.assertGreater(len(report.recommendations), 0)

    def test_generate_with_engine_learning_state(self):
        gen = self._gen()
        repo = _make_mock_repo()

        engine = MagicMock()
        engine.get_adaptive_learner_state.return_value = {
            "adaptive_learner": {
                "total_observed": 50, "score_window_size": 50,
                "score_mean": 2.1, "score_std": 0.5,
                "drift_count": 2, "recalibration_count": 1,
                "last_recalibration": None, "ph_state": {},
            },
            "tuner": {
                "total_feedback": 8, "confirmed": 4, "denied": 4,
                "overall_confirm_rate": 0.5,
                "pairs_tracked": 2, "pairs_suppressed": 0, "pairs_trusted": 0,
                "last_tuning": 0.0, "pair_stats": [],
                "suppressed_pairs": [], "trusted_pairs": [],
                "min_samples_for_tuning": 10,
            },
        }
        engine._sigma_tuner = None
        engine._anomaly_tuner = None
        engine._review_queue = None

        report = gen.generate(1, repo, engine=engine)
        self.assertEqual(report.drift_events, 2)
        self.assertEqual(report.recalibrations, 1)
        self.assertEqual(report.tuner_feedback_total, 8)

    def test_generate_duration_computed(self):
        gen = self._gen()
        repo = _make_mock_repo(session={
            "id": 1, "name": "s", "status": "completed",
            "created_at": "2026-03-05T09:00:00.000000+00:00",
            "updated_at": "2026-03-05T10:00:00.000000+00:00",
        })
        report = gen.generate(1, repo)
        self.assertAlmostEqual(report.duration_seconds, 3600.0, delta=5)

    def test_generate_status_preserved(self):
        gen = self._gen()
        repo = _make_mock_repo(session={
            "id": 1, "name": "s", "status": "running",
            "created_at": None, "updated_at": None,
        })
        report = gen.generate(1, repo)
        self.assertEqual(report.status, "running")


# ===========================================================================
# Recommendations tests
# ===========================================================================


class TestBuildRecommendations(unittest.TestCase):
    def _call(self, **kw):
        from sopilot.vigil.shift_report import _build_recommendations
        defaults = dict(
            events_by_severity={},
            tuner_stats={"total_feedback": 0, "pair_stats": []},
            sigma_state={},
            detector_summaries=[],
            drift_events=0,
            review_pending=0,
        )
        defaults.update(kw)
        return _build_recommendations(**defaults)

    def test_no_events_returns_some_recommendation(self):
        # With no events but zero feedback, at least one recommendation is generated
        recs = self._call(tuner_stats={"total_feedback": 15, "pair_stats": []})
        self.assertGreater(len(recs), 0)
        self.assertTrue(any("正常" in r or "異常なし" in r for r in recs))

    def test_many_critical_events_flagged(self):
        recs = self._call(events_by_severity={"critical": 10})
        self.assertTrue(any("重大" in r for r in recs))

    def test_drift_events_flagged(self):
        recs = self._call(drift_events=3)
        self.assertTrue(any("ドリフト" in r for r in recs))

    def test_pending_reviews_flagged(self):
        recs = self._call(review_pending=8)
        self.assertTrue(any("確認待ち" in r for r in recs))

    def test_zero_feedback_flagged(self):
        recs = self._call(tuner_stats={"total_feedback": 0, "pair_stats": []})
        self.assertTrue(any("フィードバック" in r for r in recs))

    def test_sigma_adjustments_noted(self):
        recs = self._call(sigma_state={"total_adjustments": 5})
        self.assertTrue(any("sigma" in r.lower() or "自動調整" in r or "安定" in r for r in recs))


# ===========================================================================
# API endpoint tests
# ===========================================================================


class TestShiftReportEndpoint(unittest.TestCase):
    def setUp(self):
        import os
        import tempfile
        from pathlib import Path as _P
        from fastapi.testclient import TestClient
        from sopilot.main import create_app

        self._tmp = tempfile.TemporaryDirectory()
        root = _P(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "shift-report-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"

        self.app = create_app()
        self.client = TestClient(self.app)

        # Inject mock perception engine (no real engine needed)
        engine = MagicMock()
        engine.get_adaptive_learner_state.return_value = {
            "adaptive_learner": {
                "total_observed": 0, "score_window_size": 0,
                "score_mean": 0.0, "score_std": 0.0,
                "drift_count": 0, "recalibration_count": 0,
                "last_recalibration": None, "ph_state": {},
            },
            "tuner": {
                "total_feedback": 0, "confirmed": 0, "denied": 0,
                "overall_confirm_rate": 0.0,
                "pairs_tracked": 0, "pairs_suppressed": 0, "pairs_trusted": 0,
                "last_tuning": 0.0, "pair_stats": [],
                "suppressed_pairs": [], "trusted_pairs": [],
                "min_samples_for_tuning": 10,
            },
        }
        engine._sigma_tuner = None
        engine._anomaly_tuner = None
        engine._review_queue = None
        engine.get_sigma_state.return_value = None

        vlm = MagicMock()
        vlm._engine = engine
        self.app.state.vigil_pipeline._vlm = vlm

    def tearDown(self):
        import os
        self._tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                   "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
            os.environ.pop(k, None)

    def _create_session(self) -> int:
        r = self.client.post("/vigil/sessions", json={
            "name": "テストセッション", "rules": ["安全帽着用"]
        })
        data = r.json()
        # SessionResponse uses session_id field
        return data.get("session_id") or data.get("id")

    def test_shift_report_200_for_existing_session(self):
        sid = self._create_session()
        r = self.client.get(f"/vigil/perception/shift-report/{sid}")
        self.assertEqual(r.status_code, 200)

    def test_shift_report_404_for_missing_session(self):
        r = self.client.get("/vigil/perception/shift-report/99999")
        self.assertEqual(r.status_code, 404)

    def test_shift_report_has_session_id(self):
        sid = self._create_session()
        r = self.client.get(f"/vigil/perception/shift-report/{sid}")
        self.assertEqual(r.json()["session_id"], sid)

    def test_shift_report_has_recommendations(self):
        sid = self._create_session()
        r = self.client.get(f"/vigil/perception/shift-report/{sid}")
        recs = r.json()["recommendations"]
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)

    def test_shift_report_has_required_keys(self):
        sid = self._create_session()
        r = self.client.get(f"/vigil/perception/shift-report/{sid}")
        data = r.json()
        for key in ("session_id", "total_events", "events_by_severity",
                    "top_rules", "tuner_feedback_total", "sigma_adjustments_total",
                    "drift_events", "recommendations"):
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
