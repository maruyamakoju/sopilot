"""Tests for Phase 19: HealthHistoryStore + API.

Coverage:
    - HealthHistoryStore.record(): stores snapshot, rate-limits, force-record
    - HealthHistoryStore.get_history(): time-window filter, newest-first order
    - HealthHistoryStore.get_trend(): avg, min, max, improvement
    - HealthHistoryStore.get_sparkline_data(): downsampling
    - HealthHistoryStore.clear()
    - Persistence: _save/_load round-trip
    - No-op when state_path=None
    - engine._health_history integration
    - engine.get_health_score() auto-records (rate-limited)
    - engine.get_health_history() returns dict with history/trend/sparkline
    - API: GET /vigil/perception/health-score/history
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_store(tmp_path=None, interval=0.0):
    from sopilot.perception.health_history import HealthHistoryStore
    return HealthHistoryStore(state_path=tmp_path, record_interval_seconds=interval)


def _rec(store, score=90, grade="A", factors=None, total_penalty=0.0):
    return store.record(score, grade, factors or {}, total_penalty)


# ── TestHealthHistoryStore ────────────────────────────────────────────────────

class TestHealthHistoryStore(unittest.TestCase):

    def test_record_returns_true_when_stored(self):
        s = _make_store(interval=0.0)
        self.assertTrue(_rec(s, score=85, grade="B"))

    def test_record_appends_snapshot(self):
        s = _make_store(interval=0.0)
        _rec(s, score=80, grade="B")
        hist = s.get_history()
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["score"], 80)
        self.assertEqual(hist[0]["grade"], "B")

    def test_record_rate_limited_returns_false(self):
        s = _make_store(interval=3600.0)
        _rec(s, score=90)
        result = _rec(s, score=80)  # should be throttled
        self.assertFalse(result)

    def test_record_force_bypasses_rate_limit(self):
        s = _make_store(interval=3600.0)
        _rec(s, score=90)
        result = s.record(80, "B", {}, force=True)
        self.assertTrue(result)
        self.assertEqual(len(s.get_history()), 2)

    def test_record_stores_factor_penalties(self):
        s = _make_store(interval=0.0)
        factors = {"early_warning": {"penalty": 5.0}, "fp_rate": {"penalty": 3.0}}
        _rec(s, factors=factors)
        snap = s.get_history()[0]
        self.assertIn("factor_penalties", snap)
        self.assertEqual(snap["factor_penalties"]["early_warning"], 5.0)

    def test_record_enforces_max_records(self):
        from sopilot.perception.health_history import HealthHistoryStore
        s = HealthHistoryStore(max_records=3, record_interval_seconds=0.0)
        for i in range(5):
            s.record(i * 10, "C", {})
        self.assertEqual(len(s.get_history(days=365)), 3)

    def test_get_history_newest_first(self):
        s = _make_store(interval=0.0)
        for score in [70, 80, 90]:
            s.record(score, "B", {})
        hist = s.get_history()
        scores = [h["score"] for h in hist]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_get_history_time_window_filter(self):
        from sopilot.perception.health_history import HealthHistoryStore
        s = HealthHistoryStore(record_interval_seconds=0.0)
        # Inject old record manually
        old_snap = {"score": 50, "grade": "D", "total_penalty": 50.0,
                    "recorded_at": time.time() - 10 * 86400, "factor_penalties": {}}
        with s._lock:
            s._records.append(old_snap)
        s.record(90, "A", {})
        # 1-day window should exclude the 10-day-old record
        hist = s.get_history(days=1.0)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["score"], 90)

    def test_get_history_limit(self):
        s = _make_store(interval=0.0)
        for i in range(5):
            s.record(i * 10 + 50, "C", {})
        hist = s.get_history(limit=2)
        self.assertEqual(len(hist), 2)

    def test_get_history_returns_empty_when_no_records(self):
        s = _make_store(interval=0.0)
        self.assertEqual(s.get_history(), [])

    def test_get_trend_none_when_empty(self):
        s = _make_store(interval=0.0)
        trend = s.get_trend()
        self.assertIsNone(trend["avg_score"])
        self.assertEqual(trend["record_count"], 0)

    def test_get_trend_avg_score(self):
        s = _make_store(interval=0.0)
        for score in [60, 80, 100]:
            s.record(score, "C", {})
        trend = s.get_trend(days=365)
        self.assertAlmostEqual(trend["avg_score"], 80.0, places=0)

    def test_get_trend_improvement_positive(self):
        """If latest score > first score, improvement is positive."""
        s = _make_store(interval=0.0)
        s.record(70, "C", {})
        s.record(90, "A", {})
        trend = s.get_trend(days=365)
        self.assertGreater(trend["improvement"], 0)

    def test_get_trend_improvement_negative(self):
        s = _make_store(interval=0.0)
        s.record(90, "A", {})
        s.record(70, "C", {})
        trend = s.get_trend(days=365)
        self.assertLess(trend["improvement"], 0)

    def test_get_sparkline_data_oldest_first(self):
        s = _make_store(interval=0.0)
        for score in [60, 70, 80, 90]:
            s.record(score, "B", {})
        sl = s.get_sparkline_data(days=365)
        scores = [p["score"] for p in sl]
        self.assertEqual(scores, sorted(scores))

    def test_get_sparkline_downsamples_to_points(self):
        s = _make_store(interval=0.0)
        for i in range(50):
            s.record(80, "B", {})
        sl = s.get_sparkline_data(days=365, points=10)
        self.assertLessEqual(len(sl), 10)

    def test_clear_removes_all_records(self):
        s = _make_store(interval=0.0)
        s.record(80, "B", {})
        s.clear()
        self.assertEqual(s.get_history(), [])

    def test_clear_resets_rate_limit(self):
        s = _make_store(interval=3600.0)
        s.record(80, "B", {})
        s.clear()
        self.assertTrue(s.record(90, "A", {}))

    # ── Persistence ───────────────────────────────────────────────────────────

    def test_save_creates_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "health.json"
            s = _make_store(tmp_path=path, interval=0.0)
            s.record(85, "B", {})
            self.assertTrue(path.exists())

    def test_load_restores_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "health.json"
            s1 = _make_store(tmp_path=path, interval=0.0)
            s1.record(77, "B", {})
            s2 = _make_store(tmp_path=path, interval=0.0)
            hist = s2.get_history()
            self.assertEqual(len(hist), 1)
            self.assertEqual(hist[0]["score"], 77)

    def test_no_file_written_when_state_path_none(self):
        s = _make_store(tmp_path=None, interval=0.0)
        s.record(80, "B", {})  # Should not raise, no file written
        self.assertEqual(len(s.get_history()), 1)

    def test_load_nonexistent_path_is_graceful(self):
        from sopilot.perception.health_history import HealthHistoryStore
        try:
            s = HealthHistoryStore(state_path="/nonexistent/path.json")
            self.assertEqual(s.get_history(), [])
        except Exception as exc:
            self.fail(f"Should not raise: {exc}")


# ── TestEngineHealthHistory ───────────────────────────────────────────────────

class TestEngineHealthHistory(unittest.TestCase):

    def test_engine_has_health_history(self):
        from sopilot.perception.engine import build_perception_engine
        engine = build_perception_engine()
        self.assertIsNotNone(engine._health_history)

    def test_get_health_history_returns_dict(self):
        from sopilot.perception.engine import build_perception_engine
        engine = build_perception_engine()
        result = engine.get_health_history()
        self.assertIsInstance(result, dict)
        for key in ("history", "trend", "sparkline"):
            self.assertIn(key, result)

    def test_get_health_score_records_to_history(self):
        from sopilot.perception.engine import build_perception_engine
        from sopilot.perception.health_history import HealthHistoryStore
        engine = build_perception_engine()
        # Replace with zero-interval store
        engine._health_history = HealthHistoryStore(record_interval_seconds=0.0)
        engine.get_health_score(record=True)
        hist = engine._health_history.get_history()
        self.assertGreater(len(hist), 0)

    def test_get_health_score_no_record_when_false(self):
        from sopilot.perception.engine import build_perception_engine
        from sopilot.perception.health_history import HealthHistoryStore
        engine = build_perception_engine()
        store = HealthHistoryStore(record_interval_seconds=0.0)
        engine._health_history = store
        engine.get_health_score(record=False)
        self.assertEqual(len(store.get_history()), 0)


# ── TestHealthHistoryAPI ──────────────────────────────────────────────────────

class TestHealthHistoryAPI(unittest.TestCase):

    def _make_client(self):
        import os, tempfile
        from fastapi.testclient import TestClient
        from sopilot.main import create_app

        tmp = tempfile.mkdtemp()
        os.environ.update({
            "SOPILOT_DATA_DIR": tmp,
            "SOPILOT_EMBEDDER_BACKEND": "color-motion",
            "SOPILOT_PRIMARY_TASK_ID": "hh-api-test",
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

    def test_history_endpoint_200(self):
        try:
            client, _ = self._make_client()
            r = client.get("/vigil/perception/health-score/history")
            self.assertEqual(r.status_code, 200)
        finally:
            self._cleanup()

    def test_history_response_has_required_keys(self):
        try:
            client, _ = self._make_client()
            r = client.get("/vigil/perception/health-score/history")
            body = r.json()
            for key in ("history", "trend", "sparkline"):
                self.assertIn(key, body)
        finally:
            self._cleanup()

    def test_history_days_param(self):
        try:
            client, _ = self._make_client()
            r = client.get("/vigil/perception/health-score/history?days=30")
            self.assertEqual(r.status_code, 200)
        finally:
            self._cleanup()

    def test_health_score_and_history_both_return_200(self):
        """GET /health-score and /health-score/history both work together."""
        try:
            client, _ = self._make_client()
            r1 = client.get("/vigil/perception/health-score")
            r2 = client.get("/vigil/perception/health-score/history")
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r2.status_code, 200)
            # Both responses should have valid structure
            self.assertIn("score", r1.json())
            self.assertIn("history", r2.json())
        finally:
            self._cleanup()


if __name__ == "__main__":
    unittest.main()
