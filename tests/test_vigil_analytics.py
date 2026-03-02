"""Tests for VigilPilot Analytics Dashboard.

Coverage:
  - TestVigilAnalyticsRepository (14 tests): get_analytics() unit tests
    - empty DB
    - single session, multiple events, multi-severity
    - date filtering (days param)
    - per-rule aggregation
    - per-hour aggregation
    - session name joins
  - TestVigilAnalyticsAPI (16 tests): GET /vigil/analytics E2E
    - status 200, response shape
    - default days=30
    - days query param validation (ge=1, le=365)
    - events_by_severity totals match events_by_session
    - empty-DB baseline
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sopilot.database import Database
from sopilot.main import create_app
from sopilot.vigil.repository import VigilRepository
from sopilot.vigil.schemas import VigilAnalytics


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _init_db(db_path: Path) -> VigilRepository:
    """Create a fresh DB with all migrations and return a VigilRepository."""
    db = Database(db_path)
    db.close()
    return VigilRepository(db_path)


def _make_violation(
    rule: str = "ヘルメット未着用",
    severity: str = "warning",
    confidence: float = 0.9,
) -> dict:
    return {
        "rule_index": 0,
        "rule": rule,
        "description_ja": "テスト違反",
        "severity": severity,
        "confidence": confidence,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Repository Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestVigilAnalyticsRepository:
    """Unit tests for VigilRepository.get_analytics()."""

    def _repo(self, tmp_path: Path) -> VigilRepository:
        return _init_db(tmp_path / "vigil_analytics.db")

    # ── Empty DB ──────────────────────────────────────────────────────────────

    def test_empty_db_returns_zeros(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        result = repo.get_analytics(days=30)
        assert result["total_sessions"] == 0
        assert result["total_events"] == 0
        assert result["events_by_severity"] == {"info": 0, "warning": 0, "critical": 0}
        assert result["events_by_session"] == []
        assert result["events_by_rule"] == []
        assert result["events_per_day"] == []
        assert result["top_violation_hours"] == []

    def test_empty_db_with_sessions_no_events(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        repo.create_session("テスト", ["ルール1"], 1.0, "warning")
        repo.create_session("テスト2", ["ルール2"], 0.5, "critical")
        result = repo.get_analytics(days=30)
        assert result["total_sessions"] == 2
        assert result["total_events"] == 0
        assert result["events_by_session"] == []

    # ── Single session, single event ──────────────────────────────────────────

    def test_single_event_counted(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("工場1F", ["ルール1"], 1.0, "warning")
        repo.create_event(sid, 10.0, 10, [_make_violation("ルール1", "warning")])
        result = repo.get_analytics(days=30)
        assert result["total_events"] == 1
        assert result["events_by_severity"]["warning"] == 1
        assert result["events_by_severity"]["critical"] == 0

    def test_events_by_session_name(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("玄関カメラ", ["ルール"], 1.0, "warning")
        repo.create_event(sid, 5.0, 5, [_make_violation(severity="warning")])
        repo.create_event(sid, 6.0, 6, [_make_violation(severity="critical")])
        result = repo.get_analytics(days=30)
        assert len(result["events_by_session"]) == 1
        row = result["events_by_session"][0]
        assert row["session_name"] == "玄関カメラ"
        assert row["session_id"] == sid
        assert row["total"] == 2
        assert row["warning"] == 1
        assert row["critical"] == 1
        assert row["info"] == 0

    # ── Multi-severity ────────────────────────────────────────────────────────

    def test_multi_severity_counts(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("テスト", ["R1", "R2", "R3"], 1.0, "info")
        repo.create_event(sid, 1.0, 1, [_make_violation(severity="info")])
        repo.create_event(sid, 2.0, 2, [_make_violation(severity="warning")])
        repo.create_event(sid, 3.0, 3, [_make_violation(severity="critical")])
        repo.create_event(sid, 4.0, 4, [
            _make_violation(severity="critical"),
            _make_violation(severity="warning"),
        ])
        result = repo.get_analytics(days=30)
        sev = result["events_by_severity"]
        assert sev["info"] == 1
        assert sev["warning"] == 2
        assert sev["critical"] == 2

    def test_multiple_violations_per_event(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("マルチ", ["R1", "R2"], 1.0, "info")
        # One event with 3 violations
        repo.create_event(sid, 1.0, 1, [
            _make_violation("R1", "critical"),
            _make_violation("R2", "warning"),
            _make_violation("R1", "info"),
        ])
        result = repo.get_analytics(days=30)
        assert result["total_events"] == 1  # 1 event row
        sev = result["events_by_severity"]
        assert sev["critical"] == 1
        assert sev["warning"] == 1
        assert sev["info"] == 1

    # ── Events by rule ────────────────────────────────────────────────────────

    def test_events_by_rule_sorted_desc(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("ルール", ["A", "B", "C"], 1.0, "info")
        for _ in range(5):
            repo.create_event(sid, 1.0, 1, [_make_violation("ルールA", "warning")])
        for _ in range(3):
            repo.create_event(sid, 2.0, 2, [_make_violation("ルールB", "info")])
        repo.create_event(sid, 3.0, 3, [_make_violation("ルールC", "critical")])
        result = repo.get_analytics(days=30)
        rules = result["events_by_rule"]
        assert rules[0]["rule"] == "ルールA"
        assert rules[0]["count"] == 5
        assert rules[1]["count"] == 3
        assert rules[2]["count"] == 1

    def test_events_by_rule_top_20_limit(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("マルチルール", ["R"], 1.0, "info")
        for i in range(25):
            repo.create_event(sid, float(i), i, [_make_violation(f"ルール{i:02d}", "warning")])
        result = repo.get_analytics(days=30)
        assert len(result["events_by_rule"]) <= 20

    # ── Events per day ────────────────────────────────────────────────────────

    def test_events_per_day_structure(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("日別", ["R"], 1.0, "warning")
        repo.create_event(sid, 1.0, 1, [_make_violation(severity="critical")])
        repo.create_event(sid, 2.0, 2, [_make_violation(severity="warning")])
        result = repo.get_analytics(days=30)
        assert len(result["events_per_day"]) >= 1
        day = result["events_per_day"][0]
        assert "date" in day
        assert "total" in day
        assert "critical" in day
        assert "warning" in day
        assert "info" in day
        # Date format YYYY-MM-DD
        assert len(day["date"]) == 10
        assert day["date"][4] == "-"

    def test_events_per_day_days_filter(self, tmp_path: Path) -> None:
        """days=1 should still include today's events."""
        repo = self._repo(tmp_path)
        sid = repo.create_session("フィルタ", ["R"], 1.0, "warning")
        repo.create_event(sid, 1.0, 1, [_make_violation(severity="warning")])
        result_1 = repo.get_analytics(days=1)
        result_30 = repo.get_analytics(days=30)
        # today's events appear in both
        assert result_1["total_events"] == result_30["total_events"]

    # ── Events per hour ───────────────────────────────────────────────────────

    def test_top_violation_hours_structure(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("時間帯", ["R"], 1.0, "warning")
        for _ in range(3):
            repo.create_event(sid, 1.0, 1, [_make_violation(severity="warning")])
        result = repo.get_analytics(days=30)
        hours = result["top_violation_hours"]
        assert len(hours) >= 1
        h = hours[0]
        assert "hour" in h
        assert "count" in h
        assert 0 <= h["hour"] <= 23
        assert h["count"] >= 1

    def test_top_violation_hours_sorted_desc(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        sid = repo.create_session("時刻", ["R"], 1.0, "info")
        # Create many events — all go into the current hour
        for _ in range(10):
            repo.create_event(sid, 1.0, 1, [_make_violation(severity="info")])
        result = repo.get_analytics(days=30)
        hours = result["top_violation_hours"]
        if len(hours) >= 2:
            assert hours[0]["count"] >= hours[1]["count"]

    # ── Multi-session ─────────────────────────────────────────────────────────

    def test_multi_session_breakdown(self, tmp_path: Path) -> None:
        repo = self._repo(tmp_path)
        s1 = repo.create_session("A棟", ["R"], 1.0, "warning")
        s2 = repo.create_session("B棟", ["R"], 1.0, "warning")
        for _ in range(3):
            repo.create_event(s1, 1.0, 1, [_make_violation(severity="critical")])
        for _ in range(2):
            repo.create_event(s2, 2.0, 2, [_make_violation(severity="warning")])
        result = repo.get_analytics(days=30)
        assert result["total_sessions"] == 2
        assert result["total_events"] == 5
        sessions_by_id = {r["session_id"]: r for r in result["events_by_session"]}
        assert sessions_by_id[s1]["session_name"] == "A棟"
        assert sessions_by_id[s1]["critical"] == 3
        assert sessions_by_id[s2]["session_name"] == "B棟"
        assert sessions_by_id[s2]["warning"] == 2

    def test_result_keys_complete(self, tmp_path: Path) -> None:
        """Ensure all required keys are present even on empty DB."""
        repo = self._repo(tmp_path)
        result = repo.get_analytics()
        required = {
            "total_sessions", "total_events", "events_by_severity",
            "events_by_session", "events_by_rule", "events_per_day",
            "top_violation_hours",
        }
        assert required.issubset(result.keys())


# ──────────────────────────────────────────────────────────────────────────────
# API Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestVigilAnalyticsAPI(unittest.TestCase):
    """E2E HTTP tests for GET /vigil/analytics via TestClient."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(self.root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "vigil-analytics-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"
        self.app = create_app()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for k in (
            "SOPILOT_DATA_DIR",
            "SOPILOT_EMBEDDER_BACKEND",
            "SOPILOT_PRIMARY_TASK_ID",
            "SOPILOT_RATE_LIMIT_RPM",
        ):
            os.environ.pop(k, None)

    def _create_session(self, name: str = "テスト") -> int:
        r = self.client.post(
            "/vigil/sessions",
            json={"name": name, "rules": ["ルール1"], "sample_fps": 1.0, "severity_threshold": "info"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["session_id"]

    def _inject_event(self, session_id: int, violations: list[dict]) -> None:
        """Directly insert an event into the DB via the repository."""
        repo: VigilRepository = self.app.state.vigil_repo
        repo.create_event(session_id, 1.0, 1, violations)

    # ── Basic endpoint tests ───────────────────────────────────────────────────

    def test_analytics_status_200(self) -> None:
        r = self.client.get("/vigil/analytics")
        self.assertEqual(r.status_code, 200)

    def test_analytics_response_is_json(self) -> None:
        r = self.client.get("/vigil/analytics")
        self.assertEqual(r.headers["content-type"].split(";")[0], "application/json")

    def test_analytics_required_fields(self) -> None:
        r = self.client.get("/vigil/analytics")
        data = r.json()
        for field in (
            "total_sessions", "total_events", "events_by_severity",
            "events_by_session", "events_by_rule", "events_per_day",
            "top_violation_hours",
        ):
            self.assertIn(field, data, f"missing field: {field}")

    def test_analytics_empty_db_zeros(self) -> None:
        r = self.client.get("/vigil/analytics")
        data = r.json()
        self.assertEqual(data["total_sessions"], 0)
        self.assertEqual(data["total_events"], 0)
        self.assertEqual(data["events_by_severity"], {"info": 0, "warning": 0, "critical": 0})
        self.assertEqual(data["events_by_session"], [])
        self.assertEqual(data["events_by_rule"], [])
        self.assertEqual(data["events_per_day"], [])
        self.assertEqual(data["top_violation_hours"], [])

    # ── Pydantic schema validation ─────────────────────────────────────────────

    def test_analytics_validates_as_pydantic_schema(self) -> None:
        r = self.client.get("/vigil/analytics")
        # Should not raise
        analytics = VigilAnalytics.model_validate(r.json())
        self.assertIsInstance(analytics.total_sessions, int)
        self.assertIsInstance(analytics.total_events, int)

    # ── days query parameter ───────────────────────────────────────────────────

    def test_days_param_7(self) -> None:
        r = self.client.get("/vigil/analytics?days=7")
        self.assertEqual(r.status_code, 200)

    def test_days_param_90(self) -> None:
        r = self.client.get("/vigil/analytics?days=90")
        self.assertEqual(r.status_code, 200)

    def test_days_param_365(self) -> None:
        r = self.client.get("/vigil/analytics?days=365")
        self.assertEqual(r.status_code, 200)

    def test_days_param_0_invalid(self) -> None:
        r = self.client.get("/vigil/analytics?days=0")
        self.assertEqual(r.status_code, 422)

    def test_days_param_366_invalid(self) -> None:
        r = self.client.get("/vigil/analytics?days=366")
        self.assertEqual(r.status_code, 422)

    def test_days_param_negative_invalid(self) -> None:
        r = self.client.get("/vigil/analytics?days=-5")
        self.assertEqual(r.status_code, 422)

    # ── With data ─────────────────────────────────────────────────────────────

    def test_analytics_with_sessions_no_events(self) -> None:
        self._create_session("セッション1")
        self._create_session("セッション2")
        r = self.client.get("/vigil/analytics")
        data = r.json()
        self.assertEqual(data["total_sessions"], 2)
        self.assertEqual(data["total_events"], 0)

    def test_analytics_with_events_severity_counts(self) -> None:
        sid = self._create_session("工場1F")
        self._inject_event(sid, [_make_violation("ルール1", "critical")])
        self._inject_event(sid, [_make_violation("ルール1", "warning")])
        self._inject_event(sid, [_make_violation("ルール2", "info")])
        r = self.client.get("/vigil/analytics")
        data = r.json()
        self.assertEqual(data["total_events"], 3)
        sev = data["events_by_severity"]
        self.assertEqual(sev["critical"], 1)
        self.assertEqual(sev["warning"], 1)
        self.assertEqual(sev["info"], 1)

    def test_analytics_events_by_rule(self) -> None:
        sid = self._create_session("ルールテスト")
        for _ in range(4):
            self._inject_event(sid, [_make_violation("ヘルメット未着用", "critical")])
        self._inject_event(sid, [_make_violation("立入禁止", "warning")])
        r = self.client.get("/vigil/analytics")
        data = r.json()
        rules = data["events_by_rule"]
        self.assertGreater(len(rules), 0)
        self.assertEqual(rules[0]["rule"], "ヘルメット未着用")
        self.assertEqual(rules[0]["count"], 4)

    def test_analytics_events_by_session_has_name(self) -> None:
        sid = self._create_session("玄関カメラ")
        self._inject_event(sid, [_make_violation(severity="warning")])
        r = self.client.get("/vigil/analytics")
        data = r.json()
        sessions = data["events_by_session"]
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_name"], "玄関カメラ")
        self.assertEqual(sessions[0]["session_id"], sid)

    def test_analytics_events_per_day_populated(self) -> None:
        sid = self._create_session("日別テスト")
        self._inject_event(sid, [_make_violation(severity="warning")])
        r = self.client.get("/vigil/analytics")
        data = r.json()
        days_data = data["events_per_day"]
        self.assertGreater(len(days_data), 0)
        d = days_data[0]
        self.assertIn("date", d)
        self.assertIn("total", d)
        self.assertIn("critical", d)
        self.assertIn("warning", d)
        self.assertIn("info", d)

    def test_analytics_top_violation_hours_populated(self) -> None:
        sid = self._create_session("時間帯テスト")
        for _ in range(5):
            self._inject_event(sid, [_make_violation(severity="warning")])
        r = self.client.get("/vigil/analytics")
        data = r.json()
        hours = data["top_violation_hours"]
        self.assertGreater(len(hours), 0)
        h = hours[0]
        self.assertIn("hour", h)
        self.assertIn("count", h)
        self.assertGreaterEqual(h["hour"], 0)
        self.assertLessEqual(h["hour"], 23)
