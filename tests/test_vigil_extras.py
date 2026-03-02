"""Tests for VigilPilot extra features:
  - Feature 1: Webhook CRUD and notification (10 tests)
  - Feature 2: CSV violation export (5 tests)
  - Feature 3: SSE real-time event feed (3 tests)
"""

from __future__ import annotations

import csv
import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from sopilot.database import Database
from sopilot.main import create_app
from sopilot.vigil.pipeline import VigilPipeline, _fire_webhook
from sopilot.vigil.repository import VigilRepository
from sopilot.vigil.schemas import VLMResult
from sopilot.vigil.vlm import VLMClient


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (copied from test_vigil.py to avoid cross-test-file imports)
# ──────────────────────────────────────────────────────────────────────────────


def _make_video(path: Path, seconds: int = 2, fps: float = 8.0) -> None:
    """Create a minimal synthetic AVI video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (96, 96))
    total = int(seconds * fps)
    for i in range(total):
        val = (i * 30) % 255
        frame = np.full((96, 96, 3), (val, 100, 200), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _init_db(db_path: Path) -> str:
    """Initialise a fresh SQLite database with all tables + migrations."""
    db = Database(db_path)
    db.close()
    return str(db_path)


class _MockVLMNoViolation(VLMClient):
    """VLM stub — always returns no violation."""

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        return VLMResult(has_violation=False, violations=[], raw_text="{}")


class _MockVLMWithViolation(VLMClient):
    """VLM stub — always returns one warning-level violation."""

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        return VLMResult(
            has_violation=True,
            violations=[{
                "rule_index": 0,
                "rule": rules[0] if rules else "test rule",
                "description_ja": "テスト違反が検出されました",
                "severity": "warning",
                "confidence": 0.9,
            }],
            raw_text='{"has_violation": true}',
        )


# ──────────────────────────────────────────────────────────────────────────────
# Base class for API tests
# ──────────────────────────────────────────────────────────────────────────────


class _VigilAPIBase(unittest.TestCase):
    """Shared setUp / tearDown for VigilPilot API tests."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(self.root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "vigil-extras-test"
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

    def _create_session(self, name: str = "テスト", severity_threshold: str = "warning") -> dict:
        r = self.client.post(
            "/vigil/sessions",
            json={
                "name": name,
                "rules": ["ルール1", "ルール2"],
                "sample_fps": 1.0,
                "severity_threshold": severity_threshold,
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()


# ──────────────────────────────────────────────────────────────────────────────
# Feature 1: Webhook CRUD via API
# ──────────────────────────────────────────────────────────────────────────────


class TestWebhookAPI(_VigilAPIBase):
    """Tests for PUT/GET/DELETE /vigil/sessions/{id}/webhook."""

    def test_set_webhook_returns_200_with_fields(self) -> None:
        sid = self._create_session()["session_id"]
        r = self.client.put(
            f"/vigil/sessions/{sid}/webhook",
            json={"url": "https://example.com/hook", "min_severity": "warning"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["session_id"], sid)
        self.assertEqual(data["url"], "https://example.com/hook")
        self.assertEqual(data["min_severity"], "warning")

    def test_get_webhook_returns_configured_values(self) -> None:
        sid = self._create_session()["session_id"]
        self.client.put(
            f"/vigil/sessions/{sid}/webhook",
            json={"url": "https://example.com/hook", "min_severity": "critical"},
        )
        r = self.client.get(f"/vigil/sessions/{sid}/webhook")
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["url"], "https://example.com/hook")
        self.assertEqual(data["min_severity"], "critical")

    def test_get_webhook_404_when_not_set(self) -> None:
        sid = self._create_session()["session_id"]
        r = self.client.get(f"/vigil/sessions/{sid}/webhook")
        self.assertEqual(r.status_code, 404, r.text)

    def test_clear_webhook_returns_cleared_true(self) -> None:
        sid = self._create_session()["session_id"]
        self.client.put(
            f"/vigil/sessions/{sid}/webhook",
            json={"url": "https://example.com/hook", "min_severity": "warning"},
        )
        r = self.client.delete(f"/vigil/sessions/{sid}/webhook")
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["cleared"])
        self.assertEqual(data["session_id"], sid)

    def test_get_webhook_404_after_clear(self) -> None:
        sid = self._create_session()["session_id"]
        self.client.put(
            f"/vigil/sessions/{sid}/webhook",
            json={"url": "https://example.com/hook", "min_severity": "warning"},
        )
        self.client.delete(f"/vigil/sessions/{sid}/webhook")
        r = self.client.get(f"/vigil/sessions/{sid}/webhook")
        self.assertEqual(r.status_code, 404, r.text)

    def test_set_webhook_unknown_session_404(self) -> None:
        r = self.client.put(
            "/vigil/sessions/9999/webhook",
            json={"url": "https://example.com/hook", "min_severity": "warning"},
        )
        self.assertEqual(r.status_code, 404)

    def test_get_webhook_unknown_session_404(self) -> None:
        r = self.client.get("/vigil/sessions/9999/webhook")
        self.assertEqual(r.status_code, 404)

    def test_clear_webhook_unknown_session_404(self) -> None:
        r = self.client.delete("/vigil/sessions/9999/webhook")
        self.assertEqual(r.status_code, 404)

    def test_set_webhook_invalid_severity_422(self) -> None:
        sid = self._create_session()["session_id"]
        r = self.client.put(
            f"/vigil/sessions/{sid}/webhook",
            json={"url": "https://example.com/hook", "min_severity": "extreme"},
        )
        self.assertEqual(r.status_code, 422)

    def test_update_webhook_overwrites_previous(self) -> None:
        sid = self._create_session()["session_id"]
        self.client.put(
            f"/vigil/sessions/{sid}/webhook",
            json={"url": "https://old.example.com/hook", "min_severity": "info"},
        )
        self.client.put(
            f"/vigil/sessions/{sid}/webhook",
            json={"url": "https://new.example.com/hook", "min_severity": "critical"},
        )
        r = self.client.get(f"/vigil/sessions/{sid}/webhook")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["url"], "https://new.example.com/hook")
        self.assertEqual(data["min_severity"], "critical")


# ──────────────────────────────────────────────────────────────────────────────
# Feature 1: Webhook firing on violation (unit test with mock)
# ──────────────────────────────────────────────────────────────────────────────


class TestWebhookFiring(unittest.TestCase):
    """Tests that _fire_webhook fires httpx.post in background when configured."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        _init_db(self.tmp / "test.db")
        self.repo = VigilRepository(self.tmp / "test.db")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_webhook_fired_when_configured(self) -> None:
        sid = self.repo.create_session("webhook fire", ["r"], 1.0, "warning")
        self.repo.set_webhook(sid, "https://example.com/notify", "warning")
        eid = self.repo.create_event(sid, 1.0, 1, [{"severity": "warning"}])

        violations = [{"severity": "warning", "rule": "r", "description_ja": "d", "confidence": 0.9}]
        with patch("sopilot.vigil.pipeline.httpx") as mock_httpx:
            mock_post = MagicMock()
            mock_httpx.post = mock_post
            _fire_webhook(self.repo, sid, eid, 1.0, violations)
            # Give the daemon thread a moment to execute
            time.sleep(0.3)
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        # Verify the webhook URL was used
        assert call_kwargs[0][0] == "https://example.com/notify"

    def test_webhook_not_fired_when_not_configured(self) -> None:
        sid = self.repo.create_session("no webhook", ["r"], 1.0, "warning")
        eid = self.repo.create_event(sid, 1.0, 1, [])

        violations = [{"severity": "warning"}]
        with patch("sopilot.vigil.pipeline.httpx") as mock_httpx:
            mock_post = MagicMock()
            mock_httpx.post = mock_post
            _fire_webhook(self.repo, sid, eid, 1.0, violations)
            time.sleep(0.1)
        mock_post.assert_not_called()

    def test_webhook_not_fired_below_min_severity(self) -> None:
        sid = self.repo.create_session("sev filter", ["r"], 1.0, "warning")
        self.repo.set_webhook(sid, "https://example.com/notify", "critical")
        eid = self.repo.create_event(sid, 1.0, 1, [])

        # Only info-level violations — below critical threshold
        violations = [{"severity": "info"}]
        with patch("sopilot.vigil.pipeline.httpx") as mock_httpx:
            mock_post = MagicMock()
            mock_httpx.post = mock_post
            _fire_webhook(self.repo, sid, eid, 1.0, violations)
            time.sleep(0.1)
        mock_post.assert_not_called()

    def test_webhook_payload_contains_expected_fields(self) -> None:
        sid = self.repo.create_session("payload test", ["r"], 1.0, "warning")
        self.repo.set_webhook(sid, "https://example.com/notify", "warning")
        eid = self.repo.create_event(sid, 2.5, 5, [])

        violations = [{"severity": "warning", "rule": "r", "description_ja": "違反", "confidence": 0.8}]
        captured: list[dict] = []

        def _fake_post(url, json=None, timeout=None):  # noqa: ANN001
            captured.append({"url": url, "json": json})

        with patch("sopilot.vigil.pipeline.httpx") as mock_httpx:
            mock_httpx.post = _fake_post
            _fire_webhook(self.repo, sid, eid, 2.5, violations)
            time.sleep(0.3)

        self.assertEqual(len(captured), 1)
        payload = captured[0]["json"]
        self.assertEqual(payload["session_id"], sid)
        self.assertEqual(payload["event_id"], eid)
        self.assertAlmostEqual(payload["timestamp_sec"], 2.5)
        self.assertEqual(payload["violations"], violations)


# ──────────────────────────────────────────────────────────────────────────────
# Feature 2: CSV export
# ──────────────────────────────────────────────────────────────────────────────


class TestCSVExport(_VigilAPIBase):
    """Tests for GET /vigil/sessions/{id}/report/csv."""

    def test_csv_404_on_unknown_session(self) -> None:
        r = self.client.get("/vigil/sessions/9999/report/csv")
        self.assertEqual(r.status_code, 404)

    def test_csv_returns_header_only_when_no_events(self) -> None:
        sid = self._create_session()["session_id"]
        r = self.client.get(f"/vigil/sessions/{sid}/report/csv")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("text/csv", r.headers["content-type"])
        reader = csv.DictReader(io.StringIO(r.text))
        rows = list(reader)
        self.assertEqual(rows, [])
        # Headers must match spec
        expected_headers = {
            "event_id", "timestamp_sec", "frame_number",
            "rule_index", "rule", "description_ja", "severity", "confidence", "bboxes",
        }
        assert reader.fieldnames is not None
        self.assertEqual(set(reader.fieldnames), expected_headers)

    def test_csv_has_correct_columns(self) -> None:
        sid = self._create_session()["session_id"]
        # Inject violation events directly via repo
        repo = self.app.state.vigil_repo
        violations = [{
            "rule_index": 0,
            "rule": "ヘルメット未着用",
            "description_ja": "作業者がヘルメットを着用していません",
            "severity": "warning",
            "confidence": 0.92,
            "bboxes": None,
        }]
        repo.create_event(sid, 3.5, 7, violations)

        r = self.client.get(f"/vigil/sessions/{sid}/report/csv")
        self.assertEqual(r.status_code, 200, r.text)
        reader = csv.DictReader(io.StringIO(r.text))
        rows = list(reader)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["rule"], "ヘルメット未着用")
        self.assertEqual(row["description_ja"], "作業者がヘルメットを着用していません")
        self.assertEqual(row["severity"], "warning")
        self.assertAlmostEqual(float(row["confidence"]), 0.92, places=2)
        self.assertEqual(row["frame_number"], "7")

    def test_csv_content_disposition_header(self) -> None:
        sid = self._create_session()["session_id"]
        r = self.client.get(f"/vigil/sessions/{sid}/report/csv")
        self.assertEqual(r.status_code, 200)
        cd = r.headers.get("content-disposition", "")
        self.assertIn("attachment", cd)
        self.assertIn(f"session_{sid}", cd)

    def test_csv_multiple_violations_multiple_rows(self) -> None:
        sid = self._create_session()["session_id"]
        repo = self.app.state.vigil_repo
        violations = [
            {
                "rule_index": 0, "rule": "ルール0", "description_ja": "違反0",
                "severity": "warning", "confidence": 0.8,
            },
            {
                "rule_index": 1, "rule": "ルール1", "description_ja": "違反1",
                "severity": "critical", "confidence": 0.95,
            },
        ]
        repo.create_event(sid, 1.0, 1, violations)
        repo.create_event(sid, 2.0, 2, violations[:1])

        r = self.client.get(f"/vigil/sessions/{sid}/report/csv")
        self.assertEqual(r.status_code, 200, r.text)
        reader = csv.DictReader(io.StringIO(r.text))
        rows = list(reader)
        # 2 violations in first event + 1 in second = 3 rows
        self.assertEqual(len(rows), 3)


# ──────────────────────────────────────────────────────────────────────────────
# Feature 3: SSE real-time feed
# ──────────────────────────────────────────────────────────────────────────────


class TestSSEFeed(_VigilAPIBase):
    """Tests for GET /vigil/sessions/{id}/events/stream."""

    def test_sse_404_on_unknown_session(self) -> None:
        r = self.client.get("/vigil/sessions/9999/events/stream")
        self.assertEqual(r.status_code, 404)

    def test_sse_returns_200_with_event_stream_content_type(self) -> None:
        sid = self._create_session()["session_id"]
        # Complete the session so the SSE generator terminates quickly
        repo = self.app.state.vigil_repo
        repo.update_session_status(sid, "completed", total_frames_analyzed=0, violation_count=0)
        r = self.client.get(f"/vigil/sessions/{sid}/events/stream")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("text/event-stream", r.headers["content-type"])

    def test_sse_emits_status_change_on_completed_session(self) -> None:
        sid = self._create_session()["session_id"]
        repo = self.app.state.vigil_repo
        # Pre-complete so the SSE loop exits after first iteration
        repo.update_session_status(sid, "completed", total_frames_analyzed=0, violation_count=0)
        r = self.client.get(f"/vigil/sessions/{sid}/events/stream")
        self.assertEqual(r.status_code, 200)
        # Response body should contain a status_change event
        self.assertIn("status_change", r.text)
        self.assertIn("completed", r.text)


# ──────────────────────────────────────────────────────────────────────────────
# Repository unit tests for new methods
# ──────────────────────────────────────────────────────────────────────────────


class TestRepositoryWebhookMethods(unittest.TestCase):
    """Direct unit tests for the new VigilRepository webhook methods."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        _init_db(self.tmp / "repo.db")
        self.repo = VigilRepository(self.tmp / "repo.db")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_set_and_get_webhook(self) -> None:
        sid = self.repo.create_session("wh test", ["r"], 1.0, "warning")
        self.repo.set_webhook(sid, "https://example.com/wh", "warning")
        result = self.repo.get_webhook(sid)
        self.assertIsNotNone(result)
        url, min_sev = result  # type: ignore[misc]
        self.assertEqual(url, "https://example.com/wh")
        self.assertEqual(min_sev, "warning")

    def test_get_webhook_returns_none_when_not_set(self) -> None:
        sid = self.repo.create_session("no wh", ["r"], 1.0, "warning")
        self.assertIsNone(self.repo.get_webhook(sid))

    def test_get_webhook_returns_none_for_unknown_session(self) -> None:
        self.assertIsNone(self.repo.get_webhook(9999))

    def test_clear_webhook(self) -> None:
        sid = self.repo.create_session("clear wh", ["r"], 1.0, "warning")
        self.repo.set_webhook(sid, "https://example.com/wh", "critical")
        self.repo.clear_webhook(sid)
        self.assertIsNone(self.repo.get_webhook(sid))

    def test_list_events_since_returns_only_newer_events(self) -> None:
        sid = self.repo.create_session("since test", ["r"], 1.0, "warning")
        eid1 = self.repo.create_event(sid, 1.0, 1, [])
        eid2 = self.repo.create_event(sid, 2.0, 2, [])
        eid3 = self.repo.create_event(sid, 3.0, 3, [])

        events = self.repo.list_events_since(sid, eid1)
        ids = [e["id"] for e in events]
        self.assertNotIn(eid1, ids)
        self.assertIn(eid2, ids)
        self.assertIn(eid3, ids)

    def test_list_events_since_zero_returns_all(self) -> None:
        sid = self.repo.create_session("since zero", ["r"], 1.0, "warning")
        self.repo.create_event(sid, 1.0, 1, [])
        self.repo.create_event(sid, 2.0, 2, [])
        events = self.repo.list_events_since(sid, 0)
        self.assertEqual(len(events), 2)

    def test_list_events_since_empty_when_no_newer(self) -> None:
        sid = self.repo.create_session("since empty", ["r"], 1.0, "warning")
        eid = self.repo.create_event(sid, 1.0, 1, [])
        events = self.repo.list_events_since(sid, eid)
        self.assertEqual(events, [])
