"""Tests for VigilPilot — Violation Acknowledgment and Session Templates.

Coverage:
  - TestAcknowledgeAPI (7 tests): PATCH /vigil/events/{id}/acknowledge
  - TestTemplatesAPI (5 tests): GET /vigil/templates
  - TestRepositoryAcknowledge (3 tests): direct VigilRepository.acknowledge_event
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from sopilot.database import Database
from sopilot.main import create_app
from sopilot.vigil.pipeline import VigilPipeline
from sopilot.vigil.repository import VigilRepository
from sopilot.vigil.schemas import VLMResult
from sopilot.vigil.vlm import VLMClient


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (copied locally — do not import from test_vigil.py)
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


class _MockVLMNoViolation(VLMClient):
    """VLM stub — always returns no violation."""

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        return VLMResult(has_violation=False, violations=[], raw_text="{}")


# ──────────────────────────────────────────────────────────────────────────────
# Repository-only helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_repo(tmp_path: Path) -> VigilRepository:
    db_path = tmp_path / "vigil.db"
    _init_db(db_path)
    return VigilRepository(db_path)


def _make_event(repo: VigilRepository) -> tuple[int, int]:
    """Create a session + one event. Returns (session_id, event_id)."""
    sid = repo.create_session("テスト", ["ルール1"], 1.0, "warning")
    violations = [{
        "rule_index": 0,
        "rule": "ルール1",
        "description_ja": "違反テスト",
        "severity": "warning",
        "confidence": 0.85,
    }]
    eid = repo.create_event(sid, 1.0, 1, violations)
    return sid, eid


# ──────────────────────────────────────────────────────────────────────────────
# TestRepositoryAcknowledge — direct VigilRepository unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestRepositoryAcknowledge:
    """Direct repository tests for acknowledge_event."""

    def test_acknowledge_returns_true_for_existing_event(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        _sid, eid = _make_event(repo)
        result = repo.acknowledge_event(eid, "inspector_01")
        assert result is True

    def test_acknowledge_returns_false_for_nonexistent_event(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        result = repo.acknowledge_event(99999, "inspector_01")
        assert result is False

    def test_acknowledged_at_is_set(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        _sid, eid = _make_event(repo)
        before = time.time()
        repo.acknowledge_event(eid, "tester")
        after = time.time()

        row = repo.get_event(eid)
        assert row is not None
        assert row["acknowledged_at"] is not None
        assert row["acknowledged_by"] == "tester"
        # acknowledged_at should be an ISO timestamp string
        assert isinstance(row["acknowledged_at"], str)
        assert len(row["acknowledged_at"]) > 0
        _ = before, after  # timing bounds just for reference


# ──────────────────────────────────────────────────────────────────────────────
# API test base class
# ──────────────────────────────────────────────────────────────────────────────


class _VigilAPIBase(unittest.TestCase):
    """Shared setUp / tearDown for API tests."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(self.root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "vigil-ack-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"
        self.app = create_app()
        self.client = TestClient(self.app)
        # Inject mock VLM — no real API calls
        self.app.state.vigil_pipeline._vlm = _MockVLMWithViolation()

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for k in (
            "SOPILOT_DATA_DIR",
            "SOPILOT_EMBEDDER_BACKEND",
            "SOPILOT_PRIMARY_TASK_ID",
            "SOPILOT_RATE_LIMIT_RPM",
        ):
            os.environ.pop(k, None)

    def _create_session(self, name: str = "テスト") -> dict:
        r = self.client.post(
            "/vigil/sessions",
            json={
                "name": name,
                "rules": ["ルール1", "ルール2"],
                "sample_fps": 1.0,
                "severity_threshold": "warning",
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def _upload_video_and_wait(self, sid: int, timeout: float = 15.0) -> dict:
        video_path = self.root / f"upload_{sid}.avi"
        _make_video(video_path, seconds=2, fps=8.0)
        with video_path.open("rb") as fh:
            r = self.client.post(
                f"/vigil/sessions/{sid}/analyze",
                files={"file": ("upload.avi", fh, "video/x-msvideo")},
            )
        self.assertEqual(r.status_code, 200, r.text)

        deadline = time.time() + timeout
        while time.time() < deadline:
            row = self.client.get(f"/vigil/sessions/{sid}").json()
            if row["status"] in ("completed", "failed"):
                return row
            time.sleep(0.2)
        return self.client.get(f"/vigil/sessions/{sid}").json()

    def _get_first_event_id(self, sid: int) -> int | None:
        r = self.client.get(f"/vigil/sessions/{sid}/events")
        events = r.json()
        if not events:
            return None
        return events[0]["event_id"]


# ──────────────────────────────────────────────────────────────────────────────
# TestAcknowledgeAPI
# ──────────────────────────────────────────────────────────────────────────────


class TestAcknowledgeAPI(_VigilAPIBase):
    """HTTP tests for PATCH /vigil/events/{event_id}/acknowledge."""

    def test_acknowledge_event_returns_200(self) -> None:
        """Create session, run analysis with mock VLM that returns violations,
        acknowledge an event, check 200 response with acknowledged: true."""
        session = self._create_session()
        sid = session["session_id"]
        self._upload_video_and_wait(sid)

        eid = self._get_first_event_id(sid)
        self.assertIsNotNone(eid, "Expected at least one violation event after analysis")

        r = self.client.patch(
            f"/vigil/events/{eid}/acknowledge",
            json={"acknowledged_by": "operator_01"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["acknowledged"])
        self.assertEqual(data["event_id"], eid)
        self.assertEqual(data["acknowledged_by"], "operator_01")

    def test_acknowledge_unknown_event_404(self) -> None:
        """PATCH /vigil/events/9999/acknowledge returns 404."""
        r = self.client.patch(
            "/vigil/events/9999/acknowledge",
            json={"acknowledged_by": "operator"},
        )
        self.assertEqual(r.status_code, 404)

    def test_acknowledged_event_appears_in_list(self) -> None:
        """After acknowledging, GET /vigil/sessions/{id}/events shows acknowledged_at is set."""
        session = self._create_session()
        sid = session["session_id"]
        self._upload_video_and_wait(sid)

        eid = self._get_first_event_id(sid)
        self.assertIsNotNone(eid)

        self.client.patch(
            f"/vigil/events/{eid}/acknowledge",
            json={"acknowledged_by": "supervisor"},
        )

        events = self.client.get(f"/vigil/sessions/{sid}/events").json()
        ack_event = next((e for e in events if e["event_id"] == eid), None)
        self.assertIsNotNone(ack_event)
        self.assertIsNotNone(ack_event["acknowledged_at"])
        self.assertEqual(ack_event["acknowledged_by"], "supervisor")

    def test_acknowledge_default_acknowledged_by(self) -> None:
        """When body is empty JSON {}, acknowledged_by defaults to 'operator'."""
        session = self._create_session()
        sid = session["session_id"]
        self._upload_video_and_wait(sid)

        eid = self._get_first_event_id(sid)
        self.assertIsNotNone(eid)

        r = self.client.patch(
            f"/vigil/events/{eid}/acknowledge",
            json={},
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["acknowledged_by"], "operator")

    def test_acknowledge_custom_acknowledged_by(self) -> None:
        """Custom acknowledged_by string is stored and returned."""
        session = self._create_session()
        sid = session["session_id"]
        self._upload_video_and_wait(sid)

        eid = self._get_first_event_id(sid)
        self.assertIsNotNone(eid)

        r = self.client.patch(
            f"/vigil/events/{eid}/acknowledge",
            json={"acknowledged_by": "田中安全管理者"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["acknowledged_by"], "田中安全管理者")

    def test_acknowledge_idempotent(self) -> None:
        """Acknowledging twice returns 200 both times."""
        session = self._create_session()
        sid = session["session_id"]
        self._upload_video_and_wait(sid)

        eid = self._get_first_event_id(sid)
        self.assertIsNotNone(eid)

        r1 = self.client.patch(
            f"/vigil/events/{eid}/acknowledge",
            json={"acknowledged_by": "operator"},
        )
        self.assertEqual(r1.status_code, 200)

        r2 = self.client.patch(
            f"/vigil/events/{eid}/acknowledge",
            json={"acknowledged_by": "operator"},
        )
        self.assertEqual(r2.status_code, 200)

    def test_repository_acknowledge_event(self) -> None:
        """Direct repository test: create event, acknowledge, verify fields."""
        tmp = Path(tempfile.mkdtemp())
        try:
            repo = _make_repo(tmp)
            _sid, eid = _make_event(repo)

            # Before acknowledgment
            row_before = repo.get_event(eid)
            self.assertIsNone(row_before["acknowledged_at"])
            self.assertIsNone(row_before["acknowledged_by"])

            # Acknowledge
            result = repo.acknowledge_event(eid, "test_operator")
            self.assertTrue(result)

            # After acknowledgment
            row_after = repo.get_event(eid)
            self.assertIsNotNone(row_after["acknowledged_at"])
            self.assertEqual(row_after["acknowledged_by"], "test_operator")
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# TestTemplatesAPI
# ──────────────────────────────────────────────────────────────────────────────


class TestTemplatesAPI(_VigilAPIBase):
    """HTTP tests for GET /vigil/templates."""

    def test_list_templates_returns_200(self) -> None:
        r = self.client.get("/vigil/templates")
        self.assertEqual(r.status_code, 200, r.text)

    def test_list_templates_count(self) -> None:
        """At least 5 templates returned."""
        r = self.client.get("/vigil/templates")
        self.assertEqual(r.status_code, 200)
        templates = r.json()
        self.assertGreaterEqual(len(templates), 5)

    def test_template_has_required_fields(self) -> None:
        """Each template has id, name, description, rules, sample_fps, severity_threshold."""
        r = self.client.get("/vigil/templates")
        self.assertEqual(r.status_code, 200)
        required_fields = {"id", "name", "description", "rules", "sample_fps", "severity_threshold"}
        for template in r.json():
            for field in required_fields:
                self.assertIn(field, template, f"Field '{field}' missing from template {template.get('id')}")

    def test_template_rules_are_non_empty(self) -> None:
        """Each template has at least 1 rule."""
        r = self.client.get("/vigil/templates")
        self.assertEqual(r.status_code, 200)
        for template in r.json():
            self.assertGreater(
                len(template["rules"]),
                0,
                f"Template '{template.get('id')}' has no rules",
            )

    def test_template_severity_valid(self) -> None:
        """Each template's severity_threshold is in (info, warning, critical)."""
        r = self.client.get("/vigil/templates")
        self.assertEqual(r.status_code, 200)
        valid_severities = {"info", "warning", "critical"}
        for template in r.json():
            self.assertIn(
                template["severity_threshold"],
                valid_severities,
                f"Template '{template.get('id')}' has invalid severity '{template.get('severity_threshold')}'",
            )
