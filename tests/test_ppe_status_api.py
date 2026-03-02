"""Tests for PPE Status API endpoints.

Coverage:
  - GET /vigil/sessions/{id}/ppe-status — no pose estimator (non-perception backend)
  - GET /vigil/sessions/{id}/ppe-status — with mock pose results
  - Summary calculation (helmets_ok, vests_ok counts)
  - POST /vigil/sessions/{id}/pose-enable — enable / disable toggle
  - Error cases: session not found → 404
  - PerceptionVLMClient.get_pose_results / set_pose_enabled unit tests
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from sopilot.database import Database
from sopilot.main import create_app
from sopilot.perception.types import BBox, PPEStatus, PoseKeypoint, PoseResult
from sopilot.vigil.schemas import VLMResult
from sopilot.vigil.vlm import VLMClient


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _init_db(db_path: Path) -> None:
    db = Database(db_path)
    db.close()


def _make_pose_result(
    *,
    has_helmet: bool = True,
    helmet_conf: float = 0.85,
    has_vest: bool = True,
    vest_conf: float = 0.72,
    pose_conf: float = 0.91,
) -> PoseResult:
    """Build a minimal PoseResult for testing."""
    ppe = PPEStatus(
        has_helmet=has_helmet,
        helmet_confidence=helmet_conf,
        has_vest=has_vest,
        vest_confidence=vest_conf,
    )
    keypoints = [PoseKeypoint(x=0.5, y=0.5, confidence=0.9)] * 17
    return PoseResult(
        person_bbox=BBox(x1=0.1, y1=0.1, x2=0.4, y2=0.9),
        keypoints=keypoints,
        ppe=ppe,
        pose_confidence=pose_conf,
    )


class _MockVLMNoViolation(VLMClient):
    """Plain VLM stub — no pose support, no violations."""

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        return VLMResult(has_violation=False, violations=[], raw_text="{}")


class _MockPerceptionVLM(VLMClient):
    """Minimal mock of PerceptionVLMClient that exposes pose-related methods."""

    def __init__(self, pose_results: list[PoseResult] | None = None) -> None:
        self._pose_results: list[PoseResult] = pose_results or []
        self._pose_enabled: bool = False
        # Simulate the _engine._config structure expected by the router
        self._engine = MagicMock()
        self._engine._config.pose_enabled = False
        self._engine._pose_estimator = None

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        return VLMResult(has_violation=False, violations=[], raw_text="{}")

    def get_pose_results(self) -> list[PoseResult]:
        return list(self._pose_results)

    def set_pose_enabled(self, enabled: bool) -> None:
        self._pose_enabled = enabled
        self._engine._config.pose_enabled = enabled
        if enabled and not self._engine._pose_estimator:
            self._engine._pose_estimator = MagicMock()
        elif not enabled:
            self._engine._pose_estimator = None

    def reset_session(self) -> None:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Base test class
# ──────────────────────────────────────────────────────────────────────────────


class _PPEAPIBase(unittest.TestCase):
    """Shared setUp / tearDown for PPE status API tests."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(self.root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "ppe-test"
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

    def _create_session(self, name: str = "PPEテスト") -> dict:
        r = self.client.post(
            "/vigil/sessions",
            json={
                "name": name,
                "rules": ["ヘルメット未着用を検出", "ベスト未着用を検出"],
                "sample_fps": 1.0,
                "severity_threshold": "warning",
            },
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()


# ──────────────────────────────────────────────────────────────────────────────
# GET /vigil/sessions/{id}/ppe-status — non-perception backend
# ──────────────────────────────────────────────────────────────────────────────


class TestPPEStatusNonPerception(_PPEAPIBase):
    """ppe-status with a plain VLM backend (no pose support)."""

    def setUp(self) -> None:
        super().setUp()
        # Replace with a plain VLM mock that has no pose methods
        self.app.state.vigil_pipeline._vlm = _MockVLMNoViolation()

    def test_ppe_status_returns_200(self) -> None:
        sid = self._create_session()["session_id"]
        r = self.client.get(f"/vigil/sessions/{sid}/ppe-status")
        self.assertEqual(r.status_code, 200)

    def test_ppe_status_pose_enabled_false(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()
        self.assertFalse(data["pose_enabled"])

    def test_ppe_status_persons_empty(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()
        self.assertEqual(data["persons"], [])

    def test_ppe_status_summary_zeros(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()
        s = data["summary"]
        self.assertEqual(s["total_persons"], 0)
        self.assertEqual(s["helmets_ok"], 0)
        self.assertEqual(s["vests_ok"], 0)

    def test_ppe_status_session_id_in_response(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()
        self.assertEqual(data["session_id"], sid)

    def test_ppe_status_not_found_returns_404(self) -> None:
        r = self.client.get("/vigil/sessions/9999/ppe-status")
        self.assertEqual(r.status_code, 404)


# ──────────────────────────────────────────────────────────────────────────────
# GET /vigil/sessions/{id}/ppe-status — perception backend with mock pose
# ──────────────────────────────────────────────────────────────────────────────


class TestPPEStatusWithPoseResults(_PPEAPIBase):
    """ppe-status with a mock PerceptionVLMClient returning pose data."""

    def setUp(self) -> None:
        super().setUp()
        # Two persons: one fully compliant, one not
        self._pr1 = _make_pose_result(
            has_helmet=True, helmet_conf=0.85,
            has_vest=True, vest_conf=0.72,
            pose_conf=0.91,
        )
        self._pr2 = _make_pose_result(
            has_helmet=False, helmet_conf=0.12,
            has_vest=True, vest_conf=0.88,
            pose_conf=0.78,
        )
        mock_vlm = _MockPerceptionVLM(pose_results=[self._pr1, self._pr2])
        mock_vlm._engine._config.pose_enabled = True
        mock_vlm._engine._pose_estimator = MagicMock()
        self.app.state.vigil_pipeline._vlm = mock_vlm

    def test_ppe_status_returns_200(self) -> None:
        sid = self._create_session()["session_id"]
        r = self.client.get(f"/vigil/sessions/{sid}/ppe-status")
        self.assertEqual(r.status_code, 200)

    def test_ppe_status_pose_enabled_true(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()
        self.assertTrue(data["pose_enabled"])

    def test_ppe_status_persons_count(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()
        self.assertEqual(len(data["persons"]), 2)

    def test_ppe_status_person0_helmet_ok(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()
        p0 = data["persons"][0]
        self.assertTrue(p0["has_helmet"])
        self.assertAlmostEqual(p0["helmet_confidence"], 0.85, places=3)

    def test_ppe_status_person1_helmet_ng(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()
        p1 = data["persons"][1]
        self.assertFalse(p1["has_helmet"])

    def test_ppe_status_person0_vest_ok(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()
        p0 = data["persons"][0]
        self.assertTrue(p0["has_vest"])
        self.assertAlmostEqual(p0["vest_confidence"], 0.72, places=3)

    def test_ppe_status_person1_vest_ok(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()
        p1 = data["persons"][1]
        self.assertTrue(p1["has_vest"])

    def test_ppe_status_pose_confidence_fields(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()
        self.assertAlmostEqual(data["persons"][0]["pose_confidence"], 0.91, places=3)
        self.assertAlmostEqual(data["persons"][1]["pose_confidence"], 0.78, places=3)

    def test_ppe_status_person_index_field(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()
        self.assertEqual(data["persons"][0]["index"], 0)
        self.assertEqual(data["persons"][1]["index"], 1)

    def test_ppe_status_not_found_returns_404(self) -> None:
        r = self.client.get("/vigil/sessions/9999/ppe-status")
        self.assertEqual(r.status_code, 404)


# ──────────────────────────────────────────────────────────────────────────────
# Summary calculation
# ──────────────────────────────────────────────────────────────────────────────


class TestPPEStatusSummary(_PPEAPIBase):
    """Tests for the summary sub-object."""

    def _inject_results(self, pose_results: list[PoseResult]) -> None:
        mock_vlm = _MockPerceptionVLM(pose_results=pose_results)
        mock_vlm._engine._config.pose_enabled = True
        mock_vlm._engine._pose_estimator = MagicMock()
        self.app.state.vigil_pipeline._vlm = mock_vlm

    def test_summary_all_compliant(self) -> None:
        prs = [
            _make_pose_result(has_helmet=True, has_vest=True),
            _make_pose_result(has_helmet=True, has_vest=True),
        ]
        self._inject_results(prs)
        sid = self._create_session()["session_id"]
        s = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()["summary"]
        self.assertEqual(s["total_persons"], 2)
        self.assertEqual(s["helmets_ok"], 2)
        self.assertEqual(s["vests_ok"], 2)

    def test_summary_none_compliant(self) -> None:
        prs = [
            _make_pose_result(has_helmet=False, has_vest=False),
            _make_pose_result(has_helmet=False, has_vest=False),
        ]
        self._inject_results(prs)
        sid = self._create_session()["session_id"]
        s = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()["summary"]
        self.assertEqual(s["total_persons"], 2)
        self.assertEqual(s["helmets_ok"], 0)
        self.assertEqual(s["vests_ok"], 0)

    def test_summary_partial_compliance(self) -> None:
        prs = [
            _make_pose_result(has_helmet=True, has_vest=False),
            _make_pose_result(has_helmet=False, has_vest=True),
            _make_pose_result(has_helmet=True, has_vest=True),
        ]
        self._inject_results(prs)
        sid = self._create_session()["session_id"]
        s = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()["summary"]
        self.assertEqual(s["total_persons"], 3)
        self.assertEqual(s["helmets_ok"], 2)
        self.assertEqual(s["vests_ok"], 2)

    def test_summary_single_person(self) -> None:
        prs = [_make_pose_result(has_helmet=True, has_vest=False)]
        self._inject_results(prs)
        sid = self._create_session()["session_id"]
        s = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()["summary"]
        self.assertEqual(s["total_persons"], 1)
        self.assertEqual(s["helmets_ok"], 1)
        self.assertEqual(s["vests_ok"], 0)

    def test_summary_empty_results(self) -> None:
        mock_vlm = _MockPerceptionVLM(pose_results=[])
        mock_vlm._engine._config.pose_enabled = True
        self.app.state.vigil_pipeline._vlm = mock_vlm
        sid = self._create_session()["session_id"]
        s = self.client.get(f"/vigil/sessions/{sid}/ppe-status").json()["summary"]
        self.assertEqual(s["total_persons"], 0)
        self.assertEqual(s["helmets_ok"], 0)
        self.assertEqual(s["vests_ok"], 0)


# ──────────────────────────────────────────────────────────────────────────────
# POST /vigil/sessions/{id}/pose-enable
# ──────────────────────────────────────────────────────────────────────────────


class TestPoseEnableEndpoint(_PPEAPIBase):
    """Tests for POST /vigil/sessions/{id}/pose-enable."""

    def setUp(self) -> None:
        super().setUp()
        self._mock_vlm = _MockPerceptionVLM()
        self.app.state.vigil_pipeline._vlm = self._mock_vlm

    def test_pose_enable_true_returns_200(self) -> None:
        sid = self._create_session()["session_id"]
        r = self.client.post(f"/vigil/sessions/{sid}/pose-enable?enabled=true")
        self.assertEqual(r.status_code, 200, r.text)

    def test_pose_enable_response_ok_true(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.post(f"/vigil/sessions/{sid}/pose-enable?enabled=true").json()
        self.assertTrue(data["ok"])

    def test_pose_enable_response_pose_enabled_field(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.post(f"/vigil/sessions/{sid}/pose-enable?enabled=true").json()
        self.assertTrue(data["pose_enabled"])

    def test_pose_enable_false_returns_200(self) -> None:
        sid = self._create_session()["session_id"]
        r = self.client.post(f"/vigil/sessions/{sid}/pose-enable?enabled=false")
        self.assertEqual(r.status_code, 200, r.text)

    def test_pose_enable_false_response(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.post(f"/vigil/sessions/{sid}/pose-enable?enabled=false").json()
        self.assertFalse(data["pose_enabled"])
        self.assertTrue(data["ok"])

    def test_pose_enable_calls_set_pose_enabled(self) -> None:
        sid = self._create_session()["session_id"]
        self.client.post(f"/vigil/sessions/{sid}/pose-enable?enabled=true")
        self.assertTrue(self._mock_vlm._pose_enabled)

    def test_pose_disable_calls_set_pose_enabled_false(self) -> None:
        sid = self._create_session()["session_id"]
        self.client.post(f"/vigil/sessions/{sid}/pose-enable?enabled=true")
        self.client.post(f"/vigil/sessions/{sid}/pose-enable?enabled=false")
        self.assertFalse(self._mock_vlm._pose_enabled)

    def test_pose_enable_not_found_returns_404(self) -> None:
        r = self.client.post("/vigil/sessions/9999/pose-enable?enabled=true")
        self.assertEqual(r.status_code, 404)

    def test_pose_enable_session_id_in_response(self) -> None:
        sid = self._create_session()["session_id"]
        data = self.client.post(f"/vigil/sessions/{sid}/pose-enable?enabled=true").json()
        self.assertEqual(data["session_id"], sid)

    def test_pose_enable_noop_on_plain_vlm(self) -> None:
        """For a plain VLM client without set_pose_enabled, should still return ok=True."""
        self.app.state.vigil_pipeline._vlm = _MockVLMNoViolation()
        sid = self._create_session()["session_id"]
        r = self.client.post(f"/vigil/sessions/{sid}/pose-enable?enabled=true")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])


# ──────────────────────────────────────────────────────────────────────────────
# PerceptionVLMClient unit tests (without HTTP)
# ──────────────────────────────────────────────────────────────────────────────


class TestPerceptionVLMClientPoseMethods(unittest.TestCase):
    """Unit tests for PerceptionVLMClient.get_pose_results / set_pose_enabled."""

    def _make_client(self) -> _MockPerceptionVLM:
        """Use the mock which mirrors the real PerceptionVLMClient interface."""
        return _MockPerceptionVLM()

    def test_get_pose_results_initially_empty(self) -> None:
        client = self._make_client()
        self.assertEqual(client.get_pose_results(), [])

    def test_get_pose_results_returns_copy(self) -> None:
        pr = _make_pose_result()
        client = _MockPerceptionVLM(pose_results=[pr])
        results = client.get_pose_results()
        # Mutating the returned list does not affect internal state
        results.clear()
        self.assertEqual(len(client.get_pose_results()), 1)

    def test_set_pose_enabled_true(self) -> None:
        client = self._make_client()
        client.set_pose_enabled(True)
        self.assertTrue(client._pose_enabled)

    def test_set_pose_enabled_false(self) -> None:
        client = self._make_client()
        client.set_pose_enabled(True)
        client.set_pose_enabled(False)
        self.assertFalse(client._pose_enabled)

    def test_set_pose_enabled_updates_engine_config(self) -> None:
        client = self._make_client()
        client.set_pose_enabled(True)
        self.assertTrue(client._engine._config.pose_enabled)

    def test_set_pose_enabled_false_clears_engine_estimator(self) -> None:
        client = self._make_client()
        client.set_pose_enabled(True)
        client.set_pose_enabled(False)
        self.assertIsNone(client._engine._pose_estimator)

    def test_pose_results_with_multiple_persons(self) -> None:
        prs = [_make_pose_result(has_helmet=True), _make_pose_result(has_helmet=False)]
        client = _MockPerceptionVLM(pose_results=prs)
        results = client.get_pose_results()
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].ppe.has_helmet)
        self.assertFalse(results[1].ppe.has_helmet)

    def test_pose_result_ppe_fields(self) -> None:
        pr = _make_pose_result(
            has_helmet=True, helmet_conf=0.9,
            has_vest=False, vest_conf=0.1,
        )
        client = _MockPerceptionVLM(pose_results=[pr])
        result = client.get_pose_results()[0]
        self.assertTrue(result.ppe.has_helmet)
        self.assertAlmostEqual(result.ppe.helmet_confidence, 0.9, places=3)
        self.assertFalse(result.ppe.has_vest)
        self.assertAlmostEqual(result.ppe.vest_confidence, 0.1, places=3)


if __name__ == "__main__":
    unittest.main()
