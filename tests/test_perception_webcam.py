"""Tests for Webcam -> Perception Engine direct pipeline.

Coverage:
  - POST /vigil/sessions/{id}/perception-reset endpoint (8 tests)
  - analyze_webcam_frame with mock perception backend (6 tests)
  - PerceptionVLMClient.reset_session unit tests (4 tests)
  - Integration: reset -> webcam frame flow (4 tests)
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from sopilot.database import Database
from sopilot.main import create_app
from sopilot.vigil.pipeline import VigilPipeline
from sopilot.vigil.schemas import VLMResult
from sopilot.vigil.vlm import VLMClient


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Create a minimal JPEG image in memory."""
    frame = np.full((height, width, 3), (120, 80, 200), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


class _MockVLMNoViolation(VLMClient):
    """VLM stub -- always returns no violation."""

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        return VLMResult(has_violation=False, violations=[], raw_text="{}")


class _MockVLMWithViolation(VLMClient):
    """VLM stub -- always returns one warning-level violation."""

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


class _MockVLMWithResetTracking(VLMClient):
    """VLM stub that tracks reset_session calls."""

    def __init__(self) -> None:
        self.reset_count = 0
        self.analyze_count = 0

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        self.analyze_count += 1
        return VLMResult(has_violation=False, violations=[], raw_text="{}")

    def reset_session(self) -> None:
        self.reset_count += 1


class _MockVLMStateful(VLMClient):
    """VLM stub that simulates stateful tracking (like PerceptionVLMClient).

    Returns different results depending on internal frame count.
    reset_session clears the frame count.
    """

    def __init__(self) -> None:
        self._frame_count = 0
        self._lock = threading.Lock()

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        with self._lock:
            self._frame_count += 1
            # After 3 frames, start detecting violations (simulating accumulated state)
            if self._frame_count >= 3:
                return VLMResult(
                    has_violation=True,
                    violations=[{
                        "rule_index": 0,
                        "rule": rules[0] if rules else "test",
                        "description_ja": f"フレーム{self._frame_count}で違反検出",
                        "severity": "warning",
                        "confidence": 0.85,
                    }],
                    raw_text="stateful",
                )
        return VLMResult(has_violation=False, violations=[], raw_text="no-violation")

    def reset_session(self) -> None:
        with self._lock:
            self._frame_count = 0


# ──────────────────────────────────────────────────────────────────────────────
# Test: POST /vigil/sessions/{id}/perception-reset
# ──────────────────────────────────────────────────────────────────────────────


class TestPerceptionResetEndpoint(unittest.TestCase):
    """HTTP tests for the perception-reset endpoint."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(self.root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "perception-webcam-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"
        self.app = create_app()
        self.client = TestClient(self.app)
        # Replace VLM with reset-tracking mock
        self.mock_vlm = _MockVLMWithResetTracking()
        self.app.state.vigil_pipeline._vlm = self.mock_vlm

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                  "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
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

    def test_reset_returns_ok(self) -> None:
        """POST /vigil/sessions/{id}/perception-reset returns ok: true."""
        session = self._create_session()
        sid = session["session_id"]
        r = self.client.post(f"/vigil/sessions/{sid}/perception-reset")
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["session_id"], sid)

    def test_reset_calls_vlm_reset_session(self) -> None:
        """Reset endpoint should call pipeline._vlm.reset_session()."""
        session = self._create_session()
        sid = session["session_id"]
        self.assertEqual(self.mock_vlm.reset_count, 0)
        self.client.post(f"/vigil/sessions/{sid}/perception-reset")
        self.assertEqual(self.mock_vlm.reset_count, 1)

    def test_reset_multiple_calls(self) -> None:
        """Multiple reset calls should each invoke reset_session."""
        session = self._create_session()
        sid = session["session_id"]
        for i in range(3):
            r = self.client.post(f"/vigil/sessions/{sid}/perception-reset")
            self.assertEqual(r.status_code, 200)
        self.assertEqual(self.mock_vlm.reset_count, 3)

    def test_reset_404_for_nonexistent_session(self) -> None:
        """Reset on a non-existent session returns 404."""
        r = self.client.post("/vigil/sessions/99999/perception-reset")
        self.assertEqual(r.status_code, 404)

    def test_reset_does_not_affect_session_status(self) -> None:
        """Reset should not change the session status or counters."""
        session = self._create_session()
        sid = session["session_id"]
        self.client.post(f"/vigil/sessions/{sid}/perception-reset")
        r = self.client.get(f"/vigil/sessions/{sid}")
        data = r.json()
        self.assertEqual(data["status"], "idle")
        self.assertEqual(data["total_frames_analyzed"], 0)
        self.assertEqual(data["violation_count"], 0)

    def test_reset_with_noop_backend(self) -> None:
        """Reset on a stateless backend (base VLMClient) is a no-op that succeeds."""
        self.app.state.vigil_pipeline._vlm = _MockVLMNoViolation()
        session = self._create_session()
        sid = session["session_id"]
        r = self.client.post(f"/vigil/sessions/{sid}/perception-reset")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

    def test_reset_with_violation_vlm(self) -> None:
        """Reset works even when the VLM normally returns violations."""
        self.app.state.vigil_pipeline._vlm = _MockVLMWithViolation()
        session = self._create_session()
        sid = session["session_id"]
        r = self.client.post(f"/vigil/sessions/{sid}/perception-reset")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

    def test_reset_after_delete_returns_404(self) -> None:
        """Reset on a deleted session returns 404."""
        session = self._create_session()
        sid = session["session_id"]
        self.client.delete(f"/vigil/sessions/{sid}")
        r = self.client.post(f"/vigil/sessions/{sid}/perception-reset")
        self.assertEqual(r.status_code, 404)


# ──────────────────────────────────────────────────────────────────────────────
# Test: webcam-frame with mock perception (stateful backend)
# ──────────────────────────────────────────────────────────────────────────────


class TestWebcamFrameWithPerception(unittest.TestCase):
    """Tests for analyze_webcam_frame using a stateful mock backend."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(self.root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "perception-webcam-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"
        self.app = create_app()
        self.client = TestClient(self.app)
        self.mock_vlm = _MockVLMStateful()
        self.app.state.vigil_pipeline._vlm = self.mock_vlm

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                  "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
            os.environ.pop(k, None)

    def _create_session(self) -> int:
        r = self.client.post(
            "/vigil/sessions",
            json={
                "name": "webcam-perception",
                "rules": ["ヘルメット未着用を検出"],
                "sample_fps": 1.0,
                "severity_threshold": "warning",
            },
        )
        return r.json()["session_id"]

    def _send_frame(self, sid: int) -> dict:
        jpeg = _make_jpeg_bytes()
        r = self.client.post(
            f"/vigil/sessions/{sid}/webcam-frame",
            files={"file": ("frame.jpg", jpeg, "image/jpeg")},
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_first_frames_no_violation(self) -> None:
        """First two frames should have no violation (stateful mock)."""
        sid = self._create_session()
        for _ in range(2):
            result = self._send_frame(sid)
            self.assertFalse(result["has_violation"])

    def test_third_frame_has_violation(self) -> None:
        """After accumulating state, violations appear on frame 3."""
        sid = self._create_session()
        self._send_frame(sid)
        self._send_frame(sid)
        result = self._send_frame(sid)
        self.assertTrue(result["has_violation"])
        self.assertGreater(len(result["violations"]), 0)

    def test_tracking_state_persists_across_frames(self) -> None:
        """Frame count in stateful VLM persists across webcam calls (same engine)."""
        sid = self._create_session()
        results = [self._send_frame(sid) for _ in range(5)]
        # First 2 clean, frames 3-5 have violations
        self.assertFalse(results[0]["has_violation"])
        self.assertFalse(results[1]["has_violation"])
        self.assertTrue(results[2]["has_violation"])
        self.assertTrue(results[3]["has_violation"])
        self.assertTrue(results[4]["has_violation"])

    def test_reset_clears_tracking_state(self) -> None:
        """After reset, frame count is zeroed so violations disappear."""
        sid = self._create_session()
        # Accumulate state: 3 frames -> violation
        for _ in range(3):
            self._send_frame(sid)
        # Reset perception state
        r = self.client.post(f"/vigil/sessions/{sid}/perception-reset")
        self.assertEqual(r.status_code, 200)
        # After reset, first 2 frames should be clean again
        result1 = self._send_frame(sid)
        self.assertFalse(result1["has_violation"])
        result2 = self._send_frame(sid)
        self.assertFalse(result2["has_violation"])

    def test_violation_stored_as_event(self) -> None:
        """Webcam violation is stored as a session event."""
        sid = self._create_session()
        for _ in range(3):
            self._send_frame(sid)
        # Third frame should have created an event
        r = self.client.get(f"/vigil/sessions/{sid}/events")
        self.assertEqual(r.status_code, 200)
        events = r.json()
        self.assertGreaterEqual(len(events), 1)

    def test_webcam_frame_no_store_flag(self) -> None:
        """With store=false, violations are returned but not persisted."""
        sid = self._create_session()
        # Send 3 frames with store=false to trigger stateful violation
        for _ in range(2):
            jpeg = _make_jpeg_bytes()
            self.client.post(
                f"/vigil/sessions/{sid}/webcam-frame?store=false",
                files={"file": ("frame.jpg", jpeg, "image/jpeg")},
            )
        jpeg = _make_jpeg_bytes()
        r = self.client.post(
            f"/vigil/sessions/{sid}/webcam-frame?store=false",
            files={"file": ("frame.jpg", jpeg, "image/jpeg")},
        )
        result = r.json()
        self.assertTrue(result["has_violation"])
        self.assertIsNone(result["event_id"])
        # No events should be stored
        events_r = self.client.get(f"/vigil/sessions/{sid}/events")
        self.assertEqual(len(events_r.json()), 0)


# ──────────────────────────────────────────────────────────────────────────────
# Test: PerceptionVLMClient.reset_session unit tests (mocked engine)
# ──────────────────────────────────────────────────────────────────────────────


class TestPerceptionVLMClientReset(unittest.TestCase):
    """Unit tests for PerceptionVLMClient.reset_session (mocked engine)."""

    def _make_client(self):
        """Create a PerceptionVLMClient with a mocked engine."""
        from sopilot.vigil.vlm import PerceptionVLMClient

        mock_engine = MagicMock()
        mock_engine.reset = MagicMock()
        mock_engine.close = MagicMock()

        with patch("sopilot.perception.engine.build_perception_engine", return_value=mock_engine):
            client = PerceptionVLMClient(config=None, vlm_fallback=None)

        return client, mock_engine

    def test_reset_calls_engine_reset(self) -> None:
        """reset_session should call engine.reset()."""
        client, engine = self._make_client()
        client.reset_session()
        engine.reset.assert_called_once()

    def test_reset_zeroes_frame_number(self) -> None:
        """reset_session should zero the internal frame counter."""
        client, _ = self._make_client()
        client._frame_number = 42
        client.reset_session()
        self.assertEqual(client._frame_number, 0)

    def test_reset_is_thread_safe(self) -> None:
        """reset_session should acquire the lock."""
        client, engine = self._make_client()
        # Simulate concurrent access: if lock is used, no crash
        errors = []

        def do_reset():
            try:
                for _ in range(20):
                    client.reset_session()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_reset) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)
        # Each of 4 threads calls 20 times = 80 total resets
        self.assertEqual(engine.reset.call_count, 80)

    def test_base_vlm_reset_is_noop(self) -> None:
        """Base VLMClient.reset_session is a no-op (no error)."""
        base = VLMClient()
        base.reset_session()  # Should not raise


# ──────────────────────────────────────────────────────────────────────────────
# Test: Integration - reset then webcam flow
# ──────────────────────────────────────────────────────────────────────────────


class TestResetThenWebcamIntegration(unittest.TestCase):
    """Integration tests: reset -> webcam-frame flow across multiple sessions."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(self.root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "perception-integration"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"
        self.app = create_app()
        self.client = TestClient(self.app)
        self.mock_vlm = _MockVLMStateful()
        self.app.state.vigil_pipeline._vlm = self.mock_vlm

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                  "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
            os.environ.pop(k, None)

    def _create_session(self, name: str = "integration") -> int:
        r = self.client.post(
            "/vigil/sessions",
            json={
                "name": name,
                "rules": ["安全ベスト未着用を検出"],
                "sample_fps": 1.0,
                "severity_threshold": "warning",
            },
        )
        return r.json()["session_id"]

    def _send_frame(self, sid: int) -> dict:
        jpeg = _make_jpeg_bytes()
        r = self.client.post(
            f"/vigil/sessions/{sid}/webcam-frame",
            files={"file": ("frame.jpg", jpeg, "image/jpeg")},
        )
        return r.json()

    def test_switch_sessions_without_reset_bleeds_state(self) -> None:
        """Without reset, state from session 1 bleeds into session 2."""
        sid1 = self._create_session("session-1")
        # Accumulate 2 frames on session 1
        self._send_frame(sid1)
        self._send_frame(sid1)
        # Switch to session 2 without reset
        sid2 = self._create_session("session-2")
        # Frame 3 should trigger violation because state was NOT reset
        result = self._send_frame(sid2)
        self.assertTrue(result["has_violation"], "State bled from session 1 to session 2")

    def test_switch_sessions_with_reset_isolates_state(self) -> None:
        """With reset before session 2, tracking starts fresh."""
        sid1 = self._create_session("session-1")
        # Accumulate 2 frames on session 1
        self._send_frame(sid1)
        self._send_frame(sid1)
        # Switch to session 2 WITH reset
        sid2 = self._create_session("session-2")
        self.client.post(f"/vigil/sessions/{sid2}/perception-reset")
        # First frame on session 2 should be clean
        result = self._send_frame(sid2)
        self.assertFalse(result["has_violation"], "Reset should isolate session state")

    def test_reset_between_webcam_batches(self) -> None:
        """Reset between batches of webcam frames clears accumulated state."""
        sid = self._create_session()
        # First batch: 2 frames (no violation)
        self._send_frame(sid)
        self._send_frame(sid)
        # Reset
        self.client.post(f"/vigil/sessions/{sid}/perception-reset")
        # Second batch: 2 more frames (should still be clean since reset)
        result1 = self._send_frame(sid)
        result2 = self._send_frame(sid)
        self.assertFalse(result1["has_violation"])
        self.assertFalse(result2["has_violation"])

    def test_full_lifecycle_reset_accumulate_reset(self) -> None:
        """Full cycle: reset -> accumulate -> detect -> reset -> clean again."""
        sid = self._create_session()
        # Reset at start
        self.client.post(f"/vigil/sessions/{sid}/perception-reset")
        # Accumulate 3 frames -> violation on 3rd
        r1 = self._send_frame(sid)
        r2 = self._send_frame(sid)
        r3 = self._send_frame(sid)
        self.assertFalse(r1["has_violation"])
        self.assertFalse(r2["has_violation"])
        self.assertTrue(r3["has_violation"])
        # Reset again
        self.client.post(f"/vigil/sessions/{sid}/perception-reset")
        # After reset, first frame is clean
        r4 = self._send_frame(sid)
        self.assertFalse(r4["has_violation"])
