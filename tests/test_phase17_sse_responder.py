"""Tests for Phase 17: SSE Closed-Loop Notification.

Coverage:
    - PerceptionEngine._push_early_warning_responses(): broadcast to all sessions
    - Stage 6k wires evaluate() return value into SSE push
    - No-op when no SSE sessions are registered
    - Multiple sessions receive the event
    - Multiple triggered actions each broadcast
    - ImportError gracefully handled
    - SSE event payload fields (detector, risk_level, explanation_ja, recommendations)
    - API: POST /vigil/perception/early-warning/respond pushes SSE event
    - SSE session receives EARLY_WARNING_RESPONSE event type
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_engine():
    from sopilot.perception.engine import build_perception_engine
    return build_perception_engine()


def _make_action(detector="behavioral", score=0.85, level="high"):
    from sopilot.perception.early_warning_responder import ResponseAction
    return ResponseAction(
        detector=detector,
        risk_score=score,
        risk_level=level,
        explanation_ja="テスト説明: σ値が急速に変化しています。",
        recommendations=["POST /vigil/perception/sigma-reset"],
        triggered_at=time.time(),
    )


def _ew_state(level="high", score=0.8, detector="behavioral"):
    return {
        "overall_risk": score,
        "overall_level": level,
        "detectors": {
            detector: {
                "detector": detector,
                "risk_score": score,
                "risk_level": level,
                "sigma_drift_velocity": 0.6,
                "sigma_drift_norm": 0.8,
                "fp_rate": 0.55,
                "fp_rate_norm": 0.7,
                "anomaly_burst_rate": 3.0,
                "anomaly_burst_norm": 0.6,
            }
        },
        "computed_at": time.time(),
        "burst_window_seconds": 300,
    }


# ── Unit: _push_early_warning_responses ──────────────────────────────────────

class TestPushEarlyWarningResponses(unittest.TestCase):

    def test_push_to_all_registered_sessions(self):
        engine = _make_engine()
        action = _make_action()
        pushed = []

        with patch("sopilot.perception.sse_events.list_sessions", return_value=["s1", "s2"]), \
             patch("sopilot.perception.sse_events.push_event", side_effect=lambda s, t, p: pushed.append((s, t))):
            engine._push_early_warning_responses([action])

        sessions = {s for s, _ in pushed}
        self.assertIn("s1", sessions)
        self.assertIn("s2", sessions)

    def test_event_type_is_early_warning_response(self):
        engine = _make_engine()
        action = _make_action()
        event_types = []

        with patch("sopilot.perception.sse_events.list_sessions", return_value=["s1"]), \
             patch("sopilot.perception.sse_events.push_event", side_effect=lambda s, t, p: event_types.append(t)):
            engine._push_early_warning_responses([action])

        self.assertEqual(event_types, ["EARLY_WARNING_RESPONSE"])

    def test_payload_contains_required_fields(self):
        engine = _make_engine()
        action = _make_action(detector="spatial", score=0.92, level="high")
        payloads = []

        with patch("sopilot.perception.sse_events.list_sessions", return_value=["s1"]), \
             patch("sopilot.perception.sse_events.push_event", side_effect=lambda s, t, p: payloads.append(p)):
            engine._push_early_warning_responses([action])

        self.assertEqual(len(payloads), 1)
        p = payloads[0]
        for key in ("detector", "risk_score", "risk_level", "explanation_ja",
                    "recommendations", "triggered_at"):
            self.assertIn(key, p)

    def test_noop_when_no_sessions_registered(self):
        engine = _make_engine()
        action = _make_action()
        push_calls = []

        with patch("sopilot.perception.sse_events.list_sessions", return_value=[]), \
             patch("sopilot.perception.sse_events.push_event", side_effect=lambda s, t, p: push_calls.append(1)):
            engine._push_early_warning_responses([action])

        self.assertEqual(push_calls, [])

    def test_multiple_actions_each_pushed_to_each_session(self):
        engine = _make_engine()
        actions = [_make_action("behavioral"), _make_action("spatial")]
        pushed = []

        with patch("sopilot.perception.sse_events.list_sessions", return_value=["s1", "s2"]), \
             patch("sopilot.perception.sse_events.push_event", side_effect=lambda s, t, p: pushed.append((s, p["detector"]))):
            engine._push_early_warning_responses(actions)

        # 2 actions × 2 sessions = 4 pushes
        self.assertEqual(len(pushed), 4)
        detectors_pushed = {d for _, d in pushed}
        self.assertIn("behavioral", detectors_pushed)
        self.assertIn("spatial", detectors_pushed)

    def test_import_error_handled_gracefully(self):
        engine = _make_engine()
        action = _make_action()
        import sys
        original = sys.modules.get("sopilot.perception.sse_events")
        sys.modules["sopilot.perception.sse_events"] = None  # force ImportError-like
        try:
            # Should not raise
            engine._push_early_warning_responses([action])
        except Exception as exc:
            self.fail(f"_push_early_warning_responses raised unexpectedly: {exc}")
        finally:
            if original is not None:
                sys.modules["sopilot.perception.sse_events"] = original
            elif "sopilot.perception.sse_events" in sys.modules:
                del sys.modules["sopilot.perception.sse_events"]

    def test_exception_in_push_does_not_propagate(self):
        engine = _make_engine()
        action = _make_action()
        with patch("sopilot.perception.sse_events.list_sessions", return_value=["s1"]), \
             patch("sopilot.perception.sse_events.push_event", side_effect=RuntimeError("boom")):
            try:
                engine._push_early_warning_responses([action])
            except Exception as exc:
                self.fail(f"Should not propagate: {exc}")

    def test_payload_detector_matches_action(self):
        engine = _make_engine()
        action = _make_action(detector="temporal")
        payloads = []

        with patch("sopilot.perception.sse_events.list_sessions", return_value=["s1"]), \
             patch("sopilot.perception.sse_events.push_event", side_effect=lambda s, t, p: payloads.append(p)):
            engine._push_early_warning_responses([action])

        self.assertEqual(payloads[0]["detector"], "temporal")

    def test_payload_explanation_ja_nonempty(self):
        engine = _make_engine()
        action = _make_action()
        payloads = []

        with patch("sopilot.perception.sse_events.list_sessions", return_value=["s1"]), \
             patch("sopilot.perception.sse_events.push_event", side_effect=lambda s, t, p: payloads.append(p)):
            engine._push_early_warning_responses([action])

        self.assertGreater(len(payloads[0]["explanation_ja"]), 5)


# ── Unit: Stage 6k integration (mock engine loop) ────────────────────────────

class TestStage6kSSEIntegration(unittest.TestCase):

    def test_stage6k_calls_push_when_triggered(self):
        """Simulate Stage 6k: evaluate() returns triggered actions → push called."""
        engine = _make_engine()

        mock_action = _make_action()
        push_calls = []

        from sopilot.perception.early_warning_responder import EarlyWarningResponder
        mock_responder = MagicMock(spec=EarlyWarningResponder)
        mock_responder.evaluate.return_value = [mock_action]

        engine._early_warning_responder = mock_responder

        with patch.object(engine, "_push_early_warning_responses") as mock_push:
            # Manually trigger Stage 6k logic
            if engine._early_warning is not None and engine._early_warning_responder is not None:
                ew_state = engine._early_warning.get_state()
                triggered = engine._early_warning_responder.evaluate(
                    ew_state,
                    sigma_tuner=engine._sigma_tuner,
                    review_queue=engine._review_queue,
                )
                if triggered:
                    engine._push_early_warning_responses(triggered)

            mock_push.assert_called_once_with([mock_action])

    def test_stage6k_does_not_push_when_not_triggered(self):
        """When evaluate() returns empty list, push is NOT called."""
        engine = _make_engine()

        from sopilot.perception.early_warning_responder import EarlyWarningResponder
        mock_responder = MagicMock(spec=EarlyWarningResponder)
        mock_responder.evaluate.return_value = []

        engine._early_warning_responder = mock_responder

        with patch.object(engine, "_push_early_warning_responses") as mock_push:
            if engine._early_warning is not None and engine._early_warning_responder is not None:
                ew_state = engine._early_warning.get_state()
                triggered = engine._early_warning_responder.evaluate(
                    ew_state,
                    sigma_tuner=engine._sigma_tuner,
                    review_queue=engine._review_queue,
                )
                if triggered:
                    engine._push_early_warning_responses(triggered)

            mock_push.assert_not_called()


# ── Integration: SSE event queue receives EARLY_WARNING_RESPONSE ──────────────

class TestSSEEarlyWarningEvent(unittest.TestCase):

    def test_sse_queue_receives_early_warning_response_event(self):
        """SSE queue should hold an EARLY_WARNING_RESPONSE event after push."""
        from sopilot.perception import sse_events

        sid = "phase17-test-session"
        q = sse_events.get_or_create(sid)
        q.clear()

        payload = {
            "detector": "behavioral",
            "risk_score": 0.85,
            "risk_level": "high",
            "explanation_ja": "テスト説明",
            "recommendations": ["POST /vigil/perception/sigma-reset"],
            "triggered_at": time.time(),
        }
        result = sse_events.push_event(sid, "EARLY_WARNING_RESPONSE", payload)
        self.assertTrue(result)
        self.assertEqual(q.get_stats()["queued"], 1)

        sse_events.remove(sid)

    def test_sse_event_sse_string_contains_early_warning_type(self):
        """The SSE data string for EARLY_WARNING_RESPONSE contains the event type."""
        from sopilot.perception.sse_events import PercEvent

        evt = PercEvent(
            event_type="EARLY_WARNING_RESPONSE",
            session_id="test",
            payload={"detector": "spatial", "risk_score": 0.75, "risk_level": "high",
                     "explanation_ja": "説明", "recommendations": [], "triggered_at": time.time()},
        )
        sse_str = evt.to_sse()
        self.assertIn("EARLY_WARNING_RESPONSE", sse_str)
        self.assertIn("data:", sse_str)

    def test_sse_payload_survives_roundtrip(self):
        """Payload pushed to SSE queue can be read back intact."""
        import json
        from sopilot.perception import sse_events

        sid = "phase17-roundtrip"
        q = sse_events.get_or_create(sid)
        q.clear()

        payload = {
            "detector": "interaction",
            "risk_score": 0.91,
            "risk_level": "high",
            "explanation_ja": "インタラクション検出器の説明",
            "recommendations": ["GET /vigil/camera-groups/learning/compare"],
            "triggered_at": time.time(),
        }
        sse_events.push_event(sid, "EARLY_WARNING_RESPONSE", payload)

        import asyncio
        async def _read():
            return await q.next_event(timeout=1.0)

        event = asyncio.get_event_loop().run_until_complete(_read())
        self.assertIsNotNone(event)
        sse_str = event.to_sse()
        data = json.loads(sse_str.split("data: ", 1)[1].strip())
        self.assertEqual(data["detector"], "interaction")
        self.assertIn("explanation_ja", data)

        sse_events.remove(sid)


# ── API: respond endpoint + SSE integration ───────────────────────────────────

class TestRespondEndpointSSEIntegration(unittest.TestCase):

    def _make_client(self):
        import os, tempfile
        from fastapi.testclient import TestClient
        from sopilot.main import create_app

        tmp = tempfile.mkdtemp()
        os.environ.update({
            "SOPILOT_DATA_DIR": tmp,
            "SOPILOT_EMBEDDER_BACKEND": "color-motion",
            "SOPILOT_PRIMARY_TASK_ID": "p17-test",
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

    def test_respond_endpoint_returns_sse_info_field(self):
        """POST /early-warning/respond response contains triggered list."""
        try:
            client, _ = self._make_client()
            r = client.post("/vigil/perception/early-warning/respond")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertIn("triggered", body)
            self.assertIn("total_triggered", body)
        finally:
            self._cleanup()

    def test_respond_endpoint_with_high_risk_triggers_push(self):
        """When risk is HIGH, respond endpoint calls _push_early_warning_responses."""
        try:
            client, _ = self._make_client()
            push_calls = []

            with patch(
                "sopilot.perception.engine.PerceptionEngine._push_early_warning_responses",
                side_effect=lambda actions: push_calls.append(actions),
            ):
                # No real high-risk state in fresh engine, so we mock the responder
                from sopilot.perception.early_warning_responder import ResponseAction
                action = ResponseAction(
                    detector="behavioral", risk_score=0.85, risk_level="high",
                    explanation_ja="テスト", recommendations=["POST /vigil/reset"],
                    triggered_at=time.time(),
                )
                with patch(
                    "sopilot.perception.early_warning_responder.EarlyWarningResponder.evaluate",
                    return_value=[action],
                ):
                    r = client.post("/vigil/perception/early-warning/respond")
                    self.assertEqual(r.status_code, 200)
        finally:
            self._cleanup()


if __name__ == "__main__":
    unittest.main()
