"""Tests for Phase 16: EarlyWarningResponder (自律対応).

Coverage:
    - EarlyWarningResponder.evaluate(): trigger, cooldown, no-trigger below threshold
    - EarlyWarningResponder.get_history()
    - EarlyWarningResponder.get_state()
    - EarlyWarningResponder.reset()
    - ResponseAction.to_dict()
    - Integration: engine._early_warning_responder
    - API: POST /vigil/perception/early-warning/respond
    - API: GET  /vigil/perception/early-warning/response-history
"""

from __future__ import annotations

import time
import unittest


def _ew_state(level="high", score=0.8, detector="behavioral"):
    """Build a minimal EarlyWarningEngine state dict."""
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


class TestEarlyWarningResponder(unittest.TestCase):
    def _make(self, **kw):
        from sopilot.perception.early_warning_responder import EarlyWarningResponder
        return EarlyWarningResponder(**kw)

    # ── evaluate ─────────────────────────────────────────────────────

    def test_evaluate_triggers_for_high_risk(self):
        r = self._make(risk_threshold=0.6)
        state = _ew_state(level="high", score=0.8)
        actions = r.evaluate(state)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].detector, "behavioral")
        self.assertEqual(actions[0].risk_level, "high")

    def test_evaluate_does_not_trigger_below_threshold(self):
        r = self._make(risk_threshold=0.6)
        state = _ew_state(level="low", score=0.2)
        actions = r.evaluate(state)
        self.assertEqual(len(actions), 0)

    def test_evaluate_respects_cooldown(self):
        r = self._make(risk_threshold=0.6, cooldown_seconds=3600)
        state = _ew_state(level="high", score=0.9)
        first = r.evaluate(state)
        self.assertEqual(len(first), 1)
        second = r.evaluate(state)  # within cooldown → skip
        self.assertEqual(len(second), 0)

    def test_evaluate_triggers_again_after_cooldown(self):
        r = self._make(risk_threshold=0.6, cooldown_seconds=0.01)
        state = _ew_state(level="high", score=0.9)
        r.evaluate(state)
        time.sleep(0.02)
        second = r.evaluate(state)
        self.assertEqual(len(second), 1)

    def test_evaluate_multiple_detectors(self):
        r = self._make(risk_threshold=0.6)
        state = {
            "overall_risk": 0.9, "overall_level": "high",
            "detectors": {
                "behavioral": {"risk_score": 0.8, "risk_level": "high",
                               "sigma_drift_velocity": 0.5, "sigma_drift_norm": 0.9,
                               "fp_rate": 0.6, "fp_rate_norm": 0.8,
                               "anomaly_burst_rate": 2.0, "anomaly_burst_norm": 0.4},
                "spatial": {"risk_score": 0.7, "risk_level": "high",
                            "sigma_drift_velocity": 0.3, "sigma_drift_norm": 0.5,
                            "fp_rate": 0.5, "fp_rate_norm": 0.7,
                            "anomaly_burst_rate": 1.0, "anomaly_burst_norm": 0.2},
            },
            "computed_at": time.time(), "burst_window_seconds": 300,
        }
        actions = r.evaluate(state)
        dets = {a.detector for a in actions}
        self.assertIn("behavioral", dets)
        self.assertIn("spatial", dets)

    # ── ResponseAction ────────────────────────────────────────────────

    def test_action_to_dict_has_required_fields(self):
        r = self._make(risk_threshold=0.6)
        state = _ew_state(level="high", score=0.85)
        actions = r.evaluate(state)
        self.assertEqual(len(actions), 1)
        d = actions[0].to_dict()
        for key in ("detector", "risk_score", "risk_level", "explanation_ja",
                    "recommendations", "triggered_at"):
            self.assertIn(key, d)

    def test_action_explanation_is_nonempty_japanese(self):
        r = self._make(risk_threshold=0.6)
        state = _ew_state(level="high", score=0.85)
        actions = r.evaluate(state)
        self.assertGreater(len(actions[0].explanation_ja), 10)

    def test_action_recommendations_nonempty(self):
        r = self._make(risk_threshold=0.6)
        state = _ew_state(level="high", score=0.85)
        actions = r.evaluate(state)
        self.assertGreater(len(actions[0].recommendations), 0)

    # ── get_history ───────────────────────────────────────────────────

    def test_get_history_returns_most_recent_first(self):
        r = self._make(risk_threshold=0.6, cooldown_seconds=0.0)
        for _ in range(3):
            r.evaluate(_ew_state(level="high", score=0.9))
            time.sleep(0.001)
        hist = r.get_history(limit=3)
        self.assertEqual(len(hist), 3)
        # most recent first
        self.assertGreaterEqual(hist[0]["triggered_at"], hist[-1]["triggered_at"])

    def test_get_history_respects_limit(self):
        r = self._make(risk_threshold=0.6, cooldown_seconds=0.0)
        for _ in range(5):
            r.evaluate(_ew_state(level="high", score=0.9))
            time.sleep(0.001)
        self.assertEqual(len(r.get_history(limit=2)), 2)

    # ── get_state ─────────────────────────────────────────────────────

    def test_get_state_has_required_fields(self):
        r = self._make()
        state = r.get_state()
        for k in ("total_responses", "risk_threshold", "cooldown_seconds",
                  "cooldowns_remaining", "recent_responses"):
            self.assertIn(k, state)

    def test_get_state_cooldowns_remaining_after_trigger(self):
        r = self._make(risk_threshold=0.6, cooldown_seconds=600)
        r.evaluate(_ew_state(level="high", score=0.9))
        state = r.get_state()
        self.assertIn("behavioral", state["cooldowns_remaining"])

    # ── reset ─────────────────────────────────────────────────────────

    def test_reset_clears_history_and_cooldowns(self):
        r = self._make(risk_threshold=0.6)
        r.evaluate(_ew_state(level="high", score=0.9))
        r.reset()
        self.assertEqual(r.get_history(), [])
        self.assertEqual(r.get_state()["total_responses"], 0)
        self.assertEqual(r.get_state()["cooldowns_remaining"], {})

    # ── Engine integration ────────────────────────────────────────────

    def test_engine_has_responder_after_build(self):
        from sopilot.perception.engine import build_perception_engine
        engine = build_perception_engine()
        self.assertIsNotNone(engine._early_warning_responder)

    def test_engine_get_responder_state_returns_dict(self):
        from sopilot.perception.engine import build_perception_engine
        engine = build_perception_engine()
        state = engine.get_early_warning_responder_state()
        self.assertIsInstance(state, dict)
        self.assertIn("total_responses", state)

    # ── API endpoints ─────────────────────────────────────────────────

    def _make_client(self, task_id="ewr-test"):
        import os, tempfile
        from fastapi.testclient import TestClient
        from sopilot.main import create_app

        tmp = tempfile.mkdtemp()
        os.environ.update({
            "SOPILOT_DATA_DIR": tmp,
            "SOPILOT_EMBEDDER_BACKEND": "color-motion",
            "SOPILOT_PRIMARY_TASK_ID": task_id,
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

    def test_respond_endpoint_200(self):
        try:
            client, _ = self._make_client()
            r = client.post("/vigil/perception/early-warning/respond")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertIn("triggered", body)
            self.assertIn("total_triggered", body)
        finally:
            self._cleanup()

    def test_respond_returns_empty_when_no_high_risk(self):
        try:
            client, _ = self._make_client("ewr-test2")
            r = client.post("/vigil/perception/early-warning/respond")
            # No anomalies processed yet → all risk = 0.0 → no triggers
            self.assertEqual(r.json()["total_triggered"], 0)
        finally:
            self._cleanup()

    def test_response_history_endpoint_200(self):
        try:
            client, _ = self._make_client("ewr-test3")
            r = client.get("/vigil/perception/early-warning/response-history")
            self.assertEqual(r.status_code, 200)
            self.assertIn("total_responses", r.json())
        finally:
            self._cleanup()


if __name__ == "__main__":
    unittest.main()
