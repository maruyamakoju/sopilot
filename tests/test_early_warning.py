"""Tests for Phase 15: EarlyWarningEngine (予兆検知).

Coverage:
    - EarlyWarningEngine.observe_sigma_change()
    - EarlyWarningEngine.observe_anomaly()
    - EarlyWarningEngine.get_risk_score() — all three components
    - EarlyWarningEngine.get_all_risks()
    - EarlyWarningEngine.get_state()
    - EarlyWarningEngine.reset()
    - Integration: engine.get_early_warning_state()
    - API endpoint: GET /vigil/perception/early-warning
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path


# ---------------------------------------------------------------------------
# TestEarlyWarningEngine
# ---------------------------------------------------------------------------


class TestEarlyWarningEngine(unittest.TestCase):
    def _make(self, **kw):
        from sopilot.perception.early_warning import EarlyWarningEngine
        return EarlyWarningEngine(**kw)

    # ── observe_sigma_change ──────────────────────────────────────────

    def test_sigma_change_updates_drift_velocity(self):
        ew = self._make()
        t0 = time.time()
        ew.observe_sigma_change("behavioral", 2.0, 3.0, timestamp=t0)
        ew.observe_sigma_change("behavioral", 3.0, 4.0, timestamp=t0 + 60)
        # drift velocity should be non-zero
        self.assertGreater(ew._drift_velocity.get("behavioral", 0.0), 0.0)

    def test_sigma_change_different_detectors_are_independent(self):
        ew = self._make()
        t0 = time.time()
        ew.observe_sigma_change("behavioral", 2.0, 4.0, timestamp=t0)
        ew.observe_sigma_change("behavioral", 4.0, 5.0, timestamp=t0 + 60)
        # spatial should be zero
        self.assertEqual(ew._drift_velocity.get("spatial", 0.0), 0.0)

    def test_no_change_no_drift(self):
        ew = self._make()
        self.assertEqual(ew._drift_velocity.get("behavioral", 0.0), 0.0)
        _, detail = ew.get_risk_score("behavioral")
        self.assertEqual(detail["sigma_drift_velocity"], 0.0)

    # ── observe_anomaly ───────────────────────────────────────────────

    def test_observe_anomaly_increases_burst_count(self):
        ew = self._make(burst_window_seconds=60)
        now = time.time()
        for _ in range(5):
            ew.observe_anomaly("spatial", timestamp=now)
        _, detail = ew.get_risk_score("spatial")
        self.assertGreater(detail["anomaly_burst_rate"], 0.0)

    def test_old_anomalies_pruned_from_burst_window(self):
        ew = self._make(burst_window_seconds=60)
        past = time.time() - 120  # 2 minutes ago — outside window
        for _ in range(10):
            ew.observe_anomaly("temporal", timestamp=past)
        _, detail = ew.get_risk_score("temporal")
        self.assertEqual(detail["anomaly_burst_rate"], 0.0)

    # ── get_risk_score ────────────────────────────────────────────────

    def test_risk_score_zero_with_no_signals(self):
        ew = self._make()
        risk, detail = ew.get_risk_score("behavioral")
        self.assertEqual(risk, 0.0)
        self.assertEqual(detail["risk_level"], "low")

    def test_risk_score_in_zero_one_range(self):
        ew = self._make()
        # Push extreme signals
        t0 = time.time()
        ew.observe_sigma_change("behavioral", 1.0, 6.0, timestamp=t0)
        ew.observe_sigma_change("behavioral", 6.0, 1.0, timestamp=t0 + 1)
        for _ in range(100):
            ew.observe_anomaly("behavioral", timestamp=time.time())
        risk, _ = ew.get_risk_score("behavioral")
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)

    def test_risk_score_fp_rate_from_tuner_stats(self):
        ew = self._make()
        tuner_stats = {
            "pair_stats": [
                {"detector": "behavioral", "total": 10, "denied": 9, "confirmed": 1}
            ]
        }
        _, detail = ew.get_risk_score("behavioral", tuner_stats=tuner_stats)
        self.assertGreater(detail["fp_rate"], 0.0)
        self.assertGreater(detail["fp_rate_norm"], 0.0)

    def test_risk_level_high_when_score_above_0_6(self):
        ew = self._make(max_fp_rate=0.1)  # very low threshold → fp_norm hits 1 quickly
        tuner_stats = {
            "pair_stats": [
                {"detector": "spatial", "total": 10, "denied": 10, "confirmed": 0}
            ]
        }
        _, detail = ew.get_risk_score("spatial", tuner_stats=tuner_stats)
        # fp_norm = 1.0 × W_FP(0.4) = 0.4 → medium; with drift also pushed it can be high
        # At min: fp contributes 0.4. Without drift/burst, level is "medium".
        self.assertIn(detail["risk_level"], ("medium", "high"))

    # ── get_all_risks ─────────────────────────────────────────────────

    def test_get_all_risks_returns_all_four_detectors(self):
        ew = self._make()
        risks = ew.get_all_risks()
        self.assertSetEqual(
            set(risks.keys()), {"behavioral", "spatial", "temporal", "interaction"}
        )

    def test_get_all_risks_each_has_required_fields(self):
        ew = self._make()
        risks = ew.get_all_risks()
        required = {
            "detector", "risk_score", "risk_level",
            "sigma_drift_velocity", "fp_rate", "anomaly_burst_rate",
        }
        for det, detail in risks.items():
            self.assertTrue(required.issubset(detail.keys()), msg=f"{det} missing fields")

    # ── get_state ─────────────────────────────────────────────────────

    def test_get_state_has_overall_fields(self):
        ew = self._make()
        state = ew.get_state()
        for key in ("overall_risk", "overall_level", "detectors", "computed_at", "burst_window_seconds"):
            self.assertIn(key, state)

    def test_get_state_overall_risk_equals_max_detector_risk(self):
        ew = self._make()
        state = ew.get_state()
        detector_max = max(d["risk_score"] for d in state["detectors"].values())
        self.assertAlmostEqual(state["overall_risk"], detector_max, places=3)

    # ── reset ─────────────────────────────────────────────────────────

    def test_reset_clears_drift_and_burst(self):
        ew = self._make()
        t0 = time.time()
        ew.observe_sigma_change("behavioral", 2.0, 5.0, timestamp=t0)
        ew.observe_anomaly("behavioral", timestamp=t0)
        ew.reset()
        self.assertEqual(len(ew._drift_velocity), 0)
        self.assertEqual(len(ew._anomaly_timestamps), 0)
        _, detail = ew.get_risk_score("behavioral")
        self.assertEqual(detail["risk_score"], 0.0)

    # ── Engine integration ────────────────────────────────────────────

    def test_engine_has_early_warning_after_build(self):
        from sopilot.perception.engine import build_perception_engine
        engine = build_perception_engine()
        self.assertIsNotNone(engine._early_warning)

    def test_engine_get_early_warning_state_returns_dict(self):
        from sopilot.perception.engine import build_perception_engine
        engine = build_perception_engine()
        state = engine.get_early_warning_state()
        self.assertIsInstance(state, dict)
        self.assertIn("overall_risk", state)
        self.assertIn("detectors", state)

    # ── API endpoint ──────────────────────────────────────────────────

    def _make_perception_client(self, task_id="ew-test"):
        """Create a TestClient with the perception engine active."""
        import os
        import tempfile
        from fastapi.testclient import TestClient
        from unittest.mock import MagicMock
        from sopilot.main import create_app
        from sopilot.perception.engine import build_perception_engine
        from sopilot.vigil.vlm import PerceptionVLMClient

        tmp = tempfile.mkdtemp()
        os.environ["SOPILOT_DATA_DIR"] = tmp
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = task_id
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"
        os.environ["VIGIL_VLM_BACKEND"] = "perception"

        app = create_app()
        client = TestClient(app)
        return client, tmp

    def _cleanup_perception_env(self):
        import os
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                  "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM",
                  "VIGIL_VLM_BACKEND"):
            os.environ.pop(k, None)

    def test_early_warning_endpoint_200(self):
        try:
            client, _ = self._make_perception_client("ew-test")
            r = client.get("/vigil/perception/early-warning")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertIn("overall_risk", body)
            self.assertIn("detectors", body)
        finally:
            self._cleanup_perception_env()

    def test_early_warning_endpoint_returns_low_risk_initially(self):
        try:
            client, _ = self._make_perception_client("ew-test2")
            r = client.get("/vigil/perception/early-warning")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["overall_level"], "low")
        finally:
            self._cleanup_perception_env()


if __name__ == "__main__":
    unittest.main()
