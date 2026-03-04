"""Tests for Phase 12A: SigmaTuner (自動σ調整).

Coverage:
    - SigmaTuner: get_sigma, compute_and_apply, dead zone, lr_up/lr_down,
      clamp, history, reset, get_state
    - Engine integration: _sigma_tuner field, Stage 6j wiring,
      _events_to_violations filter, get_sigma_state()
    - API endpoints: GET /sigma-state, POST /sigma-reset
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

from sopilot.perception.types import PerceptionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair_stats(
    detector: str = "behavioral",
    metric: str = "speed_zscore",
    total: int = 10,
    confirmed: int = 2,
    denied: int = 8,
) -> dict:
    return {
        "detector": detector,
        "metric": metric,
        "total": total,
        "confirmed": confirmed,
        "denied": denied,
        "confirmation_rate": confirmed / total if total else 0.0,
        "fp_rate": denied / total if total else 0.0,
    }


def _make_tuner_stats(pair_stats: list[dict]) -> dict:
    total = sum(p["total"] for p in pair_stats)
    confirmed = sum(p["confirmed"] for p in pair_stats)
    denied = sum(p["denied"] for p in pair_stats)
    return {
        "total_feedback": total,
        "confirmed": confirmed,
        "denied": denied,
        "overall_confirm_rate": confirmed / total if total else 0.0,
        "pairs_tracked": len(pair_stats),
        "pairs_suppressed": 0,
        "pairs_trusted": 0,
        "last_tuning": time.time(),
        "pair_stats": pair_stats,
        "suppressed_pairs": [],
        "trusted_pairs": [],
        "min_samples_for_tuning": 10,
    }


# ===========================================================================
# SigmaTuner unit tests
# ===========================================================================


class TestSigmaTunerDefaults(unittest.TestCase):
    def _make(self, base_sigma=2.0, **kw):
        from sopilot.perception.sigma_tuner import SigmaTuner
        return SigmaTuner(base_sigma=base_sigma, **kw)

    def test_get_sigma_returns_base_when_unadjusted(self):
        st = self._make(base_sigma=2.0)
        self.assertAlmostEqual(st.get_sigma("behavioral"), 2.0)

    def test_get_sigma_unknown_detector_returns_base(self):
        st = self._make(base_sigma=2.0)
        self.assertAlmostEqual(st.get_sigma("unknown_detector"), 2.0)

    def test_get_state_has_required_keys(self):
        st = self._make()
        state = st.get_state()
        for key in ("base_sigma", "target_fp_rate", "total_adjustments",
                    "detector_sigmas", "recent_adjustments"):
            self.assertIn(key, state)

    def test_get_state_detector_sigmas_has_all_detectors(self):
        st = self._make()
        state = st.get_state()
        for det in ("behavioral", "spatial", "temporal", "interaction"):
            self.assertIn(det, state["detector_sigmas"])

    def test_get_state_adjusted_false_initially(self):
        st = self._make()
        state = st.get_state()
        for info in state["detector_sigmas"].values():
            self.assertFalse(info["adjusted"])

    def test_total_adjustments_zero_initially(self):
        st = self._make()
        self.assertEqual(st.get_state()["total_adjustments"], 0)

    def test_recent_adjustments_empty_initially(self):
        st = self._make()
        self.assertEqual(st.get_state()["recent_adjustments"], [])


class TestSigmaTunerComputeAndApply(unittest.TestCase):
    def _make(self, base_sigma=2.0, min_samples=5, **kw):
        from sopilot.perception.sigma_tuner import SigmaTuner
        return SigmaTuner(base_sigma=base_sigma, min_samples=min_samples, **kw)

    def test_high_fp_rate_raises_sigma(self):
        st = self._make()
        # FP rate = 80% → above TARGET(30%) + DEAD_ZONE(15%) = 45%
        ps = _make_pair_stats(total=10, confirmed=2, denied=8)
        stats = _make_tuner_stats([ps])
        changes = st.compute_and_apply(stats)
        self.assertGreater(len(changes), 0)
        self.assertGreater(st.get_sigma("behavioral"), 2.0)

    def test_low_fp_rate_lowers_sigma(self):
        st = self._make()
        # FP rate = 5% → below TARGET(30%) - DEAD_ZONE(15%) = 15%
        ps = _make_pair_stats(total=10, confirmed=9, denied=1)
        stats = _make_tuner_stats([ps])
        changes = st.compute_and_apply(stats)
        self.assertGreater(len(changes), 0)
        self.assertLess(st.get_sigma("behavioral"), 2.0)

    def test_dead_zone_no_change(self):
        st = self._make()
        # FP rate = 30% → exactly at TARGET → no change
        ps = _make_pair_stats(total=10, confirmed=7, denied=3)
        stats = _make_tuner_stats([ps])
        changes = st.compute_and_apply(stats)
        self.assertEqual(len(changes), 0)

    def test_insufficient_samples_no_change(self):
        st = self._make(min_samples=10)
        # Only 5 samples → below MIN_SAMPLES
        ps = _make_pair_stats(total=5, confirmed=0, denied=5)
        stats = _make_tuner_stats([ps])
        changes = st.compute_and_apply(stats)
        self.assertEqual(len(changes), 0)

    def test_sigma_clamped_to_max(self):
        st = self._make(base_sigma=5.8, sigma_max=6.0)
        ps = _make_pair_stats(total=10, confirmed=0, denied=10)  # 100% FP
        stats = _make_tuner_stats([ps])
        st.compute_and_apply(stats)
        self.assertLessEqual(st.get_sigma("behavioral"), 6.0)

    def test_sigma_clamped_to_min(self):
        st = self._make(base_sigma=1.2, sigma_min=1.0)
        ps = _make_pair_stats(total=10, confirmed=10, denied=0)  # 0% FP
        stats = _make_tuner_stats([ps])
        st.compute_and_apply(stats)
        self.assertGreaterEqual(st.get_sigma("behavioral"), 1.0)

    def test_change_dict_has_required_keys(self):
        st = self._make()
        ps = _make_pair_stats(total=10, confirmed=0, denied=10)
        stats = _make_tuner_stats([ps])
        changes = st.compute_and_apply(stats)
        for key in ("detector", "old_sigma", "new_sigma", "fp_rate", "samples",
                    "timestamp", "direction"):
            self.assertIn(key, changes[0])

    def test_direction_up_when_fp_high(self):
        st = self._make()
        ps = _make_pair_stats(total=10, confirmed=0, denied=10)
        stats = _make_tuner_stats([ps])
        changes = st.compute_and_apply(stats)
        self.assertEqual(changes[0]["direction"], "up")

    def test_direction_down_when_fp_low(self):
        st = self._make()
        ps = _make_pair_stats(total=10, confirmed=10, denied=0)
        stats = _make_tuner_stats([ps])
        changes = st.compute_and_apply(stats)
        self.assertEqual(changes[0]["direction"], "down")

    def test_multiple_detectors_adjusted_independently(self):
        st = self._make()
        ps_b = _make_pair_stats("behavioral", total=10, confirmed=0, denied=10)
        ps_s = _make_pair_stats("spatial", total=10, confirmed=10, denied=0)
        stats = _make_tuner_stats([ps_b, ps_s])
        st.compute_and_apply(stats)
        self.assertGreater(st.get_sigma("behavioral"), 2.0)
        self.assertLess(st.get_sigma("spatial"), 2.0)

    def test_total_adjustments_increments(self):
        st = self._make()
        ps = _make_pair_stats(total=10, confirmed=0, denied=10)
        stats = _make_tuner_stats([ps])
        st.compute_and_apply(stats)
        self.assertGreater(st.get_state()["total_adjustments"], 0)

    def test_adjusted_flag_set_after_change(self):
        st = self._make()
        ps = _make_pair_stats(total=10, confirmed=0, denied=10)
        stats = _make_tuner_stats([ps])
        st.compute_and_apply(stats)
        self.assertTrue(st.get_state()["detector_sigmas"]["behavioral"]["adjusted"])

    def test_empty_pair_stats_returns_no_changes(self):
        st = self._make()
        stats = _make_tuner_stats([])
        changes = st.compute_and_apply(stats)
        self.assertEqual(changes, [])

    def test_delta_positive_when_raised(self):
        st = self._make()
        ps = _make_pair_stats(total=10, confirmed=0, denied=10)
        stats = _make_tuner_stats([ps])
        st.compute_and_apply(stats)
        delta = st.get_state()["detector_sigmas"]["behavioral"]["delta"]
        self.assertGreater(delta, 0)

    def test_delta_negative_when_lowered(self):
        st = self._make()
        ps = _make_pair_stats(total=10, confirmed=10, denied=0)
        stats = _make_tuner_stats([ps])
        st.compute_and_apply(stats)
        delta = st.get_state()["detector_sigmas"]["behavioral"]["delta"]
        self.assertLess(delta, 0)


class TestSigmaTunerReset(unittest.TestCase):
    def _make(self, **kw):
        from sopilot.perception.sigma_tuner import SigmaTuner
        return SigmaTuner(**kw)

    def test_reset_clears_sigmas(self):
        st = self._make()
        ps = _make_pair_stats(total=10, confirmed=0, denied=10)
        st.compute_and_apply(_make_tuner_stats([ps]))
        st.reset()
        self.assertAlmostEqual(st.get_sigma("behavioral"), st._base_sigma)

    def test_reset_clears_total_adjustments(self):
        st = self._make()
        ps = _make_pair_stats(total=10, confirmed=0, denied=10)
        st.compute_and_apply(_make_tuner_stats([ps]))
        st.reset()
        self.assertEqual(st.get_state()["total_adjustments"], 0)

    def test_reset_clears_history(self):
        st = self._make()
        ps = _make_pair_stats(total=10, confirmed=0, denied=10)
        st.compute_and_apply(_make_tuner_stats([ps]))
        st.reset()
        self.assertEqual(st.get_state()["recent_adjustments"], [])


# ===========================================================================
# Engine integration tests
# ===========================================================================


class TestSigmaTunerEngineIntegration(unittest.TestCase):
    def test_engine_has_sigma_tuner_field(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        self.assertTrue(hasattr(engine, "_sigma_tuner"))

    def test_sigma_tuner_none_by_default(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        self.assertIsNone(engine._sigma_tuner)

    def test_build_engine_injects_sigma_tuner(self):
        from sopilot.perception.engine import build_perception_engine
        from sopilot.perception.sigma_tuner import SigmaTuner
        engine = build_perception_engine()
        self.assertIsInstance(engine._sigma_tuner, SigmaTuner)

    def test_sigma_apply_interval_from_config(self):
        from sopilot.perception.engine import PerceptionEngine
        config = PerceptionConfig(sigma_apply_interval=5)
        engine = PerceptionEngine(config=config)
        self.assertEqual(engine._sigma_apply_interval, 5)

    def test_get_sigma_state_returns_none_without_tuner(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        self.assertIsNone(engine.get_sigma_state())

    def test_get_sigma_state_returns_dict_with_tuner(self):
        from sopilot.perception.engine import build_perception_engine
        engine = build_perception_engine()
        state = engine.get_sigma_state()
        self.assertIsNotNone(state)
        self.assertIn("detector_sigmas", state)

    def test_events_to_violations_filters_below_sigma(self):
        """ANOMALY event with z_score below detector sigma should be filtered."""
        from sopilot.perception.engine import PerceptionEngine
        from sopilot.perception.sigma_tuner import SigmaTuner
        from sopilot.perception.types import (
            EntityEvent, EntityEventType, WorldState, SceneGraph,
        )

        engine = PerceptionEngine(config=PerceptionConfig())
        sigma_tuner = SigmaTuner(base_sigma=2.0)
        # Set behavioral sigma to 5.0
        sigma_tuner._detector_sigmas["behavioral"] = 5.0
        engine._sigma_tuner = sigma_tuner

        # Create ANOMALY event with z_score=3.0 < behavioral sigma=5.0
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1, timestamp=1.0, frame_number=1,
            details={"detector": "behavioral", "metric": "speed_zscore",
                     "z_score": 3.0, "severity": "warning",
                     "description_ja": "テスト"},
        )
        sg = SceneGraph(
            timestamp=1.0, frame_number=1, entities=[], relations=[],
            frame_shape=(480, 640),
        )
        ws = WorldState(
            scene_graph=sg,
            timestamp=1.0, frame_number=1,
            active_tracks={}, zone_occupancy={},
            events=[event],
        )
        violations = engine._events_to_violations(ws)
        # Should be filtered (z=3.0 < sigma=5.0)
        self.assertEqual(len(violations), 0)

    def test_events_to_violations_passes_above_sigma(self):
        """ANOMALY event with z_score above detector sigma should pass."""
        from sopilot.perception.engine import PerceptionEngine
        from sopilot.perception.sigma_tuner import SigmaTuner
        from sopilot.perception.types import (
            EntityEvent, EntityEventType, WorldState, SceneGraph,
        )

        engine = PerceptionEngine(config=PerceptionConfig())
        sigma_tuner = SigmaTuner(base_sigma=2.0)
        engine._sigma_tuner = sigma_tuner
        # behavioral stays at default 2.0

        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1, timestamp=1.0, frame_number=1,
            details={"detector": "behavioral", "metric": "speed_zscore",
                     "z_score": 4.0, "severity": "warning",
                     "description_ja": "テスト"},
        )
        sg = SceneGraph(
            timestamp=1.0, frame_number=1, entities=[], relations=[],
            frame_shape=(480, 640),
        )
        ws = WorldState(
            scene_graph=sg,
            timestamp=1.0, frame_number=1,
            active_tracks={}, zone_occupancy={},
            events=[event],
        )
        violations = engine._events_to_violations(ws)
        # z=4.0 >= sigma=2.0 → should pass (converted to violation)
        self.assertEqual(len(violations), 1)


# ===========================================================================
# API endpoint tests
# ===========================================================================


class TestSigmaStateEndpoints(unittest.TestCase):
    def setUp(self):
        import os
        import tempfile
        from pathlib import Path as _P
        from fastapi.testclient import TestClient
        from sopilot.main import create_app
        from sopilot.perception.sigma_tuner import SigmaTuner

        self._tmp = tempfile.TemporaryDirectory()
        root = _P(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "sigma-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"

        self.app = create_app()
        self.client = TestClient(self.app)

        engine = MagicMock()
        self.sigma_tuner = SigmaTuner(base_sigma=2.0)
        engine._sigma_tuner = self.sigma_tuner
        engine.get_sigma_state.return_value = self.sigma_tuner.get_state()

        vlm = MagicMock()
        vlm._engine = engine
        self.app.state.vigil_pipeline._vlm = vlm
        self.engine = engine

    def tearDown(self):
        import os
        self._tmp.cleanup()
        for k in ("SOPILOT_DATA_DIR", "SOPILOT_EMBEDDER_BACKEND",
                   "SOPILOT_PRIMARY_TASK_ID", "SOPILOT_RATE_LIMIT_RPM"):
            os.environ.pop(k, None)

    def test_get_sigma_state_200(self):
        r = self.client.get("/vigil/perception/sigma-state")
        self.assertEqual(r.status_code, 200)

    def test_get_sigma_state_has_detector_sigmas(self):
        r = self.client.get("/vigil/perception/sigma-state")
        self.assertIn("detector_sigmas", r.json())

    def test_get_sigma_state_has_all_detectors(self):
        r = self.client.get("/vigil/perception/sigma-state")
        ds = r.json()["detector_sigmas"]
        for det in ("behavioral", "spatial", "temporal", "interaction"):
            self.assertIn(det, ds)

    def test_get_sigma_state_has_total_adjustments(self):
        r = self.client.get("/vigil/perception/sigma-state")
        self.assertIn("total_adjustments", r.json())

    def test_post_sigma_reset_200(self):
        r = self.client.post("/vigil/perception/sigma-reset")
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
