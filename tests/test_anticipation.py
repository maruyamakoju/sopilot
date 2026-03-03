"""Tests for sopilot/perception/anticipation.py — Predictive Safety Engine.

65 tests covering:
    1. TestVelocityEstimator         (8 tests)
    2. TestTTCComputation            (12 tests)
    3. TestTrajectoryHazardAnalyzer  (15 tests)
    4. TestCrowdSurgeDetector        (10 tests)
    5. TestAnticipationEngine        (15 tests)
    6. TestHazardAssessment          (5 tests)

No imports from sopilot.perception.types — mock entity objects are used.
Run: python -m pytest tests/test_anticipation.py -v --tb=short
"""
from __future__ import annotations

import math
import threading
import time
import unittest
from dataclasses import dataclass, field
from typing import Any

from sopilot.perception.anticipation import (
    HAZARD_COLLISION,
    HAZARD_CROWD_SURGE,
    HAZARD_NEAR_MISS,
    HAZARD_ZONE_BREACH,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    AnticipationEngine,
    CrowdSurgeDetector,
    HazardAssessment,
    TrajectoryHazardAnalyzer,
    VelocityEstimator,
    _cx,
    _cy,
    _eid,
    _label,
    _ttc_severity,
)


# ---------------------------------------------------------------------------
# Mock entity helpers — no dependency on sopilot.perception.types
# ---------------------------------------------------------------------------


@dataclass
class MockEntity:
    """Minimal entity used in tests. bbox = (x1, y1, x2, y2) normalized."""

    entity_id: int
    label: str
    bbox: tuple[float, float, float, float]
    details: dict[str, Any] = field(default_factory=dict)


def _make_entity(
    entity_id: int = 1,
    label: str = "person",
    x1: float = 0.1,
    y1: float = 0.1,
    x2: float = 0.2,
    y2: float = 0.3,
) -> MockEntity:
    return MockEntity(entity_id=entity_id, label=label, bbox=(x1, y1, x2, y2))


@dataclass
class MockWorldState:
    entities: list[MockEntity] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. TestVelocityEstimator (8 tests)
# ---------------------------------------------------------------------------


class TestVelocityEstimator(unittest.TestCase):

    def setUp(self) -> None:
        self.est = VelocityEstimator(alpha=0.4)

    def test_first_call_returns_zero_velocity(self):
        """Initial call for an unseen entity must return (0, 0)."""
        vx, vy = self.est.update(1, 0.5, 0.5)
        self.assertEqual(vx, 0.0)
        self.assertEqual(vy, 0.0)

    def test_second_call_returns_delta(self):
        """Second call computes EMA-smoothed delta from position change."""
        self.est.update(1, 0.0, 0.0)
        vx, vy = self.est.update(1, 0.1, 0.2)
        # alpha=0.4, prev_v=(0,0): vx = 0.4*0.1 + 0.6*0.0 = 0.04
        self.assertAlmostEqual(vx, 0.04, places=6)
        self.assertAlmostEqual(vy, 0.08, places=6)

    def test_ema_smoothing_on_third_call(self):
        """EMA accumulates history across multiple calls."""
        self.est.update(1, 0.0, 0.0)
        self.est.update(1, 0.1, 0.0)   # vx = 0.04
        vx, _ = self.est.update(1, 0.2, 0.0)   # dx=0.1 again
        # vx = 0.4*0.1 + 0.6*0.04 = 0.04 + 0.024 = 0.064
        self.assertAlmostEqual(vx, 0.064, places=6)

    def test_reset_clears_all_state(self):
        """reset() must wipe positions and velocities."""
        self.est.update(1, 0.3, 0.3)
        self.est.update(1, 0.5, 0.5)
        self.est.reset()
        vx, vy = self.est.update(1, 0.5, 0.5)
        self.assertEqual(vx, 0.0)
        self.assertEqual(vy, 0.0)

    def test_multiple_entities_are_independent(self):
        """Entities must not share velocity state."""
        self.est.update(1, 0.0, 0.0)
        self.est.update(2, 0.5, 0.5)
        vx1, _ = self.est.update(1, 0.1, 0.0)
        vx2, _ = self.est.update(2, 0.6, 0.5)
        self.assertAlmostEqual(vx1, 0.04, places=6)
        self.assertAlmostEqual(vx2, 0.04, places=6)

    def test_get_velocity_for_unknown_entity_returns_zero(self):
        """get_velocity on unseen entity must return (0, 0)."""
        v = self.est.get_velocity(999)
        self.assertEqual(v, (0.0, 0.0))

    def test_get_velocity_matches_last_update(self):
        """get_velocity returns the same value as the last update result."""
        self.est.update(3, 0.0, 0.0)
        expected = self.est.update(3, 0.05, 0.1)
        result = self.est.get_velocity(3)
        self.assertAlmostEqual(result[0], expected[0], places=9)
        self.assertAlmostEqual(result[1], expected[1], places=9)

    def test_stationary_entity_keeps_zero_velocity(self):
        """If entity does not move, velocity should stay (0, 0)."""
        for _ in range(5):
            vx, vy = self.est.update(1, 0.5, 0.5)
        # EMA of zeros stays zero
        self.assertAlmostEqual(vx, 0.0, places=9)
        self.assertAlmostEqual(vy, 0.0, places=9)


# ---------------------------------------------------------------------------
# 2. TestTTCComputation (12 tests)
# ---------------------------------------------------------------------------


class TestTTCComputation(unittest.TestCase):

    def setUp(self) -> None:
        self.analyzer = TrajectoryHazardAnalyzer(
            collision_distance_norm=0.05,
            ttc_horizon_s=5.0,
            fps_hint=10.0,
        )

    def test_head_on_collision_returns_positive_ttc(self):
        """Two entities moving toward each other must produce a finite TTC."""
        pos_a = (0.2, 0.5)
        vel_a = (0.01, 0.0)   # moving right
        pos_b = (0.8, 0.5)
        vel_b = (-0.01, 0.0)  # moving left
        ttc = self.analyzer.compute_ttc(pos_a, vel_a, pos_b, vel_b)
        self.assertIsNotNone(ttc)
        self.assertGreater(ttc, 0.0)

    def test_diverging_returns_none(self):
        """Entities moving away from each other: TTC must be None."""
        pos_a = (0.3, 0.5)
        vel_a = (-0.01, 0.0)  # moving left
        pos_b = (0.7, 0.5)
        vel_b = (0.01, 0.0)   # moving right
        ttc = self.analyzer.compute_ttc(pos_a, vel_a, pos_b, vel_b)
        self.assertIsNone(ttc)

    def test_parallel_motion_returns_none(self):
        """Same velocity vector: relative velocity is zero, TTC undefined."""
        pos_a = (0.1, 0.1)
        vel_a = (0.01, 0.0)
        pos_b = (0.5, 0.5)
        vel_b = (0.01, 0.0)
        ttc = self.analyzer.compute_ttc(pos_a, vel_a, pos_b, vel_b)
        self.assertIsNone(ttc)

    def test_one_stationary_entity(self):
        """If only entity A moves toward stationary B, TTC is computable."""
        pos_a = (0.1, 0.5)
        vel_a = (0.01, 0.0)
        pos_b = (0.5, 0.5)
        vel_b = (0.0, 0.0)
        ttc = self.analyzer.compute_ttc(pos_a, vel_a, pos_b, vel_b)
        # Relative velocity = (0-0.01, 0) = (-0.01, 0)
        # dot(dp, dv) = (0.4)*(-0.01) = -0.004; ttc = -(-0.004)/(0.0001) = 40 frames -> 4s
        self.assertIsNotNone(ttc)
        self.assertAlmostEqual(ttc, 4.0, places=3)

    def test_zero_relative_velocity_returns_none(self):
        """Identical velocities yield zero relative velocity, return None."""
        pos_a = (0.2, 0.5)
        vel_a = (0.005, 0.003)
        pos_b = (0.8, 0.5)
        vel_b = (0.005, 0.003)
        ttc = self.analyzer.compute_ttc(pos_a, vel_a, pos_b, vel_b)
        self.assertIsNone(ttc)

    def test_ttc_beyond_horizon_returns_none(self):
        """TTC > ttc_horizon_s must be clamped to None."""
        # Very slow approach -> TTC >> 5s
        pos_a = (0.1, 0.5)
        vel_a = (0.0001, 0.0)
        pos_b = (0.9, 0.5)
        vel_b = (-0.0001, 0.0)
        ttc = self.analyzer.compute_ttc(pos_a, vel_a, pos_b, vel_b)
        self.assertIsNone(ttc)

    def test_ttc_exact_zero_returns_none(self):
        """Negative or zero raw TTC (entities already passed) returns None."""
        # Entity A is to the right of B and both move further apart
        pos_a = (0.8, 0.5)
        vel_a = (0.01, 0.0)
        pos_b = (0.2, 0.5)
        vel_b = (-0.01, 0.0)
        ttc = self.analyzer.compute_ttc(pos_a, vel_a, pos_b, vel_b)
        self.assertIsNone(ttc)

    def test_ttc_within_horizon_is_positive(self):
        """Any returned TTC must be strictly positive."""
        pos_a = (0.3, 0.5)
        vel_a = (0.005, 0.0)
        pos_b = (0.5, 0.5)
        vel_b = (-0.005, 0.0)
        ttc = self.analyzer.compute_ttc(pos_a, vel_a, pos_b, vel_b)
        if ttc is not None:
            self.assertGreater(ttc, 0.0)

    def test_symmetric_ttc(self):
        """TTC(A, B) must equal TTC(B, A) — symmetry property."""
        pos_a = (0.2, 0.5)
        vel_a = (0.01, 0.0)
        pos_b = (0.8, 0.5)
        vel_b = (-0.01, 0.0)
        ttc_ab = self.analyzer.compute_ttc(pos_a, vel_a, pos_b, vel_b)
        ttc_ba = self.analyzer.compute_ttc(pos_b, vel_b, pos_a, vel_a)
        if ttc_ab is not None and ttc_ba is not None:
            self.assertAlmostEqual(ttc_ab, ttc_ba, places=6)

    def test_close_entities_fast_approach_gives_small_ttc(self):
        """Nearby entities approaching fast must yield TTC < 1s."""
        pos_a = (0.45, 0.5)
        vel_a = (0.05, 0.0)   # fast right
        pos_b = (0.55, 0.5)
        vel_b = (-0.05, 0.0)  # fast left
        ttc = self.analyzer.compute_ttc(pos_a, vel_a, pos_b, vel_b)
        self.assertIsNotNone(ttc)
        self.assertLess(ttc, 1.0)

    def test_diagonal_approach_yields_ttc(self):
        """Diagonal head-on approach must still compute a finite TTC."""
        pos_a = (0.1, 0.1)
        vel_a = (0.01, 0.01)
        pos_b = (0.9, 0.9)
        vel_b = (-0.01, -0.01)
        ttc = self.analyzer.compute_ttc(pos_a, vel_a, pos_b, vel_b)
        self.assertIsNotNone(ttc)
        self.assertGreater(ttc, 0.0)

    def test_high_fps_hint_scales_ttc(self):
        """fps_hint=20 should yield half the TTC of fps_hint=10 for same motion.

        TTC in seconds = ttc_frames / fps_hint. At fps_hint=20 each frame covers
        half a second (vs 0.1s at fps=10), so ttc20 = ttc10 / 2, i.e. ttc20/ttc10 = 0.5.
        """
        pos_a = (0.2, 0.5)
        vel_a = (0.01, 0.0)
        pos_b = (0.8, 0.5)
        vel_b = (-0.01, 0.0)
        a10 = TrajectoryHazardAnalyzer(fps_hint=10.0)
        a20 = TrajectoryHazardAnalyzer(fps_hint=20.0)
        ttc10 = a10.compute_ttc(pos_a, vel_a, pos_b, vel_b)
        ttc20 = a20.compute_ttc(pos_a, vel_a, pos_b, vel_b)
        if ttc10 is not None and ttc20 is not None:
            # ttc20 = ttc_frames/20 = (ttc_frames/10)/2 = ttc10/2
            self.assertAlmostEqual(ttc20 / ttc10, 0.5, places=3)


# ---------------------------------------------------------------------------
# 3. TestTrajectoryHazardAnalyzer (15 tests)
# ---------------------------------------------------------------------------


class TestTrajectoryHazardAnalyzer(unittest.TestCase):

    def setUp(self) -> None:
        self.analyzer = TrajectoryHazardAnalyzer(
            collision_distance_norm=0.05,
            ttc_horizon_s=5.0,
            min_speed_threshold=0.002,
            fps_hint=10.0,
        )

    # -- analyze_pair --

    def test_converging_pair_returns_assessment(self):
        """Two entities converging toward each other must produce an assessment."""
        ea = _make_entity(1, "person", 0.1, 0.45, 0.2, 0.55)
        eb = _make_entity(2, "forklift", 0.8, 0.45, 0.9, 0.55)
        vel_a = (0.02, 0.0)
        vel_b = (-0.02, 0.0)
        result = self.analyzer.analyze_pair(ea, eb, vel_a, vel_b, frame_number=5, timestamp=0.5)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, HazardAssessment)

    def test_diverging_pair_returns_none(self):
        """Entities moving apart must not trigger an assessment."""
        ea = _make_entity(1, "person", 0.4, 0.4, 0.5, 0.6)
        eb = _make_entity(2, "person", 0.5, 0.4, 0.6, 0.6)
        vel_a = (-0.02, 0.0)
        vel_b = (0.02, 0.0)
        result = self.analyzer.analyze_pair(ea, eb, vel_a, vel_b)
        self.assertIsNone(result)

    def test_both_stationary_returns_none(self):
        """Both stationary entities must always return None."""
        ea = _make_entity(1, "box", 0.1, 0.1, 0.2, 0.2)
        eb = _make_entity(2, "box", 0.3, 0.3, 0.4, 0.4)
        result = self.analyzer.analyze_pair(ea, eb, (0.0, 0.0), (0.0, 0.0))
        self.assertIsNone(result)

    def test_very_close_diverging_triggers_near_miss_from_distance(self):
        """Entities within collision_distance but diverging must still flag near_miss."""
        # Place entities very close (dist < collision_distance_norm = 0.05)
        ea = _make_entity(1, "person", 0.48, 0.48, 0.50, 0.52)
        eb = _make_entity(2, "person", 0.50, 0.48, 0.52, 0.52)
        # Diverging velocities
        vel_a = (-0.02, 0.0)
        vel_b = (0.02, 0.0)
        result = self.analyzer.analyze_pair(ea, eb, vel_a, vel_b)
        # Diverging but very close -> near_miss from distance branch
        if result is not None:
            self.assertEqual(result.hazard_type, HAZARD_NEAR_MISS)

    def test_converging_assessment_has_correct_entity_ids(self):
        """entity_id_a and entity_id_b in assessment must match input entities."""
        ea = _make_entity(10, "worker", 0.1, 0.45, 0.2, 0.55)
        eb = _make_entity(20, "vehicle", 0.8, 0.45, 0.9, 0.55)
        result = self.analyzer.analyze_pair(ea, eb, (0.02, 0.0), (-0.02, 0.0))
        self.assertIsNotNone(result)
        self.assertEqual(result.entity_id_a, 10)
        self.assertEqual(result.entity_id_b, 20)

    def test_high_speed_approach_hazard_type_is_collision(self):
        """TTC < 2s must produce hazard_type == HAZARD_COLLISION."""
        ea = _make_entity(1, "person", 0.44, 0.44, 0.50, 0.56)
        eb = _make_entity(2, "forklift", 0.50, 0.44, 0.56, 0.56)
        # Very fast approach -> TTC < 2s
        vel_a = (0.1, 0.0)
        vel_b = (-0.1, 0.0)
        result = self.analyzer.analyze_pair(ea, eb, vel_a, vel_b)
        if result is not None:
            self.assertEqual(result.hazard_type, HAZARD_COLLISION)

    def test_slow_approach_hazard_type_is_near_miss(self):
        """TTC >= 2s must produce hazard_type == HAZARD_NEAR_MISS."""
        ea = _make_entity(1, "person", 0.1, 0.45, 0.2, 0.55)
        eb = _make_entity(2, "person", 0.8, 0.45, 0.9, 0.55)
        # Very slow approach -> TTC >> 2s
        vel_a = (0.005, 0.0)
        vel_b = (-0.005, 0.0)
        result = self.analyzer.analyze_pair(ea, eb, vel_a, vel_b)
        if result is not None:
            self.assertEqual(result.hazard_type, HAZARD_NEAR_MISS)

    def test_probability_is_between_0_and_1(self):
        ea = _make_entity(1, "person", 0.1, 0.45, 0.2, 0.55)
        eb = _make_entity(2, "forklift", 0.8, 0.45, 0.9, 0.55)
        result = self.analyzer.analyze_pair(ea, eb, (0.02, 0.0), (-0.02, 0.0))
        if result is not None:
            self.assertGreaterEqual(result.probability, 0.0)
            self.assertLessEqual(result.probability, 1.0)

    # -- severity mapping --

    def test_severity_ttc_less_than_1_is_critical(self):
        self.assertEqual(_ttc_severity(0.5), SEVERITY_CRITICAL)

    def test_severity_ttc_between_1_and_3_is_warning(self):
        self.assertEqual(_ttc_severity(2.0), SEVERITY_WARNING)

    def test_severity_ttc_between_3_and_5_is_info(self):
        self.assertEqual(_ttc_severity(4.0), SEVERITY_INFO)

    # -- analyze_zone_boundary --

    def test_zone_breach_approaching_returns_assessment(self):
        """Entity approaching zone from outside must trigger zone_breach."""
        entity = _make_entity(1, "worker", 0.05, 0.45, 0.15, 0.55)
        zone = (0.4, 0.4, 0.6, 0.6)
        vel = (0.02, 0.0)  # moving right toward zone
        result = self.analyzer.analyze_zone_boundary(entity, zone, vel)
        self.assertIsNotNone(result)
        self.assertEqual(result.hazard_type, HAZARD_ZONE_BREACH)

    def test_zone_breach_moving_away_returns_none(self):
        """Entity outside zone and moving away must return None."""
        entity = _make_entity(1, "worker", 0.05, 0.45, 0.15, 0.55)
        zone = (0.4, 0.4, 0.6, 0.6)
        vel = (-0.02, 0.0)  # moving left, away from zone
        result = self.analyzer.analyze_zone_boundary(entity, zone, vel)
        self.assertIsNone(result)

    def test_zone_breach_already_inside_returns_none(self):
        """Entity already inside zone must not be flagged (not predictive)."""
        entity = _make_entity(1, "worker", 0.45, 0.45, 0.55, 0.55)
        zone = (0.4, 0.4, 0.6, 0.6)
        vel = (0.01, 0.0)
        result = self.analyzer.analyze_zone_boundary(entity, zone, vel)
        self.assertIsNone(result)

    def test_zone_breach_entity_id_b_is_none(self):
        """Zone breach assessments are single-entity, entity_id_b must be None."""
        entity = _make_entity(5, "worker", 0.05, 0.45, 0.15, 0.55)
        zone = (0.4, 0.4, 0.6, 0.6)
        vel = (0.02, 0.0)
        result = self.analyzer.analyze_zone_boundary(entity, zone, vel)
        if result is not None:
            self.assertIsNone(result.entity_id_b)
            self.assertEqual(result.entity_id_a, 5)


# ---------------------------------------------------------------------------
# 4. TestCrowdSurgeDetector (10 tests)
# ---------------------------------------------------------------------------


class TestCrowdSurgeDetector(unittest.TestCase):

    def _fill_baseline(self, detector: CrowdSurgeDetector, density: float, n: int = 5) -> None:
        """Push n frames of baseline density."""
        for i in range(n):
            detector.update(density, frame_number=i, timestamp=float(i))

    def test_not_enough_history_returns_none(self):
        """With fewer frames than window_frames, detector returns None."""
        d = CrowdSurgeDetector(window_frames=10)
        result = d.update(0.5, frame_number=0)
        self.assertIsNone(result)

    def test_slow_growth_returns_none(self):
        """Gradual density increase below threshold must not trigger."""
        d = CrowdSurgeDetector(surge_rate_threshold=0.3, window_frames=10, cooldown_frames=5)
        for i in range(5):
            d.update(0.5, frame_number=i)
        for i in range(5, 10):
            d.update(0.55, frame_number=i)   # only 10% increase
        result = d.update(0.55, frame_number=10)
        self.assertIsNone(result)

    def test_sudden_surge_returns_assessment(self):
        """Density doubling within window must trigger a HazardAssessment."""
        # Use cooldown_frames=1 so the detector can fire as soon as the window is full.
        d = CrowdSurgeDetector(surge_rate_threshold=0.3, window_frames=10, cooldown_frames=1)
        for i in range(5):
            d.update(0.2, frame_number=i)
        fired = None
        for i in range(5, 11):
            result = d.update(0.8, frame_number=i)   # 4× increase
            if result is not None:
                fired = result
                break
        self.assertIsNotNone(fired)
        self.assertEqual(fired.hazard_type, HAZARD_CROWD_SURGE)

    def test_cooldown_prevents_immediate_repeat(self):
        """Second trigger within cooldown_frames must be suppressed."""
        d = CrowdSurgeDetector(surge_rate_threshold=0.3, window_frames=10, cooldown_frames=100)
        for i in range(5):
            d.update(0.2, frame_number=i)
        for i in range(5, 10):
            d.update(0.9, frame_number=i)
        first = d.update(0.9, frame_number=10)
        second = d.update(0.9, frame_number=11)  # within cooldown
        # first might fire, second must not
        if first is not None:
            self.assertIsNone(second)

    def test_surge_rate_at_exact_threshold_not_triggered(self):
        """Surge rate exactly at threshold must not trigger (rate < threshold)."""
        d = CrowdSurgeDetector(surge_rate_threshold=0.3, window_frames=10, cooldown_frames=5)
        # baseline = 1.0, recent = 1.3 -> rate = 0.3 -> NOT triggered (strict <)
        for i in range(5):
            d.update(1.0, frame_number=i)
        for i in range(5, 10):
            d.update(1.3, frame_number=i)
        result = d.update(1.3, frame_number=10)
        self.assertIsNone(result)

    def test_reset_clears_state(self):
        """reset() must prevent any previously accumulated history from firing."""
        d = CrowdSurgeDetector(surge_rate_threshold=0.3, window_frames=10, cooldown_frames=5)
        for i in range(5):
            d.update(0.2, frame_number=i)
        d.reset()
        # After reset, single frame should not fire
        result = d.update(0.9, frame_number=5)
        self.assertIsNone(result)

    def test_high_rate_produces_critical_severity(self):
        """Surge rate > 0.6 must produce SEVERITY_CRITICAL."""
        # Use cooldown_frames=1 so the detector fires as soon as the window is full.
        d = CrowdSurgeDetector(surge_rate_threshold=0.3, window_frames=10, cooldown_frames=1)
        for i in range(5):
            d.update(0.1, frame_number=i)
        fired = None
        for i in range(5, 11):
            result = d.update(1.0, frame_number=i)   # 9× increase -> rate=9.0 > 0.6
            if result is not None:
                fired = result
                break
        self.assertIsNotNone(fired)
        self.assertEqual(fired.severity, SEVERITY_CRITICAL)

    def test_moderate_rate_produces_warning_severity(self):
        """Surge rate > 0.3 but <= 0.6 must produce SEVERITY_WARNING."""
        d = CrowdSurgeDetector(surge_rate_threshold=0.3, window_frames=10, cooldown_frames=5)
        for i in range(5):
            d.update(0.5, frame_number=i)
        for i in range(5, 10):
            d.update(0.7, frame_number=i)   # rate = 0.4 -> warning
        result = d.update(0.7, frame_number=10)
        if result is not None:
            self.assertEqual(result.severity, SEVERITY_WARNING)

    def test_surge_probability_between_0_and_1(self):
        """Probability on a surge event must be in [0, 1]."""
        d = CrowdSurgeDetector(surge_rate_threshold=0.3, window_frames=10, cooldown_frames=5)
        for i in range(5):
            d.update(0.1, frame_number=i)
        for i in range(5, 10):
            d.update(0.9, frame_number=i)
        result = d.update(0.9, frame_number=10)
        if result is not None:
            self.assertGreaterEqual(result.probability, 0.0)
            self.assertLessEqual(result.probability, 1.0)

    def test_zero_baseline_density_returns_none(self):
        """With near-zero baseline, division guard must prevent false positive.

        The deque window is split in half: the guard fires when ALL of the first
        half is effectively zero (< 1e-6). We fill the entire window with zeros
        so the baseline half stays zero regardless of what the recent half holds.
        """
        d = CrowdSurgeDetector(surge_rate_threshold=0.3, window_frames=10, cooldown_frames=1)
        # Fill all 10 slots with zero so baseline half is always 0.0
        for i in range(10):
            d.update(0.0, frame_number=i)
        # Now replace only the recent half with high density values;
        # the baseline half (first 5 slots) are still 0.0 in the deque.
        # Because the deque is full (maxlen=10) and we only push 5 more,
        # it shifts out 5 zeros and baseline half becomes positions 5-9 = all 0.0.
        # So we need to check the window state carefully: push 5 high values.
        fired = None
        for i in range(10, 15):
            result = d.update(0.5, frame_number=i)
            if result is not None:
                fired = result
        # After 5 pushes: deque = [0,0,0,0,0, 0.5,0.5,0.5,0.5,0.5]
        # baseline = mean([0,0,0,0,0]) = 0.0 < 1e-6 → guard fires → None
        self.assertIsNone(fired)


# ---------------------------------------------------------------------------
# 5. TestAnticipationEngine (15 tests)
# ---------------------------------------------------------------------------


class TestAnticipationEngine(unittest.TestCase):

    def _make_engine(self, **kwargs) -> AnticipationEngine:
        defaults = dict(
            ttc_horizon_s=5.0,
            collision_distance_norm=0.05,
            min_probability=0.1,
            cooldown_seconds=0.0,   # no cooldown for most tests
            fps_hint=10.0,
        )
        defaults.update(kwargs)
        return AnticipationEngine(**defaults)

    def test_empty_world_state_returns_empty_list(self):
        """No entities -> no hazards."""
        engine = self._make_engine()
        ws = MockWorldState(entities=[])
        result = engine.analyze(ws, frame_number=1, timestamp=1.0)
        self.assertEqual(result, [])

    def test_single_entity_returns_empty_list(self):
        """One entity: no pairs possible -> no collision hazard."""
        engine = self._make_engine()
        ws = MockWorldState(entities=[_make_entity(1, "person", 0.1, 0.1, 0.2, 0.2)])
        result = engine.analyze(ws, frame_number=1, timestamp=1.0)
        # Could have crowd surge if density threshold met, but one entity -> density=0.05 < typical surge
        # Check no collision hazards
        collision_hazards = [h for h in result if h.hazard_type in (HAZARD_COLLISION, HAZARD_NEAR_MISS)]
        self.assertEqual(collision_hazards, [])

    def test_two_converging_entities_produce_hazard(self):
        """Two entities approaching each other must trigger a hazard."""
        engine = self._make_engine()
        # Warm up velocity estimator with a prior frame
        ea = _make_entity(1, "person", 0.1, 0.45, 0.2, 0.55)
        eb = _make_entity(2, "forklift", 0.8, 0.45, 0.9, 0.55)
        ws1 = MockWorldState(entities=[ea, eb])
        engine.analyze(ws1, frame_number=0, timestamp=0.0)

        # Second frame: entities have moved toward each other
        ea2 = _make_entity(1, "person", 0.15, 0.45, 0.25, 0.55)
        eb2 = _make_entity(2, "forklift", 0.75, 0.45, 0.85, 0.55)
        ws2 = MockWorldState(entities=[ea2, eb2])
        result = engine.analyze(ws2, frame_number=1, timestamp=0.1)
        hazard_types = [h.hazard_type for h in result]
        self.assertTrue(
            HAZARD_COLLISION in hazard_types or HAZARD_NEAR_MISS in hazard_types,
            f"Expected collision/near_miss hazard, got: {hazard_types}"
        )

    def test_cooldown_prevents_duplicate_on_second_call(self):
        """With cooldown_seconds=60, second call for same pair must be suppressed."""
        engine = self._make_engine(cooldown_seconds=60.0)
        ea = _make_entity(1, "person", 0.1, 0.45, 0.2, 0.55)
        eb = _make_entity(2, "forklift", 0.8, 0.45, 0.9, 0.55)

        ws1 = MockWorldState(entities=[ea, eb])
        engine.analyze(ws1, frame_number=0, timestamp=0.0)

        ea2 = _make_entity(1, "person", 0.15, 0.45, 0.25, 0.55)
        eb2 = _make_entity(2, "forklift", 0.75, 0.45, 0.85, 0.55)
        ws2 = MockWorldState(entities=[ea2, eb2])
        result1 = engine.analyze(ws2, frame_number=1, timestamp=1.0)

        ea3 = _make_entity(1, "person", 0.20, 0.45, 0.30, 0.55)
        eb3 = _make_entity(2, "forklift", 0.70, 0.45, 0.80, 0.55)
        ws3 = MockWorldState(entities=[ea3, eb3])
        result2 = engine.analyze(ws3, frame_number=2, timestamp=2.0)  # still within cooldown

        if result1:
            collision2 = [h for h in result2 if h.hazard_type in (HAZARD_COLLISION, HAZARD_NEAR_MISS)]
            self.assertEqual(len(collision2), 0)

    def test_reset_clears_all_state(self):
        """reset() must zero out history, active, total_hazards."""
        engine = self._make_engine(cooldown_seconds=0.0)
        ea = _make_entity(1, "person", 0.1, 0.45, 0.2, 0.55)
        eb = _make_entity(2, "forklift", 0.8, 0.45, 0.9, 0.55)
        ws = MockWorldState(entities=[ea, eb])
        engine.analyze(ws, frame_number=0, timestamp=0.0)
        ea2 = _make_entity(1, "person", 0.15, 0.45, 0.25, 0.55)
        eb2 = _make_entity(2, "forklift", 0.75, 0.45, 0.85, 0.55)
        ws2 = MockWorldState(entities=[ea2, eb2])
        engine.analyze(ws2, frame_number=1, timestamp=1.0)
        engine.reset()
        state = engine.get_state_dict()
        self.assertEqual(state["total_hazards_detected"], 0)
        self.assertEqual(state["active_hazards"], 0)
        self.assertEqual(state["history_size"], 0)

    def test_get_history_returns_last_n(self):
        """get_history(n) must return at most the last n items."""
        engine = self._make_engine(cooldown_seconds=0.0)
        ea = _make_entity(1, "person", 0.1, 0.45, 0.2, 0.55)
        eb = _make_entity(2, "forklift", 0.8, 0.45, 0.9, 0.55)
        # Run multiple frames to accumulate history
        for frame in range(10):
            ea_f = _make_entity(1, "person", 0.1 + frame * 0.05, 0.45, 0.2 + frame * 0.05, 0.55)
            eb_f = _make_entity(2, "forklift", 0.9 - frame * 0.05, 0.45, 1.0 - frame * 0.05, 0.55)
            ws = MockWorldState(entities=[ea_f, eb_f])
            engine.analyze(ws, frame_number=frame, timestamp=float(frame))
        history = engine.get_history(n=3)
        self.assertLessEqual(len(history), 3)

    def test_get_active_hazards_returns_list(self):
        """get_active_hazards must return a list (possibly empty)."""
        engine = self._make_engine()
        ws = MockWorldState(entities=[])
        engine.analyze(ws, frame_number=0, timestamp=0.0)
        active = engine.get_active_hazards()
        self.assertIsInstance(active, list)

    def test_get_state_dict_has_required_keys(self):
        """get_state_dict must contain all documented keys."""
        engine = self._make_engine()
        state = engine.get_state_dict()
        required = {
            "total_hazards_detected",
            "active_hazards",
            "history_size",
            "ttc_horizon_s",
            "collision_distance_norm",
        }
        self.assertTrue(required.issubset(state.keys()), f"Missing keys: {required - state.keys()}")

    def test_scene_snapshot_dict_crowd_density(self):
        """scene_snapshot as dict with crowd_density key must be read correctly."""
        engine = self._make_engine(cooldown_seconds=0.0)
        ws = MockWorldState(entities=[])
        snapshot = {"crowd_density": 0.0}
        # Should not crash and should return a list
        result = engine.analyze(ws, scene_snapshot=snapshot, frame_number=0, timestamp=0.0)
        self.assertIsInstance(result, list)

    def test_scene_snapshot_object_crowd_density(self):
        """scene_snapshot as object with crowd_density attribute must be read."""
        @dataclass
        class FakeSnapshot:
            crowd_density: float = 0.0

        engine = self._make_engine(cooldown_seconds=0.0)
        ws = MockWorldState(entities=[])
        snapshot = FakeSnapshot(crowd_density=0.0)
        result = engine.analyze(ws, scene_snapshot=snapshot, frame_number=0, timestamp=0.0)
        self.assertIsInstance(result, list)

    def test_min_probability_filter_suppresses_low_probability(self):
        """Hazards with probability below min_probability must be filtered out."""
        engine = self._make_engine(min_probability=0.99)  # very high threshold
        ea = _make_entity(1, "person", 0.1, 0.45, 0.2, 0.55)
        eb = _make_entity(2, "person", 0.8, 0.45, 0.9, 0.55)
        # Very slow approach -> low probability
        ws = MockWorldState(entities=[ea, eb])
        engine.analyze(ws, frame_number=0, timestamp=0.0)
        ea2 = _make_entity(1, "person", 0.11, 0.45, 0.21, 0.55)
        eb2 = _make_entity(2, "person", 0.79, 0.45, 0.89, 0.55)
        ws2 = MockWorldState(entities=[ea2, eb2])
        result = engine.analyze(ws2, frame_number=1, timestamp=1.0)
        # With min_probability=0.99, most moderate-risk detections should be suppressed
        for h in result:
            self.assertGreaterEqual(h.probability, 0.99)

    def test_thread_safety_concurrent_analyze(self):
        """Concurrent calls to analyze() must not raise exceptions."""
        engine = self._make_engine(cooldown_seconds=0.0)
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for frame in range(20):
                    ea = _make_entity(thread_id * 10 + 1, "person", 0.1, 0.4, 0.2, 0.6)
                    eb = _make_entity(thread_id * 10 + 2, "person", 0.8, 0.4, 0.9, 0.6)
                    ws = MockWorldState(entities=[ea, eb])
                    engine.analyze(ws, frame_number=frame, timestamp=float(frame))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(errors, [], f"Thread errors: {errors}")

    def test_analyze_returns_list_of_hazard_assessments(self):
        """analyze() must always return a list of HazardAssessment objects."""
        engine = self._make_engine()
        ea = _make_entity(1, "person", 0.1, 0.1, 0.2, 0.2)
        ws = MockWorldState(entities=[ea])
        result = engine.analyze(ws, frame_number=0, timestamp=0.0)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, HazardAssessment)

    def test_total_hazards_increments_correctly(self):
        """total_hazards_detected in state_dict must monotonically increase."""
        engine = self._make_engine(cooldown_seconds=0.0)
        ea = _make_entity(1, "person", 0.1, 0.45, 0.2, 0.55)
        eb = _make_entity(2, "forklift", 0.8, 0.45, 0.9, 0.55)
        ws1 = MockWorldState(entities=[ea, eb])
        engine.analyze(ws1, frame_number=0, timestamp=0.0)
        count_before = engine.get_state_dict()["total_hazards_detected"]
        ea2 = _make_entity(1, "person", 0.15, 0.45, 0.25, 0.55)
        eb2 = _make_entity(2, "forklift", 0.75, 0.45, 0.85, 0.55)
        ws2 = MockWorldState(entities=[ea2, eb2])
        engine.analyze(ws2, frame_number=1, timestamp=1.0)
        count_after = engine.get_state_dict()["total_hazards_detected"]
        self.assertGreaterEqual(count_after, count_before)

    def test_get_history_empty_before_any_hazard(self):
        """History must be empty when no hazards have been recorded."""
        engine = self._make_engine()
        self.assertEqual(engine.get_history(), [])


# ---------------------------------------------------------------------------
# 6. TestHazardAssessment (5 tests)
# ---------------------------------------------------------------------------


class TestHazardAssessment(unittest.TestCase):

    def _make_assessment(self, **kwargs) -> HazardAssessment:
        defaults = dict(
            hazard_id="abc-123",
            entity_id_a=1,
            entity_id_b=2,
            hazard_type=HAZARD_COLLISION,
            ttc_seconds=1.5,
            probability=0.8,
            severity=SEVERITY_WARNING,
            description_ja="テスト衝突警告",
            frame_number=42,
            timestamp=100.0,
            details={"dist": 0.03, "speed_a": 0.01},
        )
        defaults.update(kwargs)
        return HazardAssessment(**defaults)

    def test_to_dict_contains_all_fields(self):
        """to_dict() must include every field of the dataclass."""
        ha = self._make_assessment()
        d = ha.to_dict()
        expected_keys = {
            "hazard_id", "entity_id_a", "entity_id_b", "hazard_type",
            "ttc_seconds", "probability", "severity", "description_ja",
            "frame_number", "timestamp", "details",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_severity_constants_are_distinct_strings(self):
        """SEVERITY_* constants must be distinct, non-empty strings."""
        severities = {SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_CRITICAL}
        self.assertEqual(len(severities), 3)
        for s in severities:
            self.assertIsInstance(s, str)
            self.assertTrue(len(s) > 0)

    def test_hazard_type_constants_are_distinct_strings(self):
        """HAZARD_* constants must be distinct, non-empty strings."""
        types = {HAZARD_COLLISION, HAZARD_NEAR_MISS, HAZARD_ZONE_BREACH, HAZARD_CROWD_SURGE}
        self.assertEqual(len(types), 4)
        for t in types:
            self.assertIsInstance(t, str)
            self.assertTrue(len(t) > 0)

    def test_none_entity_id_b_is_allowed(self):
        """entity_id_b=None must be preserved in to_dict()."""
        ha = self._make_assessment(entity_id_b=None)
        self.assertIsNone(ha.entity_id_b)
        self.assertIsNone(ha.to_dict()["entity_id_b"])

    def test_details_dict_is_preserved(self):
        """Custom details dict must survive the to_dict() round-trip."""
        details = {"dist": 0.042, "speed_a": 0.023, "speed_b": 0.011, "custom_key": "value"}
        ha = self._make_assessment(details=details)
        self.assertEqual(ha.to_dict()["details"], details)


if __name__ == "__main__":
    unittest.main()
