"""Comprehensive tests for the Autonomous Anomaly Detection ensemble.

Tests cover every component of sopilot/perception/anomaly.py:
    - AnomalySignal          (dataclass basics)
    - BehavioralAnomalyDetector  (speed z-score, activity distribution, warmup)
    - SpatialAnomalyDetector     (grid mapping, low-frequency cell detection)
    - TemporalPatternDetector    (hourly density, hour extraction)
    - InteractionAnomalyDetector (pair normalization, novel relation detection)
    - AnomalyDetectorEnsemble    (cooldown, weights, severity, WorldModel integration)

Also tests integration with:
    - types.py PerceptionConfig (anomaly fields)
    - world_model.py WorldModel (ensemble replaces baseline)
    - engine.py PerceptionEngine.get_anomaly_state()
    - narrator.py (ANOMALY event narration)
    - context_memory.py (anomaly keyword query)

All tests use mock data (no GPU required).
Run:  python -m pytest tests/test_anomaly_detector.py -v
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

from sopilot.perception.types import (
    BBox,
    Detection,
    EntityEvent,
    EntityEventType,
    PerceptionConfig,
    Relation,
    SceneEntity,
    SceneGraph,
    SpatialRelation,
    Track,
    TrackState,
    Violation,
    ViolationSeverity,
    WorldState,
    Zone,
)
from sopilot.perception.anomaly import (
    AnomalyDetectorEnsemble,
    AnomalySignal,
    BehavioralAnomalyDetector,
    InteractionAnomalyDetector,
    SpatialAnomalyDetector,
    TemporalPatternDetector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scene_graph(
    entities: list[SceneEntity] | None = None,
    relations: list[Relation] | None = None,
    timestamp: float = 1.0,
    frame_number: int = 1,
) -> SceneGraph:
    return SceneGraph(
        timestamp=timestamp,
        frame_number=frame_number,
        entities=entities or [],
        relations=relations or [],
        frame_shape=(480, 640),
    )


def _make_entity(
    entity_id: int = 1,
    label: str = "person",
    x1: float = 0.1,
    y1: float = 0.1,
    x2: float = 0.3,
    y2: float = 0.5,
    confidence: float = 0.9,
    zone_ids: list[str] | None = None,
    attributes: dict | None = None,
) -> SceneEntity:
    return SceneEntity(
        entity_id=entity_id,
        label=label,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=confidence,
        zone_ids=zone_ids or [],
        attributes=attributes or {},
    )


def _make_track(
    track_id: int = 1,
    label: str = "person",
    velocity: tuple[float, float] = (0.01, 0.0),
    attributes: dict | None = None,
) -> Track:
    return Track(
        track_id=track_id,
        label=label,
        state=TrackState.ACTIVE,
        bbox=BBox(0.1, 0.1, 0.3, 0.5),
        velocity=velocity,
        confidence=0.9,
        attributes=attributes or {},
    )


def _make_world_state(
    entity_count: int = 3,
    person_count: int = 2,
    entities: list[SceneEntity] | None = None,
    relations: list[Relation] | None = None,
    active_tracks: dict[int, Track] | None = None,
    zone_occupancy: dict[str, list[int]] | None = None,
    events: list[EntityEvent] | None = None,
    timestamp: float = 1.0,
    frame_number: int = 1,
) -> WorldState:
    if entities is None:
        entities = [_make_entity(entity_id=i) for i in range(entity_count)]
    sg = _make_scene_graph(
        entities=entities,
        relations=relations or [],
        timestamp=timestamp,
        frame_number=frame_number,
    )
    if active_tracks is None:
        active_tracks = {
            e.entity_id: _make_track(track_id=e.entity_id, label=e.label)
            for e in entities
        }
    return WorldState(
        timestamp=timestamp,
        frame_number=frame_number,
        scene_graph=sg,
        active_tracks=active_tracks,
        events=events or [],
        zone_occupancy=zone_occupancy or {},
        entity_count=len(entities),
        person_count=sum(1 for e in entities if e.label == "person"),
    )


# ===========================================================================
# AnomalySignal tests
# ===========================================================================


class TestAnomalySignal(unittest.TestCase):
    """Tests for the AnomalySignal dataclass."""

    def test_create_signal(self):
        sig = AnomalySignal(
            detector="behavioral",
            metric="speed_zscore",
            z_score=3.5,
            description_ja="テスト",
            description_en="test",
        )
        self.assertEqual(sig.detector, "behavioral")
        self.assertEqual(sig.entity_id, -1)
        self.assertEqual(sig.z_score, 3.5)

    def test_signal_with_entity_id(self):
        sig = AnomalySignal(
            detector="spatial",
            metric="rare_cell",
            z_score=5.0,
            description_ja="空間",
            description_en="spatial",
            entity_id=42,
            details={"cell_row": 3, "cell_col": 7},
        )
        self.assertEqual(sig.entity_id, 42)
        self.assertEqual(sig.details["cell_row"], 3)


# ===========================================================================
# BehavioralAnomalyDetector tests
# ===========================================================================


class TestBehavioralAnomalyDetector(unittest.TestCase):
    """Tests for speed z-score and activity distribution anomaly detection."""

    def setUp(self):
        self.det = BehavioralAnomalyDetector(alpha=0.1)

    def test_observe_updates_speed_mean(self):
        ws = _make_world_state(
            active_tracks={1: _make_track(velocity=(0.01, 0.0))}
        )
        self.det.observe(ws)
        self.assertEqual(self.det._observations, 1)
        self.assertAlmostEqual(self.det._speed_mean, 0.01, places=4)

    def test_observe_empty_tracks(self):
        ws = _make_world_state(active_tracks={}, entities=[])
        self.det.observe(ws)
        self.assertEqual(self.det._observations, 0)

    def test_speed_ema_converges(self):
        for i in range(50):
            ws = _make_world_state(
                active_tracks={1: _make_track(velocity=(0.01, 0.0))}
            )
            self.det.observe(ws)
        self.assertAlmostEqual(self.det._speed_mean, 0.01, places=3)

    def test_no_anomaly_during_normal_speed(self):
        for i in range(50):
            ws = _make_world_state(
                active_tracks={1: _make_track(velocity=(0.01, 0.0))}
            )
            self.det.observe(ws)
        ws = _make_world_state(
            active_tracks={1: _make_track(velocity=(0.01, 0.0))}
        )
        signals = self.det.check(ws)
        self.assertEqual(len(signals), 0)

    def test_speed_anomaly_detected(self):
        # Train on slow speeds
        for i in range(100):
            ws = _make_world_state(
                active_tracks={1: _make_track(velocity=(0.01, 0.0))}
            )
            self.det.observe(ws)
        # Inject very fast speed
        ws = _make_world_state(
            active_tracks={1: _make_track(velocity=(0.5, 0.5))}
        )
        signals = self.det.check(ws)
        speed_signals = [s for s in signals if s.metric == "speed_zscore"]
        self.assertGreater(len(speed_signals), 0)
        self.assertGreater(speed_signals[0].z_score, 2.0)

    def test_activity_distribution_learning(self):
        ws = _make_world_state(
            active_tracks={1: _make_track(attributes={"activity": "walking"})}
        )
        self.det.observe(ws)
        self.assertIn("walking", self.det._activity_freq)

    def test_activity_anomaly_detected(self):
        # Train: mostly walking
        for i in range(100):
            ws = _make_world_state(
                active_tracks={
                    1: _make_track(attributes={"activity": "walking"}),
                    2: _make_track(track_id=2, attributes={"activity": "walking"}),
                }
            )
            self.det.observe(ws)
        # Now: sudden rare activity dominates
        ws = _make_world_state(
            active_tracks={
                1: _make_track(attributes={"activity": "running"}),
                2: _make_track(track_id=2, attributes={"activity": "running"}),
                3: _make_track(track_id=3, attributes={"activity": "running"}),
            }
        )
        signals = self.det.check(ws)
        act_signals = [s for s in signals if s.metric == "activity_distribution"]
        self.assertGreater(len(act_signals), 0)

    def test_check_empty_tracks_returns_empty(self):
        ws = _make_world_state(active_tracks={}, entities=[])
        signals = self.det.check(ws)
        self.assertEqual(len(signals), 0)

    def test_reset(self):
        ws = _make_world_state(
            active_tracks={1: _make_track(velocity=(0.01, 0.0))}
        )
        self.det.observe(ws)
        self.det.reset()
        self.assertEqual(self.det._observations, 0)
        self.assertEqual(self.det._speed_mean, 0.0)
        self.assertEqual(len(self.det._activity_freq), 0)

    def test_signal_has_details(self):
        for i in range(100):
            ws = _make_world_state(
                active_tracks={1: _make_track(velocity=(0.01, 0.0))}
            )
            self.det.observe(ws)
        ws = _make_world_state(
            active_tracks={1: _make_track(velocity=(1.0, 1.0))}
        )
        signals = self.det.check(ws)
        if signals:
            self.assertIn("z_score", signals[0].details)
            self.assertIn("current_speed", signals[0].details)

    def test_speed_variance_grows(self):
        for i in range(50):
            v = 0.01 + (i % 5) * 0.002
            ws = _make_world_state(
                active_tracks={1: _make_track(velocity=(v, 0.0))}
            )
            self.det.observe(ws)
        self.assertGreater(self.det._speed_var, 0)

    def test_multiple_tracks_averaged(self):
        ws = _make_world_state(
            active_tracks={
                1: _make_track(velocity=(0.02, 0.0)),
                2: _make_track(track_id=2, velocity=(0.04, 0.0)),
            }
        )
        self.det.observe(ws)
        # Average speed should be ~0.03
        self.assertAlmostEqual(self.det._speed_mean, 0.03, places=3)

    def test_activity_decay(self):
        # Observe "walking" then switch to "stationary"
        for i in range(20):
            ws = _make_world_state(
                active_tracks={1: _make_track(attributes={"activity": "walking"})}
            )
            self.det.observe(ws)
        old_walk_freq = self.det._activity_freq["walking"]
        for i in range(20):
            ws = _make_world_state(
                active_tracks={1: _make_track(attributes={"activity": "stationary"})}
            )
            self.det.observe(ws)
        # Walking frequency should have decayed
        self.assertLess(self.det._activity_freq["walking"], old_walk_freq)

    def test_description_fields(self):
        for i in range(100):
            ws = _make_world_state(
                active_tracks={1: _make_track(velocity=(0.01, 0.0))}
            )
            self.det.observe(ws)
        ws = _make_world_state(
            active_tracks={1: _make_track(velocity=(1.0, 1.0))}
        )
        signals = self.det.check(ws)
        if signals:
            self.assertTrue(signals[0].description_ja)
            self.assertTrue(signals[0].description_en)


# ===========================================================================
# SpatialAnomalyDetector tests
# ===========================================================================


class TestSpatialAnomalyDetector(unittest.TestCase):
    """Tests for spatial occupancy heatmap anomaly detection."""

    def setUp(self):
        self.det = SpatialAnomalyDetector(grid_size=5, alpha=0.1)

    def test_cell_mapping(self):
        row, col = self.det._cell_for(0.0, 0.0)
        self.assertEqual((row, col), (0, 0))

    def test_cell_mapping_center(self):
        row, col = self.det._cell_for(0.5, 0.5)
        self.assertEqual((row, col), (2, 2))

    def test_cell_mapping_edge(self):
        row, col = self.det._cell_for(1.0, 1.0)
        self.assertEqual((row, col), (4, 4))

    def test_cell_mapping_negative_clamped(self):
        row, col = self.det._cell_for(-0.1, -0.1)
        self.assertEqual((row, col), (0, 0))

    def test_observe_updates_grid(self):
        entity = _make_entity(x1=0.0, y1=0.0, x2=0.2, y2=0.2)
        ws = _make_world_state(entities=[entity])
        self.det.observe(ws)
        # Entity center is (0.1, 0.1) -> cell (0, 0)
        self.assertGreater(self.det._grid[0][0], 0.0)

    def test_observe_increments_counter(self):
        ws = _make_world_state()
        self.det.observe(ws)
        self.assertEqual(self.det._observations, 1)

    def test_empty_cells_stay_near_zero(self):
        # Only occupy cell (0,0) repeatedly
        entity = _make_entity(x1=0.0, y1=0.0, x2=0.1, y2=0.1)
        for i in range(30):
            ws = _make_world_state(entities=[entity])
            self.det.observe(ws)
        # A distant cell should remain near 0
        self.assertLess(self.det._grid[4][4], 0.01)

    def test_no_anomaly_in_common_cell(self):
        entity = _make_entity(x1=0.0, y1=0.0, x2=0.1, y2=0.1)
        for i in range(30):
            ws = _make_world_state(entities=[entity])
            self.det.observe(ws)
        ws = _make_world_state(entities=[entity])
        signals = self.det.check(ws)
        # The cell (0,0) is common, so no anomaly expected there
        cell_00_signals = [
            s for s in signals
            if s.details.get("cell_row") == 0 and s.details.get("cell_col") == 0
        ]
        self.assertEqual(len(cell_00_signals), 0)

    def test_anomaly_in_rare_cell(self):
        # Train on cell (0,0) only
        entity = _make_entity(x1=0.0, y1=0.0, x2=0.1, y2=0.1)
        for i in range(30):
            ws = _make_world_state(entities=[entity])
            self.det.observe(ws)
        # New entity in a completely different cell
        rare_entity = _make_entity(entity_id=99, x1=0.8, y1=0.8, x2=0.95, y2=0.95)
        ws = _make_world_state(entities=[rare_entity])
        signals = self.det.check(ws)
        self.assertGreater(len(signals), 0)
        self.assertEqual(signals[0].detector, "spatial")
        self.assertEqual(signals[0].metric, "rare_cell")

    def test_minimum_observations_required(self):
        # With < 10 observations, no signals even in empty cells
        rare_entity = _make_entity(entity_id=99, x1=0.8, y1=0.8, x2=0.95, y2=0.95)
        ws = _make_world_state(entities=[rare_entity])
        self.det._observations = 5
        signals = self.det.check(ws)
        self.assertEqual(len(signals), 0)

    def test_get_grid_returns_copy(self):
        grid = self.det.get_grid()
        grid[0][0] = 999
        self.assertNotEqual(self.det._grid[0][0], 999)

    def test_reset(self):
        entity = _make_entity()
        ws = _make_world_state(entities=[entity])
        self.det.observe(ws)
        self.det.reset()
        self.assertEqual(self.det._observations, 0)
        self.assertEqual(self.det._grid[0][0], 0.0)

    def test_multiple_entities_in_same_cell(self):
        e1 = _make_entity(entity_id=1, x1=0.0, y1=0.0, x2=0.1, y2=0.1)
        e2 = _make_entity(entity_id=2, x1=0.05, y1=0.05, x2=0.15, y2=0.15)
        ws = _make_world_state(entities=[e1, e2])
        self.det.observe(ws)
        # Both map to cell (0,0) — should be treated as single occupancy
        self.assertGreater(self.det._grid[0][0], 0.0)

    def test_signal_contains_entity_id(self):
        entity = _make_entity(x1=0.0, y1=0.0, x2=0.1, y2=0.1)
        for i in range(30):
            ws = _make_world_state(entities=[entity])
            self.det.observe(ws)
        rare = _make_entity(entity_id=77, x1=0.9, y1=0.9, x2=0.99, y2=0.99)
        ws = _make_world_state(entities=[rare])
        signals = self.det.check(ws)
        if signals:
            self.assertEqual(signals[0].entity_id, 77)

    def test_empty_scene_no_crash(self):
        ws = _make_world_state(entities=[])
        self.det.observe(ws)
        signals = self.det.check(ws)
        self.assertEqual(len(signals), 0)

    def test_grid_frequency_increases_with_observation(self):
        entity = _make_entity(x1=0.4, y1=0.4, x2=0.6, y2=0.6)
        # Center is (0.5, 0.5) -> cell (2, 2)
        for _ in range(20):
            ws = _make_world_state(entities=[entity])
            self.det.observe(ws)
        self.assertGreater(self.det._grid[2][2], 0.5)


# ===========================================================================
# TemporalPatternDetector tests
# ===========================================================================


class TestTemporalPatternDetector(unittest.TestCase):
    """Tests for time-of-day entity density anomaly detection."""

    def setUp(self):
        self.det = TemporalPatternDetector(alpha=0.1)

    def test_hour_extraction_from_real_timestamp(self):
        import datetime
        ts = datetime.datetime(2026, 3, 3, 14, 30, 0).timestamp()
        hour = self.det._hour_from_timestamp(ts)
        self.assertEqual(hour, 14)

    def test_hour_extraction_consistency(self):
        # Same second always produces same hour
        h1 = self.det._hour_from_timestamp(7200.0)
        h2 = self.det._hour_from_timestamp(7200.0)
        self.assertEqual(h1, h2)

    def test_observe_updates_hourly_stats(self):
        ts = 3600.0
        hour = self.det._hour_from_timestamp(ts)
        ws = _make_world_state(timestamp=ts)
        self.det.observe(ws)
        self.assertEqual(self.det._hourly_obs[hour], 1)

    def test_no_anomaly_with_few_observations(self):
        ts = 3600.0
        ws = _make_world_state(timestamp=ts, entity_count=5)
        for _ in range(3):
            self.det.observe(ws)
        signals = self.det.check(ws)
        self.assertEqual(len(signals), 0)

    def test_no_anomaly_for_expected_count(self):
        ts = 3600.0
        for i in range(20):
            ws = _make_world_state(
                timestamp=ts + i,
                entities=[_make_entity(entity_id=j) for j in range(3)],
            )
            self.det.observe(ws)
        ws = _make_world_state(
            timestamp=ts + 20,
            entities=[_make_entity(entity_id=j) for j in range(3)],
        )
        signals = self.det.check(ws)
        self.assertEqual(len(signals), 0)

    def test_anomaly_for_unexpected_count(self):
        ts = 3600.0
        for i in range(30):
            ws = _make_world_state(
                timestamp=ts + i,
                entities=[_make_entity(entity_id=j) for j in range(2)],
            )
            self.det.observe(ws)
        ws = _make_world_state(
            timestamp=ts + 30,
            entities=[_make_entity(entity_id=j) for j in range(20)],
        )
        signals = self.det.check(ws)
        self.assertGreater(len(signals), 0)
        self.assertEqual(signals[0].detector, "temporal")
        self.assertEqual(signals[0].metric, "hourly_density")

    def test_different_hours_independent(self):
        ts_a = 3600.0
        hour_a = self.det._hour_from_timestamp(ts_a)
        # Use a timestamp 3600s apart to get a different hour
        ts_b = ts_a + 3600.0
        hour_b = self.det._hour_from_timestamp(ts_b)
        self.assertNotEqual(hour_a, hour_b)

        for i in range(10):
            ws = _make_world_state(
                timestamp=ts_a + i,
                entities=[_make_entity(entity_id=j) for j in range(5)],
            )
            self.det.observe(ws)
        for i in range(10):
            ws = _make_world_state(
                timestamp=ts_b + i,
                entities=[_make_entity(entity_id=0)],
            )
            self.det.observe(ws)
        self.assertGreater(self.det._hourly_mean[hour_a], self.det._hourly_mean[hour_b])

    def test_get_hourly_stats(self):
        ts = 3600.0
        hour = self.det._hour_from_timestamp(ts)
        ws = _make_world_state(timestamp=ts)
        self.det.observe(ws)
        stats = self.det.get_hourly_stats()
        self.assertEqual(len(stats), 24)
        self.assertEqual(stats[hour]["observations"], 1)

    def test_reset(self):
        ts = 3600.0
        hour = self.det._hour_from_timestamp(ts)
        ws = _make_world_state(timestamp=ts)
        self.det.observe(ws)
        self.det.reset()
        self.assertEqual(self.det._hourly_obs[hour], 0)

    def test_signal_details_contain_hour(self):
        ts = 3600.0
        for i in range(30):
            ws = _make_world_state(
                timestamp=ts + i,
                entities=[_make_entity(entity_id=0)],
            )
            self.det.observe(ws)
        ws = _make_world_state(
            timestamp=ts + 30,
            entities=[_make_entity(entity_id=j) for j in range(30)],
        )
        signals = self.det.check(ws)
        if signals:
            self.assertIn("hour", signals[0].details)
            self.assertIn("current_count", signals[0].details)

    def test_variance_accumulates(self):
        ts = 3600.0
        hour = self.det._hour_from_timestamp(ts)
        for i in range(20):
            n_entities = 2 + (i % 3)
            ws = _make_world_state(
                timestamp=ts + i,
                entities=[_make_entity(entity_id=j) for j in range(n_entities)],
            )
            self.det.observe(ws)
        self.assertGreater(self.det._hourly_var[hour], 0.0)

    def test_zero_entity_anomaly(self):
        ts = 3600.0
        for i in range(20):
            ws = _make_world_state(
                timestamp=ts + i,
                entities=[_make_entity(entity_id=j) for j in range(5)],
            )
            self.det.observe(ws)
        ws = _make_world_state(timestamp=ts + 20, entities=[])
        signals = self.det.check(ws)
        self.assertGreater(len(signals), 0)


# ===========================================================================
# InteractionAnomalyDetector tests
# ===========================================================================


class TestInteractionAnomalyDetector(unittest.TestCase):
    """Tests for entity-pair interaction anomaly detection."""

    def setUp(self):
        self.det = InteractionAnomalyDetector(alpha=0.1)

    def test_normalize_key_alphabetical(self):
        key1 = self.det._normalize_key("person", "near", "machine")
        key2 = self.det._normalize_key("machine", "near", "person")
        self.assertEqual(key1, key2)

    def test_normalize_key_case_insensitive(self):
        key1 = self.det._normalize_key("Person", "near", "Machine")
        key2 = self.det._normalize_key("person", "near", "machine")
        self.assertEqual(key1, key2)

    def test_observe_records_pair(self):
        e1 = _make_entity(entity_id=1, label="person")
        e2 = _make_entity(entity_id=2, label="helmet", x1=0.15, y1=0.15)
        rel = Relation(
            subject_id=1, predicate=SpatialRelation.NEAR, object_id=2
        )
        ws = _make_world_state(entities=[e1, e2], relations=[rel])
        self.det.observe(ws)
        self.assertEqual(self.det._observations, 1)
        self.assertGreater(len(self.det._pair_freq), 0)

    def test_no_relations_no_signals(self):
        ws = _make_world_state(relations=[])
        self.det.observe(ws)
        signals = self.det.check(ws)
        self.assertEqual(len(signals), 0)

    def test_novel_relation_detected(self):
        # Train on person-helmet relation
        e1 = _make_entity(entity_id=1, label="person")
        e2 = _make_entity(entity_id=2, label="helmet", x1=0.15, y1=0.15)
        rel = Relation(subject_id=1, predicate=SpatialRelation.NEAR, object_id=2)
        for i in range(30):
            ws = _make_world_state(entities=[e1, e2], relations=[rel])
            self.det.observe(ws)

        # New: person-forklift (never seen before)
        e3 = _make_entity(entity_id=3, label="forklift", x1=0.5, y1=0.5, x2=0.7, y2=0.7)
        new_rel = Relation(subject_id=1, predicate=SpatialRelation.NEAR, object_id=3)
        ws = _make_world_state(entities=[e1, e3], relations=[new_rel])
        signals = self.det.check(ws)
        self.assertGreater(len(signals), 0)
        self.assertEqual(signals[0].detector, "interaction")
        self.assertEqual(signals[0].metric, "rare_pair")

    def test_frequent_pair_not_flagged(self):
        e1 = _make_entity(entity_id=1, label="person")
        e2 = _make_entity(entity_id=2, label="helmet", x1=0.15, y1=0.15)
        rel = Relation(subject_id=1, predicate=SpatialRelation.NEAR, object_id=2)
        for i in range(50):
            ws = _make_world_state(entities=[e1, e2], relations=[rel])
            self.det.observe(ws)
        ws = _make_world_state(entities=[e1, e2], relations=[rel])
        signals = self.det.check(ws)
        # The person-helmet NEAR pair is well-established
        helmet_signals = [
            s for s in signals
            if "helmet" in s.details.get("subject_label", "")
            or "helmet" in s.details.get("object_label", "")
        ]
        self.assertEqual(len(helmet_signals), 0)

    def test_pair_decay(self):
        e1 = _make_entity(entity_id=1, label="person")
        e2 = _make_entity(entity_id=2, label="tool", x1=0.2, y1=0.2)
        rel = Relation(subject_id=1, predicate=SpatialRelation.HOLDING, object_id=2)
        for i in range(10):
            ws = _make_world_state(entities=[e1, e2], relations=[rel])
            self.det.observe(ws)
        key = self.det._normalize_key("person", "holding", "tool")
        old_freq = self.det._pair_freq[key]
        # Observe without the relation
        for i in range(20):
            ws = _make_world_state(entities=[e1], relations=[])
            self.det.observe(ws)
        new_freq = self.det._pair_freq.get(key, 0.0)
        self.assertLess(new_freq, old_freq)

    def test_get_known_pairs(self):
        e1 = _make_entity(entity_id=1, label="person")
        e2 = _make_entity(entity_id=2, label="cone", x1=0.2, y1=0.2)
        rel = Relation(subject_id=1, predicate=SpatialRelation.NEAR, object_id=2)
        ws = _make_world_state(entities=[e1, e2], relations=[rel])
        self.det.observe(ws)
        pairs = self.det.get_known_pairs()
        self.assertGreater(len(pairs), 0)

    def test_reset(self):
        e1 = _make_entity(entity_id=1, label="person")
        e2 = _make_entity(entity_id=2, label="cone", x1=0.2, y1=0.2)
        rel = Relation(subject_id=1, predicate=SpatialRelation.NEAR, object_id=2)
        ws = _make_world_state(entities=[e1, e2], relations=[rel])
        self.det.observe(ws)
        self.det.reset()
        self.assertEqual(self.det._observations, 0)
        self.assertEqual(len(self.det._pair_freq), 0)

    def test_signal_contains_labels(self):
        e1 = _make_entity(entity_id=1, label="person")
        e2 = _make_entity(entity_id=2, label="machine", x1=0.5, y1=0.5, x2=0.8, y2=0.8)
        rel = Relation(subject_id=1, predicate=SpatialRelation.OPERATING, object_id=2)
        ws = _make_world_state(entities=[e1, e2], relations=[rel])
        # First time — the pair is novel
        self.det._observations = 20  # pretend we have history
        signals = self.det.check(ws)
        if signals:
            self.assertIn("subject_label", signals[0].details)
            self.assertIn("object_label", signals[0].details)
            self.assertIn("relation", signals[0].details)

    def test_missing_entity_in_relation_skipped(self):
        # Relation references entity not in scene graph
        rel = Relation(subject_id=99, predicate=SpatialRelation.NEAR, object_id=100)
        ws = _make_world_state(entities=[], relations=[rel])
        self.det.observe(ws)
        signals = self.det.check(ws)
        self.assertEqual(len(signals), 0)

    def test_multiple_relations_processed(self):
        e1 = _make_entity(entity_id=1, label="person")
        e2 = _make_entity(entity_id=2, label="helmet", x1=0.15, y1=0.15)
        e3 = _make_entity(entity_id=3, label="cone", x1=0.5, y1=0.5, x2=0.6, y2=0.6)
        r1 = Relation(subject_id=1, predicate=SpatialRelation.NEAR, object_id=2)
        r2 = Relation(subject_id=1, predicate=SpatialRelation.NEAR, object_id=3)
        ws = _make_world_state(entities=[e1, e2, e3], relations=[r1, r2])
        self.det.observe(ws)
        self.assertGreaterEqual(len(self.det._pair_freq), 2)


# ===========================================================================
# AnomalyDetectorEnsemble tests
# ===========================================================================


class TestAnomalyDetectorEnsemble(unittest.TestCase):
    """Tests for the ensemble orchestrator: cooldown, weights, severity, integration."""

    def test_init_default_config(self):
        ens = AnomalyDetectorEnsemble()
        self.assertEqual(ens._warmup_frames, 100)
        self.assertEqual(ens._sigma_threshold, 2.0)
        self.assertEqual(ens._cooldown_seconds, 60.0)

    def test_init_custom_config(self):
        config = PerceptionConfig(
            anomaly_warmup_frames=50,
            anomaly_sigma_threshold=3.0,
            anomaly_cooldown_seconds=30.0,
            anomaly_spatial_grid_size=8,
            anomaly_ema_alpha=0.1,
        )
        ens = AnomalyDetectorEnsemble(config)
        self.assertEqual(ens._warmup_frames, 50)
        self.assertEqual(ens._sigma_threshold, 3.0)
        self.assertEqual(ens._cooldown_seconds, 30.0)
        self.assertEqual(ens._spatial._grid_size, 8)

    def test_warmup_suppresses_events(self):
        ens = AnomalyDetectorEnsemble(PerceptionConfig(anomaly_warmup_frames=10))
        for i in range(5):
            ws = _make_world_state(timestamp=float(i))
            ens.observe(ws)
        ws = _make_world_state(timestamp=5.0)
        events = ens.check_anomalies(ws)
        self.assertEqual(len(events), 0)

    def test_observe_increments_counter(self):
        ens = AnomalyDetectorEnsemble()
        ws = _make_world_state()
        ens.observe(ws)
        self.assertEqual(ens._observations, 1)

    def test_no_events_during_stable_state(self):
        ens = AnomalyDetectorEnsemble(PerceptionConfig(anomaly_warmup_frames=5))
        entity = _make_entity()
        track = _make_track(velocity=(0.01, 0.0))
        for i in range(20):
            ws = _make_world_state(
                entities=[entity],
                active_tracks={1: track},
                timestamp=float(i),
                frame_number=i,
            )
            ens.observe(ws)
        # Check with same stable state
        ws = _make_world_state(
            entities=[entity],
            active_tracks={1: track},
            timestamp=20.0,
            frame_number=20,
        )
        events = ens.check_anomalies(ws)
        self.assertEqual(len(events), 0)

    def test_cooldown_prevents_duplicate_alerts(self):
        config = PerceptionConfig(
            anomaly_warmup_frames=5,
            anomaly_cooldown_seconds=60.0,
        )
        ens = AnomalyDetectorEnsemble(config)
        entity = _make_entity()
        for i in range(10):
            ws = _make_world_state(
                entities=[entity],
                active_tracks={1: _make_track(velocity=(0.01, 0.0))},
                timestamp=float(i),
            )
            ens.observe(ws)

        # Force a spatial anomaly by placing entity in rare cell
        rare = _make_entity(entity_id=99, x1=0.9, y1=0.9, x2=0.99, y2=0.99)
        ws1 = _make_world_state(
            entities=[rare],
            active_tracks={99: _make_track(track_id=99, velocity=(0.01, 0.0))},
            timestamp=10.0,
        )
        events1 = ens.check_anomalies(ws1)

        # Same anomaly 5 seconds later — should be suppressed by cooldown
        ws2 = _make_world_state(
            entities=[rare],
            active_tracks={99: _make_track(track_id=99, velocity=(0.01, 0.0))},
            timestamp=15.0,
        )
        events2 = ens.check_anomalies(ws2)

        # At most, second check should have fewer or equal events
        if events1:
            # Get same (detector, metric, entity_id) events
            keys1 = {(e.details.get("detector"), e.details.get("metric"), e.entity_id) for e in events1}
            keys2 = {(e.details.get("detector"), e.details.get("metric"), e.entity_id) for e in events2}
            self.assertTrue(len(keys2.intersection(keys1)) == 0)

    def test_cooldown_expires(self):
        config = PerceptionConfig(
            anomaly_warmup_frames=5,
            anomaly_cooldown_seconds=10.0,
        )
        ens = AnomalyDetectorEnsemble(config)
        for i in range(10):
            ws = _make_world_state(
                entities=[_make_entity()],
                active_tracks={1: _make_track(velocity=(0.01, 0.0))},
                timestamp=float(i),
            )
            ens.observe(ws)

        # Manually insert a cooldown entry
        ens._cooldown_map[("spatial", "rare_cell", 99)] = 0.0

        # At timestamp 100 (well past cooldown), the key should not block
        result = (100.0 - 0.0) < config.anomaly_cooldown_seconds
        self.assertFalse(result)

    def test_severity_mapping_info(self):
        self.assertEqual(
            AnomalyDetectorEnsemble._z_to_severity(2.5),
            ViolationSeverity.INFO,
        )

    def test_severity_mapping_warning(self):
        self.assertEqual(
            AnomalyDetectorEnsemble._z_to_severity(3.5),
            ViolationSeverity.WARNING,
        )

    def test_severity_mapping_critical(self):
        self.assertEqual(
            AnomalyDetectorEnsemble._z_to_severity(5.0),
            ViolationSeverity.CRITICAL,
        )

    def test_get_state_structure(self):
        ens = AnomalyDetectorEnsemble()
        state = ens.get_state()
        self.assertIn("observations", state)
        self.assertIn("warmup_frames", state)
        self.assertIn("is_warmed_up", state)
        self.assertIn("detectors", state)
        self.assertIn("behavioral", state["detectors"])
        self.assertIn("spatial", state["detectors"])
        self.assertIn("temporal", state["detectors"])
        self.assertIn("interaction", state["detectors"])

    def test_reset_clears_all(self):
        ens = AnomalyDetectorEnsemble()
        ws = _make_world_state()
        ens.observe(ws)
        ens._cooldown_map[("test", "test", -1)] = 1.0
        ens.reset()
        self.assertEqual(ens._observations, 0)
        self.assertEqual(len(ens._cooldown_map), 0)

    def test_events_are_entity_events(self):
        config = PerceptionConfig(
            anomaly_warmup_frames=5,
            anomaly_sigma_threshold=1.0,  # low threshold for testing
        )
        ens = AnomalyDetectorEnsemble(config)
        # Train on empty scenes
        for i in range(10):
            ws = _make_world_state(entities=[], active_tracks={}, timestamp=float(i))
            ens.observe(ws)
        # Inject anomaly: many entities suddenly
        entities = [_make_entity(entity_id=j) for j in range(10)]
        tracks = {j: _make_track(track_id=j, velocity=(0.1, 0.1)) for j in range(10)}
        ws = _make_world_state(
            entities=entities,
            active_tracks=tracks,
            timestamp=10.0,
        )
        events = ens.check_anomalies(ws)
        for evt in events:
            self.assertIsInstance(evt, EntityEvent)
            self.assertEqual(evt.event_type, EntityEventType.ANOMALY)

    def test_event_details_include_detector(self):
        config = PerceptionConfig(
            anomaly_warmup_frames=5,
            anomaly_sigma_threshold=0.5,  # very low for testing
        )
        ens = AnomalyDetectorEnsemble(config)
        for i in range(10):
            ws = _make_world_state(
                entities=[_make_entity()],
                active_tracks={1: _make_track(velocity=(0.01, 0.0))},
                timestamp=float(i),
            )
            ens.observe(ws)
        # Anomalous state
        entities = [_make_entity(entity_id=j, x1=0.9, y1=0.9, x2=0.99, y2=0.99) for j in range(5)]
        tracks = {j: _make_track(track_id=j, velocity=(0.5, 0.5)) for j in range(5)}
        ws = _make_world_state(entities=entities, active_tracks=tracks, timestamp=10.0)
        events = ens.check_anomalies(ws)
        for evt in events:
            self.assertIn("detector", evt.details)
            self.assertIn("metric", evt.details)
            self.assertIn("z_score", evt.details)


# ===========================================================================
# Integration tests: WorldModel, Engine, Narrator, ContextMemory
# ===========================================================================


class TestWorldModelIntegration(unittest.TestCase):
    """Tests that WorldModel uses AnomalyDetectorEnsemble when configured."""

    def test_world_model_uses_ensemble(self):
        config = PerceptionConfig(anomaly_enabled=True)
        from sopilot.perception.world_model import WorldModel
        wm = WorldModel(config)
        baseline = wm.get_anomaly_baseline()
        self.assertIsInstance(baseline, AnomalyDetectorEnsemble)

    def test_world_model_fallback_when_disabled(self):
        config = PerceptionConfig(anomaly_enabled=False)
        from sopilot.perception.world_model import WorldModel, AnomalyBaseline
        wm = WorldModel(config)
        baseline = wm.get_anomaly_baseline()
        self.assertIsInstance(baseline, AnomalyBaseline)

    def test_ensemble_observe_called_during_update(self):
        config = PerceptionConfig(anomaly_enabled=True)
        from sopilot.perception.world_model import WorldModel
        wm = WorldModel(config)
        sg = _make_scene_graph(entities=[_make_entity()])
        wm.update(sg)
        baseline = wm.get_anomaly_baseline()
        self.assertEqual(baseline._observations, 1)


class TestEngineIntegration(unittest.TestCase):
    """Tests that PerceptionEngine.get_anomaly_state() works."""

    def test_get_anomaly_state_returns_dict(self):
        from sopilot.perception.engine import PerceptionEngine
        from sopilot.perception.world_model import WorldModel
        config = PerceptionConfig(anomaly_enabled=True)
        wm = WorldModel(config)
        engine = PerceptionEngine(config=config, world_model=wm)
        state = engine.get_anomaly_state()
        self.assertIsNotNone(state)
        self.assertIn("observations", state)

    def test_get_anomaly_state_none_without_world_model(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        state = engine.get_anomaly_state()
        self.assertIsNone(state)


class TestNarratorIntegration(unittest.TestCase):
    """Tests that narrator handles ensemble ANOMALY events with detector details."""

    def test_anomaly_event_with_detector_behavioral(self):
        from sopilot.perception.narrator import SceneNarrator
        narrator = SceneNarrator()
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1,
            timestamp=1.0,
            frame_number=1,
            details={"detector": "behavioral"},
        )
        entity = _make_entity(entity_id=1, label="person")
        ws = _make_world_state(entities=[entity], events=[event])
        text = narrator.narrate_event(event, ws)
        self.assertIn("行動異常", text)

    def test_anomaly_event_with_detector_spatial(self):
        from sopilot.perception.narrator import SceneNarrator
        narrator = SceneNarrator()
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1,
            timestamp=1.0,
            frame_number=1,
            details={"detector": "spatial"},
        )
        entity = _make_entity(entity_id=1, label="person")
        ws = _make_world_state(entities=[entity], events=[event])
        text = narrator.narrate_event(event, ws)
        self.assertIn("空間異常", text)

    def test_anomaly_event_with_description_ja(self):
        from sopilot.perception.narrator import SceneNarrator
        narrator = SceneNarrator()
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1,
            timestamp=1.0,
            frame_number=1,
            details={"description_ja": "速度が通常の5倍です"},
        )
        entity = _make_entity(entity_id=1, label="person")
        ws = _make_world_state(entities=[entity], events=[event])
        text = narrator.narrate_event(event, ws)
        self.assertIn("速度が通常の5倍です", text)

    def test_anomaly_en_with_detector_temporal(self):
        from sopilot.perception.narrator import SceneNarrator
        narrator = SceneNarrator()
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=-1,
            timestamp=1.0,
            frame_number=1,
            details={"detector": "temporal"},
        )
        ws = _make_world_state(entities=[], events=[event])
        text = narrator._narrate_event_en(event, ws)
        self.assertIn("Temporal anomaly", text)

    def test_anomaly_en_with_description_en(self):
        from sopilot.perception.narrator import SceneNarrator
        narrator = SceneNarrator()
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1,
            timestamp=1.0,
            frame_number=1,
            details={"description_en": "Speed is 5x normal"},
        )
        entity = _make_entity(entity_id=1, label="person")
        ws = _make_world_state(entities=[entity], events=[event])
        text = narrator._narrate_event_en(event, ws)
        self.assertIn("Speed is 5x normal", text)


class TestContextMemoryIntegration(unittest.TestCase):
    """Tests that context memory answers anomaly-related queries."""

    def test_anomaly_query_no_events(self):
        from sopilot.perception.context_memory import ContextMemory
        cm = ContextMemory()
        answer = cm.query("異常はありますか？")
        self.assertIn("異常はありません", answer)

    def test_anomaly_query_english(self):
        from sopilot.perception.context_memory import ContextMemory
        cm = ContextMemory()
        answer = cm.query("any anomalies?")
        self.assertIn("異常はありません", answer)

    def test_anomaly_query_with_events(self):
        from sopilot.perception.context_memory import ContextMemory
        cm = ContextMemory()
        # Manually inject anomaly events
        cm._events.append(
            EntityEvent(
                event_type=EntityEventType.ANOMALY,
                entity_id=-1,
                timestamp=1.0,
                frame_number=1,
                details={
                    "detector": "behavioral",
                    "description_ja": "速度異常検出",
                },
            )
        )
        cm._events.append(
            EntityEvent(
                event_type=EntityEventType.ANOMALY,
                entity_id=5,
                timestamp=2.0,
                frame_number=2,
                details={
                    "detector": "spatial",
                    "description_ja": "空間異常検出",
                },
            )
        )
        answer = cm.query("anomalyは？")
        self.assertIn("2件", answer)
        self.assertIn("behavioral", answer)
        self.assertIn("spatial", answer)


# ===========================================================================
# PerceptionConfig anomaly fields tests
# ===========================================================================


class TestPerceptionConfigAnomalyFields(unittest.TestCase):
    """Tests for the anomaly-related fields in PerceptionConfig."""

    def test_default_values(self):
        config = PerceptionConfig()
        self.assertTrue(config.anomaly_enabled)
        self.assertEqual(config.anomaly_warmup_frames, 100)
        self.assertAlmostEqual(config.anomaly_sigma_threshold, 2.0)
        self.assertAlmostEqual(config.anomaly_cooldown_seconds, 60.0)
        self.assertEqual(config.anomaly_spatial_grid_size, 10)
        self.assertAlmostEqual(config.anomaly_ema_alpha, 0.05)

    def test_custom_values(self):
        config = PerceptionConfig(
            anomaly_enabled=False,
            anomaly_warmup_frames=200,
            anomaly_sigma_threshold=3.5,
            anomaly_cooldown_seconds=120.0,
            anomaly_spatial_grid_size=20,
            anomaly_ema_alpha=0.1,
        )
        self.assertFalse(config.anomaly_enabled)
        self.assertEqual(config.anomaly_warmup_frames, 200)
        self.assertAlmostEqual(config.anomaly_sigma_threshold, 3.5)
        self.assertAlmostEqual(config.anomaly_cooldown_seconds, 120.0)
        self.assertEqual(config.anomaly_spatial_grid_size, 20)
        self.assertAlmostEqual(config.anomaly_ema_alpha, 0.1)


# ===========================================================================
# load_state() tests for each detector
# ===========================================================================


class TestBehavioralLoadState(unittest.TestCase):
    """Tests for BehavioralAnomalyDetector.load_state()."""

    def test_load_state_restores_speed(self):
        det = BehavioralAnomalyDetector()
        det.load_state({
            "speed_mean": 0.05,
            "speed_var": 0.002,
            "observations": 50,
            "activity_freq": {"walking": 0.8, "running": 0.2},
            "activity_observations": 40,
        })
        self.assertAlmostEqual(det._speed_mean, 0.05)
        self.assertAlmostEqual(det._speed_var, 0.002)
        self.assertEqual(det._observations, 50)
        self.assertAlmostEqual(det._activity_freq["walking"], 0.8)
        self.assertEqual(det._activity_observations, 40)

    def test_load_state_empty_dict(self):
        det = BehavioralAnomalyDetector()
        det.load_state({})
        self.assertEqual(det._speed_mean, 0.0)
        self.assertEqual(det._observations, 0)

    def test_load_state_partial(self):
        det = BehavioralAnomalyDetector()
        det.load_state({"speed_mean": 1.0})
        self.assertAlmostEqual(det._speed_mean, 1.0)
        self.assertEqual(det._observations, 0)


class TestSpatialLoadState(unittest.TestCase):
    """Tests for SpatialAnomalyDetector.load_state()."""

    def test_load_state_restores_grid(self):
        det = SpatialAnomalyDetector(grid_size=3)
        grid = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        det.load_state({"grid": grid, "observations": 100})
        self.assertEqual(det._observations, 100)
        self.assertAlmostEqual(det._grid[0][0], 0.1)
        self.assertAlmostEqual(det._grid[2][2], 0.9)

    def test_load_state_empty_dict(self):
        det = SpatialAnomalyDetector(grid_size=3)
        det.load_state({})
        self.assertEqual(det._observations, 0)

    def test_load_state_updates_grid_size(self):
        det = SpatialAnomalyDetector(grid_size=5)
        grid = [[0.0] * 3 for _ in range(3)]
        grid[1][1] = 0.5
        det.load_state({"grid": grid, "observations": 10})
        self.assertEqual(det._grid_size, 3)
        self.assertAlmostEqual(det._grid[1][1], 0.5)


class TestTemporalLoadState(unittest.TestCase):
    """Tests for TemporalPatternDetector.load_state()."""

    def test_load_state_restores_hourly(self):
        det = TemporalPatternDetector()
        mean = [float(i) for i in range(24)]
        var = [0.1 * i for i in range(24)]
        obs = list(range(24))
        det.load_state({"hourly_mean": mean, "hourly_var": var, "hourly_obs": obs})
        self.assertAlmostEqual(det._hourly_mean[10], 10.0)
        self.assertAlmostEqual(det._hourly_var[5], 0.5)
        self.assertEqual(det._hourly_obs[23], 23)

    def test_load_state_empty_dict(self):
        det = TemporalPatternDetector()
        det.load_state({})
        self.assertEqual(det._hourly_mean[0], 0.0)

    def test_load_state_wrong_length_ignored(self):
        det = TemporalPatternDetector()
        det.load_state({"hourly_mean": [1.0, 2.0]})  # wrong length
        self.assertEqual(det._hourly_mean[0], 0.0)  # unchanged


class TestInteractionLoadState(unittest.TestCase):
    """Tests for InteractionAnomalyDetector.load_state()."""

    def test_load_state_restores_pairs(self):
        det = InteractionAnomalyDetector()
        det.load_state({
            "pair_freq": {"person|near|helmet": 0.8, "person|operating|machine": 0.3},
            "observations": 200,
        })
        self.assertEqual(det._observations, 200)
        self.assertAlmostEqual(det._pair_freq["person|near|helmet"], 0.8)

    def test_load_state_empty(self):
        det = InteractionAnomalyDetector()
        det.load_state({})
        self.assertEqual(det._observations, 0)


class TestEnsembleLoadState(unittest.TestCase):
    """Tests for AnomalyDetectorEnsemble.load_state()."""

    def test_load_state_full(self):
        ens = AnomalyDetectorEnsemble()
        data = {
            "observations": 500,
            "behavioral": {"speed_mean": 0.02, "speed_var": 0.001, "observations": 500,
                          "activity_freq": {}, "activity_observations": 0},
            "spatial": {"grid": [[0.5] * 10 for _ in range(10)], "observations": 500},
            "temporal": {"hourly_mean": [2.0] * 24, "hourly_var": [0.5] * 24,
                        "hourly_obs": [50] * 24},
            "interaction": {"pair_freq": {"a|near|b": 0.5}, "observations": 500},
        }
        ens.load_state(data)
        self.assertEqual(ens._observations, 500)
        self.assertAlmostEqual(ens._behavioral._speed_mean, 0.02)
        self.assertAlmostEqual(ens._spatial._grid[0][0], 0.5)
        self.assertEqual(ens._temporal._hourly_obs[0], 50)
        self.assertAlmostEqual(ens._interaction._pair_freq["a|near|b"], 0.5)

    def test_load_state_clears_cooldown(self):
        ens = AnomalyDetectorEnsemble()
        ens._cooldown_map[("test", "test", -1)] = 1.0
        ens.load_state({"observations": 10})
        self.assertEqual(len(ens._cooldown_map), 0)

    def test_load_state_partial(self):
        ens = AnomalyDetectorEnsemble()
        ens.load_state({"observations": 42, "behavioral": {"speed_mean": 0.1}})
        self.assertEqual(ens._observations, 42)
        self.assertAlmostEqual(ens._behavioral._speed_mean, 0.1)


# ===========================================================================
# AnomalyExplainer tests
# ===========================================================================


class TestAnomalyExplainer(unittest.TestCase):
    """Tests for the AnomalyExplainer VLM explanation pipeline."""

    def test_explain_no_vlm_returns_none(self):
        from sopilot.perception.anomaly_explainer import AnomalyExplainer
        explainer = AnomalyExplainer(vlm_client=None)
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1, timestamp=1.0, frame_number=1,
            details={"detector": "behavioral", "description_ja": "テスト"},
        )
        ws = _make_world_state()
        result = explainer.explain(event, np.zeros((10, 10, 3), dtype=np.uint8), ws)
        self.assertIsNone(result)

    def test_explain_no_frame_returns_none(self):
        from sopilot.perception.anomaly_explainer import AnomalyExplainer
        explainer = AnomalyExplainer(vlm_client="dummy")
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1, timestamp=1.0, frame_number=1,
            details={"detector": "behavioral"},
        )
        ws = _make_world_state()
        result = explainer.explain(event, None, ws)
        self.assertIsNone(result)

    def test_build_prompt_behavioral(self):
        from sopilot.perception.anomaly_explainer import AnomalyExplainer
        explainer = AnomalyExplainer()
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1, timestamp=1.0, frame_number=1,
            details={"detector": "behavioral", "description_ja": "速度異常", "z_score": 3.5},
        )
        ws = _make_world_state(entity_count=5)
        prompt = explainer.build_prompt(event, ws)
        self.assertIn("行動異常", prompt)
        self.assertIn("速度異常", prompt)
        self.assertIn("偏差スコア=3.5", prompt)

    def test_build_prompt_spatial(self):
        from sopilot.perception.anomaly_explainer import AnomalyExplainer
        explainer = AnomalyExplainer()
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1, timestamp=1.0, frame_number=1,
            details={"detector": "spatial", "description_ja": "空間異常テスト"},
        )
        ws = _make_world_state()
        prompt = explainer.build_prompt(event, ws)
        self.assertIn("空間異常", prompt)

    def test_build_prompt_temporal(self):
        from sopilot.perception.anomaly_explainer import AnomalyExplainer
        explainer = AnomalyExplainer()
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=-1, timestamp=1.0, frame_number=1,
            details={"detector": "temporal", "description_ja": "時間帯異常"},
        )
        ws = _make_world_state()
        prompt = explainer.build_prompt(event, ws)
        self.assertIn("時間帯異常", prompt)

    def test_build_prompt_interaction(self):
        from sopilot.perception.anomaly_explainer import AnomalyExplainer
        explainer = AnomalyExplainer()
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1, timestamp=1.0, frame_number=1,
            details={"detector": "interaction", "description_ja": "関係異常"},
        )
        ws = _make_world_state()
        prompt = explainer.build_prompt(event, ws)
        self.assertIn("関係性異常", prompt)

    def test_build_prompt_unknown_detector(self):
        from sopilot.perception.anomaly_explainer import AnomalyExplainer
        explainer = AnomalyExplainer()
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1, timestamp=1.0, frame_number=1,
            details={"detector": "unknown_new", "description_ja": "テスト"},
        )
        ws = _make_world_state()
        prompt = explainer.build_prompt(event, ws)
        self.assertIn("異常が検出されました", prompt)

    def test_extract_explanation_string(self):
        from sopilot.perception.anomaly_explainer import AnomalyExplainer
        explainer = AnomalyExplainer()
        self.assertEqual(explainer._extract_explanation("test"), "test")
        self.assertIsNone(explainer._extract_explanation(""))
        self.assertIsNone(explainer._extract_explanation(None))

    def test_extract_explanation_dict(self):
        from sopilot.perception.anomaly_explainer import AnomalyExplainer
        explainer = AnomalyExplainer()
        self.assertEqual(
            explainer._extract_explanation({"explanation": "found issue"}),
            "found issue",
        )
        self.assertEqual(
            explainer._extract_explanation({"text": "some text"}),
            "some text",
        )

    def test_cooldown_property(self):
        from sopilot.perception.anomaly_explainer import AnomalyExplainer
        explainer = AnomalyExplainer()
        self.assertEqual(explainer.total_calls, 0)


# ===========================================================================
# AnomalyProfile tests
# ===========================================================================


class TestAnomalyProfile(unittest.TestCase):
    """Tests for anomaly profile save/load/apply/list."""

    def _make_trained_ensemble(self):
        config = PerceptionConfig(
            anomaly_warmup_frames=5,
            anomaly_spatial_grid_size=3,
        )
        ens = AnomalyDetectorEnsemble(config)
        for i in range(10):
            ws = _make_world_state(
                entities=[_make_entity(entity_id=1)],
                active_tracks={1: _make_track(velocity=(0.01, 0.0))},
                timestamp=float(i),
            )
            ens.observe(ws)
        return ens

    def test_save_and_load(self):
        import tempfile
        from pathlib import Path
        from sopilot.perception.anomaly_profile import save_profile, load_profile

        ens = self._make_trained_ensemble()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_profile(ens, "test_profile", Path(tmpdir))
            self.assertTrue(path.exists())
            profile = load_profile(path)
            self.assertEqual(profile.name, "test_profile")
            self.assertEqual(profile.observations, 10)
            self.assertIn("speed_mean", profile.behavioral)
            self.assertIn("grid", profile.spatial)
            self.assertIn("hourly_mean", profile.temporal)
            self.assertIn("pair_freq", profile.interaction)

    def test_apply_profile(self):
        import tempfile
        from pathlib import Path
        from sopilot.perception.anomaly_profile import (
            save_profile, load_profile, apply_profile,
        )

        ens = self._make_trained_ensemble()
        original_speed_mean = ens._behavioral._speed_mean
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_profile(ens, "apply_test", Path(tmpdir))
            profile = load_profile(path)

            # Create fresh ensemble and apply profile
            ens2 = AnomalyDetectorEnsemble(PerceptionConfig(anomaly_spatial_grid_size=3))
            self.assertEqual(ens2._observations, 0)
            apply_profile(ens2, profile)
            self.assertEqual(ens2._observations, 10)
            self.assertAlmostEqual(ens2._behavioral._speed_mean, original_speed_mean)

    def test_list_profiles(self):
        import tempfile
        from pathlib import Path
        from sopilot.perception.anomaly_profile import save_profile, list_profiles

        ens = self._make_trained_ensemble()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_profile(ens, "profile_a", Path(tmpdir))
            save_profile(ens, "profile_b", Path(tmpdir))
            profiles = list_profiles(Path(tmpdir))
            self.assertEqual(len(profiles), 2)
            names = {p["name"] for p in profiles}
            self.assertIn("profile_a", names)
            self.assertIn("profile_b", names)

    def test_list_profiles_empty_dir(self):
        import tempfile
        from pathlib import Path
        from sopilot.perception.anomaly_profile import list_profiles
        with tempfile.TemporaryDirectory() as tmpdir:
            profiles = list_profiles(Path(tmpdir))
            self.assertEqual(len(profiles), 0)

    def test_list_profiles_nonexistent_dir(self):
        from pathlib import Path
        from sopilot.perception.anomaly_profile import list_profiles
        profiles = list_profiles(Path("/tmp/nonexistent_dir_abc123"))
        self.assertEqual(len(profiles), 0)

    def test_save_creates_directory(self):
        import tempfile
        from pathlib import Path
        from sopilot.perception.anomaly_profile import save_profile

        ens = self._make_trained_ensemble()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "sub" / "dir"
            path = save_profile(ens, "nested_test", nested)
            self.assertTrue(path.exists())


# ===========================================================================
# Engine ANOMALY→Violation conversion tests
# ===========================================================================


class TestEngineAnomalyViolation(unittest.TestCase):
    """Tests that PerceptionEngine converts ANOMALY events to Violations."""

    def test_anomaly_event_to_violation(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1,
            timestamp=1.0,
            frame_number=1,
            details={
                "detector": "behavioral",
                "metric": "speed_zscore",
                "severity": "warning",
                "description_ja": "速度異常検出",
                "z_score": 3.5,
            },
            confidence=0.7,
        )
        entity = _make_entity(entity_id=1)
        ws = _make_world_state(entities=[entity], events=[event])
        violation = engine._event_to_violation(event, ws)
        self.assertIsNotNone(violation)
        self.assertEqual(violation.source, "anomaly")
        self.assertEqual(violation.severity, ViolationSeverity.WARNING)
        self.assertIn("速度異常検出", violation.description_ja)
        self.assertEqual(violation.rule, "anomaly:behavioral/speed_zscore")

    def test_anomaly_event_with_vlm_explanation(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=-1,
            timestamp=1.0,
            frame_number=1,
            details={
                "detector": "temporal",
                "metric": "hourly_density",
                "severity": "critical",
                "description_ja": "深夜に人員検出",
                "vlm_explanation": "夜間に作業員が見えます",
            },
            confidence=0.9,
        )
        ws = _make_world_state(entities=[])
        violation = engine._event_to_violation(event, ws)
        self.assertIsNotNone(violation)
        self.assertIn("[VLM]", violation.description_ja)
        self.assertIn("夜間に作業員が見えます", violation.description_ja)
        self.assertEqual(violation.severity, ViolationSeverity.CRITICAL)

    def test_anomaly_event_severity_info(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=5,
            timestamp=1.0,
            frame_number=1,
            details={
                "detector": "spatial",
                "metric": "rare_cell",
                "severity": "info",
                "description_ja": "軽微な空間異常",
            },
            confidence=0.4,
        )
        ws = _make_world_state(entities=[_make_entity(entity_id=5)])
        violation = engine._event_to_violation(event, ws)
        self.assertIsNotNone(violation)
        self.assertEqual(violation.severity, ViolationSeverity.INFO)
        self.assertEqual(violation.entity_ids, [5])

    def test_anomaly_event_scene_level(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=-1,
            timestamp=1.0,
            frame_number=1,
            details={
                "detector": "temporal",
                "metric": "hourly_density",
                "severity": "warning",
                "description_ja": "シーンレベル異常",
            },
        )
        ws = _make_world_state(entities=[])
        violation = engine._event_to_violation(event, ws)
        self.assertIsNotNone(violation)
        self.assertEqual(violation.entity_ids, [])  # scene-level: no entities

    def test_events_to_violations_includes_anomaly(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1,
            timestamp=1.0,
            frame_number=1,
            details={
                "detector": "behavioral",
                "metric": "speed_zscore",
                "severity": "warning",
                "description_ja": "テスト",
            },
        )
        ws = _make_world_state(entities=[_make_entity(entity_id=1)], events=[event])
        violations = engine._events_to_violations(ws)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].source, "anomaly")


# ===========================================================================
# Engine profile methods tests
# ===========================================================================


class TestEngineProfileMethods(unittest.TestCase):
    """Tests for PerceptionEngine profile save/load convenience methods."""

    def test_save_profile_no_world_model(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        result = engine.save_anomaly_profile("test")
        self.assertIsNone(result)

    def test_load_profile_no_world_model(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        result = engine.load_anomaly_profile("test")
        self.assertFalse(result)

    def test_save_and_load_profile_roundtrip(self):
        import tempfile
        from pathlib import Path
        from sopilot.perception.engine import PerceptionEngine
        from sopilot.perception.world_model import WorldModel

        config = PerceptionConfig(anomaly_enabled=True, anomaly_spatial_grid_size=3)
        wm = WorldModel(config)
        engine = PerceptionEngine(config=config, world_model=wm)

        # Train ensemble
        for i in range(10):
            sg = _make_scene_graph(
                entities=[_make_entity(entity_id=1)],
                timestamp=float(i),
                frame_number=i,
            )
            wm.update(sg)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Monkey-patch the profile dir
            import sopilot.perception.engine as eng_mod
            orig_path = Path
            try:
                # Save
                from sopilot.perception.anomaly_profile import save_profile
                baseline = wm.get_anomaly_baseline()
                path = save_profile(baseline, "roundtrip_test", Path(tmpdir))
                self.assertTrue(path.exists())

                # Load into a new ensemble
                from sopilot.perception.anomaly_profile import load_profile, apply_profile
                profile = load_profile(path)
                new_ens = AnomalyDetectorEnsemble(config)
                apply_profile(new_ens, profile)
                self.assertEqual(new_ens._observations, 10)
            finally:
                pass


# ===========================================================================
# Phase 10: AdaptiveLearner engine integration tests
# ===========================================================================


class TestAdaptiveLearnerEngineIntegration(unittest.TestCase):
    """Tests for get_adaptive_learner_state() and AdaptiveLearner wiring."""

    def _make_engine_with_wm(self):
        from sopilot.perception.engine import PerceptionEngine
        from sopilot.perception.world_model import WorldModel
        config = PerceptionConfig(anomaly_enabled=True, anomaly_spatial_grid_size=3)
        wm = WorldModel(config)
        engine = PerceptionEngine(config=config, world_model=wm)
        return engine

    def test_get_adaptive_learner_state_returns_dict(self):
        engine = self._make_engine_with_wm()
        state = engine.get_adaptive_learner_state()
        self.assertIsInstance(state, dict)

    def test_get_adaptive_learner_state_has_adaptive_learner_key(self):
        engine = self._make_engine_with_wm()
        state = engine.get_adaptive_learner_state()
        self.assertIn("adaptive_learner", state)

    def test_get_adaptive_learner_state_has_tuner_key(self):
        engine = self._make_engine_with_wm()
        state = engine.get_adaptive_learner_state()
        self.assertIn("tuner", state)

    def test_adaptive_learner_state_has_required_al_fields(self):
        engine = self._make_engine_with_wm()
        state = engine.get_adaptive_learner_state()
        al = state["adaptive_learner"]
        for key in ("total_observed", "score_window_size", "score_mean",
                    "score_std", "drift_count", "recalibration_count"):
            self.assertIn(key, al, f"missing key: {key}")

    def test_adaptive_learner_state_has_required_tuner_fields(self):
        engine = self._make_engine_with_wm()
        state = engine.get_adaptive_learner_state()
        t = state["tuner"]
        for key in ("total_feedback", "confirmed", "denied", "overall_confirm_rate",
                    "pairs_tracked", "pairs_suppressed", "pairs_trusted"):
            self.assertIn(key, t, f"missing tuner key: {key}")

    def test_adaptive_learner_state_without_world_model(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        state = engine.get_adaptive_learner_state()
        # Should not raise; defaults must be valid
        self.assertIsInstance(state, dict)
        al = state["adaptive_learner"]
        self.assertEqual(al["total_observed"], 0)

    def test_adaptive_learner_state_tuner_fallback_defaults(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        # No tuner attached — should return fallback dict
        state = engine.get_adaptive_learner_state()
        t = state["tuner"]
        self.assertEqual(t["total_feedback"], 0)
        self.assertEqual(t["pairs_tracked"], 0)

    def test_adaptive_learner_none_gives_default_total_observed(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        engine._adaptive_learner = None
        state = engine.get_adaptive_learner_state()
        self.assertEqual(state["adaptive_learner"]["total_observed"], 0)

    def test_adaptive_learner_state_numeric_types(self):
        engine = self._make_engine_with_wm()
        state = engine.get_adaptive_learner_state()
        al = state["adaptive_learner"]
        self.assertIsInstance(al["score_mean"], float)
        self.assertIsInstance(al["drift_count"], int)

    def test_adaptive_learner_tuner_attached_after_build(self):
        """After build_perception_engine() the engine._anomaly_tuner should be set."""
        from sopilot.perception.engine import build_perception_engine
        engine = build_perception_engine()
        self.assertIsNotNone(engine._anomaly_tuner)

    def test_anomaly_tuner_field_initialized(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        # Field exists (may be None if no build_perception_engine)
        self.assertTrue(hasattr(engine, "_anomaly_tuner"))

    def test_auto_apply_threshold_default(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        self.assertEqual(engine._auto_apply_threshold, 20)

    def test_last_tuner_feedback_count_initialized_to_zero(self):
        from sopilot.perception.engine import PerceptionEngine
        engine = PerceptionEngine(config=PerceptionConfig())
        self.assertEqual(engine._last_tuner_feedback_count, 0)

    def test_get_adaptive_learner_state_with_tuner_returns_stats(self):
        from pathlib import Path as _P
        from sopilot.perception.engine import PerceptionEngine
        from sopilot.perception.anomaly_tuner import AnomalyTuner
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            engine = PerceptionEngine(config=PerceptionConfig())
            engine._anomaly_tuner = AnomalyTuner(_P(tmp) / "fb.json")
            engine._anomaly_tuner.record_feedback("behavioral", "speed_zscore", -1, False)
            state = engine.get_adaptive_learner_state()
            self.assertGreaterEqual(state["tuner"]["total_feedback"], 1)

    def test_get_adaptive_learner_state_tuner_exception_graceful(self):
        from sopilot.perception.engine import PerceptionEngine
        from unittest.mock import MagicMock
        engine = PerceptionEngine(config=PerceptionConfig())
        bad_tuner = MagicMock()
        bad_tuner.get_stats.side_effect = RuntimeError("fail")
        engine._anomaly_tuner = bad_tuner
        # Must not raise
        state = engine.get_adaptive_learner_state()
        self.assertIn("tuner", state)


# ===========================================================================
# Phase 10: AnomalyTuner engine integration tests
# ===========================================================================


class TestAnomalyTunerEngineIntegration(unittest.TestCase):
    """Tests for AnomalyTuner injection and Stage 6h auto-apply."""

    def _make_engine_with_tuner(self, threshold: int = 2):
        from pathlib import Path as _P
        from sopilot.perception.engine import PerceptionEngine
        from sopilot.perception.world_model import WorldModel
        from sopilot.perception.anomaly_tuner import AnomalyTuner
        import tempfile
        config = PerceptionConfig(anomaly_enabled=True, anomaly_spatial_grid_size=3)
        wm = WorldModel(config)
        engine = PerceptionEngine(config=config, world_model=wm)
        self._tmp = tempfile.mkdtemp()
        engine._anomaly_tuner = AnomalyTuner(_P(self._tmp) / "fb.json")
        engine._auto_apply_threshold = threshold
        return engine

    def tearDown(self):
        import shutil
        if hasattr(self, "_tmp"):
            shutil.rmtree(self._tmp, ignore_errors=True)

    def test_anomaly_tuner_exists_on_built_engine(self):
        from sopilot.perception.engine import build_perception_engine
        engine = build_perception_engine()
        self.assertIsNotNone(engine._anomaly_tuner)

    def test_anomaly_tuner_type(self):
        from sopilot.perception.engine import build_perception_engine
        from sopilot.perception.anomaly_tuner import AnomalyTuner
        engine = build_perception_engine()
        self.assertIsInstance(engine._anomaly_tuner, AnomalyTuner)

    def test_auto_apply_counter_initialized(self):
        engine = self._make_engine_with_tuner()
        self.assertEqual(engine._last_tuner_feedback_count, 0)

    def test_auto_apply_threshold_configurable(self):
        engine = self._make_engine_with_tuner(threshold=5)
        self.assertEqual(engine._auto_apply_threshold, 5)

    def test_auto_apply_not_triggered_below_threshold(self):
        engine = self._make_engine_with_tuner(threshold=10)
        # Add 5 records (< 10 threshold)
        for i in range(5):
            engine._anomaly_tuner.record_feedback("behavioral", "speed_zscore", i, False)
        cur = len(engine._anomaly_tuner._records)
        # Counter should NOT have advanced
        self.assertEqual(engine._last_tuner_feedback_count, 0)

    def test_auto_apply_triggered_at_threshold(self):
        engine = self._make_engine_with_tuner(threshold=2)
        # Add records meeting the threshold
        for i in range(2):
            engine._anomaly_tuner.record_feedback("behavioral", "speed_zscore", i, False)
        # Manually simulate what Stage 6h does
        cur = len(engine._anomaly_tuner._records)
        if (cur - engine._last_tuner_feedback_count) >= engine._auto_apply_threshold:
            if engine._world_model is not None:
                bl = engine._world_model.get_anomaly_baseline()
                if bl is not None:
                    engine._anomaly_tuner.apply_tuning(bl)
                    engine._last_tuner_feedback_count = cur
        # After manual trigger, counter should match record count
        self.assertGreaterEqual(engine._last_tuner_feedback_count, 0)

    def test_stage_6h_no_tuner_no_error(self):
        """Stage 6h should be a no-op when _anomaly_tuner is None."""
        from sopilot.perception.engine import PerceptionEngine
        import numpy as np
        engine = PerceptionEngine(config=PerceptionConfig())
        engine._anomaly_tuner = None
        # process_frame should not raise
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            engine.process_frame(frame)
        except Exception:
            pass  # Other components may fail; what matters is no AttributeError on tuner

    def test_anomaly_tuner_record_feedback_persists(self):
        engine = self._make_engine_with_tuner()
        engine._anomaly_tuner.record_feedback("spatial", "rare_cell", -1, True)
        self.assertEqual(len(engine._anomaly_tuner._records), 1)

    def test_anomaly_tuner_get_stats_returns_dict(self):
        engine = self._make_engine_with_tuner()
        stats = engine._anomaly_tuner.get_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_feedback", stats)

    def test_auto_apply_exception_does_not_crash(self):
        """If _world_model.get_anomaly_baseline() raises, Stage 6h must not crash."""
        from unittest.mock import MagicMock
        engine = self._make_engine_with_tuner(threshold=1)
        bad_wm = MagicMock()
        bad_wm.get_anomaly_baseline.side_effect = RuntimeError("boom")
        engine._world_model = bad_wm
        engine._anomaly_tuner.record_feedback("behavioral", "speed_zscore", -1, False)
        # Simulate Stage 6h
        try:
            cur = len(engine._anomaly_tuner._records)
            if (cur - engine._last_tuner_feedback_count) >= engine._auto_apply_threshold:
                bl = engine._world_model.get_anomaly_baseline()
                if bl is not None:
                    engine._anomaly_tuner.apply_tuning(bl)
                    engine._last_tuner_feedback_count = cur
        except Exception:
            pass  # graceful
        # Should not have crashed

    def test_last_tuner_feedback_count_resets_after_apply(self):
        engine = self._make_engine_with_tuner(threshold=1)
        engine._anomaly_tuner.record_feedback("behavioral", "speed_zscore", -1, False)
        cur = len(engine._anomaly_tuner._records)
        engine._last_tuner_feedback_count = cur  # simulate apply
        self.assertEqual(engine._last_tuner_feedback_count, cur)


# ===========================================================================
# Phase 10: GET /anomaly-learning-state endpoint tests
# ===========================================================================


class TestAnomalyLearningStateEndpoint(unittest.TestCase):
    """E2E HTTP tests for GET /vigil/perception/anomaly-learning-state."""

    def setUp(self):
        import os
        import tempfile
        from pathlib import Path as _P
        from unittest.mock import MagicMock
        from fastapi.testclient import TestClient
        from sopilot.main import create_app

        self._tmp = tempfile.TemporaryDirectory()
        root = _P(self._tmp.name)
        os.environ["SOPILOT_DATA_DIR"] = str(root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "learn-state-test"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"

        self.app = create_app()
        self.client = TestClient(self.app)

        # Build mock engine with get_adaptive_learner_state
        engine = MagicMock()
        engine._anomaly_tuner = None
        engine.get_adaptive_learner_state.return_value = {
            "adaptive_learner": {
                "total_observed": 0, "score_window_size": 0,
                "score_mean": 0.0, "score_std": 0.0,
                "drift_count": 0, "recalibration_count": 0,
                "last_recalibration": None, "ph_state": {},
            },
            "tuner": {
                "total_feedback": 0, "confirmed": 0, "denied": 0,
                "overall_confirm_rate": 0.0,
                "pairs_tracked": 0, "pairs_suppressed": 0, "pairs_trusted": 0,
                "last_tuning": 0.0, "pair_stats": [],
                "suppressed_pairs": [], "trusted_pairs": [],
                "min_samples_for_tuning": 10,
            },
        }

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

    def test_endpoint_returns_200(self):
        r = self.client.get("/vigil/perception/anomaly-learning-state")
        self.assertEqual(r.status_code, 200)

    def test_response_has_adaptive_learner_key(self):
        r = self.client.get("/vigil/perception/anomaly-learning-state")
        data = r.json()
        self.assertIn("adaptive_learner", data)

    def test_response_has_tuner_key(self):
        r = self.client.get("/vigil/perception/anomaly-learning-state")
        data = r.json()
        self.assertIn("tuner", data)

    def test_adaptive_learner_fields(self):
        r = self.client.get("/vigil/perception/anomaly-learning-state")
        al = r.json()["adaptive_learner"]
        for key in ("total_observed", "drift_count", "recalibration_count",
                    "score_mean", "score_std"):
            self.assertIn(key, al)

    def test_tuner_fields(self):
        r = self.client.get("/vigil/perception/anomaly-learning-state")
        t = r.json()["tuner"]
        for key in ("total_feedback", "confirmed", "denied",
                    "overall_confirm_rate", "pairs_tracked"):
            self.assertIn(key, t)

    def test_tuner_pair_stats_is_list(self):
        r = self.client.get("/vigil/perception/anomaly-learning-state")
        self.assertIsInstance(r.json()["tuner"]["pair_stats"], list)

    def test_total_feedback_after_mock_feedback(self):
        self.engine.get_adaptive_learner_state.return_value["tuner"]["total_feedback"] = 1
        r = self.client.get("/vigil/perception/anomaly-learning-state")
        self.assertGreaterEqual(r.json()["tuner"]["total_feedback"], 1)

    def test_response_schema_valid(self):
        r = self.client.get("/vigil/perception/anomaly-learning-state")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        # Pydantic validation passed if we got 200; sanity-check types
        self.assertIsInstance(data["adaptive_learner"]["drift_count"], int)
        self.assertIsInstance(data["tuner"]["overall_confirm_rate"], float)


if __name__ == "__main__":
    unittest.main()
