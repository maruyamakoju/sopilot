"""Tests for sopilot.perception.prediction — trajectory prediction and proactive alerting.

Covers:
    - TrajectoryPredictor: velocity estimation, position prediction, zone entry, collision
    - ProactiveAlertGenerator: alert generation with time thresholds
    - Edge cases: zero velocity, single-frame tracks, no history, no zones
    - Confidence decay behaviour over prediction horizon
"""

from __future__ import annotations

import math
import unittest

from sopilot.perception.prediction import (
    CollisionPrediction,
    PredictedPosition,
    ProactiveAlertGenerator,
    TrajectoryPredictor,
    ZoneEntryPrediction,
)
from sopilot.perception.types import (
    BBox,
    EntityEventType,
    Track,
    TrackState,
    Zone,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_track(
    track_id: int = 1,
    label: str = "person",
    bbox: BBox | None = None,
    velocity: tuple[float, float] = (0.0, 0.0),
    state: TrackState = TrackState.ACTIVE,
    confidence: float = 0.9,
    history: list[BBox] | None = None,
    hits: int = 5,
) -> Track:
    """Create a Track with sensible defaults for testing."""
    if bbox is None:
        bbox = BBox(0.4, 0.4, 0.6, 0.6)
    return Track(
        track_id=track_id,
        label=label,
        state=state,
        bbox=bbox,
        velocity=velocity,
        confidence=confidence,
        first_frame=0,
        last_frame=10,
        age=10,
        hits=hits,
        misses=0,
        attributes={},
        history=history if history is not None else [],
    )


def _make_zone(
    zone_id: str = "zone_a",
    name: str = "Zone A",
    polygon: list[tuple[float, float]] | None = None,
    zone_type: str = "restricted",
) -> Zone:
    """Create a Zone with sensible defaults for testing."""
    if polygon is None:
        # A square zone in the top-right quadrant: (0.7, 0.0) to (1.0, 0.3)
        polygon = [(0.7, 0.0), (1.0, 0.0), (1.0, 0.3), (0.7, 0.3)]
    return Zone(
        zone_id=zone_id,
        name=name,
        polygon=polygon,
        zone_type=zone_type,
    )


# ---------------------------------------------------------------------------
# TrajectoryPredictor tests
# ---------------------------------------------------------------------------


class TestTrajectoryPredictorInit(unittest.TestCase):
    """Test constructor validation."""

    def test_valid_init(self) -> None:
        tp = TrajectoryPredictor(horizon_frames=10, fps=30.0, alpha=0.5)
        self.assertIsNotNone(tp)

    def test_invalid_horizon(self) -> None:
        with self.assertRaises(ValueError):
            TrajectoryPredictor(horizon_frames=0)

    def test_invalid_fps(self) -> None:
        with self.assertRaises(ValueError):
            TrajectoryPredictor(fps=0.0)
        with self.assertRaises(ValueError):
            TrajectoryPredictor(fps=-1.0)

    def test_invalid_alpha(self) -> None:
        with self.assertRaises(ValueError):
            TrajectoryPredictor(alpha=0.0)
        with self.assertRaises(ValueError):
            TrajectoryPredictor(alpha=1.5)


class TestTrajectoryPredictorPredict(unittest.TestCase):
    """Test basic position prediction."""

    def setUp(self) -> None:
        self.tp = TrajectoryPredictor(horizon_frames=10, fps=10.0, alpha=0.3)

    def test_stationary_entity_no_history(self) -> None:
        """A track with no history and zero velocity should predict at the same position."""
        track = _make_track(
            bbox=BBox(0.4, 0.4, 0.6, 0.6),
            velocity=(0.0, 0.0),
            history=[],
        )
        predictions = self.tp.predict(track)
        self.assertEqual(len(predictions), 10)

        # All predicted positions should be roughly at the original bbox center
        for pred in predictions:
            cx, cy = pred.bbox.center
            self.assertAlmostEqual(cx, 0.5, places=5)
            self.assertAlmostEqual(cy, 0.5, places=5)

    def test_stationary_entity_with_history(self) -> None:
        """A track with history at the same position should not move."""
        same_bbox = BBox(0.4, 0.4, 0.6, 0.6)
        track = _make_track(
            bbox=same_bbox,
            history=[same_bbox, same_bbox, same_bbox, same_bbox],
        )
        predictions = self.tp.predict(track)
        self.assertEqual(len(predictions), 10)

        for pred in predictions:
            cx, cy = pred.bbox.center
            self.assertAlmostEqual(cx, 0.5, places=5)
            self.assertAlmostEqual(cy, 0.5, places=5)

    def test_moving_entity_rightward(self) -> None:
        """A track moving right should predict positions further right."""
        history = [
            BBox(0.1, 0.4, 0.2, 0.6),
            BBox(0.15, 0.4, 0.25, 0.6),
            BBox(0.2, 0.4, 0.3, 0.6),
            BBox(0.25, 0.4, 0.35, 0.6),
            BBox(0.3, 0.4, 0.4, 0.6),
        ]
        track = _make_track(
            bbox=history[-1],
            history=history,
        )
        predictions = self.tp.predict(track)
        self.assertTrue(len(predictions) > 0)

        # Each successive prediction should be further right
        prev_cx = 0.35  # center of current bbox
        for pred in predictions[:5]:
            cx, _ = pred.bbox.center
            self.assertGreater(cx, prev_cx - 1e-6)
            prev_cx = cx

    def test_prediction_count_matches_horizon(self) -> None:
        track = _make_track()
        predictions = self.tp.predict(track)
        self.assertEqual(len(predictions), 10)

    def test_frame_offset_and_timestamp(self) -> None:
        """Frame offsets should be 1..horizon, timestamps should be offset/fps."""
        predictions = self.tp.predict(_make_track())
        for i, pred in enumerate(predictions):
            self.assertEqual(pred.frame_offset, i + 1)
            expected_t = (i + 1) / 10.0
            self.assertAlmostEqual(pred.timestamp_offset, expected_t, places=5)

    def test_bbox_clamped_to_unit_square(self) -> None:
        """Predicted bboxes should never go outside [0, 1]."""
        # Entity at the edge moving outward
        history = [
            BBox(0.85, 0.85, 0.95, 0.95),
            BBox(0.88, 0.88, 0.98, 0.98),
            BBox(0.91, 0.91, 1.0, 1.0),
        ]
        track = _make_track(bbox=history[-1], history=history)
        tp = TrajectoryPredictor(horizon_frames=20, fps=5.0)
        predictions = tp.predict(track)

        for pred in predictions:
            self.assertGreaterEqual(pred.bbox.x1, 0.0)
            self.assertGreaterEqual(pred.bbox.y1, 0.0)
            self.assertLessEqual(pred.bbox.x2, 1.0)
            self.assertLessEqual(pred.bbox.y2, 1.0)

    def test_exited_track_returns_empty(self) -> None:
        track = _make_track(state=TrackState.EXITED)
        self.assertEqual(self.tp.predict(track), [])

    def test_lost_track_returns_empty(self) -> None:
        track = _make_track(state=TrackState.LOST)
        self.assertEqual(self.tp.predict(track), [])

    def test_no_bbox_returns_empty(self) -> None:
        track = _make_track(bbox=None)
        # bbox=None is set manually
        track.bbox = None  # type: ignore[assignment]
        self.assertEqual(self.tp.predict(track), [])


class TestConfidenceDecay(unittest.TestCase):
    """Test that prediction confidence decays over the horizon."""

    def test_confidence_decreases_monotonically(self) -> None:
        """Confidence should decrease (or stay equal) over time."""
        tp = TrajectoryPredictor(horizon_frames=20, fps=5.0)
        history = [
            BBox(0.2, 0.4, 0.3, 0.6),
            BBox(0.25, 0.4, 0.35, 0.6),
            BBox(0.3, 0.4, 0.4, 0.6),
            BBox(0.35, 0.4, 0.45, 0.6),
        ]
        track = _make_track(bbox=history[-1], history=history, confidence=0.95)
        predictions = tp.predict(track)

        prev_conf = 1.0
        for pred in predictions:
            self.assertLessEqual(pred.confidence, prev_conf + 1e-9)
            prev_conf = pred.confidence

    def test_confidence_always_in_range(self) -> None:
        """Confidence should always be in [0, 1]."""
        tp = TrajectoryPredictor(horizon_frames=50, fps=2.0)
        track = _make_track(confidence=0.99)
        predictions = tp.predict(track)

        for pred in predictions:
            self.assertGreaterEqual(pred.confidence, 0.0)
            self.assertLessEqual(pred.confidence, 1.0)

    def test_stable_track_decays_slower(self) -> None:
        """A track with consistent velocity should retain confidence longer
        than an erratic track."""
        tp = TrajectoryPredictor(horizon_frames=10, fps=5.0)

        # Stable track: constant velocity
        stable_history = [BBox(0.1 + 0.05 * i, 0.5, 0.2 + 0.05 * i, 0.6) for i in range(6)]
        stable_track = _make_track(
            track_id=1, bbox=stable_history[-1], history=stable_history, confidence=0.9
        )

        # Erratic track: oscillating velocity
        erratic_history = [
            BBox(0.1, 0.5, 0.2, 0.6),
            BBox(0.2, 0.5, 0.3, 0.6),
            BBox(0.1, 0.5, 0.2, 0.6),
            BBox(0.2, 0.5, 0.3, 0.6),
            BBox(0.1, 0.5, 0.2, 0.6),
            BBox(0.2, 0.5, 0.3, 0.6),
        ]
        erratic_track = _make_track(
            track_id=2, bbox=erratic_history[-1], history=erratic_history, confidence=0.9
        )

        stable_preds = tp.predict(stable_track)
        erratic_preds = tp.predict(erratic_track)

        # Compare confidence at the last prediction
        last_stable_conf = stable_preds[-1].confidence
        last_erratic_conf = erratic_preds[-1].confidence
        self.assertGreater(
            last_stable_conf,
            last_erratic_conf,
            "Stable track should retain more confidence than erratic track"
        )


class TestSingleFrameTrack(unittest.TestCase):
    """Edge case: track with only a single bbox in history."""

    def test_single_history_entry(self) -> None:
        """A track with one history entry should fall back to stored velocity."""
        single_bbox = BBox(0.3, 0.3, 0.5, 0.5)
        track = _make_track(
            bbox=single_bbox,
            velocity=(0.01, 0.0),
            history=[single_bbox],
        )
        tp = TrajectoryPredictor(horizon_frames=5, fps=5.0)
        predictions = tp.predict(track)

        self.assertEqual(len(predictions), 5)
        # Should move rightward using the stored velocity
        for i, pred in enumerate(predictions):
            cx, _ = pred.bbox.center
            expected_cx = 0.4 + 0.01 * (i + 1)
            self.assertAlmostEqual(cx, expected_cx, places=4)


# ---------------------------------------------------------------------------
# Zone entry prediction tests
# ---------------------------------------------------------------------------


class TestZoneEntryPrediction(unittest.TestCase):
    """Test predict_zone_entry."""

    def setUp(self) -> None:
        self.tp = TrajectoryPredictor(horizon_frames=30, fps=10.0, alpha=0.3)

    def test_stationary_no_zone_entry(self) -> None:
        """A stationary entity should not be predicted to enter any zone."""
        same_bbox = BBox(0.1, 0.1, 0.2, 0.2)
        track = _make_track(
            bbox=same_bbox,
            velocity=(0.0, 0.0),
            history=[same_bbox, same_bbox, same_bbox],
        )
        zone = _make_zone()  # top-right quadrant
        preds = self.tp.predict_zone_entry(track, [zone])
        self.assertEqual(len(preds), 0)

    def test_moving_toward_zone(self) -> None:
        """An entity heading toward a zone should have a predicted entry."""
        # Entity at x=0.5, moving right at 0.05/frame toward zone at x=[0.7, 1.0]
        history = [
            BBox(0.35, 0.1, 0.45, 0.2),
            BBox(0.40, 0.1, 0.50, 0.2),
            BBox(0.45, 0.1, 0.55, 0.2),
            BBox(0.50, 0.1, 0.60, 0.2),
        ]
        track = _make_track(bbox=history[-1], history=history)
        zone = _make_zone()  # polygon: x in [0.7, 1.0], y in [0.0, 0.3]
        preds = self.tp.predict_zone_entry(track, [zone])

        self.assertGreater(len(preds), 0)
        entry = preds[0]
        self.assertEqual(entry.zone_id, "zone_a")
        self.assertEqual(entry.entity_id, 1)
        self.assertGreater(entry.estimated_frames, 0)
        self.assertGreater(entry.estimated_seconds, 0.0)
        self.assertGreater(entry.confidence, 0.0)
        # Entry point should be inside the zone
        px, py = entry.predicted_entry_point
        self.assertGreaterEqual(px, 0.7)
        self.assertLessEqual(py, 0.3)

    def test_moving_away_from_zone(self) -> None:
        """An entity heading away from a zone should not predict entry."""
        # Entity at x=0.5, moving LEFT (away from zone at x=[0.7, 1.0])
        history = [
            BBox(0.55, 0.1, 0.65, 0.2),
            BBox(0.50, 0.1, 0.60, 0.2),
            BBox(0.45, 0.1, 0.55, 0.2),
            BBox(0.40, 0.1, 0.50, 0.2),
        ]
        track = _make_track(bbox=history[-1], history=history)
        zone = _make_zone()
        preds = self.tp.predict_zone_entry(track, [zone])
        self.assertEqual(len(preds), 0)

    def test_already_inside_zone_not_predicted(self) -> None:
        """If entity is already inside the zone, no entry is predicted."""
        inside_bbox = BBox(0.8, 0.1, 0.9, 0.2)
        track = _make_track(bbox=inside_bbox, history=[inside_bbox, inside_bbox])
        zone = _make_zone()  # entity is inside
        preds = self.tp.predict_zone_entry(track, [zone])
        self.assertEqual(len(preds), 0)

    def test_no_zones_returns_empty(self) -> None:
        track = _make_track()
        preds = self.tp.predict_zone_entry(track, [])
        self.assertEqual(len(preds), 0)

    def test_no_bbox_returns_empty(self) -> None:
        track = _make_track()
        track.bbox = None  # type: ignore[assignment]
        preds = self.tp.predict_zone_entry(track, [_make_zone()])
        self.assertEqual(len(preds), 0)

    def test_multiple_zones(self) -> None:
        """If entity is heading toward multiple zones, predict entry for each."""
        # Entity at left side, moving right
        history = [
            BBox(0.0, 0.4, 0.1, 0.6),
            BBox(0.05, 0.4, 0.15, 0.6),
            BBox(0.10, 0.4, 0.20, 0.6),
        ]
        track = _make_track(bbox=history[-1], history=history)

        # Two zones the entity will pass through
        zone_mid = _make_zone(
            zone_id="zone_mid",
            name="Mid Zone",
            polygon=[(0.3, 0.3), (0.5, 0.3), (0.5, 0.7), (0.3, 0.7)],
        )
        zone_right = _make_zone(
            zone_id="zone_right",
            name="Right Zone",
            polygon=[(0.7, 0.3), (1.0, 0.3), (1.0, 0.7), (0.7, 0.7)],
        )

        tp = TrajectoryPredictor(horizon_frames=60, fps=10.0)
        preds = tp.predict_zone_entry(track, [zone_mid, zone_right])

        zone_ids = {p.zone_id for p in preds}
        # At minimum, the mid zone should be reachable
        self.assertIn("zone_mid", zone_ids)


# ---------------------------------------------------------------------------
# Collision prediction tests
# ---------------------------------------------------------------------------


class TestCollisionPrediction(unittest.TestCase):
    """Test predict_collision."""

    def setUp(self) -> None:
        self.tp = TrajectoryPredictor(horizon_frames=30, fps=10.0, alpha=0.3)

    def test_head_on_collision(self) -> None:
        """Two entities heading toward each other should predict a collision."""
        # Entity A moving right
        history_a = [
            BBox(0.1, 0.45, 0.2, 0.55),
            BBox(0.15, 0.45, 0.25, 0.55),
            BBox(0.2, 0.45, 0.3, 0.55),
        ]
        track_a = _make_track(track_id=1, bbox=history_a[-1], history=history_a)

        # Entity B moving left
        history_b = [
            BBox(0.8, 0.45, 0.9, 0.55),
            BBox(0.75, 0.45, 0.85, 0.55),
            BBox(0.7, 0.45, 0.8, 0.55),
        ]
        track_b = _make_track(track_id=2, bbox=history_b[-1], history=history_b)

        collision = self.tp.predict_collision(track_a, track_b)
        self.assertIsNotNone(collision)
        self.assertEqual(collision.entity_a_id, 1)
        self.assertEqual(collision.entity_b_id, 2)
        self.assertGreater(collision.estimated_frames, 0)
        self.assertGreater(collision.estimated_seconds, 0.0)
        self.assertGreater(collision.confidence, 0.0)

    def test_parallel_paths_no_collision(self) -> None:
        """Two entities moving in parallel should not collide."""
        history_a = [
            BBox(0.1, 0.2, 0.2, 0.3),
            BBox(0.15, 0.2, 0.25, 0.3),
            BBox(0.2, 0.2, 0.3, 0.3),
        ]
        track_a = _make_track(track_id=1, bbox=history_a[-1], history=history_a)

        history_b = [
            BBox(0.1, 0.7, 0.2, 0.8),
            BBox(0.15, 0.7, 0.25, 0.8),
            BBox(0.2, 0.7, 0.3, 0.8),
        ]
        track_b = _make_track(track_id=2, bbox=history_b[-1], history=history_b)

        collision = self.tp.predict_collision(track_a, track_b)
        self.assertIsNone(collision)

    def test_diverging_paths_no_collision(self) -> None:
        """Two entities moving apart should not collide."""
        history_a = [
            BBox(0.45, 0.45, 0.55, 0.55),
            BBox(0.4, 0.4, 0.5, 0.5),
            BBox(0.35, 0.35, 0.45, 0.45),
        ]
        track_a = _make_track(track_id=1, bbox=history_a[-1], history=history_a)

        history_b = [
            BBox(0.55, 0.55, 0.65, 0.65),
            BBox(0.6, 0.6, 0.7, 0.7),
            BBox(0.65, 0.65, 0.75, 0.75),
        ]
        track_b = _make_track(track_id=2, bbox=history_b[-1], history=history_b)

        collision = self.tp.predict_collision(track_a, track_b)
        self.assertIsNone(collision)

    def test_already_overlapping_not_reported(self) -> None:
        """If tracks already overlap, predict_collision returns None
        (it's not a future collision)."""
        overlap_bbox_a = BBox(0.4, 0.4, 0.6, 0.6)
        overlap_bbox_b = BBox(0.5, 0.5, 0.7, 0.7)
        track_a = _make_track(track_id=1, bbox=overlap_bbox_a, history=[overlap_bbox_a])
        track_b = _make_track(track_id=2, bbox=overlap_bbox_b, history=[overlap_bbox_b])

        collision = self.tp.predict_collision(track_a, track_b)
        self.assertIsNone(collision)

    def test_no_bbox_returns_none(self) -> None:
        track_a = _make_track(track_id=1)
        track_b = _make_track(track_id=2)
        track_a.bbox = None  # type: ignore[assignment]
        self.assertIsNone(self.tp.predict_collision(track_a, track_b))

    def test_collision_point_is_midpoint(self) -> None:
        """The collision point should be the midpoint of the two predicted centers."""
        history_a = [
            BBox(0.1, 0.45, 0.2, 0.55),
            BBox(0.15, 0.45, 0.25, 0.55),
            BBox(0.2, 0.45, 0.3, 0.55),
        ]
        track_a = _make_track(track_id=1, bbox=history_a[-1], history=history_a)

        history_b = [
            BBox(0.8, 0.45, 0.9, 0.55),
            BBox(0.75, 0.45, 0.85, 0.55),
            BBox(0.7, 0.45, 0.8, 0.55),
        ]
        track_b = _make_track(track_id=2, bbox=history_b[-1], history=history_b)

        collision = self.tp.predict_collision(track_a, track_b)
        self.assertIsNotNone(collision)
        # The collision point should be somewhere between the two starting positions
        cpx, cpy = collision.collision_point
        self.assertGreater(cpx, 0.25)
        self.assertLess(cpx, 0.75)


# ---------------------------------------------------------------------------
# Velocity estimation tests
# ---------------------------------------------------------------------------


class TestVelocityEstimation(unittest.TestCase):
    """Test internal velocity estimation logic via predict() behavior."""

    def test_zero_velocity_history(self) -> None:
        """All-same history gives zero velocity -> predictions stay put."""
        same = BBox(0.5, 0.5, 0.6, 0.6)
        track = _make_track(bbox=same, history=[same, same, same, same])
        tp = TrajectoryPredictor(horizon_frames=5, fps=5.0)
        preds = tp.predict(track)

        for pred in preds:
            cx, cy = pred.bbox.center
            self.assertAlmostEqual(cx, 0.55, places=4)
            self.assertAlmostEqual(cy, 0.55, places=4)

    def test_fallback_to_stored_velocity(self) -> None:
        """With < 2 history entries, should use track.velocity."""
        single = BBox(0.3, 0.3, 0.4, 0.4)
        track = _make_track(
            bbox=single,
            velocity=(0.02, 0.01),
            history=[single],
        )
        tp = TrajectoryPredictor(horizon_frames=5, fps=5.0)
        preds = tp.predict(track)

        # Should move in the direction of stored velocity
        for i, pred in enumerate(preds):
            cx, cy = pred.bbox.center
            step = i + 1
            self.assertAlmostEqual(cx, 0.35 + 0.02 * step, places=4)
            self.assertAlmostEqual(cy, 0.35 + 0.01 * step, places=4)


# ---------------------------------------------------------------------------
# ProactiveAlertGenerator tests
# ---------------------------------------------------------------------------


class TestProactiveAlertGenerator(unittest.TestCase):
    """Test alert generation."""

    def setUp(self) -> None:
        self.tp = TrajectoryPredictor(horizon_frames=30, fps=10.0, alpha=0.3)
        self.gen = ProactiveAlertGenerator(
            predictor=self.tp,
            zone_alert_seconds=5.0,
            collision_alert_seconds=3.0,
        )

    def test_zone_entry_alert(self) -> None:
        """Should generate ZONE_ENTRY_PREDICTED event for an approaching entity."""
        # Entity moving right toward zone
        history = [
            BBox(0.45, 0.1, 0.55, 0.2),
            BBox(0.50, 0.1, 0.60, 0.2),
            BBox(0.55, 0.1, 0.65, 0.2),
        ]
        track = _make_track(track_id=1, bbox=history[-1], history=history)
        zone = _make_zone()  # x in [0.7, 1.0], y in [0.0, 0.3]

        events = self.gen.generate_alerts(
            tracks={1: track},
            zones=[zone],
            current_frame=100,
            current_timestamp=10.0,
        )

        zone_events = [
            e for e in events
            if e.event_type == EntityEventType.ZONE_ENTRY_PREDICTED
        ]
        self.assertGreater(len(zone_events), 0)

        evt = zone_events[0]
        self.assertEqual(evt.entity_id, 1)
        self.assertEqual(evt.details["zone_id"], "zone_a")
        self.assertIn("time_to_entry", evt.details)
        self.assertIn("predicted_entry_point", evt.details)

    def test_no_alert_for_distant_zone(self) -> None:
        """No alert if zone entry is beyond the alert threshold."""
        # Entity far from zone, moving slowly
        same_bbox = BBox(0.0, 0.0, 0.05, 0.05)
        track = _make_track(
            track_id=1,
            bbox=same_bbox,
            velocity=(0.0, 0.0),
            history=[same_bbox, same_bbox],
        )
        zone = _make_zone()

        events = self.gen.generate_alerts(
            tracks={1: track},
            zones=[zone],
        )
        zone_events = [
            e for e in events
            if e.event_type == EntityEventType.ZONE_ENTRY_PREDICTED
        ]
        self.assertEqual(len(zone_events), 0)

    def test_collision_alert(self) -> None:
        """Should generate COLLISION_PREDICTED event for converging entities."""
        history_a = [
            BBox(0.1, 0.45, 0.2, 0.55),
            BBox(0.15, 0.45, 0.25, 0.55),
            BBox(0.2, 0.45, 0.3, 0.55),
        ]
        track_a = _make_track(track_id=1, bbox=history_a[-1], history=history_a)

        history_b = [
            BBox(0.8, 0.45, 0.9, 0.55),
            BBox(0.75, 0.45, 0.85, 0.55),
            BBox(0.7, 0.45, 0.8, 0.55),
        ]
        track_b = _make_track(track_id=2, bbox=history_b[-1], history=history_b)

        events = self.gen.generate_alerts(
            tracks={1: track_a, 2: track_b},
            zones=[],
            current_frame=50,
            current_timestamp=5.0,
        )

        collision_events = [
            e for e in events
            if e.event_type == EntityEventType.COLLISION_PREDICTED
        ]
        self.assertGreater(len(collision_events), 0)

        evt = collision_events[0]
        self.assertIn("entity_a_id", evt.details)
        self.assertIn("entity_b_id", evt.details)
        self.assertIn("time_to_collision", evt.details)
        self.assertIn("collision_point", evt.details)

    def test_exited_tracks_ignored(self) -> None:
        """Exited tracks should not generate alerts."""
        track = _make_track(track_id=1, state=TrackState.EXITED)
        zone = _make_zone()

        events = self.gen.generate_alerts(
            tracks={1: track},
            zones=[zone],
        )
        self.assertEqual(len(events), 0)

    def test_lost_tracks_ignored(self) -> None:
        """Lost tracks should not generate alerts."""
        track = _make_track(track_id=1, state=TrackState.LOST)
        zone = _make_zone()

        events = self.gen.generate_alerts(
            tracks={1: track},
            zones=[zone],
        )
        self.assertEqual(len(events), 0)

    def test_empty_tracks(self) -> None:
        """No tracks -> no alerts."""
        events = self.gen.generate_alerts(
            tracks={},
            zones=[_make_zone()],
        )
        self.assertEqual(len(events), 0)

    def test_no_zones_no_zone_alerts(self) -> None:
        """No zones -> no zone alerts (collision alerts still possible)."""
        track = _make_track(track_id=1)
        events = self.gen.generate_alerts(
            tracks={1: track},
            zones=[],
        )
        zone_events = [
            e for e in events
            if e.event_type == EntityEventType.ZONE_ENTRY_PREDICTED
        ]
        self.assertEqual(len(zone_events), 0)

    def test_collision_alert_respects_time_threshold(self) -> None:
        """Collision predicted beyond collision_alert_seconds should not alert."""
        # Use a very short alert window (0.1 seconds) so that only very
        # imminent collisions would alert.
        gen = ProactiveAlertGenerator(
            predictor=self.tp,
            zone_alert_seconds=5.0,
            collision_alert_seconds=0.1,  # very short window
        )

        # Entities far apart, will collide but way beyond 0.1s
        history_a = [
            BBox(0.0, 0.45, 0.1, 0.55),
            BBox(0.02, 0.45, 0.12, 0.55),
            BBox(0.04, 0.45, 0.14, 0.55),
        ]
        track_a = _make_track(track_id=1, bbox=history_a[-1], history=history_a)

        history_b = [
            BBox(0.9, 0.45, 1.0, 0.55),
            BBox(0.88, 0.45, 0.98, 0.55),
            BBox(0.86, 0.45, 0.96, 0.55),
        ]
        track_b = _make_track(track_id=2, bbox=history_b[-1], history=history_b)

        events = gen.generate_alerts(
            tracks={1: track_a, 2: track_b},
            zones=[],
        )
        collision_events = [
            e for e in events
            if e.event_type == EntityEventType.COLLISION_PREDICTED
        ]
        # Should not alert because collision is too far in the future
        self.assertEqual(len(collision_events), 0)


# ---------------------------------------------------------------------------
# Data type tests
# ---------------------------------------------------------------------------


class TestDataTypes(unittest.TestCase):
    """Test the frozen dataclass types."""

    def test_predicted_position_frozen(self) -> None:
        pp = PredictedPosition(
            frame_offset=5,
            timestamp_offset=0.5,
            bbox=BBox(0.1, 0.1, 0.2, 0.2),
            confidence=0.8,
        )
        with self.assertRaises(AttributeError):
            pp.frame_offset = 10  # type: ignore[misc]

    def test_zone_entry_prediction_frozen(self) -> None:
        zep = ZoneEntryPrediction(
            zone_id="z1",
            zone_name="Zone 1",
            entity_id=1,
            estimated_frames=5,
            estimated_seconds=0.5,
            confidence=0.8,
            predicted_entry_point=(0.5, 0.5),
        )
        with self.assertRaises(AttributeError):
            zep.zone_id = "z2"  # type: ignore[misc]

    def test_collision_prediction_frozen(self) -> None:
        cp = CollisionPrediction(
            entity_a_id=1,
            entity_b_id=2,
            estimated_frames=10,
            estimated_seconds=1.0,
            confidence=0.7,
            collision_point=(0.5, 0.5),
        )
        with self.assertRaises(AttributeError):
            cp.entity_a_id = 3  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EntityEventType extension tests
# ---------------------------------------------------------------------------


class TestEventTypes(unittest.TestCase):
    """Verify the new event types were added correctly."""

    def test_zone_entry_predicted_exists(self) -> None:
        self.assertEqual(
            EntityEventType.ZONE_ENTRY_PREDICTED.value,
            "zone_entry_predicted",
        )

    def test_collision_predicted_exists(self) -> None:
        self.assertEqual(
            EntityEventType.COLLISION_PREDICTED.value,
            "collision_predicted",
        )


if __name__ == "__main__":
    unittest.main()
