"""Comprehensive tests for Activity Recognition from Trajectory Patterns.

Tests cover:
    - ActivityFeatures extraction from known trajectories
    - Classification of each ActivityType
    - Loitering detection (entity in small area for many frames)
    - Erratic movement detection (frequent direction changes)
    - Running vs walking speed thresholds
    - Activity transition events via ActivityMonitor
    - Edge cases: empty history, single point, two points, minimal history
    - Confidence calculation behavior
    - Batch classification
    - Secondary activity reporting

Run:  python -m pytest tests/test_activity.py -v
"""

from __future__ import annotations

import math
import unittest

from sopilot.perception.types import (
    BBox,
    EntityEventType,
    Track,
    TrackState,
)
from sopilot.perception.activity import (
    ActivityClassification,
    ActivityClassifier,
    ActivityFeatures,
    ActivityMonitor,
    ActivityType,
)


# ---------------------------------------------------------------------------
# Helpers — build tracks with known trajectories
# ---------------------------------------------------------------------------


def _make_track(
    track_id: int = 1,
    label: str = "person",
    history: list[BBox] | None = None,
) -> Track:
    """Create a Track with the given bbox history."""
    h = history or []
    return Track(
        track_id=track_id,
        label=label,
        state=TrackState.ACTIVE,
        bbox=h[-1] if h else None,
        history=h,
        first_frame=0,
        last_frame=max(0, len(h) - 1),
        age=len(h),
        hits=len(h),
    )


def _make_bbox(cx: float, cy: float, size: float = 0.1) -> BBox:
    """Create a BBox centered at (cx, cy) with given size."""
    half = size / 2
    return BBox(
        x1=max(0.0, cx - half),
        y1=max(0.0, cy - half),
        x2=min(1.0, cx + half),
        y2=min(1.0, cy + half),
    )


def _straight_line_track(
    start: tuple[float, float],
    end: tuple[float, float],
    n_frames: int,
    track_id: int = 1,
) -> Track:
    """Create a track that moves in a straight line from start to end."""
    history = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        cx = start[0] + t * (end[0] - start[0])
        cy = start[1] + t * (end[1] - start[1])
        history.append(_make_bbox(cx, cy))
    return _make_track(track_id=track_id, history=history)


def _stationary_track(
    center: tuple[float, float],
    n_frames: int,
    track_id: int = 1,
) -> Track:
    """Create a track that stays at the same position for n_frames."""
    history = [_make_bbox(center[0], center[1]) for _ in range(n_frames)]
    return _make_track(track_id=track_id, history=history)


def _circular_track(
    center: tuple[float, float],
    radius: float,
    n_frames: int,
    track_id: int = 1,
) -> Track:
    """Create a track that moves in a circle (returns near start)."""
    history = []
    for i in range(n_frames):
        angle = 2 * math.pi * i / n_frames
        cx = center[0] + radius * math.cos(angle)
        cy = center[1] + radius * math.sin(angle)
        history.append(_make_bbox(cx, cy))
    return _make_track(track_id=track_id, history=history)


def _zigzag_track(
    start: tuple[float, float],
    amplitude: float,
    step_x: float,
    n_frames: int,
    track_id: int = 1,
) -> Track:
    """Create a track that zigzags up and down (frequent direction changes)."""
    history = []
    for i in range(n_frames):
        cx = start[0] + i * step_x
        # Alternate y position every frame
        cy = start[1] + amplitude * (1 if i % 2 == 0 else -1)
        history.append(_make_bbox(cx, cy))
    return _make_track(track_id=track_id, history=history)


# ---------------------------------------------------------------------------
# Test: ActivityType enum
# ---------------------------------------------------------------------------


class TestActivityType(unittest.TestCase):
    """Test the ActivityType enumeration."""

    def test_all_values(self) -> None:
        expected = {
            "stationary", "walking", "running", "loitering",
            "erratic", "approaching", "departing", "unknown",
        }
        actual = {a.value for a in ActivityType}
        self.assertEqual(actual, expected)

    def test_from_value(self) -> None:
        self.assertEqual(ActivityType("walking"), ActivityType.WALKING)
        self.assertEqual(ActivityType("loitering"), ActivityType.LOITERING)


# ---------------------------------------------------------------------------
# Test: ActivityFeatures extraction
# ---------------------------------------------------------------------------


class TestFeatureExtraction(unittest.TestCase):
    """Test extraction of motion features from track histories."""

    def setUp(self) -> None:
        self.classifier = ActivityClassifier()

    def test_empty_history(self) -> None:
        track = _make_track(history=[])
        features = self.classifier.extract_features(track)
        self.assertEqual(features.duration_frames, 0)
        self.assertEqual(features.mean_speed, 0.0)
        self.assertEqual(features.max_speed, 0.0)
        self.assertEqual(features.speed_variance, 0.0)
        self.assertEqual(features.mean_acceleration, 0.0)
        self.assertEqual(features.direction_change_rate, 0.0)
        self.assertEqual(features.displacement_ratio, 0.0)
        self.assertEqual(features.bounding_area, 0.0)

    def test_single_point(self) -> None:
        track = _make_track(history=[_make_bbox(0.5, 0.5)])
        features = self.classifier.extract_features(track)
        self.assertEqual(features.duration_frames, 1)
        self.assertEqual(features.mean_speed, 0.0)

    def test_two_points(self) -> None:
        track = _make_track(history=[
            _make_bbox(0.1, 0.1),
            _make_bbox(0.2, 0.1),
        ])
        features = self.classifier.extract_features(track)
        self.assertEqual(features.duration_frames, 2)
        self.assertAlmostEqual(features.mean_speed, 0.1, places=5)
        self.assertAlmostEqual(features.max_speed, 0.1, places=5)
        self.assertAlmostEqual(features.speed_variance, 0.0, places=5)
        # Two points: only one direction vector, no direction change possible
        self.assertEqual(features.direction_change_rate, 0.0)
        # Straight line: displacement ratio = 1.0
        self.assertAlmostEqual(features.displacement_ratio, 1.0, places=5)

    def test_stationary_features(self) -> None:
        """Stationary track: all speeds near zero, bounding area near zero."""
        track = _stationary_track((0.5, 0.5), n_frames=20)
        features = self.classifier.extract_features(track)
        self.assertEqual(features.duration_frames, 20)
        self.assertAlmostEqual(features.mean_speed, 0.0, places=5)
        self.assertAlmostEqual(features.max_speed, 0.0, places=5)
        self.assertAlmostEqual(features.bounding_area, 0.0, places=5)

    def test_straight_line_features(self) -> None:
        """Straight line: displacement ratio should be close to 1.0."""
        track = _straight_line_track((0.1, 0.5), (0.9, 0.5), n_frames=20)
        features = self.classifier.extract_features(track)
        self.assertEqual(features.duration_frames, 20)
        self.assertGreater(features.mean_speed, 0)
        # Straight line has displacement ratio ~1.0
        self.assertAlmostEqual(features.displacement_ratio, 1.0, places=2)
        # Low direction change rate (no direction changes)
        self.assertAlmostEqual(features.direction_change_rate, 0.0, places=2)

    def test_circular_features(self) -> None:
        """Circular track: displacement ratio near 0.0 (returns to start)."""
        track = _circular_track((0.5, 0.5), radius=0.1, n_frames=40)
        features = self.classifier.extract_features(track)
        # Circular movement returns near start -> low displacement ratio
        self.assertLess(features.displacement_ratio, 0.2)
        # Bounding area should be nonzero (radius 0.1 -> ~0.04 area)
        self.assertGreater(features.bounding_area, 0.0)

    def test_zigzag_features_direction_changes(self) -> None:
        """Zigzag track should have high direction change rate."""
        track = _zigzag_track(
            start=(0.1, 0.5), amplitude=0.05, step_x=0.01, n_frames=20
        )
        features = self.classifier.extract_features(track)
        # Zigzag produces frequent direction changes
        self.assertGreater(features.direction_change_rate, 0.3)

    def test_speed_consistency_constant_velocity(self) -> None:
        """Constant velocity -> low speed variance."""
        # Uniform spacing -> constant speed (start away from 0.0 to avoid
        # bbox clamping that introduces minor center shifts)
        track = _straight_line_track((0.2, 0.5), (0.7, 0.5), n_frames=10)
        features = self.classifier.extract_features(track)
        self.assertAlmostEqual(features.speed_variance, 0.0, places=6)

    def test_bounding_area_for_small_movement(self) -> None:
        """Entity that drifts slightly should have small bounding area."""
        history = [
            _make_bbox(0.50, 0.50),
            _make_bbox(0.501, 0.501),
            _make_bbox(0.502, 0.500),
            _make_bbox(0.500, 0.502),
            _make_bbox(0.501, 0.501),
        ]
        track = _make_track(history=history)
        features = self.classifier.extract_features(track)
        self.assertLess(features.bounding_area, 0.001)

    def test_features_are_frozen(self) -> None:
        """ActivityFeatures should be immutable."""
        track = _stationary_track((0.5, 0.5), n_frames=5)
        features = self.classifier.extract_features(track)
        with self.assertRaises(AttributeError):
            features.mean_speed = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: Classification of each activity type
# ---------------------------------------------------------------------------


class TestClassification(unittest.TestCase):
    """Test rule-based classification of track activities."""

    def setUp(self) -> None:
        self.classifier = ActivityClassifier()

    def test_stationary_classification(self) -> None:
        """Entity sitting still for a few frames -> STATIONARY."""
        track = _stationary_track((0.5, 0.5), n_frames=10)
        result = self.classifier.classify(track)
        self.assertEqual(result.activity, ActivityType.STATIONARY)
        self.assertGreater(result.confidence, 0.5)

    def test_walking_classification(self) -> None:
        """Entity moving at moderate speed -> WALKING."""
        # Walking range default: (0.005, 0.03)
        # For 10 frames, move 0.01 per frame in x => mean_speed ~ 0.01
        track = _straight_line_track((0.3, 0.5), (0.39, 0.5), n_frames=10)
        result = self.classifier.classify(track)
        self.assertEqual(result.activity, ActivityType.WALKING)
        self.assertGreater(result.confidence, 0.4)

    def test_running_classification(self) -> None:
        """Entity moving fast -> RUNNING."""
        # Running threshold default: 0.03
        # Move 0.05 per frame => mean_speed ~ 0.05 > 0.03
        track = _straight_line_track((0.1, 0.5), (0.6, 0.5), n_frames=11)
        features = self.classifier.extract_features(track)
        self.assertGreater(features.mean_speed, 0.03)
        result = self.classifier.classify(track)
        self.assertEqual(result.activity, ActivityType.RUNNING)
        self.assertGreater(result.confidence, 0.5)

    def test_loitering_classification(self) -> None:
        """Entity staying in small area for many frames -> LOITERING."""
        track = _stationary_track((0.5, 0.5), n_frames=50)
        result = self.classifier.classify(track)
        self.assertEqual(result.activity, ActivityType.LOITERING)
        self.assertGreater(result.confidence, 0.5)

    def test_loitering_requires_min_frames(self) -> None:
        """Stationary for fewer than loiter_min_frames -> STATIONARY, not LOITERING."""
        track = _stationary_track((0.5, 0.5), n_frames=10)
        result = self.classifier.classify(track)
        self.assertEqual(result.activity, ActivityType.STATIONARY)

    def test_erratic_classification(self) -> None:
        """Entity with frequent direction changes -> ERRATIC."""
        # Build a track with very frequent direction changes
        history = []
        for i in range(20):
            # Move in alternating diagonal directions
            if i % 4 == 0:
                cx, cy = 0.5 + 0.02, 0.5 + 0.02
            elif i % 4 == 1:
                cx, cy = 0.5 - 0.02, 0.5 + 0.02
            elif i % 4 == 2:
                cx, cy = 0.5 - 0.02, 0.5 - 0.02
            else:
                cx, cy = 0.5 + 0.02, 0.5 - 0.02
            history.append(_make_bbox(cx, cy))
        track = _make_track(history=history)
        result = self.classifier.classify(track)
        # The direction change rate should be high enough for ERRATIC
        self.assertIn(result.activity, {ActivityType.ERRATIC, ActivityType.WALKING})
        # Verify features show high direction change rate
        features = self.classifier.extract_features(track)
        if features.direction_change_rate > 0.5:
            self.assertEqual(result.activity, ActivityType.ERRATIC)

    def test_unknown_for_short_history(self) -> None:
        """Track with fewer than min_history_frames -> UNKNOWN."""
        track = _make_track(history=[
            _make_bbox(0.1, 0.1),
            _make_bbox(0.2, 0.2),
        ])
        result = self.classifier.classify(track)
        self.assertEqual(result.activity, ActivityType.UNKNOWN)
        self.assertEqual(result.confidence, 0.0)

    def test_unknown_for_empty_history(self) -> None:
        """Track with no history -> UNKNOWN."""
        track = _make_track(history=[])
        result = self.classifier.classify(track)
        self.assertEqual(result.activity, ActivityType.UNKNOWN)
        self.assertEqual(result.confidence, 0.0)

    def test_classification_result_is_frozen(self) -> None:
        """ActivityClassification is immutable."""
        track = _stationary_track((0.5, 0.5), n_frames=10)
        result = self.classifier.classify(track)
        with self.assertRaises(AttributeError):
            result.activity = ActivityType.RUNNING  # type: ignore[misc]

    def test_secondary_activity_reported(self) -> None:
        """Classification should include a secondary (runner-up) activity."""
        # A track near a boundary should have a meaningful secondary activity
        track = _stationary_track((0.5, 0.5), n_frames=50)
        result = self.classifier.classify(track)
        # LOITERING primary, STATIONARY secondary (both match slow+small area)
        self.assertEqual(result.activity, ActivityType.LOITERING)
        self.assertIsNotNone(result.secondary_activity)
        self.assertEqual(result.secondary_activity, ActivityType.STATIONARY)
        self.assertGreater(result.secondary_confidence, 0.0)

    def test_confidence_high_when_far_from_boundary(self) -> None:
        """Confidence should be high when features are far from thresholds."""
        # Very clearly running (speed much above threshold)
        track = _straight_line_track((0.0, 0.5), (0.9, 0.5), n_frames=10)
        result = self.classifier.classify(track)
        self.assertEqual(result.activity, ActivityType.RUNNING)
        self.assertGreaterEqual(result.confidence, 0.7)


# ---------------------------------------------------------------------------
# Test: Speed threshold boundaries
# ---------------------------------------------------------------------------


class TestSpeedThresholds(unittest.TestCase):
    """Test running vs walking vs stationary boundary behavior."""

    def test_just_below_stationary_threshold(self) -> None:
        """Speed just below stationary threshold -> STATIONARY."""
        # Default: stationary_speed_threshold = 0.005
        # 10 frames, move 0.004 per frame (just under threshold)
        track = _straight_line_track((0.5, 0.5), (0.536, 0.5), n_frames=10)
        features = ActivityClassifier().extract_features(track)
        self.assertLess(features.mean_speed, 0.005)
        result = ActivityClassifier().classify(track)
        self.assertEqual(result.activity, ActivityType.STATIONARY)

    def test_at_walking_speed(self) -> None:
        """Speed in walking range -> WALKING."""
        # 10 frames, 0.015 per frame (middle of walking range 0.005-0.03)
        track = _straight_line_track((0.3, 0.5), (0.435, 0.5), n_frames=10)
        features = ActivityClassifier().extract_features(track)
        self.assertGreater(features.mean_speed, 0.005)
        self.assertLess(features.mean_speed, 0.03)
        result = ActivityClassifier().classify(track)
        self.assertEqual(result.activity, ActivityType.WALKING)

    def test_above_running_threshold(self) -> None:
        """Speed above running threshold -> RUNNING."""
        # 10 frames, 0.05 per frame
        track = _straight_line_track((0.1, 0.5), (0.55, 0.5), n_frames=10)
        features = ActivityClassifier().extract_features(track)
        self.assertGreater(features.mean_speed, 0.03)
        result = ActivityClassifier().classify(track)
        self.assertEqual(result.activity, ActivityType.RUNNING)

    def test_custom_thresholds(self) -> None:
        """Classifier with custom thresholds should use them."""
        classifier = ActivityClassifier(
            stationary_speed_threshold=0.01,
            walking_speed_range=(0.01, 0.05),
            running_speed_threshold=0.05,
            min_history_frames=3,
        )
        # Speed 0.007 per frame: under custom stationary threshold
        track = _straight_line_track((0.5, 0.5), (0.521, 0.5), n_frames=4)
        result = classifier.classify(track)
        self.assertEqual(result.activity, ActivityType.STATIONARY)


# ---------------------------------------------------------------------------
# Test: Loitering detection
# ---------------------------------------------------------------------------


class TestLoiteringDetection(unittest.TestCase):
    """Test loitering (entity in small area for extended time)."""

    def test_loitering_at_exact_threshold(self) -> None:
        """Entity at exactly loiter_min_frames -> LOITERING."""
        track = _stationary_track((0.5, 0.5), n_frames=30)
        result = ActivityClassifier().classify(track)
        self.assertEqual(result.activity, ActivityType.LOITERING)

    def test_no_loitering_one_frame_short(self) -> None:
        """Entity one frame below loiter_min_frames -> STATIONARY."""
        track = _stationary_track((0.5, 0.5), n_frames=29)
        result = ActivityClassifier().classify(track)
        self.assertEqual(result.activity, ActivityType.STATIONARY)

    def test_loitering_with_slight_drift(self) -> None:
        """Entity drifting slightly in small area -> still LOITERING."""
        history = []
        for i in range(40):
            # Tiny random-like drift within a small area
            cx = 0.5 + 0.001 * (i % 3 - 1)
            cy = 0.5 + 0.001 * ((i + 1) % 3 - 1)
            history.append(_make_bbox(cx, cy))
        track = _make_track(history=history)
        features = ActivityClassifier().extract_features(track)
        self.assertLess(features.mean_speed, 0.005)
        self.assertLess(features.bounding_area, 0.01)
        result = ActivityClassifier().classify(track)
        self.assertEqual(result.activity, ActivityType.LOITERING)

    def test_custom_loiter_min_frames(self) -> None:
        """Custom loiter_min_frames is respected."""
        classifier = ActivityClassifier(loiter_min_frames=10)
        track = _stationary_track((0.5, 0.5), n_frames=10)
        result = classifier.classify(track)
        self.assertEqual(result.activity, ActivityType.LOITERING)


# ---------------------------------------------------------------------------
# Test: Erratic movement detection
# ---------------------------------------------------------------------------


class TestErraticDetection(unittest.TestCase):
    """Test erratic movement (frequent direction changes)."""

    def test_strong_zigzag_is_erratic(self) -> None:
        """Track with extreme zigzag pattern -> ERRATIC."""
        # Large amplitude zigzag where direction changes every frame
        history = []
        for i in range(15):
            cx = 0.5 + 0.01 * i
            cy = 0.5 + 0.05 * (1 if i % 2 == 0 else -1)
            history.append(_make_bbox(cx, cy))
        track = _make_track(history=history)
        features = ActivityClassifier().extract_features(track)
        self.assertGreater(features.direction_change_rate, 0.5)
        result = ActivityClassifier().classify(track)
        self.assertEqual(result.activity, ActivityType.ERRATIC)

    def test_straight_line_not_erratic(self) -> None:
        """Straight-line movement should not be classified as erratic."""
        track = _straight_line_track((0.1, 0.5), (0.5, 0.5), n_frames=20)
        features = ActivityClassifier().extract_features(track)
        self.assertLess(features.direction_change_rate, 0.1)
        result = ActivityClassifier().classify(track)
        self.assertNotEqual(result.activity, ActivityType.ERRATIC)


# ---------------------------------------------------------------------------
# Test: Batch classification
# ---------------------------------------------------------------------------


class TestBatchClassification(unittest.TestCase):
    """Test classify_batch() for multiple tracks."""

    def test_batch_returns_all_tracks(self) -> None:
        tracks = {
            1: _stationary_track((0.5, 0.5), n_frames=10),
            2: _straight_line_track((0.1, 0.5), (0.5, 0.5), n_frames=10),
            3: _straight_line_track((0.0, 0.5), (0.9, 0.5), n_frames=10),
        }
        results = ActivityClassifier().classify_batch(tracks)
        self.assertEqual(set(results.keys()), {1, 2, 3})
        for tid, result in results.items():
            self.assertIsInstance(result, ActivityClassification)

    def test_batch_empty(self) -> None:
        results = ActivityClassifier().classify_batch({})
        self.assertEqual(results, {})

    def test_batch_matches_individual(self) -> None:
        """Batch results should match individual classify() calls."""
        classifier = ActivityClassifier()
        tracks = {
            10: _stationary_track((0.3, 0.3), n_frames=10),
            20: _straight_line_track((0.1, 0.5), (0.9, 0.5), n_frames=10),
        }
        batch_results = classifier.classify_batch(tracks)
        for tid, track in tracks.items():
            individual = classifier.classify(track)
            self.assertEqual(batch_results[tid].activity, individual.activity)
            self.assertAlmostEqual(
                batch_results[tid].confidence, individual.confidence, places=5
            )


# ---------------------------------------------------------------------------
# Test: ActivityMonitor (event generation)
# ---------------------------------------------------------------------------


class TestActivityMonitor(unittest.TestCase):
    """Test the ActivityMonitor event generation for activity transitions."""

    def setUp(self) -> None:
        self.classifier = ActivityClassifier(min_history_frames=3)
        self.monitor = ActivityMonitor(self.classifier)

    def test_no_events_on_first_update(self) -> None:
        """First update should not generate events (no previous state)."""
        tracks = {
            1: _stationary_track((0.5, 0.5), n_frames=10),
        }
        events = self.monitor.update(tracks, timestamp=1.0, frame_number=30)
        self.assertEqual(len(events), 0)

    def test_event_on_activity_change(self) -> None:
        """Event generated when activity changes (stationary -> running)."""
        # First update: stationary
        tracks_1 = {
            1: _stationary_track((0.5, 0.5), n_frames=10),
        }
        self.monitor.update(tracks_1, timestamp=1.0, frame_number=10)

        # Second update: running
        tracks_2 = {
            1: _straight_line_track((0.0, 0.5), (0.9, 0.5), n_frames=10),
        }
        events = self.monitor.update(tracks_2, timestamp=2.0, frame_number=20)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.event_type, EntityEventType.STATE_CHANGED)
        self.assertEqual(event.entity_id, 1)
        self.assertEqual(event.details["change_type"], "activity")
        self.assertEqual(event.details["old_activity"], "stationary")
        self.assertEqual(event.details["new_activity"], "running")
        self.assertTrue(event.details["alert"])  # running is in default alert set

    def test_no_event_when_activity_unchanged(self) -> None:
        """No event if activity stays the same."""
        tracks = {
            1: _stationary_track((0.5, 0.5), n_frames=10),
        }
        self.monitor.update(tracks, timestamp=1.0, frame_number=10)
        events = self.monitor.update(tracks, timestamp=2.0, frame_number=20)
        self.assertEqual(len(events), 0)

    def test_departed_track_cleanup(self) -> None:
        """Departed tracks should be cleaned from monitor state."""
        tracks = {
            1: _stationary_track((0.5, 0.5), n_frames=10),
        }
        self.monitor.update(tracks, timestamp=1.0, frame_number=10)

        # Track 1 disappears
        events = self.monitor.update({}, timestamp=2.0, frame_number=20)
        self.assertEqual(len(events), 0)

        # Track 1 reappears — should not generate event (treated as new)
        events = self.monitor.update(tracks, timestamp=3.0, frame_number=30)
        self.assertEqual(len(events), 0)

    def test_multiple_track_transitions(self) -> None:
        """Multiple tracks can generate events simultaneously."""
        # First: both stationary
        tracks_1 = {
            1: _stationary_track((0.3, 0.3), n_frames=10),
            2: _stationary_track((0.7, 0.7), n_frames=10),
        }
        self.monitor.update(tracks_1, timestamp=1.0, frame_number=10)

        # Second: both running
        tracks_2 = {
            1: _straight_line_track((0.0, 0.5), (0.9, 0.5), n_frames=10),
            2: _straight_line_track((0.0, 0.3), (0.9, 0.3), n_frames=10),
        }
        events = self.monitor.update(tracks_2, timestamp=2.0, frame_number=20)
        self.assertEqual(len(events), 2)
        event_ids = {e.entity_id for e in events}
        self.assertEqual(event_ids, {1, 2})

    def test_alert_flag_for_default_activities(self) -> None:
        """Default alert activities: LOITERING, ERRATIC, RUNNING."""
        # walking -> running: alert because running is in alert set
        tracks_walk = {
            1: _straight_line_track((0.3, 0.5), (0.39, 0.5), n_frames=10),
        }
        self.monitor.update(tracks_walk, timestamp=1.0, frame_number=10)

        tracks_run = {
            1: _straight_line_track((0.0, 0.5), (0.9, 0.5), n_frames=10),
        }
        events = self.monitor.update(tracks_run, timestamp=2.0, frame_number=20)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].details["alert"])

    def test_custom_alert_activities(self) -> None:
        """Custom alert_on_activities set is respected."""
        monitor = ActivityMonitor(
            self.classifier,
            alert_on_activities={ActivityType.WALKING},
        )
        # stationary -> walking: alert because walking is in custom alert set
        tracks_stat = {
            1: _stationary_track((0.5, 0.5), n_frames=10),
        }
        monitor.update(tracks_stat, timestamp=1.0, frame_number=10)

        tracks_walk = {
            1: _straight_line_track((0.3, 0.5), (0.39, 0.5), n_frames=10),
        }
        events = monitor.update(tracks_walk, timestamp=2.0, frame_number=20)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].details["alert"])

    def test_event_timestamp_and_frame(self) -> None:
        """Events should carry the correct timestamp and frame_number."""
        tracks_1 = {1: _stationary_track((0.5, 0.5), n_frames=10)}
        self.monitor.update(tracks_1, timestamp=10.0, frame_number=300)

        tracks_2 = {
            1: _straight_line_track((0.0, 0.5), (0.9, 0.5), n_frames=10),
        }
        events = self.monitor.update(tracks_2, timestamp=11.0, frame_number=330)
        self.assertEqual(events[0].timestamp, 11.0)
        self.assertEqual(events[0].frame_number, 330)


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_three_point_minimal_history(self) -> None:
        """Track with exactly min_history_frames (default 5) points."""
        classifier = ActivityClassifier(min_history_frames=3)
        track = _make_track(history=[
            _make_bbox(0.5, 0.5),
            _make_bbox(0.5, 0.5),
            _make_bbox(0.5, 0.5),
        ])
        result = classifier.classify(track)
        # 3 frames, min_history=3 -> should classify (not UNKNOWN)
        self.assertNotEqual(result.confidence, 0.0)

    def test_exactly_min_minus_one(self) -> None:
        """Track with min_history_frames - 1 -> UNKNOWN."""
        classifier = ActivityClassifier(min_history_frames=5)
        track = _make_track(history=[
            _make_bbox(0.1, 0.1),
            _make_bbox(0.2, 0.2),
            _make_bbox(0.3, 0.3),
            _make_bbox(0.4, 0.4),
        ])
        result = classifier.classify(track)
        self.assertEqual(result.activity, ActivityType.UNKNOWN)
        self.assertEqual(result.confidence, 0.0)

    def test_all_same_position(self) -> None:
        """All bbox centers are identical -> STATIONARY or LOITERING."""
        track = _stationary_track((0.5, 0.5), n_frames=10)
        result = ActivityClassifier().classify(track)
        self.assertIn(result.activity, {ActivityType.STATIONARY, ActivityType.LOITERING})

    def test_large_history(self) -> None:
        """Large history should not cause performance issues."""
        track = _straight_line_track((0.0, 0.5), (1.0, 0.5), n_frames=1000)
        result = ActivityClassifier().classify(track)
        self.assertIsInstance(result, ActivityClassification)

    def test_bbox_at_boundary(self) -> None:
        """Track with bbox near image boundaries."""
        history = [
            _make_bbox(0.0, 0.0, size=0.05),
            _make_bbox(0.01, 0.01, size=0.05),
            _make_bbox(0.02, 0.02, size=0.05),
            _make_bbox(0.03, 0.03, size=0.05),
            _make_bbox(0.04, 0.04, size=0.05),
        ]
        track = _make_track(history=history)
        result = ActivityClassifier().classify(track)
        self.assertIsInstance(result, ActivityClassification)

    def test_acceleration_with_variable_speed(self) -> None:
        """Track with accelerating movement should have non-zero mean_acceleration."""
        # Speed increases each frame: 0.01, 0.02, 0.03, ...
        history = [_make_bbox(0.1, 0.5)]
        x = 0.1
        for i in range(1, 8):
            x += 0.005 * i  # increasing speed
            history.append(_make_bbox(x, 0.5))
        track = _make_track(history=history)
        features = ActivityClassifier().extract_features(track)
        self.assertGreater(features.mean_acceleration, 0.0)

    def test_displacement_ratio_zero_for_return_trip(self) -> None:
        """Track that returns to its start should have low displacement ratio."""
        history = []
        # Go right, then come back
        for i in range(10):
            history.append(_make_bbox(0.5 + 0.01 * i, 0.5))
        for i in range(10):
            history.append(_make_bbox(0.59 - 0.01 * i, 0.5))
        track = _make_track(history=history)
        features = ActivityClassifier().extract_features(track)
        # Returns near start -> low displacement ratio
        self.assertLess(features.displacement_ratio, 0.15)


if __name__ == "__main__":
    unittest.main()
