"""Activity Recognition from Trajectory Patterns.

Classifies the movement activity of tracked entities using rule-based
analysis of motion features extracted from bbox history.  Activities
range from simple (stationary, walking, running) to behavioral
patterns (loitering, erratic movement).

Pipeline:
    Track.history  -->  extract_features()  -->  classify()  -->  ActivityClassification

The ActivityMonitor integrates with the world model to generate
STATE_CHANGED events when an entity's activity transitions (e.g.,
walking -> running, stationary -> loitering).

Design decisions:
    - Rule-based classification (no ML) for deterministic, explainable results
    - All coordinates are normalized [0, 1] (resolution-independent)
    - Confidence reflects distance from decision boundaries
    - Frozen dataclasses for immutable result snapshots
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass

from sopilot.perception.types import (
    BBox,
    EntityEvent,
    EntityEventType,
    Track,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ActivityType enum
# ---------------------------------------------------------------------------


class ActivityType(enum.Enum):
    """Recognized movement activity types."""

    STATIONARY = "stationary"       # not moving (v < threshold)
    WALKING = "walking"             # slow, steady movement
    RUNNING = "running"             # fast movement
    LOITERING = "loitering"         # staying in small area for extended time
    ERRATIC = "erratic"             # frequent direction changes
    APPROACHING = "approaching"     # moving toward a specific point/zone
    DEPARTING = "departing"         # moving away from a specific point/zone
    UNKNOWN = "unknown"             # insufficient data


# ---------------------------------------------------------------------------
# ActivityFeatures — motion features extracted from a Track
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivityFeatures:
    """Motion features extracted from a Track's bbox history.

    All spatial measurements are in normalized coordinates [0, 1].
    Speed and acceleration are per-frame quantities.
    """

    mean_speed: float               # average displacement per frame
    max_speed: float                # peak speed in any frame
    speed_variance: float           # variance of per-frame speeds
    mean_acceleration: float        # average absolute rate of speed change
    direction_change_rate: float    # fraction of frames with >45 degree turn
    displacement_ratio: float       # net displacement / total path length (1.0=straight, 0=circle)
    bounding_area: float            # area of bounding rectangle enclosing all positions
    duration_frames: int            # number of frames of history


# ---------------------------------------------------------------------------
# ActivityClassification — result of classifying a track
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivityClassification:
    """Result of classifying a tracked entity's activity.

    Contains the primary classification, confidence, the underlying
    features, and an optional secondary (runner-up) classification.
    """

    activity: ActivityType
    confidence: float
    features: ActivityFeatures
    secondary_activity: ActivityType | None = None
    secondary_confidence: float = 0.0


# ---------------------------------------------------------------------------
# ActivityClassifier — rule-based activity classification
# ---------------------------------------------------------------------------


class ActivityClassifier:
    """Rule-based activity classifier operating on Track bbox histories.

    Extracts motion features (speed, acceleration, direction changes,
    bounding area) and applies threshold-based rules to classify the
    current activity of each tracked entity.

    All thresholds operate on normalized coordinates [0, 1] so the
    classifier is resolution-independent.

    Args:
        stationary_speed_threshold: Mean speed below which entity is stationary.
        walking_speed_range: (min, max) mean speed for walking classification.
        running_speed_threshold: Mean speed above which entity is running.
        loiter_area_threshold: Bounding area below which entity is in a small area.
        loiter_min_frames: Minimum frames to classify as loitering (vs stationary).
        erratic_direction_change_rate: Direction change rate above which movement
            is considered erratic.
        min_history_frames: Minimum bbox history length for classification.
            Tracks shorter than this return UNKNOWN.
    """

    def __init__(
        self,
        stationary_speed_threshold: float = 0.005,
        walking_speed_range: tuple[float, float] = (0.005, 0.03),
        running_speed_threshold: float = 0.03,
        loiter_area_threshold: float = 0.01,
        loiter_min_frames: int = 30,
        erratic_direction_change_rate: float = 0.5,
        min_history_frames: int = 5,
    ) -> None:
        self._stationary_threshold = stationary_speed_threshold
        self._walking_min = walking_speed_range[0]
        self._walking_max = walking_speed_range[1]
        self._running_threshold = running_speed_threshold
        self._loiter_area = loiter_area_threshold
        self._loiter_min_frames = loiter_min_frames
        self._erratic_rate = erratic_direction_change_rate
        self._min_history = min_history_frames

    # -- public API --------------------------------------------------------

    def extract_features(self, track: Track) -> ActivityFeatures:
        """Extract motion features from a track's bbox history.

        Computes per-frame velocities from consecutive bbox centers,
        then derives aggregate statistics (mean/max speed, acceleration,
        direction change rate, displacement ratio, bounding area).

        Args:
            track: A Track with bbox history to analyze.

        Returns:
            An ActivityFeatures snapshot.  If the track has fewer than
            2 history entries, all numeric fields are 0.0.
        """
        history = track.history
        n = len(history)

        if n < 2:
            return ActivityFeatures(
                mean_speed=0.0,
                max_speed=0.0,
                speed_variance=0.0,
                mean_acceleration=0.0,
                direction_change_rate=0.0,
                displacement_ratio=0.0,
                bounding_area=0.0,
                duration_frames=n,
            )

        # Extract centers
        centers = [bbox.center for bbox in history]

        # Per-frame velocities (displacement between consecutive centers)
        speeds: list[float] = []
        directions: list[float] = []
        for i in range(1, n):
            dx = centers[i][0] - centers[i - 1][0]
            dy = centers[i][1] - centers[i - 1][1]
            speed = math.hypot(dx, dy)
            speeds.append(speed)
            directions.append(math.atan2(dy, dx))

        # Speed statistics
        mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0
        speed_variance = (
            sum((s - mean_speed) ** 2 for s in speeds) / len(speeds)
            if speeds
            else 0.0
        )

        # Acceleration (absolute change in speed between consecutive frames)
        accelerations: list[float] = []
        for i in range(1, len(speeds)):
            accelerations.append(abs(speeds[i] - speeds[i - 1]))
        mean_acceleration = (
            sum(accelerations) / len(accelerations) if accelerations else 0.0
        )

        # Direction change rate (fraction of frames with >45 degree change)
        direction_changes = 0
        for i in range(1, len(directions)):
            angle_diff = abs(directions[i] - directions[i - 1])
            # Normalize to [0, pi]
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            if angle_diff > math.pi / 4:  # 45 degrees
                direction_changes += 1
        direction_change_rate = (
            direction_changes / (len(directions) - 1)
            if len(directions) > 1
            else 0.0
        )

        # Displacement ratio: net displacement / total path length
        total_path = sum(speeds)
        net_displacement = math.hypot(
            centers[-1][0] - centers[0][0],
            centers[-1][1] - centers[0][1],
        )
        displacement_ratio = (
            net_displacement / total_path if total_path > 1e-10 else 0.0
        )

        # Bounding area: axis-aligned bounding rectangle of all positions
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        bounding_area = (max(xs) - min(xs)) * (max(ys) - min(ys))

        return ActivityFeatures(
            mean_speed=mean_speed,
            max_speed=max_speed,
            speed_variance=speed_variance,
            mean_acceleration=mean_acceleration,
            direction_change_rate=direction_change_rate,
            displacement_ratio=displacement_ratio,
            bounding_area=bounding_area,
            duration_frames=n,
        )

    def classify(self, track: Track) -> ActivityClassification:
        """Classify the current activity of a tracked entity.

        Uses threshold-based rules on extracted motion features.
        Classification priority:
            1. Insufficient data -> UNKNOWN
            2. Slow + small area + long duration -> LOITERING
            3. Slow + small area + short duration -> STATIONARY
            4. Frequent direction changes -> ERRATIC
            5. Fast movement -> RUNNING
            6. Moderate movement -> WALKING
            7. Fallback -> UNKNOWN

        Confidence reflects how far the features are from the decision
        boundary thresholds (farther = more confident).

        Args:
            track: A Track with bbox history.

        Returns:
            An ActivityClassification with primary and secondary activity.
        """
        features = self.extract_features(track)

        # Insufficient data
        if features.duration_frames < self._min_history:
            return ActivityClassification(
                activity=ActivityType.UNKNOWN,
                confidence=0.0,
                features=features,
            )

        # Score each candidate activity
        candidates = self._score_candidates(features)

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        primary_activity, primary_confidence = candidates[0]
        secondary_activity: ActivityType | None = None
        secondary_confidence = 0.0
        if len(candidates) > 1 and candidates[1][1] > 0.0:
            secondary_activity = candidates[1][0]
            secondary_confidence = candidates[1][1]

        return ActivityClassification(
            activity=primary_activity,
            confidence=primary_confidence,
            features=features,
            secondary_activity=secondary_activity,
            secondary_confidence=secondary_confidence,
        )

    def classify_batch(
        self, tracks: dict[int, Track]
    ) -> dict[int, ActivityClassification]:
        """Classify activities for all tracks.

        Args:
            tracks: Mapping of track_id to Track.

        Returns:
            Mapping of track_id to ActivityClassification.
        """
        return {tid: self.classify(track) for tid, track in tracks.items()}

    # -- internal ----------------------------------------------------------

    def _score_candidates(
        self, features: ActivityFeatures
    ) -> list[tuple[ActivityType, float]]:
        """Score each activity type based on how well features match.

        Returns a list of (ActivityType, confidence) pairs.  Confidence
        is in [0, 1] and reflects how strongly the features support
        each activity.
        """
        scores: list[tuple[ActivityType, float]] = []

        ms = features.mean_speed
        ba = features.bounding_area
        dcr = features.direction_change_rate
        dur = features.duration_frames

        # --- LOITERING: slow + small area + many frames ---
        # Loitering is a more specific refinement of stationary: the entity
        # has been still in a small area for an extended period.  When the
        # loitering condition is met, it should outrank plain STATIONARY.
        is_loitering = False
        if ms < self._stationary_threshold and ba < self._loiter_area:
            if dur >= self._loiter_min_frames:
                is_loitering = True
                # Confidence: higher when duration far exceeds minimum
                duration_factor = min(1.0, dur / (self._loiter_min_frames * 2))
                # Also factor in how slow — slower = more confident
                speed_factor = 1.0 - (ms / self._stationary_threshold) if self._stationary_threshold > 0 else 1.0
                conf = 0.7 + 0.3 * min(duration_factor, speed_factor)
                scores.append((ActivityType.LOITERING, min(1.0, conf)))

        # --- STATIONARY: slow + small area ---
        if ms < self._stationary_threshold:
            # Confidence: higher when speed is much below threshold
            speed_ratio = ms / self._stationary_threshold if self._stationary_threshold > 0 else 0.0
            # When loitering is also detected, cap stationary confidence
            # below loitering so loitering wins as primary classification.
            if is_loitering:
                conf = 0.4 + 0.25 * (1.0 - speed_ratio)
            else:
                conf = 0.5 + 0.5 * (1.0 - speed_ratio)
            scores.append((ActivityType.STATIONARY, min(1.0, conf)))

        # --- ERRATIC: frequent direction changes ---
        # Only consider erratic when the entity is actually moving
        # (above stationary threshold).  Micro-jitter in a stationary
        # entity should not be classified as erratic behavior.
        if dcr > self._erratic_rate and ms >= self._stationary_threshold:
            # Confidence: higher when rate far exceeds threshold
            excess = (dcr - self._erratic_rate) / (1.0 - self._erratic_rate) if self._erratic_rate < 1.0 else 0.0
            conf = 0.5 + 0.5 * min(1.0, excess)
            scores.append((ActivityType.ERRATIC, min(1.0, conf)))

        # --- RUNNING: fast movement ---
        if ms > self._running_threshold:
            # Confidence: higher when speed far exceeds threshold
            excess = (ms - self._running_threshold) / self._running_threshold if self._running_threshold > 0 else 0.0
            conf = 0.5 + 0.5 * min(1.0, excess)
            scores.append((ActivityType.RUNNING, min(1.0, conf)))

        # --- WALKING: moderate movement ---
        if self._walking_min <= ms <= self._walking_max:
            # Confidence: highest at midpoint of walking range
            mid = (self._walking_min + self._walking_max) / 2
            half_range = (self._walking_max - self._walking_min) / 2
            if half_range > 0:
                distance_from_mid = abs(ms - mid) / half_range
                conf = 0.5 + 0.5 * (1.0 - distance_from_mid)
            else:
                conf = 0.5
            scores.append((ActivityType.WALKING, min(1.0, conf)))

        # If no candidate matched, add UNKNOWN
        if not scores:
            scores.append((ActivityType.UNKNOWN, 0.3))

        return scores


# ---------------------------------------------------------------------------
# ActivityMonitor — integrates with world model event system
# ---------------------------------------------------------------------------


class ActivityMonitor:
    """Monitors track activities and generates events on transitions.

    Wraps an ActivityClassifier and tracks per-entity activity state
    across frames.  When an entity's classified activity changes (e.g.,
    walking -> running), a STATE_CHANGED event is generated.

    Args:
        classifier: The ActivityClassifier to use.
        alert_on_activities: Activities that should generate events.
            Defaults to {LOITERING, ERRATIC, RUNNING} if not specified.
    """

    def __init__(
        self,
        classifier: ActivityClassifier,
        alert_on_activities: set[ActivityType] | None = None,
    ) -> None:
        self._classifier = classifier
        self._alert_activities = alert_on_activities or {
            ActivityType.LOITERING,
            ActivityType.ERRATIC,
            ActivityType.RUNNING,
        }
        self._previous_activities: dict[int, ActivityType] = {}

    def update(
        self,
        tracks: dict[int, Track],
        timestamp: float,
        frame_number: int,
    ) -> list[EntityEvent]:
        """Classify all tracks and generate events for notable activities.

        Generates STATE_CHANGED events when an entity's activity type
        changes (e.g., walking -> running, stationary -> loitering).
        Events include the old and new activity types and the
        classification confidence in details.

        Also cleans up state for tracks that are no longer present.

        Args:
            tracks: Current active tracks (track_id -> Track).
            timestamp: Current frame timestamp.
            frame_number: Current frame number.

        Returns:
            List of EntityEvent objects for activity transitions.
        """
        events: list[EntityEvent] = []
        current_activities: dict[int, ActivityType] = {}

        classifications = self._classifier.classify_batch(tracks)

        for tid, classification in classifications.items():
            activity = classification.activity
            current_activities[tid] = activity

            prev = self._previous_activities.get(tid)
            if prev is not None and prev != activity:
                # Activity transition detected
                is_alert = (
                    activity in self._alert_activities
                    or prev in self._alert_activities
                )
                events.append(
                    EntityEvent(
                        event_type=EntityEventType.STATE_CHANGED,
                        entity_id=tid,
                        timestamp=timestamp,
                        frame_number=frame_number,
                        details={
                            "change_type": "activity",
                            "old_activity": prev.value,
                            "new_activity": activity.value,
                            "confidence": round(classification.confidence, 3),
                            "alert": is_alert,
                        },
                        confidence=classification.confidence,
                    )
                )
                logger.info(
                    "Activity transition: entity %d %s -> %s "
                    "(confidence=%.2f, alert=%s) at t=%.3f frame=%d",
                    tid,
                    prev.value,
                    activity.value,
                    classification.confidence,
                    is_alert,
                    timestamp,
                    frame_number,
                )

        # Clean up state for tracks that disappeared
        departed = set(self._previous_activities.keys()) - set(tracks.keys())
        for tid in departed:
            del self._previous_activities[tid]

        self._previous_activities = current_activities
        return events
