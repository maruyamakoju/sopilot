"""Trajectory prediction and proactive alerting for the Perception Engine.

Predicts future positions of tracked entities using their velocity history,
enabling proactive alerts for zone entries and collisions *before* they
happen.

Architecture:

    TrajectoryPredictor
        - Computes exponentially-weighted velocity from Track.history
        - Projects future BBox positions with decaying confidence
        - Checks predicted positions against zones and other tracks

    ProactiveAlertGenerator
        - Wraps TrajectoryPredictor with configurable alert thresholds
        - Scans all active tracks for upcoming zone entries and collisions
        - Emits EntityEvent objects (ZONE_ENTRY_PREDICTED, COLLISION_PREDICTED)

Design principles:
    - Frozen dataclasses for prediction results (immutable)
    - Exponential confidence decay based on track stability
    - Thread-safe: RLock guards any mutable state
    - Graceful handling of edge cases (no history, zero velocity, single-frame)
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass

from sopilot.perception.types import (
    BBox,
    EntityEvent,
    EntityEventType,
    Track,
    TrackState,
    Zone,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictedPosition:
    """A predicted future position for a tracked entity.

    Attributes:
        frame_offset: Number of frames from now.
        timestamp_offset: Seconds from now.
        bbox: Predicted bounding box at this future time.
        confidence: Prediction confidence (decays with distance).
    """

    frame_offset: int
    timestamp_offset: float
    bbox: BBox
    confidence: float


@dataclass(frozen=True)
class ZoneEntryPrediction:
    """Prediction that a tracked entity will enter a zone.

    Attributes:
        zone_id: ID of the zone that will be entered.
        zone_name: Human-readable name of the zone.
        entity_id: Track ID of the entity.
        estimated_frames: Frames until predicted entry.
        estimated_seconds: Seconds until predicted entry.
        confidence: Prediction confidence.
        predicted_entry_point: (x, y) coordinates where the entity is
            predicted to enter the zone.
    """

    zone_id: str
    zone_name: str
    entity_id: int
    estimated_frames: int
    estimated_seconds: float
    confidence: float
    predicted_entry_point: tuple[float, float]


@dataclass(frozen=True)
class CollisionPrediction:
    """Prediction that two tracked entities will collide (bboxes overlap).

    Attributes:
        entity_a_id: Track ID of the first entity.
        entity_b_id: Track ID of the second entity.
        estimated_frames: Frames until predicted collision.
        estimated_seconds: Seconds until predicted collision.
        confidence: Prediction confidence (minimum of the two tracks).
        collision_point: (x, y) midpoint of the predicted collision.
    """

    entity_a_id: int
    entity_b_id: int
    estimated_frames: int
    estimated_seconds: float
    confidence: float
    collision_point: tuple[float, float]


# ---------------------------------------------------------------------------
# TrajectoryPredictor
# ---------------------------------------------------------------------------


class TrajectoryPredictor:
    """Predicts future positions of tracked entities using velocity history.

    Uses an exponentially-weighted moving average (EWMA) of velocity derived
    from the track's bbox history.  Recent positions are weighted more
    heavily (controlled by ``alpha``).

    Prediction confidence decays exponentially with time.  The decay rate
    (lambda) is inversely proportional to the track's velocity stability:
    stable tracks keep high confidence longer.

    Parameters:
        horizon_frames: How many frames ahead to predict.
        fps: Frame rate for time conversion (frames per second).
        alpha: EWMA smoothing factor for velocity estimation (0 < alpha <= 1).
            Higher values weight recent observations more heavily.
    """

    def __init__(
        self,
        horizon_frames: int = 30,
        fps: float = 1.0,
        alpha: float = 0.3,
    ) -> None:
        if horizon_frames < 1:
            raise ValueError(f"horizon_frames must be >= 1, got {horizon_frames}")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        self._horizon_frames = horizon_frames
        self._fps = fps
        self._alpha = alpha
        self._lock = threading.RLock()

    # -- public API --------------------------------------------------------

    def predict(self, track: Track) -> list[PredictedPosition]:
        """Predict future positions for a tracked entity.

        Uses exponentially-weighted velocity from the track's bbox history.
        Returns one ``PredictedPosition`` for each future frame up to the
        prediction horizon.

        Returns an empty list if the track has no bbox or is in an
        unsuitable state (EXITED/LOST).
        """
        with self._lock:
            if track.bbox is None:
                return []
            if track.state in (TrackState.EXITED, TrackState.LOST):
                return []

            vx, vy = self._estimate_velocity(track)
            base_confidence = track.confidence if track.confidence > 0 else 0.5
            decay_lambda = self._compute_decay_lambda(track)

            predictions: list[PredictedPosition] = []
            cx, cy = track.bbox.center
            w = track.bbox.width
            h = track.bbox.height

            for step in range(1, self._horizon_frames + 1):
                t = step / self._fps
                pred_cx = cx + vx * step
                pred_cy = cy + vy * step

                # Clamp center to [0, 1] (normalized coordinates)
                pred_cx = max(0.0, min(1.0, pred_cx))
                pred_cy = max(0.0, min(1.0, pred_cy))

                pred_bbox = BBox(
                    x1=max(0.0, pred_cx - w / 2),
                    y1=max(0.0, pred_cy - h / 2),
                    x2=min(1.0, pred_cx + w / 2),
                    y2=min(1.0, pred_cy + h / 2),
                )

                conf = base_confidence * math.exp(-decay_lambda * t)
                conf = max(0.0, min(1.0, conf))

                predictions.append(
                    PredictedPosition(
                        frame_offset=step,
                        timestamp_offset=round(t, 6),
                        bbox=pred_bbox,
                        confidence=round(conf, 6),
                    )
                )

            return predictions

    def predict_zone_entry(
        self, track: Track, zones: list[Zone]
    ) -> list[ZoneEntryPrediction]:
        """Predict if and when a track will enter any given zone.

        For each predicted future position, checks whether the bbox center
        falls inside any zone that the entity is not *currently* inside.
        Returns the first predicted entry for each zone.

        Args:
            track: The track to predict for.
            zones: List of zones to check against.

        Returns:
            A list of ``ZoneEntryPrediction`` objects, at most one per zone.
        """
        with self._lock:
            if not zones or track.bbox is None:
                return []

            predictions = self.predict(track)
            if not predictions:
                return []

            # Determine which zones the entity is currently in
            current_center = track.bbox.center
            currently_inside: set[str] = set()
            for zone in zones:
                if zone.contains_point(current_center[0], current_center[1]):
                    currently_inside.add(zone.zone_id)

            results: list[ZoneEntryPrediction] = []
            found_zones: set[str] = set()

            for pred in predictions:
                pred_cx, pred_cy = pred.bbox.center
                for zone in zones:
                    if zone.zone_id in found_zones:
                        continue
                    if zone.zone_id in currently_inside:
                        continue
                    if zone.contains_point(pred_cx, pred_cy):
                        results.append(
                            ZoneEntryPrediction(
                                zone_id=zone.zone_id,
                                zone_name=zone.name,
                                entity_id=track.track_id,
                                estimated_frames=pred.frame_offset,
                                estimated_seconds=pred.timestamp_offset,
                                confidence=pred.confidence,
                                predicted_entry_point=(
                                    round(pred_cx, 6),
                                    round(pred_cy, 6),
                                ),
                            )
                        )
                        found_zones.add(zone.zone_id)

            return results

    def predict_collision(
        self, track_a: Track, track_b: Track
    ) -> CollisionPrediction | None:
        """Predict if two tracks will collide (bboxes will overlap).

        Checks whether predicted bounding boxes of the two tracks overlap
        (IoU > 0) at any future frame within the prediction horizon.
        Returns the first predicted collision, or None if no collision is
        predicted.

        Args:
            track_a: First track.
            track_b: Second track.

        Returns:
            A ``CollisionPrediction`` if overlap is predicted, else None.
        """
        with self._lock:
            if track_a.bbox is None or track_b.bbox is None:
                return None

            preds_a = self.predict(track_a)
            preds_b = self.predict(track_b)

            if not preds_a or not preds_b:
                return None

            # Check if they already overlap -- if so, not a "future" collision
            if track_a.bbox.iou(track_b.bbox) > 0:
                return None

            for pa, pb in zip(preds_a, preds_b):
                if pa.bbox.iou(pb.bbox) > 0:
                    # Collision predicted
                    cx_a, cy_a = pa.bbox.center
                    cx_b, cy_b = pb.bbox.center
                    collision_point = (
                        round((cx_a + cx_b) / 2, 6),
                        round((cy_a + cy_b) / 2, 6),
                    )
                    confidence = min(pa.confidence, pb.confidence)
                    return CollisionPrediction(
                        entity_a_id=track_a.track_id,
                        entity_b_id=track_b.track_id,
                        estimated_frames=pa.frame_offset,
                        estimated_seconds=pa.timestamp_offset,
                        confidence=round(confidence, 6),
                        collision_point=collision_point,
                    )

            return None

    # -- internal helpers --------------------------------------------------

    def _estimate_velocity(self, track: Track) -> tuple[float, float]:
        """Estimate velocity using EWMA over the track's bbox history.

        Computes per-frame displacement between consecutive history entries
        and applies exponential weighting so that recent displacements
        dominate.

        Falls back to the track's stored velocity if history has fewer
        than 2 entries.

        Returns:
            (vx, vy) in normalized coordinates per frame.
        """
        history = track.history
        if len(history) < 2:
            # Insufficient history -- use the track's stored velocity
            return track.velocity

        alpha = self._alpha
        vx = 0.0
        vy = 0.0
        weight_sum = 0.0

        for i in range(1, len(history)):
            cx_prev, cy_prev = history[i - 1].center
            cx_curr, cy_curr = history[i].center
            dx = cx_curr - cx_prev
            dy = cy_curr - cy_prev

            # Weight: more recent observations get higher weight.
            # The last pair (i = len-1) gets the highest weight.
            # w = alpha * (1 - alpha)^(len-1-i)  but we just need relative
            # weights, so we use (1 - alpha)^(distance from end).
            distance_from_end = len(history) - 1 - i
            w = alpha * ((1.0 - alpha) ** distance_from_end)

            vx += w * dx
            vy += w * dy
            weight_sum += w

        if weight_sum > 1e-12:
            vx /= weight_sum
            vy /= weight_sum

        return (vx, vy)

    def _compute_decay_lambda(self, track: Track) -> float:
        """Compute the confidence decay rate based on track stability.

        Stable tracks (low velocity variance) decay slowly; erratic tracks
        decay quickly.  This ensures predictions for smoothly-moving
        entities stay confident further into the future.

        Returns:
            Lambda parameter for exponential decay: ``conf * exp(-lambda * t)``.
            Typical range: [0.05, 2.0].
        """
        history = track.history
        if len(history) < 3:
            # Not enough data to measure stability -- use fast decay
            return 1.0

        # Compute velocity samples
        velocities_x: list[float] = []
        velocities_y: list[float] = []
        for i in range(1, len(history)):
            cx_prev, cy_prev = history[i - 1].center
            cx_curr, cy_curr = history[i].center
            velocities_x.append(cx_curr - cx_prev)
            velocities_y.append(cy_curr - cy_prev)

        # Variance of velocity components
        if len(velocities_x) < 2:
            return 1.0

        mean_vx = sum(velocities_x) / len(velocities_x)
        mean_vy = sum(velocities_y) / len(velocities_y)
        var_vx = sum((v - mean_vx) ** 2 for v in velocities_x) / len(velocities_x)
        var_vy = sum((v - mean_vy) ** 2 for v in velocities_y) / len(velocities_y)
        total_var = var_vx + var_vy

        # Map variance to lambda: low variance -> small lambda (slow decay),
        # high variance -> large lambda (fast decay).
        # Lambda = base + scale * sqrt(variance)
        # With base=0.05 and scale=50, a perfectly stable track (var=0)
        # gets lambda=0.05, a jittery track (var=0.001) gets ~1.6.
        base_lambda = 0.05
        scale = 50.0
        decay = base_lambda + scale * math.sqrt(total_var)

        return min(decay, 2.0)


# ---------------------------------------------------------------------------
# ProactiveAlertGenerator
# ---------------------------------------------------------------------------


class ProactiveAlertGenerator:
    """Generates proactive alerts from trajectory predictions.

    Scans all active tracks for upcoming zone entries and potential
    collisions, producing ``EntityEvent`` objects with event types
    ``ZONE_ENTRY_PREDICTED`` and ``COLLISION_PREDICTED``.

    Parameters:
        predictor: The trajectory predictor to use.
        zone_alert_seconds: Alert this many seconds before a predicted
            zone entry (default 5.0).
        collision_alert_seconds: Alert this many seconds before a predicted
            collision (default 3.0).
    """

    def __init__(
        self,
        predictor: TrajectoryPredictor,
        zone_alert_seconds: float = 5.0,
        collision_alert_seconds: float = 3.0,
    ) -> None:
        self._predictor = predictor
        self._zone_alert_seconds = zone_alert_seconds
        self._collision_alert_seconds = collision_alert_seconds
        self._lock = threading.RLock()

    def generate_alerts(
        self,
        tracks: dict[int, Track],
        zones: list[Zone],
        current_frame: int = 0,
        current_timestamp: float = 0.0,
    ) -> list[EntityEvent]:
        """Check all active tracks for upcoming zone entries and collisions.

        Args:
            tracks: Mapping of track_id to Track for all active tracks.
            zones: List of zones to check for predicted entry.
            current_frame: Current frame number (for event metadata).
            current_timestamp: Current timestamp (for event metadata).

        Returns:
            List of ``EntityEvent`` objects for predicted zone entries
            and collisions.  Each event includes details such as
            ``time_to_entry``, ``predicted_entry_point``, etc.
        """
        with self._lock:
            events: list[EntityEvent] = []

            active_tracks = [
                t for t in tracks.values()
                if t.state not in (TrackState.EXITED, TrackState.LOST)
                and t.bbox is not None
            ]

            # -- Zone entry predictions --
            for track in active_tracks:
                zone_preds = self._predictor.predict_zone_entry(track, zones)
                for zp in zone_preds:
                    if zp.estimated_seconds <= self._zone_alert_seconds:
                        events.append(
                            EntityEvent(
                                event_type=EntityEventType.ZONE_ENTRY_PREDICTED,
                                entity_id=zp.entity_id,
                                timestamp=current_timestamp,
                                frame_number=current_frame,
                                details={
                                    "zone_id": zp.zone_id,
                                    "zone_name": zp.zone_name,
                                    "time_to_entry": round(zp.estimated_seconds, 3),
                                    "frames_to_entry": zp.estimated_frames,
                                    "predicted_entry_point": list(zp.predicted_entry_point),
                                },
                                confidence=zp.confidence,
                            )
                        )
                        logger.info(
                            "PREDICTED zone entry: entity %d -> zone '%s' in %.1fs "
                            "(confidence=%.2f)",
                            zp.entity_id,
                            zp.zone_name,
                            zp.estimated_seconds,
                            zp.confidence,
                        )

            # -- Collision predictions --
            for i, track_a in enumerate(active_tracks):
                for track_b in active_tracks[i + 1:]:
                    collision = self._predictor.predict_collision(track_a, track_b)
                    if collision is None:
                        continue
                    if collision.estimated_seconds <= self._collision_alert_seconds:
                        events.append(
                            EntityEvent(
                                event_type=EntityEventType.COLLISION_PREDICTED,
                                entity_id=collision.entity_a_id,
                                timestamp=current_timestamp,
                                frame_number=current_frame,
                                details={
                                    "entity_a_id": collision.entity_a_id,
                                    "entity_b_id": collision.entity_b_id,
                                    "time_to_collision": round(
                                        collision.estimated_seconds, 3
                                    ),
                                    "frames_to_collision": collision.estimated_frames,
                                    "collision_point": list(collision.collision_point),
                                },
                                confidence=collision.confidence,
                            )
                        )
                        logger.info(
                            "PREDICTED collision: entity %d <-> entity %d in %.1fs "
                            "(confidence=%.2f)",
                            collision.entity_a_id,
                            collision.entity_b_id,
                            collision.estimated_seconds,
                            collision.confidence,
                        )

            return events
