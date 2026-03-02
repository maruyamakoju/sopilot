"""Multi-object tracking for the Perception Engine (ByteTrack-inspired).

Implements persistent identity tracking across frames using:

    - A linear **Kalman filter** for motion prediction (pure numpy)
    - **Two-stage IoU assignment** (ByteTrack's key insight): first match
      high-confidence detections, then salvage unmatched tracks with
      low-confidence detections
    - **Hungarian algorithm** for optimal assignment (scipy if available,
      greedy fallback otherwise)
    - **Track lifecycle management**: TENTATIVE → ACTIVE → OCCLUDED → LOST → EXITED

No external tracking libraries are required — everything is built from
scratch on top of numpy.

Example usage::

    tracker = MultiObjectTracker(config)
    for frame_id, detections in enumerate(detection_stream):
        tracks = tracker.update(detections, frame_id)
        for t in tracks:
            print(f"Track {t.track_id}: {t.label} @ {t.bbox} [{t.state.value}]")
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np

from sopilot.perception.types import BBox, Detection, PerceptionConfig, Track, TrackState

logger = logging.getLogger(__name__)


# ── Kalman Filter ─────────────────────────────────────────────────────────────


class _KalmanBoxTracker:
    """Linear Kalman filter for a single bounding box with constant-velocity model.

    State vector (8-dimensional)::

        [cx, cy, w, h, vx, vy, vw, vh]

    where ``(cx, cy)`` is the box center, ``(w, h)`` the size, and
    ``(vx, vy, vw, vh)`` are their first-order time derivatives (velocities).

    All coordinates are in the normalized [0, 1] space inherited from
    :class:`BBox`.  The filter uses a standard predict/update cycle.
    """

    # Measurement noise (diagonal).  These values are tuned for normalized
    # coordinates where typical inter-frame displacement is ~0.01–0.05.
    _STD_WEIGHT_POSITION = 1.0 / 20.0
    _STD_WEIGHT_VELOCITY = 1.0 / 160.0

    def __init__(self, bbox: BBox) -> None:
        cx, cy = bbox.center
        w, h = bbox.width, bbox.height

        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # State transition matrix (constant velocity model).
        # x_{t+1} = F @ x_t
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # position += velocity * dt (dt=1)

        # Measurement matrix: we observe [cx, cy, w, h] directly.
        self.H = np.eye(4, 8, dtype=np.float64)

        # Covariance matrices.
        # Initial state uncertainty — large for velocities (unknown).
        self.P = np.eye(8, dtype=np.float64)
        self.P[4:, 4:] *= 100.0  # high uncertainty on initial velocities
        self.P[:4, :4] *= 10.0   # moderate uncertainty on initial position

        # Process noise.
        self.Q = np.eye(8, dtype=np.float64)
        pos_std = self._STD_WEIGHT_POSITION
        vel_std = self._STD_WEIGHT_VELOCITY
        self.Q[0, 0] = pos_std ** 2
        self.Q[1, 1] = pos_std ** 2
        self.Q[2, 2] = pos_std ** 2
        self.Q[3, 3] = pos_std ** 2
        self.Q[4, 4] = vel_std ** 2
        self.Q[5, 5] = vel_std ** 2
        self.Q[6, 6] = vel_std ** 2
        self.Q[7, 7] = vel_std ** 2

        # Measurement noise.
        self.R = np.eye(4, dtype=np.float64) * (pos_std ** 2)

    def predict(self) -> BBox:
        """Propagate state forward one time step and return the predicted bbox."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Clamp size to be non-negative.
        self.x[2] = max(self.x[2], 1e-6)
        self.x[3] = max(self.x[3], 1e-6)

        return self._state_to_bbox()

    def update(self, bbox: BBox) -> BBox:
        """Correct the state with a matched detection measurement."""
        cx, cy = bbox.center
        z = np.array([cx, cy, bbox.width, bbox.height], dtype=np.float64)

        # Innovation (measurement residual).
        y = z - self.H @ self.x

        # Innovation covariance.
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain.
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular matrix — skip update and return prediction.
            logger.warning("Kalman update: singular innovation covariance, skipping")
            return self._state_to_bbox()

        # State update.
        self.x = self.x + K @ y
        I_KH = np.eye(8) - K @ self.H
        # Joseph form for numerical stability.
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Clamp.
        self.x[2] = max(self.x[2], 1e-6)
        self.x[3] = max(self.x[3], 1e-6)

        return self._state_to_bbox()

    def _state_to_bbox(self) -> BBox:
        """Convert the internal state to a BBox (clamped to [0, 1])."""
        cx, cy, w, h = self.x[:4]
        return BBox(
            x1=float(np.clip(cx - w / 2, 0.0, 1.0)),
            y1=float(np.clip(cy - h / 2, 0.0, 1.0)),
            x2=float(np.clip(cx + w / 2, 0.0, 1.0)),
            y2=float(np.clip(cy + h / 2, 0.0, 1.0)),
        )

    @property
    def velocity(self) -> tuple[float, float]:
        """Current velocity estimate (vx, vy) in normalized coords/frame."""
        return (float(self.x[4]), float(self.x[5]))


# ── Assignment Helpers ────────────────────────────────────────────────────────


def _compute_iou_matrix(
    bboxes_a: list[BBox], bboxes_b: list[BBox]
) -> np.ndarray:
    """Compute pairwise IoU between two lists of bounding boxes.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(len(bboxes_a), len(bboxes_b))`` with IoU values.
    """
    n, m = len(bboxes_a), len(bboxes_b)
    iou_mat = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            iou_mat[i, j] = bboxes_a[i].iou(bboxes_b[j])
    return iou_mat


def _hungarian_assignment(
    cost_matrix: np.ndarray,
) -> list[tuple[int, int]]:
    """Optimal assignment using scipy's Hungarian algorithm.

    Falls back to greedy assignment if scipy is not available.

    Parameters
    ----------
    cost_matrix:
        Cost matrix (lower is better) of shape ``(N, M)``.

    Returns
    -------
    list[tuple[int, int]]
        Matched ``(row, col)`` pairs.
    """
    if cost_matrix.size == 0:
        return []

    try:
        from scipy.optimize import linear_sum_assignment

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        return list(zip(row_indices.tolist(), col_indices.tolist()))
    except ImportError:
        logger.debug("scipy not available — using greedy assignment fallback")
        return _greedy_assignment(cost_matrix)


def _greedy_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Greedy assignment: repeatedly pick the lowest-cost unmatched pair.

    This is O(N*M*min(N,M)) but perfectly adequate for the typical number
    of tracks/detections (< 100) and serves as a fallback when scipy is
    not installed.
    """
    if cost_matrix.size == 0:
        return []

    matches: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    n_matches = min(cost_matrix.shape[0], cost_matrix.shape[1])

    # Flatten and argsort all entries.
    flat_indices = np.argsort(cost_matrix, axis=None)

    for flat_idx in flat_indices:
        if len(matches) >= n_matches:
            break
        row = int(flat_idx // cost_matrix.shape[1])
        col = int(flat_idx % cost_matrix.shape[1])
        if row not in used_rows and col not in used_cols:
            matches.append((row, col))
            used_rows.add(row)
            used_cols.add(col)

    return matches


def _match_detections_to_tracks(
    track_bboxes: list[BBox],
    det_bboxes: list[BBox],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match detections to predicted track positions using IoU + Hungarian.

    Parameters
    ----------
    track_bboxes:
        Predicted bounding boxes for existing tracks.
    det_bboxes:
        Bounding boxes of new detections.
    iou_threshold:
        Minimum IoU for a valid match.

    Returns
    -------
    matches:
        List of ``(track_idx, det_idx)`` pairs.
    unmatched_tracks:
        Indices into *track_bboxes* that were not matched.
    unmatched_dets:
        Indices into *det_bboxes* that were not matched.
    """
    if not track_bboxes or not det_bboxes:
        return (
            [],
            list(range(len(track_bboxes))),
            list(range(len(det_bboxes))),
        )

    iou_mat = _compute_iou_matrix(track_bboxes, det_bboxes)

    # Convert IoU (higher = better) to cost (lower = better).
    cost_matrix = 1.0 - iou_mat

    raw_matches = _hungarian_assignment(cost_matrix)

    # Filter matches below the IoU threshold.
    matches: list[tuple[int, int]] = []
    unmatched_tracks = set(range(len(track_bboxes)))
    unmatched_dets = set(range(len(det_bboxes)))

    for t_idx, d_idx in raw_matches:
        if iou_mat[t_idx, d_idx] >= iou_threshold:
            matches.append((t_idx, d_idx))
            unmatched_tracks.discard(t_idx)
            unmatched_dets.discard(d_idx)

    return matches, sorted(unmatched_tracks), sorted(unmatched_dets)


# ── Internal Track State ──────────────────────────────────────────────────────


class _InternalTrack:
    """Augmented internal representation of a track.

    Holds the Kalman filter and label-vote history that aren't part of the
    public :class:`Track` dataclass.
    """

    __slots__ = (
        "track_id",
        "kf",
        "state",
        "label_votes",
        "confidence",
        "first_frame",
        "last_frame",
        "age",
        "hits",
        "misses",
        "history",
        "attributes",
        "_history_maxlen",
    )

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        frame_id: int,
        history_maxlen: int = 30,
    ) -> None:
        self.track_id = track_id
        self.kf = _KalmanBoxTracker(detection.bbox)
        self.state = TrackState.TENTATIVE
        self.label_votes: Counter[str] = Counter({detection.label: 1})
        self.confidence = detection.confidence
        self.first_frame = frame_id
        self.last_frame = frame_id
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.history: list[BBox] = [detection.bbox]
        self.attributes: dict[str, Any] = dict(detection.attributes)
        self._history_maxlen = history_maxlen

    # ── Properties ────────────────────────────────────────────────────

    @property
    def label(self) -> str:
        """Majority-vote label from all matched detections."""
        if not self.label_votes:
            return "unknown"
        return self.label_votes.most_common(1)[0][0]

    @property
    def bbox(self) -> BBox:
        return self.kf._state_to_bbox()

    @property
    def velocity(self) -> tuple[float, float]:
        return self.kf.velocity

    # ── Lifecycle ─────────────────────────────────────────────────────

    def predict(self) -> BBox:
        """Predict the next position (call once per frame before matching)."""
        self.age += 1
        return self.kf.predict()

    def update_with_detection(self, detection: Detection, frame_id: int) -> None:
        """Update the track with a matched detection."""
        self.kf.update(detection.bbox)
        self.label_votes[detection.label] += 1
        self.confidence = detection.confidence
        self.last_frame = frame_id
        self.hits += 1
        self.misses = 0

        # Maintain rolling history.
        self.history.append(detection.bbox)
        if len(self.history) > self._history_maxlen:
            self.history.pop(0)

        # Merge detection attributes (latest wins for overlapping keys).
        self.attributes.update(detection.attributes)

    def mark_missed(self) -> None:
        """Record a frame where this track was not matched to any detection."""
        self.misses += 1

    def to_public_track(self) -> Track:
        """Convert to the public :class:`Track` dataclass."""
        return Track(
            track_id=self.track_id,
            label=self.label,
            state=self.state,
            bbox=self.bbox,
            velocity=self.velocity,
            confidence=self.confidence,
            first_frame=self.first_frame,
            last_frame=self.last_frame,
            age=self.age,
            hits=self.hits,
            misses=self.misses,
            attributes=dict(self.attributes),
            history=list(self.history),
        )


# ── Multi-Object Tracker ─────────────────────────────────────────────────────


class MultiObjectTracker:
    """ByteTrack-inspired multi-object tracker.

    The tracker maintains a set of :class:`Track` objects, each with a
    persistent identity, across sequential frame updates.  Key design
    features:

    1. **Two-stage association** — First match high-confidence detections
       to tracks, then attempt to recover unmatched tracks using remaining
       low-confidence detections.  This is ByteTrack's core insight and
       dramatically reduces identity switches during occlusion.

    2. **Kalman-filter motion model** — Each track runs an independent
       linear Kalman filter in normalized coordinate space, providing
       velocity-aware position prediction.

    3. **Label consistency** — Track labels are determined by majority vote
       across all matched detection labels, preventing transient mis-labels
       from corrupting track identity.

    4. **Configurable lifecycle** — Tracks go through well-defined states
       (TENTATIVE → ACTIVE → OCCLUDED → LOST → EXITED) controlled by
       :class:`PerceptionConfig` thresholds.

    Parameters
    ----------
    config:
        Perception configuration.  If *None*, default thresholds are used.
    """

    def __init__(self, config: PerceptionConfig | None = None) -> None:
        self._config = config or PerceptionConfig()
        self._tracks: dict[int, _InternalTrack] = {}
        self._next_id = 1
        self._frame_count = 0
        logger.info(
            "MultiObjectTracker initialized (high_thr=%.2f, low_thr=%.2f, "
            "max_age=%d, min_hits=%d)",
            self._config.track_high_threshold,
            self._config.track_low_threshold,
            self._config.track_max_age,
            self._config.track_min_hits,
        )

    # ── Public API ────────────────────────────────────────────────────

    def update(self, detections: list[Detection], frame_id: int) -> list[Track]:
        """Process detections for a new frame and return all active tracks.

        This is the main entry point called once per frame.  The algorithm:

        1. Predict new positions for all existing tracks.
        2. Split detections into high-confidence and low-confidence sets.
        3. Match high-confidence detections to all tracks (first association).
        4. Match low-confidence detections to remaining unmatched tracks
           (second association — ByteTrack recovery).
        5. Create new tentative tracks for unmatched high-confidence detections.
        6. Update track lifecycle states.

        Parameters
        ----------
        detections:
            Detections from the current frame.
        frame_id:
            Monotonically increasing frame counter.

        Returns
        -------
        list[Track]
            All tracks that are currently TENTATIVE, ACTIVE, or OCCLUDED.
            LOST and EXITED tracks are excluded.
        """
        self._frame_count += 1

        # ── Step 1: Predict all tracks forward ────────────────────────
        predicted_bboxes: dict[int, BBox] = {}
        for tid, trk in self._tracks.items():
            if trk.state != TrackState.EXITED:
                predicted_bboxes[tid] = trk.predict()

        # ── Step 2: Split detections by confidence ────────────────────
        conf_threshold = self._config.detection_confidence_threshold
        high_conf_threshold = max(
            conf_threshold,
            (conf_threshold + 1.0) / 2,  # midpoint heuristic
        )

        high_dets: list[tuple[int, Detection]] = []
        low_dets: list[tuple[int, Detection]] = []

        for i, det in enumerate(detections):
            if det.confidence >= high_conf_threshold:
                high_dets.append((i, det))
            else:
                low_dets.append((i, det))

        # Collect active track IDs for matching (exclude EXITED).
        active_tids = [
            tid for tid, trk in self._tracks.items()
            if trk.state != TrackState.EXITED
        ]

        # ── Step 3: First association — high-confidence detections ────
        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        if active_tids and high_dets:
            trk_bboxes = [predicted_bboxes[tid] for tid in active_tids]
            det_bboxes = [d.bbox for _, d in high_dets]

            matches, unmatched_trk_idx, _ = _match_detections_to_tracks(
                trk_bboxes, det_bboxes, self._config.track_high_threshold
            )

            for t_idx, d_idx in matches:
                tid = active_tids[t_idx]
                orig_det_idx, det = high_dets[d_idx]
                self._tracks[tid].update_with_detection(det, frame_id)
                matched_track_ids.add(tid)
                matched_det_indices.add(orig_det_idx)

        # ── Step 4: Second association — low-confidence detections ────
        #    (ByteTrack's key insight: salvage unmatched tracks with
        #     low-confidence detections that might be partially occluded.)
        remaining_tids = [
            tid for tid in active_tids if tid not in matched_track_ids
        ]

        if remaining_tids and low_dets:
            trk_bboxes = [predicted_bboxes[tid] for tid in remaining_tids]
            det_bboxes = [d.bbox for _, d in low_dets]

            matches_low, _, _ = _match_detections_to_tracks(
                trk_bboxes, det_bboxes, self._config.track_low_threshold
            )

            for t_idx, d_idx in matches_low:
                tid = remaining_tids[t_idx]
                orig_det_idx, det = low_dets[d_idx]
                self._tracks[tid].update_with_detection(det, frame_id)
                matched_track_ids.add(tid)
                matched_det_indices.add(orig_det_idx)

        # ── Step 5: Mark unmatched tracks as missed ───────────────────
        for tid in active_tids:
            if tid not in matched_track_ids:
                self._tracks[tid].mark_missed()

        # ── Step 6: Create new tracks for unmatched high-conf dets ────
        for orig_idx, det in high_dets:
            if orig_idx not in matched_det_indices:
                new_track = _InternalTrack(
                    track_id=self._next_id,
                    detection=det,
                    frame_id=frame_id,
                    history_maxlen=self._config.track_history_length,
                )
                self._tracks[self._next_id] = new_track
                self._next_id += 1

        # ── Step 7: Update lifecycle states ───────────────────────────
        self._update_states()

        # ── Step 8: Garbage-collect EXITED tracks ─────────────────────
        exited = [
            tid for tid, trk in self._tracks.items()
            if trk.state == TrackState.EXITED
        ]
        for tid in exited:
            del self._tracks[tid]

        if exited:
            logger.debug("Garbage-collected %d EXITED tracks", len(exited))

        # ── Return visible tracks ─────────────────────────────────────
        visible = [
            trk.to_public_track()
            for trk in self._tracks.values()
            if trk.state in (TrackState.TENTATIVE, TrackState.ACTIVE, TrackState.OCCLUDED)
        ]

        logger.debug(
            "Frame %d: %d detections → %d visible tracks (%d total internal)",
            frame_id,
            len(detections),
            len(visible),
            len(self._tracks),
        )

        return visible

    def get_active_tracks(self) -> list[Track]:
        """Return all confirmed (ACTIVE or OCCLUDED) tracks."""
        return [
            trk.to_public_track()
            for trk in self._tracks.values()
            if trk.state in (TrackState.ACTIVE, TrackState.OCCLUDED)
        ]

    def get_track(self, track_id: int) -> Track | None:
        """Look up a specific track by ID.  Returns *None* if not found."""
        trk = self._tracks.get(track_id)
        if trk is None:
            return None
        return trk.to_public_track()

    def reset(self) -> None:
        """Clear all tracks and reset the ID counter."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0
        logger.info("MultiObjectTracker reset")

    # ── Internal state management ─────────────────────────────────────

    def _update_states(self) -> None:
        """Update the lifecycle state of every tracked object.

        State transition rules:

        - TENTATIVE + enough hits → ACTIVE
        - TENTATIVE + missed → EXITED (never confirmed, just remove)
        - ACTIVE + missed a few frames → OCCLUDED
        - OCCLUDED + re-detected → ACTIVE
        - OCCLUDED + too many misses → LOST
        - ACTIVE + too many misses → LOST
        - LOST → EXITED (immediate — we keep LOST for one cycle then GC)

        The thresholds come from :attr:`_config`.
        """
        max_age = self._config.track_max_age
        min_hits = self._config.track_min_hits
        # Number of consecutive misses before transitioning ACTIVE → OCCLUDED.
        # We use a fraction of max_age as the occlusion grace period.
        occlusion_grace = max(1, max_age // 3)

        for trk in self._tracks.values():
            state = trk.state

            if state == TrackState.TENTATIVE:
                if trk.misses > 0:
                    # Never confirmed — discard immediately.
                    trk.state = TrackState.EXITED
                    logger.debug(
                        "Track %d: TENTATIVE → EXITED (missed before confirmation)",
                        trk.track_id,
                    )
                elif trk.hits >= min_hits:
                    trk.state = TrackState.ACTIVE
                    logger.debug(
                        "Track %d: TENTATIVE → ACTIVE (hits=%d)",
                        trk.track_id,
                        trk.hits,
                    )

            elif state == TrackState.ACTIVE:
                if trk.misses > max_age:
                    trk.state = TrackState.LOST
                    logger.debug(
                        "Track %d: ACTIVE → LOST (misses=%d > max_age=%d)",
                        trk.track_id,
                        trk.misses,
                        max_age,
                    )
                elif trk.misses > occlusion_grace:
                    trk.state = TrackState.OCCLUDED
                    logger.debug(
                        "Track %d: ACTIVE → OCCLUDED (misses=%d)",
                        trk.track_id,
                        trk.misses,
                    )

            elif state == TrackState.OCCLUDED:
                if trk.misses == 0:
                    # Re-detected!
                    trk.state = TrackState.ACTIVE
                    logger.debug(
                        "Track %d: OCCLUDED → ACTIVE (re-detected)", trk.track_id
                    )
                elif trk.misses > max_age:
                    trk.state = TrackState.LOST
                    logger.debug(
                        "Track %d: OCCLUDED → LOST (misses=%d > max_age=%d)",
                        trk.track_id,
                        trk.misses,
                        max_age,
                    )

            elif state == TrackState.LOST:
                # LOST tracks are garbage-collected immediately.
                trk.state = TrackState.EXITED
                logger.debug("Track %d: LOST → EXITED", trk.track_id)
