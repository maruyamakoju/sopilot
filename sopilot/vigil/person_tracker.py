"""Within-session person tracker.

Phase 29 — Person Tracking + Behavioral Anomaly Detection.

Uses IoU-based greedy matching to maintain identity continuity across
consecutive frames without requiring a Kalman filter or neural backbone.

All bbox coordinates are normalized [0.0, 1.0] (same as FrameInferrer output).
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ── IoU helper ─────────────────────────────────────────────────────────────────


def _iou(b1: dict[str, float], b2: dict[str, float]) -> float:
    """Intersection-over-Union for two bbox dicts {x1, y1, x2, y2}."""
    ix1 = max(b1["x1"], b2["x1"])
    iy1 = max(b1["y1"], b2["y1"])
    ix2 = min(b1["x2"], b2["x2"])
    iy2 = min(b1["y2"], b2["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    a1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
    a2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
    union = a1 + a2 - inter
    return inter / union if union > 0.0 else 0.0


# ── Track state ────────────────────────────────────────────────────────────────


@dataclass
class TrackState:
    """State for one tracked person across frames."""

    track_id: str
    label: str
    trajectory: deque  # deque[dict] — {cx, cy, x1, y1, x2, y2, timestamp, frame_idx}
    first_seen: float
    last_seen: float
    last_frame_idx: int = 0
    lost_frames: int = 0
    status: str = "active"  # "active" | "lost"
    behavior_flags: set = field(default_factory=set)

    @property
    def age(self) -> float:
        """Seconds since first appearance."""
        return max(0.0, self.last_seen - self.first_seen)

    def to_dict(self) -> dict[str, Any]:
        traj = list(self.trajectory)
        latest = traj[-1] if traj else {}
        return {
            "track_id": self.track_id,
            "label": self.label,
            "status": self.status,
            "age_seconds": round(self.age, 2),
            "sightings": len(traj),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "current_bbox": (
                {k: latest[k] for k in ("x1", "y1", "x2", "y2")}
                if latest
                else None
            ),
            "behavior_flags": sorted(self.behavior_flags),
        }


# ── Tracker ────────────────────────────────────────────────────────────────────


_PERSON_LABELS = frozenset({"person", "人", "人物"})


class PersonTracker:
    """Frame-by-frame IoU-based person tracker (single session).

    Parameters
    ----------
    iou_threshold:
        Minimum IoU to consider a detection as matching an existing track.
    max_lost_frames:
        Consecutive frames a track may be missing before it is marked 'lost'.
    max_history:
        Maximum trajectory points kept per track.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 10,
        max_history: int = 60,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.max_history = max_history
        self._tracks: dict[str, TrackState] = {}
        self._frame_idx: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        detections: list[dict[str, Any]],
        timestamp: float | None = None,
    ) -> list[TrackState]:
        """Ingest one frame's detections and return all (active + lost) tracks.

        Parameters
        ----------
        detections:
            List of detection dicts from FrameInferrer:
            ``{"label": str, "confidence": float, "bbox": {"x1":…,"y1":…,"x2":…,"y2":…}}``.
        timestamp:
            Frame capture time (seconds). Defaults to ``time.time()``.
        """
        if timestamp is None:
            timestamp = time.time()

        self._frame_idx += 1
        frame_idx = self._frame_idx

        person_dets = [
            d for d in detections
            if d.get("label", "").lower() in _PERSON_LABELS
        ]

        active_tracks = [t for t in self._tracks.values() if t.status == "active"]

        # ── Greedy IoU matching (highest IoU first) ────────────────────────────
        pairs: list[tuple[float, str, int]] = []
        for track in active_tracks:
            if not track.trajectory:
                continue
            last = track.trajectory[-1]
            t_bbox = {k: last[k] for k in ("x1", "y1", "x2", "y2")}
            for i, det in enumerate(person_dets):
                iou_val = _iou(t_bbox, det["bbox"])
                if iou_val >= self.iou_threshold:
                    pairs.append((iou_val, track.track_id, i))

        pairs.sort(reverse=True)

        matched_track_ids: set[str] = set()
        matched_det_indices: set[int] = set()

        for _, track_id, det_idx in pairs:
            if track_id in matched_track_ids or det_idx in matched_det_indices:
                continue
            matched_track_ids.add(track_id)
            matched_det_indices.add(det_idx)

            track = self._tracks[track_id]
            det = person_dets[det_idx]
            bbox = det["bbox"]
            cx = (bbox["x1"] + bbox["x2"]) / 2.0
            cy = (bbox["y1"] + bbox["y2"]) / 2.0
            track.trajectory.append(
                {
                    "cx": cx, "cy": cy,
                    "x1": bbox["x1"], "y1": bbox["y1"],
                    "x2": bbox["x2"], "y2": bbox["y2"],
                    "timestamp": timestamp,
                    "frame_idx": frame_idx,
                }
            )
            track.last_seen = timestamp
            track.last_frame_idx = frame_idx
            track.lost_frames = 0
            track.status = "active"

        # ── Unmatched active tracks → increment lost counter ──────────────────
        for track in active_tracks:
            if track.track_id not in matched_track_ids:
                track.lost_frames += 1
                if track.lost_frames > self.max_lost_frames:
                    track.status = "lost"

        # ── Unmatched detections → new tracks ─────────────────────────────────
        for i, det in enumerate(person_dets):
            if i in matched_det_indices:
                continue
            bbox = det["bbox"]
            cx = (bbox["x1"] + bbox["x2"]) / 2.0
            cy = (bbox["y1"] + bbox["y2"]) / 2.0
            track_id = str(uuid.uuid4())
            new_track = TrackState(
                track_id=track_id,
                label=det.get("label", "person"),
                trajectory=deque(
                    [
                        {
                            "cx": cx, "cy": cy,
                            "x1": bbox["x1"], "y1": bbox["y1"],
                            "x2": bbox["x2"], "y2": bbox["y2"],
                            "timestamp": timestamp,
                            "frame_idx": frame_idx,
                        }
                    ],
                    maxlen=self.max_history,
                ),
                first_seen=timestamp,
                last_seen=timestamp,
                last_frame_idx=frame_idx,
            )
            self._tracks[track_id] = new_track

        return list(self._tracks.values())

    def get_active_tracks(self) -> list[dict[str, Any]]:
        """Return serialized dicts for all active tracks."""
        return [t.to_dict() for t in self._tracks.values() if t.status == "active"]

    def get_all_tracks(self) -> list[dict[str, Any]]:
        """Return serialized dicts for all tracks (active + lost)."""
        return [t.to_dict() for t in self._tracks.values()]

    def get_track_state(self, track_id: str) -> TrackState | None:
        return self._tracks.get(track_id)

    @property
    def total_tracks(self) -> int:
        return len(self._tracks)

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tracks.values() if t.status == "active")

    def clear(self) -> None:
        """Reset the tracker (e.g. when a session ends)."""
        self._tracks.clear()
        self._frame_idx = 0
