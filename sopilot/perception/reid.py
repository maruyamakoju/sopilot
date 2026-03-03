"""Cross-Camera Re-Identification (Re-ID).

Links entities across different camera sessions.
Uses cosine similarity on appearance feature vectors.
Thread-safe global tracker instance.
"""

import time
import uuid
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Any

KNOWN_LABELS = [
    "person", "worker", "vehicle", "car", "truck", "bicycle",
    "equipment", "machinery", "box", "unknown",
]
LABEL_INDEX = {lbl: i for i, lbl in enumerate(KNOWN_LABELS)}
LABEL_DIM = len(KNOWN_LABELS)   # 10
GEOM_DIM = 4
VEL_DIM = 2
COLOR_DIM = 48   # 16 bins × 3 channels
BASE_DIM = LABEL_DIM + GEOM_DIM + VEL_DIM   # 16
FULL_DIM = BASE_DIM + COLOR_DIM              # 64


@dataclass
class ReIDFeature:
    entity_id: int
    session_id: str
    label: str
    feature_vector: np.ndarray   # shape (BASE_DIM,) or (FULL_DIM,)
    bbox: tuple[float, float, float, float]  # x,y,w,h normalized [0,1]
    timestamp: float = field(default_factory=time.time)
    track_age: int = 0


@dataclass
class GlobalTrack:
    id: str
    appearances: list[ReIDFeature] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    dominant_label: str = "unknown"

    @property
    def track_count(self) -> int:
        return len(self.appearances)

    @property
    def session_ids(self) -> list[str]:
        seen: list[str] = []
        for a in self.appearances:
            if a.session_id not in seen:
                seen.append(a.session_id)
        return seen

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "track_count": self.track_count,
            "dominant_label": self.dominant_label,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "session_ids": self.session_ids,
            "appearances": [
                {"entity_id": a.entity_id, "session_id": a.session_id,
                 "label": a.label, "timestamp": a.timestamp}
                for a in self.appearances
            ],
        }


class AppearanceEncoder:
    """Encode entity appearance into a normalized feature vector."""

    @staticmethod
    def encode(
        entity_id: int,
        label: str,
        bbox: tuple[float, float, float, float],
        velocity: tuple[float, float] = (0.0, 0.0),
        frame: Any = None,  # np.ndarray H×W×3 uint8 | None
        track_age: int = 0,
    ) -> np.ndarray:
        # Label one-hot
        label_vec = np.zeros(LABEL_DIM, dtype=np.float32)
        norm_label = label.lower().strip()
        idx = LABEL_INDEX.get(norm_label)
        if idx is None:
            for kw, ki in LABEL_INDEX.items():
                if kw in norm_label or norm_label in kw:
                    idx = ki
                    break
        label_vec[idx if idx is not None else LABEL_INDEX["unknown"]] = 1.0

        # Geometry
        x, y, w, h = [float(np.clip(v, 0.0, 1.0)) for v in bbox]
        geom_vec = np.array([x, y, w, h], dtype=np.float32)

        # Velocity
        vx = float(np.clip(velocity[0], -1.0, 1.0))
        vy = float(np.clip(velocity[1], -1.0, 1.0))
        vel_vec = np.array([vx, vy], dtype=np.float32)

        base = np.concatenate([label_vec, geom_vec, vel_vec])

        # Color histogram if frame available
        if frame is not None:
            try:
                color_vec = AppearanceEncoder._color_histogram(frame, bbox)
                base = np.concatenate([base, color_vec])
            except Exception:
                pass

        # L2 normalize
        norm = np.linalg.norm(base)
        if norm > 1e-8:
            base = base / norm
        return base.astype(np.float32)

    @staticmethod
    def _color_histogram(
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        x, y, bw, bh = bbox
        x1 = max(0, int(x * w))
        y1 = max(0, int(y * h))
        x2 = min(w, int((x + bw) * w))
        y2 = min(h, int((y + bh) * h))
        if x2 <= x1 or y2 <= y1:
            return np.zeros(COLOR_DIM, dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        hist = []
        channels = min(3, crop.shape[2]) if crop.ndim == 3 else 1
        for c in range(channels):
            ch = crop[:, :, c].ravel() if crop.ndim == 3 else crop.ravel()
            h_vals, _ = np.histogram(ch, bins=16, range=(0, 256))
            hist.append(h_vals.astype(np.float32))
        # Pad to 3 channels if grayscale
        while len(hist) < 3:
            hist.append(np.zeros(16, dtype=np.float32))
        result = np.concatenate(hist)
        s = result.sum()
        if s > 0:
            result = result / s
        return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Pads shorter vector with zeros."""
    if len(a) != len(b):
        max_len = max(len(a), len(b))
        a = np.pad(a, (0, max_len - len(a)))
        b = np.pad(b, (0, max_len - len(b)))
    dot = float(np.dot(a, b))
    return float(np.clip(dot, -1.0, 1.0))


class CrossCameraTracker:
    """Maintains global entity tracks across multiple camera sessions."""
    DEFAULT_THRESHOLD = 0.80
    MAX_TIME_GAP = 600.0   # 10 minutes
    MAX_GLOBAL_TRACKS = 200

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_THRESHOLD,
        max_time_gap: float = MAX_TIME_GAP,
    ):
        self._threshold = similarity_threshold
        self._max_time_gap = max_time_gap
        self._global_tracks: dict[str, GlobalTrack] = {}
        self._entity_index: dict[tuple[str, int], str] = {}  # (session_id, entity_id) → global_id
        self._lock = threading.Lock()

    def register(self, entity_id: int, session_id: str, feature: ReIDFeature) -> str:
        """Register entity. Returns global track ID."""
        with self._lock:
            key = (session_id, entity_id)
            if key in self._entity_index:
                gid = self._entity_index[key]
                if gid in self._global_tracks:
                    gt = self._global_tracks[gid]
                    gt.appearances.append(feature)
                    gt.last_seen = feature.timestamp
                    gt.dominant_label = self._dominant_label(gt)
                return self._entity_index[key]

            best_gid, best_score = self._find_best_match(feature, session_id)
            if best_gid and best_score >= self._threshold:
                gid = best_gid
                gt = self._global_tracks[gid]
                gt.appearances.append(feature)
                gt.last_seen = feature.timestamp
                gt.dominant_label = self._dominant_label(gt)
            else:
                gid = str(uuid.uuid4())[:8]
                gt = GlobalTrack(
                    id=gid, appearances=[feature],
                    first_seen=feature.timestamp, last_seen=feature.timestamp,
                    dominant_label=feature.label,
                )
                self._global_tracks[gid] = gt

            self._entity_index[key] = gid
            if len(self._global_tracks) > self.MAX_GLOBAL_TRACKS:
                self._prune_oldest()
            return gid

    def get_global_tracks(self) -> list[GlobalTrack]:
        with self._lock:
            return list(self._global_tracks.values())

    def get_global_track(self, global_id: str) -> "GlobalTrack | None":
        with self._lock:
            return self._global_tracks.get(global_id)

    def get_global_id(self, session_id: str, entity_id: int) -> "str | None":
        with self._lock:
            return self._entity_index.get((session_id, entity_id))

    def get_cross_camera_global_tracks(self) -> list[GlobalTrack]:
        """Only tracks appearing in 2+ sessions."""
        with self._lock:
            return [
                gt for gt in self._global_tracks.values()
                if len({a.session_id for a in gt.appearances}) >= 2
            ]

    def reset(self) -> None:
        with self._lock:
            self._global_tracks.clear()
            self._entity_index.clear()

    def get_state_dict(self) -> dict:
        with self._lock:
            cross = sum(
                1 for gt in self._global_tracks.values()
                if len({a.session_id for a in gt.appearances}) >= 2
            )
            return {
                "total_global_tracks": len(self._global_tracks),
                "cross_camera_tracks": cross,
                "registered_entities": len(self._entity_index),
                "tracks": [gt.to_dict() for gt in self._global_tracks.values()],
            }

    def _find_best_match(self, feature: ReIDFeature, source_session: str) -> tuple["str | None", float]:
        best_gid = None
        best_score = -1.0
        now = feature.timestamp
        for gid, gt in self._global_tracks.items():
            sessions_in_gt = {a.session_id for a in gt.appearances}
            # Skip if only this session
            if source_session in sessions_in_gt and len(sessions_in_gt) == 1:
                continue
            if now - gt.last_seen > self._max_time_gap:
                continue
            score = cosine_similarity(feature.feature_vector, gt.appearances[-1].feature_vector)
            if score > best_score:
                best_score = score
                best_gid = gid
        return best_gid, best_score

    def _dominant_label(self, gt: GlobalTrack) -> str:
        counts: dict[str, int] = {}
        for a in gt.appearances:
            counts[a.label] = counts.get(a.label, 0) + 1
        return max(counts, key=counts.get) if counts else "unknown"

    def _prune_oldest(self) -> None:
        by_time = sorted(self._global_tracks.items(), key=lambda kv: kv[1].last_seen)
        to_remove = len(self._global_tracks) - self.MAX_GLOBAL_TRACKS
        for gid, _ in by_time[:to_remove]:
            del self._global_tracks[gid]
            dead = [k for k, v in self._entity_index.items() if v == gid]
            for k in dead:
                del self._entity_index[k]


# Module-level singleton tracker
_global_tracker: CrossCameraTracker | None = None
_global_tracker_lock = threading.Lock()


def get_global_tracker() -> CrossCameraTracker:
    """Return the module-level singleton CrossCameraTracker."""
    global _global_tracker
    with _global_tracker_lock:
        if _global_tracker is None:
            _global_tracker = CrossCameraTracker()
        return _global_tracker


def reset_global_tracker() -> None:
    """Reset the module-level singleton tracker."""
    global _global_tracker
    with _global_tracker_lock:
        if _global_tracker is not None:
            _global_tracker.reset()
        else:
            _global_tracker = CrossCameraTracker()
