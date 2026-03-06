"""Cross-camera person gallery for Re-ID tracking.

Phase 28B — Multi-Camera Person Re-Identification.

The ``ReIDGallery`` maintains an in-memory registry of all persons seen across
all active streaming sessions.  Each person is assigned a persistent UUID
(``person_id``) the first time they appear; subsequent sightings in any camera
are merged into the same entry if their appearance embedding is sufficiently
similar.

When a cross-camera match is detected (same person appearing in a different
session) a ``CrossCameraEvent`` is produced and stored so subscribers can be
notified via SSE.

Design choices
--------------
* In-memory only — no DB persistence.  Re-ID state resets on server restart.
* Thread-safe via a ``threading.Lock`` on all writes.
* Embedding similarity threshold: ``match_threshold`` (default 0.72).
  Higher = more strict matching (fewer false positives, more missed matches).
* Gallery size bounded per session (default 200 persons) to prevent unbounded
  memory growth in long-running deployments.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CameraAppearance:
    """A single sighting of a person in a specific camera session."""

    session_id: str
    session_name: str
    timestamp: float
    confidence: float
    bbox: dict[str, float]  # {x1, y1, x2, y2}


@dataclass
class PersonTrack:
    """All known sightings of one person across cameras."""

    person_id: str
    label: str  # detector label (e.g. "person")
    first_seen: float
    last_seen: float
    embedding: np.ndarray  # running mean of appearance embeddings
    n_sightings: int
    camera_history: list[CameraAppearance]
    camera_set: set[str]  # session IDs where this person was seen


@dataclass
class CrossCameraEvent:
    """Event fired when a person is matched across two different cameras."""

    event_id: str
    person_id: str
    from_session_id: str
    from_session_name: str
    to_session_id: str
    to_session_name: str
    similarity: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "cross_camera_match",
            "event_id": self.event_id,
            "person_id": self.person_id,
            "from_session_id": self.from_session_id,
            "from_session_name": self.from_session_name,
            "to_session_id": self.to_session_id,
            "to_session_name": self.to_session_name,
            "similarity": round(self.similarity, 3),
            "timestamp": self.timestamp,
        }


class ReIDGallery:
    """Thread-safe in-memory registry of person tracks across all cameras.

    Parameters
    ----------
    match_threshold:
        Cosine similarity threshold for matching a new sighting to an
        existing person.  Range [0, 1].  Default 0.72.
    max_persons_per_session:
        Maximum number of person tracks per session to prevent unbounded growth.
    """

    def __init__(
        self,
        match_threshold: float = 0.72,
        max_persons_per_session: int = 200,
    ) -> None:
        self._threshold = match_threshold
        self._max_per_session = max_persons_per_session
        self._persons: dict[str, PersonTrack] = {}
        self._lock = threading.Lock()
        # SSE: subscribers to cross-camera events
        self._event_queues: list[asyncio.Queue] = []
        self._recent_events: list[CrossCameraEvent] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        session_id: str,
        session_name: str,
        detections: list[Any],
        embeddings: list[np.ndarray],
    ) -> list[CrossCameraEvent]:
        """Process detections from one frame.

        Matches each detection to an existing person or creates a new track.
        Returns a list of ``CrossCameraEvent`` for any cross-camera matches.

        Parameters
        ----------
        session_id:
            The streaming session ID (camera identifier).
        session_name:
            Human-readable camera name.
        detections:
            List of detection objects with ``label``, ``confidence``, ``bbox``
            attributes.  Only ``"person"``-labelled detections are tracked.
        embeddings:
            Corresponding appearance embeddings, one per detection.
        """
        events: list[CrossCameraEvent] = []
        now = time.time()

        with self._lock:
            for det, emb in zip(detections, embeddings):
                if det.label.lower() not in ("person", "人", "人物"):
                    continue
                if emb is None or emb.sum() == 0:
                    continue

                best_id, best_sim = self._find_best_match(emb)
                appearance = CameraAppearance(
                    session_id=session_id,
                    session_name=session_name,
                    timestamp=now,
                    confidence=float(det.confidence),
                    bbox={
                        "x1": round(det.bbox.x1, 4),
                        "y1": round(det.bbox.y1, 4),
                        "x2": round(det.bbox.x2, 4),
                        "y2": round(det.bbox.y2, 4),
                    },
                )

                if best_id is not None:
                    # Update existing track
                    track = self._persons[best_id]
                    track.last_seen = now
                    track.n_sightings += 1
                    track.camera_history.append(appearance)
                    # Running mean of embeddings
                    n = track.n_sightings
                    track.embedding = ((n - 1) * track.embedding + emb) / n

                    # Cross-camera event?
                    if session_id not in track.camera_set:
                        track.camera_set.add(session_id)
                        prev_sessions = [a for a in track.camera_history if a.session_id != session_id]
                        if prev_sessions:
                            last_other = prev_sessions[-1]
                            ev = CrossCameraEvent(
                                event_id=str(uuid.uuid4()),
                                person_id=best_id,
                                from_session_id=last_other.session_id,
                                from_session_name=last_other.session_name,
                                to_session_id=session_id,
                                to_session_name=session_name,
                                similarity=best_sim,
                                timestamp=now,
                            )
                            events.append(ev)
                            self._recent_events.append(ev)
                            if len(self._recent_events) > 200:
                                self._recent_events.pop(0)
                else:
                    # New person — check per-session limit
                    session_count = sum(
                        1 for p in self._persons.values() if session_id in p.camera_set
                    )
                    if session_count >= self._max_per_session:
                        continue
                    pid = str(uuid.uuid4())
                    self._persons[pid] = PersonTrack(
                        person_id=pid,
                        label=det.label,
                        first_seen=now,
                        last_seen=now,
                        embedding=emb.copy(),
                        n_sightings=1,
                        camera_history=[appearance],
                        camera_set={session_id},
                    )

        # Broadcast events to SSE subscribers (outside lock)
        for ev in events:
            self._broadcast(ev)

        return events

    def get_persons(
        self,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return serialisable summaries of all tracked persons.

        Parameters
        ----------
        session_id:
            If given, only include persons seen in this session.
        """
        with self._lock:
            persons = list(self._persons.values())

        if session_id:
            persons = [p for p in persons if session_id in p.camera_set]

        persons.sort(key=lambda p: p.last_seen, reverse=True)
        return [_track_to_dict(p) for p in persons[:limit]]

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        with self._lock:
            track = self._persons.get(person_id)
        return _track_to_dict(track) if track else None

    def get_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._recent_events[-limit:])
        return [e.to_dict() for e in reversed(events)]

    def clear_session(self, session_id: str) -> int:
        """Remove all person tracks that were only seen in *session_id*.

        Returns the number of tracks removed.
        """
        with self._lock:
            to_remove = [
                pid for pid, p in self._persons.items()
                if p.camera_set == {session_id}
            ]
            for pid in to_remove:
                del self._persons[pid]
            # Strip appearances for this session from remaining tracks
            for p in self._persons.values():
                p.camera_history = [a for a in p.camera_history if a.session_id != session_id]
                p.camera_set.discard(session_id)
        return len(to_remove)

    def subscribe_events(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._event_queues.append(q)
        return q

    def unsubscribe_events(self, q: asyncio.Queue) -> None:
        try:
            self._event_queues.remove(q)
        except ValueError:
            pass

    @property
    def total_persons(self) -> int:
        with self._lock:
            return len(self._persons)

    # ── Private ────────────────────────────────────────────────────────────────

    def _find_best_match(
        self, emb: np.ndarray
    ) -> tuple[str | None, float]:
        """Find the best matching person ID for *emb*.

        Returns ``(person_id, similarity)`` or ``(None, 0.0)`` if no match
        above threshold.
        """
        best_id = None
        best_sim = 0.0
        for pid, track in self._persons.items():
            sim = _cosine_sim(emb, track.embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = pid
        if best_sim >= self._threshold:
            return best_id, best_sim
        return None, 0.0

    def _broadcast(self, event: CrossCameraEvent) -> None:
        payload = event.to_dict()
        for q in list(self._event_queues):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass


# ── Serialisation helpers ──────────────────────────────────────────────────────


def _track_to_dict(track: PersonTrack) -> dict[str, Any]:
    return {
        "person_id": track.person_id,
        "label": track.label,
        "first_seen": track.first_seen,
        "last_seen": track.last_seen,
        "n_sightings": track.n_sightings,
        "cameras": list(track.camera_set),
        "camera_count": len(track.camera_set),
        "camera_history": [
            {
                "session_id": a.session_id,
                "session_name": a.session_name,
                "timestamp": a.timestamp,
                "confidence": a.confidence,
                "bbox": a.bbox,
            }
            for a in track.camera_history[-10:]  # last 10 appearances
        ],
    }


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))
