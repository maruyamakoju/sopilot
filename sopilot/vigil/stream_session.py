"""Live streaming session state for real-time camera frame inference.

Each StreamSession represents one camera source (webcam, RTSP stream, or
frame-push client).  Sessions are stored in-memory on ``app.state.stream_sessions``
(a ``dict[str, StreamSession]``).  They are never persisted to the database.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamSession:
    """In-memory state for one live streaming session."""

    session_id: str
    name: str
    rules: list[str]
    created_at: float = field(default_factory=time.time)
    frame_count: int = 0
    last_result: dict[str, Any] | None = None
    last_frame_jpeg: bytes | None = None  # latest annotated JPEG for MJPEG preview

    # SSE subscriber queues — one per connected client
    _result_queues: list[asyncio.Queue] = field(default_factory=list)
    # Stop flag — set when DELETE is called
    _stop_event: threading.Event = field(default_factory=threading.Event)

    @property
    def is_active(self) -> bool:
        return not self._stop_event.is_set()

    def stop(self) -> None:
        """Signal the session to stop; notifies all SSE subscribers."""
        self._stop_event.set()
        # Push a sentinel so SSE generators can exit cleanly
        for q in list(self._result_queues):
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> asyncio.Queue:
        """Register a new SSE subscriber; returns a queue that receives result dicts."""
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._result_queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove an SSE subscriber queue."""
        try:
            self._result_queues.remove(q)
        except ValueError:
            pass

    def push_result(self, result: dict[str, Any]) -> None:
        """Broadcast an inference result to all SSE subscribers (non-blocking).

        Drops the event for a subscriber if their queue is full (back-pressure).
        """
        for q in list(self._result_queues):
            try:
                q.put_nowait(result)
            except asyncio.QueueFull:
                pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "rules": self.rules,
            "created_at": self.created_at,
            "frame_count": self.frame_count,
            "is_active": self.is_active,
            "subscribers": len(self._result_queues),
            "last_result": self.last_result,
        }


def create_session(name: str, rules: list[str]) -> StreamSession:
    """Factory — creates a new StreamSession with a random UUID."""
    return StreamSession(
        session_id=str(uuid.uuid4()),
        name=name,
        rules=rules,
    )
