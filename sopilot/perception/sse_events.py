"""Thread-safe SSE event queue system.

PerceptionEngine (sync thread) calls push_event() to emit events.
FastAPI SSE endpoint (async) reads via event_generator() async generator.
"""

import queue
import asyncio
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Any

_registry: dict[str, "EventQueue"] = {}
_registry_lock = threading.Lock()


@dataclass
class PercEvent:
    """Single perception event for SSE streaming."""
    event_type: str   # "ANOMALY", "GOAL_DETECTED", "DELIBERATION_RESULT", "EPISODE_CLOSED", "NL_TASK_TRIGGERED", "ENTITY_APPEARED"
    session_id: str
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        data = {"type": self.event_type, "session_id": self.session_id,
                "timestamp": self.timestamp, **self.payload}
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


class EventQueue:
    """Thread-safe bounded queue for one perception session."""
    MAX_SIZE = 500

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._q: queue.Queue[PercEvent] = queue.Queue(maxsize=self.MAX_SIZE)
        self.push_count: int = 0
        self.drop_count: int = 0
        self._lock = threading.Lock()

    def push(self, event: PercEvent) -> bool:
        """Push from sync thread. On overflow: drops oldest to make room. Returns True if enqueued."""
        with self._lock:
            try:
                self._q.put_nowait(event)
                self.push_count += 1
                return True
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(event)
                    self.push_count += 1
                except queue.Full:
                    pass
                self.drop_count += 1
                return False

    async def next_event(self, timeout: float = 15.0) -> "PercEvent | None":
        """Async get with timeout. Returns None on timeout (caller sends keepalive).
        Polls in 1-second increments via asyncio.to_thread to avoid blocking event loop."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            poll = min(1.0, remaining)
            try:
                return await asyncio.to_thread(self._q.get, True, poll)
            except queue.Empty:
                continue
        return None

    def qsize(self) -> int:
        return self._q.qsize()

    def clear(self) -> None:
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def get_stats(self) -> dict:
        return {"session_id": self.session_id, "queued": self.qsize(),
                "pushed": self.push_count, "dropped": self.drop_count}


def get_or_create(session_id: str) -> EventQueue:
    with _registry_lock:
        if session_id not in _registry:
            _registry[session_id] = EventQueue(session_id)
        return _registry[session_id]


def get(session_id: str) -> "EventQueue | None":
    with _registry_lock:
        return _registry.get(session_id)


def remove(session_id: str) -> None:
    with _registry_lock:
        _registry.pop(session_id, None)


def push_event(session_id: str, event_type: str, payload: dict[str, Any]) -> bool:
    """Push to session queue. No-op (returns False) if session not registered."""
    eq = get(session_id)
    if eq is None:
        return False
    return eq.push(PercEvent(event_type=event_type, session_id=session_id, payload=payload))


def list_sessions() -> list[str]:
    with _registry_lock:
        return list(_registry.keys())


async def event_generator(session_id: str):
    """Async generator yielding SSE strings. Yields keepalive on timeout."""
    eq = get_or_create(session_id)
    while True:
        event = await eq.next_event(timeout=15.0)
        if event is None:
            yield ": keepalive\n\n"
        else:
            yield event.to_sse()
