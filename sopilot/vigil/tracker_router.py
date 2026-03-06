"""Person tracker + behavior event API.

Phase 29 — Person Tracking + Behavioral Anomaly Detection.

Endpoints
---------
GET    /vigil/tracker/{session_id}/tracks   Active tracks in a session
GET    /vigil/tracker/{session_id}/events   Recent behavior events
GET    /vigil/tracker/{session_id}/stats    Session tracking statistics
DELETE /vigil/tracker/{session_id}          Clear tracker state for a session
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from sopilot.vigil.behavior_detector import BehaviorDetector
from sopilot.vigil.person_tracker import PersonTracker


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get_tracker(request: Request, session_id: str) -> PersonTracker:
    """Return (or lazily create) the PersonTracker for *session_id*."""
    if not hasattr(request.app.state, "stream_trackers"):
        request.app.state.stream_trackers = {}
    trackers: dict[str, PersonTracker] = request.app.state.stream_trackers
    if session_id not in trackers:
        trackers[session_id] = PersonTracker()
    return trackers[session_id]


def _get_events(request: Request, session_id: str) -> list[dict]:
    store: dict[str, list] = getattr(request.app.state, "stream_behavior_events", {})
    return store.get(session_id, [])


# ── Router ─────────────────────────────────────────────────────────────────────


def build_tracker_router() -> APIRouter:
    router = APIRouter(prefix="/vigil/tracker", tags=["PersonTracker"])

    # ── GET /{session_id}/tracks ───────────────────────────────────────────────

    @router.get(
        "/{session_id}/tracks",
        summary="Active person tracks in a streaming session",
    )
    async def get_tracks(
        session_id: str,
        request: Request,
        include_lost: bool = Query(False, description="Include lost (off-screen) tracks"),
    ) -> dict[str, Any]:
        tracker = _get_tracker(request, session_id)
        tracks = tracker.get_all_tracks() if include_lost else tracker.get_active_tracks()
        return {
            "session_id": session_id,
            "tracks": tracks,
            "total": len(tracks),
            "active_count": tracker.active_count,
        }

    # ── GET /{session_id}/events ───────────────────────────────────────────────

    @router.get(
        "/{session_id}/events",
        summary="Recent behavior anomaly events for a session",
    )
    async def get_events(
        session_id: str,
        request: Request,
        limit: int = Query(50, ge=1, le=200),
        behavior_type: str | None = Query(None, description="Filter: loitering|running|fall"),
    ) -> dict[str, Any]:
        all_events = _get_events(request, session_id)
        if behavior_type:
            all_events = [e for e in all_events if e.get("behavior_type") == behavior_type]
        recent = all_events[-limit:]
        return {
            "session_id": session_id,
            "events": recent,
            "total": len(all_events),
        }

    # ── GET /{session_id}/stats ────────────────────────────────────────────────

    @router.get(
        "/{session_id}/stats",
        summary="Tracking statistics for a session",
    )
    async def get_stats(session_id: str, request: Request) -> dict[str, Any]:
        tracker = _get_tracker(request, session_id)
        all_events = _get_events(request, session_id)

        by_type: dict[str, int] = {}
        for ev in all_events:
            bt = ev.get("behavior_type", "unknown")
            by_type[bt] = by_type.get(bt, 0) + 1

        return {
            "session_id": session_id,
            "active_tracks": tracker.active_count,
            "total_tracks": tracker.total_tracks,
            "total_behavior_events": len(all_events),
            "events_by_type": by_type,
        }

    # ── DELETE /{session_id} ───────────────────────────────────────────────────

    @router.delete(
        "/{session_id}",
        summary="Clear tracker state and behavior events for a session",
    )
    async def clear_tracker(session_id: str, request: Request) -> dict[str, Any]:
        trackers: dict[str, PersonTracker] = getattr(
            request.app.state, "stream_trackers", {}
        )
        removed_tracks = 0
        if session_id in trackers:
            removed_tracks = trackers[session_id].total_tracks
            trackers[session_id].clear()
            del trackers[session_id]

        events_store: dict[str, list] = getattr(
            request.app.state, "stream_behavior_events", {}
        )
        removed_events = len(events_store.pop(session_id, []))

        return {
            "session_id": session_id,
            "tracks_removed": removed_tracks,
            "events_removed": removed_events,
            "status": "cleared",
        }

    return router
