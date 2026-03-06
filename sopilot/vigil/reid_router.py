"""Cross-camera Re-ID API endpoints.

Phase 28B — Multi-Camera Person Re-Identification.

Endpoints
---------
GET  /vigil/reid/persons                   All tracked persons (with filters)
GET  /vigil/reid/persons/{person_id}       Single person + full camera history
GET  /vigil/reid/events                    SSE stream of cross-camera events
GET  /vigil/reid/events/recent             Last N cross-camera events (REST)
GET  /vigil/reid/stats                     Gallery statistics
DELETE /vigil/reid/sessions/{sid}          Clear Re-ID data for one camera session
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


def _get_gallery(request: Request):
    """Get the shared ReIDGallery from app.state."""
    gallery = getattr(request.app.state, "reid_gallery", None)
    if gallery is None:
        raise HTTPException(status_code=503, detail="ReID gallery not initialised")
    return gallery


def build_reid_router() -> APIRouter:
    router = APIRouter(prefix="/vigil/reid", tags=["Re-ID"])

    # ── GET /persons ───────────────────────────────────────────────────────────

    @router.get(
        "/persons",
        summary="List all tracked persons across cameras",
        response_model=None,
    )
    async def list_persons(
        request: Request,
        session_id: str | None = Query(None, description="Filter by streaming session"),
        limit: int = Query(100, ge=1, le=500),
    ) -> dict[str, Any]:
        gallery = _get_gallery(request)
        persons = gallery.get_persons(session_id=session_id, limit=limit)
        return {
            "persons": persons,
            "total": len(persons),
            "total_in_gallery": gallery.total_persons,
        }

    # ── GET /persons/{person_id} ───────────────────────────────────────────────

    @router.get(
        "/persons/{person_id}",
        summary="Get a single tracked person with full camera history",
        response_model=None,
    )
    async def get_person(person_id: str, request: Request) -> dict[str, Any]:
        gallery = _get_gallery(request)
        person = gallery.get_person(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found")
        return person

    # ── GET /events/recent ─────────────────────────────────────────────────────

    @router.get(
        "/events/recent",
        summary="Get recent cross-camera match events (REST)",
        response_model=None,
    )
    async def recent_events(
        request: Request,
        limit: int = Query(50, ge=1, le=200),
    ) -> dict[str, Any]:
        gallery = _get_gallery(request)
        events = gallery.get_recent_events(limit=limit)
        return {"events": events, "total": len(events)}

    # ── GET /events — SSE ──────────────────────────────────────────────────────

    @router.get(
        "/events",
        summary="SSE stream of cross-camera match events",
        response_class=StreamingResponse,
    )
    async def sse_events(request: Request) -> StreamingResponse:
        """Server-Sent Events — fires when a person is spotted in a new camera."""
        gallery = _get_gallery(request)

        async def _generate():
            q = gallery.subscribe_events()
            try:
                yield 'data: {"type": "connected"}\n\n'
                while True:
                    try:
                        item = await asyncio.wait_for(q.get(), timeout=30.0)
                        yield f"data: {json.dumps(item)}\n\n"
                    except asyncio.TimeoutError:
                        yield 'data: {"type": "keepalive"}\n\n'
            finally:
                gallery.unsubscribe_events(q)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── GET /stats ─────────────────────────────────────────────────────────────

    @router.get(
        "/stats",
        summary="Re-ID gallery statistics",
        response_model=None,
    )
    async def reid_stats(request: Request) -> dict[str, Any]:
        gallery = _get_gallery(request)
        persons = gallery.get_persons(limit=10000)
        multi_camera = [p for p in persons if p["camera_count"] > 1]
        recent = gallery.get_recent_events(limit=10)
        return {
            "total_persons": gallery.total_persons,
            "multi_camera_persons": len(multi_camera),
            "recent_cross_camera_events": len(recent),
            "match_threshold": gallery._threshold,
        }

    # ── DELETE /sessions/{sid} ─────────────────────────────────────────────────

    @router.delete(
        "/sessions/{sid}",
        summary="Clear Re-ID data for a streaming session",
    )
    async def clear_session(sid: str, request: Request) -> dict[str, Any]:
        gallery = _get_gallery(request)
        removed = gallery.clear_session(sid)
        return {"session_id": sid, "persons_removed": removed}

    return router
