"""Real-time camera streaming endpoints.

Phase 27 — LiveStream API.

Provides REST + WebSocket + SSE + MJPEG endpoints for pushing individual
camera frames and receiving instant detection + violation results.

Endpoint summary
----------------
POST   /vigil/stream/sessions                  Create a new streaming session
GET    /vigil/stream/sessions                  List all active sessions
DELETE /vigil/stream/sessions/{sid}            Stop and remove a session
POST   /vigil/stream/sessions/{sid}/frame      Push a JPEG frame, get JSON results
WS     /vigil/stream/sessions/{sid}/ws         Bidirectional WebSocket streaming
GET    /vigil/stream/sessions/{sid}/events     SSE stream of inference results
GET    /vigil/stream/sessions/{sid}/preview    MJPEG live preview with bbox overlay

No server required beyond the running SOPilot process.
No VLM calls per frame — only local ONNX/detector inference.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from sopilot.vigil.frame_inferrer import FrameInferrer
from sopilot.vigil.reid_extractor import ReIDExtractor
from sopilot.vigil.stream_session import StreamSession, create_session

logger = logging.getLogger(__name__)

_MJPEG_BOUNDARY = b"--frame"
_MJPEG_HEADER = b"Content-Type: image/jpeg\r\n\r\n"


# ── Request / Response schemas ─────────────────────────────────────────────────


class CreateSessionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="セッション名")
    rules: list[str] = Field(
        default_factory=list,
        description="監視ルール (1件1ルール). 空の場合は全検出を報告。",
    )


class SessionResponse(BaseModel):
    session_id: str
    name: str
    rules: list[str]
    created_at: float
    frame_count: int
    is_active: bool
    subscribers: int
    last_result: dict | None = None


class FrameResultResponse(BaseModel):
    session_id: str
    frame_count: int
    detections: list[dict]
    violations: list[str]
    frame_ms: float


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get_sessions(request: Request) -> dict[str, StreamSession]:
    """Return the shared session registry from app state."""
    if not hasattr(request.app.state, "stream_sessions"):
        request.app.state.stream_sessions = {}
    return request.app.state.stream_sessions  # type: ignore[return-value]


def _get_session(request: Request, sid: str) -> StreamSession:
    sessions = _get_sessions(request)
    session = sessions.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Stream session '{sid}' not found")
    return session


def _get_inferrer(request: Request) -> FrameInferrer:
    """Return a FrameInferrer, cached on app.state.stream_inferrer.

    Tries to use the ONNX detector from VigilPipeline; falls back to a
    detector-less inferrer (returns empty detections) if not available.
    """
    if not hasattr(request.app.state, "stream_inferrer"):
        detector = None
        try:
            pipeline = request.app.state.vigil_pipeline
            if hasattr(pipeline._vlm, "_engine"):
                engine = pipeline._vlm._engine
                if hasattr(engine, "_detector") and engine._detector is not None:
                    detector = engine._detector
        except Exception:
            pass
        request.app.state.stream_inferrer = FrameInferrer(detector=detector)
    return request.app.state.stream_inferrer  # type: ignore[return-value]


def _get_extractor(request: Request) -> ReIDExtractor:
    """Return a cached ReIDExtractor on app.state."""
    if not hasattr(request.app.state, "reid_extractor"):
        request.app.state.reid_extractor = ReIDExtractor()
    return request.app.state.reid_extractor  # type: ignore[return-value]


def _run_reid(request: Request, session: StreamSession, frame_bytes: bytes, detections: list) -> None:
    """Feed person detections into the Re-ID gallery (best-effort, non-blocking)."""
    gallery = getattr(request.app.state, "reid_gallery", None)
    if gallery is None or not detections:
        return
    persons = [d for d in detections if d["label"].lower() in ("person", "人", "人物")]
    if not persons:
        return
    try:
        import cv2
        import numpy as np
        from dataclasses import dataclass

        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        @dataclass
        class _BBox:
            x1: float; y1: float; x2: float; y2: float

        @dataclass
        class _Det:
            label: str; confidence: float; bbox: _BBox

        extractor = _get_extractor(request)
        fake_dets = [
            _Det(
                label=d["label"],
                confidence=d["confidence"],
                bbox=_BBox(**d["bbox"]),
            )
            for d in persons
        ]
        embeddings = [extractor.extract(frame, det.bbox) for det in fake_dets]
        gallery.update(session.session_id, session.name, fake_dets, embeddings)
    except Exception:
        logger.debug("ReID update skipped", exc_info=True)


def _get_tracker(request: Request, session_id: str) -> "PersonTracker":
    """Return (or lazily create) the PersonTracker for a session."""
    from sopilot.vigil.person_tracker import PersonTracker

    if not hasattr(request.app.state, "stream_trackers"):
        request.app.state.stream_trackers = {}
    trackers = request.app.state.stream_trackers
    if session_id not in trackers:
        trackers[session_id] = PersonTracker()
    return trackers[session_id]


def _run_tracker(
    request: Request,
    session: StreamSession,
    detections: list,
    timestamp: float,
) -> None:
    """Update person tracker + fire behavior events into the SSE queue. Best-effort."""
    try:
        from sopilot.vigil.behavior_detector import BehaviorDetector

        tracker = _get_tracker(request, session.session_id)
        tracks = tracker.update(detections, timestamp=timestamp)

        # Shared (stateless) behavior detector
        if not hasattr(request.app.state, "stream_behavior_detector"):
            request.app.state.stream_behavior_detector = BehaviorDetector()
        detector: BehaviorDetector = request.app.state.stream_behavior_detector

        if not hasattr(request.app.state, "stream_behavior_events"):
            request.app.state.stream_behavior_events = {}
        events_store = request.app.state.stream_behavior_events
        if session.session_id not in events_store:
            events_store[session.session_id] = []

        event_list = events_store[session.session_id]
        for track in tracks:
            if track.status != "active":
                continue
            for ev in detector.analyze(track, session.session_id):
                ev_dict = ev.to_dict()
                event_list.append(ev_dict)
                session.push_result({"type": "BEHAVIOR_EVENT", **ev_dict})
        # Cap at 200 entries
        if len(event_list) > 200:
            events_store[session.session_id] = event_list[-200:]
    except Exception:
        logger.debug("Tracker update skipped", exc_info=True)


def _session_to_response(session: StreamSession) -> SessionResponse:
    return SessionResponse(**session.to_dict())


# ── Router ─────────────────────────────────────────────────────────────────────


def build_stream_router() -> APIRouter:
    router = APIRouter(prefix="/vigil/stream", tags=["LiveStream"])

    # ── Session management ─────────────────────────────────────────────────────

    @router.post(
        "/sessions",
        response_model=SessionResponse,
        status_code=201,
        summary="Create streaming session",
    )
    async def create_stream_session(
        body: CreateSessionRequest, request: Request
    ) -> SessionResponse:
        """Create a new live streaming session."""
        sessions = _get_sessions(request)
        session = create_session(name=body.name, rules=body.rules)
        sessions[session.session_id] = session
        logger.info("Stream session created: %s (%s)", session.session_id, session.name)
        return _session_to_response(session)

    @router.get(
        "/sessions",
        response_model=list[SessionResponse],
        summary="List streaming sessions",
    )
    async def list_stream_sessions(request: Request) -> list[SessionResponse]:
        """List all streaming sessions (active and stopped)."""
        sessions = _get_sessions(request)
        return [_session_to_response(s) for s in sessions.values()]

    @router.delete(
        "/sessions/{sid}",
        summary="Stop and remove a streaming session",
    )
    async def delete_stream_session(sid: str, request: Request) -> dict[str, str]:
        """Stop a session and remove it from the registry."""
        sessions = _get_sessions(request)
        session = _get_session(request, sid)
        session.stop()
        del sessions[sid]
        logger.info("Stream session deleted: %s", sid)
        return {"session_id": sid, "status": "stopped"}

    # ── Frame push (sync REST) ─────────────────────────────────────────────────

    @router.post(
        "/sessions/{sid}/frame",
        response_model=FrameResultResponse,
        summary="Push a JPEG frame for inference",
    )
    async def push_frame(sid: str, request: Request) -> FrameResultResponse:
        """Push a raw JPEG frame and receive detection + violation results.

        The request body must be a JPEG image (Content-Type: image/jpeg).
        """
        session = _get_session(request, sid)
        if not session.is_active:
            raise HTTPException(status_code=410, detail="Session is stopped")

        frame_bytes = await request.body()
        if not frame_bytes:
            raise HTTPException(status_code=400, detail="Empty frame body")

        inferrer = _get_inferrer(request)
        result = await run_in_threadpool(inferrer.infer, frame_bytes, session.rules)

        session.frame_count += 1
        session.last_frame_jpeg = result["annotated_jpeg"]
        session.last_result = {k: v for k, v in result.items() if k != "annotated_jpeg"}

        sse_payload: dict[str, Any] = {
            "session_id": sid,
            "frame_count": session.frame_count,
            **session.last_result,
        }
        session.push_result(sse_payload)
        frame_ts = time.time()
        await run_in_threadpool(
            _run_reid, request, session, frame_bytes, result["detections"]
        )
        await run_in_threadpool(
            _run_tracker, request, session, result["detections"], frame_ts
        )

        return FrameResultResponse(**sse_payload)

    # ── WebSocket (bidirectional) ──────────────────────────────────────────────

    @router.websocket("/sessions/{sid}/ws")
    async def websocket_stream(sid: str, websocket: WebSocket) -> None:
        """Bidirectional WebSocket: client sends JPEG bytes, server replies with JSON."""
        sessions = _get_sessions(websocket)
        session = sessions.get(sid)
        if session is None or not session.is_active:
            await websocket.close(code=4004)
            return

        await websocket.accept()
        inferrer = _get_inferrer(websocket)
        logger.info("WebSocket connected: session=%s", sid)

        try:
            while session.is_active:
                try:
                    frame_bytes = await asyncio.wait_for(
                        websocket.receive_bytes(), timeout=10.0
                    )
                except asyncio.TimeoutError:
                    # Send a keepalive ping
                    await websocket.send_text(json.dumps({"type": "ping"}))
                    continue

                result = await run_in_threadpool(inferrer.infer, frame_bytes, session.rules)

                session.frame_count += 1
                session.last_frame_jpeg = result["annotated_jpeg"]
                session.last_result = {k: v for k, v in result.items() if k != "annotated_jpeg"}

                sse_payload: dict[str, Any] = {
                    "session_id": sid,
                    "frame_count": session.frame_count,
                    **session.last_result,
                }
                session.push_result(sse_payload)
                ws_ts = time.time()
                await run_in_threadpool(
                    _run_reid, websocket, session, frame_bytes, result["detections"]
                )
                await run_in_threadpool(
                    _run_tracker, websocket, session, result["detections"], ws_ts
                )

                await websocket.send_text(json.dumps(sse_payload))

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: session=%s", sid)
        except Exception:
            logger.exception("WebSocket error: session=%s", sid)
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    # ── SSE event stream ───────────────────────────────────────────────────────

    @router.get(
        "/sessions/{sid}/events",
        summary="SSE stream of inference results",
        response_class=StreamingResponse,
    )
    async def session_events(sid: str, request: Request) -> StreamingResponse:
        """Server-Sent Events stream.

        Each event is a JSON-encoded dict with detections and violations.
        The stream ends when the session is stopped or the client disconnects.
        """
        session = _get_session(request, sid)

        async def _generate():
            q = session.subscribe()
            try:
                yield "data: {\"type\": \"connected\"}\n\n"
                while True:
                    try:
                        item = await asyncio.wait_for(q.get(), timeout=30.0)
                    except asyncio.TimeoutError:
                        yield "data: {\"type\": \"keepalive\"}\n\n"
                        continue

                    if item is None:  # session stopped sentinel
                        yield "data: {\"type\": \"session_stopped\"}\n\n"
                        break

                    yield f"data: {json.dumps(item)}\n\n"

                    if not session.is_active:
                        break
            finally:
                session.unsubscribe(q)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── MJPEG preview ──────────────────────────────────────────────────────────

    @router.get(
        "/sessions/{sid}/preview",
        summary="MJPEG live preview with bbox overlay",
        response_class=StreamingResponse,
    )
    async def session_preview(sid: str, request: Request) -> StreamingResponse:
        """MJPEG stream of the latest annotated frame.

        Compatible with ``<img src="...">`` in browsers.
        """
        session = _get_session(request, sid)

        async def _mjpeg():
            while session.is_active:
                frame = session.last_frame_jpeg
                if frame:
                    yield _MJPEG_BOUNDARY + b"\r\n"
                    yield _MJPEG_HEADER
                    yield frame
                    yield b"\r\n"
                await asyncio.sleep(0.04)  # ~25 fps max

        return StreamingResponse(
            _mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return router
