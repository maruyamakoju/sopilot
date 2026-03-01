"""FastAPI router for VigilPilot endpoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from sopilot.vigil.pipeline import VigilPipeline
from sopilot.vigil.repository import VigilRepository
from sopilot.vigil.schemas import (
    AnalyzeResponse,
    SessionCreateRequest,
    SessionListItem,
    SessionReport,
    SessionResponse,
    ViolationDetail,
    ViolationEvent,
)
from sopilot.vigil.vlm import build_vlm_client

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


def _get_repo(request: Request) -> VigilRepository:
    return request.app.state.vigil_repo  # type: ignore[no-any-return]


def _get_pipeline(request: Request) -> VigilPipeline:
    return request.app.state.vigil_pipeline  # type: ignore[no-any-return]


def _row_to_session(row: dict) -> SessionResponse:
    return SessionResponse(
        session_id=row["id"],
        name=row["name"],
        rules=row["rules"],
        sample_fps=row["sample_fps"],
        severity_threshold=row["severity_threshold"],
        status=row["status"],
        video_filename=row.get("video_filename"),
        total_frames_analyzed=row["total_frames_analyzed"],
        violation_count=row["violation_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_event(row: dict) -> ViolationEvent:
    return ViolationEvent(
        event_id=row["id"],
        session_id=row["session_id"],
        timestamp_sec=row["timestamp_sec"],
        frame_number=row["frame_number"],
        violations=[ViolationDetail(**v) for v in row["violations"]],
        frame_url=(
            f"/vigil/events/{row['id']}/frame"
            if row.get("frame_path") and Path(row["frame_path"]).exists()
            else None
        ),
        created_at=row["created_at"],
    )


def build_vigil_router() -> APIRouter:
    router = APIRouter(prefix="/vigil", tags=["VigilPilot"])

    # ── Sessions ──────────────────────────────────────────────────────────

    @router.post("/sessions", summary="監視セッション作成", response_model=SessionResponse)
    async def create_session(body: SessionCreateRequest, request: Request) -> SessionResponse:
        repo = _get_repo(request)
        session_id = await run_in_threadpool(
            repo.create_session,
            body.name,
            body.rules,
            body.sample_fps,
            body.severity_threshold,
        )
        row = await run_in_threadpool(repo.get_session, session_id)
        return _row_to_session(row)  # type: ignore[arg-type]

    @router.get("/sessions", summary="セッション一覧", response_model=list[SessionListItem])
    async def list_sessions(request: Request) -> list[SessionListItem]:
        repo = _get_repo(request)
        rows = await run_in_threadpool(repo.list_sessions)
        return [
            SessionListItem(
                session_id=r["id"],
                name=r["name"],
                status=r["status"],
                violation_count=r["violation_count"],
                total_frames_analyzed=r["total_frames_analyzed"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    @router.get("/sessions/{session_id}", summary="セッション詳細", response_model=SessionResponse)
    async def get_session(session_id: int, request: Request) -> SessionResponse:
        repo = _get_repo(request)
        row = await run_in_threadpool(repo.get_session, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return _row_to_session(row)

    @router.delete("/sessions/{session_id}", summary="セッション削除")
    async def delete_session(session_id: int, request: Request) -> dict:
        repo = _get_repo(request)
        ok = await run_in_threadpool(repo.delete_session, session_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": True, "session_id": session_id}

    # ── Video analysis ────────────────────────────────────────────────────

    @router.post(
        "/sessions/{session_id}/analyze",
        summary="動画をアップロードして解析開始",
        response_model=AnalyzeResponse,
    )
    async def analyze_video(
        session_id: int,
        file: UploadFile,
        request: Request,
    ) -> AnalyzeResponse:
        repo = _get_repo(request)
        pipeline = _get_pipeline(request)

        row = await run_in_threadpool(repo.get_session, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if row["status"] == "processing":
            raise HTTPException(status_code=409, detail="Session is already processing")

        # Save uploaded video to vigil uploads dir
        settings = request.app.state.settings
        upload_dir = Path(settings.data_dir) / "vigil_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        video_path = upload_dir / f"session_{session_id}_{file.filename}"

        content = await file.read()
        video_path.write_bytes(content)
        logger.info("vigil upload: session=%d  file=%s  size=%d", session_id, file.filename, len(content))

        # Launch analysis in background thread
        pipeline.analyze_async(
            session_id=session_id,
            video_path=video_path,
            rules=row["rules"],
            sample_fps=row["sample_fps"],
            severity_threshold=row["severity_threshold"],
            cleanup_video=True,
        )

        return AnalyzeResponse(
            session_id=session_id,
            status="processing",
            message=f"解析を開始しました。GET /vigil/sessions/{session_id} でステータスを確認してください。",
        )

    # ── Events ────────────────────────────────────────────────────────────

    @router.get(
        "/sessions/{session_id}/events",
        summary="違反イベント一覧",
        response_model=list[ViolationEvent],
    )
    async def list_events(session_id: int, request: Request) -> list[ViolationEvent]:
        repo = _get_repo(request)
        rows = await run_in_threadpool(repo.list_events, session_id)
        return [_row_to_event(r) for r in rows]

    @router.get("/events/{event_id}/frame", summary="違反フレーム画像取得")
    async def get_frame(event_id: int, request: Request) -> FileResponse:
        repo = _get_repo(request)
        row = await run_in_threadpool(repo.get_event, event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Event not found")
        frame_path = row.get("frame_path")
        if not frame_path or not Path(frame_path).exists():
            raise HTTPException(status_code=404, detail="Frame image not found")
        return FileResponse(frame_path, media_type="image/jpeg")

    # ── Report ────────────────────────────────────────────────────────────

    @router.get(
        "/sessions/{session_id}/report",
        summary="違反レポート生成",
        response_model=SessionReport,
    )
    async def get_report(session_id: int, request: Request) -> SessionReport:
        repo = _get_repo(request)
        row = await run_in_threadpool(repo.get_session, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")

        events = await run_in_threadpool(repo.list_events, session_id)

        # Breakdown by severity and rule
        severity_breakdown: dict[str, int] = {"critical": 0, "warning": 0, "info": 0}
        rule_breakdown: dict[str, int] = {}
        for ev in events:
            for v in ev["violations"]:
                sev = v.get("severity", "warning")
                severity_breakdown[sev] = severity_breakdown.get(sev, 0) + 1
                rule_text = v.get("rule", "unknown")
                rule_breakdown[rule_text] = rule_breakdown.get(rule_text, 0) + 1

        return SessionReport(
            session_id=row["id"],
            name=row["name"],
            rules=row["rules"],
            status=row["status"],
            video_filename=row.get("video_filename"),
            total_frames_analyzed=row["total_frames_analyzed"],
            violation_count=row["violation_count"],
            severity_breakdown=severity_breakdown,
            rule_breakdown=rule_breakdown,
            events=[_row_to_event(e) for e in events],
            created_at=row["created_at"],
        )

    return router
