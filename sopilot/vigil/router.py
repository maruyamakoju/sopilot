"""FastAPI router for VigilPilot endpoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

from sopilot.vigil.pipeline import VigilPipeline
from sopilot.vigil.repository import VigilRepository
from sopilot.vigil.schemas import (
    AcknowledgeRequest,
    AnalyzeResponse,
    SessionCreateRequest,
    SessionListItem,
    SessionReport,
    SessionResponse,
    SessionTemplate,
    StreamRequest,
    ViolationDetail,
    ViolationEvent,
    WebcamFrameResult,
    WebhookCreate,
    WebhookEnableRequest,
    WebhookRequest,
    WebhookResponse,
    WebhookTestResult,
)
from sopilot.vigil.vlm import build_vlm_client
from sopilot.vigil.webhook_dispatcher import WebhookDispatcher
from sopilot.vigil.webhook_repository import WebhookRepository as GlobalWebhookRepository

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}

_VIGIL_TEMPLATES = [
    {
        "id": "construction_safety",
        "name": "建設現場安全",
        "description": "建設・土木現場向けの安全ルールセット",
        "rules": [
            "ヘルメット未着用の作業者を検出",
            "安全ベルトなしで高所作業している人を検出",
            "立入禁止エリアへの侵入を検出",
            "重機の安全距離違反を検出",
        ],
        "sample_fps": 1.0,
        "severity_threshold": "warning",
    },
    {
        "id": "food_factory_hygiene",
        "name": "食品工場衛生管理",
        "description": "食品製造施設向けの衛生管理ルールセット",
        "rules": [
            "手袋未着用での食品取り扱いを検出",
            "マスク未着用の作業者を検出",
            "異物混入リスクのある行動を検出",
        ],
        "sample_fps": 0.5,
        "severity_threshold": "info",
    },
    {
        "id": "warehouse_safety",
        "name": "倉庫作業安全",
        "description": "倉庫・物流施設向けの安全ルールセット",
        "rules": [
            "フォークリフトの安全速度超過を検出",
            "歩行者とフォークリフトの接近を検出",
            "不安定な積載物の取り扱いを検出",
        ],
        "sample_fps": 1.0,
        "severity_threshold": "critical",
    },
    {
        "id": "fire_safety",
        "name": "防火・避難安全",
        "description": "消防法対応・避難経路確保向けルールセット",
        "rules": [
            "非常口・避難通路のブロックを検出",
            "消火器の設置位置の遮蔽を検出",
            "禁煙エリアでの喫煙を検出",
        ],
        "sample_fps": 0.5,
        "severity_threshold": "critical",
    },
    {
        "id": "office_security",
        "name": "オフィスセキュリティ",
        "description": "オフィス・施設向けのセキュリティルールセット",
        "rules": [
            "IDカード未着用の人物を検出",
            "許可なき立入エリアへのアクセスを検出",
            "機密書類の放置を検出",
        ],
        "sample_fps": 0.5,
        "severity_threshold": "warning",
    },
]


def _get_repo(request: Request) -> VigilRepository:
    return request.app.state.vigil_repo  # type: ignore[no-any-return]


def _get_pipeline(request: Request) -> VigilPipeline:
    return request.app.state.vigil_pipeline  # type: ignore[no-any-return]


def _get_webhook_repo(request: Request) -> GlobalWebhookRepository:
    return request.app.state.vigil_webhook_repo  # type: ignore[no-any-return]


_webhook_dispatcher = WebhookDispatcher()


def _row_to_webhook(row: dict) -> WebhookResponse:
    return WebhookResponse(
        id=row["id"],
        url=row["url"],
        name=row["name"] or "",
        min_severity=row["min_severity"],
        enabled=bool(row["enabled"]),
        created_at=row["created_at"],
        last_triggered_at=row.get("last_triggered_at"),
        trigger_count=int(row["trigger_count"] or 0),
    )


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
        acknowledged_at=row.get("acknowledged_at"),
        acknowledged_by=row.get("acknowledged_by"),
    )


def _annotate_frame_with_bboxes(
    frame_path: Path,
    bbox_groups: list[tuple[list, str, str]],
) -> bytes:
    """Draw bounding boxes on a JPEG frame and return annotated JPEG bytes.

    Parameters
    ----------
    frame_path:
        Path to the source JPEG frame.
    bbox_groups:
        List of (bboxes, rule_text, severity) tuples.
        bboxes: [[x1,y1,x2,y2], ...] in 0-1000 normalized scale.
    """
    from io import BytesIO

    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(frame_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    _colors = {
        "critical": (220, 50, 50),
        "warning": (255, 140, 0),
        "info": (50, 180, 50),
    }

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(10, h // 40))
    except (OSError, IOError):
        font = ImageFont.load_default()

    for bboxes, rule_text, severity in bbox_groups:
        rgb = _colors.get(severity, _colors["warning"])
        fill_rgba = (*rgb, 45)
        label = f"[{severity.upper()}] {rule_text[:28]}"

        for bbox in bboxes:
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            # Scale from 0-1000 to pixel coordinates
            px1 = int(x1 / 1000 * w)
            py1 = int(y1 / 1000 * h)
            px2 = int(x2 / 1000 * w)
            py2 = int(y2 / 1000 * h)

            draw.rectangle([px1, py1, px2, py2], fill=fill_rgba, outline=rgb, width=3)

            # Label background + text
            text_y = max(0, py1 - 18)
            label_w = min(len(label) * 7, w - px1)
            draw.rectangle([px1, text_y, px1 + label_w, text_y + 16], fill=(*rgb, 200))
            draw.text((px1 + 2, text_y + 1), label, fill=(255, 255, 255), font=font)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


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

    # ── RTSP live-stream ──────────────────────────────────────────────────

    @router.post(
        "/sessions/{session_id}/stream",
        summary="RTSPライブストリーム解析開始",
        response_model=AnalyzeResponse,
    )
    async def start_stream(
        session_id: int,
        body: StreamRequest,
        request: Request,
    ) -> AnalyzeResponse:
        """Start analysing an RTSP live stream for an existing session.

        The session must be in ``idle`` or ``completed`` state.  The endpoint
        returns immediately; analysis runs in a background daemon thread.
        Poll ``GET /vigil/sessions/{session_id}`` to track progress.
        """
        repo = _get_repo(request)
        pipeline = _get_pipeline(request)

        row = await run_in_threadpool(repo.get_session, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if row["status"] == "processing":
            raise HTTPException(status_code=409, detail="Session is already processing")

        pipeline.stream_async(
            session_id=session_id,
            rtsp_url=body.rtsp_url,
            rules=row["rules"],
            sample_fps=row["sample_fps"],
            severity_threshold=row["severity_threshold"],
        )

        return AnalyzeResponse(
            session_id=session_id,
            status="processing",
            message=(
                f"RTSPストリーム解析を開始しました。"
                f"GET /vigil/sessions/{session_id} でステータスを確認してください。"
                f"停止するには DELETE /vigil/sessions/{session_id}/stream を呼び出してください。"
            ),
        )

    @router.delete(
        "/sessions/{session_id}/stream",
        summary="RTSPライブストリーム停止",
    )
    async def stop_stream(session_id: int, request: Request) -> dict:
        """Stop an active RTSP live-stream analysis session.

        Sends a stop signal to the background thread.  The session status will
        transition to ``"completed"`` once the current frame finishes processing.
        Returns 404 if the session does not exist, 409 if no stream is active.
        """
        repo = _get_repo(request)
        pipeline = _get_pipeline(request)

        row = await run_in_threadpool(repo.get_session, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")

        stopped = pipeline.stop_stream(session_id)
        if not stopped:
            raise HTTPException(
                status_code=409,
                detail="No active RTSP stream found for this session",
            )

        return {
            "session_id": session_id,
            "stopped": True,
            "message": "ストリーム停止シグナルを送信しました。",
        }

    # ── Webcam single-frame (synchronous) ─────────────────────────────────

    @router.post(
        "/sessions/{session_id}/webcam-frame",
        summary="Webカメラフレーム即時解析",
        response_model=WebcamFrameResult,
    )
    async def analyze_webcam_frame(
        session_id: int,
        file: UploadFile,
        request: Request,
        store: bool = Query(default=True, description="違反が検出された場合イベントとして保存する"),
    ) -> WebcamFrameResult:
        """Synchronously analyze a single JPEG frame (e.g. from a webcam).

        The VLM runs in the request thread and the result is returned immediately.
        If violations are detected and ``store=true``, the frame is saved as a
        violation event associated with this session.
        """
        import tempfile
        import time

        repo = _get_repo(request)
        pipeline = _get_pipeline(request)

        row = await run_in_threadpool(repo.get_session, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")

        # Write upload to a temp file
        content = await file.read()
        suffix = ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            frame_path = Path(tmp.name)

        try:
            rules = row["rules"]
            severity_threshold = row["severity_threshold"]

            # Run VLM synchronously
            vlm_result = await run_in_threadpool(pipeline._vlm.analyze_frame, frame_path, rules)

            # Filter by severity threshold
            sev_order = {"info": 0, "warning": 1, "critical": 2}
            threshold_level = sev_order.get(severity_threshold, 0)
            violations_above = [
                v for v in vlm_result.violations
                if sev_order.get(v.get("severity", "warning"), 1) >= threshold_level
            ]

            event_id: int | None = None
            frame_url: str | None = None

            # Store event if violations found and store=True
            if store and violations_above:
                settings = request.app.state.settings
                frame_dir = Path(settings.data_dir) / "vigil_frames" / f"session_{session_id}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                ts = time.time()
                saved_path = frame_dir / f"webcam_{ts:.3f}.jpg"
                saved_path.write_bytes(content)

                # Get current frame count for frame_number
                cur_row = await run_in_threadpool(repo.get_session, session_id)
                frame_no = (cur_row or {}).get("total_frames_analyzed", 0)

                event_id = await run_in_threadpool(
                    repo.create_event,
                    session_id,
                    ts,
                    frame_no,
                    violations_above,
                    str(saved_path),
                )
                # Update session counters
                await run_in_threadpool(
                    repo.update_session_status,
                    session_id,
                    row["status"],
                    None,  # video_filename unchanged
                    frame_no + 1,
                    (cur_row or {}).get("violation_count", 0) + len(violations_above),
                )
                frame_url = f"/vigil/events/{event_id}/frame"

            violation_details = [
                ViolationDetail(**{k: v for k, v in viol.items() if k in ViolationDetail.model_fields})
                for viol in violations_above
            ]

            return WebcamFrameResult(
                session_id=session_id,
                has_violation=bool(violations_above),
                violations=violation_details,
                event_id=event_id,
                frame_url=frame_url,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("VLM analysis failed for session %d", session_id)
            raise HTTPException(
                status_code=503,
                detail=f"VLM backend error: {exc}",
            ) from exc
        finally:
            frame_path.unlink(missing_ok=True)

    # ── Perception reset ──────────────────────────────────────────────────

    @router.post(
        "/sessions/{session_id}/perception-reset",
        summary="知覚エンジンのセッション状態をリセット",
    )
    async def reset_perception_session(session_id: int, request: Request) -> dict:
        """Reset the Perception Engine tracking state for this session.

        Call this when starting a new webcam capture session to prevent
        tracking state from bleeding across sessions. Safe to call on
        non-perception backends (no-op).
        """
        pipeline = _get_pipeline(request)
        row = await run_in_threadpool(_get_repo(request).get_session, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")
        await run_in_threadpool(pipeline._vlm.reset_session)
        logger.info("Perception session reset for vigil session %d", session_id)
        return {"ok": True, "session_id": session_id}

    # ── PPE Status ────────────────────────────────────────────────────────────

    @router.get(
        "/sessions/{session_id}/ppe-status",
        summary="PPEステータス取得（PerceptionエンジンのポーズAI使用時）",
    )
    async def get_ppe_status(session_id: int, request: Request) -> dict:
        """Get the latest per-person PPE compliance status for this session.

        Only meaningful when the VLM backend is ``perception`` with pose
        estimation enabled.  For all other backends the response will have
        ``pose_enabled=False`` and an empty ``persons`` list.
        """
        repo = _get_repo(request)
        row = await run_in_threadpool(repo.get_session, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")

        pipeline = _get_pipeline(request)
        vlm = pipeline._vlm

        # Only PerceptionVLMClient exposes pose results
        if not hasattr(vlm, "get_pose_results"):
            return {
                "session_id": session_id,
                "pose_enabled": False,
                "persons": [],
                "summary": {"total_persons": 0, "helmets_ok": 0, "vests_ok": 0},
            }

        pose_results = await run_in_threadpool(vlm.get_pose_results)

        persons = []
        for idx, pr in enumerate(pose_results):
            persons.append({
                "index": idx,
                "has_helmet": pr.ppe.has_helmet,
                "helmet_confidence": round(pr.ppe.helmet_confidence, 4),
                "has_vest": pr.ppe.has_vest,
                "vest_confidence": round(pr.ppe.vest_confidence, 4),
                "pose_confidence": round(pr.pose_confidence, 4),
            })

        helmets_ok = sum(1 for p in persons if p["has_helmet"])
        vests_ok = sum(1 for p in persons if p["has_vest"])

        # Determine whether pose is enabled on the underlying engine
        pose_enabled = False
        if hasattr(vlm, "_engine") and hasattr(vlm._engine, "_config"):
            pose_enabled = bool(getattr(vlm._engine._config, "pose_enabled", False))
        elif hasattr(vlm, "_engine") and getattr(vlm._engine, "_pose_estimator", None) is not None:
            pose_enabled = True

        return {
            "session_id": session_id,
            "pose_enabled": pose_enabled,
            "persons": persons,
            "summary": {
                "total_persons": len(persons),
                "helmets_ok": helmets_ok,
                "vests_ok": vests_ok,
            },
        }

    @router.post(
        "/sessions/{session_id}/pose-enable",
        summary="ポーズ推定の有効/無効を切替（Perceptionエンジンのみ）",
    )
    async def enable_pose_for_session(
        session_id: int,
        request: Request,
        enabled: bool = True,
    ) -> dict:
        """Enable or disable PoseEstimator on the pipeline VLM client.

        No-op (returns ``ok=True``) for non-perception backends so the UI can
        call this unconditionally without error handling.
        """
        repo = _get_repo(request)
        row = await run_in_threadpool(repo.get_session, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")

        pipeline = _get_pipeline(request)
        vlm = pipeline._vlm

        if hasattr(vlm, "set_pose_enabled"):
            await run_in_threadpool(vlm.set_pose_enabled, enabled)
            logger.info(
                "Pose estimation %s for vigil session %d",
                "enabled" if enabled else "disabled",
                session_id,
            )

        return {"ok": True, "session_id": session_id, "pose_enabled": enabled}

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

    @router.patch(
        "/events/{event_id}/acknowledge",
        summary="違反イベントを確認済みにする",
    )
    async def acknowledge_event(
        event_id: int,
        body: AcknowledgeRequest,
        request: Request,
    ) -> dict:
        """Mark a violation event as acknowledged by an operator."""
        repo = _get_repo(request)
        updated = await run_in_threadpool(repo.acknowledge_event, event_id, body.acknowledged_by)
        if not updated:
            raise HTTPException(status_code=404, detail="Event not found")
        return {
            "event_id": event_id,
            "acknowledged": True,
            "acknowledged_by": body.acknowledged_by,
        }

    @router.get("/events/{event_id}/frame", summary="違反フレーム画像取得（bbox描画対応）")
    async def get_frame(
        event_id: int,
        request: Request,
        annotate: bool = Query(default=True, description="Qwen3-VLのbboxをフレームに描画する"),
    ) -> Response:
        repo = _get_repo(request)
        row = await run_in_threadpool(repo.get_event, event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Event not found")
        frame_path = row.get("frame_path")
        if not frame_path or not Path(frame_path).exists():
            raise HTTPException(status_code=404, detail="Frame image not found")

        # Draw bounding boxes if any violation has bbox data (Qwen3-VL backend)
        if annotate:
            bbox_groups = [
                (v["bboxes"], v.get("rule", ""), v.get("severity", "warning"))
                for v in row.get("violations", [])
                if v.get("bboxes")
            ]
            if bbox_groups:
                try:
                    annotated = await run_in_threadpool(
                        _annotate_frame_with_bboxes, Path(frame_path), bbox_groups
                    )
                    return Response(content=annotated, media_type="image/jpeg")
                except Exception:
                    logger.exception("bbox annotation failed for event %d", event_id)

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

    # ── Webhook management ────────────────────────────────────────────────────

    @router.put("/sessions/{session_id}/webhook", summary="Webhookを設定")
    async def set_webhook(session_id: int, body: WebhookRequest, request: Request) -> dict:
        repo = _get_repo(request)
        row = await run_in_threadpool(repo.get_session, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")
        await run_in_threadpool(repo.set_webhook, session_id, body.url, body.min_severity)
        return {"session_id": session_id, "url": body.url, "min_severity": body.min_severity}

    @router.delete("/sessions/{session_id}/webhook", summary="Webhookを削除")
    async def clear_webhook(session_id: int, request: Request) -> dict:
        repo = _get_repo(request)
        row = await run_in_threadpool(repo.get_session, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")
        await run_in_threadpool(repo.clear_webhook, session_id)
        return {"session_id": session_id, "cleared": True}

    @router.get("/sessions/{session_id}/webhook", summary="Webhook設定を取得")
    async def get_webhook(session_id: int, request: Request) -> dict:
        repo = _get_repo(request)
        row = await run_in_threadpool(repo.get_session, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")
        result = await run_in_threadpool(repo.get_webhook, session_id)
        if result is None:
            raise HTTPException(status_code=404, detail="No webhook configured")
        url, min_severity = result
        return {"session_id": session_id, "url": url, "min_severity": min_severity}

    # ── CSV report ────────────────────────────────────────────────────────────

    @router.get("/sessions/{session_id}/report/csv", summary="違反レポートCSVダウンロード")
    async def download_report_csv(session_id: int, request: Request):  # type: ignore[return]
        import csv
        import io
        import json as _json

        from fastapi.responses import StreamingResponse as _StreamingResponse

        repo = _get_repo(request)
        row = await run_in_threadpool(repo.get_session, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")
        events = await run_in_threadpool(repo.list_events, session_id)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "event_id", "timestamp_sec", "frame_number",
            "rule_index", "rule", "description_ja", "severity", "confidence", "bboxes",
        ])
        for ev in events:
            for v in ev.get("violations", []):
                bboxes_val = v.get("bboxes")
                bboxes_str = _json.dumps(bboxes_val) if bboxes_val else ""
                writer.writerow([
                    ev["id"], ev["timestamp_sec"], ev["frame_number"],
                    v.get("rule_index", ""), v.get("rule", ""), v.get("description_ja", ""),
                    v.get("severity", ""), v.get("confidence", ""), bboxes_str,
                ])
        output.seek(0)

        filename = f"vigil_session_{session_id}_violations.csv"
        return _StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    # ── SSE real-time feed ────────────────────────────────────────────────────

    @router.get("/sessions/{session_id}/events/stream", summary="SSEリアルタイム違反フィード")
    async def stream_events(session_id: int, request: Request):  # type: ignore[return]
        import asyncio
        import json as _json

        from fastapi.responses import StreamingResponse as _StreamingResponse

        repo = _get_repo(request)
        row = await run_in_threadpool(repo.get_session, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")

        async def _generate():  # type: ignore[return]
            last_id = 0
            while True:
                if await request.is_disconnected():
                    break
                new_events = repo.list_events_since(session_id, last_id)
                for ev in new_events:
                    data = {
                        "type": "violation",
                        "event_id": ev["id"],
                        "timestamp_sec": ev["timestamp_sec"],
                        "frame_number": ev["frame_number"],
                        "violations": ev.get("violations", []),
                        "created_at": ev["created_at"],
                    }
                    yield f"data: {_json.dumps(data)}\n\n"
                    last_id = ev["id"]
                cur = repo.get_session(session_id)
                status = cur["status"] if cur else "failed"
                if status in ("completed", "failed"):
                    yield f"data: {_json.dumps({'type': 'status_change', 'status': status})}\n\n"
                    break
                yield f"data: {_json.dumps({'type': 'heartbeat'})}\n\n"
                await asyncio.sleep(2)

        return _StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Templates ─────────────────────────────────────────────────────────────

    @router.get(
        "/templates",
        summary="セッションテンプレート一覧",
        response_model=list[SessionTemplate],
    )
    async def list_templates() -> list[SessionTemplate]:
        """Return predefined rule-set templates for common surveillance scenarios."""
        return [SessionTemplate(**t) for t in _VIGIL_TEMPLATES]

    # ── Global webhook management ──────────────────────────────────────────────

    @router.post(
        "/webhooks",
        summary="Webhookを登録",
        response_model=WebhookResponse,
        status_code=201,
    )
    async def create_global_webhook(body: WebhookCreate, request: Request) -> WebhookResponse:
        """Register a new global webhook for violation alert delivery."""
        if not body.url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=422,
                detail="url must start with http:// or https://",
            )
        wh_repo = _get_webhook_repo(request)
        row = await run_in_threadpool(
            wh_repo.create, body.url, body.name, body.secret, body.min_severity
        )
        return _row_to_webhook(row)

    @router.get(
        "/webhooks",
        summary="Webhook一覧",
        response_model=list[WebhookResponse],
    )
    async def list_global_webhooks(request: Request) -> list[WebhookResponse]:
        """Return all registered global webhooks."""
        wh_repo = _get_webhook_repo(request)
        rows = await run_in_threadpool(wh_repo.list_all)
        return [_row_to_webhook(r) for r in rows]

    @router.get(
        "/webhooks/{webhook_id}",
        summary="Webhook詳細",
        response_model=WebhookResponse,
    )
    async def get_global_webhook(webhook_id: int, request: Request) -> WebhookResponse:
        """Return a single registered webhook by ID."""
        wh_repo = _get_webhook_repo(request)
        row = await run_in_threadpool(wh_repo.get, webhook_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Webhook not found")
        return _row_to_webhook(row)

    @router.delete(
        "/webhooks/{webhook_id}",
        summary="Webhookを削除",
        status_code=204,
    )
    async def delete_global_webhook(webhook_id: int, request: Request) -> Response:
        """Delete a registered webhook by ID."""
        wh_repo = _get_webhook_repo(request)
        deleted = await run_in_threadpool(wh_repo.delete, webhook_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Webhook not found")
        return Response(status_code=204)

    @router.patch(
        "/webhooks/{webhook_id}/enable",
        summary="Webhookを有効/無効化",
        response_model=WebhookResponse,
    )
    async def toggle_global_webhook(
        webhook_id: int,
        body: WebhookEnableRequest,
        request: Request,
    ) -> WebhookResponse:
        """Enable or disable a webhook without deleting it."""
        wh_repo = _get_webhook_repo(request)
        updated = await run_in_threadpool(wh_repo.update_enabled, webhook_id, body.enabled)
        if not updated:
            raise HTTPException(status_code=404, detail="Webhook not found")
        row = await run_in_threadpool(wh_repo.get, webhook_id)
        return _row_to_webhook(row)  # type: ignore[arg-type]

    @router.post(
        "/webhooks/{webhook_id}/test",
        summary="Webhookテスト送信",
        response_model=WebhookTestResult,
    )
    async def test_global_webhook(webhook_id: int, request: Request) -> WebhookTestResult:
        """Send a test payload to a registered webhook and return the result."""
        wh_repo = _get_webhook_repo(request)
        row = await run_in_threadpool(wh_repo.get, webhook_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Webhook not found")
        result = await run_in_threadpool(_webhook_dispatcher.test_webhook, row)
        return WebhookTestResult(**result)

    return router
