from __future__ import annotations

import base64
from contextlib import asynccontextmanager
import hmac
from pathlib import Path
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.responses import JSONResponse

from .config import get_settings
from .db import Database
from .schemas import (
    AuditTrailItem,
    AuditTrailResponse,
    AuditExportResponse,
    HealthResponse,
    NightlyStatusResponse,
    QueueMetricsResponse,
    ScoreCreateResponse,
    ScoreRequest,
    ScoreResultResponse,
    SearchResponse,
    TrainingCreateResponse,
    TrainingResultResponse,
    VideoDeleteResponse,
    VideoInfoResponse,
    VideoIngestCreateResponse,
    VideoIngestResultResponse,
    VideoListResponse,
)
from .service import SopilotService
from .storage import ensure_directories
from .utils import safe_filename


VALID_ROLES = {"gold", "trainee", "audit"}
ROLE_ORDER = {"viewer": 1, "operator": 2, "admin": 3}
UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
PUBLIC_PATHS = {
    "/",
    "/health",
    "/metrics",  # Prometheus metrics endpoint
    "/ui",
    "/docs",
    "/docs/oauth2-redirect",
    "/openapi.json",
    "/redoc",
}


def _is_auth_configured(
    *,
    api_token: str,
    api_role_tokens: list[tuple[str, str]],
    basic_user: str,
    basic_password: str,
) -> bool:
    if api_token.strip():
        return True
    if any(role and token for role, token in api_role_tokens):
        return True
    if basic_user.strip() and basic_password.strip():
        return True
    return False


def _safe_filename(name: str) -> str:
    return safe_filename(name, fallback="upload.mp4")


def _normalize_role(role: str, *, default: str = "viewer") -> str:
    value = role.strip().lower()
    return value if value in ROLE_ORDER else default


def _has_required_role(role: str, required: str) -> bool:
    lhs = ROLE_ORDER.get(_normalize_role(role, default="viewer"), 0)
    rhs = ROLE_ORDER.get(_normalize_role(required, default="viewer"), 0)
    return lhs >= rhs


def _parse_role_tokens(spec: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    raw = spec.strip()
    if not raw:
        return out
    for token in raw.replace(";", ",").split(","):
        item = token.strip()
        if not item:
            continue
        role, sep, value = item.partition(":")
        if not sep:
            continue
        role_norm = _normalize_role(role, default="")
        token_value = value.strip()
        if not role_norm or not token_value:
            continue
        out.append((role_norm, token_value))
    return out


def _is_public_path(path: str) -> bool:
    if path in PUBLIC_PATHS:
        return True
    if len(path) > 1 and path.endswith("/"):
        return path[:-1] in PUBLIC_PATHS
    return False


def _require_role(request: Request, required: str) -> None:
    actor = getattr(request.state, "actor", "unknown")
    role = _normalize_role(getattr(request.state, "role", "viewer"), default="viewer")
    if _has_required_role(role, required):
        return
    raise HTTPException(
        status_code=403,
        detail=f"{required} role required for this operation (actor={actor}, role={role})",
    )


def _map_service_error(
    exc: Exception,
    *,
    value_error_status: int = 400,
    value_error_detail: str | None = None,
    runtime_error_status: int = 500,
    runtime_error_detail: str | None = None,
    default_status: int = 500,
    default_detail: str = "internal server error",
) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ValueError):
        detail = value_error_detail if value_error_detail is not None else str(exc)
        return HTTPException(status_code=value_error_status, detail=detail)
    if isinstance(exc, RuntimeError):
        detail = runtime_error_detail if runtime_error_detail is not None else str(exc)
        return HTTPException(status_code=runtime_error_status, detail=detail)
    return HTTPException(status_code=default_status, detail=default_detail)


async def _persist_upload(upload: UploadFile, target: Path, max_bytes: int | None = None) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with target.open("wb") as f:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise ValueError(f"upload too large: {total} bytes exceeds limit {max_bytes} bytes")
    finally:
        await upload.close()
    return total


def _parse_basic(auth_header: str) -> tuple[str, str] | None:
    if not auth_header.startswith("Basic "):
        return None
    raw = auth_header[6:].strip()
    if not raw:
        return None
    try:
        decoded = base64.b64decode(raw).decode("utf-8")
    except Exception:
        return None
    user, sep, pwd = decoded.partition(":")
    if not sep:
        return None
    return user, pwd


def _resolve_identity(
    auth_header: str,
    *,
    api_token: str,
    api_token_role: str,
    api_role_tokens: list[tuple[str, str]],
    basic_user: str,
    basic_password: str,
    basic_role: str,
    auth_default_role: str,
) -> tuple[str, str] | None:
    token = api_token.strip()
    user = basic_user.strip()
    pwd = basic_password.strip()
    default_role = _normalize_role(auth_default_role, default="admin")
    token_entries = [(role, value) for role, value in api_role_tokens if role and value]
    auth_enabled = bool(token_entries) or bool(token) or bool(user and pwd)
    if not auth_enabled:
        return "anonymous", default_role

    if auth_header.startswith("Bearer "):
        supplied = auth_header[7:].strip()
        if supplied:
            for role, value in token_entries:
                if hmac.compare_digest(supplied, value):
                    return f"token:{role}", role
            if token and hmac.compare_digest(supplied, token):
                return "token:api", _normalize_role(api_token_role, default="admin")

    pair = _parse_basic(auth_header)
    if pair is not None and user and pwd:
        supplied_user, supplied_pwd = pair
        if hmac.compare_digest(supplied_user, user) and hmac.compare_digest(supplied_pwd, pwd):
            return f"basic:{supplied_user}", _normalize_role(basic_role, default="admin")

    return None


async def _handle_upload_enqueue(
    *,
    request: Request,
    settings,
    service: SopilotService,
    file: UploadFile,
    task_id: str,
    role: str,
    site_id: str | None,
    camera_id: str | None,
    operator_id_hash: str | None,
) -> VideoIngestCreateResponse:
    role = role.lower().strip()
    if role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"invalid role: {role}")

    incoming = (
        settings.raw_dir
        / ".incoming"
        / f"{uuid.uuid4().hex}_{_safe_filename(file.filename or 'upload.mp4')}"
    )
    try:
        max_bytes = max(1, int(settings.upload_max_mb)) * 1024 * 1024
        size = await _persist_upload(file, incoming, max_bytes=max_bytes)
        if size <= 0:
            incoming.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="empty upload")

        payload = service.enqueue_ingest_from_path(
            file_name=file.filename or "upload.mp4",
            staged_path=incoming,
            task_id=task_id,
            role=role,
            requested_by=request.state.actor,
            site_id=site_id,
            camera_id=camera_id,
            operator_id_hash=operator_id_hash,
        )
        return VideoIngestCreateResponse(**payload)
    except HTTPException:
        incoming.unlink(missing_ok=True)
        raise
    except ValueError as exc:
        incoming.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        incoming.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="upload enqueue failed") from exc


def create_app() -> FastAPI:
    settings = get_settings()
    ensure_directories(settings)
    api_role_tokens = _parse_role_tokens(settings.api_role_tokens)

    db = Database(settings.db_path)
    service = SopilotService(settings=settings, db=db)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            app.state.service.shutdown()

    app = FastAPI(title="SOPilot MVP", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    ui_index = Path(__file__).resolve().parent / "ui" / "index.html"

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if _is_public_path(request.url.path):
            request.state.actor = "anonymous"
            request.state.role = "viewer"
            return await call_next(request)

        auth_is_configured = _is_auth_configured(
            api_token=settings.api_token,
            api_role_tokens=api_role_tokens,
            basic_user=settings.basic_user,
            basic_password=settings.basic_password,
        )
        if not auth_is_configured:
            if settings.auth_required:
                return JSONResponse(
                    status_code=503,
                    content={"detail": "authentication is required but credentials are not configured"},
                )
            request.state.actor = "anonymous"
            request.state.role = _normalize_role(settings.auth_default_role, default="viewer")
            return await call_next(request)

        identity = _resolve_identity(
            request.headers.get("authorization", ""),
            api_token=settings.api_token,
            api_token_role=settings.api_token_role,
            api_role_tokens=api_role_tokens,
            basic_user=settings.basic_user,
            basic_password=settings.basic_password,
            basic_role=settings.basic_role,
            auth_default_role=settings.auth_default_role,
        )
        if identity is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "authentication required"},
                headers={"WWW-Authenticate": 'Bearer realm="sopilot", Basic realm="sopilot"'},
            )
        request.state.actor = identity[0]
        request.state.role = identity[1]
        return await call_next(request)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        db_ok = True
        try:
            db.list_videos(limit=1)
        except Exception:
            db_ok = False
        return HealthResponse(status="ok" if db_ok else "degraded", db=db_ok)

    @app.get("/")
    def root() -> dict:
        return {"message": "open /ui for Field PoC Console"}

    @app.get("/ui")
    def ui() -> FileResponse:
        if not ui_index.exists():
            raise HTTPException(status_code=404, detail="ui not found")
        return FileResponse(str(ui_index), media_type="text/html")

    @app.post("/videos", response_model=VideoIngestCreateResponse)
    async def upload_video(
        request: Request,
        file: UploadFile = File(...),
        task_id: str = Form(...),
        role: str = Form("trainee"),
        site_id: str | None = Form(None),
        camera_id: str | None = Form(None),
        operator_id_hash: str | None = Form(None),
    ) -> VideoIngestCreateResponse:
        _require_role(request, "operator")
        return await _handle_upload_enqueue(
            request=request,
            settings=settings,
            service=app.state.service,
            file=file,
            task_id=task_id,
            role=role,
            site_id=site_id,
            camera_id=camera_id,
            operator_id_hash=operator_id_hash,
        )

    @app.post("/gold", response_model=VideoIngestCreateResponse)
    async def upload_gold(
        request: Request,
        file: UploadFile = File(...),
        task_id: str = Form(...),
        site_id: str | None = Form(None),
        camera_id: str | None = Form(None),
        operator_id_hash: str | None = Form(None),
    ) -> VideoIngestCreateResponse:
        _require_role(request, "operator")
        return await _handle_upload_enqueue(
            request=request,
            settings=settings,
            service=app.state.service,
            file=file,
            task_id=task_id,
            role="gold",
            site_id=site_id,
            camera_id=camera_id,
            operator_id_hash=operator_id_hash,
        )

    @app.get("/videos/jobs/{ingest_job_id}", response_model=VideoIngestResultResponse)
    def get_ingest_job(ingest_job_id: str) -> VideoIngestResultResponse:
        item = app.state.service.get_ingest_job(ingest_job_id)
        if item is None:
            raise HTTPException(status_code=404, detail="ingest job not found")
        return VideoIngestResultResponse(**item)

    @app.post("/score", response_model=ScoreCreateResponse)
    def create_score(payload: ScoreRequest, request: Request) -> ScoreCreateResponse:
        _require_role(request, "operator")
        try:
            result = app.state.service.enqueue_score(
                gold_video_id=payload.gold_video_id,
                trainee_video_id=payload.trainee_video_id,
                requested_by=request.state.actor,
            )
            return ScoreCreateResponse(**result)
        except Exception as exc:
            raise _map_service_error(
                exc,
                value_error_status=400,
                runtime_error_status=500,
                default_status=500,
                default_detail="score job enqueue failed",
            ) from exc

    @app.get("/score/{score_job_id}", response_model=ScoreResultResponse)
    def get_score(score_job_id: str) -> ScoreResultResponse:
        item = app.state.service.get_score(score_job_id)
        if item is None:
            raise HTTPException(status_code=404, detail="score job not found")
        return ScoreResultResponse(**item)

    @app.get("/score/{score_job_id}/report.pdf")
    def get_score_pdf(score_job_id: str) -> FileResponse:
        try:
            path = app.state.service.build_score_pdf(score_job_id)
            return FileResponse(str(path), media_type="application/pdf", filename=f"score_{score_job_id}.pdf")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/search", response_model=SearchResponse)
    def search(
        task_id: str,
        video_id: int,
        clip_idx: int,
        k: int = Query(default=5, ge=1, le=50),
    ) -> SearchResponse:
        try:
            result = app.state.service.search(
                task_id=task_id,
                video_id=video_id,
                clip_idx=clip_idx,
                k=k,
            )
            return SearchResponse(**result)
        except Exception as exc:
            raise _map_service_error(
                exc,
                value_error_status=400,
                runtime_error_status=500,
                default_status=500,
                default_detail="search failed",
            ) from exc

    @app.get("/videos", response_model=VideoListResponse)
    def list_videos(
        task_id: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> VideoListResponse:
        items = app.state.service.list_videos(task_id=task_id, limit=limit)
        return VideoListResponse(items=[VideoInfoResponse(**x) for x in items])

    @app.get("/videos/{video_id}", response_model=VideoInfoResponse)
    def get_video(video_id: int) -> VideoInfoResponse:
        item = app.state.service.get_video_info(video_id)
        if item is None:
            raise HTTPException(status_code=404, detail="video not found")
        return VideoInfoResponse(**item)

    @app.get("/videos/{video_id}/file")
    def get_video_file(video_id: int) -> FileResponse:
        try:
            path = app.state.service.get_video_file_path(video_id)
            return FileResponse(str(path))
        except Exception as exc:
            raise _map_service_error(
                exc,
                value_error_status=404,
                runtime_error_status=500,
                default_status=500,
                default_detail="video file fetch failed",
            ) from exc

    @app.delete("/videos/{video_id}", response_model=VideoDeleteResponse)
    def delete_video(video_id: int, request: Request) -> VideoDeleteResponse:
        _require_role(request, "admin")
        try:
            payload = app.state.service.delete_video(video_id)
            return VideoDeleteResponse(**payload)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="video delete failed") from exc

    @app.post("/train/nightly", response_model=TrainingCreateResponse)
    def trigger_training(request: Request) -> TrainingCreateResponse:
        _require_role(request, "admin")
        try:
            result = app.state.service.enqueue_training(
                trigger="manual",
                requested_by=request.state.actor,
            )
            return TrainingCreateResponse(**result)
        except Exception as exc:
            raise _map_service_error(
                exc,
                value_error_status=400,
                runtime_error_status=500,
                default_status=500,
                default_detail="training job enqueue failed",
            ) from exc

    @app.get("/train/jobs/{training_job_id}", response_model=TrainingResultResponse)
    def get_training(training_job_id: str) -> TrainingResultResponse:
        item = app.state.service.get_training_job(training_job_id)
        if item is None:
            raise HTTPException(status_code=404, detail="training job not found")
        return TrainingResultResponse(**item)

    @app.get("/train/nightly/status", response_model=NightlyStatusResponse)
    def nightly_status() -> NightlyStatusResponse:
        return NightlyStatusResponse(**app.state.service.get_nightly_status())

    @app.get("/audit/trail", response_model=AuditTrailResponse)
    def get_audit_trail(limit: int = Query(default=100, ge=1, le=1000)) -> AuditTrailResponse:
        items = app.state.service.get_audit_trail(limit=limit)
        return AuditTrailResponse(items=[AuditTrailItem(**x) for x in items])

    @app.post("/audit/export", response_model=AuditExportResponse)
    def export_audit_trail(
        request: Request,
        limit: int = Query(default=500, ge=1, le=5000),
    ) -> AuditExportResponse:
        _require_role(request, "admin")
        try:
            payload = app.state.service.export_signed_audit_trail(limit=limit)
            return AuditExportResponse(**payload)
        except Exception as exc:
            raise _map_service_error(
                exc,
                value_error_status=400,
                runtime_error_status=503,
                default_status=500,
                default_detail="audit export failed",
            ) from exc

    @app.get("/audit/export/{export_id}/file")
    def get_audit_export_file(export_id: str, request: Request) -> FileResponse:
        _require_role(request, "admin")
        try:
            path = app.state.service.get_audit_export_path(export_id)
            return FileResponse(
                str(path),
                media_type="application/json",
                filename=path.name,
            )
        except Exception as exc:
            raise _map_service_error(
                exc,
                value_error_status=404,
                runtime_error_status=500,
                default_status=500,
                default_detail="audit export file fetch failed",
            ) from exc

    @app.get("/ops/queue", response_model=QueueMetricsResponse)
    def get_queue_metrics(request: Request) -> QueueMetricsResponse:
        _require_role(request, "operator")
        return QueueMetricsResponse(**app.state.service.get_queue_metrics())

    @app.get("/metrics")
    def prometheus_metrics():
        """Prometheus metrics endpoint for monitoring."""
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            from .metrics import collect_gpu_metrics
            # Collect latest GPU metrics before exposing
            collect_gpu_metrics()
            metrics_output = generate_latest()
            return JSONResponse(
                content=metrics_output.decode("utf-8"),
                media_type=CONTENT_TYPE_LATEST,
            )
        except ImportError:
            return JSONResponse(
                content={"error": "prometheus-client not installed"},
                status_code=503,
            )

    return app
