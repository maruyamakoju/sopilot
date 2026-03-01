import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from sopilot import __version__
from sopilot.api.auth import ApiKeyMiddleware
from sopilot.api.correlation import CorrelationIDMiddleware
from sopilot.api.rate_limit import RateLimitMiddleware
from sopilot.api.routes import build_router
from sopilot.config import Settings
from sopilot.database import Database
from sopilot.logging_config import setup_logging
from sopilot.services.embedder import build_embedder
from sopilot.services.score_queue import ScoreJobQueue
from sopilot.services.sopilot_service import SOPilotService
from sopilot.services.storage import FileStorage
from sopilot.services.video_processor import VideoProcessor
from sopilot.vigil import build_vigil_router
from sopilot.vigil.pipeline import VigilPipeline
from sopilot.vigil.repository import VigilRepository
from sopilot.vigil.vlm import build_vlm_client

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = Settings.from_env()
    setup_logging(level=settings.log_level, use_json=settings.log_json)
    database = Database(settings.database_path)
    embedder = build_embedder(settings)
    video_processor = VideoProcessor(
        sample_fps=settings.sample_fps,
        clip_seconds=settings.clip_seconds,
        frame_size=settings.frame_size,
        embedder=embedder,
    )
    storage = FileStorage(settings.raw_video_dir, max_upload_bytes=settings.max_upload_mb * 1024 * 1024)
    service = SOPilotService(
        settings=settings,
        database=database,
        storage=storage,
        video_processor=video_processor,
    )
    queue = ScoreJobQueue(service, worker_count=settings.score_worker_threads, max_retries=settings.score_job_max_retries)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "starting SOPilot v%s task=%s enforce=%s workers=%s embedder=%s "
            "data_dir=%s max_upload=%sMB rate_limit=%srpm api_key=%s",
            __version__,
            settings.primary_task_id,
            settings.enforce_primary_task,
            settings.score_worker_threads,
            embedder.name,
            settings.data_dir,
            settings.max_upload_mb,
            settings.rate_limit_rpm,
            "configured" if settings.api_key else "disabled",
        )
        queue.start()
        try:
            yield
        finally:
            queue.stop()
            database.close()
            logger.info("SOPilot v%s shutdown complete", __version__)

    app = FastAPI(
        title="SOPilot",
        version=__version__,
        description="On-prem SOP (Standard Operating Procedure) evaluation service powered by video embeddings. "
                    "Upload gold and trainee videos, run DTW-based alignment scoring, and generate audit reports.",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "Health", "description": "Service health and observability endpoints"},
            {"name": "Videos", "description": "Video upload, listing, and streaming"},
            {"name": "Scoring", "description": "Score job management, results, and reports"},
            {"name": "Search", "description": "Clip similarity search across videos"},
            {"name": "Configuration", "description": "Task profile and dataset configuration"},
            {"name": "Admin", "description": "Database administration and maintenance"},
        ],
    )
    # Middleware execution order (outermost → innermost):
    #   1. CorrelationIDMiddleware — assign X-Request-ID + security headers
    #   2. CORSMiddleware          — handle preflight / CORS headers
    #   3. ApiKeyMiddleware        — authenticate before consuming rate budget
    #   4. RateLimitMiddleware     — only rate-limit authenticated requests
    #
    # FastAPI processes middleware in REVERSE order of add_middleware() calls,
    # so we add innermost first.
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_rpm,
        burst=settings.rate_limit_burst,
    )
    app.add_middleware(ApiKeyMiddleware, api_key=settings.api_key)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(CorrelationIDMiddleware)
    # VigilPilot — surveillance camera violation detection
    vigil_repo = VigilRepository(settings.database_path)
    vigil_vlm_key = os.environ.get("VIGIL_VLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    vigil_frames_root = Path(settings.data_dir) / "vigil_frames"
    if vigil_vlm_key:
        vigil_vlm = build_vlm_client(api_key=vigil_vlm_key)
    else:
        vigil_vlm = build_vlm_client(api_key="not-configured")  # will fail gracefully at analysis time
    vigil_pipeline = VigilPipeline(repo=vigil_repo, vlm=vigil_vlm, frames_root=vigil_frames_root)

    app.state.sopilot_service = service
    app.state.score_queue = queue
    app.state.settings = settings
    app.state.embedder = embedder
    app.state.vigil_repo = vigil_repo
    app.state.vigil_pipeline = vigil_pipeline
    app.include_router(build_vigil_router())
    app.include_router(build_router(), prefix="/api/v1")
    # Backward-compatible: mount same routes at root for existing clients
    app.include_router(build_router())

    @app.get("/", response_model=None)
    def ui_index() -> FileResponse | dict[str, str]:
        index_path = Path(settings.ui_dir) / "index.html"
        if not index_path.exists():
            return {"message": "UI not found"}
        return FileResponse(index_path)

    return app
