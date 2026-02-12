from __future__ import annotations

import threading

from .config import get_settings
from .db import Database
from .service import SopilotService
from .storage import ensure_directories

_service_lock = threading.Lock()
_service: SopilotService | None = None


def _get_service() -> SopilotService:
    global _service
    if _service is not None:
        return _service
    with _service_lock:
        if _service is None:
            settings = get_settings()
            ensure_directories(settings)
            db = Database(settings.db_path)
            _service = SopilotService(settings=settings, db=db, runtime_mode="worker")
    return _service


def run_ingest_job(job_id: str) -> None:
    _get_service().run_ingest_job(job_id)


def run_score_job(job_id: str) -> None:
    _get_service().run_score_job(job_id)


def run_training_job(job_id: str) -> None:
    _get_service().run_training_job(job_id)


def shutdown_worker_service() -> None:
    global _service
    with _service_lock:
        if _service is not None:
            _service.shutdown()
            _service = None
