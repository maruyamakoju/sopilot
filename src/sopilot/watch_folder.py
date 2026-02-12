from __future__ import annotations

import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import get_settings
from .db import Database
from .service import SopilotService
from .storage import ensure_directories
from .utils import now_tag, write_json

logger = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}
ROLES = {"gold", "trainee", "audit"}


_now_tag = now_tag


def _is_hidden_path(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    return any(part.startswith(".") for part in rel.parts)


def _scan_candidates(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _is_hidden_path(path, root):
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        out.append(path)
    return sorted(out)


def _derive_task_role(
    *,
    root: Path,
    source_path: Path,
    default_task_id: str,
    default_role: str,
) -> tuple[str, str]:
    rel = source_path.relative_to(root)
    parts = [p.strip() for p in rel.parts]
    task_id = default_task_id
    role = default_role

    if len(parts) >= 2:
        if parts[1].lower() in ROLES:
            task_id = parts[0] or task_id
            role = parts[1].lower()
        else:
            task_id = parts[0] or task_id
            if len(parts) >= 3 and parts[2].lower() in ROLES:
                role = parts[2].lower()
            elif parts[-2].lower() in ROLES:
                role = parts[-2].lower()
    elif len(parts) == 1:
        # Single-file inbox requires explicit default task.
        task_id = default_task_id

    if role not in ROLES:
        role = "trainee"
    return task_id.strip(), role


_write_json = write_json


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    ensure_directories(settings)

    if not settings.watch_enabled:
        raise RuntimeError("folder watch is disabled; set SOPILOT_WATCH_ENABLED=true")

    watch_dir = settings.watch_dir.resolve()
    watch_dir.mkdir(parents=True, exist_ok=True)
    processing_dir = watch_dir / ".processing"
    processed_dir = watch_dir / ".processed"
    failed_dir = watch_dir / ".failed"
    processing_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    db = Database(settings.db_path)
    service = SopilotService(settings=settings, db=db, runtime_mode="watcher")
    poll_sec = max(1, int(settings.watch_poll_sec))
    default_task = settings.watch_task_id.strip()
    default_role = settings.watch_role if settings.watch_role in ROLES else "trainee"

    logger.info(
        "folder watcher started watch_dir=%s poll_sec=%s default_task_id=%s default_role=%s",
        watch_dir,
        poll_sec,
        default_task or "<required in root>",
        default_role,
    )

    try:
        while True:
            candidates = _scan_candidates(watch_dir)
            if not candidates:
                time.sleep(poll_sec)
                continue

            for source in candidates:
                src_rel = str(source.relative_to(watch_dir))
                tag = f"{_now_tag()}_{uuid.uuid4().hex[:8]}_{source.name}"
                staged = processing_dir / tag

                try:
                    shutil.move(str(source), str(staged))
                    task_id, role = _derive_task_role(
                        root=watch_dir,
                        source_path=source,
                        default_task_id=default_task,
                        default_role=default_role,
                    )
                    if not task_id:
                        raise ValueError(
                            "task_id could not be derived from path; set SOPILOT_WATCH_TASK_ID "
                            "or use watch_dir/<task_id>/[role]/video.mp4 layout"
                        )

                    result = service.enqueue_ingest_from_path(
                        file_name=source.name,
                        staged_path=staged,
                        task_id=task_id,
                        role=role,
                        requested_by="watcher:folder",
                        site_id=None,
                        camera_id=None,
                        operator_id_hash=None,
                    )
                    receipt = {
                        "status": "queued",
                        "source": src_rel,
                        "task_id": task_id,
                        "role": role,
                        "ingest_job_id": result["ingest_job_id"],
                        "queued_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                    }
                    _write_json(processed_dir / f"{tag}.json", receipt)
                    logger.info(
                        "queued source=%s job=%s task=%s role=%s", src_rel, result["ingest_job_id"], task_id, role
                    )
                except Exception as exc:
                    err_target = failed_dir / tag
                    if staged.exists():
                        shutil.move(str(staged), str(err_target))
                    _write_json(
                        failed_dir / f"{tag}.error.json",
                        {
                            "status": "failed",
                            "source": src_rel,
                            "error": str(exc),
                            "failed_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                        },
                    )
                    logger.exception("failed to queue source=%s", src_rel)

            time.sleep(poll_sec)
    finally:
        service.shutdown()


if __name__ == "__main__":
    run()
