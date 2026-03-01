from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sopilot.config import Settings
from sopilot.database import Database
from sopilot.eval.harness import job_detected_critical
from sopilot.services.embedder import build_embedder
from sopilot.services.sopilot_service import SOPilotService
from sopilot.services.storage import FileStorage
from sopilot.services.video_processor import VideoProcessor

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def classify(path: Path, mode: str) -> str:
    if mode != "auto":
        return mode
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    if "gold" in parts or name.startswith(("gold_", "g_")):
        return "gold"
    if "trainee" in parts or "train" in parts or name.startswith(("trainee_", "t_")):
        return "trainee"
    return "trainee"


def iter_videos(base_dir: Path, recursive: bool) -> list[Path]:
    if not base_dir.exists():
        raise SystemExit(f"base-dir not found: {base_dir}")
    globber = base_dir.rglob("*") if recursive else base_dir.glob("*")
    return sorted([path for path in globber if path.is_file() and path.suffix.lower() in VIDEO_EXTS])


def configure_env(
    *,
    data_dir: str | Path,
    task_id: str,
    task_name: str,
    embedder_backend: str,
    allow_embedder_fallback: bool = True,
    run_id: str | None = None,
    run_seed: int | None = None,
    vjepa2_pooling: str | None = None,
) -> None:
    data_root = Path(data_dir).resolve()
    os.environ["SOPILOT_DATA_DIR"] = str(Path(data_dir).resolve())
    os.environ["SOPILOT_PRIMARY_TASK_ID"] = task_id
    os.environ["SOPILOT_PRIMARY_TASK_NAME"] = task_name
    os.environ["SOPILOT_ENFORCE_PRIMARY_TASK"] = "true"
    os.environ["SOPILOT_EMBEDDER_BACKEND"] = embedder_backend
    os.environ["SOPILOT_ALLOW_EMBEDDER_FALLBACK"] = "true" if allow_embedder_fallback else "false"
    if vjepa2_pooling:
        os.environ["SOPILOT_VJEPA2_POOLING"] = str(vjepa2_pooling).strip().lower()
    else:
        os.environ.pop("SOPILOT_VJEPA2_POOLING", None)
    os.environ["SOPILOT_SCORE_WORKERS"] = "1"
    os.environ["SOPILOT_RUN_ID"] = (run_id or task_id).strip() or task_id
    if run_seed is None:
        os.environ.pop("SOPILOT_RUN_SEED", None)
    else:
        os.environ["SOPILOT_RUN_SEED"] = str(int(run_seed))
    os.environ["SOPILOT_ENABLE_FAILURE_CAPSULE"] = "true"
    os.environ["SOPILOT_EMBEDDER_FAILURE_CAPSULE_PATH"] = str(data_root / "embedder_failure_capsules.jsonl")
    os.environ["SOPILOT_EMBEDDER_IMMEDIATE_RETRY"] = "true"


def build_service(settings: Settings) -> SOPilotService:
    database = Database(settings.database_path)
    embedder = build_embedder(settings)
    processor = VideoProcessor(
        sample_fps=settings.sample_fps,
        clip_seconds=settings.clip_seconds,
        frame_size=settings.frame_size,
        embedder=embedder,
    )
    storage = FileStorage(settings.raw_video_dir)
    return SOPilotService(
        settings=settings,
        database=database,
        storage=storage,
        video_processor=processor,
    )


def collect_embedder_runtime(service: SOPilotService) -> dict[str, Any]:
    embedder = service.video_processor.embedder
    payload: dict[str, Any] = {
        "embedder_name": getattr(embedder, "name", type(embedder).__name__),
        "embedder_type": type(embedder).__name__,
        "vjepa2_pooling": os.getenv("SOPILOT_VJEPA2_POOLING", "mean_tokens"),
    }
    stats_fn = getattr(embedder, "get_stats", None)
    if callable(stats_fn):
        try:
            stats = stats_fn()
            if isinstance(stats, dict):
                payload.update(stats)
        except Exception as exc:  # pragma: no cover - defensive guard
            payload["stats_error"] = str(exc)
    fallback_uses = payload.get("fallback_uses")
    if fallback_uses is not None:
        try:
            payload["fallback_contaminated"] = int(fallback_uses) > 0
        except Exception:
            payload["fallback_contaminated"] = False
    else:
        payload["fallback_contaminated"] = False
    return payload


def build_labels_template(
    task_id: str,
    completed_jobs: list[dict[str, Any]],
    existing_expected: dict[int, bool | None] | None = None,
) -> dict[str, Any]:
    expected_map = existing_expected or {}
    rows: list[dict[str, Any]] = []
    for job in completed_jobs:
        if job.get("score") is None:
            continue
        job_id = int(job["id"])
        expected = expected_map.get(job_id, None)
        if expected is not None:
            expected = bool(expected)
        rows.append(
            {
                "job_id": job_id,
                "critical_expected": expected,
                "predicted_critical": bool(job_detected_critical(job["score"])),
            }
        )
    return {"task_id": task_id, "jobs": rows}
