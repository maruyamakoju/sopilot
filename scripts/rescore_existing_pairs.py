from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sopilot.config import Settings
from sopilot.database import Database
from sopilot.services.embedder import build_embedder
from sopilot.services.sopilot_service import SOPilotService
from sopilot.services.storage import FileStorage
from sopilot.services.video_processor import VideoProcessor


def _build_service(data_dir: Path, task_id: str, task_name: str, backend: str) -> SOPilotService:
    os.environ["SOPILOT_DATA_DIR"] = str(data_dir.resolve())
    os.environ["SOPILOT_PRIMARY_TASK_ID"] = task_id
    os.environ["SOPILOT_PRIMARY_TASK_NAME"] = task_name
    os.environ["SOPILOT_ENFORCE_PRIMARY_TASK"] = "true"
    os.environ["SOPILOT_EMBEDDER_BACKEND"] = backend
    os.environ["SOPILOT_ALLOW_EMBEDDER_FALLBACK"] = "true"
    settings = Settings.from_env()
    db = Database(settings.database_path)
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
        database=db,
        storage=storage,
        video_processor=processor,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore existing trainee videos against one gold id.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-name", default="PoC Task")
    parser.add_argument("--gold-id", type=int, required=True)
    parser.add_argument("--site-id", default=None)
    parser.add_argument("--backend", choices=["vjepa2", "color-motion"], default="color-motion")
    parser.add_argument("--repeat", type=int, default=1, help="How many times to rescore all trainees")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    service = _build_service(Path(args.data_dir), args.task_id, args.task_name, args.backend)
    trainees = service.list_videos(site_id=args.site_id, is_gold=False, limit=100000)
    trainee_ids = [int(item["video_id"]) for item in trainees if item["status"] == "ready"]
    results: list[dict] = []

    for cycle in range(max(1, args.repeat)):
        print(f"[cycle {cycle + 1}/{max(1, args.repeat)}] trainees={len(trainee_ids)}")
        for trainee_id in trainee_ids:
            try:
                job = service.queue_score_job(gold_video_id=args.gold_id, trainee_video_id=trainee_id)
                job_id = int(job["job_id"])
                service.run_score_job(job_id)
                finished = service.get_score_job(job_id)
                score = None
                if finished.get("result"):
                    score = finished["result"].get("score")
                print(f"  job_id={job_id} trainee_id={trainee_id} status={finished['status']} score={score}")
                results.append(
                    {
                        "job_id": job_id,
                        "trainee_id": trainee_id,
                        "status": finished["status"],
                        "score": score,
                    }
                )
            except Exception as exc:
                print(f"  trainee_id={trainee_id} failed={exc}")
                results.append(
                    {
                        "job_id": None,
                        "trainee_id": trainee_id,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
