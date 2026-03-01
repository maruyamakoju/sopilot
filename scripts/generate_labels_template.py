from __future__ import annotations

import argparse
import json
from pathlib import Path

from sopilot.database import Database
from sopilot.eval.harness import job_detected_critical


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate critical label template from completed score jobs.")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    db = Database(Path(args.db_path))
    jobs = db.list_completed_score_jobs(task_id=args.task_id)
    jobs = sorted(jobs, key=lambda x: int(x["id"]), reverse=True)[: max(1, args.limit)]

    payload = {
        "task_id": args.task_id,
        "jobs": [
            {
                "job_id": int(job["id"]),
                "critical_expected": False,
                "predicted_critical": bool(job_detected_critical(job["score"])),
            }
            for job in jobs
        ],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()

