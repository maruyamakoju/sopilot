from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sopilot.database import Database


def main() -> None:
    parser = argparse.ArgumentParser(description="Show SOPilot PoC operational status.")
    parser.add_argument("--db-path", default="data/sopilot.db")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--target-gold", type=int, default=20)
    parser.add_argument("--target-trainee", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    db = Database(Path(args.db_path))
    profile = db.get_task_profile(args.task_id)
    videos = db.list_videos(task_id=args.task_id, limit=100000)
    jobs = db.list_completed_score_jobs(task_id=args.task_id)

    gold = [v for v in videos if bool(v["is_gold"])]
    trainee = [v for v in videos if not bool(v["is_gold"])]
    ready = [v for v in videos if v["status"] == "ready"]

    decisions = {"pass": 0, "retrain": 0, "fail": 0, "needs_review": 0, "unknown": 0}
    for job in jobs:
        summary = job["score"].get("summary", {})
        decision = str(summary.get("decision", "unknown"))
        if decision not in decisions:
            decision = "unknown"
        decisions[decision] += 1

    out = {
        "task_id": args.task_id,
        "profile_exists": profile is not None,
        "profile": profile,
        "videos": {
            "total": len(videos),
            "gold": len(gold),
            "trainee": len(trainee),
            "ready": len(ready),
        },
        "targets": {"gold": args.target_gold, "trainee": args.target_trainee},
        "progress": {
            "gold_pct": round(100.0 * len(gold) / max(1, args.target_gold), 1),
            "trainee_pct": round(100.0 * len(trainee) / max(1, args.target_trainee), 1),
        },
        "completed_jobs": len(jobs),
        "decision_counts": decisions,
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"task_id: {out['task_id']}")
        print(f"profile_exists: {out['profile_exists']}")
        print(f"videos total={out['videos']['total']} ready={out['videos']['ready']}")
        print(
            f"gold={out['videos']['gold']}/{args.target_gold} ({out['progress']['gold_pct']}%), "
            f"trainee={out['videos']['trainee']}/{args.target_trainee} ({out['progress']['trainee_pct']}%)"
        )
        print(f"completed_jobs: {out['completed_jobs']}")
        print(f"decision_counts: {out['decision_counts']}")

    if args.strict:
        ok = True
        if profile is None:
            ok = False
        if len(gold) < args.target_gold:
            ok = False
        if len(trainee) < args.target_trainee:
            ok = False
        if not ok:
            sys.exit(2)


if __name__ == "__main__":
    main()

