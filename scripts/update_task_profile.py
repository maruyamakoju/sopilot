from __future__ import annotations

import argparse
import json
from pathlib import Path

from sopilot.database import Database


def main() -> None:
    parser = argparse.ArgumentParser(description="Update SOPilot task profile in local DB.")
    parser.add_argument("--db-path", default="data/sopilot.db")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--pass-score", type=float, default=None)
    parser.add_argument("--retrain-score", type=float, default=None)
    parser.add_argument("--w-miss", type=float, default=None)
    parser.add_argument("--w-swap", type=float, default=None)
    parser.add_argument("--w-dev", type=float, default=None)
    parser.add_argument("--w-time", type=float, default=None)
    parser.add_argument("--policy-json", default=None, help="JSON string for deviation policy")
    parser.add_argument("--policy-file", default=None, help="JSON file for deviation policy")
    parser.add_argument("--create-if-missing", action="store_true")
    args = parser.parse_args()

    db = Database(Path(args.db_path))
    current = db.get_task_profile(args.task_id)
    if current is None and not args.create_if_missing:
        raise SystemExit(f"Task profile not found: {args.task_id} (use --create-if-missing)")

    if current is None:
        current = {
            "task_name": args.task_id,
            "pass_score": 90.0,
            "retrain_score": 80.0,
            "default_weights": {"w_miss": 0.45, "w_swap": 0.20, "w_dev": 0.25, "w_time": 0.10},
            "deviation_policy": {
                "missing_step": "critical",
                "step_deviation": "quality",
                "order_swap": "quality",
                "over_time": "efficiency",
            },
        }

    task_name = args.task_name if args.task_name is not None else current["task_name"]
    pass_score = float(args.pass_score) if args.pass_score is not None else float(current["pass_score"])
    retrain_score = (
        float(args.retrain_score) if args.retrain_score is not None else float(current["retrain_score"])
    )
    if retrain_score > pass_score:
        raise SystemExit("retrain_score must be <= pass_score")

    weights = dict(current["default_weights"])
    if args.w_miss is not None:
        weights["w_miss"] = float(args.w_miss)
    if args.w_swap is not None:
        weights["w_swap"] = float(args.w_swap)
    if args.w_dev is not None:
        weights["w_dev"] = float(args.w_dev)
    if args.w_time is not None:
        weights["w_time"] = float(args.w_time)

    policy = dict(current["deviation_policy"])
    if args.policy_json:
        policy = json.loads(args.policy_json)
    if args.policy_file:
        policy = json.loads(Path(args.policy_file).read_text(encoding="utf-8"))

    db.upsert_task_profile(
        task_id=args.task_id,
        task_name=task_name,
        pass_score=pass_score,
        retrain_score=retrain_score,
        default_weights=weights,
        deviation_policy=policy,
    )
    updated = db.get_task_profile(args.task_id)
    print(json.dumps(updated, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
