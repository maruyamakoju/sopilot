from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sopilot.database import Database
from sopilot.eval.harness import (
    available_critical_scoring_modes,
    compute_poc_metrics,
    job_detected_critical,
    load_critical_labels,
    policy_critical_threshold,
    policy_scoring_mode,
)


def _load_policy(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_mode(
    *,
    explicit_mode: str | None,
    policy: dict[str, Any] | None,
    fallback: str,
) -> str:
    if explicit_mode:
        return explicit_mode
    from_policy = policy_scoring_mode(policy)
    if from_policy:
        return from_policy
    return fallback


def _resolve_threshold(
    *,
    explicit_threshold: float | None,
    policy: dict[str, Any] | None,
    fallback: float,
) -> float:
    if explicit_threshold is not None:
        return float(explicit_threshold)
    from_policy = policy_critical_threshold(policy)
    if from_policy is not None:
        return float(from_policy)
    return float(fallback)


def _shadow_diff(
    *,
    completed_jobs: list[dict[str, Any]],
    baseline_mode: str,
    baseline_threshold: float,
    baseline_policy: dict[str, Any] | None,
    candidate_mode: str,
    candidate_threshold: float,
    candidate_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_only = 0
    candidate_only = 0
    both = 0
    neither = 0
    changed_job_ids: list[int] = []

    for job in completed_jobs:
        job_id = int(job["id"])
        score_payload = job["score"]
        base = job_detected_critical(
            score_payload,
            scoring_mode=baseline_mode,
            critical_threshold=baseline_threshold,
            critical_policy=baseline_policy,
        )
        cand = job_detected_critical(
            score_payload,
            scoring_mode=candidate_mode,
            critical_threshold=candidate_threshold,
            critical_policy=candidate_policy,
        )
        if base and cand:
            both += 1
        elif base and not cand:
            baseline_only += 1
            changed_job_ids.append(job_id)
        elif (not base) and cand:
            candidate_only += 1
            changed_job_ids.append(job_id)
        else:
            neither += 1

    total = len(completed_jobs)
    return {
        "total_jobs": int(total),
        "both_detected": int(both),
        "baseline_only": int(baseline_only),
        "candidate_only": int(candidate_only),
        "neither_detected": int(neither),
        "decision_change_rate": round(float((baseline_only + candidate_only) / max(1, total)), 6),
        "changed_job_ids": sorted(changed_job_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow evaluation: compare baseline and candidate detectors on same jobs.")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--labels", default=None, help="Optional labels to compare miss/FPR deltas.")

    parser.add_argument("--baseline-mode", choices=available_critical_scoring_modes(), default=None)
    parser.add_argument("--baseline-threshold", type=float, default=None)
    parser.add_argument("--baseline-policy", default=None)

    parser.add_argument("--candidate-mode", choices=available_critical_scoring_modes(), default=None)
    parser.add_argument("--candidate-threshold", type=float, default=None)
    parser.add_argument("--candidate-policy", default=None)

    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    baseline_policy = _load_policy(args.baseline_policy)
    candidate_policy = _load_policy(args.candidate_policy)

    baseline_mode = _resolve_mode(
        explicit_mode=args.baseline_mode,
        policy=baseline_policy,
        fallback="guarded_binary_v1",
    )
    candidate_mode = _resolve_mode(
        explicit_mode=args.candidate_mode,
        policy=candidate_policy,
        fallback="guarded_binary_v2",
    )
    baseline_threshold = _resolve_threshold(
        explicit_threshold=args.baseline_threshold,
        policy=baseline_policy,
        fallback=0.5,
    )
    candidate_threshold = _resolve_threshold(
        explicit_threshold=args.candidate_threshold,
        policy=candidate_policy,
        fallback=0.5,
    )

    db = Database(Path(args.db_path))
    completed_jobs = db.list_completed_score_jobs(task_id=args.task_id)
    labels = None
    label_scope = None
    if args.labels:
        label_set = load_critical_labels(Path(args.labels))
        labels = label_set.labels
        label_scope = {
            "labels_total_jobs": int(label_set.total_jobs),
            "labels_labeled_jobs": int(label_set.labeled_jobs),
            "labels_unknown_jobs": int(label_set.unknown_jobs),
        }

    baseline_report = compute_poc_metrics(
        completed_jobs,
        labels,
        label_scope=label_scope,
        critical_scoring_mode=baseline_mode,
        critical_threshold=baseline_threshold,
        critical_policy=baseline_policy,
    )
    candidate_report = compute_poc_metrics(
        completed_jobs,
        labels,
        label_scope=label_scope,
        critical_scoring_mode=candidate_mode,
        critical_threshold=candidate_threshold,
        critical_policy=candidate_policy,
    )

    shadow = _shadow_diff(
        completed_jobs=completed_jobs,
        baseline_mode=baseline_mode,
        baseline_threshold=baseline_threshold,
        baseline_policy=baseline_policy,
        candidate_mode=candidate_mode,
        candidate_threshold=candidate_threshold,
        candidate_policy=candidate_policy,
    )

    baseline_detect_rate = (
        float(baseline_report["critical_detected_jobs"]) / max(1, int(baseline_report["num_completed_jobs"]))
    )
    candidate_detect_rate = (
        float(candidate_report["critical_detected_jobs"]) / max(1, int(candidate_report["num_completed_jobs"]))
    )
    payload = {
        "task_id": args.task_id,
        "db_path": str(Path(args.db_path).resolve()),
        "labels_path": str(Path(args.labels).resolve()) if args.labels else None,
        "baseline": {
            "mode": baseline_mode,
            "threshold": float(baseline_threshold),
            "policy_path": str(Path(args.baseline_policy).resolve()) if args.baseline_policy else None,
            "report": baseline_report,
        },
        "candidate": {
            "mode": candidate_mode,
            "threshold": float(candidate_threshold),
            "policy_path": str(Path(args.candidate_policy).resolve()) if args.candidate_policy else None,
            "report": candidate_report,
        },
        "shadow_diff": shadow,
        "delta": {
            "critical_detected_rate_baseline": round(float(baseline_detect_rate), 6),
            "critical_detected_rate_candidate": round(float(candidate_detect_rate), 6),
            "critical_detected_rate_delta": round(float(candidate_detect_rate - baseline_detect_rate), 6),
            "critical_miss_rate_delta": (
                round(
                    float((candidate_report.get("critical_miss_rate") or 0.0) - (baseline_report.get("critical_miss_rate") or 0.0)),
                    6,
                )
                if baseline_report.get("critical_miss_rate") is not None and candidate_report.get("critical_miss_rate") is not None
                else None
            ),
            "critical_false_positive_rate_delta": (
                round(
                    float(
                        (candidate_report.get("critical_false_positive_rate") or 0.0)
                        - (baseline_report.get("critical_false_positive_rate") or 0.0)
                    ),
                    6,
                )
                if baseline_report.get("critical_false_positive_rate") is not None
                and candidate_report.get("critical_false_positive_rate") is not None
                else None
            ),
        },
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
