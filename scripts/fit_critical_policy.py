from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sopilot.database import Database
from sopilot.eval.gates import GateConfig, available_gate_profiles, get_gate_profile
from sopilot.eval.harness import compute_poc_metrics, load_critical_labels
from sopilot.eval.integrity import attach_payload_hash, verify_payload_hash


def _parse_grid(raw: str) -> list[float]:
    values: list[float] = []
    for chunk in str(raw).split(","):
        token = chunk.strip()
        if not token:
            continue
        parsed = float(token)
        if parsed < 0.0:
            continue
        values.append(float(parsed))
    out = sorted({round(float(v), 6) for v in values})
    if not out:
        raise SystemExit("grid must include at least one non-negative value")
    return out


def _load_split_manifest(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "artifact_hash_sha256" in payload and not verify_payload_hash(payload):
        raise SystemExit(f"split manifest hash verification failed: {path}")
    raw = payload.get("split_job_ids")
    if not isinstance(raw, dict):
        raise SystemExit("split manifest must include object field 'split_job_ids'")
    out: dict[str, list[int]] = {}
    for key in ("dev", "test", "challenge"):
        rows = raw.get(key, [])
        if not isinstance(rows, list):
            raise SystemExit(f"split manifest field split_job_ids.{key} must be a list")
        out[key] = sorted({int(job_id) for job_id in rows})
    return out


def _build_candidate_policy(
    *,
    dtw_min: float,
    mean_distance_max: float,
    expected_span_max: float,
    critical_threshold: float,
) -> dict[str, Any]:
    return {
        "version": "critical_policy_v1",
        "scoring_mode": "guarded_binary_v2",
        "critical_threshold": round(float(critical_threshold), 6),
        "guardrails": {
            "guarded_binary_v1": {
                "min_dtw": round(float(dtw_min), 6),
            },
            "guarded_binary_v2": {
                "min_dtw": round(float(dtw_min), 6),
                "max_critical_missing_mean_distance": round(float(mean_distance_max), 6),
                "max_critical_missing_expected_span": round(float(expected_span_max), 6),
            },
        },
    }


def _metric_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _constraint_excess(value: float | None, threshold: float | None) -> float:
    if threshold is None:
        return 0.0
    if value is None:
        return 1.0
    return max(0.0, float(value) - float(threshold))


def _fit_constraints(
    *,
    report: dict[str, Any],
    gate_config: GateConfig,
    use_ci_constraints: bool,
) -> tuple[bool, float]:
    miss = _metric_float(report, "critical_miss_rate")
    fpr = _metric_float(report, "critical_false_positive_rate")
    miss_ci_hi = _metric_float(((report.get("critical_confidence") or {}).get("miss_rate") or {}).get("ci95") or {}, "high")
    fpr_ci_hi = _metric_float(
        ((report.get("critical_confidence") or {}).get("false_positive_rate") or {}).get("ci95") or {},
        "high",
    )

    excess = 0.0
    excess += _constraint_excess(miss, gate_config.max_critical_miss_rate)
    excess += _constraint_excess(fpr, gate_config.max_critical_false_positive_rate)
    if use_ci_constraints:
        excess += _constraint_excess(miss_ci_hi, gate_config.max_critical_miss_rate_ci95_high)
        excess += _constraint_excess(fpr_ci_hi, gate_config.max_critical_false_positive_rate_ci95_high)
    return excess <= 1e-12, round(float(excess), 8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit guarded_binary_v2 policy on DEV split only.")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument(
        "--gate-profile",
        choices=available_gate_profiles(),
        default="research_v2",
        help="Used as fit constraints (miss/fpr, optionally CI).",
    )
    parser.add_argument("--critical-threshold", type=float, default=0.5)
    parser.add_argument(
        "--dtw-min-grid",
        default="0.015,0.02,0.025,0.03,0.035,0.04",
        help="Comma-separated grid.",
    )
    parser.add_argument(
        "--mean-distance-max-grid",
        default="0.08,0.09,0.1,0.11,0.12,0.13,0.14",
        help="Comma-separated grid.",
    )
    parser.add_argument(
        "--expected-span-max-grid",
        default="1.0,1.2,1.4,1.5,1.6,1.8,2.0",
        help="Comma-separated grid.",
    )
    parser.add_argument(
        "--use-ci-constraints",
        action="store_true",
        help="Enforce CI upper-bound constraints in fit objective.",
    )
    parser.add_argument("--output-policy", required=True)
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--policy-id-prefix", default="critical-policy")
    parser.add_argument("--git-commit", default="unknown")
    args = parser.parse_args()

    split_ids = _load_split_manifest(Path(args.split_manifest))
    dev_ids = {int(job_id) for job_id in split_ids.get("dev", [])}
    if not dev_ids:
        raise SystemExit("split manifest dev set is empty")

    labels = load_critical_labels(Path(args.labels)).labels
    db = Database(Path(args.db_path))
    completed_jobs = db.list_completed_score_jobs(task_id=args.task_id)
    job_by_id = {int(job["id"]): job for job in completed_jobs}

    dev_jobs = [job_by_id[job_id] for job_id in sorted(dev_ids) if job_id in job_by_id]
    dev_labels = {job_id: labels[job_id] for job_id in sorted(dev_ids) if job_id in labels}
    if not dev_jobs:
        raise SystemExit("no completed jobs found for dev split")
    if not dev_labels:
        raise SystemExit("no labels found for dev split")

    gate_config = get_gate_profile(args.gate_profile)
    dtw_grid = _parse_grid(args.dtw_min_grid)
    mean_grid = _parse_grid(args.mean_distance_max_grid)
    span_grid = _parse_grid(args.expected_span_max_grid)

    candidates: list[dict[str, Any]] = []
    for dtw_min in dtw_grid:
        for mean_max in mean_grid:
            for span_max in span_grid:
                policy = _build_candidate_policy(
                    dtw_min=dtw_min,
                    mean_distance_max=mean_max,
                    expected_span_max=span_max,
                    critical_threshold=float(args.critical_threshold),
                )
                report = compute_poc_metrics(
                    dev_jobs,
                    dev_labels,
                    label_scope={
                        "labels_total_jobs": len(dev_ids),
                        "labels_labeled_jobs": len(dev_ids),
                        "labels_unknown_jobs": 0,
                    },
                    critical_scoring_mode="guarded_binary_v2",
                    critical_threshold=float(args.critical_threshold),
                    critical_policy=policy,
                )
                feasible, violation = _fit_constraints(
                    report=report,
                    gate_config=gate_config,
                    use_ci_constraints=bool(args.use_ci_constraints),
                )
                candidates.append(
                    {
                        "policy": policy,
                        "metrics": {
                            "critical_miss_rate": report.get("critical_miss_rate"),
                            "critical_false_positive_rate": report.get("critical_false_positive_rate"),
                            "critical_miss_rate_ci95_high": (
                                ((report.get("critical_confidence") or {}).get("miss_rate") or {}).get("ci95") or {}
                            ).get("high"),
                            "critical_false_positive_rate_ci95_high": (
                                ((report.get("critical_confidence") or {}).get("false_positive_rate") or {}).get("ci95")
                                or {}
                            ).get("high"),
                            "critical_confusion": report.get("critical_confusion"),
                            "critical_positives": report.get("critical_positives"),
                            "critical_negatives": report.get("critical_negatives"),
                        },
                        "fit": {
                            "feasible": bool(feasible),
                            "constraint_violation": float(violation),
                            "params": {
                                "min_dtw": float(dtw_min),
                                "max_critical_missing_mean_distance": float(mean_max),
                                "max_critical_missing_expected_span": float(span_max),
                            },
                        },
                    }
                )

    if not candidates:
        raise SystemExit("no candidates evaluated")

    def _candidate_rank(item: dict[str, Any]) -> tuple[float, float, float, float, float]:
        metrics = item.get("metrics", {})
        fit = item.get("fit", {})
        feasible = 0.0 if bool(fit.get("feasible")) else 1.0
        violation = float(fit.get("constraint_violation") or 0.0)
        fpr = float(metrics.get("critical_false_positive_rate") or 0.0)
        miss = float(metrics.get("critical_miss_rate") or 0.0)
        params = fit.get("params", {})
        regularization = (
            abs(float(params.get("min_dtw") or 0.0) - 0.025)
            + abs(float(params.get("max_critical_missing_mean_distance") or 0.0) - 0.11)
            + abs(float(params.get("max_critical_missing_expected_span") or 0.0) - 1.5)
        )
        return (feasible, violation, fpr, miss, regularization)

    best = sorted(candidates, key=_candidate_rank)[0]
    best_policy = dict(best["policy"])
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    best_policy["fit"] = {
        "split": "dev",
        "split_manifest_path": str(Path(args.split_manifest).resolve()),
        "db_path": str(Path(args.db_path).resolve()),
        "labels_path": str(Path(args.labels).resolve()),
        "task_id": args.task_id,
        "gate_profile": args.gate_profile,
        "gate_config": {key: value for key, value in asdict(gate_config).items() if value is not None},
        "use_ci_constraints": bool(args.use_ci_constraints),
        "search_grid": {
            "dtw_min": dtw_grid,
            "max_critical_missing_mean_distance": mean_grid,
            "max_critical_missing_expected_span": span_grid,
        },
        "dev_jobs": int(len(dev_jobs)),
        "dev_labeled_jobs": int(len(dev_labels)),
        "best_metrics": best.get("metrics", {}),
        "best_rank": {
            "feasible": bool(best.get("fit", {}).get("feasible")),
            "constraint_violation": float(best.get("fit", {}).get("constraint_violation") or 0.0),
        },
    }
    best_policy["created_at"] = str(created_at)
    best_policy["git_commit"] = str(args.git_commit or "unknown")
    best_policy_hashed = attach_payload_hash(
        best_policy,
        exclude_extra_keys={"policy_id"},
        method="sha256(canonical_json,exclude=artifact_hash_sha256|artifact_hash_method|policy_id)",
    )
    best_policy_hashed["policy_id"] = f"{str(args.policy_id_prefix).strip() or 'critical-policy'}-{best_policy_hashed['artifact_hash_sha256'][:12]}"

    policy_path = Path(args.output_policy).resolve()
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(best_policy_hashed, ensure_ascii=False, indent=2), encoding="utf-8")

    report_payload = {
        "task_id": args.task_id,
        "db_path": str(Path(args.db_path).resolve()),
        "labels_path": str(Path(args.labels).resolve()),
        "split_manifest_path": str(Path(args.split_manifest).resolve()),
        "gate_profile": args.gate_profile,
        "use_ci_constraints": bool(args.use_ci_constraints),
        "critical_threshold": float(args.critical_threshold),
        "num_candidates": int(len(candidates)),
        "best_candidate": best,
        "policy_id": best_policy_hashed.get("policy_id"),
        "policy_hash_sha256": best_policy_hashed.get("artifact_hash_sha256"),
        "top_candidates": sorted(candidates, key=_candidate_rank)[:20],
        "output_policy": str(policy_path),
    }
    report_text = json.dumps(report_payload, ensure_ascii=False, indent=2)
    print(report_text)
    if args.output_report:
        report_path = Path(args.output_report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
