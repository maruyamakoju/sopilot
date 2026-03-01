from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sopilot.database import Database
from sopilot.eval.gates import available_gate_profiles, evaluate_gates, get_gate_profile
from sopilot.eval.harness import compute_poc_metrics, load_critical_labels
from sopilot.eval.integrity import attach_payload_hash

_AXES = ("site", "gold", "trainee")


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
    gate_config: Any,
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


def _group_key_for_axis(job: dict[str, Any], axis: str) -> str:
    normalized = str(axis).strip().lower()
    if normalized == "site":
        return str(job.get("trainee_site_id") or "unknown")
    if normalized == "gold":
        return f"gold:{int(job['gold_video_id'])}"
    if normalized == "trainee":
        return f"trainee:{int(job['trainee_video_id'])}"
    raise ValueError(f"unsupported holdout axis: {axis}")


def _parse_axis_order(raw: str, *, fallback: str = "site,gold,trainee") -> list[str]:
    tokens = [token.strip().lower() for token in str(raw or fallback).split(",") if token.strip()]
    out: list[str] = []
    for token in tokens:
        if token not in _AXES:
            raise SystemExit(f"unsupported axis in --fallback-axis-order: {token}")
        if token not in out:
            out.append(token)
    if not out:
        return list(_AXES)
    return out


def _count_pos_neg(ids: list[int], labels: dict[int, bool]) -> tuple[int, int]:
    pos = int(sum(1 for job_id in ids if bool(labels.get(int(job_id), False))))
    neg = int(len(ids) - pos)
    return pos, neg


def _axis_group_summary(
    *,
    axis: str,
    group_to_ids: dict[str, list[int]],
    labels: dict[int, bool],
    min_holdout_positives: int,
    min_holdout_negatives: int,
) -> dict[str, Any]:
    holdout_ready_groups = 0
    for ids in group_to_ids.values():
        pos, neg = _count_pos_neg(ids, labels)
        if pos >= int(min_holdout_positives) and neg >= int(min_holdout_negatives):
            holdout_ready_groups += 1
    return {
        "axis": axis,
        "groups_total": int(len(group_to_ids)),
        "groups_holdout_ready": int(holdout_ready_groups),
    }


def _select_axis(
    *,
    requested_axis: str,
    auto_fallback_axis: bool,
    axis_order: list[str],
    axis_summaries: dict[str, dict[str, Any]],
    min_groups_for_generalization: int,
) -> tuple[str, dict[str, Any]]:
    if not auto_fallback_axis:
        return requested_axis, {"fallback_used": False, "requested_axis": requested_axis, "axis_order": axis_order}

    for axis in axis_order:
        summary = axis_summaries.get(axis) or {}
        groups_total = int(summary.get("groups_total") or 0)
        groups_ready = int(summary.get("groups_holdout_ready") or 0)
        if groups_total >= int(min_groups_for_generalization) and groups_ready >= int(min_groups_for_generalization):
            return axis, {
                "fallback_used": axis != requested_axis,
                "requested_axis": requested_axis,
                "axis_order": axis_order,
            }

    return requested_axis, {
        "fallback_used": False,
        "requested_axis": requested_axis,
        "axis_order": axis_order,
        "reason": "no_axis_met_min_groups",
    }


def _fit_policy_on_dev(
    *,
    dev_jobs: list[dict[str, Any]],
    dev_labels: dict[int, bool],
    gate_profile: str,
    critical_threshold: float,
    dtw_grid: list[float],
    mean_grid: list[float],
    span_grid: list[float],
    use_ci_constraints: bool,
    meta: dict[str, Any],
) -> dict[str, Any]:
    gate_config = get_gate_profile(gate_profile)
    candidates: list[dict[str, Any]] = []
    for dtw_min in dtw_grid:
        for mean_max in mean_grid:
            for span_max in span_grid:
                policy = {
                    "version": "critical_policy_v1",
                    "scoring_mode": "guarded_binary_v2",
                    "critical_threshold": round(float(critical_threshold), 6),
                    "guardrails": {
                        "guarded_binary_v1": {"min_dtw": round(float(dtw_min), 6)},
                        "guarded_binary_v2": {
                            "min_dtw": round(float(dtw_min), 6),
                            "max_critical_missing_mean_distance": round(float(mean_max), 6),
                            "max_critical_missing_expected_span": round(float(span_max), 6),
                        },
                    },
                }
                report = compute_poc_metrics(
                    dev_jobs,
                    dev_labels,
                    label_scope={
                        "labels_total_jobs": len(dev_labels),
                        "labels_labeled_jobs": len(dev_labels),
                        "labels_unknown_jobs": 0,
                    },
                    critical_scoring_mode="guarded_binary_v2",
                    critical_threshold=float(critical_threshold),
                    critical_policy=policy,
                )
                feasible, violation = _fit_constraints(
                    report=report,
                    gate_config=gate_config,
                    use_ci_constraints=bool(use_ci_constraints),
                )
                candidates.append(
                    {
                        "policy": policy,
                        "metrics": {
                            "critical_miss_rate": report.get("critical_miss_rate"),
                            "critical_false_positive_rate": report.get("critical_false_positive_rate"),
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
        raise SystemExit("no LOSO candidates evaluated")

    def _rank(item: dict[str, Any]) -> tuple[float, float, float, float]:
        fit = item.get("fit", {})
        metrics = item.get("metrics", {})
        feasible = 0.0 if bool(fit.get("feasible")) else 1.0
        violation = float(fit.get("constraint_violation") or 0.0)
        fpr = float(metrics.get("critical_false_positive_rate") or 0.0)
        miss = float(metrics.get("critical_miss_rate") or 0.0)
        return (feasible, violation, fpr, miss)

    best = sorted(candidates, key=_rank)[0]
    policy = dict(best["policy"])
    policy["fit"] = {
        "scope": "leave_one_group_out",
        "meta": meta,
        "gate_profile": gate_profile,
        "gate_config": {key: value for key, value in asdict(gate_config).items() if value is not None},
        "use_ci_constraints": bool(use_ci_constraints),
        "best_rank": {
            "feasible": bool(best.get("fit", {}).get("feasible")),
            "constraint_violation": float(best.get("fit", {}).get("constraint_violation") or 0.0),
        },
    }
    policy_hashed = attach_payload_hash(
        policy,
        exclude_extra_keys={"policy_id"},
        method="sha256(canonical_json,exclude=artifact_hash_sha256|artifact_hash_method|policy_id)",
    )
    policy_hashed["policy_id"] = f"loso-policy-{policy_hashed['artifact_hash_sha256'][:12]}"
    return {
        "policy": policy_hashed,
        "best": best,
        "num_candidates": int(len(candidates)),
    }


def _row_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# LOSO Table",
        "",
        "|axis|group|status|dev_jobs|holdout_jobs|pos|neg|miss|fpr|miss_ci95_hi|fpr_ci95_hi|violation|pass|policy_id|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        axis = str(row.get("axis") or "site")
        group = row.get("holdout_group")
        if group is None:
            group = row.get("site")
        lines.append(
            "|"
            + "|".join(
                [
                    axis,
                    str(group),
                    str(row.get("status")),
                    str(row.get("dev_jobs", "-")),
                    str(row.get("holdout_jobs", "-")),
                    str(row.get("holdout_positives", "-")),
                    str(row.get("holdout_negatives", "-")),
                    str(row.get("miss_rate", "-")),
                    str(row.get("fpr", "-")),
                    str(row.get("miss_ci95_high", "-")),
                    str(row.get("fpr_ci95_high", "-")),
                    str(row.get("constraint_violation", "-")),
                    str(row.get("overall_pass", "-")),
                    str(row.get("policy_id", "-")),
                ]
            )
            + "|"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leave-one-site-out sweep with dev-fit policies.")
    parser.add_argument("--db-path", "--db", dest="db_path", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--gate-profile", "--profile", dest="gate_profile", choices=available_gate_profiles(), default="research_v2")
    parser.add_argument("--scoring-mode", default="guarded_binary_v2")
    parser.add_argument("--critical-threshold", type=float, default=0.5)
    parser.add_argument("--dtw-min-grid", default="0.015,0.02,0.025,0.03,0.035,0.04")
    parser.add_argument("--mean-distance-max-grid", default="0.08,0.09,0.1,0.11,0.12,0.13,0.14")
    parser.add_argument("--expected-span-max-grid", default="1.0,1.2,1.4,1.5,1.6,1.8,2.0")
    parser.add_argument("--use-ci-constraints", action="store_true")
    parser.add_argument("--holdout-axis", choices=list(_AXES), default="site")
    parser.add_argument("--auto-fallback-axis", action="store_true")
    parser.add_argument("--fallback-axis-order", default="site,gold,trainee")
    parser.add_argument("--min-groups-for-generalization", type=int, default=2)
    parser.add_argument("--min-holdout-positives", type=int, default=1)
    parser.add_argument("--min-holdout-negatives", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if str(args.scoring_mode).strip().lower() != "guarded_binary_v2":
        raise SystemExit("evaluate_loso_sweep currently supports --scoring-mode=guarded_binary_v2 only")

    labels = load_critical_labels(Path(args.labels)).labels
    db = Database(Path(args.db_path))
    completed = db.list_completed_score_jobs(task_id=args.task_id)
    completed_by_id = {int(job["id"]): job for job in completed}
    labeled_ids = sorted([int(job_id) for job_id in labels if int(job_id) in completed_by_id])
    if not labeled_ids:
        raise SystemExit("no labeled completed jobs for LOSO")

    gate_config = get_gate_profile(args.gate_profile)
    dtw_grid = _parse_grid(args.dtw_min_grid)
    mean_grid = _parse_grid(args.mean_distance_max_grid)
    span_grid = _parse_grid(args.expected_span_max_grid)

    axis_order = _parse_axis_order(args.fallback_axis_order)
    requested_axis = str(args.holdout_axis)
    if requested_axis not in axis_order:
        axis_order = [requested_axis] + [axis for axis in axis_order if axis != requested_axis]

    axis_group_maps: dict[str, dict[str, list[int]]] = {}
    axis_summaries: dict[str, dict[str, Any]] = {}
    for axis in axis_order:
        group_map: dict[str, list[int]] = {}
        for job_id in labeled_ids:
            group = _group_key_for_axis(completed_by_id[job_id], axis)
            group_map.setdefault(group, []).append(int(job_id))
        axis_group_maps[axis] = {group: sorted(ids) for group, ids in group_map.items()}
        axis_summaries[axis] = _axis_group_summary(
            axis=axis,
            group_to_ids=axis_group_maps[axis],
            labels=labels,
            min_holdout_positives=int(args.min_holdout_positives),
            min_holdout_negatives=int(args.min_holdout_negatives),
        )

    axis_used, axis_selection = _select_axis(
        requested_axis=requested_axis,
        auto_fallback_axis=bool(args.auto_fallback_axis),
        axis_order=axis_order,
        axis_summaries=axis_summaries,
        min_groups_for_generalization=max(2, int(args.min_groups_for_generalization)),
    )
    group_to_ids = axis_group_maps.get(axis_used, {})

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    per_holdout_reports: dict[str, Any] = {}
    for holdout_group in sorted(group_to_ids.keys()):
        holdout_ids = sorted(group_to_ids.get(holdout_group, []))
        holdout_labels = {job_id: labels[job_id] for job_id in holdout_ids if job_id in labels}
        holdout_pos = int(sum(1 for value in holdout_labels.values() if value))
        holdout_neg = int(sum(1 for value in holdout_labels.values() if not value))
        if holdout_pos < int(args.min_holdout_positives) or holdout_neg < int(args.min_holdout_negatives):
            rows.append(
                {
                    "axis": axis_used,
                    "holdout_group": holdout_group,
                    "site": holdout_group,
                    "status": "skipped_insufficient_holdout_evidence",
                    "holdout_jobs": int(len(holdout_ids)),
                    "holdout_positives": holdout_pos,
                    "holdout_negatives": holdout_neg,
                }
            )
            continue

        dev_ids = sorted(set(labeled_ids) - set(holdout_ids))
        dev_jobs = [completed_by_id[job_id] for job_id in dev_ids]
        dev_labels = {job_id: labels[job_id] for job_id in dev_ids if job_id in labels}
        if not dev_jobs or not dev_labels:
            rows.append(
                {
                    "axis": axis_used,
                    "holdout_group": holdout_group,
                    "site": holdout_group,
                    "status": "skipped_empty_dev",
                    "holdout_jobs": int(len(holdout_ids)),
                    "holdout_positives": holdout_pos,
                    "holdout_negatives": holdout_neg,
                }
            )
            continue

        fit = _fit_policy_on_dev(
            dev_jobs=dev_jobs,
            dev_labels=dev_labels,
            gate_profile=args.gate_profile,
            critical_threshold=float(args.critical_threshold),
            dtw_grid=dtw_grid,
            mean_grid=mean_grid,
            span_grid=span_grid,
            use_ci_constraints=bool(args.use_ci_constraints),
            meta={
                "holdout_axis": axis_used,
                "holdout_group": holdout_group,
                "task_id": args.task_id,
                "db_path": str(Path(args.db_path).resolve()),
                "labels_path": str(Path(args.labels).resolve()),
                "dev_jobs": int(len(dev_jobs)),
                "holdout_jobs": int(len(holdout_ids)),
            },
        )
        policy = fit["policy"]

        holdout_jobs = [completed_by_id[job_id] for job_id in holdout_ids]
        holdout_report = compute_poc_metrics(
            holdout_jobs,
            holdout_labels,
            label_scope={
                "labels_total_jobs": len(holdout_ids),
                "labels_labeled_jobs": len(holdout_ids),
                "labels_unknown_jobs": 0,
            },
            critical_scoring_mode="guarded_binary_v2",
            critical_threshold=float(args.critical_threshold),
            critical_policy=policy,
        )
        gates = evaluate_gates(holdout_report, gate_config)
        holdout_report["gates"] = gates
        miss_ci_high = ((holdout_report.get("critical_confidence") or {}).get("miss_rate") or {}).get("ci95", {}).get("high")
        fpr_ci_high = ((holdout_report.get("critical_confidence") or {}).get("false_positive_rate") or {}).get("ci95", {}).get("high")
        _, violation = _fit_constraints(
            report=holdout_report,
            gate_config=gate_config,
            use_ci_constraints=bool(args.use_ci_constraints),
        )

        row = {
            "axis": axis_used,
            "holdout_group": holdout_group,
            "site": holdout_group,
            "status": "ok",
            "dev_jobs": int(len(dev_jobs)),
            "holdout_jobs": int(len(holdout_ids)),
            "holdout_positives": holdout_pos,
            "holdout_negatives": holdout_neg,
            "miss_rate": holdout_report.get("critical_miss_rate"),
            "fpr": holdout_report.get("critical_false_positive_rate"),
            "miss_ci95_high": miss_ci_high,
            "fpr_ci95_high": fpr_ci_high,
            "constraint_violation": float(violation),
            "overall_pass": bool(gates.get("overall_pass")),
            "policy_id": policy.get("policy_id"),
            "policy_hash_sha256": policy.get("artifact_hash_sha256"),
        }
        rows.append(row)
        per_holdout_reports[holdout_group] = {
            "row": row,
            "fit": fit,
            "holdout_report": holdout_report,
        }

        holdout_safe = holdout_group.replace(":", "_")
        holdout_dir = out_dir / f"{axis_used}_{holdout_safe}"
        holdout_dir.mkdir(parents=True, exist_ok=True)
        (holdout_dir / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
        (holdout_dir / "report.json").write_text(
            json.dumps(per_holdout_reports[holdout_group], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    ok_rows = [row for row in rows if str(row.get("status")) == "ok"]
    worst_case: dict[str, Any] | None = None
    if ok_rows:
        ranked = sorted(
            ok_rows,
            key=lambda row: (
                1.0 if bool(row.get("overall_pass")) else 0.0,
                -float(row.get("constraint_violation") or 0.0),
                -float(row.get("fpr") or 0.0),
                -float(row.get("miss_rate") or 0.0),
            ),
        )
        worst_case = ranked[0]

    payload = {
        "task_id": args.task_id,
        "db_path": str(Path(args.db_path).resolve()),
        "labels_path": str(Path(args.labels).resolve()),
        "gate_profile": args.gate_profile,
        "gate_config": {key: value for key, value in asdict(gate_config).items() if value is not None},
        "scoring_mode": "guarded_binary_v2",
        "critical_threshold": float(args.critical_threshold),
        "use_ci_constraints": bool(args.use_ci_constraints),
        "holdout_axis_requested": requested_axis,
        "holdout_axis_used": axis_used,
        "axis_selection": axis_selection,
        "axis_summaries": axis_summaries,
        "summary": {
            "total_rows": int(len(rows)),
            "ok_rows": int(len(ok_rows)),
            "skipped_rows": int(len(rows) - len(ok_rows)),
            "overall_pass_rows": int(sum(1 for row in ok_rows if bool(row.get("overall_pass")))),
            "worst_case_row": worst_case,
        },
        "rows": rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    (out_dir / "loso_table.json").write_text(text, encoding="utf-8")
    (out_dir / "loso_table.md").write_text(_row_markdown(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
