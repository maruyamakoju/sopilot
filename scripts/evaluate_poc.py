from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from sopilot.database import Database
from sopilot.eval.gates import (
    GateConfig,
    available_gate_profiles,
    evaluate_gates,
    get_gate_profile,
    is_gate_profile_locked,
    merge_gate_config,
)
from sopilot.eval.harness import (
    available_critical_scoring_modes,
    compute_critical_threshold_sweep,
    compute_poc_metrics,
    load_critical_labels,
    policy_critical_threshold,
    policy_scoring_mode,
    recommend_threshold_from_sweep,
)

_GATE_OVERRIDE_KEYS = (
    "max_critical_miss_rate",
    "max_critical_fpr",
    "max_critical_miss_ci95_high",
    "max_critical_fpr_ci95_high",
    "max_rescore_jitter",
    "max_dtw_p90",
    "max_drift_critical_score_psi",
    "max_drift_score_psi",
    "max_drift_dtw_psi",
    "max_critical_detected_rate_shift_abs",
    "min_completed_jobs",
    "min_labels_total_jobs",
    "min_labeled_jobs",
    "min_critical_positives",
    "min_critical_negatives",
    "min_coverage_rate",
    "min_rescore_pairs",
)


def _clamp_threshold(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return round(float(value), 6)


def _parse_sweep_thresholds(args: argparse.Namespace) -> list[float]:
    values: list[float] = []
    if args.critical_sweep_auto:
        values.extend([round(i * 0.05, 6) for i in range(0, 21)])

    if args.critical_sweep_values:
        for chunk in str(args.critical_sweep_values).split(","):
            token = chunk.strip()
            if not token:
                continue
            values.append(_clamp_threshold(float(token)))

    has_range = args.critical_sweep_start is not None or args.critical_sweep_stop is not None
    if has_range:
        if args.critical_sweep_start is None or args.critical_sweep_stop is None:
            raise SystemExit("both --critical-sweep-start and --critical-sweep-stop are required")
        step = float(args.critical_sweep_step)
        if step <= 0:
            raise SystemExit("--critical-sweep-step must be > 0")
        start = float(args.critical_sweep_start)
        stop = float(args.critical_sweep_stop)
        direction = 1.0 if stop >= start else -1.0
        step = abs(step) * direction
        cursor = start
        for _ in range(0, 10001):
            if direction > 0 and cursor > (stop + 1e-12):
                break
            if direction < 0 and cursor < (stop - 1e-12):
                break
            values.append(_clamp_threshold(cursor))
            cursor += step

    return sorted({float(v) for v in values})


def _gate_overrides(args: argparse.Namespace) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for key in _GATE_OVERRIDE_KEYS:
        value = getattr(args, key, None)
        if value is None:
            continue
        out[key] = value
    return out


def _build_gate_config(args: argparse.Namespace) -> GateConfig:
    base = get_gate_profile(args.gate_profile) if args.gate_profile else GateConfig()
    overrides = _gate_overrides(args)
    if (
        args.gate_profile
        and is_gate_profile_locked(args.gate_profile)
        and overrides
        and not args.allow_profile_overrides
    ):
        names = ", ".join(sorted(overrides.keys()))
        raise SystemExit(
            f"gate profile '{args.gate_profile}' is locked and cannot be overridden "
            f"without --allow-profile-overrides (attempted: {names})"
        )
    return merge_gate_config(
        base,
        max_critical_miss_rate=args.max_critical_miss_rate,
        max_critical_false_positive_rate=args.max_critical_fpr,
        max_critical_miss_rate_ci95_high=args.max_critical_miss_ci95_high,
        max_critical_false_positive_rate_ci95_high=args.max_critical_fpr_ci95_high,
        max_rescore_jitter=args.max_rescore_jitter,
        max_dtw_p90=args.max_dtw_p90,
        max_drift_critical_score_psi=args.max_drift_critical_score_psi,
        max_drift_score_psi=args.max_drift_score_psi,
        max_drift_dtw_psi=args.max_drift_dtw_psi,
        max_critical_detected_rate_shift_abs=args.max_critical_detected_rate_shift_abs,
        min_num_completed_jobs=args.min_completed_jobs,
        min_labels_total_jobs=args.min_labels_total_jobs,
        min_labeled_jobs=args.min_labeled_jobs,
        min_critical_positives=args.min_critical_positives,
        min_critical_negatives=args.min_critical_negatives,
        min_coverage_rate=args.min_coverage_rate,
        min_rescore_pairs=args.min_rescore_pairs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SOPilot PoC metrics from completed score jobs.")
    parser.add_argument("--db-path", required=True, help="Path to sopilot.db")
    parser.add_argument("--task-id", default=None, help="Optional task_id filter")
    parser.add_argument("--labels", default=None, help="Optional JSON labels file for critical miss/fp metrics")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument(
        "--critical-policy",
        default=None,
        help="Optional critical policy JSON (fit artifact).",
    )
    parser.add_argument(
        "--critical-scoring-mode",
        choices=available_critical_scoring_modes(),
        default=None,
        help="Critical detection mode used for metric computation.",
    )
    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=None,
        help="Threshold used when critical-scoring-mode=continuous_v1.",
    )
    parser.add_argument(
        "--critical-sweep-auto",
        action="store_true",
        help="Enable threshold sweep with default grid [0.00, 0.05, ..., 1.00].",
    )
    parser.add_argument(
        "--critical-sweep-values",
        default=None,
        help="Comma-separated threshold list for sweep (e.g. 0.2,0.3,0.4).",
    )
    parser.add_argument("--critical-sweep-start", type=float, default=None)
    parser.add_argument("--critical-sweep-stop", type=float, default=None)
    parser.add_argument("--critical-sweep-step", type=float, default=0.05)
    parser.add_argument(
        "--critical-sweep-scoring-mode",
        choices=available_critical_scoring_modes(),
        default="continuous_v1",
        help="Scoring mode used for threshold sweep table.",
    )
    parser.add_argument(
        "--gate-profile",
        choices=available_gate_profiles(),
        default=None,
        help="Optional named gate profile. Explicit threshold flags override profile values.",
    )
    parser.add_argument(
        "--allow-profile-overrides",
        action="store_true",
        help="Allow overriding thresholds when gate-profile is contract-locked.",
    )
    parser.add_argument("--max-critical-miss-rate", type=float, default=None)
    parser.add_argument("--max-critical-fpr", type=float, default=None)
    parser.add_argument("--max-critical-miss-ci95-high", type=float, default=None)
    parser.add_argument("--max-critical-fpr-ci95-high", type=float, default=None)
    parser.add_argument("--max-rescore-jitter", type=float, default=None)
    parser.add_argument("--max-dtw-p90", type=float, default=None)
    parser.add_argument("--max-drift-critical-score-psi", type=float, default=None)
    parser.add_argument("--max-drift-score-psi", type=float, default=None)
    parser.add_argument("--max-drift-dtw-psi", type=float, default=None)
    parser.add_argument("--max-critical-detected-rate-shift-abs", type=float, default=None)
    parser.add_argument("--min-completed-jobs", type=int, default=None)
    parser.add_argument("--min-labels-total-jobs", type=int, default=None)
    parser.add_argument("--min-labeled-jobs", type=int, default=None)
    parser.add_argument("--min-critical-positives", type=int, default=None)
    parser.add_argument("--min-critical-negatives", type=int, default=None)
    parser.add_argument("--min-coverage-rate", type=float, default=None)
    parser.add_argument("--min-rescore-pairs", type=int, default=None)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    db = Database(Path(args.db_path))
    completed = db.list_completed_score_jobs(task_id=args.task_id)
    critical_policy = None
    if args.critical_policy:
        critical_policy = json.loads(Path(args.critical_policy).read_text(encoding="utf-8"))
    scoring_mode = args.critical_scoring_mode or policy_scoring_mode(critical_policy) or "legacy_binary"
    critical_threshold = args.critical_threshold
    if critical_threshold is None:
        critical_threshold = policy_critical_threshold(critical_policy)
    if critical_threshold is None:
        critical_threshold = 0.5

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
    report = compute_poc_metrics(
        completed,
        labels,
        label_scope=label_scope,
        critical_scoring_mode=scoring_mode,
        critical_threshold=critical_threshold,
        critical_policy=critical_policy,
    )
    report["task_id"] = args.task_id
    if args.gate_profile:
        report["gate_profile"] = args.gate_profile
        report["gate_profile_locked"] = is_gate_profile_locked(args.gate_profile)
    report["gate_overrides"] = _gate_overrides(args)
    report["critical_policy_path"] = str(Path(args.critical_policy).resolve()) if args.critical_policy else None
    report["critical_sweep_requested"] = bool(
        args.critical_sweep_auto
        or args.critical_sweep_values
        or args.critical_sweep_start is not None
        or args.critical_sweep_stop is not None
    )

    gate_config = _build_gate_config(args)
    report["gate_config"] = {
        key: value for key, value in asdict(gate_config).items() if value is not None
    }
    gates = evaluate_gates(
        report,
        gate_config,
    )
    report["gates"] = gates

    sweep_thresholds = _parse_sweep_thresholds(args)
    if sweep_thresholds:
        if labels is None:
            report["critical_threshold_sweep"] = {
                "error": "labels_required",
                "scoring_mode": args.critical_sweep_scoring_mode,
                "thresholds": sweep_thresholds,
                "rows": [],
                "recommended": None,
            }
        else:
            rows = compute_critical_threshold_sweep(
                completed,
                labels,
                sweep_thresholds,
                scoring_mode=args.critical_sweep_scoring_mode,
                critical_policy=critical_policy,
            )
            recommended = recommend_threshold_from_sweep(
                rows,
                max_miss_rate=gate_config.max_critical_miss_rate,
                max_false_positive_rate=gate_config.max_critical_false_positive_rate,
            )
            report["critical_threshold_sweep"] = {
                "scoring_mode": args.critical_sweep_scoring_mode,
                "thresholds": sweep_thresholds,
                "constraints": {
                    "max_critical_miss_rate": gate_config.max_critical_miss_rate,
                    "max_critical_false_positive_rate": gate_config.max_critical_false_positive_rate,
                },
                "rows": rows,
                "recommended": recommended,
            }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.fail_on_gate and not gates["overall_pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
