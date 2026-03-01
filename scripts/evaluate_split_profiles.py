from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    compute_critical_score,
    compute_poc_metrics,
    policy_critical_threshold,
    policy_scoring_mode,
)
from sopilot.eval.integrity import attach_payload_hash, verify_payload_hash

_MIN_EVIDENCE_KEYS = (
    "min_num_completed_jobs",
    "min_labels_total_jobs",
    "min_labeled_jobs",
    "min_critical_positives",
    "min_critical_negatives",
    "min_coverage_rate",
    "min_rescore_pairs",
)
_SPLIT_STRATEGIES = (
    "random_stratified",
    "group_trainee",
    "group_gold",
    "group_site",
    "leave_one_site_out",
)


def _load_labels_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_labeled_job_ids(labels_payload: dict[str, Any]) -> tuple[dict[int, bool], dict[int, dict[str, Any]]]:
    labels: dict[int, bool] = {}
    rows: dict[int, dict[str, Any]] = {}
    for item in labels_payload.get("jobs", []):
        if not isinstance(item, dict):
            continue
        if item.get("job_id") is None:
            continue
        try:
            job_id = int(item["job_id"])
        except Exception:
            continue
        expected = item.get("critical_expected")
        if expected is None:
            continue
        labels[job_id] = bool(expected)
        rows[job_id] = dict(item)
    return labels, rows


def _split_ids(
    *,
    job_ids: list[int],
    labels: dict[int, bool],
    hardness_by_id: dict[int, float],
    dev_ratio: float,
    test_ratio: float,
    challenge_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    if not job_ids:
        return {"dev": [], "test": [], "challenge": []}
    if dev_ratio <= 0 or test_ratio <= 0:
        raise ValueError("dev_ratio and test_ratio must be > 0")
    if challenge_ratio < 0 or challenge_ratio >= 1.0:
        raise ValueError("challenge_ratio must be in [0, 1)")

    positives = sorted([job_id for job_id in job_ids if labels.get(job_id, False)])
    negatives = sorted([job_id for job_id in job_ids if not labels.get(job_id, False)])

    def _challenge_take(ids: list[int]) -> int:
        if not ids:
            return 0
        return min(len(ids), max(1, int(round(len(ids) * challenge_ratio))))

    challenge_pos_n = _challenge_take(positives)
    challenge_neg_n = _challenge_take(negatives)

    positives_hard = sorted(positives, key=lambda job_id: (-float(hardness_by_id.get(job_id, 0.0)), job_id))
    negatives_hard = sorted(negatives, key=lambda job_id: (-float(hardness_by_id.get(job_id, 0.0)), job_id))
    challenge_ids = set(positives_hard[:challenge_pos_n] + negatives_hard[:challenge_neg_n])

    rem_pos = [job_id for job_id in positives if job_id not in challenge_ids]
    rem_neg = [job_id for job_id in negatives if job_id not in challenge_ids]
    rng = random.Random(int(seed))
    rng.shuffle(rem_pos)
    rng.shuffle(rem_neg)

    dev_frac = float(dev_ratio / (dev_ratio + test_ratio))
    dev_pos_n = int(round(len(rem_pos) * dev_frac))
    dev_neg_n = int(round(len(rem_neg) * dev_frac))

    dev_ids = set(rem_pos[:dev_pos_n] + rem_neg[:dev_neg_n])
    test_ids = set(rem_pos[dev_pos_n:] + rem_neg[dev_neg_n:])

    if challenge_ids & dev_ids or challenge_ids & test_ids or dev_ids & test_ids:
        raise RuntimeError("split overlap detected")

    return {
        "dev": sorted(dev_ids),
        "test": sorted(test_ids),
        "challenge": sorted(challenge_ids),
    }


def _split_ids_grouped(
    *,
    job_ids: list[int],
    labels: dict[int, bool],
    hardness_by_id: dict[int, float],
    group_by_id: dict[int, str],
    dev_ratio: float,
    test_ratio: float,
    challenge_ratio: float,
    seed: int,
    challenge_fixed_ids: set[int] | None = None,
) -> dict[str, list[int]]:
    if not job_ids:
        return {"dev": [], "test": [], "challenge": []}
    if dev_ratio <= 0 or test_ratio <= 0:
        raise ValueError("dev_ratio and test_ratio must be > 0")
    if challenge_ratio < 0 or challenge_ratio >= 1.0:
        raise ValueError("challenge_ratio must be in [0, 1)")

    groups: dict[str, list[int]] = {}
    for job_id in job_ids:
        group = group_by_id.get(job_id, f"job:{job_id}")
        groups.setdefault(group, []).append(job_id)

    group_rows: list[dict[str, Any]] = []
    for group, ids in groups.items():
        ids_sorted = sorted(ids)
        pos = int(sum(1 for job_id in ids_sorted if labels.get(job_id, False)))
        neg = int(len(ids_sorted) - pos)
        hardness = float(sum(float(hardness_by_id.get(job_id, 0.0)) for job_id in ids_sorted) / max(1, len(ids_sorted)))
        group_rows.append(
            {
                "group": group,
                "ids": ids_sorted,
                "jobs": int(len(ids_sorted)),
                "pos": pos,
                "neg": neg,
                "hardness": hardness,
            }
        )
    group_rows.sort(key=lambda row: str(row["group"]))

    challenge_ids: set[int] = set()
    if challenge_fixed_ids is not None:
        challenge_ids = {int(job_id) for job_id in challenge_fixed_ids if int(job_id) in set(job_ids)}
    else:
        total_pos = int(sum(int(row["pos"]) for row in group_rows))
        total_neg = int(sum(int(row["neg"]) for row in group_rows))
        target_challenge_jobs = max(0, int(round(len(job_ids) * challenge_ratio)))
        target_challenge_pos = 0
        target_challenge_neg = 0
        if challenge_ratio > 0.0 and total_pos > 0:
            target_challenge_pos = min(total_pos, max(1, int(round(total_pos * challenge_ratio))))
        if challenge_ratio > 0.0 and total_neg > 0:
            target_challenge_neg = min(total_neg, max(1, int(round(total_neg * challenge_ratio))))

        hard_groups = sorted(group_rows, key=lambda row: (-float(row["hardness"]), -int(row["jobs"]), str(row["group"])))
        selected_groups: set[str] = set()
        challenge_pos = 0
        challenge_neg = 0

        for row in hard_groups:
            if challenge_pos >= target_challenge_pos:
                break
            if int(row["pos"]) <= 0:
                continue
            selected_groups.add(str(row["group"]))
            challenge_ids.update(int(job_id) for job_id in row["ids"])
            challenge_pos += int(row["pos"])
            challenge_neg += int(row["neg"])

        for row in hard_groups:
            if challenge_neg >= target_challenge_neg:
                break
            if int(row["neg"]) <= 0:
                continue
            if str(row["group"]) in selected_groups:
                continue
            selected_groups.add(str(row["group"]))
            challenge_ids.update(int(job_id) for job_id in row["ids"])
            challenge_pos += int(row["pos"])
            challenge_neg += int(row["neg"])

        for row in hard_groups:
            if len(challenge_ids) >= target_challenge_jobs:
                break
            if str(row["group"]) in selected_groups:
                continue
            selected_groups.add(str(row["group"]))
            challenge_ids.update(int(job_id) for job_id in row["ids"])

        if len(challenge_ids) >= len(job_ids) and hard_groups:
            for job_id in hard_groups[-1]["ids"]:
                challenge_ids.discard(int(job_id))

    remaining_rows: list[dict[str, Any]] = []
    for row in group_rows:
        rem_ids = [int(job_id) for job_id in row["ids"] if int(job_id) not in challenge_ids]
        if not rem_ids:
            continue
        rem_pos = int(sum(1 for job_id in rem_ids if labels.get(job_id, False)))
        rem_neg = int(len(rem_ids) - rem_pos)
        remaining_rows.append(
            {
                "group": row["group"],
                "ids": rem_ids,
                "jobs": int(len(rem_ids)),
                "pos": rem_pos,
                "neg": rem_neg,
            }
        )

    rng = random.Random(int(seed))
    rng.shuffle(remaining_rows)

    total_remaining_jobs = int(sum(int(row["jobs"]) for row in remaining_rows))
    total_remaining_pos = int(sum(int(row["pos"]) for row in remaining_rows))
    total_remaining_neg = int(sum(int(row["neg"]) for row in remaining_rows))
    dev_frac = float(dev_ratio / (dev_ratio + test_ratio))
    target_dev_jobs = float(total_remaining_jobs) * dev_frac
    target_dev_pos = float(total_remaining_pos) * dev_frac
    target_dev_neg = float(total_remaining_neg) * dev_frac

    dev_ids: set[int] = set()
    dev_jobs = 0
    dev_pos = 0
    dev_neg = 0

    for row in remaining_rows:
        row_jobs = int(row["jobs"])
        row_pos = int(row["pos"])
        row_neg = int(row["neg"])

        keep_loss = _split_loss(
            dev_jobs=float(dev_jobs),
            dev_pos=float(dev_pos),
            dev_neg=float(dev_neg),
            target_dev_jobs=target_dev_jobs,
            target_dev_pos=target_dev_pos,
            target_dev_neg=target_dev_neg,
        )
        add_loss = _split_loss(
            dev_jobs=float(dev_jobs + row_jobs),
            dev_pos=float(dev_pos + row_pos),
            dev_neg=float(dev_neg + row_neg),
            target_dev_jobs=target_dev_jobs,
            target_dev_pos=target_dev_pos,
            target_dev_neg=target_dev_neg,
        )
        if add_loss <= keep_loss:
            dev_ids.update(int(job_id) for job_id in row["ids"])
            dev_jobs += row_jobs
            dev_pos += row_pos
            dev_neg += row_neg

    remaining_ids = {int(job_id) for row in remaining_rows for job_id in row["ids"]}
    test_ids = set(remaining_ids - dev_ids)

    if not dev_ids and test_ids and remaining_rows:
        move = remaining_rows[0]["ids"]
        for job_id in move:
            dev_ids.add(int(job_id))
            test_ids.discard(int(job_id))
    if not test_ids and dev_ids and remaining_rows:
        move = remaining_rows[-1]["ids"]
        for job_id in move:
            test_ids.add(int(job_id))
            dev_ids.discard(int(job_id))

    if challenge_ids & dev_ids or challenge_ids & test_ids or dev_ids & test_ids:
        raise RuntimeError("split overlap detected")

    return {
        "dev": sorted(int(job_id) for job_id in dev_ids),
        "test": sorted(int(job_id) for job_id in test_ids),
        "challenge": sorted(int(job_id) for job_id in challenge_ids),
    }


def _split_loss(
    *,
    dev_jobs: float,
    dev_pos: float,
    dev_neg: float,
    target_dev_jobs: float,
    target_dev_pos: float,
    target_dev_neg: float,
) -> float:
    return (
        abs(float(dev_jobs) - float(target_dev_jobs))
        + abs(float(dev_pos) - float(target_dev_pos))
        + abs(float(dev_neg) - float(target_dev_neg))
    )


def _split_ids_from_strategy(
    *,
    strategy: str,
    completed_jobs: list[dict[str, Any]],
    labels: dict[int, bool],
    eligible_ids: list[int],
    hardness_by_id: dict[int, float],
    dev_ratio: float,
    test_ratio: float,
    challenge_ratio: float,
    seed: int,
    holdout_site: str | None,
) -> dict[str, list[int]]:
    job_by_id = {int(job["id"]): job for job in completed_jobs}
    if strategy == "random_stratified":
        return _split_ids(
            job_ids=eligible_ids,
            labels=labels,
            hardness_by_id=hardness_by_id,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            challenge_ratio=challenge_ratio,
            seed=seed,
        )

    if strategy == "leave_one_site_out":
        if not holdout_site:
            raise SystemExit("--holdout-site is required when --split-strategy=leave_one_site_out")
        challenge_ids = {
            int(job_id)
            for job_id in eligible_ids
            if str(job_by_id[int(job_id)].get("trainee_site_id") or "unknown") == str(holdout_site)
        }
        if not challenge_ids:
            raise SystemExit(f"holdout site '{holdout_site}' has no eligible labeled jobs")
        group_by_id = {
            int(job_id): f"trainee:{int(job_by_id[int(job_id)]['trainee_video_id'])}"
            for job_id in eligible_ids
        }
        return _split_ids_grouped(
            job_ids=eligible_ids,
            labels=labels,
            hardness_by_id=hardness_by_id,
            group_by_id=group_by_id,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            challenge_ratio=0.0,
            seed=seed,
            challenge_fixed_ids=challenge_ids,
        )

    group_by_id: dict[int, str] = {}
    for job_id in eligible_ids:
        job = job_by_id[int(job_id)]
        if strategy == "group_trainee":
            group_by_id[int(job_id)] = f"trainee:{int(job['trainee_video_id'])}"
        elif strategy == "group_gold":
            group_by_id[int(job_id)] = f"gold:{int(job['gold_video_id'])}"
        elif strategy == "group_site":
            site = str(job.get("trainee_site_id") or "unknown")
            group_by_id[int(job_id)] = f"site:{site}"
        else:
            raise SystemExit(f"unsupported split strategy: {strategy}")
    return _split_ids_grouped(
        job_ids=eligible_ids,
        labels=labels,
        hardness_by_id=hardness_by_id,
        group_by_id=group_by_id,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        challenge_ratio=challenge_ratio,
        seed=seed,
        challenge_fixed_ids=None,
    )


def _validate_split_ids(*, split_ids: dict[str, list[int]], eligible_ids: list[int]) -> None:
    dev = {int(job_id) for job_id in split_ids.get("dev", [])}
    test = {int(job_id) for job_id in split_ids.get("test", [])}
    challenge = {int(job_id) for job_id in split_ids.get("challenge", [])}

    if dev & test or dev & challenge or test & challenge:
        raise SystemExit("split manifest has overlapping split_job_ids")

    eligible = {int(job_id) for job_id in eligible_ids}
    used = dev | test | challenge
    unknown = sorted(used - eligible)
    missing = sorted(eligible - used)
    if unknown:
        raise SystemExit(f"split manifest contains unknown job ids: {unknown[:10]}")
    if missing:
        raise SystemExit(f"split manifest is missing eligible job ids: {missing[:10]}")


def _load_split_manifest(path: Path) -> tuple[dict[str, list[int]], dict[str, Any]]:
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
    return out, payload


def _build_split_manifest_payload(
    *,
    task_id: str | None,
    strategy: str,
    holdout_site: str | None,
    seed: int,
    split_ratios: dict[str, float],
    split_ids: dict[str, list[int]],
) -> dict[str, Any]:
    payload = {
        "version": "split_manifest_v1",
        "task_id": task_id,
        "strategy": strategy,
        "holdout_site": holdout_site,
        "seed": int(seed),
        "split_ratios": split_ratios,
        "split_job_ids": {
            "dev": [int(job_id) for job_id in split_ids.get("dev", [])],
            "test": [int(job_id) for job_id in split_ids.get("test", [])],
            "challenge": [int(job_id) for job_id in split_ids.get("challenge", [])],
        },
    }
    return attach_payload_hash(payload)


def _labels_subset_payload(
    *,
    labels_payload: dict[str, Any],
    split_ids: set[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in labels_payload.get("jobs", []):
        if not isinstance(item, dict):
            continue
        if item.get("job_id") is None:
            continue
        try:
            job_id = int(item["job_id"])
        except Exception:
            continue
        if job_id not in split_ids:
            continue
        rows.append(dict(item))
    rows.sort(key=lambda row: int(row["job_id"]))
    return {
        "task_id": labels_payload.get("task_id"),
        "jobs": rows,
    }


def _counts(labels: dict[int, bool], ids: list[int]) -> dict[str, int]:
    pos = 0
    neg = 0
    for job_id in ids:
        if labels.get(job_id, False):
            pos += 1
        else:
            neg += 1
    return {
        "jobs": int(len(ids)),
        "critical_positives": int(pos),
        "critical_negatives": int(neg),
    }


def _without_min_evidence(config: GateConfig) -> GateConfig:
    payload = asdict(config)
    for key in _MIN_EVIDENCE_KEYS:
        payload[key] = None
    return GateConfig(**payload)


def _build_gate_config(args: argparse.Namespace) -> GateConfig:
    base = get_gate_profile(args.gate_profile)
    has_overrides = any(
        value is not None
        for value in (
            args.max_critical_miss_rate,
            args.max_critical_fpr,
            args.max_critical_miss_ci95_high,
            args.max_critical_fpr_ci95_high,
            args.max_rescore_jitter,
            args.max_dtw_p90,
            args.max_drift_critical_score_psi,
            args.max_drift_score_psi,
            args.max_drift_dtw_psi,
            args.max_critical_detected_rate_shift_abs,
            args.min_completed_jobs,
            args.min_labels_total_jobs,
            args.min_labeled_jobs,
            args.min_critical_positives,
            args.min_critical_negatives,
            args.min_coverage_rate,
            args.min_rescore_pairs,
        )
    )
    if is_gate_profile_locked(args.gate_profile) and has_overrides and not args.allow_profile_overrides:
        raise SystemExit(
            f"gate profile '{args.gate_profile}' is locked; pass --allow-profile-overrides to override thresholds"
        )

    merged = merge_gate_config(
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
    if args.strict_evidence_gates:
        return merged
    return _without_min_evidence(merged)


def evaluate_split_profile(
    *,
    completed_jobs: list[dict[str, Any]],
    labels: dict[int, bool],
    split_ids: dict[str, list[int]],
    candidate_modes: list[str],
    critical_threshold: float,
    critical_policy: dict[str, Any] | None,
    gate_config: GateConfig,
) -> dict[str, Any]:
    job_by_id = {int(job["id"]): job for job in completed_jobs}
    out: dict[str, Any] = {}
    all_ids = sorted({int(job_id) for job_id in labels if int(job_id) in job_by_id})
    eval_targets = {"full": all_ids, **split_ids}

    for mode in candidate_modes:
        mode_reports: dict[str, Any] = {}
        for split_name, ids in eval_targets.items():
            ids_set = set(ids)
            subset_jobs = [job_by_id[job_id] for job_id in ids if job_id in job_by_id]
            subset_labels = {job_id: labels[job_id] for job_id in ids if job_id in labels}
            report = compute_poc_metrics(
                subset_jobs,
                subset_labels,
                label_scope={
                    "labels_total_jobs": len(ids_set),
                    "labels_labeled_jobs": len(ids_set),
                    "labels_unknown_jobs": 0,
                },
                critical_scoring_mode=mode,
                critical_threshold=critical_threshold,
                critical_policy=critical_policy,
            )
            report["split"] = split_name
            report["critical_scoring_mode"] = mode
            report["critical_threshold"] = critical_threshold
            report["gates"] = evaluate_gates(report, gate_config)
            mode_reports[split_name] = report
        out[mode] = mode_reports
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate gate profiles on Dev/Test/Challenge splits.")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--output-dir", default=None, help="Output directory for split labels and report")
    parser.add_argument("--seed", type=int, default=20260223)
    parser.add_argument("--dev-ratio", type=float, default=0.6)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--challenge-ratio", type=float, default=0.2)
    parser.add_argument(
        "--split-strategy",
        choices=list(_SPLIT_STRATEGIES),
        default="random_stratified",
    )
    parser.add_argument(
        "--holdout-site",
        default=None,
        help="Required for split-strategy=leave_one_site_out",
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Optional path to existing split manifest JSON. When provided, split generation is skipped.",
    )
    parser.add_argument(
        "--write-split-manifest",
        default=None,
        help="Optional output path for generated split manifest JSON.",
    )
    parser.add_argument(
        "--candidate-mode",
        action="append",
        choices=available_critical_scoring_modes(),
        default=None,
        help="Candidate critical scoring mode to evaluate. Repeatable.",
    )
    parser.add_argument("--critical-policy", default=None, help="Optional policy JSON (fit artifact).")
    parser.add_argument("--critical-threshold", type=float, default=None)
    parser.add_argument(
        "--gate-profile",
        choices=available_gate_profiles(),
        default="research_v2",
    )
    parser.add_argument("--allow-profile-overrides", action="store_true")
    parser.add_argument(
        "--strict-evidence-gates",
        action="store_true",
        help="Keep min-evidence thresholds enabled for per-split gate checks.",
    )
    parser.add_argument("--write-split-labels", action="store_true")

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
    args = parser.parse_args()

    db = Database(Path(args.db_path))
    completed_jobs = db.list_completed_score_jobs(task_id=args.task_id)
    labels_payload = _load_labels_payload(Path(args.labels))
    labels, _ = _extract_labeled_job_ids(labels_payload)
    completed_ids = {int(job["id"]) for job in completed_jobs}
    eligible_ids = sorted([job_id for job_id in labels if job_id in completed_ids])
    if not eligible_ids:
        raise SystemExit("no labeled completed jobs found for split evaluation")

    critical_policy = None
    if args.critical_policy:
        critical_policy = json.loads(Path(args.critical_policy).read_text(encoding="utf-8"))
    critical_threshold = args.critical_threshold
    if critical_threshold is None:
        critical_threshold = policy_critical_threshold(critical_policy)
    if critical_threshold is None:
        critical_threshold = 0.5

    job_by_id = {int(job["id"]): job for job in completed_jobs}
    hardness_by_id: dict[int, float] = {}
    for job_id in eligible_ids:
        score = job_by_id[job_id]["score"]
        critical_score = float(compute_critical_score(score))
        if labels[job_id]:
            hardness_by_id[job_id] = 1.0 - critical_score
        else:
            hardness_by_id[job_id] = critical_score

    split_manifest_payload: dict[str, Any] | None = None
    if args.split_manifest:
        split_manifest_path = Path(args.split_manifest).resolve()
        split_ids, split_manifest_payload = _load_split_manifest(split_manifest_path)
        _validate_split_ids(split_ids=split_ids, eligible_ids=eligible_ids)
        strategy_used = str(split_manifest_payload.get("strategy") or "manifest")
    else:
        strategy_used = str(args.split_strategy)
        split_ids = _split_ids_from_strategy(
            strategy=strategy_used,
            completed_jobs=completed_jobs,
            labels=labels,
            eligible_ids=eligible_ids,
            hardness_by_id=hardness_by_id,
            dev_ratio=float(args.dev_ratio),
            test_ratio=float(args.test_ratio),
            challenge_ratio=float(args.challenge_ratio),
            seed=int(args.seed),
            holdout_site=args.holdout_site,
        )
        _validate_split_ids(split_ids=split_ids, eligible_ids=eligible_ids)
        split_manifest_payload = _build_split_manifest_payload(
            task_id=args.task_id,
            strategy=strategy_used,
            holdout_site=args.holdout_site,
            seed=int(args.seed),
            split_ratios={
                "dev": float(args.dev_ratio),
                "test": float(args.test_ratio),
                "challenge": float(args.challenge_ratio),
            },
            split_ids=split_ids,
        )

    gate_config = _build_gate_config(args)
    policy_mode = policy_scoring_mode(critical_policy)
    candidate_modes = args.candidate_mode or ([policy_mode] if policy_mode else ["guarded_binary_v1"])
    report_payload = {
        "task_id": args.task_id,
        "db_path": str(Path(args.db_path).resolve()),
        "labels_path": str(Path(args.labels).resolve()),
        "seed": int(args.seed),
        "split_strategy": strategy_used,
        "holdout_site": args.holdout_site,
        "split_manifest_path": str(Path(args.split_manifest).resolve()) if args.split_manifest else None,
        "split_ratios": {
            "dev": float(args.dev_ratio),
            "test": float(args.test_ratio),
            "challenge": float(args.challenge_ratio),
        },
        "critical_threshold": float(critical_threshold),
        "candidate_modes": candidate_modes,
        "critical_policy_path": str(Path(args.critical_policy).resolve()) if args.critical_policy else None,
        "gate_profile": args.gate_profile,
        "gate_profile_locked": is_gate_profile_locked(args.gate_profile),
        "gate_config_used": {key: value for key, value in asdict(gate_config).items() if value is not None},
        "strict_evidence_gates": bool(args.strict_evidence_gates),
        "eligible_counts": _counts(labels, eligible_ids),
        "split_counts": {
            split_name: _counts(labels, ids)
            for split_name, ids in split_ids.items()
        },
        "split_job_ids": split_ids,
        "split_manifest": split_manifest_payload,
    }
    report_payload["results"] = evaluate_split_profile(
        completed_jobs=completed_jobs,
        labels=labels,
        split_ids=split_ids,
        candidate_modes=candidate_modes,
        critical_threshold=float(critical_threshold),
        critical_policy=critical_policy,
        gate_config=gate_config,
    )

    output_dir: Path | None = None
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_split_labels:
        if output_dir is None:
            raise SystemExit("--write-split-labels requires --output-dir")
        for split_name, ids in split_ids.items():
            subset = _labels_subset_payload(labels_payload=labels_payload, split_ids=set(ids))
            (output_dir / f"labels_{split_name}.json").write_text(
                json.dumps(subset, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    manifest_out_path: Path | None = None
    if args.write_split_manifest:
        manifest_out_path = Path(args.write_split_manifest).resolve()
    elif output_dir is not None and split_manifest_payload is not None:
        manifest_out_path = output_dir / "split_manifest.json"
    if manifest_out_path is not None and split_manifest_payload is not None:
        manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_out_path.write_text(
            json.dumps(split_manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    text = json.dumps(report_payload, ensure_ascii=False, indent=2)
    print(text)

    out_path: Path | None = None
    if args.output:
        out_path = Path(args.output).resolve()
    elif output_dir is not None:
        out_path = output_dir / "split_evaluation_report.json"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
