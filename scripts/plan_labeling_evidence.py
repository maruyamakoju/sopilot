from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from sopilot.database import Database
from sopilot.eval.harness import load_critical_labels


def _wilson_upper_zero_errors(total: int) -> float | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    n = float(total)
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = (z2 / (2.0 * n)) / denom
    margin = (z / denom) * ((z2 / (4.0 * n * n)) ** 0.5)
    return round(float(min(1.0, center + margin)), 6)


def _min_n_for_zero_error_ci(target_upper: float, *, max_n: int = 20000) -> int:
    target = float(target_upper)
    if target <= 0.0:
        return max_n
    for n in range(1, int(max_n) + 1):
        upper = _wilson_upper_zero_errors(n)
        if upper is not None and upper <= target:
            return int(n)
    return int(max_n)


def _load_split_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_mode(split_report: dict[str, Any], mode: str | None) -> str:
    if mode:
        return str(mode)
    results = split_report.get("results", {}) or {}
    if not isinstance(results, dict) or not results:
        raise SystemExit("split report has no results")
    return sorted(results.keys())[0]


def _split_row(split_report: dict[str, Any], mode: str, split_name: str) -> dict[str, Any]:
    results = split_report.get("results", {}) or {}
    mode_payload = (results.get(mode) or {}) if isinstance(results, dict) else {}
    row = mode_payload.get(split_name) if isinstance(mode_payload, dict) else None
    if not isinstance(row, dict):
        raise SystemExit(f"missing results row for mode={mode}, split={split_name}")
    return row


def _metric_thresholds(split_report: dict[str, Any], miss_ci_target: float | None, fpr_ci_target: float | None) -> tuple[float, float]:
    gate = split_report.get("gate_config_used", {}) or {}
    miss = miss_ci_target
    fpr = fpr_ci_target
    if miss is None:
        miss = float(gate.get("max_critical_miss_rate_ci95_high") or 0.2)
    if fpr is None:
        fpr = float(gate.get("max_critical_false_positive_rate_ci95_high") or 0.2)
    return float(miss), float(fpr)


def _wilson_ci(positives: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if total <= 0:
        return None
    n = float(total)
    phat = float(max(0, positives)) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return float(low), float(high)


def _binomial_tail_at_least(*, n: int, k: int, p: float) -> float:
    if k <= 0:
        return 1.0
    if n < k:
        return 0.0
    p = max(0.0, min(1.0, float(p)))
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    q = 1.0 - p
    pmf = q**n  # P(X=0)
    cdf = pmf
    x = 0
    while x < (k - 1):
        num = float(n - x)
        den = float(x + 1)
        pmf = pmf * (num / den) * (p / q)
        cdf += pmf
        x += 1
    tail = max(0.0, min(1.0, 1.0 - cdf))
    return float(tail)


def _min_labels_for_positive_target(
    *,
    positives_needed: int,
    prevalence: float,
    hit_probability: float,
    max_n: int,
) -> int:
    need = int(max(0, positives_needed))
    if need <= 0:
        return 0
    p = max(0.0, min(1.0, float(prevalence)))
    if p <= 0.0:
        return int(max_n)
    target_prob = max(0.5, min(0.999999, float(hit_probability)))
    for n in range(need, int(max_n) + 1):
        if _binomial_tail_at_least(n=n, k=need, p=p) >= target_prob:
            return int(n)
    return int(max_n)


def _difficulty_score(job: dict[str, Any]) -> tuple[float, list[str]]:
    score_payload = job.get("score", {}) or {}
    metrics = score_payload.get("metrics", {}) or {}
    deviations = score_payload.get("deviations", []) or []

    score = float(score_payload.get("score") or 0.0)
    dtw = float(metrics.get("dtw_normalized_cost") or 0.0)
    over_time = float(metrics.get("over_time_ratio") or 0.0)
    miss_steps = float(metrics.get("miss_steps") or 0.0)
    deviation_steps = float(metrics.get("deviation_steps") or 0.0)

    tags: list[str] = []
    points = 0.0
    if score < 85.0:
        points += (85.0 - score) / 30.0
        tags.append("low_score")
    if dtw > 0.06:
        points += min(2.0, dtw / 0.1)
        tags.append("high_dtw")
    if over_time > 0.25:
        points += min(2.0, over_time / 0.5)
        tags.append("high_over_time")
    if miss_steps > 0:
        points += min(1.5, miss_steps / 2.0)
        tags.append("has_miss_steps")
    if deviation_steps > 0:
        points += min(1.0, deviation_steps / 2.0)
        tags.append("has_deviation_steps")
    if any(str(dev.get("type", "")).strip().lower() == "missing_step" for dev in deviations):
        points += 1.0
        tags.append("deviation_missing_step")
    if not tags:
        tags.append("default")
    return round(float(points), 6), tags


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _select_challenge_candidates(
    *,
    rows_all: list[dict[str, Any]],
    limit: int,
    max_per_site: int,
) -> list[dict[str, Any]]:
    take = max(0, int(limit))
    if take <= 0:
        return []
    if int(max_per_site) <= 0:
        return rows_all[:take]

    per_site: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows_all:
        site = str(row.get("site_id") or "unknown")
        if per_site.get(site, 0) < int(max_per_site):
            selected.append(row)
            per_site[site] = int(per_site.get(site, 0) + 1)
            if len(selected) >= take:
                return selected
        else:
            skipped.append(row)

    for row in skipped:
        selected.append(row)
        if len(selected) >= take:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan label expansion for CI-tight Test/Challenge evidence.")
    parser.add_argument("--db-path", "--db", dest="db_path", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split-report", required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--candidate-mode", default=None)
    parser.add_argument("--target-miss-ci95-high", type=float, default=None)
    parser.add_argument("--target-fpr-ci95-high", type=float, default=None)
    parser.add_argument("--target-positive-hit-prob", type=float, default=0.9)
    parser.add_argument("--prevalence-floor", type=float, default=0.05)
    parser.add_argument("--max-label-budget", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260223)
    parser.add_argument("--test-random-candidates", type=int, default=200)
    parser.add_argument("--challenge-rule-candidates", type=int, default=200)
    parser.add_argument("--challenge-max-per-site", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    split_report = _load_split_report(Path(args.split_report))
    mode = _candidate_mode(split_report, args.candidate_mode)
    miss_ci_target, fpr_ci_target = _metric_thresholds(
        split_report,
        args.target_miss_ci95_high,
        args.target_fpr_ci95_high,
    )

    labels = load_critical_labels(Path(args.labels)).labels
    db = Database(Path(args.db_path))
    completed = db.list_completed_score_jobs(task_id=args.task_id)
    completed_by_id = {int(job["id"]): job for job in completed}

    test_row = _split_row(split_report, mode, "test")
    challenge_row = _split_row(split_report, mode, "challenge")
    full_row = _split_row(split_report, mode, "full")

    required_pos = _min_n_for_zero_error_ci(miss_ci_target)
    required_neg = _min_n_for_zero_error_ci(fpr_ci_target)
    test_pos = int(test_row.get("critical_positives") or 0)
    test_neg = int(test_row.get("critical_negatives") or 0)
    challenge_pos = int(challenge_row.get("critical_positives") or 0)
    challenge_neg = int(challenge_row.get("critical_negatives") or 0)

    overall_pos_ratio = 0.0
    full_pos = int(full_row.get("critical_positives") or 0)
    full_total = int((full_row.get("critical_positives") or 0) + (full_row.get("critical_negatives") or 0))
    prevalence_ci = _wilson_ci(full_pos, full_total)
    if full_total > 0:
        overall_pos_ratio = float(full_pos / full_total)
    prevalence_floor = max(0.0, min(1.0, float(args.prevalence_floor)))
    prevalence_lower = prevalence_floor
    prevalence_upper = 1.0
    if prevalence_ci is not None:
        prevalence_lower = max(prevalence_floor, float(prevalence_ci[0]))
        prevalence_upper = float(prevalence_ci[1])

    def _label_budget_naive(need_pos: int) -> int:
        if need_pos <= 0:
            return 0
        prevalence = max(prevalence_floor, float(overall_pos_ratio))
        return int(round(need_pos / prevalence + 0.499999))

    def _label_budget_conservative(need_pos: int) -> int:
        return _min_labels_for_positive_target(
            positives_needed=need_pos,
            prevalence=prevalence_lower,
            hit_probability=float(args.target_positive_hit_prob),
            max_n=max(1, int(args.max_label_budget)),
        )

    completed_ids = set(completed_by_id.keys())
    labeled_ids = {int(job_id) for job_id in labels if int(job_id) in completed_ids}
    unlabeled_ids = sorted(completed_ids - labeled_ids)

    rng = random.Random(int(args.seed))
    random_pool = list(unlabeled_ids)
    rng.shuffle(random_pool)
    test_pos_need = int(max(0, required_pos - test_pos))
    challenge_pos_need = int(max(0, required_pos - challenge_pos))
    test_budget_naive = int(_label_budget_naive(test_pos_need))
    challenge_budget_naive = int(_label_budget_naive(challenge_pos_need))
    test_budget_conservative = int(_label_budget_conservative(test_pos_need))
    challenge_budget_conservative = int(_label_budget_conservative(challenge_pos_need))

    test_candidate_n = int(
        max(
            int(args.test_random_candidates),
            test_budget_conservative,
        )
    )
    challenge_candidate_n = int(
        max(
            int(args.challenge_rule_candidates),
            challenge_budget_conservative,
        )
    )

    test_rows: list[dict[str, Any]] = []
    for job_id in random_pool[: max(0, int(test_candidate_n))]:
        job = completed_by_id[job_id]
        test_rows.append(
            {
                "job_id": int(job_id),
                "selection_method": "model_independent_random",
                "seed": int(args.seed),
                "task_id": job.get("task_id"),
                "site_id": job.get("trainee_site_id"),
                "gold_video_id": int(job.get("gold_video_id")),
                "trainee_video_id": int(job.get("trainee_video_id")),
            }
        )

    challenge_rows_all: list[dict[str, Any]] = []
    for job_id in unlabeled_ids:
        job = completed_by_id[job_id]
        difficulty, tags = _difficulty_score(job)
        challenge_rows_all.append(
            {
                "job_id": int(job_id),
                "selection_method": "predefined_hard_conditions",
                "difficulty_score": float(difficulty),
                "difficulty_tags": tags,
                "task_id": job.get("task_id"),
                "site_id": job.get("trainee_site_id"),
                "gold_video_id": int(job.get("gold_video_id")),
                "trainee_video_id": int(job.get("trainee_video_id")),
            }
        )
    challenge_rows_all.sort(key=lambda row: (-float(row["difficulty_score"]), int(row["job_id"])))
    challenge_rows = _select_challenge_candidates(
        rows_all=challenge_rows_all,
        limit=max(0, int(challenge_candidate_n)),
        max_per_site=max(0, int(args.challenge_max_per_site)),
    )

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _jsonl(out_dir / "test_random_label_candidates.jsonl", test_rows)
    _jsonl(out_dir / "challenge_rule_label_candidates.jsonl", challenge_rows)

    requirements = {
        "task_id": args.task_id,
        "split_report": str(Path(args.split_report).resolve()),
        "candidate_mode": mode,
        "seed": int(args.seed),
        "ci_targets": {
            "miss_ci95_high": float(miss_ci_target),
            "fpr_ci95_high": float(fpr_ci_target),
        },
        "required_n_if_zero_error": {
            "critical_positives": int(required_pos),
            "critical_negatives": int(required_neg),
        },
        "required_for_zero_error": {
            "critical_positives": int(required_pos),
            "critical_negatives": int(required_neg),
        },
        "prevalence_estimate": {
            "full_positive_ratio_point": round(float(overall_pos_ratio), 6),
            "full_positive_ratio_wilson95_low": round(float(prevalence_lower), 6),
            "full_positive_ratio_wilson95_high": round(float(prevalence_upper), 6),
            "prevalence_floor": float(prevalence_floor),
            "conservative_prevalence_used": round(float(prevalence_lower), 6),
        },
        "label_budget_model": {
            "type": "binomial_tail",
            "description": "minimum n where P(X>=positives_needed) >= target_positive_hit_prob, X~Binomial(n, conservative_prevalence)",
            "target_positive_hit_prob": float(args.target_positive_hit_prob),
            "max_label_budget": int(args.max_label_budget),
        },
        "current_counts": {
            "test": {"critical_positives": test_pos, "critical_negatives": test_neg},
            "challenge": {"critical_positives": challenge_pos, "critical_negatives": challenge_neg},
        },
        "deficits": {
            "test": {
                "positives_needed": int(test_pos_need),
                "negatives_needed": int(max(0, required_neg - test_neg)),
                "suggested_labels_to_add": int(test_budget_conservative),
                "suggested_labels_to_add_naive": int(test_budget_naive),
                "suggested_labels_to_add_conservative": int(test_budget_conservative),
            },
            "challenge": {
                "positives_needed": int(challenge_pos_need),
                "negatives_needed": int(max(0, required_neg - challenge_neg)),
                "suggested_labels_to_add": int(challenge_budget_conservative),
                "suggested_labels_to_add_naive": int(challenge_budget_naive),
                "suggested_labels_to_add_conservative": int(challenge_budget_conservative),
            },
        },
        "selection_protocol": {
            "test": "random sampling from unlabeled completed jobs (seed fixed)",
            "challenge": "predefined hard-condition rules independent from candidate detector",
        },
        "unlabeled_completed_jobs": int(len(unlabeled_ids)),
        "candidate_pool_sizes": {
            "test_random": int(len(test_rows)),
            "challenge_rules": int(len(challenge_rows)),
        },
        "candidate_generation": {
            "test_random_candidates_requested": int(args.test_random_candidates),
            "challenge_rule_candidates_requested": int(args.challenge_rule_candidates),
            "challenge_max_per_site": int(args.challenge_max_per_site),
        },
        "candidate_files": {
            "test_random": str(out_dir / "test_random_label_candidates.jsonl"),
            "challenge_rules": str(out_dir / "challenge_rule_label_candidates.jsonl"),
        },
    }
    text = json.dumps(requirements, ensure_ascii=False, indent=2)
    print(text)
    (out_dir / "ci_labeling_plan.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
