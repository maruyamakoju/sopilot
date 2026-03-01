import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CRITICAL_SCORING_MODE = "legacy_binary"
DEFAULT_CRITICAL_THRESHOLD = 0.5
GUARDED_BINARY_V1_MIN_DTW = 0.025
GUARDED_BINARY_V2_MIN_DTW = 0.025
GUARDED_BINARY_V2_MAX_CRITICAL_MISSING_MEAN_DISTANCE = 0.11
GUARDED_BINARY_V2_MAX_CRITICAL_MISSING_EXPECTED_SPAN = 1.5
_DRIFT_MIN_WINDOW = 20
_CRITICAL_SCORING_MODES = {"legacy_binary", "continuous_v1", "guarded_binary_v1", "guarded_binary_v2"}


@dataclass(frozen=True)
class CriticalLabel:
    job_id: int
    critical_expected: bool


@dataclass(frozen=True)
class CriticalLabelSet:
    labels: dict[int, bool]
    total_jobs: int
    labeled_jobs: int
    unknown_jobs: int


def available_critical_scoring_modes() -> list[str]:
    return sorted(_CRITICAL_SCORING_MODES)


def normalize_critical_scoring_mode(mode: str | None) -> str:
    if not mode:
        return DEFAULT_CRITICAL_SCORING_MODE
    normalized = str(mode).strip().lower()
    if normalized not in _CRITICAL_SCORING_MODES:
        raise ValueError(
            f"unsupported critical scoring mode: {mode} "
            f"(expected one of: {', '.join(available_critical_scoring_modes())})"
        )
    return normalized


def load_critical_labels(path: Path) -> CriticalLabelSet:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs", [])
    labels: dict[int, bool] = {}
    all_job_ids: set[int] = set()
    labeled_job_ids: set[int] = set()
    for item in jobs:
        if not isinstance(item, dict):
            continue
        raw_job_id = item.get("job_id")
        if raw_job_id is None:
            continue
        try:
            job_id = int(raw_job_id)
        except Exception:
            continue
        all_job_ids.add(job_id)
        expected = item.get("critical_expected")
        if expected is None:
            continue
        labels[job_id] = bool(expected)
        labeled_job_ids.add(job_id)
    total_jobs = int(len(all_job_ids))
    labeled_jobs = int(len(labeled_job_ids))
    unknown_jobs = int(max(0, total_jobs - labeled_jobs))
    return CriticalLabelSet(
        labels=labels,
        total_jobs=total_jobs,
        labeled_jobs=labeled_jobs,
        unknown_jobs=unknown_jobs,
    )


def parse_critical_labels(path: Path) -> dict[int, bool]:
    return load_critical_labels(path).labels


def build_critical_score_breakdown(job_score: dict[str, Any]) -> dict[str, Any]:
    """Build a continuous critical score in [0, 1] with interpretable components.

    The score is intentionally conservative:
    - Any explicit critical deviation should produce a high score.
    - Non-critical noise stays low unless multiple weak signals accumulate.
    """
    severity_counts = _severity_counts(job_score)
    metrics = dict(job_score.get("metrics", {}) or {})
    critical_count = int(severity_counts.get("critical", 0))
    quality_count = int(severity_counts.get("quality", 0))
    efficiency_count = int(severity_counts.get("efficiency", 0))

    miss_steps = _non_negative_int(metrics.get("miss_steps"))
    swap_steps = _non_negative_int(metrics.get("swap_steps"))
    deviation_steps = _non_negative_int(metrics.get("deviation_steps"))
    dtw = _non_negative_float(metrics.get("dtw_normalized_cost"))
    over_time_ratio = _non_negative_float(metrics.get("over_time_ratio"))

    components = {
        "critical_deviation": 2.50 * float(critical_count),
        "quality_deviation": 0.35 * float(quality_count),
        "efficiency_deviation": 0.15 * float(efficiency_count),
        "miss_steps": 0.20 * float(miss_steps),
        "swap_steps": 0.10 * float(swap_steps),
        "deviation_steps": 0.15 * float(deviation_steps),
        "dtw_cost_excess": max(0.0, float(dtw) - 0.08) * 3.0,
        "over_time_excess": max(0.0, float(over_time_ratio) - 0.20) * 0.8,
    }
    raw_score = float(sum(components.values()))
    critical_score = 1.0 - math.exp(-raw_score)
    critical_score = max(0.0, min(1.0, float(critical_score)))

    return {
        "critical_score": round(float(critical_score), 6),
        "raw_score": round(float(raw_score), 6),
        "severity_counts": severity_counts,
        "metrics_snapshot": {
            "miss_steps": int(miss_steps),
            "swap_steps": int(swap_steps),
            "deviation_steps": int(deviation_steps),
            "dtw_normalized_cost": round(float(dtw), 6),
            "over_time_ratio": round(float(over_time_ratio), 6),
        },
        "components": {k: round(float(v), 6) for k, v in components.items()},
    }


def compute_critical_score(job_score: dict[str, Any]) -> float:
    return float(build_critical_score_breakdown(job_score)["critical_score"])


def policy_scoring_mode(critical_policy: dict[str, Any] | None) -> str | None:
    if not isinstance(critical_policy, dict):
        return None
    raw = critical_policy.get("scoring_mode")
    if raw is None:
        return None
    try:
        return normalize_critical_scoring_mode(str(raw))
    except Exception:
        return None


def policy_critical_threshold(critical_policy: dict[str, Any] | None) -> float | None:
    if not isinstance(critical_policy, dict):
        return None
    return _to_float(critical_policy.get("critical_threshold"))


def job_detected_critical(
    job_score: dict[str, Any],
    *,
    scoring_mode: str | None = None,
    critical_threshold: float | None = DEFAULT_CRITICAL_THRESHOLD,
    critical_policy: dict[str, Any] | None = None,
) -> bool:
    mode = normalize_critical_scoring_mode(scoring_mode)
    if mode in {"legacy_binary", "guarded_binary_v1", "guarded_binary_v2"}:
        deviations = job_score.get("deviations", [])
        has_critical = False
        for dev in deviations:
            severity = str(dev.get("severity", ""))
            # Treat as critical only when task policy marks the deviation critical.
            # This keeps evaluation aligned with task-specific severity mapping.
            if severity == "critical":
                has_critical = True
                break
        if not has_critical:
            return False
        if mode == "legacy_binary":
            return True
        metrics = dict(job_score.get("metrics", {}) or {})
        dtw = _non_negative_float(metrics.get("dtw_normalized_cost"))
        if mode == "guarded_binary_v1":
            min_dtw = _resolve_guardrail_non_negative(
                critical_policy,
                mode="guarded_binary_v1",
                key="min_dtw",
                default=GUARDED_BINARY_V1_MIN_DTW,
            )
            return dtw >= min_dtw

        # Stage-B v2 guardrail:
        # 1) Keep v1 DTW floor
        # 2) For critical missing-step deviations, require collapse pattern
        #    consistent with true freeze (tight expected span, moderate distance).
        min_dtw = _resolve_guardrail_non_negative(
            critical_policy,
            mode="guarded_binary_v2",
            key="min_dtw",
            default=GUARDED_BINARY_V2_MIN_DTW,
        )
        if dtw < min_dtw:
            return False
        missing_stats = _critical_missing_stats(job_score)
        max_mean_distance = _resolve_guardrail_non_negative(
            critical_policy,
            mode="guarded_binary_v2",
            key="max_critical_missing_mean_distance",
            default=GUARDED_BINARY_V2_MAX_CRITICAL_MISSING_MEAN_DISTANCE,
        )
        max_expected_span = _resolve_guardrail_non_negative(
            critical_policy,
            mode="guarded_binary_v2",
            key="max_critical_missing_expected_span",
            default=GUARDED_BINARY_V2_MAX_CRITICAL_MISSING_EXPECTED_SPAN,
        )
        return not (missing_stats["num_critical_missing"] > 0 and (missing_stats["mean_distance_avg"] > max_mean_distance or missing_stats["expected_span_avg"] > max_expected_span))
    threshold = _normalize_threshold(critical_threshold)
    return compute_critical_score(job_score) >= threshold


def compute_poc_metrics(
    completed_jobs: list[dict[str, Any]],
    critical_labels: dict[int, bool] | None = None,
    *,
    label_scope: dict[str, int] | None = None,
    critical_scoring_mode: str | None = None,
    critical_threshold: float | None = DEFAULT_CRITICAL_THRESHOLD,
    critical_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scoring_mode = normalize_critical_scoring_mode(critical_scoring_mode)
    threshold = _normalize_threshold(critical_threshold)

    dtw_values: list[float] = []
    scores: list[float] = []
    critical_scores: list[float] = []
    detected_flags: list[float] = []
    critical_detected = 0

    for job in completed_jobs:
        score_payload = job["score"]
        metrics = score_payload.get("metrics", {})
        dtw = metrics.get("dtw_normalized_cost")
        if dtw is not None:
            dtw_values.append(float(dtw))
        scores.append(float(score_payload.get("score", 0.0)))
        critical_scores.append(float(compute_critical_score(score_payload)))
        detected = job_detected_critical(
            score_payload,
            scoring_mode=scoring_mode,
            critical_threshold=threshold,
            critical_policy=critical_policy,
        )
        detected_flags.append(1.0 if detected else 0.0)
        if detected:
            critical_detected += 1

    metrics_out: dict[str, Any] = {
        "num_completed_jobs": len(completed_jobs),
        "critical_detected_jobs": critical_detected,
        "critical_scoring_mode": scoring_mode,
        "critical_threshold": threshold,
        "critical_score_stats": _stats(critical_scores),
        "score_stats": _stats(scores),
        "dtw_normalized_cost_stats": _stats(dtw_values),
        "rescore_jitter": _rescore_jitter(completed_jobs),
        "drift": _compute_drift_summary(
            scores=scores,
            critical_scores=critical_scores,
            dtw_values=dtw_values,
            detected_flags=detected_flags,
        ),
        "critical_policy": _critical_policy_summary(critical_policy),
    }

    if critical_labels is not None:
        miss_rate, fp_rate, confusion = _critical_rates(
            completed_jobs,
            critical_labels,
            scoring_mode=scoring_mode,
            critical_threshold=threshold,
            critical_policy=critical_policy,
        )
        critical_positives = int(confusion["tp"] + confusion["fn"])
        critical_negatives = int(confusion["fp"] + confusion["tn"])
        completed_labeled_jobs = int(critical_positives + critical_negatives)

        labels_total_jobs = _to_int((label_scope or {}).get("labels_total_jobs"))
        labels_labeled_jobs = _to_int((label_scope or {}).get("labels_labeled_jobs"))
        labels_unknown_jobs = _to_int((label_scope or {}).get("labels_unknown_jobs"))

        if labels_total_jobs is None:
            labels_total_jobs = int(len(critical_labels))
        if labels_labeled_jobs is None:
            labels_labeled_jobs = int(len(critical_labels))
        if labels_unknown_jobs is None:
            labels_unknown_jobs = int(max(0, labels_total_jobs - labels_labeled_jobs))

        coverage_rate = None
        if labels_labeled_jobs > 0:
            coverage_rate = round(float(completed_labeled_jobs / labels_labeled_jobs), 6)

        metrics_out["critical_miss_rate"] = miss_rate
        metrics_out["critical_false_positive_rate"] = fp_rate
        metrics_out["critical_confusion"] = confusion
        metrics_out["critical_positives"] = critical_positives
        metrics_out["critical_negatives"] = critical_negatives
        metrics_out["completed_labeled_jobs"] = completed_labeled_jobs
        metrics_out["labels_total_jobs"] = labels_total_jobs
        metrics_out["labels_labeled_jobs"] = labels_labeled_jobs
        metrics_out["labels_unknown_jobs"] = labels_unknown_jobs
        metrics_out["coverage_rate"] = coverage_rate
        metrics_out["critical_confidence"] = {
            "miss_rate": _rate_confidence(
                point=miss_rate,
                positives=int(confusion["fn"]),
                total=critical_positives,
            ),
            "false_positive_rate": _rate_confidence(
                point=fp_rate,
                positives=int(confusion["fp"]),
                total=critical_negatives,
            ),
        }
    else:
        metrics_out["critical_miss_rate"] = None
        metrics_out["critical_false_positive_rate"] = None
        metrics_out["critical_confusion"] = None
        metrics_out["critical_positives"] = None
        metrics_out["critical_negatives"] = None
        metrics_out["completed_labeled_jobs"] = None
        metrics_out["labels_total_jobs"] = _to_int((label_scope or {}).get("labels_total_jobs"))
        metrics_out["labels_labeled_jobs"] = _to_int((label_scope or {}).get("labels_labeled_jobs"))
        metrics_out["labels_unknown_jobs"] = _to_int((label_scope or {}).get("labels_unknown_jobs"))
        metrics_out["coverage_rate"] = None
        metrics_out["critical_confidence"] = None

    return metrics_out


def compute_critical_threshold_sweep(
    completed_jobs: list[dict[str, Any]],
    labels: dict[int, bool],
    thresholds: Iterable[float],
    *,
    scoring_mode: str = "continuous_v1",
    critical_policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate miss/FPR across thresholds for a fixed scoring mode."""
    mode = normalize_critical_scoring_mode(scoring_mode)
    if mode != "continuous_v1":
        raise ValueError("critical threshold sweep requires scoring_mode='continuous_v1'")
    rows: list[dict[str, Any]] = []
    for threshold_value in sorted({_normalize_threshold(v) for v in thresholds}):
        miss_rate, fp_rate, confusion = _critical_rates(
            completed_jobs,
            labels,
            scoring_mode=mode,
            critical_threshold=threshold_value,
            critical_policy=critical_policy,
        )
        rows.append(
            {
                "threshold": threshold_value,
                "critical_miss_rate": miss_rate,
                "critical_false_positive_rate": fp_rate,
                "critical_confusion": confusion,
                "critical_positives": int(confusion["tp"] + confusion["fn"]),
                "critical_negatives": int(confusion["fp"] + confusion["tn"]),
            }
        )
    return rows


def recommend_threshold_from_sweep(
    sweep_rows: list[dict[str, Any]],
    *,
    max_miss_rate: float | None,
    max_false_positive_rate: float | None,
) -> dict[str, Any] | None:
    if not sweep_rows:
        return None

    def _fits(row: dict[str, Any]) -> bool:
        miss = _to_float(row.get("critical_miss_rate"))
        fpr = _to_float(row.get("critical_false_positive_rate"))
        if miss is None or fpr is None:
            return False
        if max_miss_rate is not None and miss > max_miss_rate:
            return False
        return not (max_false_positive_rate is not None and fpr > max_false_positive_rate)

    def _sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
        miss = _to_float(row.get("critical_miss_rate"))
        fpr = _to_float(row.get("critical_false_positive_rate"))
        threshold = _to_float(row.get("threshold"))
        return (
            float(fpr if fpr is not None else 1.0),
            float(miss if miss is not None else 1.0),
            -float(threshold if threshold is not None else 0.0),
        )

    admissible = [row for row in sweep_rows if _fits(row)]
    if admissible:
        best = sorted(admissible, key=_sort_key)[0]
        return {
            "threshold": best.get("threshold"),
            "critical_miss_rate": best.get("critical_miss_rate"),
            "critical_false_positive_rate": best.get("critical_false_positive_rate"),
            "reason": "meets_constraints_min_fpr",
        }

    rows_with_rates = [
        row
        for row in sweep_rows
        if _to_float(row.get("critical_miss_rate")) is not None and _to_float(row.get("critical_false_positive_rate")) is not None
    ]
    if not rows_with_rates:
        return None

    def _violation(row: dict[str, Any]) -> float:
        miss = float(_to_float(row.get("critical_miss_rate")) or 0.0)
        fpr = float(_to_float(row.get("critical_false_positive_rate")) or 0.0)
        miss_excess = max(0.0, miss - float(max_miss_rate)) if max_miss_rate is not None else 0.0
        fpr_excess = max(0.0, fpr - float(max_false_positive_rate)) if max_false_positive_rate is not None else 0.0
        return miss_excess + fpr_excess

    best = sorted(rows_with_rates, key=lambda row: (_violation(row), _sort_key(row)))[0]
    return {
        "threshold": best.get("threshold"),
        "critical_miss_rate": best.get("critical_miss_rate"),
        "critical_false_positive_rate": best.get("critical_false_positive_rate"),
        "reason": "minimum_constraint_violation",
    }


def _severity_counts(job_score: dict[str, Any]) -> dict[str, int]:
    summary = job_score.get("summary", {}) or {}
    raw = summary.get("severity_counts")
    counts: dict[str, int] = {}
    if isinstance(raw, dict):
        for key in ("critical", "quality", "efficiency"):
            counts[key] = _non_negative_int(raw.get(key))
    else:
        counts = {"critical": 0, "quality": 0, "efficiency": 0}

    if sum(counts.values()) > 0:
        return counts

    # Fallback for partial score payloads that do not include summary.severity_counts
    deviations = job_score.get("deviations", []) or []
    for dev in deviations:
        severity = str(dev.get("severity", "quality")).strip().lower()
        if severity not in counts:
            counts[severity] = 0
        counts[severity] += 1
    for key in ("critical", "quality", "efficiency"):
        counts.setdefault(key, 0)
    return counts


def _critical_missing_stats(job_score: dict[str, Any]) -> dict[str, float]:
    deviations = job_score.get("deviations", []) or []
    critical_missing = [
        dev
        for dev in deviations
        if str(dev.get("severity", "")).strip().lower() == "critical"
        and str(dev.get("type", "")).strip().lower() == "missing_step"
    ]
    mean_distances: list[float] = []
    expected_spans: list[float] = []
    for dev in critical_missing:
        mean_distances.append(_non_negative_float(dev.get("mean_distance")))
        expected_spans.append(_non_negative_float(dev.get("expected_span_len")))

    def _avg(items: list[float]) -> float:
        if not items:
            return 0.0
        return float(sum(items) / len(items))

    return {
        "num_critical_missing": float(len(critical_missing)),
        "mean_distance_avg": _avg(mean_distances),
        "expected_span_avg": _avg(expected_spans),
    }


def _normalize_threshold(value: float | None) -> float:
    if value is None:
        return float(DEFAULT_CRITICAL_THRESHOLD)
    try:
        threshold = float(value)
    except Exception:
        threshold = float(DEFAULT_CRITICAL_THRESHOLD)
    if threshold < 0.0:
        return 0.0
    if threshold > 1.0:
        return 1.0
    return round(float(threshold), 6)


def _non_negative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except Exception:
        return 0
    return int(parsed) if parsed > 0 else 0


def _non_negative_float(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return 0.0
    return float(parsed) if parsed > 0 else 0.0


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _rate_confidence(
    *,
    point: float | None,
    positives: int,
    total: int,
) -> dict[str, Any]:
    ci95 = _binomial_wilson_ci95(positives=positives, total=total)
    return {
        "n": int(total),
        "point": point,
        "ci95": ci95,
    }


def _binomial_wilson_ci95(*, positives: int, total: int) -> dict[str, float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    n = float(total)
    phat = float(positives) / n
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = (phat + (z2 / (2.0 * n))) / denom
    margin = (z / denom) * np.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))
    low = max(0.0, center - float(margin))
    high = min(1.0, center + float(margin))
    return {
        "low": round(float(low), 6),
        "high": round(float(high), 6),
    }


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def _critical_rates(
    completed_jobs: list[dict[str, Any]],
    labels: dict[int, bool],
    *,
    scoring_mode: str,
    critical_threshold: float,
    critical_policy: dict[str, Any] | None = None,
) -> tuple[float | None, float | None, dict]:
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for job in completed_jobs:
        job_id = int(job["id"])
        if job_id not in labels:
            continue
        expected = labels[job_id]
        detected = job_detected_critical(
            job["score"],
            scoring_mode=scoring_mode,
            critical_threshold=critical_threshold,
            critical_policy=critical_policy,
        )
        if expected and detected:
            tp += 1
        elif expected and not detected:
            fn += 1
        elif not expected and detected:
            fp += 1
        else:
            tn += 1

    miss_rate = (fn / (tp + fn)) if (tp + fn) > 0 else None
    fp_rate = (fp / (fp + tn)) if (fp + tn) > 0 else None
    confusion = {"tp": tp, "fn": fn, "fp": fp, "tn": tn}
    if miss_rate is not None:
        miss_rate = round(float(miss_rate), 6)
    if fp_rate is not None:
        fp_rate = round(float(fp_rate), 6)
    return miss_rate, fp_rate, confusion


def _rescore_jitter(completed_jobs: list[dict[str, Any]]) -> dict[str, float | int | None]:
    grouped: dict[tuple[int, int], list[float]] = {}
    for job in completed_jobs:
        key = (int(job["gold_video_id"]), int(job["trainee_video_id"]))
        grouped.setdefault(key, []).append(float(job["score"].get("score", 0.0)))

    jitter_values: list[float] = []
    for scores in grouped.values():
        if len(scores) < 2:
            continue
        jitter_values.append(float(max(scores) - min(scores)))

    if not jitter_values:
        return {
            "num_pairs_with_repeats": 0,
            "mean_delta": None,
            "p90_delta": None,
            "max_delta": None,
        }
    arr = np.asarray(jitter_values, dtype=np.float64)
    return {
        "num_pairs_with_repeats": int(len(jitter_values)),
        "mean_delta": round(float(np.mean(arr)), 6),
        "p90_delta": round(float(np.percentile(arr, 90)), 6),
        "max_delta": round(float(np.max(arr)), 6),
    }


def _resolve_guardrail_non_negative(
    critical_policy: dict[str, Any] | None,
    *,
    mode: str,
    key: str,
    default: float,
) -> float:
    if not isinstance(critical_policy, dict):
        return float(default)
    guardrails = critical_policy.get("guardrails", {})
    if not isinstance(guardrails, dict):
        return float(default)

    mode_payload = guardrails.get(mode, {})
    if not isinstance(mode_payload, dict):
        mode_payload = {}

    fallback_mode_payload = guardrails.get("guarded_binary_v1", {})
    if not isinstance(fallback_mode_payload, dict):
        fallback_mode_payload = {}

    value = mode_payload.get(key)
    if value is None and key == "min_dtw":
        value = fallback_mode_payload.get("min_dtw")
    parsed = _to_float(value)
    if parsed is None or parsed < 0.0:
        return float(default)
    return float(parsed)


def _critical_policy_summary(critical_policy: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(critical_policy, dict):
        return None
    out: dict[str, Any] = {}
    for key in ("policy_id", "version", "scoring_mode", "critical_threshold", "created_at"):
        value = critical_policy.get(key)
        if value is None:
            continue
        out[key] = value
    guardrails = critical_policy.get("guardrails")
    if isinstance(guardrails, dict):
        out["guardrails"] = guardrails
    return out or {}


def _compute_drift_summary(
    *,
    scores: list[float],
    critical_scores: list[float],
    dtw_values: list[float],
    detected_flags: list[float],
) -> dict[str, Any]:
    windows = _split_windows(max(len(scores), len(critical_scores), len(detected_flags)))
    out: dict[str, Any] = {
        "reference_jobs": windows["reference_jobs"],
        "current_jobs": windows["current_jobs"],
        "min_window_jobs_required": int(_DRIFT_MIN_WINDOW),
    }
    if not windows["enabled"]:
        out["enabled"] = False
        out["reason"] = "insufficient_jobs"
        out["critical_score_psi"] = None
        out["score_psi"] = None
        out["dtw_normalized_cost_psi"] = None
        out["critical_detected_rate_reference"] = None
        out["critical_detected_rate_current"] = None
        out["critical_detected_rate_shift_abs"] = None
        return out

    out["enabled"] = True
    out["reason"] = "ok"
    r_start = 0
    r_end = int(windows["reference_jobs"])
    c_start = r_end
    c_end = r_end + int(windows["current_jobs"])

    out["critical_score_psi"] = _population_stability_index(
        critical_scores[r_start:r_end],
        critical_scores[c_start:c_end],
    )
    out["score_psi"] = _population_stability_index(
        scores[r_start:r_end],
        scores[c_start:c_end],
    )
    if len(dtw_values) >= (r_end + int(windows["current_jobs"])):
        out["dtw_normalized_cost_psi"] = _population_stability_index(
            dtw_values[r_start:r_end],
            dtw_values[c_start:c_end],
        )
    else:
        out["dtw_normalized_cost_psi"] = None

    ref_rate = _safe_mean(detected_flags[r_start:r_end])
    cur_rate = _safe_mean(detected_flags[c_start:c_end])
    out["critical_detected_rate_reference"] = None if ref_rate is None else round(float(ref_rate), 6)
    out["critical_detected_rate_current"] = None if cur_rate is None else round(float(cur_rate), 6)
    if ref_rate is None or cur_rate is None:
        out["critical_detected_rate_shift_abs"] = None
    else:
        out["critical_detected_rate_shift_abs"] = round(abs(float(cur_rate) - float(ref_rate)), 6)
    return out


def _split_windows(total_jobs: int) -> dict[str, Any]:
    n = int(max(0, total_jobs))
    half = n // 2
    if half < _DRIFT_MIN_WINDOW or (n - half) < _DRIFT_MIN_WINDOW:
        return {
            "enabled": False,
            "reference_jobs": int(half),
            "current_jobs": int(max(0, n - half)),
        }
    return {
        "enabled": True,
        "reference_jobs": int(half),
        "current_jobs": int(n - half),
    }


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _population_stability_index(reference: list[float], current: list[float], *, bins: int = 10) -> float | None:
    if not reference or not current:
        return None
    ref_arr = np.asarray(reference, dtype=np.float64)
    cur_arr = np.asarray(current, dtype=np.float64)
    if ref_arr.size == 0 or cur_arr.size == 0:
        return None

    quantiles = np.linspace(0.0, 1.0, num=max(2, int(bins) + 1))
    edges = np.quantile(ref_arr, quantiles)
    edges = np.unique(edges)
    if edges.size < 3:
        low = float(min(np.min(ref_arr), np.min(cur_arr)))
        high = float(max(np.max(ref_arr), np.max(cur_arr)))
        if abs(high - low) < 1e-12:
            return 0.0
        edges = np.linspace(low, high, num=max(3, int(bins) + 1))

    ref_hist, _ = np.histogram(ref_arr, bins=edges)
    cur_hist, _ = np.histogram(cur_arr, bins=edges)
    eps = 1e-8
    ref_pct = np.maximum(ref_hist.astype(np.float64) / max(1.0, float(ref_arr.size)), eps)
    cur_pct = np.maximum(cur_hist.astype(np.float64) / max(1.0, float(cur_arr.size)), eps)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(max(0.0, psi)), 6)
