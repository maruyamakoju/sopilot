from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sopilot.eval.gates import get_gate_profile

DEFAULT_BACKENDS = ("color-motion", "vjepa2")
ALLOWED_BACKENDS = frozenset(DEFAULT_BACKENDS)
DEFAULT_BASE_SEED = 1729
DEFAULT_TASK_NAME = "PoC Task"
DEFAULT_SITE_ID = "site-a"
DEFAULT_GOLD_ID = 2
DEFAULT_MAX_SOURCE = 6
DEFAULT_VJEPA2_POOLING = "mean_tokens"


@dataclass(frozen=True)
class RunDefaults:
    task_name: str
    site_id: str
    gold_id: int
    max_source: int
    base_seed: int
    vjepa2_pooling: str


@dataclass(frozen=True)
class ReportContext:
    task_id: str
    task_name: str
    base_dir: str | None
    trainee_dir: str | None
    trainee_bad_dir: str | None
    site_id: str
    gold_id: int
    max_source: int


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _safe_stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def _bootstrap_mean_ci95(values: list[float], *, samples: int = 2000, seed: int = 42) -> dict[str, float] | None:
    if not values:
        return None
    if len(values) == 1:
        value = float(values[0])
        return {"low": value, "high": value}

    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(float(statistics.fmean(draw)))
    means.sort()
    low_idx = max(0, int(0.025 * (samples - 1)))
    high_idx = min(samples - 1, int(0.975 * (samples - 1)))
    return {"low": float(means[low_idx]), "high": float(means[high_idx])}


def _rate(predicate_values: list[bool]) -> float | None:
    if not predicate_values:
        return None
    return float(sum(1 for item in predicate_values if item) / len(predicate_values))


def _sign_test_pvalue_two_sided(values: list[float]) -> float | None:
    filtered = [value for value in values if value != 0.0]
    n = len(filtered)
    if n == 0:
        return None
    positive = sum(1 for value in filtered if value > 0.0)
    denom = 2 ** n
    cdf = sum(math.comb(n, idx) for idx in range(0, positive + 1)) / denom
    sf = sum(math.comb(n, idx) for idx in range(positive, n + 1)) / denom
    return float(min(1.0, 2.0 * min(cdf, sf)))


def _tail(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _extract_gate_metrics(gate_payload: dict[str, Any]) -> dict[str, Any]:
    gates = gate_payload.get("gates") if isinstance(gate_payload.get("gates"), dict) else {}
    jitter = gate_payload.get("rescore_jitter") if isinstance(gate_payload.get("rescore_jitter"), dict) else {}
    dtw = (
        gate_payload.get("dtw_normalized_cost_stats")
        if isinstance(gate_payload.get("dtw_normalized_cost_stats"), dict)
        else {}
    )
    return {
        "overall_pass": gates.get("overall_pass"),
        "critical_miss_rate": gate_payload.get("critical_miss_rate"),
        "critical_false_positive_rate": gate_payload.get("critical_false_positive_rate"),
        "dtw_p90": dtw.get("p90"),
        "rescore_jitter_max_delta": jitter.get("max_delta"),
        "num_completed_jobs": gate_payload.get("num_completed_jobs"),
    }


def _extract_embedder_runtime(summary_payload: dict[str, Any]) -> dict[str, Any]:
    value = summary_payload.get("embedder_runtime")
    if isinstance(value, dict):
        return value
    return {}


def _extract_label_scope(labels_payload: dict[str, Any]) -> dict[str, int]:
    jobs = labels_payload.get("jobs")
    if not isinstance(jobs, list):
        return {
            "labels_total_jobs": 0,
            "labels_labeled_jobs": 0,
            "labels_unknown_jobs": 0,
        }
    total = len(jobs)
    unknown = 0
    for row in jobs:
        if not isinstance(row, dict):
            continue
        if row.get("critical_expected") is None:
            unknown += 1
    labeled = max(0, total - unknown)
    return {
        "labels_total_jobs": int(total),
        "labels_labeled_jobs": int(labeled),
        "labels_unknown_jobs": int(unknown),
    }


def _extract_coverage_and_confusion(
    *,
    gate_payload: dict[str, Any],
    local_summary: dict[str, Any],
    label_scope: dict[str, int],
) -> dict[str, Any]:
    expected_raw = label_scope.get("labels_total_jobs")
    if expected_raw is None:
        expected_raw = local_summary.get("scored_total")
    expected = _to_int(expected_raw)

    confusion_raw = gate_payload.get("critical_confusion")
    confusion = confusion_raw if isinstance(confusion_raw, dict) else {}
    tp = _to_int(confusion.get("tp"))
    fn = _to_int(confusion.get("fn"))
    fp = _to_int(confusion.get("fp"))
    tn = _to_int(confusion.get("tn"))

    completed_from_confusion: int | None = None
    if any(value is not None for value in (tp, fn, fp, tn)):
        completed_from_confusion = int((tp or 0) + (fn or 0) + (fp or 0) + (tn or 0))

    completed = completed_from_confusion
    if completed is None:
        completed_fallback_raw = local_summary.get("scored_completed")
        if completed_fallback_raw is None:
            completed_fallback_raw = label_scope.get("labels_labeled_jobs")
        if completed_fallback_raw is None:
            completed_fallback_raw = gate_payload.get("num_completed_jobs")
        completed = _to_int(completed_fallback_raw)

    if expected is None:
        expected = completed

    completed_score_jobs_total_raw = gate_payload.get("num_completed_jobs")
    completed_score_jobs_total = _to_int(completed_score_jobs_total_raw)
    coverage: float | None = None
    if expected is not None and expected > 0 and completed is not None:
        coverage = float(completed / expected)

    negatives = None
    if fp is not None or tn is not None:
        negatives = int((fp or 0) + (tn or 0))
    positives = None
    if tp is not None or fn is not None:
        positives = int((tp or 0) + (fn or 0))
    return {
        "expected_job_count": expected,
        "completed_job_count": completed,
        "completed_score_jobs_total": completed_score_jobs_total,
        "coverage_rate": coverage,
        "critical_fp_count": fp,
        "critical_tn_count": tn,
        "critical_tp_count": tp,
        "critical_fn_count": fn,
        "critical_negatives": negatives,
        "critical_positives": positives,
        **label_scope,
    }


def _has_gate_metrics(row: dict[str, Any]) -> bool:
    gate = row.get("gate_metrics")
    return isinstance(gate, dict) and bool(gate)


def _metric_values(rows: list[dict[str, Any]], metric_key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        gate = row.get("gate_metrics")
        if not isinstance(gate, dict):
            continue
        value = gate.get(metric_key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except Exception:
            continue
    return values


def _row_float_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            out.append(float(value))
        except Exception:
            continue
    return out


def _row_int_values(rows: list[dict[str, Any]], key: str) -> list[int]:
    out: list[int] = []
    for row in rows:
        value = _to_int(row.get(key))
        if value is not None:
            out.append(int(value))
    return out


def _metric_delta(left_row: dict[str, Any], right_row: dict[str, Any], metric_key: str) -> float | None:
    left_gate = left_row.get("gate_metrics")
    right_gate = right_row.get("gate_metrics")
    if not isinstance(left_gate, dict) or not isinstance(right_gate, dict):
        return None
    left_value = left_gate.get(metric_key)
    right_value = right_gate.get(metric_key)
    if left_value is None or right_value is None:
        return None
    try:
        return float(right_value) - float(left_value)
    except Exception:
        return None


def _row_delta(left_row: dict[str, Any], right_row: dict[str, Any], key: str) -> float | None:
    left_value = left_row.get(key)
    right_value = right_row.get(key)
    if left_value is None or right_value is None:
        return None
    try:
        return float(right_value) - float(left_value)
    except Exception:
        return None


def _summarize_backend_rows(rows: list[dict[str, Any]], gate_thresholds: dict[str, float]) -> dict[str, Any]:
    metrics_rows = [row for row in rows if _has_gate_metrics(row)]
    strict_metrics_rows = [row for row in metrics_rows if not bool(row.get("contaminated_by_fallback"))]
    pass_rows = [row for row in metrics_rows if row["gate_metrics"].get("overall_pass") is True]
    strict_pass_rows = [row for row in strict_metrics_rows if row["gate_metrics"].get("overall_pass") is True]
    fpr_values = _metric_values(metrics_rows, "critical_false_positive_rate")
    strict_fpr_values = _metric_values(strict_metrics_rows, "critical_false_positive_rate")
    miss_values = _metric_values(metrics_rows, "critical_miss_rate")
    dtw_values = _metric_values(metrics_rows, "dtw_p90")
    strict_dtw_values = _metric_values(strict_metrics_rows, "dtw_p90")
    duration_values = _row_float_values(rows, "duration_sec")
    strict_duration_values = _row_float_values(strict_metrics_rows, "duration_sec")
    expected_values = _row_int_values(rows, "expected_job_count")
    completed_values = _row_int_values(rows, "completed_job_count")
    completed_score_job_values = _row_int_values(rows, "completed_score_jobs_total")
    coverage_values = _row_float_values(rows, "coverage_rate")
    critical_negative_values = _row_int_values(rows, "critical_negatives")
    fp_values = _row_int_values(rows, "critical_fp_count")
    tn_values = _row_int_values(rows, "critical_tn_count")
    unknown_label_values = _row_int_values(rows, "labels_unknown_jobs")
    contaminated_runs = sum(1 for row in rows if bool(row.get("contaminated_by_fallback")))
    return {
        "runs": len(rows),
        "runs_with_metrics": len(metrics_rows),
        "strict_runs_with_metrics": len(strict_metrics_rows),
        "pass_runs": len(pass_rows),
        "strict_pass_runs": len(strict_pass_rows),
        "nonzero_return_runs": sum(1 for row in rows if row.get("returncode") not in (0, None)),
        "pass_rate": (len(pass_rows) / len(metrics_rows)) if metrics_rows else None,
        "strict_pass_rate": (len(strict_pass_rows) / len(strict_metrics_rows)) if strict_metrics_rows else None,
        "contaminated_runs": contaminated_runs,
        "contamination_rate": (float(contaminated_runs) / float(len(rows))) if rows else None,
        "mean_duration_sec": _safe_mean(duration_values),
        "std_duration_sec": _safe_stdev(duration_values),
        "duration_ci95": _bootstrap_mean_ci95(duration_values),
        "strict_mean_duration_sec": _safe_mean(strict_duration_values),
        "mean_critical_false_positive_rate": _safe_mean(fpr_values),
        "std_critical_false_positive_rate": _safe_stdev(fpr_values),
        "critical_false_positive_rate_ci95": _bootstrap_mean_ci95(fpr_values),
        "strict_mean_critical_false_positive_rate": _safe_mean(strict_fpr_values),
        "mean_critical_miss_rate": _safe_mean(miss_values),
        "mean_dtw_p90": _safe_mean(dtw_values),
        "std_dtw_p90": _safe_stdev(dtw_values),
        "dtw_p90_ci95": _bootstrap_mean_ci95(dtw_values),
        "strict_mean_dtw_p90": _safe_mean(strict_dtw_values),
        "critical_fpr_over_threshold_rate": _rate(
            [value > gate_thresholds["critical_false_positive_rate"] for value in fpr_values]
        ),
        "dtw_p90_over_threshold_rate": _rate([value > gate_thresholds["dtw_p90"] for value in dtw_values]),
        "strict_critical_fpr_over_threshold_rate": _rate(
            [value > gate_thresholds["critical_false_positive_rate"] for value in strict_fpr_values]
        ),
        "strict_dtw_p90_over_threshold_rate": _rate([value > gate_thresholds["dtw_p90"] for value in strict_dtw_values]),
        "mean_expected_job_count": _safe_mean([float(v) for v in expected_values]),
        "mean_completed_job_count": _safe_mean([float(v) for v in completed_values]),
        "mean_completed_score_jobs_total": _safe_mean([float(v) for v in completed_score_job_values]),
        "mean_coverage_rate": _safe_mean(coverage_values),
        "min_coverage_rate": float(min(coverage_values)) if coverage_values else None,
        "coverage_lt_100_runs": sum(1 for value in coverage_values if value < 1.0),
        "mean_critical_negatives": _safe_mean([float(v) for v in critical_negative_values]),
        "mean_critical_fp_count": _safe_mean([float(v) for v in fp_values]),
        "mean_critical_tn_count": _safe_mean([float(v) for v in tn_values]),
        "mean_unknown_label_count": _safe_mean([float(v) for v in unknown_label_values]),
    }


def _summarize_pairwise_two_backends(runs: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    left_rows: dict[int, dict[str, Any]] = {}
    right_rows: dict[int, dict[str, Any]] = {}
    for row in runs:
        repeat = _to_int(row.get("repeat"))
        if repeat is None:
            continue
        if row.get("backend") == left:
            left_rows[int(repeat)] = row
        elif row.get("backend") == right:
            right_rows[int(repeat)] = row

    repeats = sorted(set(left_rows.keys()) & set(right_rows.keys()))
    delta_fpr: list[float] = []
    delta_dtw: list[float] = []
    delta_duration: list[float] = []
    strict_repeats: list[int] = []
    strict_delta_fpr: list[float] = []
    strict_delta_dtw: list[float] = []
    strict_delta_duration: list[float] = []
    expected_match_flags: list[bool] = []
    delta_coverage: list[float] = []

    for rep in repeats:
        left_row = left_rows[rep]
        right_row = right_rows[rep]
        fpr_delta = _metric_delta(left_row, right_row, "critical_false_positive_rate")
        if fpr_delta is not None:
            delta_fpr.append(fpr_delta)
        dtw_delta = _metric_delta(left_row, right_row, "dtw_p90")
        if dtw_delta is not None:
            delta_dtw.append(dtw_delta)
        duration_delta = _row_delta(left_row, right_row, "duration_sec")
        if duration_delta is not None:
            delta_duration.append(duration_delta)

        left_expected = _to_int(left_row.get("expected_job_count"))
        right_expected = _to_int(right_row.get("expected_job_count"))
        if left_expected is not None and right_expected is not None:
            expected_match_flags.append(int(left_expected) == int(right_expected))

        coverage_delta = _row_delta(left_row, right_row, "coverage_rate")
        if coverage_delta is not None:
            delta_coverage.append(coverage_delta)

        if bool(left_row.get("contaminated_by_fallback")) or bool(right_row.get("contaminated_by_fallback")):
            continue

        strict_repeats.append(rep)
        if fpr_delta is not None:
            strict_delta_fpr.append(fpr_delta)
        if dtw_delta is not None:
            strict_delta_dtw.append(dtw_delta)
        if duration_delta is not None:
            strict_delta_duration.append(duration_delta)

    return {
        "left_backend": left,
        "right_backend": right,
        "paired_repeats": repeats,
        "mean_delta_fpr_right_minus_left": _safe_mean(delta_fpr),
        "delta_fpr_ci95": _bootstrap_mean_ci95(delta_fpr),
        "right_wins_fpr_rate": _rate([value < 0.0 for value in delta_fpr]),
        "sign_test_pvalue_fpr": _sign_test_pvalue_two_sided(delta_fpr),
        "mean_delta_dtw_p90_right_minus_left": _safe_mean(delta_dtw),
        "delta_dtw_p90_ci95": _bootstrap_mean_ci95(delta_dtw),
        "right_wins_dtw_rate": _rate([value < 0.0 for value in delta_dtw]),
        "sign_test_pvalue_dtw_p90": _sign_test_pvalue_two_sided(delta_dtw),
        "mean_delta_duration_sec_right_minus_left": _safe_mean(delta_duration),
        "delta_duration_sec_ci95": _bootstrap_mean_ci95(delta_duration),
        "right_wins_duration_rate": _rate([value < 0.0 for value in delta_duration]),
        "sign_test_pvalue_duration_sec": _sign_test_pvalue_two_sided(delta_duration),
        "strict_paired_repeats": strict_repeats,
        "strict_mean_delta_fpr_right_minus_left": _safe_mean(strict_delta_fpr),
        "strict_mean_delta_dtw_p90_right_minus_left": _safe_mean(strict_delta_dtw),
        "strict_mean_delta_duration_sec_right_minus_left": _safe_mean(strict_delta_duration),
        "expected_job_count_match_rate": _rate(expected_match_flags),
        "mean_delta_coverage_rate_right_minus_left": _safe_mean(delta_coverage),
    }


def summarize_runs(runs: list[dict[str, Any]], backends: list[str]) -> dict[str, Any]:
    legacy_gate = get_gate_profile("legacy_poc")
    gate_thresholds = {
        "critical_false_positive_rate": float(legacy_gate.max_critical_false_positive_rate or 0.30),
        "dtw_p90": float(legacy_gate.max_dtw_p90 or 0.60),
    }
    by_backend: dict[str, Any] = {}
    for backend in backends:
        rows = [row for row in runs if row.get("backend") == backend]
        by_backend[backend] = _summarize_backend_rows(rows, gate_thresholds)

    pairwise: list[dict[str, Any]] = []
    if len(backends) == 2:
        pairwise.append(_summarize_pairwise_two_backends(runs, backends[0], backends[1]))

    return {"by_backend": by_backend, "pairwise": pairwise}


def build_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Backend Ablation Report")
    lines.append("")
    lines.append(f"- generated_at_utc: `{report.get('generated_at_utc')}`")
    lines.append(f"- task_id: `{report.get('task_id')}`")
    lines.append(f"- run_count: `{len(report.get('runs', []))}`")
    lines.append("")
    lines.append("## Backend Summary")
    lines.append("")
    lines.append(
        "| backend | runs | failed_runs | runs_with_metrics | pass_runs | pass_rate | contaminated_runs | contamination_rate | mean_duration_sec ± std | mean_fpr ± std | mean_miss | mean_dtw_p90 ± std |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    summary = report.get("summary", {}).get("by_backend", {})
    for backend in report.get("backends", []):
        row = summary.get(backend, {})
        lines.append(
            "| "
            f"{backend} | {row.get('runs')} | {row.get('nonzero_return_runs')} | "
            f"{row.get('runs_with_metrics')} | {row.get('pass_runs')} | "
            f"{_fmt_float(row.get('pass_rate'))} | {_fmt_int(row.get('contaminated_runs'))} | {_fmt_float(row.get('contamination_rate'))} | "
            f"{_fmt_mean_std(row.get('mean_duration_sec'), row.get('std_duration_sec'))} | "
            f"{_fmt_mean_std(row.get('mean_critical_false_positive_rate'), row.get('std_critical_false_positive_rate'))} | "
            f"{_fmt_float(row.get('mean_critical_miss_rate'))} | {_fmt_mean_std(row.get('mean_dtw_p90'), row.get('std_dtw_p90'))} |"
        )

    lines.append("")
    lines.append("## Coverage and Label Scope")
    lines.append("")
    lines.append(
        "| backend | mean_expected_jobs | mean_completed_jobs | mean_score_jobs_total | mean_coverage_rate | min_coverage_rate | coverage_lt_100_runs | mean_critical_negatives | mean_fp | mean_tn | mean_unknown_labels |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for backend in report.get("backends", []):
        row = summary.get(backend, {})
        lines.append(
            "| "
            f"{backend} | {_fmt_float(row.get('mean_expected_job_count'))} | {_fmt_float(row.get('mean_completed_job_count'))} | "
            f"{_fmt_float(row.get('mean_completed_score_jobs_total'))} | {_fmt_float(row.get('mean_coverage_rate'))} | {_fmt_float(row.get('min_coverage_rate'))} | "
            f"{_fmt_int(row.get('coverage_lt_100_runs'))} | {_fmt_float(row.get('mean_critical_negatives'))} | "
            f"{_fmt_float(row.get('mean_critical_fp_count'))} | {_fmt_float(row.get('mean_critical_tn_count'))} | "
            f"{_fmt_float(row.get('mean_unknown_label_count'))} |"
        )

    lines.append("")
    lines.append("## Strict Summary (fallback-clean runs only)")
    lines.append("")
    lines.append(
        "| backend | strict_runs_with_metrics | strict_pass_runs | strict_pass_rate | strict_mean_duration_sec | strict_mean_fpr | strict_mean_dtw_p90 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for backend in report.get("backends", []):
        row = summary.get(backend, {})
        lines.append(
            "| "
            f"{backend} | {row.get('strict_runs_with_metrics')} | {row.get('strict_pass_runs')} | "
            f"{_fmt_float(row.get('strict_pass_rate'))} | {_fmt_float(row.get('strict_mean_duration_sec'))} | "
            f"{_fmt_float(row.get('strict_mean_critical_false_positive_rate'))} | {_fmt_float(row.get('strict_mean_dtw_p90'))} |"
        )

    pairwise = report.get("summary", {}).get("pairwise", [])
    if pairwise:
        lines.append("")
        lines.append("## Pairwise Delta")
        lines.append("")
        lines.append(
            "| left_backend | right_backend | paired_repeats | strict_paired_repeats | delta_fpr (right-left) | delta_fpr_ci95 | right_wins_fpr_rate | sign_p_fpr | strict_delta_fpr | delta_dtw_p90 | delta_dtw_ci95 | right_wins_dtw_rate | sign_p_dtw | strict_delta_dtw | delta_duration_sec | delta_duration_ci95 | right_wins_duration_rate | sign_p_duration | strict_delta_duration |"
        )
        lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|")
        for row in pairwise:
            lines.append(
                "| "
                f"{row.get('left_backend')} | {row.get('right_backend')} | "
                f"{len(row.get('paired_repeats') or [])} | "
                f"{len(row.get('strict_paired_repeats') or [])} | "
                f"{_fmt_float(row.get('mean_delta_fpr_right_minus_left'))} | "
                f"{_fmt_ci(row.get('delta_fpr_ci95'))} | "
                f"{_fmt_float(row.get('right_wins_fpr_rate'))} | "
                f"{_fmt_float(row.get('sign_test_pvalue_fpr'))} | "
                f"{_fmt_float(row.get('strict_mean_delta_fpr_right_minus_left'))} | "
                f"{_fmt_float(row.get('mean_delta_dtw_p90_right_minus_left'))} | "
                f"{_fmt_ci(row.get('delta_dtw_p90_ci95'))} | "
                f"{_fmt_float(row.get('right_wins_dtw_rate'))} | "
                f"{_fmt_float(row.get('sign_test_pvalue_dtw_p90'))} | "
                f"{_fmt_float(row.get('strict_mean_delta_dtw_p90_right_minus_left'))} | "
                f"{_fmt_float(row.get('mean_delta_duration_sec_right_minus_left'))} | "
                f"{_fmt_ci(row.get('delta_duration_sec_ci95'))} | "
                f"{_fmt_float(row.get('right_wins_duration_rate'))} | "
                f"{_fmt_float(row.get('sign_test_pvalue_duration_sec'))} | "
                f"{_fmt_float(row.get('strict_mean_delta_duration_sec_right_minus_left'))} |"
            )
        lines.append("")
        lines.append("| expected_job_match_rate | delta_coverage_rate (right-left) |")
        lines.append("|---:|---:|")
        for row in pairwise:
            lines.append(
                f"| {_fmt_float(row.get('expected_job_count_match_rate'))} | "
                f"{_fmt_float(row.get('mean_delta_coverage_rate_right_minus_left'))} |"
            )

    return "\n".join(lines) + "\n"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def _fmt_mean_std(mean_value: Any, std_value: Any) -> str:
    if mean_value is None:
        return "n/a"
    if std_value is None:
        return f"{_fmt_float(mean_value)} ± n/a"
    return f"{_fmt_float(mean_value)} ± {_fmt_float(std_value)}"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return str(int(value))
    except Exception:
        return str(value)


def _fmt_ci(value: Any) -> str:
    if not isinstance(value, dict):
        return "n/a"
    return f"[{_fmt_float(value.get('low'))}, {_fmt_float(value.get('high'))}]"


def _run_one(cmd: list[str], cwd: Path | None = None) -> tuple[int, float, str, str]:
    started = time.perf_counter()
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True, cwd=str(cwd) if cwd else None)
    duration = time.perf_counter() - started
    return int(proc.returncode), float(duration), proc.stdout or "", proc.stderr or ""


def _probe_runtime_info() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    try:
        import torch  # type: ignore

        payload["torch_version"] = getattr(torch, "__version__", None)
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            payload["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        payload["torch_probe_error"] = str(exc)
    return payload


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_repeat_from_name(name: str) -> int | None:
    match = re.search(r"(\d+)$", str(name))
    if not match:
        return None
    return _to_int(match.group(1))


def _duration_from_manifest(run_manifest: dict[str, Any]) -> float | None:
    started = run_manifest.get("started_at_utc")
    completed = run_manifest.get("completed_at_utc")
    if not started or not completed:
        return None
    try:
        started_dt = datetime.fromisoformat(str(started))
        completed_dt = datetime.fromisoformat(str(completed))
    except Exception:
        return None
    if started_dt.tzinfo is None:
        started_dt = started_dt.replace(tzinfo=UTC)
    if completed_dt.tzinfo is None:
        completed_dt = completed_dt.replace(tzinfo=UTC)
    delta = (completed_dt - started_dt).total_seconds()
    if delta < 0:
        return None
    return float(delta)


def _normalize_duration(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except Exception:
        return None


def _resolve_run_id(run_manifest: dict[str, Any], existing: dict[str, Any], backend: str, repeat: int) -> str:
    run_id = run_manifest.get("run_id") or existing.get("run_id")
    if run_id:
        return str(run_id)
    return f"summarize:{backend}:rep{repeat:02d}"


def _infer_returncode(run_manifest: dict[str, Any], preferred: Any) -> int | None:
    value = _to_int(preferred)
    if value is not None:
        return int(value)
    status = str(run_manifest.get("status", "")).lower()
    if status == "completed":
        return 0
    if status == "failed":
        return 1
    return None


def _is_fallback_contaminated(embedder_runtime: dict[str, Any]) -> bool:
    contaminated = bool(embedder_runtime.get("fallback_contaminated"))
    fallback_uses = _to_int(embedder_runtime.get("fallback_uses"))
    if fallback_uses is not None and fallback_uses > 0:
        contaminated = True
    return contaminated


def _build_run_row(
    *,
    backend: str,
    repeat: int,
    data_dir: Path,
    gate_payload: dict[str, Any],
    local_summary: dict[str, Any],
    labels_payload: dict[str, Any],
    run_manifest: dict[str, Any],
    run_id: str,
    seed: int | None,
    returncode: int | None,
    duration_sec: Any,
    stdout_tail: str,
    stderr_tail: str,
) -> dict[str, Any]:
    embedder_runtime = _extract_embedder_runtime(local_summary)
    label_scope = _extract_label_scope(labels_payload)
    coverage_scope = _extract_coverage_and_confusion(
        gate_payload=gate_payload,
        local_summary=local_summary,
        label_scope=label_scope,
    )
    return {
        "backend": backend,
        "repeat": int(repeat),
        "run_id": str(run_id),
        "seed": seed,
        "data_dir": str(data_dir),
        "returncode": _to_int(returncode),
        "duration_sec": _normalize_duration(duration_sec),
        "stdout_tail": str(stdout_tail),
        "stderr_tail": str(stderr_tail),
        "gate_metrics": _extract_gate_metrics(gate_payload),
        "embedder_runtime": embedder_runtime,
        "contaminated_by_fallback": _is_fallback_contaminated(embedder_runtime),
        **coverage_scope,
        "gate_report_path": str((data_dir / "gate_report.json").resolve()),
        "local_summary_path": str((data_dir / "local_pipeline_summary.json").resolve()),
        "labels_path": str((data_dir / "labels_template.json").resolve()),
        "run_manifest_path": str((data_dir / "run_manifest.json").resolve()),
        "run_manifest_status": run_manifest.get("status"),
    }


def _existing_runs_index(existing_report: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    rows = existing_report.get("runs")
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        backend = row.get("backend")
        repeat = _to_int(row.get("repeat"))
        if backend is None or repeat is None:
            continue
        out[(str(backend), int(repeat))] = row
    return out


def _discover_backends(root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        backends = [str(item).strip() for item in requested if str(item).strip()]
        return sorted(dict.fromkeys(backends))

    discovered: list[str] = []
    runs_dir = root / "runs"
    if runs_dir.exists():
        for path in sorted(runs_dir.iterdir(), key=lambda p: p.name):
            if path.is_dir():
                discovered.append(path.name)
    if discovered:
        return discovered
    return list(DEFAULT_BACKENDS)


def _build_run_row_from_data_dir(
    *,
    backend: str,
    repeat: int,
    data_dir: Path,
    existing_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = existing_row or {}
    gate_payload = _read_json(data_dir / "gate_report.json")
    local_summary = _read_json(data_dir / "local_pipeline_summary.json")
    labels_payload = _read_json(data_dir / "labels_template.json")
    run_manifest = _read_json(data_dir / "run_manifest.json")
    seed = _to_int(run_manifest.get("seed"))
    if seed is None:
        seed = _to_int(existing.get("seed"))
    duration_sec = existing.get("duration_sec", _duration_from_manifest(run_manifest))
    return _build_run_row(
        backend=backend,
        repeat=int(repeat),
        data_dir=data_dir,
        gate_payload=gate_payload,
        local_summary=local_summary,
        labels_payload=labels_payload,
        run_manifest=run_manifest,
        run_id=_resolve_run_id(run_manifest, existing, backend, repeat),
        seed=seed,
        returncode=_infer_returncode(run_manifest, existing.get("returncode")),
        duration_sec=duration_sec,
        stdout_tail=str(existing.get("stdout_tail", "")),
        stderr_tail=str(existing.get("stderr_tail", "")),
    )


def _existing_bool_value(report: dict[str, Any], key: str, fallback: bool) -> bool:
    if key in report:
        return bool(report.get(key))
    return bool(fallback)


def _existing_list_value(report: dict[str, Any], key: str, fallback: list[str]) -> list[str]:
    value = report.get(key)
    if isinstance(value, list):
        return [str(item) for item in value]
    return list(fallback)


def _resolve_report_controls(
    *,
    summarize_only: bool,
    runs: list[dict[str, Any]],
    existing_report: dict[str, Any],
    args_repeats: int,
    args_base_seed: int | None,
    args_vjepa2_pooling: str | None,
    args_critical_patterns: list[str] | None,
    args_skip_generate_bad: bool,
    args_skip_prefill_labels: bool,
    args_disable_embedder_fallback: bool,
    args_fail_on_gate: bool,
) -> dict[str, Any]:
    if not summarize_only:
        return {
            "repeats": int(args_repeats),
            "base_seed": int(args_base_seed) if args_base_seed is not None else DEFAULT_BASE_SEED,
            "vjepa2_pooling": (args_vjepa2_pooling or DEFAULT_VJEPA2_POOLING),
            "critical_patterns": list(args_critical_patterns or []),
            "skip_generate_bad": bool(args_skip_generate_bad),
            "skip_prefill_labels": bool(args_skip_prefill_labels),
            "disable_embedder_fallback": bool(args_disable_embedder_fallback),
            "fail_on_gate": bool(args_fail_on_gate),
        }

    run_repeats = [repeat for repeat in (_to_int(row.get("repeat")) for row in runs) if repeat is not None]
    repeats_value = max(run_repeats) if run_repeats else None
    if repeats_value is None:
        repeats_value = _to_int(existing_report.get("repeats"))
    if repeats_value is None:
        repeats_value = int(args_repeats)

    base_seed_value = _to_int(existing_report.get("base_seed"))
    if base_seed_value is None:
        base_seed_value = int(args_base_seed) if args_base_seed is not None else DEFAULT_BASE_SEED

    vjepa2_pooling_value = existing_report.get("vjepa2_pooling")
    if not isinstance(vjepa2_pooling_value, str) or not vjepa2_pooling_value.strip():
        vjepa2_pooling_value = args_vjepa2_pooling or DEFAULT_VJEPA2_POOLING

    return {
        "repeats": int(repeats_value),
        "base_seed": int(base_seed_value),
        "vjepa2_pooling": str(vjepa2_pooling_value),
        "critical_patterns": _existing_list_value(existing_report, "critical_patterns", list(args_critical_patterns or [])),
        "skip_generate_bad": _existing_bool_value(existing_report, "skip_generate_bad", args_skip_generate_bad),
        "skip_prefill_labels": _existing_bool_value(existing_report, "skip_prefill_labels", args_skip_prefill_labels),
        "disable_embedder_fallback": _existing_bool_value(
            existing_report,
            "disable_embedder_fallback",
            args_disable_embedder_fallback,
        ),
        "fail_on_gate": _existing_bool_value(existing_report, "fail_on_gate", args_fail_on_gate),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run backend ablation with reproducible reports.")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--trainee-dir", default=None)
    parser.add_argument("--trainee-bad-dir", default=None)
    parser.add_argument("--site-id", default=None)
    parser.add_argument("--gold-id", type=int, default=None)
    parser.add_argument("--max-source", type=int, default=None)
    parser.add_argument("--critical-pattern", action="append", default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Repeat seed base; run seed is base-seed + repeat.",
    )
    parser.add_argument("--backends", nargs="+", default=None)
    parser.add_argument(
        "--vjepa2-pooling",
        choices=["mean_tokens", "first_token", "flatten"],
        default=None,
        help="Pooling strategy when backend=vjepa2.",
    )
    parser.add_argument("--output-root", default="artifacts/backend_ablation")
    parser.add_argument("--skip-generate-bad", action="store_true")
    parser.add_argument("--skip-prefill-labels", action="store_true")
    parser.add_argument(
        "--disable-embedder-fallback",
        action="store_true",
        help="Strict mode: disable fallback and fail fast on primary embedder errors.",
    )
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Propagate gate failures as non-zero exit code from each run.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Do not run new experiments; rebuild report from existing runs under output-root.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, backends: list[str]) -> None:
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if not backends:
        raise SystemExit("--backends is empty")
    invalid = [backend for backend in backends if backend not in ALLOWED_BACKENDS]
    if invalid:
        raise SystemExit(f"invalid backend(s): {invalid}")

    if args.summarize_only:
        return
    required_when_running = {
        "--task-id": args.task_id,
        "--base-dir": args.base_dir,
        "--trainee-dir": args.trainee_dir,
        "--trainee-bad-dir": args.trainee_bad_dir,
    }
    missing = [key for key, value in required_when_running.items() if not value]
    if missing:
        raise SystemExit(f"missing required arguments when running experiments: {', '.join(missing)}")


def _resolve_run_defaults(args: argparse.Namespace) -> RunDefaults:
    return RunDefaults(
        task_name=str(args.task_name or DEFAULT_TASK_NAME),
        site_id=str(args.site_id or DEFAULT_SITE_ID),
        gold_id=int(args.gold_id) if args.gold_id is not None else DEFAULT_GOLD_ID,
        max_source=int(args.max_source) if args.max_source is not None else DEFAULT_MAX_SOURCE,
        base_seed=int(args.base_seed) if args.base_seed is not None else DEFAULT_BASE_SEED,
        vjepa2_pooling=str(args.vjepa2_pooling or DEFAULT_VJEPA2_POOLING),
    )


def _build_flow_command(
    *,
    args: argparse.Namespace,
    defaults: RunDefaults,
    backend: str,
    repeat: int,
    data_dir: Path,
) -> tuple[list[str], str, int]:
    rep_name = f"rep{repeat:02d}"
    repeat_seed = int(defaults.base_seed) + int(repeat)
    run_id = f"{args.task_id}:{backend}:{rep_name}"
    cmd = [
        sys.executable,
        "scripts/run_poc_critical_gate_flow.py",
        "--task-id",
        str(args.task_id),
        "--task-name",
        defaults.task_name,
        "--run-id",
        run_id,
        "--base-dir",
        str(Path(str(args.base_dir)).resolve()),
        "--trainee-dir",
        str(Path(str(args.trainee_dir)).resolve()),
        "--trainee-bad-dir",
        str(Path(str(args.trainee_bad_dir)).resolve()),
        "--data-dir",
        str(data_dir),
        "--site-id",
        defaults.site_id,
        "--gold-id",
        str(int(defaults.gold_id)),
        "--backend",
        backend,
        "--vjepa2-pooling",
        defaults.vjepa2_pooling,
        "--max-source",
        str(int(defaults.max_source)),
        "--seed",
        str(repeat_seed),
        "--reset-data-dir",
    ]
    if args.skip_generate_bad:
        cmd.append("--skip-generate-bad")
    if args.skip_prefill_labels:
        cmd.append("--skip-prefill-labels")
    if args.disable_embedder_fallback:
        cmd.append("--disable-embedder-fallback")
    if not args.fail_on_gate:
        cmd.append("--no-fail-on-gate")
    for pattern in (args.critical_pattern or []):
        cmd.extend(["--critical-pattern", pattern])
    return cmd, run_id, repeat_seed


def _build_dry_run_row(*, backend: str, repeat: int, run_id: str, seed: int, data_dir: Path) -> dict[str, Any]:
    return {
        "backend": backend,
        "repeat": repeat,
        "run_id": run_id,
        "seed": seed,
        "data_dir": str(data_dir),
        "returncode": None,
        "duration_sec": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "gate_metrics": {},
        "embedder_runtime": {},
        "contaminated_by_fallback": False,
        "expected_job_count": None,
        "completed_job_count": None,
        "completed_score_jobs_total": None,
        "coverage_rate": None,
        "critical_fp_count": None,
        "critical_tn_count": None,
        "critical_tp_count": None,
        "critical_fn_count": None,
        "critical_negatives": None,
        "critical_positives": None,
        "labels_total_jobs": 0,
        "labels_labeled_jobs": 0,
        "labels_unknown_jobs": 0,
    }


def _collect_summarize_rows(
    *,
    root: Path,
    backends: list[str],
    existing_report: dict[str, Any],
    existing_index: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backend in backends:
        backend_root = root / "runs" / backend
        if not backend_root.exists():
            continue
        rep_dirs = sorted([path for path in backend_root.iterdir() if path.is_dir()], key=lambda path: path.name)
        for pos, rep_dir in enumerate(rep_dirs, start=1):
            repeat = _parse_repeat_from_name(rep_dir.name) or pos
            existing_row = existing_index.get((backend, int(repeat)))
            rows.append(
                _build_run_row_from_data_dir(
                    backend=backend,
                    repeat=int(repeat),
                    data_dir=rep_dir,
                    existing_row=existing_row,
                )
            )
    if rows:
        return rows
    if isinstance(existing_report.get("runs"), list):
        return [row for row in existing_report.get("runs", []) if isinstance(row, dict)]
    return []


def _execute_runs(
    *,
    root: Path,
    args: argparse.Namespace,
    backends: list[str],
    defaults: RunDefaults,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backend in backends:
        for repeat in range(1, int(args.repeats) + 1):
            rep_name = f"rep{repeat:02d}"
            data_dir = root / "runs" / backend / rep_name
            data_dir.mkdir(parents=True, exist_ok=True)
            cmd, run_id, repeat_seed = _build_flow_command(
                args=args,
                defaults=defaults,
                backend=backend,
                repeat=repeat,
                data_dir=data_dir,
            )

            print(f"[ablation] backend={backend} repeat={repeat} data_dir={data_dir}", flush=True)
            print("$", " ".join(cmd), flush=True)

            if args.dry_run:
                rows.append(
                    _build_dry_run_row(
                        backend=backend,
                        repeat=repeat,
                        run_id=run_id,
                        seed=repeat_seed,
                        data_dir=data_dir,
                    )
                )
                continue

            returncode, duration_sec, stdout, stderr = _run_one(cmd)
            gate_payload = _read_json(data_dir / "gate_report.json")
            local_summary = _read_json(data_dir / "local_pipeline_summary.json")
            labels_payload = _read_json(data_dir / "labels_template.json")
            run_manifest = _read_json(data_dir / "run_manifest.json")
            rows.append(
                _build_run_row(
                    backend=backend,
                    repeat=repeat,
                    data_dir=data_dir,
                    gate_payload=gate_payload,
                    local_summary=local_summary,
                    labels_payload=labels_payload,
                    run_manifest=run_manifest,
                    run_id=run_id,
                    seed=repeat_seed,
                    returncode=returncode,
                    duration_sec=duration_sec,
                    stdout_tail=_tail(stdout),
                    stderr_tail=_tail(stderr),
                )
            )
    return rows


def _sort_runs(runs: list[dict[str, Any]], backends: list[str]) -> None:
    backend_rank = {name: idx for idx, name in enumerate(backends)}
    runs.sort(
        key=lambda row: (
            backend_rank.get(str(row.get("backend")), 999),
            _to_int(row.get("repeat")) or 0,
        )
    )


def _resolve_int(primary: Any, fallback: Any, default: int) -> int:
    value = _to_int(primary)
    if value is not None:
        return int(value)
    value = _to_int(fallback)
    if value is not None:
        return int(value)
    return int(default)


def _resolve_path(value: Any) -> str | None:
    if not value:
        return None
    return str(Path(str(value)).resolve())


def _resolve_report_context(
    *,
    args: argparse.Namespace,
    existing_report: dict[str, Any],
    defaults: RunDefaults,
) -> ReportContext:
    if args.summarize_only:
        return ReportContext(
            task_id=str(args.task_id or existing_report.get("task_id") or "ablation_summarize"),
            task_name=str(args.task_name or existing_report.get("task_name") or defaults.task_name),
            base_dir=args.base_dir or existing_report.get("base_dir"),
            trainee_dir=args.trainee_dir or existing_report.get("trainee_dir"),
            trainee_bad_dir=args.trainee_bad_dir or existing_report.get("trainee_bad_dir"),
            site_id=str(args.site_id or existing_report.get("site_id") or defaults.site_id),
            gold_id=_resolve_int(args.gold_id, existing_report.get("gold_id"), defaults.gold_id),
            max_source=_resolve_int(args.max_source, existing_report.get("max_source"), defaults.max_source),
        )
    return ReportContext(
        task_id=str(args.task_id),
        task_name=defaults.task_name,
        base_dir=args.base_dir,
        trainee_dir=args.trainee_dir,
        trainee_bad_dir=args.trainee_bad_dir,
        site_id=defaults.site_id,
        gold_id=defaults.gold_id,
        max_source=defaults.max_source,
    )


def _build_report(
    *,
    context: ReportContext,
    backends: list[str],
    controls: dict[str, Any],
    runs: list[dict[str, Any]],
    summarize_only: bool,
) -> dict[str, Any]:
    report = {
        "generated_at_utc": now_iso(),
        "task_id": context.task_id,
        "task_name": context.task_name,
        "base_dir": _resolve_path(context.base_dir),
        "trainee_dir": _resolve_path(context.trainee_dir),
        "trainee_bad_dir": _resolve_path(context.trainee_bad_dir),
        "site_id": context.site_id,
        "gold_id": context.gold_id,
        "max_source": context.max_source,
        "backends": backends,
        "vjepa2_pooling": controls["vjepa2_pooling"],
        "repeats": controls["repeats"],
        "base_seed": controls["base_seed"],
        "critical_patterns": controls["critical_patterns"],
        "skip_generate_bad": controls["skip_generate_bad"],
        "skip_prefill_labels": controls["skip_prefill_labels"],
        "disable_embedder_fallback": controls["disable_embedder_fallback"],
        "fail_on_gate": controls["fail_on_gate"],
        "summarize_only": bool(summarize_only),
        "runtime_info": _probe_runtime_info(),
        "runs": runs,
    }
    report["summary"] = summarize_runs(runs, backends)
    return report


def main() -> None:
    args = _build_parser().parse_args()
    root = Path(args.output_root).resolve()
    backends = _discover_backends(root, args.backends)
    _validate_args(args, backends)

    root.mkdir(parents=True, exist_ok=True)
    existing_report = _read_json(root / "ablation_report.json")
    existing_index = _existing_runs_index(existing_report)
    defaults = _resolve_run_defaults(args)

    if args.summarize_only:
        runs = _collect_summarize_rows(
            root=root,
            backends=backends,
            existing_report=existing_report,
            existing_index=existing_index,
        )
    else:
        runs = _execute_runs(
            root=root,
            args=args,
            backends=backends,
            defaults=defaults,
        )
    _sort_runs(runs, backends)

    controls = _resolve_report_controls(
        summarize_only=bool(args.summarize_only),
        runs=runs,
        existing_report=existing_report,
        args_repeats=int(args.repeats),
        args_base_seed=args.base_seed,
        args_vjepa2_pooling=args.vjepa2_pooling,
        args_critical_patterns=args.critical_pattern,
        args_skip_generate_bad=bool(args.skip_generate_bad),
        args_skip_prefill_labels=bool(args.skip_prefill_labels),
        args_disable_embedder_fallback=bool(args.disable_embedder_fallback),
        args_fail_on_gate=bool(args.fail_on_gate),
    )
    context = _resolve_report_context(
        args=args,
        existing_report=existing_report,
        defaults=defaults,
    )
    report = _build_report(
        context=context,
        backends=backends,
        controls=controls,
        runs=runs,
        summarize_only=bool(args.summarize_only),
    )

    report_path = root / "ablation_report.json"
    md_path = root / "ablation_report.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown_report(report), encoding="utf-8")

    print(f"[ablation] report_json={report_path}", flush=True)
    print(f"[ablation] report_md={md_path}", flush=True)


if __name__ == "__main__":
    main()
