from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from sopilot.database import Database
from sopilot.eval.harness import (
    available_critical_scoring_modes,
    build_critical_score_breakdown,
    job_detected_critical,
    load_critical_labels,
)


def _query_video_paths(db_path: Path, video_ids: set[int]) -> dict[int, str]:
    if not video_ids:
        return {}
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        placeholders = ",".join(["?"] * len(video_ids))
        cur.execute(
            f"SELECT id, file_path FROM videos WHERE id IN ({placeholders})",
            tuple(sorted(video_ids)),
        )
        out: dict[int, str] = {}
        for video_id, file_path in cur.fetchall():
            if file_path:
                out[int(video_id)] = str(file_path)
        return out
    finally:
        conn.close()


def _load_backend_hint(summary_path: Path | None, explicit_backend: str | None) -> str:
    if explicit_backend:
        return str(explicit_backend).strip()
    if summary_path is None or not summary_path.exists():
        return "unknown"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return "unknown"
    runtime = payload.get("embedder_runtime", {}) or {}
    if not isinstance(runtime, dict):
        return "unknown"
    for key in ("backend", "embedder_backend", "embedder_name"):
        value = runtime.get(key)
        if value:
            return str(value)
    return "unknown"


def _deviation_types(job_score: dict[str, Any]) -> list[str]:
    deviations = job_score.get("deviations", []) or []
    items: set[str] = set()
    for dev in deviations:
        dtype = str(dev.get("type", "unknown")).strip().lower()
        if dtype:
            items.add(dtype)
    if not items:
        return ["none"]
    return sorted(items)


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
        mean_distances.append(float(dev.get("mean_distance") or 0.0))
        expected_spans.append(float(dev.get("expected_span_len") or 0.0))

    def _avg(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    return {
        "num_critical_missing": float(len(critical_missing)),
        "mean_distance_avg": _avg(mean_distances),
        "expected_span_avg": _avg(expected_spans),
    }


def _infer_fp_reason(
    *,
    job_score: dict[str, Any],
    missing_stats: dict[str, float],
) -> tuple[str, list[str]]:
    deviations = job_score.get("deviations", []) or []
    metrics = job_score.get("metrics", {}) or {}
    tags: list[str] = []
    if missing_stats["num_critical_missing"] <= 0:
        tags.append("non_missing_critical")
        return "non_missing_critical", tags

    mean_distance_avg = float(missing_stats["mean_distance_avg"])
    expected_span_avg = float(missing_stats["expected_span_avg"])
    dtw = float(metrics.get("dtw_normalized_cost") or 0.0)
    over_time_ratio = float(metrics.get("over_time_ratio") or 0.0)
    critical_deviation_count = 0
    for dev in deviations:
        if str(dev.get("severity", "")).strip().lower() == "critical":
            critical_deviation_count += 1

    if expected_span_avg > 1.5:
        tags.append("expected_span_gt_1_5")
    if mean_distance_avg > 0.11:
        tags.append("mean_distance_gt_0_11")
    if dtw > 0.12:
        tags.append("dtw_gt_0_12")
    if over_time_ratio > 0.50:
        tags.append("over_time_gt_0_50")
    if critical_deviation_count >= 2:
        tags.append("multi_critical_missing")

    if "expected_span_gt_1_5" in tags and "mean_distance_gt_0_11" not in tags:
        return "near_freeze_alignment_span", tags
    if "mean_distance_gt_0_11" in tags and "expected_span_gt_1_5" not in tags:
        return "high_distance_alignment_mismatch", tags
    if "mean_distance_gt_0_11" in tags and "expected_span_gt_1_5" in tags:
        return "severe_alignment_mismatch", tags
    if "multi_critical_missing" in tags:
        return "multi_missing_step_pattern", tags
    return "single_missing_step_low_distance", tags


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_rate_rows(
    *,
    fp_counts: Counter[str],
    negative_counts: Counter[str],
    key_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, negatives in sorted(negative_counts.items(), key=lambda x: (-x[1], x[0])):
        fp = int(fp_counts.get(key, 0))
        rate = (fp / negatives) if negatives > 0 else None
        rows.append(
            {
                key_name: key,
                "negative_jobs": int(negatives),
                "false_positive_jobs": fp,
                "false_positive_rate": round(float(rate), 6) if rate is not None else None,
            }
        )
    return rows


def _escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _build_manifest(
    *,
    fp_rows: list[dict[str, Any]],
    fn_rows: list[dict[str, Any]],
    breakdown: dict[str, Any],
    top_k: int,
    scoring_mode: str,
    threshold: float,
) -> str:
    fp_top = sorted(
        fp_rows,
        key=lambda row: (
            -float(_safe_float(row.get("critical_score")) or 0.0),
            -float(_safe_float(row.get("threshold_margin")) or 0.0),
            int(row.get("job_id", 0)),
        ),
    )[: max(0, int(top_k))]
    fn_top = sorted(
        fn_rows,
        key=lambda row: (
            float(_safe_float(row.get("critical_score")) or 0.0),
            float(_safe_float(row.get("threshold_margin")) or 0.0),
            int(row.get("job_id", 0)),
        ),
    )[: max(0, int(top_k))]

    lines: list[str] = []
    lines.append("# Critical Error Cases")
    lines.append("")
    lines.append(f"- critical_scoring_mode: `{scoring_mode}`")
    lines.append(f"- critical_threshold: `{threshold}`")
    lines.append(f"- labeled_jobs: `{breakdown['overall']['labeled_jobs']}`")
    lines.append(f"- labeled_negatives: `{breakdown['overall']['labeled_negative_jobs']}`")
    lines.append(f"- false_positives: `{breakdown['overall']['false_positive_jobs']}`")
    lines.append(f"- false_negatives: `{breakdown['overall']['false_negative_jobs']}`")
    lines.append(f"- false_positive_rate: `{breakdown['overall']['false_positive_rate']}`")
    lines.append(f"- miss_rate: `{breakdown['overall']['miss_rate']}`")
    lines.append("")
    lines.append("## Top False Positives")
    lines.append("")
    lines.append("|rank|job_id|critical_score|margin|site_id|gold_id|backend|deviation_types|trainee_path|")
    lines.append("|---:|---:|---:|---:|---|---:|---|---|---|")
    for idx, row in enumerate(fp_top, start=1):
        lines.append(
            "|"
            + "|".join(
                [
                    str(idx),
                    str(row.get("job_id", "")),
                    str(row.get("critical_score", "")),
                    str(row.get("threshold_margin", "")),
                    _escape_md(row.get("site_id", "unknown")),
                    str(row.get("gold_video_id", "")),
                    _escape_md(row.get("backend", "unknown")),
                    _escape_md(",".join(row.get("deviation_types", []))),
                    _escape_md(row.get("trainee_path", "")),
                ]
            )
            + "|"
        )
    lines.append("")
    lines.append("## Top False Negatives")
    lines.append("")
    lines.append("|rank|job_id|critical_score|margin|site_id|gold_id|backend|deviation_types|trainee_path|")
    lines.append("|---:|---:|---:|---:|---|---:|---|---|---|")
    for idx, row in enumerate(fn_top, start=1):
        lines.append(
            "|"
            + "|".join(
                [
                    str(idx),
                    str(row.get("job_id", "")),
                    str(row.get("critical_score", "")),
                    str(row.get("threshold_margin", "")),
                    _escape_md(row.get("site_id", "unknown")),
                    str(row.get("gold_video_id", "")),
                    _escape_md(row.get("backend", "unknown")),
                    _escape_md(",".join(row.get("deviation_types", []))),
                    _escape_md(row.get("trainee_path", "")),
                ]
            )
            + "|"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def extract_error_cases(
    *,
    completed_jobs: list[dict[str, Any]],
    labels: dict[int, bool],
    video_path_by_id: dict[int, str],
    scoring_mode: str,
    critical_threshold: float,
    backend: str,
    critical_policy: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fp_rows: list[dict[str, Any]] = []
    fn_rows: list[dict[str, Any]] = []

    labeled_jobs = 0
    labeled_positive_jobs = 0
    labeled_negative_jobs = 0
    fp_count = 0
    fn_count = 0
    tp_count = 0
    tn_count = 0

    neg_site_counter: Counter[str] = Counter()
    neg_backend_counter: Counter[str] = Counter()
    neg_gold_counter: Counter[str] = Counter()
    neg_devtype_counter: Counter[str] = Counter()
    neg_reason_counter: Counter[str] = Counter()

    fp_site_counter: Counter[str] = Counter()
    fp_backend_counter: Counter[str] = Counter()
    fp_gold_counter: Counter[str] = Counter()
    fp_devtype_counter: Counter[str] = Counter()
    fp_reason_counter: Counter[str] = Counter()

    for job in completed_jobs:
        job_id = int(job["id"])
        if job_id not in labels:
            continue
        labeled_jobs += 1
        expected = bool(labels[job_id])
        score_payload = job["score"]
        score_breakdown = build_critical_score_breakdown(score_payload)
        detected = job_detected_critical(
            score_payload,
            scoring_mode=scoring_mode,
            critical_threshold=critical_threshold,
            critical_policy=critical_policy,
        )
        critical_score = float(score_breakdown["critical_score"])
        deviation_types = _deviation_types(score_payload)
        missing_stats = _critical_missing_stats(score_payload)
        fp_reason, fp_reason_tags = _infer_fp_reason(job_score=score_payload, missing_stats=missing_stats)
        site_id = str(job.get("trainee_site_id") or "unknown")
        gold_id = int(job["gold_video_id"])
        gold_id_key = str(gold_id)
        backend_key = str(backend or "unknown")
        trainee_id = int(job["trainee_video_id"])
        gold_path = video_path_by_id.get(gold_id)
        trainee_path = video_path_by_id.get(trainee_id)

        row = {
            "job_id": job_id,
            "task_id": job.get("task_id"),
            "gold_video_id": gold_id,
            "trainee_video_id": trainee_id,
            "gold_path": gold_path,
            "trainee_path": trainee_path,
            "site_id": site_id,
            "backend": backend_key,
            "critical_expected": expected,
            "critical_detected": bool(detected),
            "critical_scoring_mode": scoring_mode,
            "critical_threshold": round(float(critical_threshold), 6),
            "critical_policy_id": (critical_policy or {}).get("policy_id"),
            "critical_score": round(float(critical_score), 6),
            "critical_raw_score": score_breakdown["raw_score"],
            "threshold_margin": round(float(critical_score - critical_threshold), 6),
            "score": _safe_float(score_payload.get("score")),
            "decision": (score_payload.get("summary", {}) or {}).get("decision"),
            "decision_reason": (score_payload.get("summary", {}) or {}).get("decision_reason"),
            "severity_counts": score_breakdown["severity_counts"],
            "metrics": score_payload.get("metrics", {}),
            "deviation_types": deviation_types,
            "fp_reason_auto": fp_reason,
            "fp_reason_tags": fp_reason_tags,
            "critical_missing_stats": {
                "num_critical_missing": int(missing_stats["num_critical_missing"]),
                "mean_distance_avg": round(float(missing_stats["mean_distance_avg"]), 6),
                "expected_span_avg": round(float(missing_stats["expected_span_avg"]), 6),
            },
            "deviations": score_payload.get("deviations", []),
            "threshold_decision": {
                "mode": scoring_mode,
                "threshold": round(float(critical_threshold), 6),
                "detected": bool(detected),
                "score_breakdown": score_breakdown,
            },
        }

        if expected:
            labeled_positive_jobs += 1
        else:
            labeled_negative_jobs += 1
            neg_site_counter[site_id] += 1
            neg_backend_counter[backend_key] += 1
            neg_gold_counter[gold_id_key] += 1
            for dtype in deviation_types:
                neg_devtype_counter[dtype] += 1
            neg_reason_counter[fp_reason] += 1

        if expected and detected:
            tp_count += 1
        elif expected and not detected:
            fn_count += 1
            fn_rows.append(row)
        elif (not expected) and detected:
            fp_count += 1
            fp_rows.append(row)
            fp_site_counter[site_id] += 1
            fp_backend_counter[backend_key] += 1
            fp_gold_counter[gold_id_key] += 1
            for dtype in deviation_types:
                fp_devtype_counter[dtype] += 1
            fp_reason_counter[fp_reason] += 1
        else:
            tn_count += 1

    miss_rate = (fn_count / labeled_positive_jobs) if labeled_positive_jobs > 0 else None
    fp_rate = (fp_count / labeled_negative_jobs) if labeled_negative_jobs > 0 else None
    breakdown = {
        "overall": {
            "labeled_jobs": int(labeled_jobs),
            "labeled_positive_jobs": int(labeled_positive_jobs),
            "labeled_negative_jobs": int(labeled_negative_jobs),
            "false_positive_jobs": int(fp_count),
            "false_negative_jobs": int(fn_count),
            "true_positive_jobs": int(tp_count),
            "true_negative_jobs": int(tn_count),
            "false_positive_rate": round(float(fp_rate), 6) if fp_rate is not None else None,
            "miss_rate": round(float(miss_rate), 6) if miss_rate is not None else None,
            "critical_scoring_mode": scoring_mode,
            "critical_threshold": round(float(critical_threshold), 6),
            "backend": backend,
        },
        "by_deviation_type": _make_rate_rows(
            fp_counts=fp_devtype_counter,
            negative_counts=neg_devtype_counter,
            key_name="deviation_type",
        ),
        "by_site_id": _make_rate_rows(
            fp_counts=fp_site_counter,
            negative_counts=neg_site_counter,
            key_name="site_id",
        ),
        "by_backend": _make_rate_rows(
            fp_counts=fp_backend_counter,
            negative_counts=neg_backend_counter,
            key_name="backend",
        ),
        "by_gold_id": _make_rate_rows(
            fp_counts=fp_gold_counter,
            negative_counts=neg_gold_counter,
            key_name="gold_id",
        ),
        "by_fp_reason": _make_rate_rows(
            fp_counts=fp_reason_counter,
            negative_counts=neg_reason_counter,
            key_name="fp_reason",
        ),
    }
    return fp_rows, fn_rows, breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract labeled critical FP/FN cases from completed jobs.")
    parser.add_argument("--db-path", required=True, help="Path to sopilot.db")
    parser.add_argument("--labels", required=True, help="Path to labels_template.json")
    parser.add_argument("--task-id", default=None, help="Optional task_id filter")
    parser.add_argument("--site-id", default=None, help="Optional trainee site filter")
    parser.add_argument("--gold-id", type=int, default=None, help="Optional gold_video_id filter")
    parser.add_argument("--backend", default=None, help="Optional backend label for breakdown rows")
    parser.add_argument("--summary", default=None, help="Optional summary JSON used to infer backend label")
    parser.add_argument(
        "--critical-scoring-mode",
        choices=available_critical_scoring_modes(),
        default="legacy_binary",
    )
    parser.add_argument("--critical-threshold", type=float, default=0.5)
    parser.add_argument("--critical-policy", default=None, help="Optional policy JSON used by guarded modes")
    parser.add_argument("--top-k", type=int, default=30, help="Rows included in topK manifest tables")
    parser.add_argument("--output-dir", default="artifacts/errors", help="Directory for output files")
    args = parser.parse_args()

    db_path = Path(args.db_path).resolve()
    labels_path = Path(args.labels).resolve()
    output_dir = Path(args.output_dir).resolve()
    summary_path = Path(args.summary).resolve() if args.summary else None

    labels = load_critical_labels(labels_path).labels
    critical_policy = None
    if args.critical_policy:
        critical_policy = json.loads(Path(args.critical_policy).read_text(encoding="utf-8"))
    db = Database(db_path)
    completed_jobs = db.list_completed_score_jobs(task_id=args.task_id)
    if args.site_id:
        completed_jobs = [row for row in completed_jobs if str(row.get("trainee_site_id") or "") == str(args.site_id)]
    if args.gold_id is not None:
        completed_jobs = [row for row in completed_jobs if int(row["gold_video_id"]) == int(args.gold_id)]

    video_ids: set[int] = set()
    for row in completed_jobs:
        video_ids.add(int(row["gold_video_id"]))
        video_ids.add(int(row["trainee_video_id"]))
    video_paths = _query_video_paths(db_path, video_ids)
    backend = _load_backend_hint(summary_path, args.backend)

    fp_rows, fn_rows, breakdown = extract_error_cases(
        completed_jobs=completed_jobs,
        labels=labels,
        video_path_by_id=video_paths,
        scoring_mode=args.critical_scoring_mode,
        critical_threshold=float(args.critical_threshold),
        backend=backend,
        critical_policy=critical_policy,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fp_path = output_dir / "false_positives.jsonl"
    fn_path = output_dir / "false_negatives.jsonl"
    breakdown_path = output_dir / "fp_breakdown.json"
    manifest_path = output_dir / "topK_fp_manifest.md"

    _write_jsonl(fp_path, fp_rows)
    _write_jsonl(fn_path, fn_rows)
    breakdown_path.write_text(json.dumps(breakdown, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(
        _build_manifest(
            fp_rows=fp_rows,
            fn_rows=fn_rows,
            breakdown=breakdown,
            top_k=max(0, int(args.top_k)),
            scoring_mode=args.critical_scoring_mode,
            threshold=float(args.critical_threshold),
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "false_positives": str(fp_path),
                "false_negatives": str(fn_path),
                "fp_breakdown": str(breakdown_path),
                "topk_manifest": str(manifest_path),
                "false_positive_jobs": len(fp_rows),
                "false_negative_jobs": len(fn_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
