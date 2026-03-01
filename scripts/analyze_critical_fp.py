from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bucket_name(file_path: str | None) -> str:
    if not file_path:
        return "unknown"
    name = Path(file_path).name.lower()
    if "_bad_freeze" in name:
        return "bad_freeze"
    if "_bad_skip_start" in name:
        return "bad_skip_start"
    if "_bad_cut_tail" in name:
        return "bad_cut_tail"
    if "_bad_" in name:
        return "other_bad"
    return "normal"


def _db_mappings(db_path: Path, job_ids: set[int]) -> tuple[dict[int, int], dict[int, str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        job_to_trainee: dict[int, int] = {}
        if job_ids:
            placeholders = ",".join(["?"] * len(job_ids))
            cur.execute(
                f"SELECT id, trainee_video_id FROM score_jobs WHERE id IN ({placeholders})",
                tuple(sorted(job_ids)),
            )
            for job_id, trainee_video_id in cur.fetchall():
                if trainee_video_id is not None:
                    job_to_trainee[int(job_id)] = int(trainee_video_id)

        video_ids = sorted(set(job_to_trainee.values()))
        video_path_by_id: dict[int, str] = {}
        if video_ids:
            placeholders = ",".join(["?"] * len(video_ids))
            cur.execute(
                f"SELECT id, file_path FROM videos WHERE id IN ({placeholders})",
                tuple(video_ids),
            )
            for video_id, file_path in cur.fetchall():
                if file_path:
                    video_path_by_id[int(video_id)] = str(file_path)
        return job_to_trainee, video_path_by_id
    finally:
        conn.close()


def analyze_fp(
    *,
    labels_payload: dict,
    summary_payload: dict,
    sample_limit: int,
    db_path: Path | None = None,
) -> dict:
    labels_jobs = labels_payload.get("jobs", [])
    labeled_jobs = [row for row in labels_jobs if row.get("critical_expected", None) is not None]
    all_jobs = {
        int(row["job_id"]): row
        for row in (summary_payload.get("all_score_jobs", []) or [])
        if row.get("job_id") is not None
    }
    video_path_by_id = {
        int(video_id): str(path)
        for video_id, path in (summary_payload.get("video_path_by_id", {}) or {}).items()
    }
    all_job_ids = {int(row["job_id"]) for row in labeled_jobs if row.get("job_id") is not None}
    db_job_to_trainee: dict[int, int] = {}
    db_video_paths: dict[int, str] = {}
    if db_path is not None and db_path.exists():
        db_job_to_trainee, db_video_paths = _db_mappings(db_path, all_job_ids)

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    fp_rows = []
    for row in labeled_jobs:
        expected = bool(row.get("critical_expected"))
        predicted = bool(row.get("predicted_critical", False))
        if expected and predicted:
            tp += 1
        elif expected and not predicted:
            fn += 1
        elif (not expected) and predicted:
            fp += 1
        else:
            tn += 1
        if expected or not predicted:
            continue
        job_id = int(row["job_id"])
        linked = all_jobs.get(job_id, {})
        trainee_video_id = linked.get("trainee_video_id")
        if trainee_video_id is None:
            trainee_video_id = db_job_to_trainee.get(job_id)
        trainee_path = None
        if trainee_video_id is not None:
            trainee_path = video_path_by_id.get(int(trainee_video_id)) or db_video_paths.get(int(trainee_video_id))
        fp_rows.append(
            {
                "job_id": job_id,
                "trainee_video_id": trainee_video_id,
                "trainee_path": trainee_path,
                "bucket": _bucket_name(trainee_path),
            }
        )

    bucket_counts = Counter(row["bucket"] for row in fp_rows)
    total_labeled = len(labeled_jobs)
    fp_count = len(fp_rows)
    fp_ratio_labeled = (fp_count / total_labeled) if total_labeled > 0 else None
    fp_rate_gate = (fp / (fp + tn)) if (fp + tn) > 0 else None

    return {
        "total_jobs_in_template": len(labels_jobs),
        "total_labeled_jobs": total_labeled,
        "fp_jobs": fp_count,
        "fp_ratio_over_labeled": round(float(fp_ratio_labeled), 6) if fp_ratio_labeled is not None else None,
        "fp_rate_gate_equivalent": round(float(fp_rate_gate), 6) if fp_rate_gate is not None else None,
        "confusion_over_labeled": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "fp_bucket_counts": dict(bucket_counts),
        "sample_fp_jobs": fp_rows[: max(0, sample_limit)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze critical false positives from labels + summary.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--labels", default=None, help="Path to labels template json (default: <data-dir>/labels_template.json)")
    parser.add_argument(
        "--summary",
        default=None,
        help="Path to incremental summary json (default: <data-dir>/incremental_summary.json)",
    )
    parser.add_argument("--db-path", default=None, help="Optional DB path for path/job-id fallback mapping.")
    parser.add_argument("--sample-limit", type=int, default=30)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    labels_path = Path(args.labels).resolve() if args.labels else (data_dir / "labels_template.json")
    summary_path = Path(args.summary).resolve() if args.summary else (data_dir / "incremental_summary.json")

    labels_payload = _load_json(labels_path)
    summary_payload = _load_json(summary_path)
    report = analyze_fp(
        labels_payload=labels_payload,
        summary_payload=summary_payload,
        sample_limit=max(0, int(args.sample_limit)),
        db_path=Path(args.db_path).resolve() if args.db_path else (data_dir / "sopilot.db"),
    )
    report["data_dir"] = str(data_dir)
    report["labels_path"] = str(labels_path)
    report["summary_path"] = str(summary_path)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
