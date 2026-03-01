from __future__ import annotations

import argparse
import json
from pathlib import Path


def _is_internal_raw_path(path_text: str) -> bool:
    normalized = path_text.replace("\\", "/").lower()
    return "/raw/" in normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefill critical_expected labels from pipeline summary path patterns.")
    parser.add_argument("--summary", required=True, help="local_pipeline_summary.json")
    parser.add_argument("--labels", required=True, help="labels_template.json to update in-place")
    parser.add_argument(
        "--critical-pattern",
        action="append",
        default=None,
        help="Path substring to mark as critical_expected=true (repeatable)",
    )
    args = parser.parse_args()

    patterns = [p.strip().lower() for p in (args.critical_pattern or ["_bad_freeze"])]
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))

    video_to_path: dict[int, str] = {}
    for item in summary.get("uploads", []):
        if item.get("video_id") is None:
            continue
        try:
            video_to_path[int(item["video_id"])] = str(item["path"])
        except Exception:
            continue
    for raw_key, path in (summary.get("video_path_by_id") or {}).items():
        try:
            video_to_path[int(raw_key)] = str(path)
        except Exception:
            continue

    score_rows = summary.get("all_score_jobs") or summary.get("scores_completed") or []
    job_to_trainee: dict[int, int] = {}
    for item in score_rows:
        try:
            job_to_trainee[int(item["job_id"])] = int(item["trainee_video_id"])
        except Exception:
            continue

    updated = 0
    skipped = 0
    unknown = 0
    for row in labels.get("jobs", []):
        job_id = int(row["job_id"])
        trainee_id = job_to_trainee.get(job_id)
        path_text = video_to_path.get(trainee_id, "").lower() if trainee_id is not None else ""
        if not path_text or _is_internal_raw_path(path_text):
            row["critical_expected"] = None
            unknown += 1
            skipped += 1
            continue
        expected = any(pattern in path_text for pattern in patterns)
        row["critical_expected"] = bool(expected)
        updated += 1

    Path(args.labels).write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"updated={updated} skipped={skipped} unknown={unknown} patterns={patterns} labels={args.labels}")


if __name__ == "__main__":
    main()
