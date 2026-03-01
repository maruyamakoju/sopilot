"""convert_labels.py — Convert critical_expected annotations to human_labels format.

Reads labels_template.json (critical_expected per job_id) and produces
human_labels.json (human_verdict per video_id) compatible with
run_loso_evaluation.py --human-labels.

Mapping:
    critical_expected = True  → human_verdict = "fail"
    critical_expected = False → human_verdict = "pass"

When multiple jobs exist for the same trainee video, majority vote decides.

Usage:
    python scripts/convert_labels.py \
        --db data_trip_96h_official_20260212/sopilot.db \
        --template data_trip_96h_official_20260212/labels_template.json \
        --output human_labels.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _open_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise SystemExit(f"[ERROR] Database not found: {db_path}")
    conn = sqlite3.connect(str(db_path), timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA query_only=ON")
    return conn


def _load_job_to_video_map(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    """Map job_id → {trainee_video_id, score, decision}."""
    rows = conn.execute("""
        SELECT
            sj.id AS job_id,
            sj.trainee_video_id,
            json_extract(sj.score_json, '$.score') AS score,
            json_extract(sj.score_json, '$.summary.decision') AS decision
        FROM score_jobs sj
        WHERE sj.status = 'completed' AND sj.score_json IS NOT NULL
    """).fetchall()
    return {int(r["job_id"]): dict(r) for r in rows}


def convert(
    template_path: Path,
    db_path: Path,
    output_path: Path,
) -> None:
    # Load template
    data = json.loads(template_path.read_text(encoding="utf-8"))
    jobs = data.get("jobs", [])
    annotated = [j for j in jobs if j.get("critical_expected") is not None]

    if not annotated:
        raise SystemExit("[ERROR] No annotated entries found in template.")

    print(f"Template: {len(annotated)} annotated entries out of {len(jobs)} total")

    # Map job_id → video info
    conn = _open_db(db_path)
    job_map = _load_job_to_video_map(conn)
    conn.close()

    # Group annotations by trainee_video_id
    video_annotations: dict[int, list[dict[str, Any]]] = {}
    skipped = 0
    for entry in annotated:
        job_id = entry["job_id"]
        info = job_map.get(job_id)
        if info is None:
            skipped += 1
            continue
        vid = int(info["trainee_video_id"])
        if vid not in video_annotations:
            video_annotations[vid] = []
        video_annotations[vid].append({
            "job_id": job_id,
            "critical_expected": entry["critical_expected"],
            "score": info.get("score"),
            "decision": info.get("decision"),
        })

    if skipped:
        print(f"  Skipped {skipped} entries (job_id not found in DB)")

    # Collapse to per-video verdict via majority vote
    labels: list[dict[str, Any]] = []
    stats = Counter()

    for vid, anns in sorted(video_annotations.items()):
        verdicts = []
        scores = []
        for a in anns:
            # critical_expected=True → fail, False → pass
            v = "fail" if a["critical_expected"] else "pass"
            verdicts.append(v)
            if a["score"] is not None:
                scores.append(float(a["score"]))

        # Majority vote
        vote_counter = Counter(verdicts)
        majority_verdict = vote_counter.most_common(1)[0][0]

        # Average system score for reference (not used as ground truth)
        avg_score = round(sum(scores) / len(scores), 2) if scores else None

        label = {
            "video_id": vid,
            "human_score": None,  # No numeric score from critical_expected
            "human_verdict": majority_verdict,
            "annotator_id": "labels_template_conversion",
            "annotation_time": datetime.now(timezone.utc).isoformat(),
            "notes": f"Converted from critical_expected. {len(anns)} jobs, majority={majority_verdict}. Avg system score={avg_score}",
            "source_job_count": len(anns),
            "system_avg_score": avg_score,
        }
        labels.append(label)
        stats[majority_verdict] += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nConverted {len(labels)} video annotations:")
    print(f"  pass: {stats['pass']}")
    print(f"  fail: {stats['fail']}")
    print(f"  Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert critical_expected labels to human_labels format."
    )
    parser.add_argument("--db", required=True, help="Path to sopilot.db")
    parser.add_argument("--template", required=True, help="Path to labels_template.json")
    parser.add_argument("--output", required=True, help="Output human_labels.json path")
    args = parser.parse_args()

    convert(
        template_path=Path(args.template),
        db_path=Path(args.db),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
