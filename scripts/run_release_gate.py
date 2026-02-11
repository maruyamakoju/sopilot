from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import time

import requests


@dataclass
class ScoreOutcome:
    name: str
    score_job_id: str
    score: float | None
    status: str
    error_message: str | None
    clip_count_gold: int | None
    clip_count_trainee: int | None


def _headers(token: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if token.strip():
        out["Authorization"] = f"Bearer {token.strip()}"
    return out


def _poll_json(base_url: str, headers: dict[str, str], path: str, timeout_sec: int) -> dict:
    started = time.time()
    while time.time() - started <= timeout_sec:
        res = requests.get(f"{base_url}{path}", headers=headers, timeout=30)
        res.raise_for_status()
        payload = res.json()
        if payload.get("status") in {"completed", "failed"}:
            return payload
        time.sleep(1.0)
    raise TimeoutError(f"timeout while polling {path}")


def _upload_and_wait(
    *,
    base_url: str,
    headers: dict[str, str],
    path: str,
    task_id: str,
    file_path: Path,
    role: str,
    timeout_sec: int,
) -> int:
    data = {"task_id": task_id}
    if role != "gold":
        data["role"] = role
    with file_path.open("rb") as f:
        res = requests.post(
            f"{base_url}{path}",
            headers=headers,
            data=data,
            files={"file": (file_path.name, f, "video/mp4")},
            timeout=120,
        )
    res.raise_for_status()
    ingest_job_id = str(res.json()["ingest_job_id"])
    done = _poll_json(base_url, headers, f"/videos/jobs/{ingest_job_id}", timeout_sec=timeout_sec)
    if done.get("status") != "completed":
        raise RuntimeError(f"ingest failed path={file_path} detail={done}")
    video_id = done.get("video_id")
    if video_id is None:
        raise RuntimeError(f"ingest completed without video_id path={file_path}")
    return int(video_id)


def _score_pair(
    *,
    base_url: str,
    headers: dict[str, str],
    gold_video_id: int,
    trainee_video_id: int,
    timeout_sec: int,
) -> dict:
    req_headers = dict(headers)
    req_headers["Content-Type"] = "application/json"
    res = requests.post(
        f"{base_url}/score",
        headers=req_headers,
        json={"gold_video_id": gold_video_id, "trainee_video_id": trainee_video_id},
        timeout=60,
    )
    res.raise_for_status()
    score_job_id = str(res.json()["score_job_id"])
    final = _poll_json(base_url, headers, f"/score/{score_job_id}", timeout_sec=timeout_sec)
    final["_score_job_id"] = score_job_id
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SOPilot release quality gate")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="")
    parser.add_argument("--task-id", default="maintenance_filter_swap")
    parser.add_argument("--gold", default="demo_videos/maintenance_filter_swap/gold.mp4")
    parser.add_argument(
        "--variants",
        default="missing,swap,deviation,time_over,mixed",
        help="comma separated trainee variants (filenames under --variants-dir)",
    )
    parser.add_argument("--variants-dir", default="demo_videos/maintenance_filter_swap")
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--baseline-min-score", type=float, default=90.0)
    parser.add_argument("--failure-max-score", type=float, default=95.0)
    parser.add_argument("--failure-strict-max-score", type=float, default=90.0)
    parser.add_argument("--min-failures-below-strict", type=int, default=3)
    parser.add_argument("--min-baseline-gap", type=float, default=5.0)
    parser.add_argument("--out-json", default="demo_artifacts/release_gate_report.json")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    headers = _headers(args.token)
    variants_dir = Path(args.variants_dir).resolve()
    gold_path = Path(args.gold).resolve()
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    requests.get(f"{base_url}/health", timeout=20).raise_for_status()

    baseline_gold_id = _upload_and_wait(
        base_url=base_url,
        headers=headers,
        path="/gold",
        task_id=args.task_id,
        file_path=gold_path,
        role="gold",
        timeout_sec=args.timeout_sec,
    )
    baseline_trainee_id = _upload_and_wait(
        base_url=base_url,
        headers=headers,
        path="/videos",
        task_id=args.task_id,
        file_path=gold_path,
        role="trainee",
        timeout_sec=args.timeout_sec,
    )
    baseline = _score_pair(
        base_url=base_url,
        headers=headers,
        gold_video_id=baseline_gold_id,
        trainee_video_id=baseline_trainee_id,
        timeout_sec=args.timeout_sec,
    )
    outcomes: list[ScoreOutcome] = [
        ScoreOutcome(
            name="gold_vs_gold",
            score_job_id=str(baseline["_score_job_id"]),
            score=float(baseline["score"]) if baseline.get("score") is not None else None,
            status=str(baseline.get("status")),
            error_message=baseline.get("error_message"),
            clip_count_gold=(baseline.get("result") or {}).get("clip_count", {}).get("gold"),
            clip_count_trainee=(baseline.get("result") or {}).get("clip_count", {}).get("trainee"),
        )
    ]

    variants = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    for variant in variants:
        trainee_file = variants_dir / f"{variant}.mp4"
        if not trainee_file.exists():
            raise FileNotFoundError(f"variant file not found: {trainee_file}")
        trainee_id = _upload_and_wait(
            base_url=base_url,
            headers=headers,
            path="/videos",
            task_id=args.task_id,
            file_path=trainee_file,
            role="trainee",
            timeout_sec=args.timeout_sec,
        )
        scored = _score_pair(
            base_url=base_url,
            headers=headers,
            gold_video_id=baseline_gold_id,
            trainee_video_id=trainee_id,
            timeout_sec=args.timeout_sec,
        )
        outcomes.append(
            ScoreOutcome(
                name=variant,
                score_job_id=str(scored["_score_job_id"]),
                score=float(scored["score"]) if scored.get("score") is not None else None,
                status=str(scored.get("status")),
                error_message=scored.get("error_message"),
                clip_count_gold=(scored.get("result") or {}).get("clip_count", {}).get("gold"),
                clip_count_trainee=(scored.get("result") or {}).get("clip_count", {}).get("trainee"),
            )
        )

    baseline_score = outcomes[0].score if outcomes[0].score is not None else 0.0
    failures = outcomes[1:]
    failure_scores = [x.score for x in failures if x.score is not None]
    strict_failures = [x for x in failures if x.score is not None and x.score <= args.failure_strict_max_score]
    median_failure = statistics.median(failure_scores) if failure_scores else 0.0
    gap = float(baseline_score - median_failure)

    gate_results = {
        "baseline_min_score": baseline_score >= float(args.baseline_min_score),
        "all_failures_below_max": all(
            (x.score is not None and x.score <= float(args.failure_max_score)) for x in failures
        ),
        "strict_failure_count_ok": len(strict_failures) >= int(args.min_failures_below_strict),
        "baseline_gap_ok": gap >= float(args.min_baseline_gap),
    }
    passed = all(bool(v) for v in gate_results.values())

    report = {
        "passed": passed,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_id": args.task_id,
        "baseline_score": baseline_score,
        "median_failure_score": median_failure,
        "baseline_gap": gap,
        "gate_results": gate_results,
        "config": {
            "baseline_min_score": float(args.baseline_min_score),
            "failure_max_score": float(args.failure_max_score),
            "failure_strict_max_score": float(args.failure_strict_max_score),
            "min_failures_below_strict": int(args.min_failures_below_strict),
            "min_baseline_gap": float(args.min_baseline_gap),
        },
        "outcomes": [
            {
                "name": x.name,
                "score_job_id": x.score_job_id,
                "status": x.status,
                "score": x.score,
                "error_message": x.error_message,
                "clip_count_gold": x.clip_count_gold,
                "clip_count_trainee": x.clip_count_trainee,
            }
            for x in outcomes
        ],
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
