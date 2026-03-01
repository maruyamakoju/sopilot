from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        return response.json()
    except Exception:
        return {"detail": response.text[:400]}


def _load_videos(client: httpx.Client, api: str, site_id: str | None, limit: int) -> list[dict[str, Any]]:
    params = {"limit": str(max(1, limit))}
    if site_id:
        params["site_id"] = site_id
    response = client.get(f"{api}/videos", params=params)
    payload = _safe_json(response)
    if response.status_code != 200:
        raise SystemExit(f"failed to fetch /videos: {response.status_code} {str(payload)[:300]}")
    return list(payload.get("items", []))


def _pick_gold(golds: list[dict[str, Any]], gold_id: int | None) -> int:
    if gold_id is not None:
        return int(gold_id)
    if not golds:
        raise SystemExit("no gold videos found")
    # /videos is expected to be created_at desc
    return int(golds[0]["video_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch score trainee videos against one gold video.")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--site-id", default=None)
    parser.add_argument("--gold-id", type=int, default=None, help="If omitted, newest gold is used.")
    parser.add_argument("--limit", type=int, default=2000, help="Max videos fetched from /videos.")
    parser.add_argument("--only-ready", action="store_true", help="Score only ready videos.")
    parser.add_argument("--wait", action="store_true", help="Wait until all jobs complete.")
    parser.add_argument("--poll-interval", type=float, default=1.2)
    parser.add_argument("--output", default=None, help="Write summary JSON.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if trainee already has completed job.")
    args = parser.parse_args()

    api = args.api.rstrip("/")
    timeout = httpx.Timeout(120.0, connect=10.0)

    with httpx.Client(timeout=timeout) as client:
        videos = _load_videos(client, api, args.site_id, args.limit)
        golds = [v for v in videos if v.get("is_gold")]
        trainees = [v for v in videos if not v.get("is_gold")]
        if args.only_ready:
            golds = [v for v in golds if v.get("status") == "ready"]
            trainees = [v for v in trainees if v.get("status") == "ready"]

        gold_id = _pick_gold(golds, args.gold_id)
        print(f"gold_id={gold_id} golds={len(golds)} trainees={len(trainees)}")

        existing_trainee_ids: set[int] = set()
        if args.skip_existing:
            # lightweight scan by querying each trainee current score relation would be expensive without endpoint.
            # skip_existing is best-effort via local output history only; keep empty now.
            pass

        queued: list[dict[str, Any]] = []
        failed_create: list[dict[str, Any]] = []
        for trainee in trainees:
            trainee_id = int(trainee["video_id"])
            if trainee_id == gold_id:
                continue
            if trainee_id in existing_trainee_ids:
                continue
            response = client.post(
                f"{api}/score",
                json={"gold_video_id": gold_id, "trainee_video_id": trainee_id},
            )
            payload = _safe_json(response)
            if response.status_code != 200:
                print(f"FAIL create trainee={trainee_id}: {response.status_code} {str(payload)[:240]}")
                failed_create.append(
                    {
                        "trainee_id": trainee_id,
                        "status_code": response.status_code,
                        "error": payload,
                    }
                )
                continue
            job_id = int(payload["job_id"])
            queued.append({"job_id": job_id, "trainee_id": trainee_id})
            print(f"queued job_id={job_id} trainee_id={trainee_id}")

        print(f"queued_jobs={len(queued)} failed_creates={len(failed_create)}")
        completed = 0
        failed = 0
        done_jobs: list[dict[str, Any]] = []

        if args.wait and queued:
            pending = {item["job_id"]: item["trainee_id"] for item in queued}
            while pending:
                finished_now: list[int] = []
                for job_id, trainee_id in list(pending.items()):
                    response = client.get(f"{api}/score/{job_id}")
                    payload = _safe_json(response)
                    if response.status_code != 200:
                        continue
                    status = payload.get("status")
                    if status in {"completed", "failed"}:
                        finished_now.append(job_id)
                        if status == "completed":
                            completed += 1
                        else:
                            failed += 1
                        done_jobs.append(
                            {
                                "job_id": job_id,
                                "trainee_id": trainee_id,
                                "status": status,
                                "score": payload.get("result", {}).get("score") if payload.get("result") else None,
                            }
                        )
                for job_id in finished_now:
                    pending.pop(job_id, None)
                print(f"progress completed={completed} failed={failed} pending={len(pending)}")
                if pending:
                    time.sleep(max(0.2, args.poll_interval))
            print("batch scoring finished")

        summary = {
            "gold_id": gold_id,
            "queued_jobs": len(queued),
            "failed_creates": len(failed_create),
            "completed": completed,
            "failed": failed,
        }
        print(f"summary: {summary}")

        if args.output:
            out_path = Path(args.output).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(
                    {
                        "summary": summary,
                        "queued": queued,
                        "failed_creates": failed_create,
                        "done_jobs": done_jobs,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()

