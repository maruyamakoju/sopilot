from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def _poll(base_url: str, headers: dict[str, str], path: str, done: set[str], timeout_sec: int) -> dict:
    started = time.time()
    while time.time() - started <= timeout_sec:
        res = requests.get(f"{base_url}{path}", headers=headers, timeout=30)
        res.raise_for_status()
        payload = res.json()
        if payload.get("status") in done:
            return payload
        time.sleep(1.0)
    raise TimeoutError(f"timeout while polling {path}")


def _safe_json_request(method: str, url: str, *, headers: dict[str, str], timeout: int = 30) -> tuple[int, dict | None]:
    res = requests.request(method=method, url=url, headers=headers, timeout=timeout)
    status = int(res.status_code)
    try:
        payload = res.json()
    except Exception:
        payload = None
    return status, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SOPilot demo flow end-to-end")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="", help="Bearer token value")
    parser.add_argument("--task-id", default="maintenance_filter_swap")
    parser.add_argument("--gold", default="demo_videos/maintenance_filter_swap/gold.mp4")
    parser.add_argument("--trainee", default="demo_videos/maintenance_filter_swap/missing.mp4")
    parser.add_argument("--out-dir", default="demo_artifacts")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--audit-export-limit", type=int, default=100)
    parser.add_argument(
        "--skip-signed-audit-export",
        action="store_true",
        help="Skip /audit/export call",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    headers: dict[str, str] = {}
    if args.token.strip():
        headers["Authorization"] = f"Bearer {args.token.strip()}"

    requests.get(f"{base_url}/health", timeout=20).raise_for_status()

    with Path(args.gold).open("rb") as f:
        res = requests.post(
            f"{base_url}/gold",
            headers=headers,
            data={"task_id": args.task_id},
            files={"file": (Path(args.gold).name, f, "video/mp4")},
            timeout=120,
        )
    res.raise_for_status()
    gold_job = res.json()["ingest_job_id"]
    gold_done = _poll(base_url, headers, f"/videos/jobs/{gold_job}", {"completed", "failed"}, args.timeout_sec)
    if gold_done["status"] != "completed":
        raise RuntimeError(f"gold ingest failed: {gold_done}")

    with Path(args.trainee).open("rb") as f:
        res = requests.post(
            f"{base_url}/videos",
            headers=headers,
            data={"task_id": args.task_id, "role": "trainee"},
            files={"file": (Path(args.trainee).name, f, "video/mp4")},
            timeout=120,
        )
    res.raise_for_status()
    trainee_job = res.json()["ingest_job_id"]
    trainee_done = _poll(base_url, headers, f"/videos/jobs/{trainee_job}", {"completed", "failed"}, args.timeout_sec)
    if trainee_done["status"] != "completed":
        raise RuntimeError(f"trainee ingest failed: {trainee_done}")

    score_req_headers = dict(headers)
    score_req_headers["Content-Type"] = "application/json"
    res = requests.post(
        f"{base_url}/score",
        headers=score_req_headers,
        json={
            "gold_video_id": int(gold_done["video_id"]),
            "trainee_video_id": int(trainee_done["video_id"]),
        },
        timeout=60,
    )
    res.raise_for_status()
    score_job_id = res.json()["score_job_id"]
    score_done = _poll(base_url, headers, f"/score/{score_job_id}", {"completed", "failed"}, args.timeout_sec)
    if score_done["status"] != "completed":
        raise RuntimeError(f"score failed: {score_done}")

    pdf = requests.get(f"{base_url}/score/{score_job_id}/report.pdf", headers=headers, timeout=120)
    pdf.raise_for_status()
    pdf_path = out_dir / f"score_{score_job_id}.pdf"
    pdf_path.write_bytes(pdf.content)

    audit = requests.get(f"{base_url}/audit/trail?limit=30", headers=headers, timeout=30)
    audit.raise_for_status()
    audit_path = out_dir / "audit_trail_latest.json"
    audit_path.write_text(json.dumps(audit.json(), ensure_ascii=False, indent=2), encoding="utf-8")

    queue_status, queue_payload = _safe_json_request(
        "GET",
        f"{base_url}/ops/queue",
        headers=headers,
        timeout=30,
    )
    queue_path = out_dir / "queue_metrics_latest.json"
    if queue_payload is not None:
        queue_path.write_text(json.dumps(queue_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    signed_audit = None
    signed_audit_file = None
    if not args.skip_signed_audit_export:
        export_status, export_payload = _safe_json_request(
            "POST",
            f"{base_url}/audit/export?limit={max(1, int(args.audit_export_limit))}",
            headers=headers,
            timeout=60,
        )
        if export_status == 200 and isinstance(export_payload, dict):
            signed_audit = export_payload
            export_id = str(export_payload.get("export_id", "")).strip()
            if export_id:
                file_res = requests.get(
                    f"{base_url}/audit/export/{export_id}/file",
                    headers=headers,
                    timeout=60,
                )
                if file_res.status_code == 200:
                    signed_audit_file = out_dir / f"audit_export_{export_id}.json"
                    signed_audit_file.write_bytes(file_res.content)

    score_path = out_dir / f"score_{score_job_id}.json"
    score_path.write_text(json.dumps(score_done, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "task_id": args.task_id,
        "gold_video_id": int(gold_done["video_id"]),
        "trainee_video_id": int(trainee_done["video_id"]),
        "score_job_id": score_job_id,
        "score": score_done.get("score"),
        "requested_by": score_done.get("requested_by"),
        "pdf": str(pdf_path),
        "score_json": str(score_path),
        "audit_json": str(audit_path),
        "queue_metrics_status": queue_status,
        "queue_metrics_json": str(queue_path) if queue_path.exists() else None,
        "signed_audit_export": signed_audit,
        "signed_audit_export_file": str(signed_audit_file) if signed_audit_file else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
