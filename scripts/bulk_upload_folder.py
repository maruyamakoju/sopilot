from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
from typing import Literal

import httpx

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def classify(path: Path, mode: Literal["gold", "trainee", "auto"]) -> Literal["gold", "trainee"]:
    if mode != "auto":
        return mode

    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    if "gold" in parts or name.startswith(("gold_", "g_")):
        return "gold"
    if "trainee" in parts or "train" in parts or name.startswith(("trainee_", "t_")):
        return "trainee"
    return "trainee"


def iter_videos(base_dir: Path, recursive: bool) -> list[Path]:
    if not base_dir.exists():
        raise SystemExit(f"base-dir not found: {base_dir}")
    globber = base_dir.rglob("*") if recursive else base_dir.glob("*")
    files = [path for path in globber if path.is_file() and path.suffix.lower() in VIDEO_EXTS]
    return sorted(files)


def _safe_json(response: httpx.Response) -> dict:
    try:
        return response.json()
    except Exception:
        return {"detail": response.text[:400]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk upload PoC videos (gold/trainee) to SOPilot API.")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="Base URL of SOPilot API")
    parser.add_argument("--base-dir", required=True, help="Folder containing videos")
    parser.add_argument("--task-id", required=True, help="task_id (must match primary task if enforced)")
    parser.add_argument("--site-id", default=None)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--recorded-at", default=None)
    parser.add_argument("--mode", choices=["auto", "gold", "trainee"], default="auto")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=None, help="Write detailed result JSON")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    files = iter_videos(base_dir, recursive=args.recursive)
    if args.max_files is not None:
        files = files[: max(1, args.max_files)]

    if not files:
        raise SystemExit("no video files found")

    if args.dry_run:
        for path in files:
            kind = classify(path, args.mode)
            print(f"[DRY] {kind}\t{path}")
        return

    api = args.api.rstrip("/")
    data_base = {
        "task_id": args.task_id,
        "site_id": args.site_id,
        "camera_id": args.camera_id,
        "recorded_at": args.recorded_at,
    }
    data_base = {k: v for k, v in data_base.items() if v is not None}

    timeout = httpx.Timeout(1200.0, connect=10.0)
    results: list[dict] = []
    success = 0
    failed = 0

    with httpx.Client(timeout=timeout) as client:
        for idx, path in enumerate(files, start=1):
            kind = classify(path, args.mode)
            endpoint = "/gold" if kind == "gold" else "/videos"
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

            print(f"[{idx}/{len(files)}] upload {kind}: {path.name}")
            with path.open("rb") as fh:
                response = client.post(
                    f"{api}{endpoint}",
                    data=data_base,
                    files={"file": (path.name, fh, mime)},
                )

            payload = _safe_json(response)
            if response.status_code != 200:
                failed += 1
                print(f"  -> FAIL {response.status_code}: {str(payload)[:300]}")
                results.append(
                    {
                        "path": str(path),
                        "kind": kind,
                        "ok": False,
                        "status_code": response.status_code,
                        "error": payload,
                    }
                )
                continue

            success += 1
            result = {
                "path": str(path),
                "kind": kind,
                "ok": True,
                "status_code": response.status_code,
                "video_id": payload.get("video_id"),
                "clip_count": payload.get("clip_count"),
                "status": payload.get("status"),
                "task_id": payload.get("task_id"),
            }
            print(
                f"  -> OK video_id={result['video_id']} clips={result['clip_count']} status={result['status']}"
            )
            results.append(result)

    summary = {
        "total": len(files),
        "success": success,
        "failed": failed,
        "success_rate": round(100.0 * success / len(files), 1),
    }
    print(f"summary: {summary}")

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
