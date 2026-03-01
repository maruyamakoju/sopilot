from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sopilot.config import Settings
from sopilot.eval.harness import compute_poc_metrics

try:
    from scripts.poc_pipeline_common import (
        build_labels_template as common_build_labels_template,
    )
    from scripts.poc_pipeline_common import (
        build_service as common_build_service,
    )
    from scripts.poc_pipeline_common import (
        classify as common_classify,
    )
    from scripts.poc_pipeline_common import (
        collect_embedder_runtime as common_collect_embedder_runtime,
    )
    from scripts.poc_pipeline_common import (
        configure_env as common_configure_env,
    )
    from scripts.poc_pipeline_common import (
        iter_videos as common_iter_videos,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution path
    from poc_pipeline_common import (
        build_labels_template as common_build_labels_template,
    )
    from poc_pipeline_common import (
        build_service as common_build_service,
    )
    from poc_pipeline_common import (
        classify as common_classify,
    )
    from poc_pipeline_common import (
        collect_embedder_runtime as common_collect_embedder_runtime,
    )
    from poc_pipeline_common import (
        configure_env as common_configure_env,
    )
    from poc_pipeline_common import (
        iter_videos as common_iter_videos,
    )


def classify(path: Path, mode: str) -> str:
    return common_classify(path, mode)


def iter_videos(base_dir: Path, recursive: bool) -> list[Path]:
    return common_iter_videos(base_dir, recursive)


def _configure_env(args: argparse.Namespace) -> None:
    common_configure_env(
        data_dir=args.data_dir,
        task_id=args.task_id,
        task_name=args.task_name,
        embedder_backend=args.embedder_backend,
        allow_embedder_fallback=bool(args.allow_embedder_fallback),
        run_id=args.run_id,
        run_seed=args.seed,
        vjepa2_pooling=args.vjepa2_pooling,
    )


def _build_service(settings: Settings):
    return common_build_service(settings)


def _collect_embedder_runtime(service) -> dict[str, Any]:
    return common_collect_embedder_runtime(service)


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
    }


def _build_labels_template(task_id: str, done_jobs: list[dict[str, Any]]) -> dict[str, Any]:
    completed_jobs = []
    for item in done_jobs:
        if item.get("result") is None:
            continue
        completed_jobs.append({"id": int(item["job_id"]), "score": item["result"]})
    return common_build_labels_template(task_id, completed_jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PoC ingest+score locally without starting API server.")
    parser.add_argument("--base-dir", required=True, help="Folder with gold/trainee videos")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-name", default="PoC Task")
    parser.add_argument("--run-id", default=None, help="Optional run identifier for reproducibility metadata")
    parser.add_argument("--site-id", default=None)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--recorded-at", default=None)
    parser.add_argument("--mode", choices=["auto", "gold", "trainee"], default="auto")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Run seed for reproducibility metadata")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--embedder-backend", choices=["vjepa2", "color-motion"], default="vjepa2")
    parser.add_argument(
        "--vjepa2-pooling",
        choices=["mean_tokens", "first_token", "flatten"],
        default="mean_tokens",
        help="Pooling strategy when embedder-backend=vjepa2.",
    )
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument(
        "--allow-embedder-fallback",
        dest="allow_embedder_fallback",
        action="store_true",
        help="Allow fallback embedder when primary backend fails (default).",
    )
    fallback_group.add_argument(
        "--disable-embedder-fallback",
        dest="allow_embedder_fallback",
        action="store_false",
        help="Disable fallback; fail fast on primary embedder errors.",
    )
    parser.set_defaults(allow_embedder_fallback=True)
    parser.add_argument(
        "--score-scope",
        choices=["uploaded", "all"],
        default="uploaded",
        help="uploaded: score only trainee videos uploaded in this run / all: score all trainees in DB",
    )
    parser.add_argument("--gold-id", type=int, default=None, help="Use explicit gold video id for scoring")
    parser.add_argument("--labels-output", default=None, help="Optional labels template JSON path")
    parser.add_argument("--eval-output", default=None, help="Optional metrics JSON path")
    parser.add_argument("--output", default="data/local_pipeline_summary.json")
    args = parser.parse_args()

    _configure_env(args)
    settings = Settings.from_env()
    service = _build_service(settings)

    files = iter_videos(Path(args.base_dir), recursive=args.recursive)
    if args.max_files is not None:
        files = files[: max(1, args.max_files)]
    if not files:
        raise SystemExit("no videos found")

    input_manifest: list[dict[str, Any]] = []
    for path in files:
        kind = classify(path, args.mode)
        input_manifest.append(
            {
                "path": str(path.resolve()),
                "kind": kind,
                **_file_fingerprint(path),
            }
        )

    upload_results: list[dict[str, Any]] = []
    uploaded_gold_ids: list[int] = []
    uploaded_trainee_ids: list[int] = []

    for idx, item in enumerate(input_manifest, start=1):
        path = Path(item["path"])
        kind = str(item["kind"])
        print(f"[{idx}/{len(files)}] ingest {kind}: {path.name}")
        with path.open("rb") as fh:
            try:
                payload = service.ingest_video(
                    original_filename=path.name,
                    file_obj=fh,
                    task_id=args.task_id,
                    site_id=args.site_id,
                    camera_id=args.camera_id,
                    operator_id_hash=None,
                    recorded_at=args.recorded_at,
                    is_gold=(kind == "gold"),
                )
                video_id = int(payload["video_id"])
                if kind == "gold":
                    uploaded_gold_ids.append(video_id)
                else:
                    uploaded_trainee_ids.append(video_id)
                print(f"  -> OK video_id={video_id} clips={payload['clip_count']}")
                upload_results.append(
                    {
                        "path": str(path.resolve()),
                        "kind": kind,
                        "ok": True,
                        "video_id": video_id,
                        "clip_count": int(payload["clip_count"]),
                    }
                )
            except Exception as exc:
                print(f"  -> FAIL {exc}")
                upload_results.append(
                    {
                        "path": str(path.resolve()),
                        "kind": kind,
                        "ok": False,
                        "error": str(exc),
                    }
                )

    if args.gold_id is not None:
        gold_id = int(args.gold_id)
    elif uploaded_gold_ids:
        gold_id = uploaded_gold_ids[-1]
    else:
        golds = service.list_videos(site_id=args.site_id, is_gold=True, limit=1000)
        if not golds:
            raise SystemExit("no gold video available for scoring")
        gold_id = int(golds[0]["video_id"])

    if args.score_scope == "uploaded":
        trainee_ids = list(uploaded_trainee_ids)
    else:
        trainees = service.list_videos(site_id=args.site_id, is_gold=False, limit=100000)
        trainee_ids = [int(v["video_id"]) for v in trainees if v["status"] == "ready"]

    done_jobs: list[dict[str, Any]] = []
    failed_jobs: list[dict[str, Any]] = []
    for trainee_id in trainee_ids:
        try:
            job = service.queue_score_job(gold_video_id=gold_id, trainee_video_id=trainee_id)
            job_id = int(job["job_id"])
            service.run_score_job(job_id)
            finished = service.get_score_job(job_id)
            status = finished["status"]
            if status == "completed":
                result = finished["result"]
                print(f"[score] OK job_id={job_id} trainee_id={trainee_id} score={result.get('score')}")
                done_jobs.append(
                    {
                        "job_id": job_id,
                        "gold_video_id": gold_id,
                        "trainee_video_id": trainee_id,
                        "status": status,
                        "result": result,
                    }
                )
            else:
                print(f"[score] FAIL job_id={job_id} trainee_id={trainee_id} error={finished.get('error')}")
                failed_jobs.append(
                    {
                        "job_id": job_id,
                        "gold_video_id": gold_id,
                        "trainee_video_id": trainee_id,
                        "status": status,
                        "error": finished.get("error"),
                    }
                )
        except Exception as exc:
            print(f"[score] FAIL trainee_id={trainee_id} error={exc}")
            failed_jobs.append(
                {
                    "job_id": None,
                    "gold_video_id": gold_id,
                    "trainee_video_id": trainee_id,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    metrics_jobs = [
        {
            "id": int(item["job_id"]),
            "gold_video_id": int(item["gold_video_id"]),
            "trainee_video_id": int(item["trainee_video_id"]),
            "score": item["result"],
        }
        for item in done_jobs
    ]
    metrics = compute_poc_metrics(metrics_jobs, critical_labels=None)
    metrics["task_id"] = args.task_id

    if args.labels_output:
        labels_payload = _build_labels_template(args.task_id, done_jobs)
        labels_path = Path(args.labels_output).resolve()
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text(json.dumps(labels_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"labels template: {labels_path}")

    if args.eval_output:
        eval_path = Path(args.eval_output).resolve()
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"metrics: {eval_path}")

    summary = {
        "task_id": args.task_id,
        "base_dir": str(Path(args.base_dir).resolve()),
        "seed": int(args.seed),
        "allow_embedder_fallback": bool(args.allow_embedder_fallback),
        "embedder_runtime": _collect_embedder_runtime(service),
        "input_manifest": input_manifest,
        "uploaded_total": len(upload_results),
        "uploaded_success": sum(1 for item in upload_results if item["ok"]),
        "uploaded_failed": sum(1 for item in upload_results if not item["ok"]),
        "gold_id_used": gold_id,
        "scored_total": len(trainee_ids),
        "scored_completed": len(done_jobs),
        "scored_failed": len(failed_jobs),
        "metrics": metrics,
        "uploads": upload_results,
        "scores_completed": done_jobs,
        "scores_failed": failed_jobs,
    }
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {out_path}")


if __name__ == "__main__":
    main()
