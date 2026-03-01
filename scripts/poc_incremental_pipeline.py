from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sopilot.config import Settings
from sopilot.eval.harness import compute_poc_metrics
from sopilot.services.sopilot_service import SOPilotService

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


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "files": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = payload.get("files")
        if not isinstance(files, dict):
            return {"version": 1, "files": {}}
        return {"version": 1, "files": files}
    except Exception:
        return {"version": 1, "files": {}}


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_env(args: argparse.Namespace) -> None:
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


def build_service(settings: Settings):
    return common_build_service(settings)


def collect_embedder_runtime(service) -> dict[str, Any]:
    return common_collect_embedder_runtime(service)


def to_metrics_jobs(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in scored:
        if row.get("result") is None:
            continue
        items.append(
            {
                "id": int(row["job_id"]),
                "gold_video_id": int(row["gold_video_id"]),
                "trainee_video_id": int(row["trainee_video_id"]),
                "score": row["result"],
            }
        )
    return items


def load_existing_expected(labels_path: Path) -> dict[int, bool | None]:
    if not labels_path.exists():
        return {}
    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[int, bool | None] = {}
    for row in payload.get("jobs", []):
        try:
            if "critical_expected" not in row:
                continue
            expected = row.get("critical_expected")
            if expected is None:
                out[int(row["job_id"])] = None
            else:
                out[int(row["job_id"])] = bool(expected)
        except Exception:
            continue
    return out


def build_labels_template(
    task_id: str,
    completed_jobs: list[dict[str, Any]],
    existing_expected: dict[int, bool | None] | None = None,
) -> dict[str, Any]:
    return common_build_labels_template(task_id, completed_jobs, existing_expected=existing_expected)


def build_video_path_map(manifest: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        return out
    for path, meta in files.items():
        if not isinstance(meta, dict):
            continue
        video_id = meta.get("video_id")
        if video_id is None:
            continue
        try:
            out[int(video_id)] = str(path)
        except Exception:
            continue
    return out


def choose_gold_id(service: SOPilotService, args: argparse.Namespace, new_gold_ids: list[int]) -> int:
    if args.gold_id is not None:
        return int(args.gold_id)
    if new_gold_ids:
        return int(new_gold_ids[-1])
    golds = service.list_videos(site_id=args.site_id, is_gold=True, limit=10000)
    if not golds:
        raise SystemExit("no gold videos available")
    return int(golds[0]["video_id"])


def select_trainee_ids(
    service: SOPilotService,
    *,
    task_id: str,
    site_id: str | None,
    gold_id: int,
    score_scope: str,
    new_trainee_ids: list[int],
) -> list[int]:
    if score_scope == "new":
        return list(new_trainee_ids)

    trainees = service.list_videos(site_id=site_id, is_gold=False, limit=100000)
    ready_ids = [int(item["video_id"]) for item in trainees if item["status"] == "ready"]
    if score_scope == "all":
        return ready_ids

    # unscored: score only pairs not completed yet for selected gold
    completed = service.database.list_completed_score_jobs(task_id=task_id)
    seen_pairs = {(int(job["gold_video_id"]), int(job["trainee_video_id"])) for job in completed}
    return [trainee_id for trainee_id in ready_ids if (gold_id, trainee_id) not in seen_pairs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental PoC ingest+score pipeline.")
    parser.add_argument("--base-dir", required=True)
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
    parser.add_argument("--embedder-backend", choices=["vjepa2", "color-motion"], default="color-motion")
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
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--gold-id", type=int, default=None)
    parser.add_argument("--score-scope", choices=["new", "all", "unscored"], default="unscored")
    parser.add_argument("--labels-output", default=None)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    configure_env(args)
    settings = Settings.from_env()
    service = build_service(settings)

    manifest_path = (
        Path(args.manifest_path).resolve()
        if args.manifest_path
        else (Path(args.data_dir).resolve() / "ingest_manifest.json")
    )
    manifest = load_manifest(manifest_path)

    base_dir = Path(args.base_dir).resolve()
    files = iter_videos(base_dir, recursive=args.recursive)
    if args.max_files is not None:
        files = files[: max(1, args.max_files)]
    selected_input_manifest = [
        {
            "path": str(path.resolve()),
            "kind": classify(path, args.mode),
            **file_fingerprint(path),
        }
        for path in files
    ]

    new_gold_ids: list[int] = []
    new_trainee_ids: list[int] = []
    ingested: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for idx, path in enumerate(files, start=1):
        abs_path = str(path.resolve())
        kind = classify(path, args.mode)
        fp = file_fingerprint(path)
        prev = manifest["files"].get(abs_path)
        if prev and prev.get("size") == fp["size"] and prev.get("mtime_ns") == fp["mtime_ns"]:
            skipped.append({"path": abs_path, "kind": kind, "reason": "unchanged"})
            continue

        print(f"[{idx}/{len(files)}] ingest {kind}: {path.name}")
        with path.open("rb") as fh:
            try:
                result = service.ingest_video(
                    original_filename=path.name,
                    file_obj=fh,
                    task_id=args.task_id,
                    site_id=args.site_id,
                    camera_id=args.camera_id,
                    operator_id_hash=None,
                    recorded_at=args.recorded_at,
                    is_gold=(kind == "gold"),
                )
                video_id = int(result["video_id"])
                manifest["files"][abs_path] = {
                    **fp,
                    "kind": kind,
                    "video_id": video_id,
                }
                ingested.append(
                    {
                        "path": abs_path,
                        "kind": kind,
                        "video_id": video_id,
                        "clip_count": int(result["clip_count"]),
                    }
                )
                if kind == "gold":
                    new_gold_ids.append(video_id)
                else:
                    new_trainee_ids.append(video_id)
                print(f"  -> OK video_id={video_id} clips={result['clip_count']}")
            except Exception as exc:
                skipped.append({"path": abs_path, "kind": kind, "reason": f"ingest_failed: {exc}"})
                print(f"  -> FAIL {exc}")

    save_manifest(manifest_path, manifest)
    gold_id = choose_gold_id(service, args, new_gold_ids)
    trainee_ids = select_trainee_ids(
        service,
        task_id=args.task_id,
        site_id=args.site_id,
        gold_id=gold_id,
        score_scope=args.score_scope,
        new_trainee_ids=new_trainee_ids,
    )

    scored_ok: list[dict[str, Any]] = []
    scored_fail: list[dict[str, Any]] = []
    for trainee_id in trainee_ids:
        try:
            job = service.queue_score_job(gold_video_id=gold_id, trainee_video_id=trainee_id)
            job_id = int(job["job_id"])
            service.run_score_job(job_id)
            final = service.get_score_job(job_id)
            if final["status"] == "completed":
                score = final["result"].get("score") if final.get("result") else None
                print(f"[score] OK job_id={job_id} trainee_id={trainee_id} score={score}")
                scored_ok.append(
                    {
                        "job_id": job_id,
                        "gold_video_id": gold_id,
                        "trainee_video_id": trainee_id,
                        "result": final["result"],
                    }
                )
            else:
                print(f"[score] FAIL job_id={job_id} trainee_id={trainee_id} error={final.get('error')}")
                scored_fail.append(
                    {
                        "job_id": job_id,
                        "gold_video_id": gold_id,
                        "trainee_video_id": trainee_id,
                        "error": final.get("error"),
                    }
                )
        except Exception as exc:
            print(f"[score] FAIL trainee_id={trainee_id} error={exc}")
            scored_fail.append(
                {
                    "job_id": None,
                    "gold_video_id": gold_id,
                    "trainee_video_id": trainee_id,
                    "error": str(exc),
                }
            )

    completed_all = service.database.list_completed_score_jobs(task_id=args.task_id)
    metrics_all = compute_poc_metrics(completed_all, critical_labels=None)
    metrics_all["task_id"] = args.task_id
    metrics_new = compute_poc_metrics(to_metrics_jobs(scored_ok), critical_labels=None)
    metrics_new["task_id"] = args.task_id

    if args.labels_output:
        labels_path = Path(args.labels_output).resolve()
        existing_expected = load_existing_expected(labels_path)
        labels_payload = build_labels_template(args.task_id, completed_all, existing_expected=existing_expected)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text(json.dumps(labels_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"labels: {labels_path}")

    if args.eval_output:
        eval_path = Path(args.eval_output).resolve()
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps(metrics_all, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"eval: {eval_path}")

    summary = {
        "task_id": args.task_id,
        "base_dir": str(base_dir),
        "seed": int(args.seed),
        "allow_embedder_fallback": bool(args.allow_embedder_fallback),
        "embedder_runtime": collect_embedder_runtime(service),
        "selected_input_manifest": selected_input_manifest,
        "manifest_path": str(manifest_path),
        "ingested_count": len(ingested),
        "skipped_count": len(skipped),
        "new_gold_ids": new_gold_ids,
        "new_trainee_ids": new_trainee_ids,
        "gold_id_used": gold_id,
        "scored_count": len(scored_ok),
        "score_failed_count": len(scored_fail),
        "metrics_new": metrics_new,
        "metrics_all": metrics_all,
        "video_path_by_id": build_video_path_map(manifest),
        "all_score_jobs": [
            {
                "job_id": int(job["id"]),
                "gold_video_id": int(job["gold_video_id"]),
                "trainee_video_id": int(job["trainee_video_id"]),
            }
            for job in completed_all
        ],
        "ingested": ingested,
        "skipped": skipped,
        "scores_completed": scored_ok,
        "scores_failed": scored_fail,
    }

    output_path = (
        Path(args.output).resolve()
        if args.output
        else (Path(args.data_dir).resolve() / "incremental_summary.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {output_path}")


if __name__ == "__main__":
    main()
