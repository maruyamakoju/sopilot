from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from sopilot.utils import write_json as _write_json


@dataclass
class DistRuntime:
    rank: int
    world_size: int
    local_rank: int
    torch: Any | None
    dist: Any | None
    enabled: bool

    def all_reduce_np(self, value: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return value
        assert self.torch is not None and self.dist is not None
        tensor = self.torch.from_numpy(value)
        self.dist.all_reduce(tensor, op=self.dist.ReduceOp.SUM)
        return tensor.cpu().numpy()

    def all_reduce_float(self, value: float) -> float:
        if not self.enabled:
            return float(value)
        assert self.torch is not None and self.dist is not None
        tensor = self.torch.tensor([value], dtype=self.torch.float64)
        self.dist.all_reduce(tensor, op=self.dist.ReduceOp.SUM)
        return float(tensor.item())

    def broadcast_object(self, value: Any, src: int = 0) -> Any:
        if not self.enabled:
            return value
        assert self.dist is not None
        holder = [value]
        self.dist.broadcast_object_list(holder, src=src)
        return holder[0]

    def barrier(self) -> None:
        if self.enabled:
            assert self.dist is not None
            self.dist.barrier()

    def shutdown(self) -> None:
        if self.enabled:
            assert self.dist is not None
            self.dist.destroy_process_group()


def _init_dist_runtime() -> DistRuntime:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    try:
        import torch
        import torch.distributed as dist
    except Exception:
        torch = None
        dist = None

    if world_size <= 1:
        return DistRuntime(
            rank=rank,
            world_size=1,
            local_rank=local_rank,
            torch=torch,
            dist=dist,
            enabled=False,
        )

    if torch is None or dist is None:
        raise RuntimeError("WORLD_SIZE>1 requires torch distributed runtime")

    backend = "gloo"
    if torch.cuda.is_available() and os.name != "nt":
        backend = "nccl"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return DistRuntime(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        torch=torch,
        dist=dist,
        enabled=True,
    )


def _load_video_rows(db_path: Path, since: str, max_videos: int) -> list[dict]:
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        query = (
            "SELECT id, task_id, role, created_at, raw_embedding_path, embedding_path "
            "FROM videos "
            "WHERE ((raw_embedding_path IS NOT NULL AND raw_embedding_path != '') "
            "OR (embedding_path IS NOT NULL AND embedding_path != ''))"
        )
        params: list[Any] = []
        if since:
            query += " AND created_at > ?"
            params.append(since)
        query += " ORDER BY created_at ASC"
        if max_videos > 0:
            query += " LIMIT ?"
            params.append(max_videos)
        cur.execute(query, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _select_source_path(row: dict, mode: str) -> Path | None:
    raw = str(row.get("raw_embedding_path") or "").strip()
    eff = str(row.get("embedding_path") or "").strip()

    if mode == "raw":
        candidate = raw
    elif mode == "effective":
        candidate = eff
    else:
        candidate = raw or eff

    if not candidate:
        return None
    path = Path(candidate)
    if not path.exists():
        return None
    return path


def _infer_dim(candidates: list[dict]) -> int:
    for item in candidates:
        path = Path(item["path"])
        try:
            mat = np.load(path, mmap_mode="r")
        except Exception:
            continue
        if mat.ndim == 2 and int(mat.shape[0]) > 0 and int(mat.shape[1]) > 0:
            return int(mat.shape[1])
    return 0


def _write_skip_report(
    *,
    report_path: Path,
    reason: str,
    runtime: DistRuntime,
    total_videos: int,
    selected_videos: int,
    clips_used: int,
) -> None:
    if runtime.rank != 0:
        return
    report = {
        "status": "skipped",
        "reason": reason,
        "world_size": int(runtime.world_size),
        "total_videos": int(total_videos),
        "selected_videos": int(selected_videos),
        "clips_used": int(clips_used),
        "finished_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json(report_path, report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOPilot domain adapter trainer")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--since", default="")
    parser.add_argument("--embedding-source", default="auto", choices=["auto", "raw", "effective"])
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--min-clips", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runtime = _init_dist_runtime()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    db_path = data_dir / "sopilot.db"
    report_path = reports_dir / f"adapt_{args.job_id}.json"

    rows = _load_video_rows(db_path=db_path, since=args.since.strip(), max_videos=int(args.max_videos))
    selected: list[dict] = []
    for row in rows:
        source_path = _select_source_path(row, args.embedding_source)
        if source_path is None:
            continue
        selected.append(
            {
                "video_id": int(row["id"]),
                "task_id": row["task_id"],
                "created_at": row["created_at"],
                "path": str(source_path),
            }
        )

    if runtime.enabled:
        if runtime.rank == 0:
            payload = selected
        else:
            payload = None
        selected = runtime.broadcast_object(payload, src=0) or []

    if not selected:
        _write_skip_report(
            report_path=report_path,
            reason="no embedding artifacts",
            runtime=runtime,
            total_videos=len(rows),
            selected_videos=0,
            clips_used=0,
        )
        runtime.barrier()
        runtime.shutdown()
        return 0

    if runtime.rank == 0:
        expected_dim = _infer_dim(selected)
    else:
        expected_dim = 0
    expected_dim = int(runtime.broadcast_object(expected_dim, src=0))
    if expected_dim <= 0:
        _write_skip_report(
            report_path=report_path,
            reason="unable to infer embedding dimension",
            runtime=runtime,
            total_videos=len(rows),
            selected_videos=len(selected),
            clips_used=0,
        )
        runtime.barrier()
        runtime.shutdown()
        return 0

    local_sum = np.zeros((expected_dim,), dtype=np.float64)
    local_sq = np.zeros((expected_dim,), dtype=np.float64)
    local_clips = 0.0
    local_videos = 0.0

    for idx, item in enumerate(selected):
        if idx % runtime.world_size != runtime.rank:
            continue
        path = Path(item["path"])
        try:
            mat = np.load(path, mmap_mode="r")
        except Exception:
            continue
        if mat.ndim != 2:
            continue
        if int(mat.shape[0]) <= 0 or int(mat.shape[1]) != expected_dim:
            continue
        local_sum += mat.sum(axis=0, dtype=np.float64)
        local_sq += np.square(mat, dtype=np.float64).sum(axis=0)
        local_clips += float(mat.shape[0])
        local_videos += 1.0

    global_sum = runtime.all_reduce_np(local_sum)
    global_sq = runtime.all_reduce_np(local_sq)
    clips_used = runtime.all_reduce_float(local_clips)
    videos_used = runtime.all_reduce_float(local_videos)

    if clips_used < float(max(1, int(args.min_clips))):
        _write_skip_report(
            report_path=report_path,
            reason="insufficient clips",
            runtime=runtime,
            total_videos=len(rows),
            selected_videos=len(selected),
            clips_used=int(clips_used),
        )
        runtime.barrier()
        runtime.shutdown()
        return 0

    mean = global_sum / float(clips_used)
    var = np.maximum(global_sq / float(clips_used) - np.square(mean), float(args.eps))
    std = np.sqrt(var)

    if runtime.rank == 0:
        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        adapter_path = models_dir / f"feature_adapter_{args.job_id}.npz"
        np.savez(adapter_path, mean=mean.astype(np.float32), std=std.astype(np.float32))

        pointer = {
            "adapter_path": str(adapter_path),
            "job_id": args.job_id,
            "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "source": "torchrun_domain_adapter",
            "since": args.since.strip() or None,
            "world_size": int(runtime.world_size),
        }
        _write_json(models_dir / "current_adapter.json", pointer)

        report = {
            "status": "completed",
            "mode": "torchrun_domain_adapter",
            "job_id": args.job_id,
            "since": args.since.strip() or None,
            "world_size": int(runtime.world_size),
            "embedding_dim": int(expected_dim),
            "total_videos": int(len(rows)),
            "selected_videos": int(len(selected)),
            "videos_used": int(videos_used),
            "clips_used": int(clips_used),
            "adapter_path": str(adapter_path),
            "finished_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        }
        _write_json(report_path, report)

    runtime.barrier()
    runtime.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
