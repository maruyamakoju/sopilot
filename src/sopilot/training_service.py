from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import shlex
import subprocess
import threading

import numpy as np

from .config import Settings
from .db import Database
from .embedding_manager import EmbeddingManager
from .queueing import QueueManager
from .utils import write_json
from .vector_index import NpyVectorIndex

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        index: NpyVectorIndex,
        embedding_mgr: EmbeddingManager,
        queue: QueueManager | None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.index = index
        self.embedding_mgr = embedding_mgr
        self._queue = queue

        self._scheduler_stop = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        self._next_nightly_run: datetime | None = None

    def start_nightly_scheduler_if_enabled(self, enqueue_fn: object = None) -> None:
        if not self.settings.nightly_enabled:
            return
        now = datetime.now()
        self._next_nightly_run = self._compute_next_nightly_run(now)
        self._enqueue_fn = enqueue_fn or self.enqueue_training
        self._scheduler_thread = threading.Thread(
            target=self._nightly_loop,
            name="sopilot-nightly-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()

    def stop_scheduler(self) -> None:
        self._scheduler_stop.set()
        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=2.0)

    def enqueue_training(self, trigger: str = "manual", requested_by: str | None = None) -> dict:
        if self._queue is None:
            raise RuntimeError("queue manager is not available in this runtime mode")
        trigger = trigger.strip().lower()
        if trigger not in {"manual", "nightly"}:
            raise ValueError(f"unsupported training trigger: {trigger}")
        job_id = self.db.create_training_job(trigger=trigger, requested_by=requested_by)
        try:
            self._queue.enqueue_training(job_id)
        except Exception as exc:
            self.db.fail_training_job(job_id, f"enqueue failed: {exc}")
            raise
        return {"training_job_id": job_id, "status": "queued", "trigger": trigger}

    def run_training_job(self, job_id: str) -> None:
        self.db.mark_training_job_running(job_id)
        try:
            row = self.db.get_training_job(job_id)
            if row is None:
                raise ValueError(f"training job not found: {job_id}")

            trigger = str(row["trigger"])
            since = self.db.latest_training_finished_at()
            new_videos = self.db.count_videos_since(since)

            if trigger == "nightly" and new_videos < self.settings.nightly_min_new_videos:
                summary = {
                    "status": "skipped",
                    "trigger": trigger,
                    "reason": "not enough new videos",
                    "new_videos": int(new_videos),
                    "threshold": int(self.settings.nightly_min_new_videos),
                    "since": since,
                }
                summary_path = self.settings.reports_dir / f"training_{job_id}.json"
                write_json(summary_path, summary)
                self.db.complete_training_job(
                    job_id=job_id,
                    status="skipped",
                    summary_path=str(summary_path),
                    metrics_json=json.dumps(summary, ensure_ascii=True),
                )
                return

            videos = self.db.list_videos_for_training(since_created_at=since)
            if not videos:
                summary = {
                    "status": "skipped",
                    "trigger": trigger,
                    "reason": "no videos available for adaptation",
                    "since": since,
                }
                summary_path = self.settings.reports_dir / f"training_{job_id}.json"
                write_json(summary_path, summary)
                self.db.complete_training_job(
                    job_id=job_id,
                    status="skipped",
                    summary_path=str(summary_path),
                    metrics_json=json.dumps(summary, ensure_ascii=True),
                )
                return

            if self.settings.adapt_command:
                metrics = self._run_external_adaptation(job_id=job_id, since=since)
            elif self.settings.neural_mode:
                metrics = self._run_neural_training(job_id=job_id, videos=videos)
            else:
                metrics = self._run_builtin_feature_adaptation(job_id=job_id, videos=videos)

            metrics["trigger"] = trigger
            metrics["since"] = since
            metrics["new_videos"] = int(new_videos)
            summary_path = self.settings.reports_dir / f"training_{job_id}.json"
            write_json(summary_path, metrics)
            self.db.complete_training_job(
                job_id=job_id,
                status=metrics.get("status", "completed"),
                summary_path=str(summary_path),
                metrics_json=json.dumps(metrics, ensure_ascii=True),
            )
        except Exception as exc:
            self.db.fail_training_job(job_id, str(exc))
            logger.exception("Training job failed job_id=%s", job_id)
            raise

    def _run_external_adaptation(self, job_id: str, since: str | None) -> dict:
        command_template = self.settings.adapt_command.strip()
        if not command_template:
            raise ValueError("adapt command is empty")

        adapter_pointer_path = self.settings.models_dir / "current_adapter.json"
        before_pointer = None
        if adapter_pointer_path.exists():
            before_pointer = adapter_pointer_path.read_text(encoding="utf-8")

        command = command_template.format(
            job_id=job_id,
            data_dir=str(self.settings.data_dir),
            models_dir=str(self.settings.models_dir),
            reports_dir=str(self.settings.reports_dir),
            since=since or "",
        )
        try:
            args = shlex.split(command)
        except Exception as exc:
            raise ValueError(f"invalid adapt command: {exc}") from exc
        if not args:
            raise ValueError("adapt command resolved to empty")
        started = datetime.now(tz=timezone.utc)
        proc = subprocess.run(
            args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=max(1, self.settings.adapt_timeout_sec),
        )
        ended = datetime.now(tz=timezone.utc)
        if proc.returncode != 0:
            raise RuntimeError(
                f"adapt command failed rc={proc.returncode}: {proc.stderr[-800:]}"
            )

        refresh_stats = None
        after_pointer = None
        if adapter_pointer_path.exists():
            after_pointer = adapter_pointer_path.read_text(encoding="utf-8")
        if self.settings.enable_feature_adapter and after_pointer and after_pointer != before_pointer:
            self.embedding_mgr.invalidate_cache()
            refresh_stats = self._refresh_all_embeddings_and_reindex()

        payload = {
            "status": "completed",
            "mode": "external_command",
            "command": command,
            "return_code": int(proc.returncode),
            "duration_sec": float((ended - started).total_seconds()),
            "stdout_tail": proc.stdout[-4000:],
        }
        if refresh_stats is not None:
            payload["reindex"] = refresh_stats
        return payload

    def _run_builtin_feature_adaptation(self, job_id: str, videos: list[dict]) -> dict:
        sum_vec: np.ndarray | None = None
        sum_sq: np.ndarray | None = None
        clip_count = 0
        used_videos = 0
        dim: int | None = None

        for row in videos:
            source = row.get("raw_embedding_path") or row.get("embedding_path")
            if not source:
                continue
            path = Path(source)
            if not path.exists():
                continue
            mat = np.load(path).astype(np.float64)
            if mat.ndim != 2 or mat.shape[0] == 0:
                continue
            if dim is None:
                dim = int(mat.shape[1])
                sum_vec = np.zeros((dim,), dtype=np.float64)
                sum_sq = np.zeros((dim,), dtype=np.float64)
            elif int(mat.shape[1]) != dim:
                continue

            sum_vec += mat.sum(axis=0)
            sum_sq += np.square(mat).sum(axis=0)
            clip_count += int(mat.shape[0])
            used_videos += 1

        if dim is None or clip_count < 2 or sum_vec is None or sum_sq is None:
            return {
                "status": "skipped",
                "mode": "builtin_feature_adapter",
                "reason": "insufficient embeddings for adaptation",
                "videos_used": int(used_videos),
                "clips_used": int(clip_count),
            }

        mean = sum_vec / float(clip_count)
        var = np.maximum(sum_sq / float(clip_count) - np.square(mean), 1e-6)
        std = np.sqrt(var)
        adapter_path = self.settings.models_dir / f"feature_adapter_{job_id}.npz"
        np.savez(adapter_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
        adapter_pointer_path = self.settings.models_dir / "current_adapter.json"
        write_json(
            adapter_pointer_path,
            {
                "adapter_path": str(adapter_path),
                "job_id": job_id,
                "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            },
        )
        self.embedding_mgr.invalidate_cache()

        refresh_stats = self._refresh_all_embeddings_and_reindex()
        return {
            "status": "completed",
            "mode": "builtin_feature_adapter",
            "adapter_path": str(adapter_path),
            "videos_used": int(used_videos),
            "clips_used": int(clip_count),
            "embedding_dim": int(dim),
            "reindex": refresh_stats,
        }

    def _run_neural_training(self, job_id: str, videos: list[dict]) -> dict:
        """Run gradient-based neural training pipeline.

        Trains ProjectionHead, StepSegmenter, and ScoringHead using
        the multi-phase SOPilotTrainer.
        """
        from .nn.trainer import SOPilotTrainer, TrainingConfig
        from .step_engine import detect_step_boundaries, evaluate_sop, invalidate_neural_caches

        # Collect embeddings from gold videos
        embeddings_list: list[np.ndarray] = []
        boundaries_list: list[list[int]] = []
        used_videos = 0
        clip_count = 0

        for row in videos:
            source = row.get("raw_embedding_path") or row.get("embedding_path")
            if not source:
                continue
            path = Path(source)
            if not path.exists():
                continue
            mat = np.load(path).astype(np.float32)
            if mat.ndim != 2 or mat.shape[0] < 3:
                continue

            embeddings_list.append(mat)
            bounds = detect_step_boundaries(
                mat,
                self.settings.change_threshold_factor,
                self.settings.min_step_clips,
            )
            boundaries_list.append(bounds)
            clip_count += int(mat.shape[0])
            used_videos += 1

        if not embeddings_list:
            return {
                "status": "skipped",
                "mode": "neural",
                "reason": "no valid embeddings for neural training",
                "videos_used": 0,
                "clips_used": 0,
            }

        # Resolve device
        device = self.settings.neural_device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        d_in = embeddings_list[0].shape[1]
        config = TrainingConfig(
            device=device,
            proj_d_in=d_in,
            output_dir=self.settings.neural_model_dir,
        )
        trainer = SOPilotTrainer(config)

        # Phase 1a: Contrastive projection
        proj_log = trainer.train_projection_head(embeddings_list, boundaries_list)

        # Phase 1b: Step segmenter (MS-TCN++)
        seg_log = trainer.train_step_segmenter(embeddings_list, boundaries_list)

        # Phase 1c: ASFormer segmenter (transformer-based)
        if self.settings.neural_asformer_enabled:
            asformer_log = trainer.train_asformer(embeddings_list, boundaries_list)

        # Phase 2: Warm-start scoring from current formula
        metrics_list: list[dict] = []
        scores_list: list[float] = []

        # Generate training data by running evaluate_sop on all pairs
        for i in range(len(embeddings_list)):
            for j in range(len(embeddings_list)):
                if i == j:
                    continue
                try:
                    result = evaluate_sop(
                        gold_embeddings=embeddings_list[i],
                        trainee_embeddings=embeddings_list[j],
                        gold_meta=[{"start_sec": 0.0, "end_sec": 1.0, "clip_idx": k}
                                   for k in range(embeddings_list[i].shape[0])],
                        trainee_meta=[{"start_sec": 0.0, "end_sec": 1.0, "clip_idx": k}
                                      for k in range(embeddings_list[j].shape[0])],
                        threshold_factor=self.settings.change_threshold_factor,
                        min_step_clips=self.settings.min_step_clips,
                        low_similarity_threshold=self.settings.low_similarity_threshold,
                        w_miss=self.settings.w_miss,
                        w_swap=self.settings.w_swap,
                        w_dev=self.settings.w_dev,
                        w_time=self.settings.w_time,
                        w_warp=self.settings.w_warp,
                    )
                    metrics_list.append(result["metrics"])
                    scores_list.append(result["score"])
                except Exception:
                    continue

        score_log = None
        if len(metrics_list) >= 2:
            from .nn.scoring_head import METRIC_KEYS
            metrics_array = np.array(
                [[m.get(k, 0.0) for k in METRIC_KEYS] for m in metrics_list],
                dtype=np.float32,
            )
            scores_array = np.array(scores_list, dtype=np.float32)
            score_log = trainer.train_scoring_head(metrics_array, scores_array)

            # Phase 4: Calibration (use same data as warm-start for now)
            import torch
            scoring_head = trainer.scoring_head
            if scoring_head is not None:
                scoring_head.eval()
                with torch.no_grad():
                    x = torch.from_numpy(metrics_array).to(device)
                    predicted = scoring_head(x).squeeze(-1).cpu().numpy()
                trainer.calibrate(predicted, scores_array)

        # Save all models
        saved_paths = trainer.save_all()

        # Invalidate caches so new models are picked up
        self.embedding_mgr.invalidate_cache()
        invalidate_neural_caches()

        # Refresh embeddings with new neural adapter
        refresh_stats = self._refresh_all_embeddings_and_reindex()

        summary = trainer.training_summary()
        return {
            "status": "completed",
            "mode": "neural",
            "videos_used": used_videos,
            "clips_used": clip_count,
            "embedding_dim": d_in,
            "saved_paths": saved_paths,
            "training_summary": summary,
            "reindex": refresh_stats,
        }

    def _refresh_all_embeddings_and_reindex(self) -> dict:
        rows = self.db.list_videos_with_artifacts()
        old_version = self.index.current_version()
        staging_version = self.index.create_staging_version()

        videos_refreshed = 0
        clips_indexed = 0
        tasks_touched: set[str] = set()
        task_dims: dict[str, int] = {}
        embedder_name = f"{self.embedding_mgr.active_model_name()}+adapter"

        try:
            for row in rows:
                raw_source = row.get("raw_embedding_path") or row.get("embedding_path")
                emb_target = row.get("embedding_path")
                clip_meta_path = row.get("clip_meta_path")
                if not raw_source or not emb_target or not clip_meta_path:
                    continue

                raw_path = Path(raw_source)
                target_path = Path(emb_target)
                meta_path = Path(clip_meta_path)
                if not raw_path.exists() or not meta_path.exists():
                    continue

                raw_mat = np.load(raw_path).astype(np.float32)
                out = self.embedding_mgr.apply_adapter(raw_mat)
                np.save(target_path, out.astype(np.float32))
                self.db.update_video_embedding(
                    video_id=int(row["id"]),
                    embedding_path=str(target_path),
                    embedding_model=embedder_name,
                )

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                n = min(int(out.shape[0]), len(meta))
                if n <= 0:
                    continue

                index_rows = [
                    {
                        "video_id": int(row["id"]),
                        "clip_idx": int(meta[i]["clip_idx"]),
                        "start_sec": float(meta[i]["start_sec"]),
                        "end_sec": float(meta[i]["end_sec"]),
                        "role": row.get("role", "unknown"),
                    }
                    for i in range(n)
                ]
                task_id = str(row["task_id"])
                vec_dim = int(out.shape[1])
                existing_dim = task_dims.get(task_id)
                if existing_dim is None:
                    task_dims[task_id] = vec_dim
                elif existing_dim != vec_dim:
                    logger.warning(
                        "Skip reindex row due mixed embedding dims task_id=%s expected=%s got=%s video_id=%s",
                        task_id,
                        existing_dim,
                        vec_dim,
                        int(row["id"]),
                    )
                    continue
                self.index.add_to_version(
                    staging_version,
                    task_id=task_id,
                    vectors=out[:n],
                    metadata=index_rows,
                )
                tasks_touched.add(task_id)
                videos_refreshed += 1
                clips_indexed += n

            self.index.activate_version(staging_version)
        except Exception:
            self.index.delete_version(staging_version)
            raise

        return {
            "old_index_version": old_version,
            "new_index_version": staging_version,
            "videos_refreshed": int(videos_refreshed),
            "clips_indexed": int(clips_indexed),
            "tasks_touched": sorted(tasks_touched),
        }

    def get_training_job(self, training_job_id: str) -> dict | None:
        row = self.db.get_training_job(training_job_id)
        if row is None:
            return None

        payload = {
            "training_job_id": row["id"],
            "trigger": row["trigger"],
            "status": row["status"],
            "requested_by": row.get("requested_by"),
            "error_message": row["error_message"],
            "queued_at": row.get("queued_at"),
            "started_at": row.get("started_at"),
            "finished_at": row.get("finished_at"),
            "result": None,
        }
        summary_path = row.get("summary_path")
        if summary_path:
            p = Path(summary_path)
            if p.exists():
                payload["result"] = json.loads(p.read_text(encoding="utf-8"))
        return payload

    def get_nightly_status(self) -> dict:
        return {
            "enabled": bool(self.settings.nightly_enabled),
            "next_run_local": self._next_nightly_run.isoformat(timespec="seconds")
            if self._next_nightly_run is not None
            else None,
            "hour_local": int(self.settings.nightly_hour_local),
            "min_new_videos": int(self.settings.nightly_min_new_videos),
        }

    def _compute_next_nightly_run(self, now_local: datetime) -> datetime:
        hour = max(0, min(23, int(self.settings.nightly_hour_local)))
        target = now_local.replace(hour=hour, minute=0, second=0, microsecond=0)
        if target <= now_local:
            target += timedelta(days=1)
        return target

    def _nightly_loop(self) -> None:
        while not self._scheduler_stop.is_set():
            if self._next_nightly_run is None:
                self._next_nightly_run = self._compute_next_nightly_run(datetime.now())
            now = datetime.now()
            if now >= self._next_nightly_run:
                try:
                    if not self.db.has_active_training_job():
                        enqueue = getattr(self, "_enqueue_fn", self.enqueue_training)
                        enqueue(trigger="nightly", requested_by="system:nightly")
                except Exception:
                    logger.exception("Failed to enqueue nightly training job")
                self._next_nightly_run = self._compute_next_nightly_run(now + timedelta(seconds=1))
            self._scheduler_stop.wait(timeout=max(5, self.settings.nightly_check_interval_sec))
