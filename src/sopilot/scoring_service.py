from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .config import Settings
from .db import Database
from .embedding_manager import EmbeddingManager
from .queueing import QueueManager
from .step_engine import evaluate_sop
from .utils import write_json
from .vector_index import NpyVectorIndex

logger = logging.getLogger(__name__)


class ScoringService:
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

    def _load_video_artifacts(self, video_id: int) -> tuple[dict, np.ndarray, list[dict]]:
        video = self.db.get_video(video_id)
        if video is None:
            raise ValueError(f"video not found: {video_id}")
        emb_path = Path(video["embedding_path"])
        meta_path = Path(video["clip_meta_path"])
        if not emb_path.exists() or not meta_path.exists():
            raise ValueError(f"artifacts missing for video: {video_id}")

        emb = np.load(emb_path).astype(np.float32)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return video, emb, meta

    def enqueue_score(
        self,
        gold_video_id: int,
        trainee_video_id: int,
        requested_by: str | None = None,
    ) -> dict:
        if self._queue is None:
            raise RuntimeError("queue manager is not available in this runtime mode")
        job_id = self.db.create_score_job(
            gold_video_id=gold_video_id,
            trainee_video_id=trainee_video_id,
            requested_by=requested_by,
        )
        try:
            self._queue.enqueue_score(job_id)
        except Exception as exc:
            self.db.fail_score_job(job_id, f"enqueue failed: {exc}")
            raise
        return {"score_job_id": job_id, "status": "queued", "score": None}

    def run_score_job(self, job_id: str) -> None:
        self.db.mark_score_job_running(job_id)
        try:
            row = self.db.get_score_job(job_id)
            if row is None:
                raise ValueError(f"score job not found: {job_id}")

            gold_video_id = int(row["gold_video_id"])
            trainee_video_id = int(row["trainee_video_id"])
            gold_video, gold_emb, gold_meta = self._load_video_artifacts(gold_video_id)
            trainee_video, trainee_emb, trainee_meta = self._load_video_artifacts(trainee_video_id)
            if gold_video["task_id"] != trainee_video["task_id"]:
                raise ValueError("gold and trainee task_id must match")
            min_clips = max(1, int(self.settings.min_scoring_clips))
            if int(gold_emb.shape[0]) < min_clips or int(trainee_emb.shape[0]) < min_clips:
                raise ValueError(
                    "insufficient clip coverage for scoring: "
                    f"gold={int(gold_emb.shape[0])}, trainee={int(trainee_emb.shape[0])}, min={min_clips}. "
                    "Increase video duration or reduce SOPILOT_CLIP_SECONDS."
                )

            result = evaluate_sop(
                gold_embeddings=gold_emb,
                trainee_embeddings=trainee_emb,
                gold_meta=gold_meta,
                trainee_meta=trainee_meta,
                threshold_factor=self.settings.change_threshold_factor,
                min_step_clips=self.settings.min_step_clips,
                low_similarity_threshold=self.settings.low_similarity_threshold,
                w_miss=self.settings.w_miss,
                w_swap=self.settings.w_swap,
                w_dev=self.settings.w_dev,
                w_time=self.settings.w_time,
                w_warp=self.settings.w_warp,
                neural_mode=self.settings.neural_mode,
                neural_model_dir=self.settings.neural_model_dir if self.settings.neural_mode else None,
                neural_device=self.settings.neural_device,
                neural_soft_dtw_gamma=self.settings.neural_soft_dtw_gamma,
                neural_uncertainty_samples=self.settings.neural_uncertainty_samples,
                neural_calibration_enabled=self.settings.neural_calibration_enabled,
                neural_cuda_dtw=self.settings.neural_cuda_dtw,
                neural_ot_alignment=self.settings.neural_ot_alignment,
                neural_conformal_alpha=self.settings.neural_conformal_alpha,
            )
            result["gold_video_id"] = int(gold_video_id)
            result["trainee_video_id"] = int(trainee_video_id)
            result["task_id"] = gold_video["task_id"]
            gold_model = str(gold_video.get("embedding_model") or "")
            trainee_model = str(trainee_video.get("embedding_model") or "")
            if gold_model and gold_model == trainee_model:
                result["embedding_model"] = gold_model
            elif gold_model or trainee_model:
                result["embedding_model"] = f"mixed:{gold_model}|{trainee_model}"
            else:
                result["embedding_model"] = self.embedding_mgr.active_model_name()

            result_path = self.settings.reports_dir / f"score_{job_id}.json"
            write_json(result_path, result)
            self.db.complete_score_job(
                job_id=job_id,
                score=float(result["score"]),
                result_path=str(result_path),
            )
        except Exception as exc:
            self.db.fail_score_job(job_id, str(exc))
            logger.exception("Score job failed job_id=%s", job_id)
            raise

    def get_score(self, score_job_id: str) -> dict | None:
        row = self.db.get_score_job(score_job_id)
        if row is None:
            return None

        payload = {
            "score_job_id": row["id"],
            "status": row["status"],
            "gold_video_id": int(row["gold_video_id"]),
            "trainee_video_id": int(row["trainee_video_id"]),
            "requested_by": row.get("requested_by"),
            "score": float(row["score"]) if row["score"] is not None else None,
            "error_message": row["error_message"],
            "queued_at": row.get("queued_at"),
            "started_at": row.get("started_at"),
            "finished_at": row.get("finished_at"),
            "result": None,
        }

        result_path = row.get("result_path")
        if result_path:
            rp = Path(result_path)
            if rp.exists():
                payload["result"] = json.loads(rp.read_text(encoding="utf-8"))
        return payload

    def build_score_pdf(self, score_job_id: str) -> Path:
        row = self.db.get_score_job(score_job_id)
        if row is None:
            raise ValueError("score job not found")
        if row.get("status") != "completed":
            raise ValueError("score job is not completed yet")
        result_path = row.get("result_path")
        if not result_path:
            raise ValueError("score result artifact is missing")
        src = Path(result_path)
        if not src.exists():
            raise ValueError("score result file not found on disk")
        result = json.loads(src.read_text(encoding="utf-8"))

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
        except Exception as exc:
            raise RuntimeError("reportlab is required for PDF export") from exc

        out = self.settings.reports_dir / f"score_{score_job_id}.pdf"
        c = canvas.Canvas(str(out), pagesize=A4)
        _, height = A4
        x = 42
        y = height - 48

        def line(text: str, step: int = 16) -> None:
            nonlocal y
            c.drawString(x, y, text[:1500])
            y -= step
            if y < 50:
                c.showPage()
                y = height - 48

        line(self.settings.report_title, 20)
        line(f"Score Job: {score_job_id}")
        line(f"Requested By: {row.get('requested_by') or 'unknown'}")
        line(f"Task: {result.get('task_id', '')}")
        line(f"Gold Video ID: {result.get('gold_video_id', '')}")
        line(f"Trainee Video ID: {result.get('trainee_video_id', '')}")
        line(f"Embedding Model: {result.get('embedding_model', '')}")
        line(f"Score: {result.get('score', '')}")
        line("")
        metrics = result.get("metrics", {})
        line("Metrics:")
        for key in [
            "miss",
            "swap",
            "deviation",
            "over_time",
            "temporal_warp",
            "path_stretch",
            "duplicate_ratio",
            "order_violation_ratio",
            "temporal_drift",
            "confidence_loss",
            "local_similarity_gap",
            "hard_miss_ratio",
            "mean_alignment_cost",
        ]:
            if key in metrics:
                line(f" - {key}: {metrics[key]}")
        line("")
        deviations = result.get("deviations", [])
        line(f"Deviations ({len(deviations)}):")
        for idx, item in enumerate(deviations[:100], start=1):
            typ = item.get("type", "")
            conf = item.get("confidence", "")
            reason = item.get("reason", "")
            gt = item.get("gold_time", {})
            tt = item.get("trainee_time", {})
            line(
                f"{idx}. {typ} conf={conf} reason={reason} "
                f"gold=({gt.get('start_sec')},{gt.get('end_sec')}) "
                f"trainee=({tt.get('start_sec')},{tt.get('end_sec')})"
            )
        c.save()
        return out

    def search(self, task_id: str, video_id: int, clip_idx: int, k: int) -> dict:
        video, emb, _ = self._load_video_artifacts(video_id)
        if video["task_id"] != task_id:
            raise ValueError("requested task_id does not match query video")
        if clip_idx < 0 or clip_idx >= emb.shape[0]:
            raise ValueError(f"clip_idx out of range: {clip_idx}")

        query = emb[clip_idx]
        items = self.index.search(
            task_id=task_id,
            query=query,
            k=max(1, k),
            exclude_video_id=video_id,
            exclude_clip_idx=clip_idx,
        )
        return {
            "task_id": task_id,
            "query_video_id": int(video_id),
            "query_clip_idx": int(clip_idx),
            "items": items,
        }
