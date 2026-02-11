from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import uuid

import numpy as np

from .config import Settings
from .db import Database, VideoCreateInput
from .embedding_manager import EmbeddingManager
from .queueing import QueueManager
from .utils import now_tag, safe_filename
from .vector_index import NpyVectorIndex
from .video import ClipWindowStream

logger = logging.getLogger(__name__)


class IngestService:
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

    def enqueue_ingest(
        self,
        *,
        file_name: str,
        payload: bytes,
        task_id: str,
        role: str,
        requested_by: str | None = None,
        site_id: str | None,
        camera_id: str | None,
        operator_id_hash: str | None,
    ) -> dict:
        incoming_dir = self.settings.raw_dir / ".incoming"
        incoming_dir.mkdir(parents=True, exist_ok=True)
        staged = incoming_dir / f"{now_tag()}_{uuid.uuid4().hex[:8]}_{safe_filename(file_name)}"
        staged.write_bytes(payload)
        return self.enqueue_ingest_from_path(
            file_name=file_name,
            staged_path=staged,
            task_id=task_id,
            role=role,
            requested_by=requested_by,
            site_id=site_id,
            camera_id=camera_id,
            operator_id_hash=operator_id_hash,
        )

    def enqueue_ingest_from_path(
        self,
        *,
        file_name: str,
        staged_path: Path,
        task_id: str,
        role: str,
        requested_by: str | None = None,
        site_id: str | None,
        camera_id: str | None,
        operator_id_hash: str | None,
    ) -> dict:
        if self._queue is None:
            raise RuntimeError("queue manager is not available in this runtime mode")
        if not staged_path.exists():
            raise ValueError(f"staged file not found: {staged_path}")

        safe_name = safe_filename(file_name)
        raw_path = self.settings.raw_dir / f"{now_tag()}_{uuid.uuid4().hex[:8]}_{safe_name}"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(staged_path), str(raw_path))
        except Exception:
            shutil.copy2(str(staged_path), str(raw_path))
            staged_path.unlink(missing_ok=True)

        job_id = self.db.create_ingest_job(
            task_id=task_id,
            role=role,
            requested_by=requested_by,
            file_name=safe_name,
            file_path=str(raw_path),
            site_id=site_id,
            camera_id=camera_id,
            operator_id_hash=operator_id_hash,
        )
        try:
            self._queue.enqueue_ingest(job_id)
        except Exception as exc:
            self.db.fail_ingest_job(job_id, f"enqueue failed: {exc}")
            raise

        return {"ingest_job_id": job_id, "status": "queued"}

    def run_ingest_job(self, job_id: str) -> None:
        self.db.mark_ingest_job_running(job_id)
        try:
            row = self.db.get_ingest_job(job_id)
            if row is None:
                raise ValueError(f"ingest job not found: {job_id}")
            result = self._process_ingest_job(row)
            self.db.complete_ingest_job(
                job_id=job_id,
                video_id=int(result["video_id"]),
                num_clips=int(result["num_clips"]),
                source_fps=float(result["source_fps"]),
                sampled_fps=float(result["sampled_fps"]),
                embedding_model=str(result["embedding_model"]),
            )
        except Exception as exc:
            self.db.fail_ingest_job(job_id, str(exc))
            logger.exception("Ingest job failed job_id=%s", job_id)
            raise

    def _process_ingest_job(self, job_row: dict) -> dict:
        task_id = str(job_row["task_id"])
        role = str(job_row["role"])
        file_path = Path(job_row["file_path"])
        if not file_path.exists():
            raise ValueError(f"ingest file missing: {file_path}")

        video_id = self.db.create_video(
            VideoCreateInput(
                task_id=task_id,
                role=role,
                file_path=str(file_path),
                embedding_model=self.embedding_mgr.active_model_name(),
                site_id=job_row.get("site_id"),
                camera_id=job_row.get("camera_id"),
                operator_id_hash=job_row.get("operator_id_hash"),
            )
        )

        stream = ClipWindowStream(
            video_path=file_path,
            target_fps=self.settings.target_fps,
            clip_seconds=self.settings.clip_seconds,
            max_side=self.settings.max_side,
            min_clip_coverage=self.settings.min_clip_coverage,
            privacy_mask_enabled=self.settings.privacy_mask_enabled,
            privacy_mask_mode=self.settings.privacy_mask_mode,
            privacy_mask_rects=self.settings.privacy_mask_rects,
            privacy_face_blur=self.settings.privacy_face_blur,
        )

        clip_meta: list[dict] = []
        raw_chunks: list[np.ndarray] = []
        eff_chunks: list[np.ndarray] = []
        clip_batch: list = []
        ingest_batch = max(1, int(self.settings.ingest_embed_batch_size))

        for clip in stream:
            clip_meta.append(
                {
                    "clip_idx": int(clip.clip_idx),
                    "start_sec": float(clip.start_sec),
                    "end_sec": float(clip.end_sec),
                    "quality_flags": ",".join(clip.quality_flags),
                }
            )
            clip_batch.append(clip)
            if len(clip_batch) >= ingest_batch:
                raw, effective = self.embedding_mgr.embed_batch(clip_batch)
                raw_chunks.append(raw)
                eff_chunks.append(effective)
                clip_batch = []

        if clip_batch:
            raw, effective = self.embedding_mgr.embed_batch(clip_batch)
            raw_chunks.append(raw)
            eff_chunks.append(effective)

        if not clip_meta or not raw_chunks or not eff_chunks:
            raise ValueError("no decodable clip produced from video")

        raw_embeddings = np.concatenate(raw_chunks, axis=0).astype(np.float32)
        effective_embeddings = np.concatenate(eff_chunks, axis=0).astype(np.float32)
        embedder_name = self.embedding_mgr.active_model_name()

        raw_embedding_path = self.settings.embeddings_dir / f"video_{video_id}.raw.npy"
        embedding_path = self.settings.embeddings_dir / f"video_{video_id}.npy"
        clip_meta_path = self.settings.embeddings_dir / f"video_{video_id}.json"
        np.save(raw_embedding_path, raw_embeddings)
        np.save(embedding_path, effective_embeddings)
        clip_meta_path.write_text(
            json.dumps(clip_meta, ensure_ascii=True, indent=2), encoding="utf-8"
        )

        self.db.finalize_video(
            video_id=video_id,
            raw_embedding_path=str(raw_embedding_path),
            embedding_path=str(embedding_path),
            clip_meta_path=str(clip_meta_path),
            embedding_model=embedder_name,
            num_clips=len(clip_meta),
        )
        self.db.add_clips(video_id=video_id, task_id=task_id, role=role, rows=clip_meta)

        index_rows = [
            {
                "video_id": int(video_id),
                "clip_idx": int(meta["clip_idx"]),
                "start_sec": float(meta["start_sec"]),
                "end_sec": float(meta["end_sec"]),
                "role": role,
            }
            for meta in clip_meta
        ]
        try:
            self.index.add(task_id=task_id, vectors=effective_embeddings, metadata=index_rows)
        except ValueError as exc:
            if "embedding dimension mismatch" not in str(exc):
                raise
            rebuilt = self.rebuild_task_index(task_id, preferred_dim=int(effective_embeddings.shape[1]))
            logger.warning(
                "Rebuilt task index due embedding dimension mismatch task_id=%s preferred_dim=%s indexed_clips=%s",
                task_id,
                int(effective_embeddings.shape[1]),
                rebuilt,
            )

        stats = stream.stats or {"source_fps": 0.0, "sampled_fps": 0.0}
        return {
            "video_id": int(video_id),
            "task_id": task_id,
            "role": role,
            "num_clips": int(len(clip_meta)),
            "source_fps": float(stats.get("source_fps", 0.0)),
            "sampled_fps": float(stats.get("sampled_fps", 0.0)),
            "embedding_model": embedder_name,
        }

    def get_ingest_job(self, ingest_job_id: str) -> dict | None:
        row = self.db.get_ingest_job(ingest_job_id)
        if row is None:
            return None
        return {
            "ingest_job_id": row["id"],
            "status": row["status"],
            "task_id": row["task_id"],
            "role": row["role"],
            "requested_by": row.get("requested_by"),
            "video_id": int(row["video_id"]) if row.get("video_id") is not None else None,
            "num_clips": int(row["num_clips"]) if row.get("num_clips") is not None else None,
            "source_fps": float(row["source_fps"]) if row.get("source_fps") is not None else None,
            "sampled_fps": float(row["sampled_fps"]) if row.get("sampled_fps") is not None else None,
            "embedding_model": row.get("embedding_model"),
            "error_message": row.get("error_message"),
            "queued_at": row.get("queued_at"),
            "started_at": row.get("started_at"),
            "finished_at": row.get("finished_at"),
        }

    def rebuild_task_index(self, task_id: str, preferred_dim: int | None = None) -> int:
        rows = self.db.list_task_videos_with_artifacts(task_id)
        meta_rows: list[dict] = []
        vec_chunks: list[np.ndarray] = []
        dim_counts: dict[int, int] = {}
        candidate_rows: list[tuple[dict, np.ndarray, list[dict]]] = []

        for row in rows:
            emb_path = Path(str(row.get("embedding_path") or ""))
            meta_path = Path(str(row.get("clip_meta_path") or ""))
            if not emb_path.exists() or not meta_path.exists():
                continue
            emb = np.load(emb_path).astype(np.float32)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            n = min(int(emb.shape[0]), len(meta))
            if n <= 0:
                continue
            dim = int(emb.shape[1])
            dim_counts[dim] = dim_counts.get(dim, 0) + n
            candidate_rows.append((row, emb[:n], meta[:n]))

        target_dim: int | None
        if preferred_dim is not None and preferred_dim > 0:
            target_dim = int(preferred_dim)
        elif dim_counts:
            target_dim = max(dim_counts.items(), key=lambda x: x[1])[0]
        else:
            target_dim = None

        for row, emb, meta in candidate_rows:
            if target_dim is not None and int(emb.shape[1]) != target_dim:
                continue
            vec_chunks.append(emb)
            for i in range(int(emb.shape[0])):
                meta_rows.append(
                    {
                        "video_id": int(row["id"]),
                        "clip_idx": int(meta[i]["clip_idx"]),
                        "start_sec": float(meta[i]["start_sec"]),
                        "end_sec": float(meta[i]["end_sec"]),
                        "role": row.get("role", "unknown"),
                    }
                )

        if vec_chunks and meta_rows:
            vectors = np.concatenate(vec_chunks, axis=0).astype(np.float32)
        else:
            vectors = np.zeros((0, 0), dtype=np.float32)
            meta_rows = []

        self.index.overwrite_task(task_id=task_id, vectors=vectors, metadata=meta_rows)
        return int(len(meta_rows))
