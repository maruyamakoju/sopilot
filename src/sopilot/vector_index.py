from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .utils import normalize_rows as _normalize_rows


def _safe_task_id(task_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)
    return cleaned[:120] if cleaned else "default"


class NpyVectorIndex:
    """
    Versioned on-disk vector index.

    - Current reads use pointer file (`current_version.txt`).
    - Rebuild can happen in a staging version and atomically activate at the end.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._pointer_path = self.root / "current_version.txt"
        self._lock = threading.RLock()
        self._ensure_default_version()

    def _ensure_default_version(self) -> None:
        with self._lock:
            if self._pointer_path.exists():
                return
            default = "v1"
            self._pointer_path.write_text(default, encoding="utf-8")
            self._version_dir(default).mkdir(parents=True, exist_ok=True)

    def _version_dir(self, version: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", version)
        return self.root / safe

    def _current_version(self) -> str:
        with self._lock:
            self._ensure_default_version()
            return self._pointer_path.read_text(encoding="utf-8").strip() or "v1"

    def current_version(self) -> str:
        with self._lock:
            return self._current_version()

    def _paths(self, task_id: str, *, version: str | None = None) -> tuple[Path, Path]:
        if version is None:
            version = self._current_version()
        version_dir = self._version_dir(version)
        version_dir.mkdir(parents=True, exist_ok=True)
        safe = _safe_task_id(task_id)
        return version_dir / f"{safe}.npy", version_dir / f"{safe}.json"

    def _load(self, task_id: str, *, version: str | None = None) -> tuple[np.ndarray, list[dict]]:
        with self._lock:
            vec_path, meta_path = self._paths(task_id, version=version)
            if not vec_path.exists() or not meta_path.exists():
                return np.zeros((0, 0), dtype=np.float32), []
            vectors = np.load(vec_path)
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            return vectors.astype(np.float32), metadata

    def _save(
        self,
        task_id: str,
        vectors: np.ndarray,
        metadata: list[dict],
        *,
        version: str | None = None,
    ) -> None:
        with self._lock:
            vec_path, meta_path = self._paths(task_id, version=version)
            temp_vec = vec_path.with_suffix(f"{vec_path.suffix}.{uuid.uuid4().hex}.tmp")
            temp_meta = meta_path.with_suffix(f"{meta_path.suffix}.{uuid.uuid4().hex}.tmp")
            with temp_vec.open("wb") as f:
                np.save(f, vectors.astype(np.float32))
            temp_meta.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
            os.replace(temp_vec, vec_path)
            os.replace(temp_meta, meta_path)

    def add(self, task_id: str, vectors: np.ndarray, metadata: list[dict]) -> None:
        with self._lock:
            self.add_to_version(self._current_version(), task_id, vectors, metadata)

    def add_to_version(self, version: str, task_id: str, vectors: np.ndarray, metadata: list[dict]) -> None:
        with self._lock:
            existing_vecs, existing_meta = self._load(task_id, version=version)

            if existing_vecs.size == 0:
                combined_vecs = vectors.astype(np.float32)
            else:
                if existing_vecs.shape[1] != vectors.shape[1]:
                    raise ValueError(f"embedding dimension mismatch: {existing_vecs.shape[1]} vs {vectors.shape[1]}")
                combined_vecs = np.concatenate([existing_vecs, vectors.astype(np.float32)], axis=0)

            combined_meta = existing_meta + metadata
            self._save(task_id, combined_vecs, combined_meta, version=version)

    def create_staging_version(self) -> str:
        with self._lock:
            now = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            version = f"staging_{now}_{uuid.uuid4().hex[:8]}"
            self._version_dir(version).mkdir(parents=True, exist_ok=True)
            return version

    def activate_version(self, version: str) -> None:
        with self._lock:
            version_dir = self._version_dir(version)
            if not version_dir.exists():
                raise ValueError(f"index version does not exist: {version}")
            self._pointer_path.write_text(version, encoding="utf-8")

    def delete_version(self, version: str) -> None:
        with self._lock:
            version = version.strip()
            if not version:
                return
            current = self._current_version()
            if version == current:
                return
            vdir = self._version_dir(version)
            if not vdir.exists():
                return
            shutil.rmtree(vdir, ignore_errors=True)

    def clear(self) -> None:
        with self._lock:
            current = self._current_version()
            vdir = self._version_dir(current)
            for path in vdir.glob("*.npy"):
                path.unlink(missing_ok=True)
            for path in vdir.glob("*.json"):
                path.unlink(missing_ok=True)

    def overwrite_task(self, task_id: str, vectors: np.ndarray, metadata: list[dict]) -> None:
        with self._lock:
            vec_path, meta_path = self._paths(task_id, version=None)
            if vectors.size == 0 or len(metadata) == 0:
                vec_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return
            self._save(task_id, vectors.astype(np.float32), metadata, version=None)

    def search(
        self,
        task_id: str,
        query: np.ndarray,
        k: int,
        exclude_video_id: int | None = None,
        exclude_clip_idx: int | None = None,
    ) -> list[dict]:
        with self._lock:
            vectors, metadata = self._load(task_id, version=None)
            if vectors.size == 0:
                return []

            if vectors.ndim != 2:
                raise ValueError("stored vector index has invalid shape")

            if query.ndim != 1:
                raise ValueError("query vector must be 1D")

            if query.shape[0] != vectors.shape[1]:
                raise ValueError("query and index embedding dimensions differ")

            mat = _normalize_rows(vectors.astype(np.float32))
            q = query.astype(np.float32)
            q = q / max(float(np.linalg.norm(q)), 1e-12)
            sims = mat @ q

            order = np.argsort(-sims)
            out: list[dict] = []
            for idx in order:
                meta = metadata[int(idx)]
                if exclude_video_id is not None and exclude_clip_idx is not None:
                    if (
                        int(meta.get("video_id", -1)) == exclude_video_id
                        and int(meta.get("clip_idx", -1)) == exclude_clip_idx
                    ):
                        continue
                out.append(
                    {
                        "similarity": float(sims[int(idx)]),
                        "video_id": int(meta["video_id"]),
                        "clip_idx": int(meta["clip_idx"]),
                        "start_sec": float(meta["start_sec"]),
                        "end_sec": float(meta["end_sec"]),
                        "role": meta.get("role", "unknown"),
                    }
                )
                if len(out) >= k:
                    break
            return out
