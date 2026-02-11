from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2-D matrix in-place-safe."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (vectors / norms).astype(np.float32)


def safe_filename(name: str, *, fallback: str = "video.mp4") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return cleaned[:120] or fallback


def now_tag() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
