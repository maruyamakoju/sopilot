from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import zipfile
from typing import Any


def resolve_path(workspace: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = value if isinstance(value, Path) else Path(value)
    if path.is_absolute():
        return path.resolve()
    return (workspace / path).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(read_text(path))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def extract_sha256(text: str) -> str | None:
    match = re.search(r"\b([0-9a-fA-F]{64})\b", text)
    if not match:
        return None
    return match.group(1).lower()


def read_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return extract_sha256(read_text(path))


def zip_entries(path: Path) -> list[str]:
    with zipfile.ZipFile(path, "r") as zf:
        return sorted(info.filename for info in zf.infolist())


def latest_subdir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [path for path in root.iterdir() if path.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.name)[-1]
