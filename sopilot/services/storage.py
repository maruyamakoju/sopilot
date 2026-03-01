from pathlib import Path
from typing import BinaryIO

from sopilot.exceptions import FileTooLargeError

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


class FileStorage:
    def __init__(self, raw_video_dir: Path, *, max_upload_bytes: int = 0) -> None:
        self.raw_video_dir = raw_video_dir
        self.raw_video_dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_upload_bytes  # 0 = unlimited

    def save_upload(self, video_id: int, original_filename: str, source: BinaryIO) -> str:
        suffix = Path(original_filename or "").suffix.lower()
        if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            suffix = ".mp4"
        target = self.raw_video_dir / f"{video_id:08d}{suffix}"
        source.seek(0)
        written = 0
        with target.open("wb") as out:
            while True:
                chunk = source.read(_CHUNK_SIZE)
                if not chunk:
                    break
                written += len(chunk)
                if self._max_bytes and written > self._max_bytes:
                    out.close()
                    target.unlink(missing_ok=True)
                    max_mb = self._max_bytes / (1024 * 1024)
                    raise FileTooLargeError(
                        f"Upload exceeds {max_mb:.0f} MB limit"
                    )
                out.write(chunk)
        return str(target)
