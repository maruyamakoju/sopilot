"""VigilPilot analysis pipeline.

Orchestrates frame extraction → VLM analysis → violation event storage.
Runs in a background thread (non-blocking for the API caller).
"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

from sopilot.vigil.extractor import iter_frames
from sopilot.vigil.repository import VigilRepository
from sopilot.vigil.vlm import VLMClient

logger = logging.getLogger(__name__)


class VigilPipeline:
    """Background pipeline: video → frames → VLM → violation events."""

    def __init__(self, repo: VigilRepository, vlm: VLMClient, frames_root: Path) -> None:
        self._repo = repo
        self._vlm = vlm
        self._frames_root = frames_root
        self._frames_root.mkdir(parents=True, exist_ok=True)

    def analyze_async(
        self,
        session_id: int,
        video_path: Path,
        rules: list[str],
        sample_fps: float,
        severity_threshold: str,
        cleanup_video: bool = False,
    ) -> None:
        """Launch analysis in a daemon thread and return immediately."""
        t = threading.Thread(
            target=self._run,
            args=(session_id, video_path, rules, sample_fps, severity_threshold, cleanup_video),
            daemon=True,
            name=f"vigil-{session_id}",
        )
        t.start()
        logger.info("vigil session=%d analysis started (thread=%s)", session_id, t.name)

    def _run(
        self,
        session_id: int,
        video_path: Path,
        rules: list[str],
        sample_fps: float,
        severity_threshold: str,
        cleanup_video: bool,
    ) -> None:
        frame_dir = self._frames_root / f"session_{session_id}"
        frames_analyzed = 0
        violation_count = 0
        _SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}
        threshold_level = _SEVERITY_ORDER.get(severity_threshold, 1)

        try:
            self._repo.update_session_status(
                session_id,
                status="processing",
                video_filename=video_path.name,
            )

            for frame_num, ts_sec, frame_path in iter_frames(
                video_path, sample_fps=sample_fps, output_dir=frame_dir
            ):
                try:
                    result = self._vlm.analyze_frame(frame_path, rules)
                except Exception as e:
                    logger.warning("VLM error at frame %d (%.1fs): %s", frame_num, ts_sec, e)
                    frames_analyzed += 1
                    continue

                frames_analyzed += 1

                if result.has_violation and result.violations:
                    # Filter by severity threshold
                    filtered = [
                        v for v in result.violations
                        if _SEVERITY_ORDER.get(v.get("severity", "warning"), 1) >= threshold_level
                    ]
                    if filtered:
                        self._repo.create_event(
                            session_id=session_id,
                            timestamp_sec=ts_sec,
                            frame_number=frame_num,
                            violations=filtered,
                            frame_path=str(frame_path),
                        )
                        violation_count += len(filtered)
                        logger.info(
                            "vigil session=%d  frame=%d  ts=%.1fs  violations=%d",
                            session_id, frame_num, ts_sec, len(filtered),
                        )

            self._repo.update_session_status(
                session_id,
                status="completed",
                total_frames_analyzed=frames_analyzed,
                violation_count=violation_count,
            )
            logger.info(
                "vigil session=%d completed  frames=%d  violations=%d",
                session_id, frames_analyzed, violation_count,
            )

        except Exception as e:
            logger.exception("vigil session=%d failed: %s", session_id, e)
            self._repo.update_session_status(
                session_id,
                status="failed",
                total_frames_analyzed=frames_analyzed,
                violation_count=violation_count,
            )
        finally:
            if cleanup_video and video_path.exists():
                video_path.unlink(missing_ok=True)
