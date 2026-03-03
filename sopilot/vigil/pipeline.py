"""VigilPilot analysis pipeline.

Orchestrates frame extraction → VLM analysis → violation event storage.
Runs in a background thread (non-blocking for the API caller).
"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

import httpx

from sopilot.vigil.extractor import iter_frames, iter_frames_rtsp
from sopilot.vigil.repository import VigilRepository
from sopilot.vigil.vlm import VLMClient
from sopilot.vigil.webhook_dispatcher import WebhookDispatcher

logger = logging.getLogger(__name__)

_WEBHOOK_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


def _fire_webhook(
    repo: VigilRepository,
    session_id: int,
    event_id: int,
    timestamp: float,
    violations_above_threshold: list[dict],
) -> None:
    """Fire a webhook notification for a violation event in a daemon thread."""
    row = repo.get_session(session_id)
    webhook_url = row.get("webhook_url") if row else None
    if not webhook_url:
        return
    min_sev = row.get("webhook_min_severity") or "warning"
    min_level = _WEBHOOK_SEVERITY_ORDER.get(min_sev, 1)
    should_fire = any(
        _WEBHOOK_SEVERITY_ORDER.get(v.get("severity", "warning"), 1) >= min_level
        for v in violations_above_threshold
    )
    if not should_fire:
        return

    def _fire() -> None:
        try:
            httpx.post(
                webhook_url,
                json={
                    "session_id": session_id,
                    "event_id": event_id,
                    "timestamp_sec": timestamp,
                    "violations": violations_above_threshold,
                },
                timeout=5,
            )
        except Exception:
            pass

    threading.Thread(target=_fire, daemon=True).start()


class VigilPipeline:
    """Background pipeline: video → frames → VLM → violation events."""

    def __init__(
        self,
        repo: VigilRepository,
        vlm: VLMClient,
        frames_root: Path,
        webhook_repo=None,  # WebhookRepository | None
    ) -> None:
        self._repo = repo
        self._vlm = vlm
        self._frames_root = frames_root
        self._frames_root.mkdir(parents=True, exist_ok=True)
        self._stream_stop_events: dict[int, threading.Event] = {}
        self._webhook_repo = webhook_repo
        self._webhook_dispatcher = WebhookDispatcher() if webhook_repo is not None else None

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

    def stream_async(
        self,
        session_id: int,
        rtsp_url: str,
        rules: list[str],
        sample_fps: float,
        severity_threshold: str,
    ) -> None:
        """Launch RTSP live-stream analysis in a daemon thread and return immediately.

        The session status will be set to ``"processing"`` while streaming is
        active and changed to ``"completed"`` once the stream stops (either
        because ``stop_stream()`` was called, ``max_frames`` was reached, or the
        stream ended on its own).

        Parameters
        ----------
        session_id:
            ID of an existing vigil session (must be in ``"idle"`` state).
        rtsp_url:
            RTSP URL passed directly to :func:`iter_frames_rtsp`.
        rules:
            Monitoring rules forwarded to the VLM.
        sample_fps:
            Frames per second to sample from the live stream.
        severity_threshold:
            Minimum severity level (``"info"``, ``"warning"``, ``"critical"``).
        """
        stop_event = threading.Event()
        self._stream_stop_events[session_id] = stop_event

        t = threading.Thread(
            target=self._stream_worker,
            args=(session_id, rtsp_url, rules, sample_fps, severity_threshold, stop_event),
            daemon=True,
            name=f"vigil-rtsp-{session_id}",
        )
        t.start()
        logger.info(
            "vigil session=%d RTSP stream started (url=%s thread=%s)",
            session_id, rtsp_url, t.name,
        )

    def stop_stream(self, session_id: int) -> bool:
        """Signal the running RTSP stream for *session_id* to stop.

        Returns ``True`` if a stop event was found and set, ``False`` if no
        active stream exists for this session.
        """
        event = self._stream_stop_events.get(session_id)
        if event is None:
            logger.warning("stop_stream called but no active stream for session=%d", session_id)
            return False
        event.set()
        logger.info("vigil session=%d RTSP stop requested", session_id)
        return True

    # ── Global webhook dispatch ────────────────────────────────────────────

    def _dispatch_global_webhooks(
        self,
        session_id: int,
        event_id: int,
        timestamp_sec: float,
        violations: list[dict],
    ) -> None:
        """Dispatch violation to all registered global webhooks (fire-and-forget).

        Called after each event is saved.  Silently no-ops if no webhook repo
        or dispatcher is configured.
        """
        if self._webhook_repo is None or self._webhook_dispatcher is None:
            return
        try:
            webhooks = self._webhook_repo.list_all()
            if not webhooks:
                return
            # Use the highest severity among the violations in this event
            from sopilot.vigil.webhook_dispatcher import SEVERITY_ORDER
            top_sev = max(
                violations,
                key=lambda v: SEVERITY_ORDER.get(v.get("severity", "info"), 0),
                default=None,
            )
            severity = top_sev.get("severity", "warning") if top_sev else "warning"
            from datetime import UTC, datetime
            payload = {
                "event": "violation",
                "session_id": session_id,
                "event_id": event_id,
                "severity": severity,
                "timestamp_sec": timestamp_sec,
                "violations": violations,
                "timestamp": datetime.now(UTC).isoformat(),
                "source": "sopilot-vigil",
            }
            self._webhook_dispatcher.dispatch_violation(payload, webhooks, self._webhook_repo)
        except Exception:
            logger.exception("global webhook dispatch failed for session=%d", session_id)

    def _stream_worker(
        self,
        session_id: int,
        rtsp_url: str,
        rules: list[str],
        sample_fps: float,
        severity_threshold: str,
        stop_event: threading.Event,
    ) -> None:
        """Background thread body for RTSP stream analysis."""
        frame_dir = self._frames_root / f"session_{session_id}_rtsp"
        frames_analyzed = 0
        violation_count = 0
        _SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}
        threshold_level = _SEVERITY_ORDER.get(severity_threshold, 1)

        # Reset stateful VLM backends between sessions (see _run()).
        self._vlm.reset_session(session_id=str(session_id))

        try:
            self._repo.update_session_status(
                session_id,
                status="processing",
                video_filename=rtsp_url,
            )

            for frame_num, ts_sec, frame_path in iter_frames_rtsp(
                rtsp_url,
                sample_fps=sample_fps,
                output_dir=frame_dir,
                stop_event=stop_event,
            ):
                try:
                    result = self._vlm.analyze_frame(frame_path, rules)
                except Exception as e:
                    logger.warning(
                        "VLM error at rtsp frame %d (%.1fs): %s", frame_num, ts_sec, e
                    )
                    frames_analyzed += 1
                    continue

                frames_analyzed += 1

                if result.has_violation and result.violations:
                    filtered = [
                        v for v in result.violations
                        if _SEVERITY_ORDER.get(v.get("severity", "warning"), 1) >= threshold_level
                    ]
                    if filtered:
                        event_id = self._repo.create_event(
                            session_id=session_id,
                            timestamp_sec=ts_sec,
                            frame_number=frame_num,
                            violations=filtered,
                            frame_path=str(frame_path),
                        )
                        violation_count += len(filtered)
                        logger.info(
                            "vigil rtsp session=%d  frame=%d  ts=%.1fs  violations=%d",
                            session_id, frame_num, ts_sec, len(filtered),
                        )
                        _fire_webhook(self._repo, session_id, event_id, ts_sec, filtered)
                        self._dispatch_global_webhooks(
                            session_id, event_id, ts_sec, filtered
                        )

            self._repo.update_session_status(
                session_id,
                status="completed",
                total_frames_analyzed=frames_analyzed,
                violation_count=violation_count,
            )
            logger.info(
                "vigil rtsp session=%d completed  frames=%d  violations=%d",
                session_id, frames_analyzed, violation_count,
            )

        except Exception as e:
            logger.exception("vigil rtsp session=%d failed: %s", session_id, e)
            self._repo.update_session_status(
                session_id,
                status="failed",
                total_frames_analyzed=frames_analyzed,
                violation_count=violation_count,
            )
        finally:
            # Clean up the stop event entry so memory doesn't grow unboundedly
            self._stream_stop_events.pop(session_id, None)

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

        # Reset stateful VLM backends (e.g. PerceptionVLMClient) so that
        # tracker / world-model state from a previous session does not leak
        # into this one.
        self._vlm.reset_session(session_id=str(session_id))

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
                        event_id = self._repo.create_event(
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
                        _fire_webhook(self._repo, session_id, event_id, ts_sec, filtered)
                        self._dispatch_global_webhooks(
                            session_id, event_id, ts_sec, filtered
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
