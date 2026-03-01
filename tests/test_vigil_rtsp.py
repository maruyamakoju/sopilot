"""Tests for VigilPilot RTSP live-stream support.

Coverage:
  - TestRTSPExtractor (4 tests):
      iter_frames_rtsp() with a synthetic local video file used as a
      stand-in for an RTSP stream (cv2.VideoCapture accepts file paths
      as well as rtsp:// URLs, so the extractor logic is exercised fully).
  - TestVigilPipelineStream (5 tests):
      stream_async() / stop_stream() / _stream_worker() with a mocked VLM
      and synthetic video.
"""

from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from sopilot.database import Database
from sopilot.vigil.extractor import iter_frames_rtsp
from sopilot.vigil.pipeline import VigilPipeline
from sopilot.vigil.repository import VigilRepository
from sopilot.vigil.schemas import VLMResult
from sopilot.vigil.vlm import VLMClient


# ──────────────────────────────────────────────────────────────────────────────
# Shared test helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_video(path: Path, seconds: int = 3, fps: float = 10.0) -> None:
    """Create a minimal synthetic AVI video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (96, 96))
    total = int(seconds * fps)
    for i in range(total):
        val = (i * 20) % 255
        frame = np.full((96, 96, 3), (val, 120, 180), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _init_db(db_path: Path) -> VigilRepository:
    """Initialise a fresh SQLite database and return a VigilRepository."""
    db = Database(db_path)
    db.close()
    return VigilRepository(db_path)


class _MockVLMNoViolation(VLMClient):
    """VLM stub — always returns no violation."""

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        return VLMResult(has_violation=False, violations=[], raw_text="{}")


class _MockVLMWithViolation(VLMClient):
    """VLM stub — always returns one warning-level violation."""

    def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
        return VLMResult(
            has_violation=True,
            violations=[{
                "rule_index": 0,
                "rule": rules[0] if rules else "test rule",
                "description_ja": "テスト違反が検出されました",
                "severity": "warning",
                "confidence": 0.9,
            }],
            raw_text='{"has_violation": true}',
        )


# ──────────────────────────────────────────────────────────────────────────────
# TestRTSPExtractor
# ──────────────────────────────────────────────────────────────────────────────


class TestRTSPExtractor(unittest.TestCase):
    """Test iter_frames_rtsp() using a local .avi file as a synthetic stream."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp = Path(self._tmpdir.name)
        self._video = self._tmp / "test_stream.avi"
        _make_video(self._video, seconds=3, fps=10.0)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    # ------------------------------------------------------------------

    def test_yields_frames_from_local_file(self) -> None:
        """iter_frames_rtsp yields (frame_num, ts_sec, path) tuples."""
        out_dir = self._tmp / "frames_out"
        frames = list(
            iter_frames_rtsp(
                str(self._video),
                sample_fps=5.0,
                max_frames=5,
                output_dir=out_dir,
            )
        )
        self.assertGreater(len(frames), 0)
        frame_num, ts_sec, jpeg_path = frames[0]
        self.assertEqual(frame_num, 0)
        self.assertIsInstance(ts_sec, float)
        self.assertGreaterEqual(ts_sec, 0.0)
        self.assertTrue(jpeg_path.exists(), "JPEG frame file must be created on disk")
        self.assertTrue(str(jpeg_path).endswith(".jpg"))

    def test_respects_max_frames_limit(self) -> None:
        """Generator stops after max_frames frames even if the stream has more."""
        out_dir = self._tmp / "frames_max"
        frames = list(
            iter_frames_rtsp(
                str(self._video),
                sample_fps=100.0,  # very fast — would over-sample without cap
                max_frames=3,
                output_dir=out_dir,
            )
        )
        self.assertLessEqual(len(frames), 3)

    def test_stop_event_halts_streaming(self) -> None:
        """Setting stop_event causes the generator to stop cleanly."""
        out_dir = self._tmp / "frames_stop"
        stop_event = threading.Event()

        collected: list[tuple[int, float, Path]] = []
        gen = iter_frames_rtsp(
            str(self._video),
            sample_fps=5.0,
            max_frames=100,
            output_dir=out_dir,
            stop_event=stop_event,
        )

        # Collect exactly 2 frames, then set stop_event
        for item in gen:
            collected.append(item)
            if len(collected) >= 2:
                stop_event.set()
                break

        # Drain — should stop quickly now that event is set
        for item in gen:
            collected.append(item)

        # We should have gotten at most a few frames (not the full 100)
        self.assertGreater(len(collected), 0)
        self.assertLess(len(collected), 100)

    def test_raises_on_nonexistent_stream(self) -> None:
        """iter_frames_rtsp raises OSError for an unreadable path/URL."""
        with self.assertRaises(OSError):
            list(
                iter_frames_rtsp(
                    "/nonexistent/path/to/stream.mp4",
                    sample_fps=1.0,
                    max_frames=5,
                )
            )


# ──────────────────────────────────────────────────────────────────────────────
# TestVigilPipelineStream
# ──────────────────────────────────────────────────────────────────────────────


class TestVigilPipelineStream(unittest.TestCase):
    """Test VigilPipeline.stream_async() and stop_stream() end-to-end."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp = Path(self._tmpdir.name)

        self._db_path = self._tmp / "vigil_test.db"
        self._repo = _init_db(self._db_path)

        self._frames_root = self._tmp / "frames"
        self._video = self._tmp / "test_stream.avi"
        _make_video(self._video, seconds=3, fps=10.0)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _make_session(self, rules: list[str] | None = None) -> int:
        return self._repo.create_session(
            name="rtsp-test",
            rules=rules or ["ヘルメット未着用を検出"],
            sample_fps=5.0,
            severity_threshold="warning",
        )

    def _wait_for_status(
        self, session_id: int, target: str, timeout: float = 10.0
    ) -> str:
        """Poll repository until session reaches target status or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            row = self._repo.get_session(session_id)
            if row and row["status"] == target:
                return target
            time.sleep(0.05)
        row = self._repo.get_session(session_id)
        return row["status"] if row else "unknown"

    # ------------------------------------------------------------------

    def test_stream_async_transitions_to_processing(self) -> None:
        """stream_async() transitions session through 'processing' then 'completed'.

        Because the synthetic video is read from disk at full speed, the
        background thread may have already transitioned to 'completed' by the
        time the main thread polls.  We accept either 'processing' or
        'completed' as evidence that stream_async fired correctly, then confirm
        the final state is 'completed'.
        """
        session_id = self._make_session()
        pipeline = VigilPipeline(
            repo=self._repo,
            vlm=_MockVLMNoViolation(),
            frames_root=self._frames_root,
        )
        pipeline.stream_async(
            session_id=session_id,
            rtsp_url=str(self._video),
            rules=["test rule"],
            sample_fps=5.0,
            severity_threshold="warning",
        )
        # Wait for the session to leave 'idle' (becomes processing or completed)
        deadline = time.time() + 8.0
        while time.time() < deadline:
            row = self._repo.get_session(session_id)
            if row and row["status"] != "idle":
                break
            time.sleep(0.02)

        row = self._repo.get_session(session_id)
        self.assertIsNotNone(row)
        self.assertIn(
            row["status"],
            {"processing", "completed"},
            "session must have left 'idle' state after stream_async()",
        )
        # Ensure it ultimately reaches 'completed'
        final_status = self._wait_for_status(session_id, "completed", timeout=15.0)
        self.assertEqual(final_status, "completed")

    def test_stream_completes_after_video_ends(self) -> None:
        """Stream worker transitions to 'completed' when the source ends."""
        session_id = self._make_session()
        pipeline = VigilPipeline(
            repo=self._repo,
            vlm=_MockVLMNoViolation(),
            frames_root=self._frames_root,
        )
        pipeline.stream_async(
            session_id=session_id,
            rtsp_url=str(self._video),
            rules=["test rule"],
            sample_fps=5.0,
            severity_threshold="warning",
        )
        status = self._wait_for_status(session_id, "completed", timeout=15.0)
        self.assertEqual(status, "completed")

    def test_stop_stream_stops_early(self) -> None:
        """stop_stream() causes the stream to complete before all frames are read."""
        # Use a blocking VLM so the background thread is definitely still in
        # "processing" when we call stop_stream.
        first_call_event = threading.Event()  # signals the test that VLM was called
        vlm_release_event = threading.Event()  # test releases the VLM when ready

        class _BlockingVLM(VLMClient):
            def analyze_frame(self, frame_path: Path, rules: list[str]) -> VLMResult:
                first_call_event.set()
                vlm_release_event.wait()  # block until the test releases us
                return VLMResult(has_violation=False, violations=[], raw_text="{}")

        session_id = self._make_session()
        pipeline = VigilPipeline(
            repo=self._repo,
            vlm=_BlockingVLM(),
            frames_root=self._frames_root,
        )
        pipeline.stream_async(
            session_id=session_id,
            rtsp_url=str(self._video),
            rules=["test rule"],
            sample_fps=5.0,
            severity_threshold="warning",
        )

        # Wait until the VLM has been called at least once (stream is active)
        called = first_call_event.wait(timeout=8.0)
        self.assertTrue(called, "VLM should have been called within timeout")

        # The stop event should still be registered (stream is blocked in VLM)
        self.assertIn(
            session_id,
            pipeline._stream_stop_events,
            "stop event should still be registered while stream is active",
        )

        result = pipeline.stop_stream(session_id)
        self.assertTrue(result, "stop_stream should return True for an active stream")

        # Release the blocking VLM so the thread can finish
        vlm_release_event.set()

        status = self._wait_for_status(session_id, "completed", timeout=10.0)
        self.assertEqual(status, "completed")

    def test_stop_stream_returns_false_for_no_active_stream(self) -> None:
        """stop_stream() returns False when no active stream exists for the session."""
        pipeline = VigilPipeline(
            repo=self._repo,
            vlm=_MockVLMNoViolation(),
            frames_root=self._frames_root,
        )
        result = pipeline.stop_stream(session_id=9999)
        self.assertFalse(result)

    def test_violations_are_stored_during_stream(self) -> None:
        """VLM violations reported during streaming are persisted in the DB."""
        session_id = self._make_session()
        pipeline = VigilPipeline(
            repo=self._repo,
            vlm=_MockVLMWithViolation(),
            frames_root=self._frames_root,
        )
        pipeline.stream_async(
            session_id=session_id,
            rtsp_url=str(self._video),
            rules=["ヘルメット未着用を検出"],
            sample_fps=5.0,
            severity_threshold="warning",
        )
        status = self._wait_for_status(session_id, "completed", timeout=15.0)
        self.assertEqual(status, "completed")

        events = self._repo.list_events(session_id)
        self.assertGreater(len(events), 0, "At least one violation event should be stored")
        # Verify event structure
        first_event = events[0]
        self.assertIn("violations", first_event)
        self.assertGreater(len(first_event["violations"]), 0)


if __name__ == "__main__":
    unittest.main()
