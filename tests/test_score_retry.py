"""Tests for ScoreJobQueue exponential-backoff retry logic."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from sopilot.config import Settings
from sopilot.database import Database
from sopilot.services.embedder import ColorMotionEmbedder
from sopilot.services.score_queue import ScoreJobQueue
from sopilot.services.sopilot_service import SOPilotService
from sopilot.services.storage import FileStorage
from sopilot.services.video_processor import VideoProcessor


def _make_video(path: Path, colors: list[tuple]) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 8.0, (64, 64))
    for color in colors:
        frame = np.full((64, 64, 3), color, dtype=np.uint8)
        # Write enough frames to produce at least 2 clips (need >= 8 sec at 8fps = 64 frames)
        for _ in range(48):
            writer.write(frame)
    writer.release()


class _CountingService:
    """Minimal service stub that counts run_score_job calls and fails on demand."""

    def __init__(self, fail_first_n: int = 0):
        self.calls: list[int] = []
        self.fail_first_n = fail_first_n
        self.database = MagicMock()
        self.database.reset_score_job_for_retry = MagicMock()

    def run_score_job(self, job_id: int) -> None:
        self.calls.append(job_id)
        attempt = self.calls.count(job_id)
        if attempt <= self.fail_first_n:
            raise RuntimeError(f"simulated failure attempt {attempt}")

    def requeue_pending_jobs(self, enqueue_fn) -> None:
        pass


class RetryLogicTests(unittest.TestCase):
    def test_successful_job_no_retry(self):
        svc = _CountingService(fail_first_n=0)
        queue = ScoreJobQueue(svc, worker_count=1, max_retries=2)
        queue.start()
        try:
            queue.enqueue(1)
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if svc.calls.count(1) >= 1:
                    break
                time.sleep(0.05)
        finally:
            queue.stop()
        self.assertEqual(svc.calls.count(1), 1, "Job should run exactly once when it succeeds")

    def test_retry_on_failure(self):
        """Job that fails once should be retried."""
        svc = _CountingService(fail_first_n=1)
        # Override delay to zero for fast tests
        queue = ScoreJobQueue(svc, worker_count=1, max_retries=2)

        with patch("sopilot.services.score_queue._BASE_RETRY_DELAY_SEC", 0.05):
            queue.start()
            try:
                queue.enqueue(42)
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if svc.calls.count(42) >= 2:
                        break
                    time.sleep(0.05)
            finally:
                queue.stop()

        self.assertGreaterEqual(svc.calls.count(42), 2, "Job should have been attempted at least twice")

    def test_db_reset_called_on_retry(self):
        """reset_score_job_for_retry should be called once per retry attempt."""
        svc = _CountingService(fail_first_n=1)

        with patch("sopilot.services.score_queue._BASE_RETRY_DELAY_SEC", 0.05):
            queue = ScoreJobQueue(svc, worker_count=1, max_retries=2)
            queue.start()
            try:
                queue.enqueue(7)
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if svc.calls.count(7) >= 2:
                        break
                    time.sleep(0.05)
            finally:
                queue.stop()

        svc.database.reset_score_job_for_retry.assert_called_with(7)

    def test_permanently_failed_after_max_retries(self):
        """After max_retries exhausted, job should NOT be re-queued again."""
        svc = _CountingService(fail_first_n=99)  # always fail

        with patch("sopilot.services.score_queue._BASE_RETRY_DELAY_SEC", 0.05):
            queue = ScoreJobQueue(svc, worker_count=1, max_retries=1)
            queue.start()
            try:
                queue.enqueue(99)
                # Wait long enough for both attempts + backoff
                time.sleep(0.5)
            finally:
                queue.stop()

        # max_retries=1 → attempt 1 (fail) + 1 retry (fail) = 2 total, no more
        self.assertEqual(svc.calls.count(99), 2,
                         f"Expected 2 attempts (1 original + 1 retry), got {svc.calls.count(99)}")

    def test_dedup_prevents_double_enqueue(self):
        """Enqueueing the same job_id twice while it is pending should be a no-op."""
        barrier_entered = []

        class _SlowService:
            database = MagicMock()

            def run_score_job(self, job_id: int) -> None:
                barrier_entered.append(job_id)
                time.sleep(0.15)

            def requeue_pending_jobs(self, enqueue_fn) -> None:
                pass

        svc = _SlowService()
        queue = ScoreJobQueue(svc, worker_count=1, max_retries=0)
        queue.start()
        try:
            queue.enqueue(5)
            queue.enqueue(5)  # duplicate — should be ignored
            time.sleep(0.4)
        finally:
            queue.stop()

        self.assertEqual(barrier_entered.count(5), 1, "Duplicate enqueue should be ignored while job is pending")

    def test_max_retries_zero_no_retry(self):
        """max_retries=0 means the job fails immediately with no retry."""
        svc = _CountingService(fail_first_n=99)
        queue = ScoreJobQueue(svc, worker_count=1, max_retries=0)

        with patch("sopilot.services.score_queue._BASE_RETRY_DELAY_SEC", 0.05):
            queue.start()
            try:
                queue.enqueue(3)
                time.sleep(0.3)
            finally:
                queue.stop()

        self.assertEqual(svc.calls.count(3), 1, "max_retries=0 should attempt the job exactly once")

    def test_retry_counts_cleared_on_success(self):
        """After a retry succeeds, internal retry state should be cleaned up."""
        svc = _CountingService(fail_first_n=1)

        with patch("sopilot.services.score_queue._BASE_RETRY_DELAY_SEC", 0.05):
            queue = ScoreJobQueue(svc, worker_count=1, max_retries=3)
            queue.start()
            try:
                queue.enqueue(10)
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if svc.calls.count(10) >= 2:
                        break
                    time.sleep(0.05)
                time.sleep(0.1)  # let cleanup settle
            finally:
                queue.stop()

        with queue._lock:
            self.assertNotIn(10, queue._retry_counts,
                             "Retry counts should be cleared after a successful retry")


class IntegrationRetryTests(unittest.TestCase):
    """Full stack test: real DB + service, synthetic failure via mock."""

    def _make_settings(self, tmp_dir: str) -> Settings:
        root = Path(tmp_dir)
        data_dir = root / "data"
        raw_dir = data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        return Settings(
            data_dir=data_dir,
            raw_video_dir=raw_dir,
            ui_dir=Path(__file__).resolve().parents[1] / "sopilot" / "ui",
            database_path=data_dir / "sopilot.db",
            embedder_backend="color-motion",
            primary_task_id="task-retry",
            primary_task_name="Retry Task",
            enforce_primary_task=True,
            score_job_max_retries=1,
        )

    def test_job_completes_after_one_transient_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._make_settings(tmp)
            db = Database(settings.database_path)
            storage = FileStorage(settings.raw_video_dir)
            processor = VideoProcessor(
                sample_fps=settings.sample_fps,
                clip_seconds=settings.clip_seconds,
                frame_size=settings.frame_size,
                embedder=ColorMotionEmbedder(),
            )
            service = SOPilotService(settings=settings, database=db, storage=storage, video_processor=processor)

            # Ingest gold and trainee
            root = Path(tmp)
            gold_file = root / "g.avi"
            trainee_file = root / "t.avi"
            _make_video(gold_file, [(0, 0, 200), (0, 200, 0)])
            _make_video(trainee_file, [(0, 0, 200), (200, 0, 0)])

            with gold_file.open("rb") as fh:
                gold = service.ingest_video(
                    original_filename="g.avi", file_obj=fh,
                    task_id="task-retry", site_id=None, camera_id=None,
                    operator_id_hash=None, recorded_at=None, is_gold=True,
                )
            with trainee_file.open("rb") as fh:
                trainee = service.ingest_video(
                    original_filename="t.avi", file_obj=fh,
                    task_id="task-retry", site_id=None, camera_id=None,
                    operator_id_hash=None, recorded_at=None, is_gold=False,
                )

            job = service.queue_score_job(
                gold_video_id=gold["video_id"],
                trainee_video_id=trainee["video_id"],
            )
            job_id = job["job_id"]

            call_count = [0]
            original_run = service.run_score_job

            def _fail_once(jid):
                call_count[0] += 1
                if call_count[0] == 1:
                    # Only raise — do NOT call db.fail_score_job here.
                    # The queue's _handle_failure will manage the DB state.
                    raise RuntimeError("transient error")
                original_run(jid)

            service.run_score_job = _fail_once

            with patch("sopilot.services.score_queue._BASE_RETRY_DELAY_SEC", 0.05):
                queue = ScoreJobQueue(service, worker_count=1, max_retries=1)
                queue.start()
                try:
                    queue.enqueue(job_id)
                    deadline = time.time() + 8.0
                    current = {"status": "queued"}
                    while time.time() < deadline:
                        current = service.get_score_job(job_id)
                        if current["status"] == "completed":
                            break
                        time.sleep(0.1)
                finally:
                    queue.stop()

            self.assertEqual(current["status"], "completed",
                             f"Job should complete after one retry. Final status: {current['status']!r}")
            self.assertEqual(call_count[0], 2, "run_score_job should have been called twice")


if __name__ == "__main__":
    unittest.main()
