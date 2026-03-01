import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from sopilot.config import Settings
from sopilot.database import Database
from sopilot.services.embedder import ColorMotionEmbedder
from sopilot.services.score_queue import ScoreJobQueue
from sopilot.services.sopilot_service import SOPilotService
from sopilot.services.storage import FileStorage
from sopilot.services.video_processor import VideoProcessor


def _make_video(path: Path, colors: list[tuple[int, int, int]]) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 8.0, (96, 96))
    for color in colors:
        frame = np.full((96, 96, 3), color, dtype=np.uint8)
        for _ in range(24):
            writer.write(frame)
    writer.release()


class AsyncScoreQueueTests(unittest.TestCase):
    def test_queue_processes_job_to_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "data"
            raw_dir = data_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            settings = Settings(
                data_dir=data_dir,
                raw_video_dir=raw_dir,
                ui_dir=Path(__file__).resolve().parents[1] / "sopilot" / "ui",
                database_path=data_dir / "sopilot.db",
                embedder_backend="color-motion",
                primary_task_id="task-a",
                primary_task_name="Task A",
                enforce_primary_task=True,
            )

            db = Database(settings.database_path)
            storage = FileStorage(settings.raw_video_dir)
            processor = VideoProcessor(
                sample_fps=settings.sample_fps,
                clip_seconds=settings.clip_seconds,
                frame_size=settings.frame_size,
                embedder=ColorMotionEmbedder(),
            )
            service = SOPilotService(settings=settings, database=db, storage=storage, video_processor=processor)
            queue = ScoreJobQueue(service, worker_count=1)
            queue.start()
            try:
                gold_file = root / "gold.avi"
                trainee_file = root / "trainee.avi"
                _make_video(gold_file, [(0, 0, 255), (0, 255, 0)])
                _make_video(trainee_file, [(0, 0, 255), (255, 0, 0)])

                with gold_file.open("rb") as fh:
                    gold = service.ingest_video(
                        original_filename="gold.avi",
                        file_obj=fh,
                        task_id="task-a",
                        site_id=None,
                        camera_id=None,
                        operator_id_hash=None,
                        recorded_at=None,
                        is_gold=True,
                    )
                with trainee_file.open("rb") as fh:
                    trainee = service.ingest_video(
                        original_filename="trainee.avi",
                        file_obj=fh,
                        task_id="task-a",
                        site_id=None,
                        camera_id=None,
                        operator_id_hash=None,
                        recorded_at=None,
                        is_gold=False,
                    )

                job = service.queue_score_job(
                    gold_video_id=gold["video_id"],
                    trainee_video_id=trainee["video_id"],
                    weights=None,
                )
                queue.enqueue(job["job_id"])

                deadline = time.time() + 10.0
                current = job
                while time.time() < deadline:
                    current = service.get_score_job(job["job_id"])
                    if current["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.1)

                self.assertEqual(current["status"], "completed")
                self.assertIsNotNone(current["result"])
                self.assertIn("score", current["result"])
            finally:
                queue.stop()


if __name__ == "__main__":
    unittest.main()
