"""Unit tests for the decomposed sub-services."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sopilot.config import Settings
from sopilot.constants import (
    DATASET_TARGET_GOLD,
    DATASET_TARGET_TRAINEE,
    DEFAULT_DEVIATION_POLICY,
    DEFAULT_WEIGHTS,
)
from sopilot.database import Database
from sopilot.exceptions import InvalidStateError, NotFoundError
from sopilot.services.embedder import ColorMotionEmbedder
from sopilot.services.scoring_service import ScoringService
from sopilot.services.search_service import SearchService
from sopilot.services.storage import FileStorage
from sopilot.services.task_profile_service import TaskProfileService
from sopilot.services.video_processor import VideoProcessor
from sopilot.services.video_service import VideoLoadResult, VideoService


def _make_settings(data_dir: Path, raw_dir: Path) -> Settings:
    return Settings(
        data_dir=data_dir,
        raw_video_dir=raw_dir,
        ui_dir=Path(__file__).resolve().parents[1] / "sopilot" / "ui",
        database_path=data_dir / "sopilot.db",
        embedder_backend="color-motion",
        primary_task_id="task-test",
        primary_task_name="Test Task",
        enforce_primary_task=True,
    )


def _make_video(path: Path, colors: list[tuple[int, int, int]]) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 8.0, (96, 96))
    for color in colors:
        frame = np.full((96, 96, 3), color, dtype=np.uint8)
        for _ in range(24):
            writer.write(frame)
    writer.release()


class _Env:
    """Shared test environment setup."""

    def __init__(self, tmp_dir: str) -> None:
        root = Path(tmp_dir)
        self.root = root
        self.data_dir = root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.settings = _make_settings(self.data_dir, self.raw_dir)
        self.db = Database(self.settings.database_path)
        self.storage = FileStorage(self.settings.raw_video_dir)
        self.processor = VideoProcessor(
            sample_fps=self.settings.sample_fps,
            clip_seconds=self.settings.clip_seconds,
            frame_size=self.settings.frame_size,
            embedder=ColorMotionEmbedder(),
        )
        self.task_profile_svc = TaskProfileService(self.settings, self.db)
        self.video_svc = VideoService(
            self.settings, self.db, self.storage, self.processor
        )
        self.search_svc = SearchService(self.db)
        self.scoring_svc = ScoringService(
            self.settings,
            self.db,
            self.video_svc,
            self.task_profile_svc.get_task_profile_for,
        )

    def ingest(self, filename: str, colors: list[tuple[int, int, int]], is_gold: bool) -> dict:
        video_file = self.root / filename
        _make_video(video_file, colors)
        with video_file.open("rb") as fh:
            return self.video_svc.ingest_video(
                original_filename=filename,
                file_obj=fh,
                task_id="task-test",
                site_id="site-a",
                camera_id=None,
                operator_id_hash=None,
                recorded_at=None,
                is_gold=is_gold,
            )


# ---- TaskProfileService ----


class TaskProfileServiceTests(unittest.TestCase):
    def test_ensure_primary_creates_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            profile = env.task_profile_svc.get_task_profile()
            self.assertEqual(profile["task_id"], "task-test")
            self.assertEqual(profile["default_weights"], DEFAULT_WEIGHTS)
            self.assertEqual(profile["deviation_policy"], DEFAULT_DEVIATION_POLICY)

    def test_update_task_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            updated = env.task_profile_svc.update_task_profile(
                task_name="New Name",
                pass_score=85.0,
                retrain_score=70.0,
                default_weights=None,
                deviation_policy=None,
            )
            self.assertEqual(updated["task_name"], "New Name")
            self.assertEqual(updated["pass_score"], 85.0)
            self.assertEqual(updated["retrain_score"], 70.0)

    def test_update_rejects_retrain_above_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            with self.assertRaises(InvalidStateError):
                env.task_profile_svc.update_task_profile(
                    task_name=None,
                    pass_score=50.0,
                    retrain_score=60.0,
                    default_weights=None,
                    deviation_policy=None,
                )

    def test_get_task_profile_for_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            with self.assertRaises(NotFoundError):
                env.task_profile_svc.get_task_profile_for("no-such-task")


# ---- VideoService ----


class VideoServiceTests(unittest.TestCase):
    def test_ingest_and_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            result = env.ingest("gold.avi", [(0, 0, 255), (0, 255, 0)], is_gold=True)
            self.assertEqual(result["status"], "ready")
            self.assertTrue(result["is_gold"])
            self.assertGreater(result["clip_count"], 0)

            detail = env.video_svc.get_video_detail(result["video_id"])
            self.assertEqual(detail["video_id"], result["video_id"])
            self.assertTrue(detail["is_gold"])

    def test_original_filename_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            result = env.ingest("my_gold_video.avi", [(0, 0, 255)], is_gold=True)
            # ingest response carries original_filename
            self.assertEqual(result["original_filename"], "my_gold_video.avi")
            # detail response carries original_filename
            detail = env.video_svc.get_video_detail(result["video_id"])
            self.assertEqual(detail["original_filename"], "my_gold_video.avi")
            # list response carries original_filename
            videos = env.video_svc.list_videos(site_id=None, is_gold=None, limit=100)
            self.assertEqual(videos[0]["original_filename"], "my_gold_video.avi")

    def test_get_video_detail_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            with self.assertRaises(NotFoundError):
                env.video_svc.get_video_detail(9999)

    def test_list_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            env.ingest("gold.avi", [(0, 0, 255)], is_gold=True)
            env.ingest("trainee.avi", [(255, 0, 0)], is_gold=False)
            videos = env.video_svc.list_videos(site_id=None, is_gold=None, limit=100)
            self.assertEqual(len(videos), 2)
            gold_count = sum(1 for v in videos if v["is_gold"])
            self.assertEqual(gold_count, 1)

    def test_dataset_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            env.ingest("gold.avi", [(0, 0, 255)], is_gold=True)
            env.ingest("trainee.avi", [(255, 0, 0)], is_gold=False)
            summary = env.video_svc.get_dataset_summary()
            self.assertEqual(summary["gold_videos"], 1)
            self.assertEqual(summary["trainee_videos"], 1)
            self.assertEqual(summary["target"]["gold"], DATASET_TARGET_GOLD)
            self.assertEqual(summary["target"]["trainee"], DATASET_TARGET_TRAINEE)

    def test_stream_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            result = env.ingest("gold.avi", [(0, 0, 255)], is_gold=True)
            path = env.video_svc.get_video_stream_path(result["video_id"])
            self.assertTrue(path.exists())

    def test_stream_path_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            with self.assertRaises(NotFoundError):
                env.video_svc.get_video_stream_path(9999)

    def test_assert_task_allowed_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            with self.assertRaises(InvalidStateError):
                env.video_svc.assert_task_allowed("wrong-task")

    def test_load_video_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            result = env.ingest("gold.avi", [(0, 0, 255), (0, 255, 0)], is_gold=True)
            load_result = env.video_svc.load_video_data(result["video_id"])
            # Returns VideoLoadResult dataclass
            self.assertIsInstance(load_result, VideoLoadResult)
            self.assertEqual(load_result.video["status"], "ready")
            self.assertGreater(len(load_result.clips), 0)
            self.assertEqual(load_result.embeddings.shape[0], len(load_result.clips))
            self.assertIsInstance(load_result.boundaries, list)

    def test_load_video_data_tuple_unpack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            result = env.ingest("gold.avi", [(0, 0, 255), (0, 255, 0)], is_gold=True)
            # Backward-compatible tuple unpacking via __iter__
            video, clips, embeddings, boundaries = env.video_svc.load_video_data(result["video_id"])
            self.assertEqual(video["status"], "ready")
            self.assertGreater(len(clips), 0)
            self.assertEqual(embeddings.shape[0], len(clips))
            self.assertIsInstance(boundaries, list)


# ---- ScoringService ----


class ScoringServiceTests(unittest.TestCase):
    def test_queue_and_run_score_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            gold = env.ingest("gold.avi", [(0, 0, 255), (0, 255, 0)], is_gold=True)
            trainee = env.ingest("trainee.avi", [(0, 0, 255), (255, 0, 0)], is_gold=False)

            job = env.scoring_svc.queue_score_job(
                gold_video_id=gold["video_id"],
                trainee_video_id=trainee["video_id"],
            )
            self.assertEqual(job["status"], "queued")

            env.scoring_svc.run_score_job(job["job_id"])
            completed = env.scoring_svc.get_score_job(job["job_id"])
            self.assertEqual(completed["status"], "completed")
            self.assertIn("score", completed["result"])
            self.assertIn("summary", completed["result"])
            self.assertIn("decision", completed["result"]["summary"])

    def test_score_job_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            with self.assertRaises(NotFoundError):
                env.scoring_svc.get_score_job(9999)

    def test_score_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            gold = env.ingest("gold.avi", [(0, 0, 255), (0, 255, 0)], is_gold=True)
            trainee = env.ingest("trainee.avi", [(0, 0, 255), (255, 0, 0)], is_gold=False)
            job = env.scoring_svc.queue_score_job(
                gold_video_id=gold["video_id"],
                trainee_video_id=trainee["video_id"],
            )
            env.scoring_svc.run_score_job(job["job_id"])
            review = env.scoring_svc.update_score_review(
                job_id=job["job_id"], verdict="pass", note="looks good"
            )
            self.assertEqual(review["verdict"], "pass")
            self.assertEqual(review["note"], "looks good")

    def test_review_uncompleted_job_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            gold = env.ingest("gold.avi", [(0, 0, 255), (0, 255, 0)], is_gold=True)
            trainee = env.ingest("trainee.avi", [(0, 0, 255), (255, 0, 0)], is_gold=False)
            job = env.scoring_svc.queue_score_job(
                gold_video_id=gold["video_id"],
                trainee_video_id=trainee["video_id"],
            )
            with self.assertRaises(InvalidStateError):
                env.scoring_svc.update_score_review(
                    job_id=job["job_id"], verdict="pass", note=None
                )

    def test_export_before_completion_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            gold = env.ingest("gold.avi", [(0, 0, 255), (0, 255, 0)], is_gold=True)
            trainee = env.ingest("trainee.avi", [(0, 0, 255), (255, 0, 0)], is_gold=False)
            job = env.scoring_svc.queue_score_job(
                gold_video_id=gold["video_id"],
                trainee_video_id=trainee["video_id"],
            )
            with self.assertRaises(InvalidStateError):
                env.scoring_svc.export_score_job(job["job_id"])

    def test_validate_rejects_non_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            t1 = env.ingest("t1.avi", [(0, 0, 255)], is_gold=False)
            t2 = env.ingest("t2.avi", [(255, 0, 0)], is_gold=False)
            with self.assertRaises(InvalidStateError):
                env.scoring_svc.queue_score_job(
                    gold_video_id=t1["video_id"],
                    trainee_video_id=t2["video_id"],
                )

    def test_requeue_pending_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            gold = env.ingest("gold.avi", [(0, 0, 255), (0, 255, 0)], is_gold=True)
            trainee = env.ingest("trainee.avi", [(0, 0, 255), (255, 0, 0)], is_gold=False)
            env.scoring_svc.queue_score_job(
                gold_video_id=gold["video_id"],
                trainee_video_id=trainee["video_id"],
            )
            requeued: list[int] = []
            env.scoring_svc.requeue_pending_jobs(requeued.append)
            self.assertEqual(len(requeued), 1)


# ---- SearchService ----


class SearchServiceTests(unittest.TestCase):
    def test_search_returns_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            env.ingest("gold.avi", [(0, 0, 255), (0, 255, 0)], is_gold=True)
            env.ingest("trainee.avi", [(0, 0, 255), (255, 0, 0)], is_gold=False)
            result = env.search_svc.search(
                query_video_id=1, query_clip_index=0, k=5, task_id=None
            )
            self.assertEqual(result["query_video_id"], 1)
            self.assertIsInstance(result["results"], list)

    def test_search_clip_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            with self.assertRaises(NotFoundError):
                env.search_svc.search(
                    query_video_id=9999, query_clip_index=0, k=5, task_id=None
                )


if __name__ == "__main__":
    unittest.main()
