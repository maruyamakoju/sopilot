"""Shared test fixtures for SOPilot test suite."""
import os

# Disable connection pooling in tests to avoid Windows file-lock issues
# during temp directory cleanup.  Must be set before any Database import.
os.environ.setdefault("SOPILOT_DB_POOL_SIZE", "0")

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from sopilot.config import Settings
from sopilot.database import Database
from sopilot.services.embedder import ColorMotionEmbedder
from sopilot.services.sopilot_service import SOPilotService
from sopilot.services.storage import FileStorage
from sopilot.services.video_processor import VideoProcessor


def make_test_video(path: Path, colors: list[tuple], resolution: int = 96, fps: float = 8.0, frames_per_color: int = 24) -> None:
    """Create a synthetic test video with colored segments."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (resolution, resolution))
    for color in colors:
        frame = np.full((resolution, resolution, 3), color, dtype=np.uint8)
        for _ in range(frames_per_color):
            writer.write(frame)
    writer.release()


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def settings(tmp_dir):
    """Create test Settings pointing to a temp directory."""
    data_dir = tmp_dir / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        data_dir=data_dir,
        raw_video_dir=raw_dir,
        ui_dir=Path(__file__).resolve().parents[1] / "sopilot" / "ui",
        database_path=data_dir / "sopilot.db",
        embedder_backend="color-motion",
        primary_task_id="test-task",
        primary_task_name="Test Task",
        enforce_primary_task=True,
    )


@pytest.fixture
def database(settings):
    """Create a fresh Database instance."""
    db = Database(settings.database_path)
    yield db
    db.close()


@pytest.fixture
def service(settings, database):
    """Create a fully wired SOPilotService."""
    storage = FileStorage(settings.raw_video_dir)
    processor = VideoProcessor(
        sample_fps=settings.sample_fps,
        clip_seconds=settings.clip_seconds,
        frame_size=settings.frame_size,
        embedder=ColorMotionEmbedder(),
    )
    return SOPilotService(
        settings=settings,
        database=database,
        storage=storage,
        video_processor=processor,
    )


@pytest.fixture
def gold_and_trainee(tmp_dir, service):
    """Ingest a gold and trainee video pair, returning (gold_payload, trainee_payload)."""
    gold_file = tmp_dir / "gold.avi"
    trainee_file = tmp_dir / "trainee.avi"
    make_test_video(gold_file, [(0, 0, 255), (0, 255, 0)])
    make_test_video(trainee_file, [(0, 0, 255), (255, 0, 0)])

    with gold_file.open("rb") as fh:
        gold = service.ingest_video(
            original_filename="gold.avi", file_obj=fh,
            task_id="test-task", site_id=None, camera_id=None,
            operator_id_hash=None, recorded_at=None, is_gold=True,
        )
    with trainee_file.open("rb") as fh:
        trainee = service.ingest_video(
            original_filename="trainee.avi", file_obj=fh,
            task_id="test-task", site_id=None, camera_id=None,
            operator_id_hash=None, recorded_at=None, is_gold=False,
        )
    return gold, trainee


def make_test_client(tmp_dir: Path, *, api_key: str = "", task_id: str = "test-task"):
    """Create a FastAPI TestClient with env vars pointing to tmp_dir."""
    os.environ["SOPILOT_DATA_DIR"] = str(tmp_dir / "data")
    os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
    os.environ["SOPILOT_PRIMARY_TASK_ID"] = task_id
    if api_key:
        os.environ["SOPILOT_API_KEY"] = api_key
    else:
        os.environ.pop("SOPILOT_API_KEY", None)
    from fastapi.testclient import TestClient

    from sopilot.main import create_app
    return TestClient(create_app())
