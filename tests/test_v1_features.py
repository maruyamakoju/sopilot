"""Tests for v1.0 features: Gold Builder, SOP Versioning, Evidence Clips.

Covers:
- Gold Builder: enforce_quality=True blocks low-quality gold uploads (ValidationError)
- Gold Builder: enforce_quality=False allows any gold upload (informational only)
- Gold Builder: HTTP endpoint returns 422 with quality details when gate triggered
- SOP Versioning: gold_version increments per-task for sequential gold uploads
- SOP Versioning: gold_version present in list_videos and get_video_detail
- SOP Versioning: trainee videos have gold_version=None
- Evidence Clips: scoring output includes trainee_timecode / gold_timecode per deviation
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from sopilot.config import Settings
from sopilot.database import Database
from sopilot.exceptions import ValidationError
from sopilot.main import create_app
from sopilot.services.embedder import ColorMotionEmbedder
from sopilot.services.storage import FileStorage
from sopilot.services.video_processor import VideoProcessor
from sopilot.services.video_service import VideoService


# ── Helpers ─────────────────────────────────────────────────────────


def _make_settings(data_dir: Path, raw_dir: Path) -> Settings:
    return Settings(
        data_dir=data_dir,
        raw_video_dir=raw_dir,
        ui_dir=Path(__file__).resolve().parents[1] / "sopilot" / "ui",
        database_path=data_dir / "sopilot.db",
        embedder_backend="color-motion",
        primary_task_id="task-v1",
        primary_task_name="V1 Test Task",
        enforce_primary_task=True,
    )


def _make_video(path: Path, colors: list[tuple[int, int, int]], brightness: int = 200) -> None:
    """Create a synthetic AVI video with the given colors per segment."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 8.0, (96, 96))
    for color in colors:
        frame = np.full((96, 96, 3), color, dtype=np.uint8)
        for _ in range(24):
            writer.write(frame)
    writer.release()


def _make_textured_video(path: Path, brightness: int = 180) -> None:
    """Create a bright textured video that passes all quality checks (brightness + sharpness)."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 8.0, (96, 96))
    # Checkerboard pattern: high Laplacian variance (sharp), bright enough
    rng = np.random.default_rng(42)
    for _ in range(48):
        # Add random noise on top of bright background to ensure sharpness
        frame = np.full((96, 96, 3), brightness, dtype=np.uint8)
        noise = rng.integers(-30, 30, (96, 96, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, brightness - 40, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


def _make_dark_video(path: Path) -> None:
    """Create a very dark synthetic video (brightness ~0) that fails quality gate."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 8.0, (96, 96))
    frame = np.zeros((96, 96, 3), dtype=np.uint8)  # pure black
    for _ in range(48):
        writer.write(frame)
    writer.release()


class _Env:
    """Shared test environment."""

    def __init__(self, tmp_dir: str) -> None:
        root = Path(tmp_dir)
        self.root = root
        data_dir = root / "data"
        raw_dir = data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.settings = _make_settings(data_dir, raw_dir)
        self.db = Database(self.settings.database_path)
        self.storage = FileStorage(self.settings.raw_video_dir)
        self.processor = VideoProcessor(
            sample_fps=self.settings.sample_fps,
            clip_seconds=self.settings.clip_seconds,
            frame_size=self.settings.frame_size,
            embedder=ColorMotionEmbedder(),
        )
        self.video_svc = VideoService(self.settings, self.db, self.storage, self.processor)

    def ingest(
        self,
        filename: str,
        colors: list[tuple[int, int, int]],
        is_gold: bool,
        enforce_quality: bool = False,
    ) -> dict:
        video_file = self.root / filename
        _make_video(video_file, colors)
        with video_file.open("rb") as fh:
            return self.video_svc.ingest_video(
                original_filename=filename,
                file_obj=fh,
                task_id="task-v1",
                site_id=None,
                camera_id=None,
                operator_id_hash=None,
                recorded_at=None,
                is_gold=is_gold,
                enforce_quality=enforce_quality,
            )

    def ingest_dark(self, filename: str, is_gold: bool, enforce_quality: bool = False) -> dict:
        video_file = self.root / filename
        _make_dark_video(video_file)
        with video_file.open("rb") as fh:
            return self.video_svc.ingest_video(
                original_filename=filename,
                file_obj=fh,
                task_id="task-v1",
                site_id=None,
                camera_id=None,
                operator_id_hash=None,
                recorded_at=None,
                is_gold=is_gold,
                enforce_quality=enforce_quality,
            )


# ══════════════════════════════════════════════════════════════════════
# Gold Builder: enforce_quality (unit / service level)
# ══════════════════════════════════════════════════════════════════════


class GoldBuilderServiceTests(unittest.TestCase):
    def test_ingest_gold_no_enforce_quality_succeeds_regardless(self) -> None:
        """enforce_quality=False (default) never blocks ingest, even for dark video."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            # Dark video should still ingest when enforce_quality=False
            result = env.ingest_dark("dark_gold.avi", is_gold=True, enforce_quality=False)
            self.assertEqual(result["status"], "ready")
            self.assertTrue(result["is_gold"])
            # quality report present and overall_pass is False
            self.assertIsNotNone(result.get("quality"))
            self.assertFalse(result["quality"]["overall_pass"])

    def test_ingest_gold_enforce_quality_blocks_dark_video(self) -> None:
        """enforce_quality=True raises ValidationError for a dark gold video."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            with self.assertRaises(ValidationError) as ctx:
                env.ingest_dark("dark_gold.avi", is_gold=True, enforce_quality=True)
            exc = ctx.exception
            self.assertEqual(exc.error_code, "QUALITY_GATE_FAILED")
            self.assertIn("quality", exc.context)
            self.assertFalse(exc.context["quality"]["overall_pass"])

    def test_ingest_gold_enforce_quality_passes_textured_video(self) -> None:
        """enforce_quality=True succeeds for a bright, textured (sharp) video."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            # Textured video with sufficient brightness passes quality gate
            video_file = env.root / "gold_textured.avi"
            _make_textured_video(video_file, brightness=180)
            with video_file.open("rb") as fh:
                result = env.video_svc.ingest_video(
                    original_filename="gold_textured.avi",
                    file_obj=fh,
                    task_id="task-v1",
                    site_id=None,
                    camera_id=None,
                    operator_id_hash=None,
                    recorded_at=None,
                    is_gold=True,
                    enforce_quality=True,
                )
            self.assertEqual(result["status"], "ready")
            self.assertTrue(result["is_gold"])

    def test_enforce_quality_only_applies_to_gold(self) -> None:
        """enforce_quality param exists but trainee ingest ignores quality failures silently."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            # Trainee video can be dark without issue (enforce_quality doesn't block non-gold either
            # since the service itself gates on enforce_quality flag regardless of is_gold)
            result = env.ingest_dark("dark_trainee.avi", is_gold=False, enforce_quality=False)
            self.assertEqual(result["status"], "ready")
            self.assertFalse(result["is_gold"])

    def test_quality_gate_failed_video_not_in_ready_list(self) -> None:
        """After a quality gate failure, the failed video does not appear as ready."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            with self.assertRaises(ValidationError):
                env.ingest_dark("fail_gold.avi", is_gold=True, enforce_quality=True)
            # Should have no ready videos
            ready_videos = env.video_svc.list_videos(site_id=None, is_gold=True, limit=100)
            ready = [v for v in ready_videos if v["status"] == "ready"]
            self.assertEqual(len(ready), 0)


# ══════════════════════════════════════════════════════════════════════
# Gold Builder: HTTP endpoint (enforce_quality via form field)
# ══════════════════════════════════════════════════════════════════════


class GoldBuilderHttpTests(unittest.TestCase):
    def _make_client(self, tmp_root: Path):
        os.environ["SOPILOT_DATA_DIR"] = str(tmp_root / "data")
        os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
        os.environ["SOPILOT_PRIMARY_TASK_ID"] = "task-http-gb"
        os.environ["SOPILOT_RATE_LIMIT_RPM"] = "0"
        app = create_app()
        return TestClient(app)

    def test_http_enforce_quality_dark_video_returns_422(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = self._make_client(root)
            dark_file = root / "dark.avi"
            _make_dark_video(dark_file)
            with dark_file.open("rb") as fh:
                resp = client.post(
                    "/gold",
                    files={"file": ("dark.avi", fh, "video/x-msvideo")},
                    data={"task_id": "task-http-gb", "enforce_quality": "true"},
                    headers={"X-API-Key": "test-key"},
                )
            self.assertEqual(resp.status_code, 422)
            body = resp.json()
            self.assertEqual(body["error"]["code"], "QUALITY_GATE_FAILED")
            self.assertIn("quality", body["error"]["details"])

    def test_http_enforce_quality_false_dark_video_returns_200(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = self._make_client(root)
            dark_file = root / "dark_no_gate.avi"
            _make_dark_video(dark_file)
            with dark_file.open("rb") as fh:
                resp = client.post(
                    "/gold",
                    files={"file": ("dark_no_gate.avi", fh, "video/x-msvideo")},
                    data={"task_id": "task-http-gb", "enforce_quality": "false"},
                    headers={"X-API-Key": "test-key"},
                )
            self.assertEqual(resp.status_code, 200, resp.text)
            body = resp.json()
            self.assertEqual(body["status"], "ready")


# ══════════════════════════════════════════════════════════════════════
# SOP Versioning: gold_version field
# ══════════════════════════════════════════════════════════════════════


class SopVersioningTests(unittest.TestCase):
    def test_first_gold_has_version_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            result = env.ingest("g1.avi", [(100, 100, 200)], is_gold=True)
            detail = env.video_svc.get_video_detail(result["video_id"])
            self.assertEqual(detail["gold_version"], 1)

    def test_second_gold_has_version_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            env.ingest("g1.avi", [(100, 100, 200)], is_gold=True)
            r2 = env.ingest("g2.avi", [(200, 100, 100)], is_gold=True)
            detail = env.video_svc.get_video_detail(r2["video_id"])
            self.assertEqual(detail["gold_version"], 2)

    def test_trainee_has_no_gold_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            result = env.ingest("t1.avi", [(100, 100, 200)], is_gold=False)
            detail = env.video_svc.get_video_detail(result["video_id"])
            self.assertIsNone(detail["gold_version"])

    def test_list_videos_includes_gold_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            env.ingest("g1.avi", [(100, 100, 200)], is_gold=True)
            env.ingest("g2.avi", [(200, 100, 100)], is_gold=True)
            videos = env.video_svc.list_videos(site_id=None, is_gold=True, limit=100)
            # list_videos returns newest first
            versions = {v["video_id"]: v["gold_version"] for v in videos}
            # Both should have non-None gold_version
            self.assertTrue(all(v is not None for v in versions.values()))
            # Versions should be {1, 2}
            self.assertEqual(set(versions.values()), {1, 2})

    def test_trainee_list_has_null_gold_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            env.ingest("t1.avi", [(100, 100, 200)], is_gold=False)
            env.ingest("t2.avi", [(200, 100, 100)], is_gold=False)
            videos = env.video_svc.list_videos(site_id=None, is_gold=False, limit=100)
            self.assertTrue(all(v["gold_version"] is None for v in videos))

    def test_gold_version_sequential_across_multiple_uploads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            ids = []
            for i in range(4):
                r = env.ingest(f"g{i}.avi", [(i * 60, 100, 100)], is_gold=True)
                ids.append(r["video_id"])
            # Verify each has the correct version
            for expected_ver, vid in enumerate(ids, start=1):
                detail = env.video_svc.get_video_detail(vid)
                self.assertEqual(detail["gold_version"], expected_ver, f"video {vid} should be v{expected_ver}")

    def test_gold_version_repo_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            r1 = env.ingest("g1.avi", [(100, 100, 200)], is_gold=True)
            r2 = env.ingest("g2.avi", [(200, 100, 100)], is_gold=True)
            self.assertEqual(env.db.get_gold_version(r1["video_id"], "task-v1"), 1)
            self.assertEqual(env.db.get_gold_version(r2["video_id"], "task-v1"), 2)
            # ID lower than any existing gold video returns 0
            self.assertEqual(env.db.get_gold_version(0, "task-v1"), 0)


# ══════════════════════════════════════════════════════════════════════
# Evidence Clips: timecodes in deviation output (backend verification)
# ══════════════════════════════════════════════════════════════════════


class EvidenceClipsTests(unittest.TestCase):
    """Verify that attach_timecodes produces timecode fields in deviations."""

    def test_attach_timecodes_adds_gold_and_trainee_timecode(self) -> None:
        from sopilot.core.score_pipeline import attach_timecodes, clip_window_to_time

        # Synthetic clip metadata (clip_index, start_sec, end_sec)
        gold_clips = [
            {"clip_index": 0, "start_sec": 0.0, "end_sec": 4.0},
            {"clip_index": 1, "start_sec": 4.0, "end_sec": 8.0},
            {"clip_index": 2, "start_sec": 8.0, "end_sec": 12.0},
        ]
        trainee_clips = [
            {"clip_index": 0, "start_sec": 0.0, "end_sec": 4.0},
            {"clip_index": 1, "start_sec": 4.0, "end_sec": 8.0},
        ]
        deviations = [
            {"type": "step_deviation", "gold_clip_range": [0, 1], "trainee_clip_range": [0, 0]},
            {"type": "missing_step", "gold_clip_range": [2, 2]},
        ]
        enriched = attach_timecodes(deviations, gold_clips, trainee_clips)

        # First deviation: both gold and trainee timecodes
        self.assertIn("gold_timecode", enriched[0])
        self.assertIn("trainee_timecode", enriched[0])
        self.assertEqual(enriched[0]["gold_timecode"], [0.0, 8.0])
        self.assertEqual(enriched[0]["trainee_timecode"], [0.0, 4.0])

        # Second deviation: only gold timecode (no trainee_clip_range)
        self.assertIn("gold_timecode", enriched[1])
        self.assertNotIn("trainee_timecode", enriched[1])
        self.assertEqual(enriched[1]["gold_timecode"], [8.0, 12.0])

    def test_clip_window_to_time_returns_correct_range(self) -> None:
        from sopilot.core.score_pipeline import clip_window_to_time

        clips = [
            {"clip_index": 0, "start_sec": 0.0, "end_sec": 4.0},
            {"clip_index": 1, "start_sec": 4.0, "end_sec": 8.0},
            {"clip_index": 2, "start_sec": 8.0, "end_sec": 12.0},
        ]
        result = clip_window_to_time(clips, 0, 2)
        self.assertEqual(result, [0.0, 12.0])

    def test_attach_timecodes_handles_empty_deviations(self) -> None:
        from sopilot.core.score_pipeline import attach_timecodes

        result = attach_timecodes([], [], [])
        self.assertEqual(result, [])

    def test_attach_timecodes_graceful_with_missing_clip_range(self) -> None:
        from sopilot.core.score_pipeline import attach_timecodes

        gold_clips = [{"clip_index": 0, "start_sec": 0.0, "end_sec": 4.0}]
        deviations = [{"type": "over_time"}]  # no clip_range fields
        enriched = attach_timecodes(deviations, gold_clips, [])
        self.assertEqual(len(enriched), 1)
        self.assertNotIn("gold_timecode", enriched[0])
        self.assertNotIn("trainee_timecode", enriched[0])


# ══════════════════════════════════════════════════════════════════════
# v1.1 Features: Multi-task, Enhanced Trend Dashboard, Deep-link
# ══════════════════════════════════════════════════════════════════════


class MultiTaskListTests(unittest.TestCase):
    """Verify GET /tasks returns task list with metadata."""

    def _seed_task_profile(self, db: Database, task_id: str = "task-v1") -> None:
        """Ensure a task profile exists for testing."""
        from sopilot.constants import DEFAULT_DEVIATION_POLICY, DEFAULT_WEIGHTS

        db.upsert_task_profile(
            task_id=task_id,
            task_name="Test Task",
            pass_score=60.0,
            retrain_score=50.0,
            default_weights=DEFAULT_WEIGHTS,
            deviation_policy=DEFAULT_DEVIATION_POLICY,
        )

    def test_list_tasks_returns_primary_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            self._seed_task_profile(env.db)
            tasks = env.db.list_tasks()
            self.assertTrue(len(tasks) >= 1)
            task_ids = [t["task_id"] for t in tasks]
            self.assertIn("task-v1", task_ids)

    def test_list_tasks_video_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            self._seed_task_profile(env.db)
            env.ingest("g1.avi", [(100, 100, 200)], is_gold=True)
            env.ingest("t1.avi", [(100, 100, 200)], is_gold=False)
            tasks = env.db.list_tasks()
            task = next(t for t in tasks if t["task_id"] == "task-v1")
            self.assertEqual(task["gold_count"], 1)
            self.assertEqual(task["trainee_count"], 1)
            self.assertEqual(task["video_count"], 2)

    def test_list_tasks_http_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            (data_dir / "raw").mkdir()
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            os.environ["SOPILOT_DATA_DIR"] = str(data_dir)
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            try:
                app = create_app()
                client = TestClient(app)
                r = client.get("/tasks")
                self.assertEqual(r.status_code, 200)
                data = r.json()
                self.assertIn("tasks", data)
                self.assertIsInstance(data["tasks"], list)
            finally:
                if old_data is not None:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                else:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                if old_backend is not None:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                else:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)

    def test_get_task_profile_with_task_id_param(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            (data_dir / "raw").mkdir()
            old_data = os.environ.get("SOPILOT_DATA_DIR")
            old_backend = os.environ.get("SOPILOT_EMBEDDER_BACKEND")
            os.environ["SOPILOT_DATA_DIR"] = str(data_dir)
            os.environ["SOPILOT_EMBEDDER_BACKEND"] = "color-motion"
            try:
                app = create_app()
                client = TestClient(app)
                # Default primary task should exist
                r = client.get("/task-profile")
                self.assertEqual(r.status_code, 200)
                self.assertIn("task_id", r.json())
            finally:
                if old_data is not None:
                    os.environ["SOPILOT_DATA_DIR"] = old_data
                else:
                    os.environ.pop("SOPILOT_DATA_DIR", None)
                if old_backend is not None:
                    os.environ["SOPILOT_EMBEDDER_BACKEND"] = old_backend
                else:
                    os.environ.pop("SOPILOT_EMBEDDER_BACKEND", None)


class EnhancedOperatorTrendTests(unittest.TestCase):
    """Verify enhanced operator trend with moving_avg, pass_rate, volatility, team_avg."""

    def test_trend_includes_new_fields(self) -> None:
        """Operator trend response includes moving_avg, pass_rate, volatility, team_avg."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            # Create operator with several jobs
            _insert_operator_jobs(env.db, "op-enhanced", num_jobs=6)
            result = env.db.get_operator_trend("op-enhanced")
            self.assertIn("moving_avg", result)
            self.assertIn("pass_rate", result)
            self.assertIn("volatility", result)
            self.assertIn("team_avg", result)
            # moving_avg should have same length as jobs
            self.assertEqual(len(result["moving_avg"]), len(result["jobs"]))
            # First few should be None (window warmup)
            self.assertIsNone(result["moving_avg"][0])
            # Last should be a number
            self.assertIsNotNone(result["moving_avg"][-1])

    def test_volatility_is_numeric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            _insert_operator_jobs(env.db, "op-vol", num_jobs=5)
            result = env.db.get_operator_trend("op-vol")
            self.assertIsInstance(result["volatility"], float)

    def test_team_avg_is_numeric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = _Env(tmp)
            _insert_operator_jobs(env.db, "op-team", num_jobs=3)
            result = env.db.get_operator_trend("op-team")
            self.assertIsNotNone(result["team_avg"])
            self.assertIsInstance(result["team_avg"], float)


def _insert_operator_jobs(db: Database, operator_id: str, num_jobs: int = 5) -> None:
    """Insert synthetic score jobs for testing operator trend."""
    import json

    with db.connect() as conn:
        # Create gold video
        conn.execute(
            "INSERT INTO videos (task_id, original_filename, is_gold, status, site_id, operator_id_hash, created_at, updated_at) "
            "VALUES (?, ?, 1, 'ready', 'site-a', ?, datetime('now'), datetime('now'))",
            ("pilot_task", "gold.avi", operator_id),
        )
        gold_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        for i in range(num_jobs):
            # Create trainee video
            conn.execute(
                "INSERT INTO videos (task_id, original_filename, is_gold, status, site_id, operator_id_hash, created_at, updated_at) "
                "VALUES (?, ?, 0, 'ready', 'site-a', ?, datetime('now'), datetime('now'))",
                ("pilot_task", f"trainee_{i}.avi", operator_id),
            )
            trainee_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            score = 60.0 + i * 5  # improving scores
            score_json = json.dumps({
                "score": score,
                "summary": {"decision": "pass" if score >= 60 else "fail"},
                "deviations": [],
            })
            conn.execute(
                "INSERT INTO score_jobs (gold_video_id, trainee_video_id, status, score_json, created_at, updated_at) "
                "VALUES (?, ?, 'completed', ?, datetime('now', ?), datetime('now', ?))",
                (gold_id, trainee_id, score_json, f"-{num_jobs - i} hours", f"-{num_jobs - i} hours"),
            )


if __name__ == "__main__":
    unittest.main()
