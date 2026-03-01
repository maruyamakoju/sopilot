"""Tests for sopilot.database.Database — schema, CRUD, and job lifecycle."""
import sqlite3

import pytest

from sopilot.database import Database

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    """Return a fresh database path inside a temp directory."""
    return tmp_path / "test_sopilot.db"


@pytest.fixture
def db(db_path):
    """Return a freshly initialised Database."""
    return Database(db_path)


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

class TestSchema:
    def test_tables_created(self, db, db_path):
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "videos" in tables
        assert "clips" in tables
        assert "score_jobs" in tables
        assert "task_profiles" in tables
        assert "score_reviews" in tables

    def test_videos_table_has_expected_columns(self, db, db_path):
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("PRAGMA table_info(videos)").fetchall()
        col_names = {row["name"] for row in rows}
        conn.close()
        for expected in ("id", "task_id", "is_gold", "file_path", "status",
                         "clip_count", "created_at", "updated_at",
                         "original_filename"):
            assert expected in col_names, f"Missing column: {expected}"

    def test_score_jobs_has_weights_and_timing_columns(self, db, db_path):
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("PRAGMA table_info(score_jobs)").fetchall()
        col_names = {row["name"] for row in rows}
        conn.close()
        assert "weights_json" in col_names
        assert "started_at" in col_names
        assert "finished_at" in col_names


# ---------------------------------------------------------------------------
# _ensure_column
# ---------------------------------------------------------------------------

class TestEnsureColumn:
    def test_adds_column_when_missing(self, db, db_path):
        with db.connect() as conn:
            db._ensure_column(conn, "videos", "test_col", "TEXT")
            rows = conn.execute("PRAGMA table_info(videos)").fetchall()
            col_names = {row["name"] for row in rows}
        assert "test_col" in col_names

    def test_idempotent_when_column_exists(self, db, db_path):
        with db.connect() as conn:
            db._ensure_column(conn, "videos", "test_col2", "TEXT")
            # Running again should not raise
            db._ensure_column(conn, "videos", "test_col2", "TEXT")
            rows = conn.execute("PRAGMA table_info(videos)").fetchall()
            col_names = [row["name"] for row in rows]
        # Column should appear exactly once
        assert col_names.count("test_col2") == 1


# ---------------------------------------------------------------------------
# insert_video / get_video roundtrip
# ---------------------------------------------------------------------------

class TestVideoRoundtrip:
    def test_insert_and_retrieve(self, db):
        vid = db.insert_video(
            task_id="task-1",
            site_id="site-a",
            camera_id="cam-1",
            operator_id_hash="op123",
            recorded_at="2026-01-01T00:00:00Z",
            is_gold=True,
            original_filename="gold.mp4",
        )
        assert isinstance(vid, int)
        assert vid >= 1

        row = db.get_video(vid)
        assert row is not None
        assert row["task_id"] == "task-1"
        assert row["site_id"] == "site-a"
        assert row["camera_id"] == "cam-1"
        assert row["is_gold"] is True
        assert row["status"] == "processing"
        assert row["original_filename"] == "gold.mp4"

    def test_get_nonexistent_video_returns_none(self, db):
        assert db.get_video(99999) is None

    def test_insert_video_without_optional_fields(self, db):
        vid = db.insert_video(
            task_id="task-2",
            site_id=None,
            camera_id=None,
            operator_id_hash=None,
            recorded_at=None,
            is_gold=False,
        )
        row = db.get_video(vid)
        assert row is not None
        assert row["is_gold"] is False
        assert row["site_id"] is None


# ---------------------------------------------------------------------------
# Score job lifecycle
# ---------------------------------------------------------------------------

class TestScoreJobLifecycle:
    def _insert_pair(self, db):
        gold_id = db.insert_video(
            task_id="t", site_id=None, camera_id=None,
            operator_id_hash=None, recorded_at=None, is_gold=True,
        )
        trainee_id = db.insert_video(
            task_id="t", site_id=None, camera_id=None,
            operator_id_hash=None, recorded_at=None, is_gold=False,
        )
        return gold_id, trainee_id

    def test_create_score_job(self, db):
        gold_id, trainee_id = self._insert_pair(db)
        job_id = db.create_score_job(gold_id, trainee_id)
        assert isinstance(job_id, int)
        assert job_id >= 1

    def test_create_score_job_with_weights(self, db):
        gold_id, trainee_id = self._insert_pair(db)
        weights = {"w_miss": 0.3, "w_swap": 0.2, "w_dev": 0.3, "w_time": 0.2}
        job_id = db.create_score_job(gold_id, trainee_id, weights=weights)
        job = db.get_score_job(job_id)
        assert job is not None
        assert job["weights"] == weights

    def test_claim_score_job(self, db):
        gold_id, trainee_id = self._insert_pair(db)
        job_id = db.create_score_job(gold_id, trainee_id)
        claimed = db.claim_score_job(job_id)
        assert claimed is True
        job = db.get_score_job(job_id)
        assert job["status"] == "running"

    def test_claim_nonexistent_job_returns_false(self, db):
        assert db.claim_score_job(99999) is False

    def test_claim_completed_job_returns_false(self, db):
        gold_id, trainee_id = self._insert_pair(db)
        job_id = db.create_score_job(gold_id, trainee_id)
        db.complete_score_job(job_id, {"score": 95.0})
        claimed = db.claim_score_job(job_id)
        assert claimed is False

    def test_complete_score_job(self, db):
        gold_id, trainee_id = self._insert_pair(db)
        job_id = db.create_score_job(gold_id, trainee_id)
        db.claim_score_job(job_id)
        db.complete_score_job(job_id, {"score": 88.5, "summary": {"decision": "pass"}})
        job = db.get_score_job(job_id)
        assert job["status"] == "completed"
        assert job["score"]["score"] == 88.5

    def test_fail_score_job(self, db):
        gold_id, trainee_id = self._insert_pair(db)
        job_id = db.create_score_job(gold_id, trainee_id)
        db.claim_score_job(job_id)
        db.fail_score_job(job_id, "something broke")
        job = db.get_score_job(job_id)
        assert job["status"] == "failed"
        assert job["error"] == "something broke"


# ---------------------------------------------------------------------------
# list_pending_score_job_ids
# ---------------------------------------------------------------------------

class TestListPendingJobs:
    def _insert_pair(self, db):
        gold_id = db.insert_video(
            task_id="t", site_id=None, camera_id=None,
            operator_id_hash=None, recorded_at=None, is_gold=True,
        )
        trainee_id = db.insert_video(
            task_id="t", site_id=None, camera_id=None,
            operator_id_hash=None, recorded_at=None, is_gold=False,
        )
        return gold_id, trainee_id

    def test_pending_jobs_listed(self, db):
        gold_id, trainee_id = self._insert_pair(db)
        j1 = db.create_score_job(gold_id, trainee_id)
        j2 = db.create_score_job(gold_id, trainee_id)
        pending = db.list_pending_score_job_ids()
        assert j1 in pending
        assert j2 in pending

    def test_completed_jobs_not_in_pending(self, db):
        gold_id, trainee_id = self._insert_pair(db)
        j1 = db.create_score_job(gold_id, trainee_id)
        db.complete_score_job(j1, {"score": 100})
        pending = db.list_pending_score_job_ids()
        assert j1 not in pending

    def test_running_jobs_in_pending(self, db):
        gold_id, trainee_id = self._insert_pair(db)
        j1 = db.create_score_job(gold_id, trainee_id)
        db.claim_score_job(j1)
        pending = db.list_pending_score_job_ids()
        assert j1 in pending


# ---------------------------------------------------------------------------
# reset_score_job_for_retry
# ---------------------------------------------------------------------------

class TestResetScoreJob:
    def _insert_pair(self, db):
        gold_id = db.insert_video(
            task_id="t", site_id=None, camera_id=None,
            operator_id_hash=None, recorded_at=None, is_gold=True,
        )
        trainee_id = db.insert_video(
            task_id="t", site_id=None, camera_id=None,
            operator_id_hash=None, recorded_at=None, is_gold=False,
        )
        return gold_id, trainee_id

    def test_reset_failed_job(self, db):
        gold_id, trainee_id = self._insert_pair(db)
        job_id = db.create_score_job(gold_id, trainee_id)
        db.claim_score_job(job_id)
        db.fail_score_job(job_id, "error")
        db.reset_score_job_for_retry(job_id)
        job = db.get_score_job(job_id)
        assert job["status"] == "queued"
        assert job["error"] is None

    def test_reset_puts_job_back_in_pending(self, db):
        gold_id, trainee_id = self._insert_pair(db)
        job_id = db.create_score_job(gold_id, trainee_id)
        db.claim_score_job(job_id)
        db.fail_score_job(job_id, "error")
        db.reset_score_job_for_retry(job_id)
        pending = db.list_pending_score_job_ids()
        assert job_id in pending


# ---------------------------------------------------------------------------
# count_videos with filters
# ---------------------------------------------------------------------------

class TestCountVideos:
    def test_count_all(self, db):
        db.insert_video(task_id="t1", site_id=None, camera_id=None,
                        operator_id_hash=None, recorded_at=None, is_gold=True)
        db.insert_video(task_id="t1", site_id=None, camera_id=None,
                        operator_id_hash=None, recorded_at=None, is_gold=False)
        assert db.count_videos() == 2

    def test_count_by_task(self, db):
        db.insert_video(task_id="a", site_id=None, camera_id=None,
                        operator_id_hash=None, recorded_at=None, is_gold=True)
        db.insert_video(task_id="b", site_id=None, camera_id=None,
                        operator_id_hash=None, recorded_at=None, is_gold=False)
        assert db.count_videos(task_id="a") == 1
        assert db.count_videos(task_id="b") == 1
        assert db.count_videos(task_id="c") == 0

    def test_count_by_is_gold(self, db):
        db.insert_video(task_id="t", site_id=None, camera_id=None,
                        operator_id_hash=None, recorded_at=None, is_gold=True)
        db.insert_video(task_id="t", site_id=None, camera_id=None,
                        operator_id_hash=None, recorded_at=None, is_gold=False)
        db.insert_video(task_id="t", site_id=None, camera_id=None,
                        operator_id_hash=None, recorded_at=None, is_gold=False)
        assert db.count_videos(is_gold=True) == 1
        assert db.count_videos(is_gold=False) == 2

    def test_count_by_status(self, db):
        db.insert_video(task_id="t", site_id=None, camera_id=None,
                              operator_id_hash=None, recorded_at=None, is_gold=True)
        assert db.count_videos(status="processing") == 1
        assert db.count_videos(status="ready") == 0

    def test_count_empty_database(self, db):
        assert db.count_videos() == 0

    def test_count_combined_filters(self, db):
        db.insert_video(task_id="t1", site_id=None, camera_id=None,
                        operator_id_hash=None, recorded_at=None, is_gold=True)
        db.insert_video(task_id="t1", site_id=None, camera_id=None,
                        operator_id_hash=None, recorded_at=None, is_gold=False)
        db.insert_video(task_id="t2", site_id=None, camera_id=None,
                        operator_id_hash=None, recorded_at=None, is_gold=True)
        assert db.count_videos(task_id="t1", is_gold=True) == 1
        assert db.count_videos(task_id="t1", is_gold=False) == 1
        assert db.count_videos(task_id="t2", is_gold=False) == 0
