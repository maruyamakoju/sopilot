"""Tests for db.py (P0-2).

Covers:
- Schema initialization (tables, indexes, _ensure_column)
- Video CRUD (create, finalize, update embedding, get, delete, list)
- Ingest job lifecycle (create → running → completed/failed)
- Score job lifecycle (create → running → completed/failed)
- Training job lifecycle (create → running → completed/failed)
- Generalized job helpers (_mark_job_running, _fail_job, _get_job)
- Audit trail query (list_audit_trail, job_status_counts)
- Edge cases (unknown table, invalid column, SQL injection guards)
"""

from __future__ import annotations

import pytest

from sopilot.db import Database, VideoCreateInput


@pytest.fixture()
def db(tmp_path):
    """Create a fresh in-memory-like Database in tmp_path."""
    d = Database(tmp_path / "test.db")
    yield d
    d.close()


def _make_video(db: Database, task_id: str = "t1", role: str = "gold") -> int:
    return db.create_video(
        VideoCreateInput(
            task_id=task_id,
            role=role,
            file_path="/tmp/video.mp4",
            embedding_model="heuristic",
        )
    )


class TestSchemaInit:
    """Test schema creation and column migration."""

    def test_tables_created(self, db):
        cur = db._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = {row["name"] for row in cur.fetchall()}
        assert "videos" in tables
        assert "clips" in tables
        assert "ingest_jobs" in tables
        assert "score_jobs" in tables
        assert "training_jobs" in tables

    def test_indexes_created(self, db):
        cur = db._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row["name"] for row in cur.fetchall()}
        assert "idx_videos_task" in indexes
        assert "idx_clips_video" in indexes
        assert "idx_ingest_jobs_status" in indexes

    def test_ensure_column_unknown_table(self, db):
        with pytest.raises(ValueError, match="unknown table"):
            db._ensure_column("nonexistent", "col", "TEXT")

    def test_ensure_column_invalid_name(self, db):
        with pytest.raises(ValueError, match="invalid column name"):
            db._ensure_column("videos", "1bad-name", "TEXT")

    def test_ensure_column_invalid_ddl(self, db):
        with pytest.raises(ValueError, match="invalid column ddl"):
            db._ensure_column("videos", "new_col", "TEXT; DROP TABLE videos")

    def test_ensure_column_idempotent(self, db):
        # Adding an existing column should be a no-op
        db._ensure_column("videos", "task_id", "TEXT")
        # No error raised


class TestVideoCreateInput:
    """Test the VideoCreateInput dataclass."""

    def test_required_fields(self):
        v = VideoCreateInput(task_id="t1", role="gold", file_path="/tmp/v.mp4", embedding_model="h")
        assert v.task_id == "t1"
        assert v.site_id is None

    def test_optional_fields(self):
        v = VideoCreateInput(
            task_id="t1",
            role="gold",
            file_path="/tmp/v.mp4",
            embedding_model="h",
            site_id="s1",
            camera_id="c1",
            operator_id_hash="abc",
        )
        assert v.site_id == "s1"
        assert v.camera_id == "c1"


class TestVideoCrud:
    """Test video table operations."""

    def test_create_video(self, db):
        vid = _make_video(db)
        assert isinstance(vid, int)
        assert vid > 0

    def test_get_video(self, db):
        vid = _make_video(db, task_id="test-task", role="trainee")
        row = db.get_video(vid)
        assert row is not None
        assert row["task_id"] == "test-task"
        assert row["role"] == "trainee"
        assert row["created_at"] is not None

    def test_get_nonexistent_video(self, db):
        assert db.get_video(99999) is None

    def test_finalize_video(self, db):
        vid = _make_video(db)
        db.finalize_video(vid, "/raw.npy", "/emb.npy", "/meta.json", "heuristic-v1", 10)
        row = db.get_video(vid)
        assert row["raw_embedding_path"] == "/raw.npy"
        assert row["embedding_path"] == "/emb.npy"
        assert row["clip_meta_path"] == "/meta.json"
        assert row["embedding_model"] == "heuristic-v1"
        assert row["num_clips"] == 10

    def test_update_video_embedding(self, db):
        vid = _make_video(db)
        db.update_video_embedding(vid, "/new_emb.npy", "vjepa2")
        row = db.get_video(vid)
        assert row["embedding_path"] == "/new_emb.npy"
        assert row["embedding_model"] == "vjepa2"

    def test_delete_video(self, db):
        vid = _make_video(db)
        deleted = db.delete_video(vid)
        assert deleted is not None
        assert deleted["id"] == vid
        assert db.get_video(vid) is None

    def test_delete_nonexistent_video(self, db):
        assert db.delete_video(99999) is None

    def test_delete_cascades_clips(self, db):
        vid = _make_video(db)
        db.add_clips(
            vid,
            "t1",
            "gold",
            [
                {"clip_idx": 0, "start_sec": 0.0, "end_sec": 4.0},
                {"clip_idx": 1, "start_sec": 4.0, "end_sec": 8.0},
            ],
        )
        db.delete_video(vid)
        cur = db._conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM clips WHERE video_id = ?", (vid,))
        assert cur.fetchone()["c"] == 0


class TestListVideos:
    """Test video listing operations."""

    def test_list_videos_empty(self, db):
        assert db.list_videos() == []

    def test_list_videos_all(self, db):
        _make_video(db, task_id="t1")
        _make_video(db, task_id="t2")
        videos = db.list_videos()
        assert len(videos) == 2

    def test_list_videos_by_task(self, db):
        _make_video(db, task_id="t1")
        _make_video(db, task_id="t1")
        _make_video(db, task_id="t2")
        videos = db.list_videos(task_id="t1")
        assert len(videos) == 2
        assert all(v["task_id"] == "t1" for v in videos)

    def test_list_videos_limit_clamped(self, db):
        for i in range(5):
            _make_video(db, task_id=f"t{i}")
        videos = db.list_videos(limit=3)
        assert len(videos) == 3

    def test_list_videos_limit_min(self, db):
        _make_video(db)
        videos = db.list_videos(limit=0)
        # limit clamped to max(1, ...)
        assert len(videos) == 1

    def test_list_videos_with_artifacts(self, db):
        vid = _make_video(db)
        db.finalize_video(vid, "/raw.npy", "/emb.npy", "/meta.json", "h", 5)
        _make_video(db)  # Not finalized
        result = db.list_videos_with_artifacts()
        assert len(result) == 1
        assert result[0]["id"] == vid

    def test_list_task_videos_with_artifacts(self, db):
        vid1 = _make_video(db, task_id="t1")
        db.finalize_video(vid1, "/raw.npy", "/emb.npy", "/meta.json", "h", 5)
        vid2 = _make_video(db, task_id="t2")
        db.finalize_video(vid2, "/raw.npy", "/emb.npy", "/meta.json", "h", 5)
        result = db.list_task_videos_with_artifacts("t1")
        assert len(result) == 1
        assert result[0]["task_id"] == "t1"

    def test_list_videos_for_training(self, db):
        vid = _make_video(db)
        db.finalize_video(vid, "/raw.npy", "/emb.npy", "/meta.json", "h", 5)
        result = db.list_videos_for_training()
        assert len(result) == 1

    def test_count_videos_since(self, db):
        _make_video(db)
        _make_video(db)
        assert db.count_videos_since(None) == 2
        # All were just created, so counting since epoch should give 2
        assert db.count_videos_since("1970-01-01T00:00:00+00:00") == 2
        # Counting since far future gives 0
        assert db.count_videos_since("2099-01-01T00:00:00+00:00") == 0


class TestClips:
    """Test clip operations."""

    def test_add_clips(self, db):
        vid = _make_video(db)
        db.add_clips(
            vid,
            "t1",
            "gold",
            [
                {"clip_idx": 0, "start_sec": 0.0, "end_sec": 4.0, "quality_flags": "ok"},
                {"clip_idx": 1, "start_sec": 4.0, "end_sec": 8.0},
            ],
        )
        cur = db._conn.cursor()
        cur.execute("SELECT * FROM clips WHERE video_id = ?", (vid,))
        clips = [dict(r) for r in cur.fetchall()]
        assert len(clips) == 2
        assert clips[0]["clip_idx"] == 0
        assert clips[1]["quality_flags"] == ""


class TestIngestJobLifecycle:
    """Test ingest job state machine: queued → running → completed/failed."""

    def _create_job(self, db):
        return db.create_ingest_job(
            task_id="t1",
            role="trainee",
            requested_by="user",
            file_name="v.mp4",
            file_path="/tmp/v.mp4",
            site_id="s1",
            camera_id="c1",
            operator_id_hash="hash1",
        )

    def test_create_ingest_job(self, db):
        job_id = self._create_job(db)
        assert isinstance(job_id, str)
        assert len(job_id) == 36  # UUID format

    def test_queued_state(self, db):
        job_id = self._create_job(db)
        job = db.get_ingest_job(job_id)
        assert job["status"] == "queued"
        assert job["queued_at"] is not None
        assert job["started_at"] is None

    def test_mark_running(self, db):
        job_id = self._create_job(db)
        db.mark_ingest_job_running(job_id)
        job = db.get_ingest_job(job_id)
        assert job["status"] == "running"
        assert job["started_at"] is not None

    def test_complete(self, db):
        job_id = self._create_job(db)
        vid = _make_video(db)
        db.mark_ingest_job_running(job_id)
        db.complete_ingest_job(
            job_id=job_id,
            video_id=vid,
            num_clips=10,
            source_fps=30.0,
            sampled_fps=4.0,
            embedding_model="heuristic",
        )
        job = db.get_ingest_job(job_id)
        assert job["status"] == "completed"
        assert job["video_id"] == vid
        assert job["num_clips"] == 10
        assert job["finished_at"] is not None

    def test_fail(self, db):
        job_id = self._create_job(db)
        db.mark_ingest_job_running(job_id)
        db.fail_ingest_job(job_id, "test error message")
        job = db.get_ingest_job(job_id)
        assert job["status"] == "failed"
        assert job["error_message"] == "test error message"
        assert job["finished_at"] is not None

    def test_get_nonexistent(self, db):
        assert db.get_ingest_job("nonexistent-id") is None


class TestScoreJobLifecycle:
    """Test score job state machine."""

    def test_create_and_complete(self, db):
        vid1 = _make_video(db)
        vid2 = _make_video(db, role="trainee")
        job_id = db.create_score_job(vid1, vid2, requested_by="admin")
        assert db.get_score_job(job_id)["status"] == "queued"

        db.mark_score_job_running(job_id)
        assert db.get_score_job(job_id)["status"] == "running"

        db.complete_score_job(job_id, score=87.5, result_path="/result.json")
        job = db.get_score_job(job_id)
        assert job["status"] == "completed"
        assert job["score"] == pytest.approx(87.5)
        assert job["result_path"] == "/result.json"

    def test_fail(self, db):
        vid = _make_video(db)
        job_id = db.create_score_job(vid, vid, requested_by=None)
        db.fail_score_job(job_id, "error")
        job = db.get_score_job(job_id)
        assert job["status"] == "failed"


class TestTrainingJobLifecycle:
    """Test training job state machine."""

    def test_create_and_complete(self, db):
        job_id = db.create_training_job("nightly", requested_by="scheduler")
        assert db.get_training_job(job_id)["status"] == "queued"

        db.mark_training_job_running(job_id)
        assert db.get_training_job(job_id)["status"] == "running"

        db.complete_training_job(
            job_id,
            status="completed",
            summary_path="/summary.json",
            metrics_json='{"loss": 0.1}',
        )
        job = db.get_training_job(job_id)
        assert job["status"] == "completed"
        assert job["summary_path"] == "/summary.json"
        assert job["metrics_json"] == '{"loss": 0.1}'

    def test_fail(self, db):
        job_id = db.create_training_job("manual", requested_by="admin")
        db.fail_training_job(job_id, "OOM")
        assert db.get_training_job(job_id)["status"] == "failed"

    def test_has_active_training_job(self, db):
        assert db.has_active_training_job() is False
        job_id = db.create_training_job("nightly", requested_by=None)
        assert db.has_active_training_job() is True
        db.mark_training_job_running(job_id)
        assert db.has_active_training_job() is True
        db.complete_training_job(job_id, "completed", "/s", "{}")
        assert db.has_active_training_job() is False

    def test_latest_training_finished_at(self, db):
        assert db.latest_training_finished_at() is None
        job_id = db.create_training_job("nightly", requested_by=None)
        db.mark_training_job_running(job_id)
        db.complete_training_job(job_id, "completed", "/s", "{}")
        finished = db.latest_training_finished_at()
        assert finished is not None


class TestGeneralizedJobHelpers:
    """Test the _mark_job_running, _fail_job, _get_job guards."""

    def test_mark_running_unknown_table(self, db):
        with pytest.raises(ValueError, match="unknown table"):
            db._mark_job_running("evil_table", "id")

    def test_fail_job_unknown_table(self, db):
        with pytest.raises(ValueError, match="unknown table"):
            db._fail_job("evil_table", "id", "msg")

    def test_get_job_unknown_table(self, db):
        with pytest.raises(ValueError, match="unknown table"):
            db._get_job("evil_table", "id")


class TestAuditTrailQuery:
    """Test list_audit_trail and job_status_counts."""

    def test_audit_trail_ordering(self, db):
        db.create_training_job("a", requested_by=None)
        db.create_training_job("b", requested_by=None)
        trail = db.list_audit_trail(limit=10)
        # Most recent first
        assert len(trail) == 2

    def test_audit_trail_limit_clamped(self, db):
        for i in range(5):
            db.create_training_job(f"t{i}", requested_by=None)
        trail = db.list_audit_trail(limit=3)
        assert len(trail) == 3

    def test_job_status_counts_empty(self, db):
        counts = db.job_status_counts()
        assert "ingest" in counts
        assert "score" in counts
        assert "training" in counts
        assert counts["ingest"]["total"] == 0

    def test_job_status_counts_mixed(self, db):
        db.create_training_job("a", requested_by=None)
        job2 = db.create_training_job("b", requested_by=None)
        db.mark_training_job_running(job2)
        db.complete_training_job(job2, "completed", "/s", "{}")

        counts = db.job_status_counts()
        assert counts["training"]["queued"] == 1
        assert counts["training"]["completed"] == 1
        assert counts["training"]["total"] == 2


class TestClose:
    """Test database close."""

    def test_close(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.close()
        # Double close should not raise (sqlite3 handles this)
