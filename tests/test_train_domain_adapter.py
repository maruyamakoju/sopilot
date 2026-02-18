"""Tests for adapt/train_domain_adapter.py (P2-1).

Covers DistRuntime, _init_dist_runtime, _load_video_rows, _select_source_path,
_infer_dim, _write_skip_report, _parse_args, and main() integration.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from sopilot.adapt.train_domain_adapter import (
    DistRuntime,
    _infer_dim,
    _init_dist_runtime,
    _load_video_rows,
    _parse_args,
    _select_source_path,
    _write_skip_report,
    main,
)


# ---------------------------------------------------------------------------
# DistRuntime
# ---------------------------------------------------------------------------
class TestDistRuntime:
    """Test DistRuntime dataclass methods."""

    def _make_single(self) -> DistRuntime:
        return DistRuntime(rank=0, world_size=1, local_rank=0, torch=None, dist=None, enabled=False)

    def test_all_reduce_np_single(self):
        """Single-process: all_reduce_np returns input unchanged."""
        rt = self._make_single()
        arr = np.array([1.0, 2.0, 3.0])
        result = rt.all_reduce_np(arr)
        np.testing.assert_array_equal(result, arr)

    def test_all_reduce_float_single(self):
        """Single-process: all_reduce_float returns input."""
        rt = self._make_single()
        assert rt.all_reduce_float(3.14) == pytest.approx(3.14)

    def test_broadcast_object_single(self):
        """Single-process: broadcast_object returns input."""
        rt = self._make_single()
        data = {"key": "value"}
        assert rt.broadcast_object(data, src=0) is data

    def test_barrier_single(self):
        """Single-process: barrier is a no-op."""
        rt = self._make_single()
        rt.barrier()

    def test_shutdown_single(self):
        """Single-process: shutdown is a no-op."""
        rt = self._make_single()
        rt.shutdown()

    def test_broadcast_object_returns_identity(self):
        """broadcast_object preserves list type."""
        rt = self._make_single()
        data = [1, 2, 3]
        assert rt.broadcast_object(data) is data


# ---------------------------------------------------------------------------
# _init_dist_runtime
# ---------------------------------------------------------------------------
class TestInitDistRuntime:
    """Test _init_dist_runtime function."""

    def test_default_single_process(self, monkeypatch):
        """Default env (no RANK/WORLD_SIZE) returns single-process runtime."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        rt = _init_dist_runtime()
        assert rt.rank == 0
        assert rt.world_size == 1
        assert rt.local_rank == 0
        assert rt.enabled is False

    def test_world_size_1_is_single(self, monkeypatch):
        """Explicit WORLD_SIZE=1 returns single-process."""
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        rt = _init_dist_runtime()
        assert rt.world_size == 1
        assert rt.enabled is False

    def test_world_size_gt1_without_torch_raises(self, monkeypatch):
        """WORLD_SIZE>1 without torch raises RuntimeError."""
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_RANK", "0")

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def fail_import(name, *args, **kwargs):
            if name in ("torch", "torch.distributed"):
                raise ImportError("no torch")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fail_import)
        with pytest.raises(RuntimeError, match="torch distributed"):
            _init_dist_runtime()


# ---------------------------------------------------------------------------
# _load_video_rows
# ---------------------------------------------------------------------------
class TestLoadVideoRows:
    """Test _load_video_rows function."""

    def _setup_db(self, db_path: Path, rows: list[tuple]) -> None:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE videos ("
            "id INTEGER PRIMARY KEY, task_id TEXT, role TEXT, "
            "created_at TEXT, raw_embedding_path TEXT, embedding_path TEXT)"
        )
        for row in rows:
            conn.execute(
                "INSERT INTO videos (id, task_id, role, created_at, raw_embedding_path, embedding_path) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                row,
            )
        conn.commit()
        conn.close()

    def test_missing_db_returns_empty(self, tmp_path):
        """Non-existent DB returns empty list."""
        result = _load_video_rows(tmp_path / "nope.db", since="", max_videos=0)
        assert result == []

    def test_empty_table(self, tmp_path):
        """Empty videos table returns empty list."""
        db_path = tmp_path / "sopilot.db"
        self._setup_db(db_path, [])
        result = _load_video_rows(db_path, since="", max_videos=0)
        assert result == []

    def test_no_embeddings(self, tmp_path):
        """Videos without embeddings are excluded."""
        db_path = tmp_path / "sopilot.db"
        self._setup_db(db_path, [(1, "t1", "ref", "2026-01-01T00:00:00", "", "")])
        result = _load_video_rows(db_path, since="", max_videos=0)
        assert result == []

    def test_raw_embedding(self, tmp_path):
        """Video with raw_embedding_path is returned."""
        db_path = tmp_path / "sopilot.db"
        self._setup_db(db_path, [(1, "t1", "ref", "2026-01-01T00:00:00", "/data/emb.npy", "")])
        result = _load_video_rows(db_path, since="", max_videos=0)
        assert len(result) == 1
        assert result[0]["raw_embedding_path"] == "/data/emb.npy"

    def test_effective_embedding(self, tmp_path):
        """Video with embedding_path is returned."""
        db_path = tmp_path / "sopilot.db"
        self._setup_db(db_path, [(1, "t1", "ref", "2026-01-01T00:00:00", "", "/data/eff.npy")])
        result = _load_video_rows(db_path, since="", max_videos=0)
        assert len(result) == 1
        assert result[0]["embedding_path"] == "/data/eff.npy"

    def test_since_filter(self, tmp_path):
        """Only videos created after 'since' are returned."""
        db_path = tmp_path / "sopilot.db"
        self._setup_db(
            db_path,
            [
                (1, "t1", "ref", "2025-01-01T00:00:00", "/a.npy", ""),
                (2, "t2", "ref", "2026-06-01T00:00:00", "/b.npy", ""),
            ],
        )
        result = _load_video_rows(db_path, since="2026-01-01T00:00:00", max_videos=0)
        assert len(result) == 1
        assert result[0]["id"] == 2

    def test_max_videos_limit(self, tmp_path):
        """max_videos limits the number of returned rows."""
        db_path = tmp_path / "sopilot.db"
        rows = [(i, f"t{i}", "ref", f"2026-01-{i:02d}T00:00:00", f"/{i}.npy", "") for i in range(1, 6)]
        self._setup_db(db_path, rows)
        result = _load_video_rows(db_path, since="", max_videos=2)
        assert len(result) == 2

    def test_order_by_created_at(self, tmp_path):
        """Results ordered by created_at ASC."""
        db_path = tmp_path / "sopilot.db"
        self._setup_db(
            db_path,
            [
                (1, "t1", "ref", "2026-06-01T00:00:00", "/a.npy", ""),
                (2, "t2", "ref", "2026-01-01T00:00:00", "/b.npy", ""),
            ],
        )
        result = _load_video_rows(db_path, since="", max_videos=0)
        assert result[0]["id"] == 2
        assert result[1]["id"] == 1


# ---------------------------------------------------------------------------
# _select_source_path
# ---------------------------------------------------------------------------
class TestSelectSourcePath:
    """Test _select_source_path function."""

    def test_raw_mode(self, tmp_path):
        """Mode 'raw' selects raw_embedding_path."""
        f = tmp_path / "raw.npy"
        f.write_bytes(b"x")
        row = {"raw_embedding_path": str(f), "embedding_path": ""}
        assert _select_source_path(row, "raw") == f

    def test_effective_mode(self, tmp_path):
        """Mode 'effective' selects embedding_path."""
        f = tmp_path / "eff.npy"
        f.write_bytes(b"x")
        row = {"raw_embedding_path": "", "embedding_path": str(f)}
        assert _select_source_path(row, "effective") == f

    def test_auto_mode_prefers_raw(self, tmp_path):
        """Mode 'auto' prefers raw_embedding_path."""
        raw = tmp_path / "raw.npy"
        raw.write_bytes(b"x")
        eff = tmp_path / "eff.npy"
        eff.write_bytes(b"x")
        row = {"raw_embedding_path": str(raw), "embedding_path": str(eff)}
        assert _select_source_path(row, "auto") == raw

    def test_auto_mode_falls_back_to_effective(self, tmp_path):
        """Mode 'auto' falls back to embedding_path if raw empty."""
        eff = tmp_path / "eff.npy"
        eff.write_bytes(b"x")
        row = {"raw_embedding_path": "", "embedding_path": str(eff)}
        assert _select_source_path(row, "auto") == eff

    def test_none_when_empty(self):
        """Returns None when both paths empty."""
        row = {"raw_embedding_path": "", "embedding_path": ""}
        assert _select_source_path(row, "auto") is None

    def test_none_when_file_missing(self, tmp_path):
        """Returns None when file doesn't exist."""
        row = {"raw_embedding_path": str(tmp_path / "nonexistent.npy"), "embedding_path": ""}
        assert _select_source_path(row, "raw") is None

    def test_none_values_handled(self):
        """None values in row dict are handled gracefully."""
        row = {"raw_embedding_path": None, "embedding_path": None}
        assert _select_source_path(row, "auto") is None

    def test_whitespace_stripped(self, tmp_path):
        """Whitespace in paths is stripped."""
        f = tmp_path / "emb.npy"
        f.write_bytes(b"x")
        row = {"raw_embedding_path": f"  {f}  ", "embedding_path": ""}
        assert _select_source_path(row, "raw") == f


# ---------------------------------------------------------------------------
# _infer_dim
# ---------------------------------------------------------------------------
class TestInferDim:
    """Test _infer_dim function."""

    def test_valid_2d_file(self, tmp_path):
        """Returns correct dim from valid 2D array."""
        f = tmp_path / "emb.npy"
        np.save(f, np.zeros((5, 128)))
        assert _infer_dim([{"path": str(f)}]) == 128

    def test_skips_missing_file(self, tmp_path):
        """Missing files are skipped."""
        f = tmp_path / "emb.npy"
        np.save(f, np.ones((3, 64)))
        candidates = [
            {"path": str(tmp_path / "nope.npy")},
            {"path": str(f)},
        ]
        assert _infer_dim(candidates) == 64

    def test_skips_1d_file(self, tmp_path):
        """1D arrays are skipped."""
        f = tmp_path / "emb.npy"
        np.save(f, np.zeros((10,)))
        assert _infer_dim([{"path": str(f)}]) == 0

    def test_empty_candidates(self):
        """Empty candidates returns 0."""
        assert _infer_dim([]) == 0

    def test_skips_corrupt_file(self, tmp_path):
        """Corrupt files are skipped."""
        f = tmp_path / "corrupt.npy"
        f.write_bytes(b"not numpy data")
        assert _infer_dim([{"path": str(f)}]) == 0


# ---------------------------------------------------------------------------
# _write_skip_report
# ---------------------------------------------------------------------------
class TestWriteSkipReport:
    """Test _write_skip_report function."""

    def test_rank0_writes_report(self, tmp_path):
        """Rank 0 writes skip report JSON."""
        rt = DistRuntime(rank=0, world_size=1, local_rank=0, torch=None, dist=None, enabled=False)
        report_path = tmp_path / "report.json"
        _write_skip_report(
            report_path=report_path,
            reason="no embeddings",
            runtime=rt,
            total_videos=10,
            selected_videos=0,
            clips_used=0,
        )
        data = json.loads(report_path.read_text())
        assert data["status"] == "skipped"
        assert data["reason"] == "no embeddings"
        assert data["total_videos"] == 10
        assert data["selected_videos"] == 0
        assert "finished_at" in data

    def test_non_rank0_skips(self, tmp_path):
        """Non-rank-0 processes don't write."""
        rt = DistRuntime(rank=1, world_size=2, local_rank=1, torch=None, dist=None, enabled=False)
        report_path = tmp_path / "report.json"
        _write_skip_report(
            report_path=report_path,
            reason="test",
            runtime=rt,
            total_videos=5,
            selected_videos=0,
            clips_used=0,
        )
        assert not report_path.exists()


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------
class TestParseArgs:
    """Test _parse_args function."""

    def test_required_args(self):
        """All required args are parsed."""
        with patch(
            "sys.argv", ["prog", "--job-id", "j1", "--data-dir", "/d", "--models-dir", "/m", "--reports-dir", "/r"]
        ):
            args = _parse_args()
        assert args.job_id == "j1"
        assert args.data_dir == "/d"
        assert args.models_dir == "/m"
        assert args.reports_dir == "/r"

    def test_defaults(self):
        """Default values for optional args."""
        with patch(
            "sys.argv", ["prog", "--job-id", "j1", "--data-dir", "/d", "--models-dir", "/m", "--reports-dir", "/r"]
        ):
            args = _parse_args()
        assert args.since == ""
        assert args.embedding_source == "auto"
        assert args.max_videos == 0
        assert args.min_clips == 2
        assert args.eps == pytest.approx(1e-6)

    def test_custom_args(self):
        """Custom optional args are parsed."""
        with patch(
            "sys.argv",
            [
                "prog",
                "--job-id",
                "j2",
                "--data-dir",
                "/d",
                "--models-dir",
                "/m",
                "--reports-dir",
                "/r",
                "--since",
                "2026-01-01",
                "--embedding-source",
                "raw",
                "--max-videos",
                "50",
                "--min-clips",
                "10",
                "--eps",
                "0.001",
            ],
        ):
            args = _parse_args()
        assert args.since == "2026-01-01"
        assert args.embedding_source == "raw"
        assert args.max_videos == 50
        assert args.min_clips == 10
        assert args.eps == pytest.approx(0.001)

    def test_invalid_embedding_source_rejected(self):
        """Invalid embedding-source choice rejected."""
        with patch(
            "sys.argv",
            [
                "prog",
                "--job-id",
                "j1",
                "--data-dir",
                "/d",
                "--models-dir",
                "/m",
                "--reports-dir",
                "/r",
                "--embedding-source",
                "invalid",
            ],
        ):
            with pytest.raises(SystemExit):
                _parse_args()


# ---------------------------------------------------------------------------
# main() integration (single-process, temp DB + files)
# ---------------------------------------------------------------------------
class TestMain:
    """Integration tests for main() function."""

    def _setup_db(self, db_path: Path, rows: list[tuple]) -> None:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE videos ("
            "id INTEGER PRIMARY KEY, task_id TEXT, role TEXT, "
            "created_at TEXT, raw_embedding_path TEXT, embedding_path TEXT)"
        )
        for row in rows:
            conn.execute(
                "INSERT INTO videos (id, task_id, role, created_at, raw_embedding_path, embedding_path) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                row,
            )
        conn.commit()
        conn.close()

    def test_no_db_writes_skip(self, tmp_path, monkeypatch):
        """main() with no DB writes skip report."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"
        with patch(
            "sys.argv",
            [
                "prog",
                "--job-id",
                "j1",
                "--data-dir",
                str(data_dir),
                "--models-dir",
                str(models_dir),
                "--reports-dir",
                str(reports_dir),
            ],
        ):
            rc = main()
        assert rc == 0
        report = json.loads((reports_dir / "adapt_j1.json").read_text())
        assert report["status"] == "skipped"
        assert report["reason"] == "no embedding artifacts"

    def test_no_embeddings_writes_skip(self, tmp_path, monkeypatch):
        """main() with empty table writes skip report."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        self._setup_db(data_dir / "sopilot.db", [(1, "t1", "ref", "2026-01-01", "", "")])
        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"
        with patch(
            "sys.argv",
            [
                "prog",
                "--job-id",
                "j2",
                "--data-dir",
                str(data_dir),
                "--models-dir",
                str(models_dir),
                "--reports-dir",
                str(reports_dir),
            ],
        ):
            rc = main()
        assert rc == 0
        report = json.loads((reports_dir / "adapt_j2.json").read_text())
        assert report["status"] == "skipped"

    def test_successful_adapter_creation(self, tmp_path, monkeypatch):
        """main() with valid embeddings creates adapter + report."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create embedding files
        emb_dir = tmp_path / "embs"
        emb_dir.mkdir()
        for i in range(3):
            np.save(emb_dir / f"emb{i}.npy", np.random.randn(4, 64).astype(np.float32))

        self._setup_db(
            data_dir / "sopilot.db",
            [(i + 1, f"t{i}", "ref", f"2026-01-0{i + 1}T00:00:00", str(emb_dir / f"emb{i}.npy"), "") for i in range(3)],
        )

        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"
        with patch(
            "sys.argv",
            [
                "prog",
                "--job-id",
                "j3",
                "--data-dir",
                str(data_dir),
                "--models-dir",
                str(models_dir),
                "--reports-dir",
                str(reports_dir),
            ],
        ):
            rc = main()

        assert rc == 0

        # Check adapter
        adapter_path = models_dir / "feature_adapter_j3.npz"
        assert adapter_path.exists()
        data = np.load(adapter_path)
        assert "mean" in data
        assert "std" in data
        assert data["mean"].shape == (64,)
        assert data["std"].shape == (64,)
        assert np.all(data["std"] > 0)

        # Check pointer
        pointer = json.loads((models_dir / "current_adapter.json").read_text())
        assert pointer["job_id"] == "j3"
        assert "adapter_path" in pointer

        # Check report
        report = json.loads((reports_dir / "adapt_j3.json").read_text())
        assert report["status"] == "completed"
        assert report["embedding_dim"] == 64
        assert report["clips_used"] == 12  # 3 videos * 4 clips
        assert report["videos_used"] == 3

    def test_insufficient_clips_writes_skip(self, tmp_path, monkeypatch):
        """main() with too few clips writes skip report."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        emb_dir = tmp_path / "embs"
        emb_dir.mkdir()
        np.save(emb_dir / "emb0.npy", np.random.randn(1, 32).astype(np.float32))

        self._setup_db(
            data_dir / "sopilot.db",
            [(1, "t1", "ref", "2026-01-01T00:00:00", str(emb_dir / "emb0.npy"), "")],
        )

        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"
        with patch(
            "sys.argv",
            [
                "prog",
                "--job-id",
                "j4",
                "--data-dir",
                str(data_dir),
                "--models-dir",
                str(models_dir),
                "--reports-dir",
                str(reports_dir),
                "--min-clips",
                "5",
            ],
        ):
            rc = main()

        assert rc == 0
        report = json.loads((reports_dir / "adapt_j4.json").read_text())
        assert report["status"] == "skipped"
        assert report["reason"] == "insufficient clips"

    def test_adapter_mean_std_correctness(self, tmp_path, monkeypatch):
        """Adapter mean/std matches manual computation."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        emb_dir = tmp_path / "embs"
        emb_dir.mkdir()

        # Deterministic data
        rng = np.random.RandomState(42)
        all_data = []
        for i in range(2):
            mat = rng.randn(3, 16).astype(np.float32)
            np.save(emb_dir / f"emb{i}.npy", mat)
            all_data.append(mat.astype(np.float64))

        combined = np.concatenate(all_data, axis=0)
        expected_mean = combined.mean(axis=0)
        expected_var = np.maximum(combined.var(axis=0), 1e-6)
        expected_std = np.sqrt(expected_var)

        self._setup_db(
            data_dir / "sopilot.db",
            [(i + 1, f"t{i}", "ref", f"2026-01-0{i + 1}T00:00:00", str(emb_dir / f"emb{i}.npy"), "") for i in range(2)],
        )

        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"
        with patch(
            "sys.argv",
            [
                "prog",
                "--job-id",
                "j5",
                "--data-dir",
                str(data_dir),
                "--models-dir",
                str(models_dir),
                "--reports-dir",
                str(reports_dir),
            ],
        ):
            main()

        data = np.load(models_dir / "feature_adapter_j5.npz")
        np.testing.assert_allclose(data["mean"], expected_mean.astype(np.float32), atol=1e-5)
        np.testing.assert_allclose(data["std"], expected_std.astype(np.float32), atol=1e-5)

    def test_since_filter_in_main(self, tmp_path, monkeypatch):
        """main() respects --since argument."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        emb_dir = tmp_path / "embs"
        emb_dir.mkdir()
        np.save(emb_dir / "old.npy", np.random.randn(3, 16).astype(np.float32))
        np.save(emb_dir / "new.npy", np.random.randn(3, 16).astype(np.float32))

        self._setup_db(
            data_dir / "sopilot.db",
            [
                (1, "t1", "ref", "2025-01-01T00:00:00", str(emb_dir / "old.npy"), ""),
                (2, "t2", "ref", "2026-06-01T00:00:00", str(emb_dir / "new.npy"), ""),
            ],
        )

        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"
        with patch(
            "sys.argv",
            [
                "prog",
                "--job-id",
                "j6",
                "--data-dir",
                str(data_dir),
                "--models-dir",
                str(models_dir),
                "--reports-dir",
                str(reports_dir),
                "--since",
                "2026-01-01T00:00:00",
            ],
        ):
            main()

        report = json.loads((reports_dir / "adapt_j6.json").read_text())
        assert report["status"] == "completed"
        assert report["selected_videos"] == 1
        assert report["clips_used"] == 3

    def test_max_videos_in_main(self, tmp_path, monkeypatch):
        """main() respects --max-videos argument."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        emb_dir = tmp_path / "embs"
        emb_dir.mkdir()
        for i in range(5):
            np.save(emb_dir / f"emb{i}.npy", np.random.randn(3, 16).astype(np.float32))

        self._setup_db(
            data_dir / "sopilot.db",
            [
                (i + 1, f"t{i}", "ref", f"2026-01-{i + 1:02d}T00:00:00", str(emb_dir / f"emb{i}.npy"), "")
                for i in range(5)
            ],
        )

        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"
        with patch(
            "sys.argv",
            [
                "prog",
                "--job-id",
                "j7",
                "--data-dir",
                str(data_dir),
                "--models-dir",
                str(models_dir),
                "--reports-dir",
                str(reports_dir),
                "--max-videos",
                "2",
            ],
        ):
            main()

        report = json.loads((reports_dir / "adapt_j7.json").read_text())
        assert report["status"] == "completed"
        assert report["selected_videos"] == 2

    def test_embedding_source_effective(self, tmp_path, monkeypatch):
        """main() with --embedding-source effective uses embedding_path."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        emb_dir = tmp_path / "embs"
        emb_dir.mkdir()
        np.save(emb_dir / "eff.npy", np.random.randn(5, 32).astype(np.float32))

        self._setup_db(
            data_dir / "sopilot.db",
            [(1, "t1", "ref", "2026-01-01T00:00:00", "", str(emb_dir / "eff.npy"))],
        )

        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"
        with patch(
            "sys.argv",
            [
                "prog",
                "--job-id",
                "j8",
                "--data-dir",
                str(data_dir),
                "--models-dir",
                str(models_dir),
                "--reports-dir",
                str(reports_dir),
                "--embedding-source",
                "effective",
            ],
        ):
            rc = main()

        assert rc == 0
        report = json.loads((reports_dir / "adapt_j8.json").read_text())
        assert report["status"] == "completed"
        assert report["clips_used"] == 5
