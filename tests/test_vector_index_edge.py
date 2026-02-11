from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from sopilot.vector_index import NpyVectorIndex


def _make_meta(video_id: int, n: int, role: str = "gold") -> list[dict]:
    return [
        {
            "video_id": video_id,
            "clip_idx": i,
            "start_sec": float(i * 4),
            "end_sec": float((i + 1) * 4),
            "role": role,
        }
        for i in range(n)
    ]


class TestDimensionMismatch:
    def test_add_raises_on_dimension_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            idx = NpyVectorIndex(Path(td))
            vec_a = np.array([[1.0, 0.0]], dtype=np.float32)
            idx.add("task1", vec_a, _make_meta(1, 1))

            vec_b = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            with pytest.raises(ValueError, match="embedding dimension mismatch"):
                idx.add("task1", vec_b, _make_meta(2, 1))


class TestClear:
    def test_clear_removes_task_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            idx = NpyVectorIndex(Path(td))
            vec = np.array([[1.0, 0.0]], dtype=np.float32)
            idx.add("task1", vec, _make_meta(1, 1))

            results = idx.search("task1", np.array([1.0, 0.0], dtype=np.float32), k=5)
            assert len(results) == 1

            idx.clear()
            results_after = idx.search("task1", np.array([1.0, 0.0], dtype=np.float32), k=5)
            assert len(results_after) == 0


class TestDeleteVersion:
    def test_delete_non_current_version(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            idx = NpyVectorIndex(Path(td))
            staging = idx.create_staging_version()
            vec = np.array([[0.0, 1.0]], dtype=np.float32)
            idx.add_to_version(staging, "task1", vec, _make_meta(1, 1))

            # Delete staging version (not current)
            idx.delete_version(staging)
            version_dir = Path(td) / staging.replace("/", "_")
            # The directory should be removed (or not accessible)

    def test_delete_current_version_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            idx = NpyVectorIndex(Path(td))
            current = idx.current_version()
            vec = np.array([[1.0, 0.0]], dtype=np.float32)
            idx.add("task1", vec, _make_meta(1, 1))

            idx.delete_version(current)
            # Should NOT delete current - data should still be there
            results = idx.search("task1", np.array([1.0, 0.0], dtype=np.float32), k=5)
            assert len(results) == 1

    def test_delete_empty_string_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            idx = NpyVectorIndex(Path(td))
            idx.delete_version("")
            idx.delete_version("  ")


class TestOverwriteTask:
    def test_overwrite_replaces_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            idx = NpyVectorIndex(Path(td))
            vec_a = np.array([[1.0, 0.0]], dtype=np.float32)
            idx.add("task1", vec_a, _make_meta(1, 1))

            vec_b = np.array([[0.0, 1.0]], dtype=np.float32)
            idx.overwrite_task("task1", vec_b, _make_meta(2, 1))

            results = idx.search("task1", np.array([0.0, 1.0], dtype=np.float32), k=5)
            assert len(results) == 1
            assert results[0]["video_id"] == 2

    def test_overwrite_with_empty_removes_task(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            idx = NpyVectorIndex(Path(td))
            vec = np.array([[1.0, 0.0]], dtype=np.float32)
            idx.add("task1", vec, _make_meta(1, 1))

            idx.overwrite_task("task1", np.zeros((0, 2), dtype=np.float32), [])
            results = idx.search("task1", np.array([1.0, 0.0], dtype=np.float32), k=5)
            assert len(results) == 0


class TestSearchExclusion:
    def test_exclude_self_from_results(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            idx = NpyVectorIndex(Path(td))
            vec = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
            meta = [
                {"video_id": 1, "clip_idx": 0, "start_sec": 0.0, "end_sec": 4.0, "role": "gold"},
                {"video_id": 1, "clip_idx": 1, "start_sec": 4.0, "end_sec": 8.0, "role": "gold"},
                {"video_id": 2, "clip_idx": 0, "start_sec": 0.0, "end_sec": 4.0, "role": "trainee"},
            ]
            idx.add("task1", vec, meta)

            # Search excluding video_id=1, clip_idx=0
            results = idx.search(
                "task1",
                np.array([1.0, 0.0], dtype=np.float32),
                k=5,
                exclude_video_id=1,
                exclude_clip_idx=0,
            )
            assert all(
                not (r["video_id"] == 1 and r["clip_idx"] == 0) for r in results
            )
            assert len(results) == 2

    def test_search_empty_index_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            idx = NpyVectorIndex(Path(td))
            results = idx.search("nonexistent_task", np.array([1.0, 0.0], dtype=np.float32), k=5)
            assert results == []


class TestSearchValidation:
    def test_query_must_be_1d(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            idx = NpyVectorIndex(Path(td))
            vec = np.array([[1.0, 0.0]], dtype=np.float32)
            idx.add("task1", vec, _make_meta(1, 1))

            with pytest.raises(ValueError, match="query vector must be 1D"):
                idx.search("task1", np.array([[1.0, 0.0]], dtype=np.float32), k=5)

    def test_query_dimension_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            idx = NpyVectorIndex(Path(td))
            vec = np.array([[1.0, 0.0]], dtype=np.float32)
            idx.add("task1", vec, _make_meta(1, 1))

            with pytest.raises(ValueError, match="dimensions differ"):
                idx.search("task1", np.array([1.0, 0.0, 0.0], dtype=np.float32), k=5)


class TestMultipleTasksIsolation:
    def test_tasks_dont_interfere(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            idx = NpyVectorIndex(Path(td))
            vec_a = np.array([[1.0, 0.0]], dtype=np.float32)
            vec_b = np.array([[0.0, 1.0]], dtype=np.float32)
            idx.add("task_a", vec_a, _make_meta(1, 1))
            idx.add("task_b", vec_b, _make_meta(2, 1))

            results_a = idx.search("task_a", np.array([1.0, 0.0], dtype=np.float32), k=5)
            assert len(results_a) == 1
            assert results_a[0]["video_id"] == 1

            results_b = idx.search("task_b", np.array([0.0, 1.0], dtype=np.float32), k=5)
            assert len(results_b) == 1
            assert results_b[0]["video_id"] == 2
