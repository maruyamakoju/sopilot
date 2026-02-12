from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from sopilot.vector_index import NpyVectorIndex


def test_versioned_index_activation_swap() -> None:
    with tempfile.TemporaryDirectory() as td:
        idx = NpyVectorIndex(Path(td))

        base_vec = np.array([[1.0, 0.0]], dtype=np.float32)
        base_meta = [{"video_id": 1, "clip_idx": 0, "start_sec": 0.0, "end_sec": 4.0, "role": "gold"}]
        idx.add("task_a", base_vec, base_meta)

        query = np.array([1.0, 0.0], dtype=np.float32)
        current = idx.search("task_a", query, k=5)
        assert len(current) == 1
        assert current[0]["video_id"] == 1

        old_version = idx.current_version()
        staging = idx.create_staging_version()

        new_vec = np.array([[0.0, 1.0]], dtype=np.float32)
        new_meta = [{"video_id": 2, "clip_idx": 0, "start_sec": 0.0, "end_sec": 4.0, "role": "trainee"}]
        idx.add_to_version(staging, "task_a", new_vec, new_meta)

        # Before activate, search still uses old version.
        before = idx.search("task_a", query, k=5)
        assert len(before) == 1
        assert before[0]["video_id"] == 1

        idx.activate_version(staging)

        after = idx.search("task_a", np.array([0.0, 1.0], dtype=np.float32), k=5)
        assert len(after) == 1
        assert after[0]["video_id"] == 2
        assert idx.current_version() != old_version
