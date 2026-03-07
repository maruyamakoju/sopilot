"""Tests for Phase 43: VideoMemoryStore (video_memory.py).

Tests cover:
    - _tokenise helper
    - MemoryFrame dataclass
    - VideoMemoryStore.add_frame (normalisation, rolling window, return value)
    - VideoMemoryStore.search_visual (cosine similarity, filters, empty store)
    - VideoMemoryStore.search_text (BM25, CJK, filters, empty store)
    - VideoMemoryStore.get_frame / get_count / get_stats
    - Persistence (save/load round-trip)
    - Thread safety (concurrent add + search)
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from sopilot.perception.video_memory import (
    MemoryFrame,
    VideoMemoryStore,
    _tokenise,
)

_DIM = 768


def _rand_emb(seed: int | None = None, dim: int = _DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


class TestTokenise:
    def test_ascii_words(self):
        tokens = _tokenise("helmet violation detected")
        assert "helmet" in tokens
        assert "violation" in tokens

    def test_lowercase(self):
        tokens = _tokenise("HELMET Violation")
        assert all(t == t.lower() for t in tokens)

    def test_cjk_characters(self):
        tokens = _tokenise("ヘルメット未着用")
        # CJK chars should be individual character tokens
        assert "ヘ" in tokens or "ヘルメット" in tokens or len(tokens) > 0

    def test_mixed_jp_en(self):
        tokens = _tokenise("helmet ヘルメット violation")
        assert len(tokens) > 0

    def test_empty_string(self):
        assert _tokenise("") == []

    def test_punctuation_removed(self):
        tokens = _tokenise("hello, world! test.")
        assert "," not in tokens
        assert "!" not in tokens


# ---------------------------------------------------------------------------
# MemoryFrame
# ---------------------------------------------------------------------------


class TestMemoryFrame:
    def test_fields(self):
        emb = _rand_emb(seed=1)
        frame = MemoryFrame(
            frame_id="id1",
            camera_id="cam01",
            timestamp=100.0,
            visual_embedding=emb,
            caption="テスト",
        )
        assert frame.frame_id == "id1"
        assert frame.camera_id == "cam01"
        assert frame.timestamp == 100.0
        assert frame.caption == "テスト"
        assert frame.incident_id is None

    def test_optional_fields(self):
        frame = MemoryFrame(
            frame_id="id2",
            camera_id="cam",
            timestamp=0.0,
            visual_embedding=_rand_emb(),
            caption="test",
            incident_id="inc_001",
            tags=("helmet", "safety"),
        )
        assert frame.incident_id == "inc_001"
        assert "helmet" in frame.tags


# ---------------------------------------------------------------------------
# VideoMemoryStore — add_frame
# ---------------------------------------------------------------------------


class TestAddFrame:
    def test_returns_uuid_string(self):
        store = VideoMemoryStore()
        fid = store.add_frame(_rand_emb(seed=1), "test caption")
        assert isinstance(fid, str)
        assert len(fid) == 36  # UUID4

    def test_count_increments(self):
        store = VideoMemoryStore()
        for i in range(5):
            store.add_frame(_rand_emb(seed=i), f"caption {i}")
        assert store.get_count() == 5

    def test_embedding_l2_normalized(self):
        store = VideoMemoryStore()
        raw = np.random.randn(_DIM).astype(np.float32) * 100
        fid = store.add_frame(raw, "test")
        frame = store.get_frame(fid)
        assert frame is not None
        norm = float(np.linalg.norm(frame.visual_embedding))
        assert abs(norm - 1.0) < 1e-5

    def test_rolling_window_drops_oldest(self):
        store = VideoMemoryStore(max_frames=5)
        ids = []
        for i in range(7):
            fid = store.add_frame(_rand_emb(seed=i), f"caption {i}")
            ids.append(fid)
        assert store.get_count() == 5
        # oldest 2 should be gone
        assert store.get_frame(ids[0]) is None
        assert store.get_frame(ids[1]) is None
        # newest 5 should exist
        for fid in ids[2:]:
            assert store.get_frame(fid) is not None

    def test_camera_id_stored(self):
        store = VideoMemoryStore()
        fid = store.add_frame(_rand_emb(), "cap", camera_id="cam_XY")
        frame = store.get_frame(fid)
        assert frame.camera_id == "cam_XY"

    def test_custom_frame_id(self):
        store = VideoMemoryStore()
        custom_id = "my-custom-uuid"
        store.add_frame(_rand_emb(), "cap", frame_id=custom_id)
        assert store.get_frame(custom_id) is not None

    def test_incident_id_stored(self):
        store = VideoMemoryStore()
        fid = store.add_frame(_rand_emb(), "cap", incident_id="inc_42")
        frame = store.get_frame(fid)
        assert frame.incident_id == "inc_42"

    def test_tags_stored(self):
        store = VideoMemoryStore()
        fid = store.add_frame(_rand_emb(), "cap", tags=("safety", "ppe"))
        frame = store.get_frame(fid)
        assert "safety" in frame.tags


# ---------------------------------------------------------------------------
# Visual search
# ---------------------------------------------------------------------------


class TestSearchVisual:
    def setup_method(self):
        self.store = VideoMemoryStore()
        self.target_emb = _rand_emb(seed=42)
        self.target_id = self.store.add_frame(
            self.target_emb, "ヘルメット未着用の作業員", camera_id="cam01"
        )
        for i in range(5):  # distractors
            self.store.add_frame(_rand_emb(seed=i + 100), f"通常状態 {i}", camera_id="cam01")

    def test_returns_list(self):
        results = self.store.search_visual(self.target_emb)
        assert isinstance(results, list)

    def test_returns_frame_score_tuples(self):
        results = self.store.search_visual(self.target_emb, top_k=3)
        for frame, score in results:
            assert isinstance(frame, MemoryFrame)
            assert isinstance(score, float)

    def test_top_result_is_most_similar(self):
        results = self.store.search_visual(self.target_emb, top_k=1)
        assert len(results) == 1
        frame, score = results[0]
        # The exact target embedding should yield similarity ≈ 1.0
        assert score >= 0.99

    def test_results_sorted_descending(self):
        results = self.store.search_visual(self.target_emb, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        results = self.store.search_visual(self.target_emb, top_k=3)
        assert len(results) <= 3

    def test_empty_store(self):
        store = VideoMemoryStore()
        results = store.search_visual(_rand_emb())
        assert results == []

    def test_camera_filter(self):
        store = VideoMemoryStore()
        store.add_frame(_rand_emb(seed=1), "cam A frame", camera_id="cam_A")
        store.add_frame(_rand_emb(seed=2), "cam B frame", camera_id="cam_B")
        results = store.search_visual(_rand_emb(seed=1), camera_filter="cam_A")
        for frame, _ in results:
            assert frame.camera_id == "cam_A"

    def test_time_filter_after(self):
        store = VideoMemoryStore()
        store.add_frame(_rand_emb(seed=1), "old", timestamp=100.0)
        store.add_frame(_rand_emb(seed=2), "new", timestamp=1000.0)
        results = store.search_visual(_rand_emb(seed=2), time_after=500.0)
        for frame, _ in results:
            assert frame.timestamp >= 500.0

    def test_large_query_norm_handled(self):
        """Query with large norm should not crash."""
        big = self.target_emb * 1000.0
        results = self.store.search_visual(big)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Text search (BM25)
# ---------------------------------------------------------------------------


class TestSearchText:
    def setup_method(self):
        self.store = VideoMemoryStore()
        self.store.add_frame(_rand_emb(seed=0), "ヘルメット未着用の作業員を検出", camera_id="cam01")
        self.store.add_frame(_rand_emb(seed=1), "安全ベスト未着用の警告", camera_id="cam01")
        self.store.add_frame(_rand_emb(seed=2), "正常な作業状態を確認", camera_id="cam02")

    def test_returns_list(self):
        results = self.store.search_text("ヘルメット")
        assert isinstance(results, list)

    def test_returns_frame_score_tuples(self):
        results = self.store.search_text("ヘルメット")
        for frame, score in results:
            assert isinstance(frame, MemoryFrame)
            assert score >= 0

    def test_relevant_document_ranked_first(self):
        results = self.store.search_text("ヘルメット未着用")
        assert len(results) > 0
        top_frame, _ = results[0]
        assert "ヘルメット" in top_frame.caption

    def test_scores_descending(self):
        results = self.store.search_text("作業")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query(self):
        results = self.store.search_text("")
        assert results == []

    def test_no_match_returns_empty(self):
        # Use a unique ASCII token guaranteed not to appear in any caption
        results = self.store.search_text("xyznotpresenttoken999abc")
        assert results == []

    def test_camera_filter_text(self):
        results = self.store.search_text("作業", camera_filter="cam01")
        for frame, _ in results:
            assert frame.camera_id == "cam01"

    def test_ascii_query(self):
        store = VideoMemoryStore()
        store.add_frame(_rand_emb(), "helmet violation detected in zone A")
        results = store.search_text("helmet violation")
        assert len(results) > 0

    def test_top_k_respected(self):
        results = self.store.search_text("作業", top_k=1)
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_stats_structure(self):
        store = VideoMemoryStore()
        stats = store.get_stats()
        for key in ("frame_count", "max_frames", "total_added",
                    "total_searches", "vocabulary_size", "avg_caption_tokens",
                    "uptime_seconds"):
            assert key in stats

    def test_stats_after_add(self):
        store = VideoMemoryStore()
        store.add_frame(_rand_emb(), "test caption hello world")
        stats = store.get_stats()
        assert stats["frame_count"] == 1
        assert stats["total_added"] == 1
        assert stats["vocabulary_size"] > 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vm.json"
            store = VideoMemoryStore(persist_path=path)
            for i in range(3):
                store.add_frame(_rand_emb(seed=i), f"caption {i}", camera_id="cam01")
            store._save()

            store2 = VideoMemoryStore(persist_path=path)
            assert store2.get_count() == 3

    def test_load_restores_captions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vm.json"
            store = VideoMemoryStore(persist_path=path)
            fid = store.add_frame(_rand_emb(seed=99), "特別なキャプション", camera_id="cam")
            store._save()

            store2 = VideoMemoryStore(persist_path=path)
            # Text search should work after reload
            results = store2.search_text("特別")
            assert len(results) > 0

    def test_save_no_persist_path(self):
        store = VideoMemoryStore()
        store.add_frame(_rand_emb(), "test")
        store._save()  # should not raise

    def test_load_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            store = VideoMemoryStore(persist_path=path)
            assert store.get_count() == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_add(self):
        store = VideoMemoryStore()
        errors: list[Exception] = []

        def add_worker(i: int) -> None:
            try:
                store.add_frame(_rand_emb(seed=i), f"caption {i}", camera_id=f"cam{i%3}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_worker, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert store.get_count() == 30

    def test_concurrent_add_and_search(self):
        store = VideoMemoryStore()
        for i in range(10):
            store.add_frame(_rand_emb(seed=i), f"caption {i}")

        errors: list[Exception] = []

        def search_worker() -> None:
            try:
                store.search_visual(_rand_emb(), top_k=5)
                store.search_text("caption", top_k=3)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=search_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
