"""Tests for Phase 43: RAGEngine (rag_engine.py).

Tests cover:
    - RAGResult dataclass + to_dict
    - RAGEngine: query (visual-only, text-only, hybrid)
    - Alpha weighting (visual vs text)
    - Filter propagation (camera, time)
    - generate_answer (with/without LLaVA)
    - full_query pipeline
    - get_stats
    - Query count tracking
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from sopilot.perception.rag_engine import RAGEngine, RAGResult
from sopilot.perception.video_memory import VideoMemoryStore

_DIM = 768


def _rand_emb(seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_store_with_frames(n: int = 5) -> VideoMemoryStore:
    store = VideoMemoryStore()
    for i in range(n):
        store.add_frame(
            _rand_emb(seed=i),
            f"フレーム {i}: ヘルメット未着用の作業員を検出",
            camera_id=f"cam{i % 3}",
            timestamp=float(1000 + i * 10),
        )
    return store


# ---------------------------------------------------------------------------
# RAGResult
# ---------------------------------------------------------------------------


class TestRAGResult:
    def test_defaults(self):
        r = RAGResult(frame_id="id1", camera_id="cam", timestamp=100.0, caption="test")
        assert r.visual_score == 0.0
        assert r.text_score == 0.0
        assert r.combined_score == 0.0
        assert r.incident_id is None

    def test_to_dict_keys(self):
        r = RAGResult(
            frame_id="id1",
            camera_id="cam01",
            timestamp=100.0,
            caption="test caption",
            visual_score=0.8,
            text_score=3.5,
            combined_score=0.75,
        )
        d = r.to_dict()
        for key in ("frame_id", "camera_id", "timestamp", "caption",
                    "visual_score", "text_score", "combined_score",
                    "incident_id", "tags"):
            assert key in d

    def test_to_dict_scores_rounded(self):
        r = RAGResult(
            frame_id="x", camera_id="c", timestamp=0.0,
            caption="t", combined_score=0.123456789,
        )
        d = r.to_dict()
        assert d["combined_score"] == pytest.approx(0.1235, abs=1e-3)


# ---------------------------------------------------------------------------
# RAGEngine init
# ---------------------------------------------------------------------------


class TestRAGEngineInit:
    def test_default_alpha(self):
        store = VideoMemoryStore()
        engine = RAGEngine(store)
        assert engine._alpha == 0.6

    def test_alpha_clamped(self):
        store = VideoMemoryStore()
        engine = RAGEngine(store, alpha=1.5)
        assert engine._alpha == 1.0
        engine2 = RAGEngine(store, alpha=-0.5)
        assert engine2._alpha == 0.0

    def test_no_llava(self):
        store = VideoMemoryStore()
        engine = RAGEngine(store, llava_client=None)
        assert engine._llava is None

    def test_with_llava(self):
        store = VideoMemoryStore()
        mock_llava = MagicMock()
        engine = RAGEngine(store, llava_client=mock_llava)
        assert engine._llava is mock_llava


# ---------------------------------------------------------------------------
# RAGEngine.query — text-only
# ---------------------------------------------------------------------------


class TestQueryTextOnly:
    def setup_method(self):
        self.store = _make_store_with_frames(5)
        self.engine = RAGEngine(self.store, alpha=0.0)  # text-only

    def test_returns_list(self):
        results = self.engine.query("ヘルメット")
        assert isinstance(results, list)

    def test_results_are_rag_result(self):
        results = self.engine.query("ヘルメット")
        for r in results:
            assert isinstance(r, RAGResult)

    def test_sorted_by_combined_score(self):
        results = self.engine.query("ヘルメット 作業員")
        scores = [r.combined_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        results = self.engine.query("ヘルメット", top_k=2)
        assert len(results) <= 2

    def test_no_match_empty(self):
        results = self.engine.query("存在しないXYZ987")
        assert results == []

    def test_camera_filter(self):
        results = self.engine.query("フレーム", camera_filter="cam0")
        for r in results:
            assert r.camera_id == "cam0"

    def test_time_filter(self):
        results = self.engine.query("フレーム", time_after=1030.0)
        for r in results:
            assert r.timestamp >= 1030.0


# ---------------------------------------------------------------------------
# RAGEngine.query — visual-only
# ---------------------------------------------------------------------------


class TestQueryVisualOnly:
    def setup_method(self):
        self.store = _make_store_with_frames(5)
        self.engine = RAGEngine(self.store, alpha=1.0)  # visual-only
        # Add a frame with known embedding
        self.known_emb = _rand_emb(seed=999)
        self.store.add_frame(
            self.known_emb, "特定の映像", camera_id="cam_target", timestamp=9000.0
        )

    def test_visual_returns_results(self):
        results = self.engine.query("", query_visual=self.known_emb, top_k=5)
        assert len(results) > 0

    def test_top_result_is_similar(self):
        results = self.engine.query("", query_visual=self.known_emb, top_k=1)
        assert len(results) == 1
        # The known embedding should match itself highly
        assert results[0].visual_score >= 0.99

    def test_combined_score_equals_visual_when_alpha_1(self):
        results = self.engine.query("", query_visual=self.known_emb, top_k=3)
        for r in results:
            assert abs(r.combined_score - r.visual_score) < 1e-5


# ---------------------------------------------------------------------------
# RAGEngine.query — hybrid
# ---------------------------------------------------------------------------


class TestQueryHybrid:
    def setup_method(self):
        self.store = VideoMemoryStore()
        # Frame that matches both text AND visual
        self.target_emb = _rand_emb(seed=42)
        self.store.add_frame(
            self.target_emb,
            "ヘルメット未着用の危険な状況",
            camera_id="cam01",
            timestamp=500.0,
        )
        for i in range(4):
            self.store.add_frame(
                _rand_emb(seed=i + 200),
                f"通常状態 {i}",
                camera_id="cam01",
                timestamp=float(100 + i),
            )
        self.engine = RAGEngine(self.store, alpha=0.6)

    def test_hybrid_returns_results(self):
        results = self.engine.query("ヘルメット", query_visual=self.target_emb, top_k=5)
        assert len(results) > 0

    def test_hybrid_result_has_both_scores(self):
        results = self.engine.query("ヘルメット", query_visual=self.target_emb, top_k=3)
        for r in results:
            assert r.visual_score >= 0 or r.text_score >= 0
            # combined should be non-negative
            assert r.combined_score >= 0


# ---------------------------------------------------------------------------
# generate_answer
# ---------------------------------------------------------------------------


class TestGenerateAnswer:
    def test_no_results_returns_message(self):
        engine = RAGEngine(VideoMemoryStore())
        answer = engine.generate_answer("ヘルメット", [])
        assert "見つかりませんでした" in answer

    def test_fallback_without_llava(self):
        engine = RAGEngine(VideoMemoryStore(), llava_client=None)
        results = [
            RAGResult(
                frame_id="id1", camera_id="cam01",
                timestamp=time.time(), caption="ヘルメット未着用",
                combined_score=0.9,
            )
        ]
        answer = engine.generate_answer("ヘルメットを探して", results)
        assert "ヘルメット" in answer or "検索結果" in answer

    def test_with_llava_calls_answer_question(self):
        mock_llava = MagicMock()
        mock_llava.answer_question.return_value = "ヘルメット未着用の作業員が1名検出されました。"
        engine = RAGEngine(VideoMemoryStore(), llava_client=mock_llava)
        results = [
            RAGResult(
                frame_id="id1", camera_id="cam01",
                timestamp=time.time(), caption="ヘルメット未着用",
                combined_score=0.9,
            )
        ]
        answer = engine.generate_answer("ヘルメット", results)
        assert "ヘルメット" in answer
        mock_llava.answer_question.assert_called_once()

    def test_llava_failure_falls_back(self):
        mock_llava = MagicMock()
        mock_llava.answer_question.side_effect = RuntimeError("LLaVA failed")
        engine = RAGEngine(VideoMemoryStore(), llava_client=mock_llava)
        results = [
            RAGResult(
                frame_id="id1", camera_id="cam01",
                timestamp=time.time(), caption="test caption",
                combined_score=0.5,
            )
        ]
        answer = engine.generate_answer("query", results)
        assert isinstance(answer, str) and len(answer) > 0


# ---------------------------------------------------------------------------
# full_query
# ---------------------------------------------------------------------------


class TestFullQuery:
    def setup_method(self):
        self.store = _make_store_with_frames(5)
        self.engine = RAGEngine(self.store, alpha=0.0)  # text-only for determinism

    def test_returns_dict(self):
        result = self.engine.full_query("ヘルメット")
        assert isinstance(result, dict)

    def test_result_keys(self):
        result = self.engine.full_query("ヘルメット")
        for key in ("query", "results", "answer", "total_results",
                    "query_time_ms", "memory_size"):
            assert key in result

    def test_query_echoed(self):
        result = self.engine.full_query("テストクエリ")
        assert result["query"] == "テストクエリ"

    def test_results_are_dicts(self):
        result = self.engine.full_query("フレーム")
        for r in result["results"]:
            assert isinstance(r, dict)
            assert "frame_id" in r

    def test_query_time_positive(self):
        result = self.engine.full_query("テスト")
        assert result["query_time_ms"] >= 0

    def test_memory_size_reported(self):
        result = self.engine.full_query("test")
        assert result["memory_size"] == 5


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_stats_structure(self):
        store = VideoMemoryStore()
        engine = RAGEngine(store)
        stats = engine.get_stats()
        for key in ("query_count", "avg_query_ms", "alpha",
                    "memory_size", "llava_available", "uptime_seconds"):
            assert key in stats

    def test_query_count_increments(self):
        store = _make_store_with_frames(3)
        engine = RAGEngine(store, alpha=0.0)
        engine.full_query("test")
        engine.full_query("test2")
        assert engine.get_stats()["query_count"] == 2

    def test_avg_query_ms_positive(self):
        store = _make_store_with_frames(3)
        engine = RAGEngine(store, alpha=0.0)
        engine.full_query("test")
        assert engine.get_stats()["avg_query_ms"] >= 0

    def test_llava_available_false_by_default(self):
        engine = RAGEngine(VideoMemoryStore())
        assert engine.get_stats()["llava_available"] is False

    def test_llava_available_with_client(self):
        engine = RAGEngine(VideoMemoryStore(), llava_client=MagicMock())
        assert engine.get_stats()["llava_available"] is True
