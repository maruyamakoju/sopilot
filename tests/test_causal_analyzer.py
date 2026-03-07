"""Tests for Phase 44: CausalAnalyzer (causal_analyzer.py).

Tests cover:
    - CausalEvent dataclass + to_dict
    - CausalChain dataclass + to_dict
    - _causal_relevance helper
    - CausalAnalyzer.observe_event (returns ID, buffer growth, thread safety)
    - CausalAnalyzer.analyze (chain building, lookback, causal scoring,
      JP/EN explanations, LLaVA enhancement)
    - CausalAnalyzer.get_recent_chains + get_buffer_size + get_stats
    - Persistence (save/load)
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sopilot.perception.causal_analyzer import (
    CausalAnalyzer,
    CausalChain,
    CausalEvent,
    _causal_relevance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(offset: float = 0.0) -> float:
    """Return current time minus offset seconds."""
    return time.time() - offset


# ---------------------------------------------------------------------------
# CausalEvent
# ---------------------------------------------------------------------------


class TestCausalEvent:
    def test_fields(self):
        ev = CausalEvent(
            event_id="id1",
            event_type="PPE未着用",
            timestamp=100.0,
            camera_id="cam01",
            confidence=0.95,
            description_ja="ヘルメット未着用を検出",
        )
        assert ev.event_id == "id1"
        assert ev.event_type == "PPE未着用"
        assert ev.confidence == 0.95

    def test_to_dict_keys(self):
        ev = CausalEvent(
            event_id="id1",
            event_type="走行",
            timestamp=time.time(),
            camera_id="cam",
        )
        d = ev.to_dict()
        for key in ("event_id", "event_type", "timestamp", "camera_id",
                    "confidence", "description_ja", "seconds_ago"):
            assert key in d

    def test_seconds_ago_non_negative(self):
        ev = CausalEvent(
            event_id="id",
            event_type="test",
            timestamp=time.time() - 5,
            camera_id="cam",
        )
        assert ev.to_dict()["seconds_ago"] >= 0


# ---------------------------------------------------------------------------
# CausalChain
# ---------------------------------------------------------------------------


class TestCausalChain:
    def test_defaults(self):
        chain = CausalChain(
            chain_id="c1",
            incident_type="転倒",
            incident_time=time.time(),
            camera_id="cam01",
        )
        assert chain.chain == []
        assert chain.causal_confidence == 0.0

    def test_to_dict_keys(self):
        chain = CausalChain(
            chain_id="c1",
            incident_type="転倒",
            incident_time=time.time(),
            camera_id="cam01",
            explanation_ja="テスト説明",
            causal_confidence=0.8,
        )
        d = chain.to_dict()
        for key in ("chain_id", "incident_type", "incident_time", "camera_id",
                    "chain", "explanation_ja", "explanation_en",
                    "time_span_seconds", "causal_confidence", "chain_length"):
            assert key in d

    def test_chain_length_in_dict(self):
        ev = CausalEvent("e1", "PPE未着用", time.time(), "cam", 0.9)
        chain = CausalChain(
            chain_id="c1", incident_type="転倒",
            incident_time=time.time(), camera_id="cam01",
            chain=[ev],
        )
        assert chain.to_dict()["chain_length"] == 1


# ---------------------------------------------------------------------------
# _causal_relevance helper
# ---------------------------------------------------------------------------


class TestCausalRelevance:
    def test_direct_match_is_nonzero(self):
        score = _causal_relevance("転倒", "転倒")
        assert score > 0

    def test_known_causal_pair_high_score(self):
        # 走行 → 転倒 is a known causal rule
        score = _causal_relevance("走行", "転倒")
        assert score >= 0.8

    def test_known_causal_pair_ppe(self):
        score = _causal_relevance("ヘルメット", "安全違反")
        assert score >= 0.5

    def test_unrelated_zero(self):
        score = _causal_relevance("xyz_unrelated_event", "abc_incident")
        assert score == 0.0

    def test_returns_float(self):
        assert isinstance(_causal_relevance("走行", "転倒"), float)


# ---------------------------------------------------------------------------
# CausalAnalyzer.observe_event
# ---------------------------------------------------------------------------


class TestObserveEvent:
    def test_returns_uuid_string(self):
        a = CausalAnalyzer()
        eid = a.observe_event("PPE未着用", camera_id="cam01")
        assert isinstance(eid, str)
        assert len(eid) == 36

    def test_buffer_grows(self):
        a = CausalAnalyzer()
        a.observe_event("走行")
        a.observe_event("PPE未着用")
        assert a.get_buffer_size() == 2

    def test_buffer_rolls_when_full(self):
        a = CausalAnalyzer(buffer_size=5)
        for i in range(8):
            a.observe_event(f"event_{i}")
        assert a.get_buffer_size() == 5

    def test_observe_stores_camera_id(self):
        a = CausalAnalyzer()
        a.observe_event("走行", camera_id="cam_XY")
        # inspect buffer
        assert a._buffer[0].camera_id == "cam_XY"

    def test_observe_stores_confidence(self):
        a = CausalAnalyzer()
        a.observe_event("転倒", confidence=0.77)
        assert abs(a._buffer[0].confidence - 0.77) < 1e-5

    def test_observe_default_description(self):
        a = CausalAnalyzer()
        a.observe_event("走行")
        assert "走行" in a._buffer[0].description_ja


# ---------------------------------------------------------------------------
# CausalAnalyzer.analyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_returns_causal_chain(self):
        a = CausalAnalyzer()
        chain = a.analyze("転倒")
        assert isinstance(chain, CausalChain)

    def test_empty_buffer_returns_empty_chain(self):
        a = CausalAnalyzer()
        chain = a.analyze("転倒")
        assert chain.chain == []
        assert "先行イベントは検出されませんでした" in chain.explanation_ja

    def test_causal_events_in_chain(self):
        a = CausalAnalyzer()
        now = time.time()
        a.observe_event("走行", timestamp=now - 5, camera_id="cam01")
        chain = a.analyze("転倒", incident_time=now, camera_id="cam01")
        assert len(chain.chain) > 0

    def test_lookback_window_respected(self):
        """Events older than lookback_seconds should not appear in chain."""
        a = CausalAnalyzer(lookback_seconds=10.0)
        now = time.time()
        # This event is 30s ago — outside the 10s window
        a.observe_event("走行", timestamp=now - 30, camera_id="cam01")
        chain = a.analyze("転倒", incident_time=now, camera_id="cam01")
        assert chain.chain == []

    def test_camera_filter_in_analyze(self):
        """Events from different cameras should not appear in chain if camera_id specified."""
        a = CausalAnalyzer()
        now = time.time()
        a.observe_event("走行", timestamp=now - 5, camera_id="cam_A")
        a.observe_event("PPE未着用", timestamp=now - 3, camera_id="cam_B")
        chain = a.analyze("転倒", incident_time=now, camera_id="cam_A")
        for ev in chain.chain:
            assert ev.camera_id == "cam_A"

    def test_explanation_ja_not_empty(self):
        a = CausalAnalyzer()
        now = time.time()
        a.observe_event("走行", timestamp=now - 5, camera_id="cam01")
        chain = a.analyze("転倒", incident_time=now, camera_id="cam01")
        assert len(chain.explanation_ja) > 5

    def test_explanation_en_not_empty(self):
        a = CausalAnalyzer()
        now = time.time()
        a.observe_event("走行", timestamp=now - 5, camera_id="cam01")
        chain = a.analyze("転倒", incident_time=now, camera_id="cam01")
        assert len(chain.explanation_en) > 5

    def test_chain_sorted_chronologically(self):
        a = CausalAnalyzer()
        now = time.time()
        a.observe_event("PPE未着用", timestamp=now - 15, camera_id="cam01")
        a.observe_event("走行", timestamp=now - 3, camera_id="cam01")
        chain = a.analyze("転倒", incident_time=now, camera_id="cam01")
        if len(chain.chain) >= 2:
            timestamps = [ev.timestamp for ev in chain.chain]
            assert timestamps == sorted(timestamps)

    def test_chain_stored(self):
        a = CausalAnalyzer()
        a.analyze("転倒")
        assert len(a._chains) == 1

    def test_max_chain_length_5(self):
        """Chain should contain at most 5 causal events."""
        a = CausalAnalyzer()
        now = time.time()
        for i in range(10):
            a.observe_event(f"走行", timestamp=now - (10 - i), camera_id="cam01")
        chain = a.analyze("転倒", incident_time=now, camera_id="cam01")
        assert len(chain.chain) <= 5

    def test_causal_confidence_in_range(self):
        a = CausalAnalyzer()
        now = time.time()
        a.observe_event("走行", timestamp=now - 3, camera_id="cam01")
        chain = a.analyze("転倒", incident_time=now, camera_id="cam01")
        assert 0.0 <= chain.causal_confidence <= 1.0

    def test_time_span_calculated(self):
        a = CausalAnalyzer()
        now = time.time()
        a.observe_event("走行", timestamp=now - 10, camera_id="cam01")
        chain = a.analyze("転倒", incident_time=now, camera_id="cam01")
        if chain.chain:
            assert chain.time_span_seconds > 0


# ---------------------------------------------------------------------------
# LLaVA integration
# ---------------------------------------------------------------------------


class TestLLaVAIntegration:
    def test_llava_called_when_chain_nonempty(self):
        mock_llava = MagicMock()
        mock_llava.answer_question.return_value = "走行後に転倒が発生しました。"
        a = CausalAnalyzer(llava_client=mock_llava)
        now = time.time()
        a.observe_event("走行", timestamp=now - 3, camera_id="cam01")
        chain = a.analyze("転倒", incident_time=now, camera_id="cam01")
        if chain.chain:  # only if events found
            mock_llava.answer_question.assert_called()

    def test_llava_failure_falls_back(self):
        mock_llava = MagicMock()
        mock_llava.answer_question.side_effect = RuntimeError("fail")
        a = CausalAnalyzer(llava_client=mock_llava)
        now = time.time()
        a.observe_event("走行", timestamp=now - 3, camera_id="cam01")
        chain = a.analyze("転倒", incident_time=now, camera_id="cam01")
        assert isinstance(chain.explanation_ja, str)


# ---------------------------------------------------------------------------
# get_recent_chains + get_stats
# ---------------------------------------------------------------------------


class TestGetRecentChains:
    def test_returns_list(self):
        a = CausalAnalyzer()
        chains = a.get_recent_chains()
        assert isinstance(chains, list)

    def test_limit_respected(self):
        a = CausalAnalyzer()
        for _ in range(5):
            a.analyze("転倒")
        chains = a.get_recent_chains(limit=3)
        assert len(chains) <= 3

    def test_chains_are_dicts(self):
        a = CausalAnalyzer()
        a.analyze("転倒")
        chains = a.get_recent_chains()
        for c in chains:
            assert isinstance(c, dict)
            assert "incident_type" in c


class TestGetStats:
    def test_stats_structure(self):
        a = CausalAnalyzer()
        stats = a.get_stats()
        for key in ("buffer_events", "chains_stored", "total_observed",
                    "total_analyzed", "lookback_seconds", "llava_available",
                    "uptime_seconds"):
            assert key in stats

    def test_total_observed_increments(self):
        a = CausalAnalyzer()
        a.observe_event("走行")
        a.observe_event("PPE未着用")
        assert a.get_stats()["total_observed"] == 2

    def test_total_analyzed_increments(self):
        a = CausalAnalyzer()
        a.analyze("転倒")
        a.analyze("ppe")
        assert a.get_stats()["total_analyzed"] == 2


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_no_error(self):
        a = CausalAnalyzer()
        a.analyze("転倒")
        a._save()  # no persist_path → silent no-op

    def test_save_and_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chains.json"
            a = CausalAnalyzer(persist_path=path)
            a.analyze("転倒")
            a._save()
            assert path.exists()

    def test_load_missing_silent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            a = CausalAnalyzer(persist_path=path)
            # _load is called in __init__, should not raise
            assert a.get_buffer_size() == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_observe(self):
        a = CausalAnalyzer()
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                a.observe_event(f"event_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_observe_and_analyze(self):
        a = CausalAnalyzer()
        errors: list[Exception] = []

        def obs_worker() -> None:
            try:
                a.observe_event("走行", camera_id="cam01")
            except Exception as e:
                errors.append(e)

        def analyze_worker() -> None:
            try:
                a.analyze("転倒", camera_id="cam01")
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=obs_worker) for _ in range(15)]
            + [threading.Thread(target=analyze_worker) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
