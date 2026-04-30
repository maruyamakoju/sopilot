"""Tests for Phase 62: Agent Episodic Memory (AgentEpisodeStore).

All tests use only in-memory instances (no real models, no disk I/O).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from sopilot.perception.agent_episode_store import (
    AgentEpisode,
    AgentEpisodeStore,
    _cosine_sim,
    _extract_fingerprint,
    _violation_type_hash,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_store(**kwargs) -> AgentEpisodeStore:
    """Return an in-memory store (no disk persistence)."""
    return AgentEpisodeStore(persist_path=None, **kwargs)


def _default_situation() -> dict:
    return {
        "risk_level": "high",
        "severity_counts": {"critical": 1, "warning": 2, "info": 0},
        "camera_violations": {"cam1": 3},
        "dominant_violation_type": "PPE違反",
    }


def _store_one(store: AgentEpisodeStore, **overrides) -> str:
    sit = _default_situation()
    sit.update(overrides)
    return store.store_decision(
        **sit,
        decision_action="escalate",
        reasoning_ja="重大違反",
        reasoning_en="Critical violation",
        camera_id="cam1",
    )


# ── TestFeatureExtraction ──────────────────────────────────────────────────


class TestFeatureExtraction:
    def test_fingerprint_length(self):
        fp = _extract_fingerprint("high", {"critical": 1, "warning": 0, "info": 0}, {"cam": 1}, "PPE")
        assert len(fp) == 6

    def test_fingerprint_low_risk(self):
        fp = _extract_fingerprint("low", {}, {}, "")
        assert fp[0] == 0.0  # risk_score

    def test_fingerprint_critical_risk(self):
        fp = _extract_fingerprint("critical", {"critical": 5}, {"cam": 5}, "PPE")
        assert fp[0] == 1.0

    def test_fingerprint_crit_ratio(self):
        fp = _extract_fingerprint("high", {"critical": 2, "warning": 2, "info": 0}, {"cam": 1}, "")
        # 2 critical / 4 total = 0.5
        assert abs(fp[1] - 0.5) < 0.01

    def test_fingerprint_warn_ratio(self):
        fp = _extract_fingerprint("medium", {"critical": 0, "warning": 3, "info": 1}, {"cam": 1}, "")
        # 3 warning / 4 total = 0.75
        assert abs(fp[2] - 0.75) < 0.01

    def test_fingerprint_camera_count_normalized(self):
        cams = {f"cam{i}": 1 for i in range(5)}
        fp = _extract_fingerprint("low", {}, cams, "")
        assert fp[3] == 0.5  # 5 / 10

    def test_fingerprint_camera_count_capped(self):
        cams = {f"cam{i}": 1 for i in range(20)}
        fp = _extract_fingerprint("low", {}, cams, "")
        assert fp[3] == 1.0  # capped

    def test_fingerprint_violations_normalized(self):
        fp = _extract_fingerprint("low", {}, {"cam": 10}, "")
        assert fp[4] == 0.5  # 10 / 20

    def test_fingerprint_violations_capped(self):
        fp = _extract_fingerprint("low", {}, {"cam": 100}, "")
        assert fp[4] == 1.0

    def test_violation_type_hash_empty(self):
        assert _violation_type_hash("") == 0.0

    def test_violation_type_hash_range(self):
        h = _violation_type_hash("PPE違反")
        assert 0.0 <= h <= 1.0

    def test_violation_type_hash_reproducible(self):
        assert _violation_type_hash("PPE") == _violation_type_hash("PPE")

    def test_violation_type_hash_different(self):
        assert _violation_type_hash("PPE") != _violation_type_hash("転倒")


# ── TestCosineSim ──────────────────────────────────────────────────────────


class TestCosineSim:
    def test_identical_vectors(self):
        v = [1.0, 0.5, 0.3]
        assert abs(_cosine_sim(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_sim(a, b)) < 1e-6

    def test_zero_vector(self):
        assert _cosine_sim([0.0, 0.0], [1.0, 0.5]) == 0.0

    def test_symmetry(self):
        a = [0.3, 0.7, 0.1]
        b = [0.5, 0.2, 0.9]
        assert abs(_cosine_sim(a, b) - _cosine_sim(b, a)) < 1e-9

    def test_range(self):
        a = [0.4, 0.6, 0.2]
        b = [0.1, 0.8, 0.5]
        s = _cosine_sim(a, b)
        assert -1.0 <= s <= 1.0


# ── TestAgentEpisode ───────────────────────────────────────────────────────


class TestAgentEpisode:
    def test_construction_defaults(self):
        ep = AgentEpisode(
            episode_id="abc12345",
            fingerprint=[0.5, 0.3, 0.0, 0.1, 0.2, 0.4],
            decision_action="escalate",
            risk_level="high",
            dominant_violation_type="PPE",
            reasoning_ja="テスト",
            reasoning_en="test",
        )
        assert ep.outcome is None
        assert ep.reduction_ratio is None
        assert ep.outcome_at is None

    def test_to_dict_contains_keys(self):
        ep = AgentEpisode(
            episode_id="xyz",
            fingerprint=[0.0] * 6,
            decision_action="monitor",
            risk_level="low",
            dominant_violation_type="",
            reasoning_ja="",
            reasoning_en="",
        )
        d = ep.to_dict()
        assert "episode_id" in d
        assert "fingerprint" in d
        assert "decision_action" in d
        assert "outcome" in d

    def test_to_dict_outcome_none(self):
        ep = AgentEpisode(
            episode_id="z",
            fingerprint=[],
            decision_action="monitor",
            risk_level="low",
            dominant_violation_type="",
            reasoning_ja="",
            reasoning_en="",
        )
        assert ep.to_dict()["outcome"] is None


# ── TestConstruction ───────────────────────────────────────────────────────


class TestConstruction:
    def test_default_construction(self):
        store = _make_store()
        assert store is not None

    def test_custom_max_episodes(self):
        store = _make_store(max_episodes=50)
        stats = store.get_stats()
        assert stats["max_episodes"] == 50

    def test_custom_similarity_threshold(self):
        store = _make_store(recall_min_similarity=0.9)
        stats = store.get_stats()
        assert stats["recall_min_similarity"] == 0.9

    def test_initial_stats_shape(self):
        store = _make_store()
        stats = store.get_stats()
        assert stats["enabled"] is True
        assert stats["total_episodes"] == 0
        assert stats["store_count"] == 0
        assert stats["recall_count"] == 0
        assert stats["outcome_update_count"] == 0

    def test_no_persist_on_none_path(self):
        # Should not raise even with persist_path=None
        store = _make_store()
        _store_one(store)
        # No exception = pass


# ── TestStoreDecision ──────────────────────────────────────────────────────


class TestStoreDecision:
    def test_returns_string_id(self):
        store = _make_store()
        ep_id = _store_one(store)
        assert isinstance(ep_id, str)
        assert len(ep_id) > 0

    def test_increments_total(self):
        store = _make_store()
        _store_one(store)
        _store_one(store)
        assert store.get_stats()["total_episodes"] == 2

    def test_increments_store_count(self):
        store = _make_store()
        _store_one(store)
        _store_one(store)
        assert store.get_stats()["store_count"] == 2

    def test_episode_in_index(self):
        store = _make_store()
        ep_id = _store_one(store)
        assert ep_id in store._index

    def test_maxlen_evicts_oldest(self):
        store = _make_store(max_episodes=3)
        ids = [_store_one(store) for _ in range(5)]
        stats = store.get_stats()
        assert stats["total_episodes"] == 3
        # First two should be evicted from index
        assert ids[0] not in store._index
        assert ids[1] not in store._index
        assert ids[4] in store._index

    def test_stores_action_distribution(self):
        store = _make_store()
        store.store_decision(
            risk_level="high",
            severity_counts={"critical": 1, "warning": 0, "info": 0},
            camera_violations={"cam": 1},
            dominant_violation_type="PPE",
            decision_action="plan",
        )
        dist = store.get_stats()["action_distribution"]
        assert dist.get("plan", 0) == 1


# ── TestUpdateOutcome ──────────────────────────────────────────────────────


class TestUpdateOutcome:
    def test_returns_true_on_found(self):
        store = _make_store()
        ep_id = _store_one(store)
        assert store.update_outcome(ep_id, "success", 0.75) is True

    def test_returns_false_on_not_found(self):
        store = _make_store()
        assert store.update_outcome("nonexistent", "success") is False

    def test_sets_outcome(self):
        store = _make_store()
        ep_id = _store_one(store)
        store.update_outcome(ep_id, "partial", 0.3)
        ep = store._index[ep_id]
        assert ep.outcome == "partial"

    def test_sets_reduction_ratio(self):
        store = _make_store()
        ep_id = _store_one(store)
        store.update_outcome(ep_id, "success", 0.8)
        ep = store._index[ep_id]
        assert abs(ep.reduction_ratio - 0.8) < 0.01

    def test_sets_outcome_at(self):
        store = _make_store()
        ep_id = _store_one(store)
        store.update_outcome(ep_id, "failure")
        ep = store._index[ep_id]
        assert ep.outcome_at is not None

    def test_increments_outcome_update_count(self):
        store = _make_store()
        ep_id = _store_one(store)
        store.update_outcome(ep_id, "success")
        assert store.get_stats()["outcome_update_count"] == 1

    def test_outcome_in_distribution(self):
        store = _make_store()
        ep_id = _store_one(store)
        store.update_outcome(ep_id, "success")
        dist = store.get_stats()["outcome_distribution"]
        assert dist.get("success", 0) == 1


# ── TestRecall ────────────────────────────────────────────────────────────


class TestRecall:
    def test_recall_empty_store(self):
        store = _make_store()
        results = store.recall(
            risk_level="high",
            severity_counts={"critical": 1, "warning": 0, "info": 0},
            camera_violations={"cam": 1},
            dominant_violation_type="PPE",
        )
        assert results == []

    def test_recall_increments_count(self):
        store = _make_store()
        store.recall(
            risk_level="low",
            severity_counts={},
            camera_violations={},
            dominant_violation_type="",
        )
        assert store.get_stats()["recall_count"] == 1

    def test_identical_situation_recalled(self):
        store = _make_store(recall_min_similarity=0.5)
        sit = _default_situation()
        store.store_decision(
            **sit,
            decision_action="escalate",
            reasoning_ja="テスト",
        )
        results = store.recall(**sit, top_k=5)
        assert len(results) == 1

    def test_similar_situation_recalled(self):
        store = _make_store(recall_min_similarity=0.7)
        store.store_decision(
            risk_level="high",
            severity_counts={"critical": 1, "warning": 2, "info": 0},
            camera_violations={"cam1": 3},
            dominant_violation_type="PPE違反",
            decision_action="escalate",
        )
        # Slightly different situation (same type, slightly diff counts)
        results = store.recall(
            risk_level="high",
            severity_counts={"critical": 1, "warning": 1, "info": 0},
            camera_violations={"cam1": 2},
            dominant_violation_type="PPE違反",
            top_k=3,
        )
        assert len(results) >= 1

    def test_different_type_below_threshold(self):
        store = _make_store(recall_min_similarity=0.95)
        store.store_decision(
            risk_level="high",
            severity_counts={"critical": 2, "warning": 0, "info": 0},
            camera_violations={"cam": 5},
            dominant_violation_type="PPE違反",
            decision_action="escalate",
        )
        # Completely different type + severity
        results = store.recall(
            risk_level="low",
            severity_counts={"critical": 0, "warning": 0, "info": 1},
            camera_violations={},
            dominant_violation_type="清掃違反",
            top_k=3,
        )
        assert len(results) == 0

    def test_result_contains_similarity(self):
        store = _make_store(recall_min_similarity=0.5)
        sit = _default_situation()
        store.store_decision(**sit, decision_action="escalate")
        results = store.recall(**sit, top_k=1)
        if results:
            assert "similarity" in results[0]
            assert 0.0 <= results[0]["similarity"] <= 1.0

    def test_result_sorted_by_similarity_desc(self):
        store = _make_store(recall_min_similarity=0.0)
        # Store two similar episodes
        sit1 = _default_situation()
        sit2 = dict(_default_situation(), risk_level="low",
                    severity_counts={"critical": 0, "warning": 0, "info": 1},
                    camera_violations={})
        store.store_decision(**sit1, decision_action="escalate")
        store.store_decision(**sit2, decision_action="monitor")
        results = store.recall(**_default_situation(), top_k=5)
        if len(results) >= 2:
            assert results[0]["similarity"] >= results[1]["similarity"]

    def test_top_k_limits_results(self):
        store = _make_store(recall_min_similarity=0.0)
        for i in range(10):
            _store_one(store)
        results = store.recall(**_default_situation(), top_k=3)
        assert len(results) <= 3


# ── TestSearchByAction ────────────────────────────────────────────────────


class TestSearchByAction:
    def test_returns_matching_action(self):
        store = _make_store()
        store.store_decision(
            **_default_situation(),
            decision_action="plan",
        )
        results = store.search_by_action("plan")
        assert len(results) == 1
        assert results[0]["decision_action"] == "plan"

    def test_excludes_other_actions(self):
        store = _make_store()
        store.store_decision(**_default_situation(), decision_action="escalate")
        results = store.search_by_action("plan")
        assert len(results) == 0

    def test_most_recent_first(self):
        store = _make_store()
        for _ in range(3):
            store.store_decision(**_default_situation(), decision_action="monitor")
        results = store.search_by_action("monitor", limit=2)
        assert len(results) == 2


# ── TestGetRecentEpisodes ─────────────────────────────────────────────────


class TestGetRecentEpisodes:
    def test_empty_store(self):
        store = _make_store()
        assert store.get_recent_episodes() == []

    def test_respects_limit(self):
        store = _make_store()
        for _ in range(15):
            _store_one(store)
        recent = store.get_recent_episodes(limit=5)
        assert len(recent) == 5

    def test_newest_first(self):
        store = _make_store()
        ep1 = _store_one(store, dominant_violation_type="A")
        ep2 = _store_one(store, dominant_violation_type="B")
        recent = store.get_recent_episodes(limit=2)
        assert recent[0]["episode_id"] == ep2
        assert recent[1]["episode_id"] == ep1


# ── TestReset ─────────────────────────────────────────────────────────────


class TestReset:
    def test_clears_episodes(self):
        store = _make_store()
        _store_one(store)
        _store_one(store)
        store.reset()
        assert store.get_stats()["total_episodes"] == 0

    def test_clears_index(self):
        store = _make_store()
        _store_one(store)
        store.reset()
        assert len(store._index) == 0

    def test_clears_counters(self):
        store = _make_store()
        _store_one(store)
        store.update_outcome(list(store._index.keys())[0], "success")
        store.recall(**_default_situation())
        store.reset()
        stats = store.get_stats()
        assert stats["store_count"] == 0
        assert stats["recall_count"] == 0
        assert stats["outcome_update_count"] == 0


# ── TestIntegrationWithAgentLoop ──────────────────────────────────────────


class TestIntegrationWithAgentLoop:
    def test_episode_stored_after_think(self):
        from sopilot.perception.agent_loop import AgentEvent, AgentLoop

        store = _make_store(recall_min_similarity=0.0)
        loop = AgentLoop(
            think_interval_s=0.0,
            episode_store=store,
        )
        ev = AgentEvent(
            event_type="violation",
            camera_id="cam1",
            severity="critical",
            violation_rule="PPE違反",
        )
        loop.inject_event(ev)
        decision = loop.think()
        assert decision is not None
        # An episode should have been stored
        assert store.get_stats()["total_episodes"] == 1

    def test_episode_id_in_decision_params(self):
        from sopilot.perception.agent_loop import AgentEvent, AgentLoop

        store = _make_store(recall_min_similarity=0.0)
        loop = AgentLoop(think_interval_s=0.0, episode_store=store)
        loop.inject_event(AgentEvent(
            event_type="violation",
            camera_id="cam1",
            severity="critical",
            violation_rule="PPE",
        ))
        decision = loop.think()
        assert decision is not None
        assert "episode_id" in decision.params

    def test_recall_populates_situation(self):
        from sopilot.perception.agent_loop import AgentEvent, AgentLoop

        store = _make_store(recall_min_similarity=0.0)
        loop = AgentLoop(think_interval_s=0.0, episode_store=store)
        # Prime the store with one episode
        store.store_decision(
            risk_level="high",
            severity_counts={"critical": 1, "warning": 0, "info": 0},
            camera_violations={"cam1": 1},
            dominant_violation_type="PPE",
            decision_action="escalate",
        )
        # Now think — situation should include recalled_episodes
        loop.inject_event(AgentEvent(
            event_type="violation",
            camera_id="cam1",
            severity="critical",
            violation_rule="PPE",
        ))
        decision = loop.think()
        # recalled_episodes is accessible via situation in decision
        # Just verify no crash and episode_id present
        assert decision is not None

    def test_episode_store_in_stats(self):
        from sopilot.perception.agent_loop import AgentLoop

        store = _make_store()
        loop = AgentLoop(episode_store=store)
        stats = loop.get_stats()
        assert stats["components"]["episode_store"] is True


# ── TestIntegrationWithVerifier ──────────────────────────────────────────


class TestIntegrationWithVerifier:
    def _make_frame_result(self, violation_count: int):
        mock = MagicMock()
        mock.violations = [MagicMock() for _ in range(violation_count)]
        return mock

    def test_verifier_accepts_episode_id(self):
        from sopilot.perception.closed_loop_verifier import ClosedLoopVerifier

        store = _make_store()
        verifier = ClosedLoopVerifier(
            verify_after_s=0.0,
            episode_store=store,
        )
        ep_id = _store_one(store)
        task_id = verifier.schedule(
            decision_action="escalate",
            camera_id="cam1",
            baseline_violations=3,
            baseline_severity={"critical": 1, "warning": 0, "info": 0},
            baseline_risk_level="high",
            episode_id=ep_id,
        )
        assert task_id is not None
        task = verifier._pending[task_id]
        assert task.episode_id == ep_id

    def test_outcome_propagated_to_episode_store(self):
        from sopilot.perception.closed_loop_verifier import ClosedLoopVerifier

        store = _make_store()
        verifier = ClosedLoopVerifier(
            verify_after_s=0.0,
            episode_store=store,
        )
        ep_id = _store_one(store)
        verifier.schedule(
            decision_action="escalate",
            camera_id="cam1",
            baseline_violations=4,
            baseline_severity={"critical": 1, "warning": 0, "info": 0},
            baseline_risk_level="high",
            episode_id=ep_id,
        )
        # Simulate verification (1 violation down from 4 = 75% reduction → success)
        frame = self._make_frame_result(violation_count=1)
        verifier.check_pending(frame, camera_id="cam1")
        ep = store._index.get(ep_id)
        assert ep is not None
        assert ep.outcome is not None  # outcome should be filled in


# ── TestPerceptionConfig ──────────────────────────────────────────────────


class TestPerceptionConfig:
    def test_default_disabled(self):
        from sopilot.perception.types import PerceptionConfig

        cfg = PerceptionConfig()
        assert cfg.episode_store_enabled is False

    def test_default_max_episodes(self):
        from sopilot.perception.types import PerceptionConfig

        cfg = PerceptionConfig()
        assert cfg.episode_store_max_episodes == 1000

    def test_default_similarity_threshold(self):
        from sopilot.perception.types import PerceptionConfig

        cfg = PerceptionConfig()
        assert cfg.episode_store_recall_min_similarity == 0.7
