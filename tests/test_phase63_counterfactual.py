"""Tests for Phase 63 — CounterfactualReasoner.

Covers:
    * Construction and validation
    * Evaluation with empty / sparse / rich episode store
    * Per-action grouping and ranking
    * Confidence heuristic
    * Regret and is_chosen_best flag
    * Narrative content (JP and EN, both branches)
    * Stats accumulation
    * Reset
    * Wiring with a real AgentEpisodeStore
    * Integration: post-decision evaluation
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sopilot.perception.counterfactual import (
    CounterfactualEstimate,
    CounterfactualReasoner,
    CounterfactualResult,
)


# ── Fakes ─────────────────────────────────────────────────────────────────────


class _FakeEpisodeStore:
    """Minimal duck-typed AgentEpisodeStore for unit testing."""

    def __init__(self, episodes: list[dict] | None = None) -> None:
        self._episodes = episodes or []
        self._recall_min_similarity = 0.7
        self.recall_calls: list[dict] = []

    def recall(
        self,
        *,
        risk_level: str,
        severity_counts: dict,
        camera_violations: dict,
        dominant_violation_type: str,
        top_k: int = 50,
    ) -> list[dict]:
        self.recall_calls.append({
            "risk_level": risk_level,
            "severity_counts": severity_counts,
            "camera_violations": camera_violations,
            "dominant_violation_type": dominant_violation_type,
            "top_k": top_k,
            "min_similarity": self._recall_min_similarity,
        })
        return list(self._episodes[:top_k])


def _ep(action: str, reduction: float | None, **kwargs):
    base = {
        "decision_action": action,
        "reduction_ratio": reduction,
        "outcome": "success" if reduction and reduction > 0.5 else "partial",
        "risk_level": "high",
    }
    base.update(kwargs)
    return base


# ── Construction ──────────────────────────────────────────────────────────────


class TestConstruction:
    def test_requires_episode_store(self) -> None:
        with pytest.raises(ValueError):
            CounterfactualReasoner(None)  # type: ignore[arg-type]

    def test_default_thresholds(self) -> None:
        r = CounterfactualReasoner(_FakeEpisodeStore())
        assert r._min_episodes == 2
        assert r._recall_top_k == 50
        assert 0.0 <= r._recall_min_similarity <= 1.0

    def test_clamps_min_episodes(self) -> None:
        r = CounterfactualReasoner(_FakeEpisodeStore(), min_episodes_per_action=0)
        assert r._min_episodes == 1

    def test_clamps_recall_top_k(self) -> None:
        r = CounterfactualReasoner(_FakeEpisodeStore(), recall_top_k=0)
        assert r._recall_top_k == 1

    def test_clamps_min_similarity(self) -> None:
        r = CounterfactualReasoner(
            _FakeEpisodeStore(), recall_min_similarity=1.7
        )
        assert r._recall_min_similarity == 1.0

    def test_initial_stats(self) -> None:
        r = CounterfactualReasoner(_FakeEpisodeStore())
        stats = r.get_stats()
        assert stats["enabled"] is True
        assert stats["evaluation_count"] == 0
        assert stats["regret_observations"] == 0
        assert stats["average_regret"] is None


# ── Empty / sparse store ──────────────────────────────────────────────────────


class TestEmpty:
    def test_empty_store_returns_no_data_result(self) -> None:
        r = CounterfactualReasoner(_FakeEpisodeStore([]))
        result = r.evaluate(
            chosen_action="escalate",
            risk_level="high",
            severity_counts={"critical": 1, "warning": 0, "info": 0},
            camera_violations={"cam-A": 1},
            dominant_violation_type="ヘルメット未着用",
        )
        assert isinstance(result, CounterfactualResult)
        assert result.has_data is False
        assert result.sample_size_total == 0
        assert result.alternatives == []
        assert result.best_alternative is None
        assert result.regret is None
        assert result.is_chosen_best is False

    def test_empty_store_narrative_japanese(self) -> None:
        r = CounterfactualReasoner(_FakeEpisodeStore([]))
        result = r.evaluate(
            chosen_action="escalate",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert "サンプル数=0" in result.reasoning_ja
        assert "保留" in result.reasoning_ja

    def test_empty_store_narrative_english(self) -> None:
        r = CounterfactualReasoner(_FakeEpisodeStore([]))
        result = r.evaluate(
            chosen_action="escalate",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert "Insufficient" in result.reasoning_en
        assert "n=0" in result.reasoning_en

    def test_only_unverified_episodes_yields_no_estimate(self) -> None:
        store = _FakeEpisodeStore([
            _ep("escalate", reduction=None),
            _ep("escalate", reduction=None),
            _ep("escalate", reduction=None),
        ])
        r = CounterfactualReasoner(store, min_episodes_per_action=2)
        result = r.evaluate(
            chosen_action="escalate",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        # has_data is False because no estimate was computable
        assert result.has_data is False
        # But the action is still listed (so caller sees it was tried)
        assert any(a.action == "escalate" for a in result.alternatives)
        chosen = next(a for a in result.alternatives if a.action == "escalate")
        assert chosen.expected_reduction is None
        assert chosen.verified_size == 0
        assert chosen.sample_size == 3


# ── Single-action estimates ───────────────────────────────────────────────────


class TestSingleAction:
    def test_three_escalates_yield_estimate(self) -> None:
        store = _FakeEpisodeStore([
            _ep("escalate", 0.6),
            _ep("escalate", 0.7),
            _ep("escalate", 0.5),
        ])
        r = CounterfactualReasoner(store, min_episodes_per_action=2)
        result = r.evaluate(
            chosen_action="escalate",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert result.has_data is True
        assert result.chosen_estimate is not None
        # Mean of 0.6, 0.7, 0.5 = 0.6
        assert abs(result.chosen_estimate.expected_reduction - 0.6) < 1e-9
        assert result.chosen_estimate.verified_size == 3
        assert result.chosen_estimate.sample_size == 3

    def test_below_min_episodes_no_estimate(self) -> None:
        store = _FakeEpisodeStore([_ep("escalate", 0.6)])
        r = CounterfactualReasoner(store, min_episodes_per_action=2)
        result = r.evaluate(
            chosen_action="escalate",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert result.has_data is False
        chosen = next(a for a in result.alternatives if a.action == "escalate")
        assert chosen.expected_reduction is None
        assert chosen.verified_size == 1

    def test_chosen_is_only_estimate_so_chosen_best(self) -> None:
        store = _FakeEpisodeStore([
            _ep("escalate", 0.6),
            _ep("escalate", 0.7),
        ])
        r = CounterfactualReasoner(store, min_episodes_per_action=2)
        result = r.evaluate(
            chosen_action="escalate",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert result.is_chosen_best is True
        assert result.regret == 0.0

    def test_stddev_zero_for_identical_outcomes(self) -> None:
        store = _FakeEpisodeStore([
            _ep("escalate", 0.5),
            _ep("escalate", 0.5),
            _ep("escalate", 0.5),
        ])
        r = CounterfactualReasoner(store)
        result = r.evaluate(
            chosen_action="escalate",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert result.chosen_estimate is not None
        assert result.chosen_estimate.stddev_reduction == 0.0


# ── Multi-action ranking ──────────────────────────────────────────────────────


class TestRanking:
    def _make_mixed_store(self) -> _FakeEpisodeStore:
        # plan averages 0.7, escalate averages 0.6, monitor averages 0.1
        return _FakeEpisodeStore([
            _ep("plan", 0.7), _ep("plan", 0.7), _ep("plan", 0.7),
            _ep("escalate", 0.6), _ep("escalate", 0.6),
            _ep("monitor", 0.1), _ep("monitor", 0.1),
        ])

    def test_best_is_highest_expected(self) -> None:
        r = CounterfactualReasoner(self._make_mixed_store())
        result = r.evaluate(
            chosen_action="escalate",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert result.best_alternative is not None
        assert result.best_alternative.action == "plan"

    def test_chosen_not_best_yields_positive_regret(self) -> None:
        r = CounterfactualReasoner(self._make_mixed_store())
        result = r.evaluate(
            chosen_action="escalate",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert result.is_chosen_best is False
        # regret = 0.7 - 0.6 = 0.1
        assert result.regret is not None
        assert abs(result.regret - 0.1) < 1e-9

    def test_chosen_is_best_yields_zero_regret(self) -> None:
        r = CounterfactualReasoner(self._make_mixed_store())
        result = r.evaluate(
            chosen_action="plan",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert result.is_chosen_best is True
        assert result.regret == 0.0

    def test_alternatives_ordered_by_rank(self) -> None:
        r = CounterfactualReasoner(self._make_mixed_store())
        result = r.evaluate(
            chosen_action="escalate",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        # The first three are estimable; the rest (none here) come last
        ranked = [a for a in result.alternatives if a.expected_reduction is not None]
        assert ranked[0].action == "plan"
        assert ranked[1].action == "escalate"
        assert ranked[2].action == "monitor"
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        assert ranked[2].rank == 3

    def test_unestimable_actions_listed_last(self) -> None:
        store = _FakeEpisodeStore([
            _ep("plan", 0.7), _ep("plan", 0.7), _ep("plan", 0.7),
            _ep("query", None),  # unverified — no estimate
        ])
        r = CounterfactualReasoner(store, min_episodes_per_action=2)
        result = r.evaluate(
            chosen_action="plan",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        # plan estimable, query not
        last = result.alternatives[-1]
        assert last.expected_reduction is None
        assert last.action == "query"

    def test_chosen_action_not_in_recalled_episodes(self) -> None:
        store = _FakeEpisodeStore([
            _ep("plan", 0.7), _ep("plan", 0.7),
        ])
        r = CounterfactualReasoner(store)
        result = r.evaluate(
            chosen_action="adapt",  # never seen in store
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert result.chosen_estimate is None
        # has_data is True because plan has an estimate
        assert result.has_data is True
        # No regret available because chosen has no estimate
        assert result.regret is None
        assert result.is_chosen_best is False


# ── Confidence heuristic ──────────────────────────────────────────────────────


class TestConfidence:
    def test_zero_for_no_data(self) -> None:
        assert CounterfactualReasoner._confidence(0, 0.0) == 0.0

    def test_higher_for_more_samples(self) -> None:
        c4 = CounterfactualReasoner._confidence(4, 0.0)
        c100 = CounterfactualReasoner._confidence(100, 0.0)
        assert c100 > c4

    def test_lower_for_high_variance(self) -> None:
        c_low = CounterfactualReasoner._confidence(10, 0.0)
        c_high = CounterfactualReasoner._confidence(10, 0.8)
        assert c_low > c_high

    def test_bounded_zero_one(self) -> None:
        c = CounterfactualReasoner._confidence(1000, 0.0)
        assert 0.0 <= c <= 1.0

    def test_stddev_above_one_clamps_to_zero_spread_factor(self) -> None:
        # stddev=1.0 → spread_factor=0 → confidence=0
        c = CounterfactualReasoner._confidence(100, 1.5)
        assert c == 0.0


# ── Stats accumulation ───────────────────────────────────────────────────────


class TestStats:
    def test_evaluation_count_increments(self) -> None:
        r = CounterfactualReasoner(_FakeEpisodeStore([]))
        for _ in range(5):
            r.evaluate(
                chosen_action="monitor",
                risk_level="low",
                severity_counts={},
                camera_violations={},
                dominant_violation_type="",
            )
        assert r.get_stats()["evaluation_count"] == 5

    def test_average_regret_computed_after_observations(self) -> None:
        store = _FakeEpisodeStore([
            _ep("plan", 0.7), _ep("plan", 0.7),
            _ep("escalate", 0.6), _ep("escalate", 0.6),
        ])
        r = CounterfactualReasoner(store)
        # Twice picking escalate when plan is better → regret ≈ 0.1 each
        for _ in range(2):
            r.evaluate(
                chosen_action="escalate",
                risk_level="high",
                severity_counts={"critical": 1},
                camera_violations={"cam-A": 1},
                dominant_violation_type="x",
            )
        stats = r.get_stats()
        assert stats["regret_observations"] == 2
        assert stats["average_regret"] is not None
        assert abs(stats["average_regret"] - 0.1) < 1e-3

    def test_chosen_best_rate(self) -> None:
        store = _FakeEpisodeStore([
            _ep("plan", 0.7), _ep("plan", 0.7),
            _ep("escalate", 0.6), _ep("escalate", 0.6),
        ])
        r = CounterfactualReasoner(store)
        # 2 plans (best), 1 escalate (not best)
        for _ in range(2):
            r.evaluate(
                chosen_action="plan",
                risk_level="high",
                severity_counts={"critical": 1},
                camera_violations={"cam-A": 1},
                dominant_violation_type="x",
            )
        r.evaluate(
            chosen_action="escalate",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        stats = r.get_stats()
        assert stats["chosen_best_rate"] is not None
        # 2/3 ≈ 0.667
        assert abs(stats["chosen_best_rate"] - 2.0 / 3.0) < 1e-3

    def test_evaluations_with_data_only_counts_when_estimable(self) -> None:
        store = _FakeEpisodeStore([_ep("plan", None), _ep("plan", None)])
        r = CounterfactualReasoner(store, min_episodes_per_action=2)
        r.evaluate(
            chosen_action="plan",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert r.get_stats()["evaluations_with_data"] == 0


# ── Reset ─────────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_counters(self) -> None:
        store = _FakeEpisodeStore([
            _ep("plan", 0.7), _ep("plan", 0.7),
        ])
        r = CounterfactualReasoner(store)
        r.evaluate(
            chosen_action="plan",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert r.get_stats()["evaluation_count"] == 1
        r.reset()
        stats = r.get_stats()
        assert stats["evaluation_count"] == 0
        assert stats["regret_observations"] == 0
        assert stats["average_regret"] is None


# ── Narrative ─────────────────────────────────────────────────────────────────


class TestNarrative:
    def test_optimal_narrative_japanese(self) -> None:
        store = _FakeEpisodeStore([
            _ep("plan", 0.7), _ep("plan", 0.7), _ep("plan", 0.7),
            _ep("escalate", 0.5), _ep("escalate", 0.5),
        ])
        r = CounterfactualReasoner(store)
        result = r.evaluate(
            chosen_action="plan",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert "妥当" in result.reasoning_ja or "最良" in result.reasoning_ja

    def test_optimal_narrative_english(self) -> None:
        store = _FakeEpisodeStore([
            _ep("plan", 0.7), _ep("plan", 0.7), _ep("plan", 0.7),
        ])
        r = CounterfactualReasoner(store)
        result = r.evaluate(
            chosen_action="plan",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert "Optimal" in result.reasoning_en

    def test_suboptimal_narrative_includes_regret(self) -> None:
        store = _FakeEpisodeStore([
            _ep("plan", 0.7), _ep("plan", 0.7),
            _ep("escalate", 0.5), _ep("escalate", 0.5),
        ])
        r = CounterfactualReasoner(store)
        result = r.evaluate(
            chosen_action="escalate",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert "PLAN" in result.reasoning_ja
        assert "後悔" in result.reasoning_ja
        assert "regret" in result.reasoning_en.lower()


# ── to_dict ───────────────────────────────────────────────────────────────────


class TestToDict:
    def test_estimate_to_dict_rounds(self) -> None:
        e = CounterfactualEstimate(
            action="plan",
            sample_size=3,
            verified_size=3,
            expected_reduction=0.123456789,
            stddev_reduction=0.0001234,
            confidence=0.987654321,
            rank=1,
        )
        d = e.to_dict()
        assert d["expected_reduction"] == 0.1235
        assert d["confidence"] == 0.9877

    def test_estimate_to_dict_handles_none(self) -> None:
        e = CounterfactualEstimate(
            action="x", sample_size=1, verified_size=0,
            expected_reduction=None, stddev_reduction=None,
            confidence=0.0, rank=0,
        )
        d = e.to_dict()
        assert d["expected_reduction"] is None
        assert d["stddev_reduction"] is None

    def test_result_to_dict_has_expected_keys(self) -> None:
        r = CounterfactualReasoner(_FakeEpisodeStore([]))
        result = r.evaluate(
            chosen_action="monitor",
            risk_level="low",
            severity_counts={},
            camera_violations={},
            dominant_violation_type="",
        )
        d = result.to_dict()
        for key in (
            "chosen_action", "chosen_estimate", "alternatives",
            "best_alternative", "is_chosen_best", "regret",
            "sample_size_total", "has_data", "reasoning_ja",
            "reasoning_en", "evaluated_at",
        ):
            assert key in d


# ── Recall threshold management ───────────────────────────────────────────────


class TestRecallThreshold:
    def test_relaxes_and_restores_threshold(self) -> None:
        store = _FakeEpisodeStore([_ep("plan", 0.7), _ep("plan", 0.7)])
        store._recall_min_similarity = 0.9  # store starts strict
        r = CounterfactualReasoner(store, recall_min_similarity=0.5)
        r.evaluate(
            chosen_action="plan",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        # Threshold restored after the call
        assert store._recall_min_similarity == 0.9
        # Recall was made with the relaxed threshold
        assert store.recall_calls[-1]["min_similarity"] == 0.5

    def test_top_k_passed_through(self) -> None:
        store = _FakeEpisodeStore([])
        r = CounterfactualReasoner(store, recall_top_k=17)
        r.evaluate(
            chosen_action="monitor",
            risk_level="low",
            severity_counts={},
            camera_violations={},
            dominant_violation_type="",
        )
        assert store.recall_calls[-1]["top_k"] == 17


# ── Failure tolerance ─────────────────────────────────────────────────────────


class TestFailureTolerance:
    def test_recall_exception_returns_no_data(self) -> None:
        store = MagicMock()
        store.recall.side_effect = RuntimeError("boom")
        store._recall_min_similarity = 0.7
        r = CounterfactualReasoner(store)
        result = r.evaluate(
            chosen_action="escalate",
            risk_level="high",
            severity_counts={"critical": 1},
            camera_violations={"cam-A": 1},
            dominant_violation_type="x",
        )
        assert result.has_data is False
        assert result.sample_size_total == 0


# ── Real AgentEpisodeStore integration ────────────────────────────────────────


class TestRealStoreIntegration:
    def test_with_real_store(self) -> None:
        from sopilot.perception.agent_episode_store import AgentEpisodeStore

        with tempfile.TemporaryDirectory() as td:
            store = AgentEpisodeStore(persist_path=str(Path(td) / "ep.json"))
            # Same fingerprint situations × 3 escalate verifications
            ids = []
            for _ in range(3):
                ep_id = store.store_decision(
                    risk_level="high",
                    severity_counts={"critical": 1, "warning": 0, "info": 0},
                    camera_violations={"cam-A": 1},
                    dominant_violation_type="ヘルメット未着用",
                    decision_action="escalate",
                    reasoning_ja="r",
                    reasoning_en="r",
                    camera_id="cam-A",
                )
                ids.append(ep_id)
                store.update_outcome(ep_id, "success", reduction_ratio=0.6)

            r = CounterfactualReasoner(store, min_episodes_per_action=2)
            result = r.evaluate(
                chosen_action="escalate",
                risk_level="high",
                severity_counts={"critical": 1, "warning": 0, "info": 0},
                camera_violations={"cam-A": 1},
                dominant_violation_type="ヘルメット未着用",
            )
            assert result.has_data is True
            assert result.chosen_estimate is not None
            # At least one verified estimate should be present
            assert result.chosen_estimate.verified_size >= 2
            assert result.chosen_estimate.expected_reduction is not None
            assert abs(result.chosen_estimate.expected_reduction - 0.6) < 1e-6


# ── Engine integration smoke test ─────────────────────────────────────────────


class TestEngineIntegrationSmoke:
    """Verify counterfactual_reasoner gets wired into PerceptionEngine when both
    agent_loop and episode_store are enabled."""

    def test_engine_initializes_counterfactual_when_episode_store_present(self) -> None:
        from sopilot.perception.engine import build_perception_engine
        from sopilot.perception.types import PerceptionConfig

        cfg = PerceptionConfig()
        cfg.agent_loop_enabled = True
        cfg.episode_store_enabled = True
        cfg.counterfactual_enabled = True
        cfg.orchestrator_enabled = False
        engine = build_perception_engine(cfg)
        assert hasattr(engine, "_counterfactual_reasoner")
        assert engine._counterfactual_reasoner is not None
        # Wired to the same episode store the agent loop uses
        assert (
            engine._counterfactual_reasoner._episode_store
            is engine._agent_episode_store
        )

    def test_engine_skips_counterfactual_when_episode_store_disabled(self) -> None:
        from sopilot.perception.engine import build_perception_engine
        from sopilot.perception.types import PerceptionConfig

        cfg = PerceptionConfig()
        cfg.agent_loop_enabled = False
        cfg.episode_store_enabled = False
        cfg.counterfactual_enabled = True
        cfg.orchestrator_enabled = False
        engine = build_perception_engine(cfg)
        assert getattr(engine, "_counterfactual_reasoner", None) is None
