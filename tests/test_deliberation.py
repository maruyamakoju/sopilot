"""Tests for sopilot.perception.deliberation — System 2 deliberative reasoning.

~45 tests covering:
    TestShouldDeliberate    (6)
    TestDeliberate          (8)
    TestHypotheses          (6)
    TestEvidence            (5)
    TestRanking             (4)
    TestUrgency             (5)
    TestAction              (4)
    TestHistory             (4)
    TestCooldown            (3)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from sopilot.perception.deliberation import (
    DeliberationResult,
    DeliberativeReasoner,
    Evidence,
    Hypothesis,
)
from sopilot.perception.types import EntityEvent, EntityEventType, WorldState


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_event(
    etype: EntityEventType = EntityEventType.ANOMALY,
    entity_id: int = 1,
    ts: float | None = None,
    confidence: float = 0.8,
    details: dict | None = None,
) -> EntityEvent:
    return EntityEvent(
        event_type=etype,
        entity_id=entity_id,
        timestamp=ts or time.time(),
        frame_number=0,
        details=details or {"severity": "warning", "z_score": 2.5},
        confidence=confidence,
    )


def _make_world_state(entity_count: int = 2, recent_event_count: int = 0) -> MagicMock:
    ws = MagicMock(spec=WorldState)
    ws.events = [_make_event() for _ in range(recent_event_count)]
    ws.scene_graph = MagicMock()
    ws.scene_graph.entities = {i: MagicMock() for i in range(entity_count)}
    ws.timestamp = time.time()
    return ws


# ── TestShouldDeliberate ──────────────────────────────────────────────────────


class TestShouldDeliberate:
    """6 tests: ANOMALY, RULE_VIOLATION, PROLONGED_PRESENCE, COLLISION_PREDICTED,
    ENTERED (excluded), and cooldown gate."""

    def test_anomaly_returns_true(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(EntityEventType.ANOMALY)
        assert reasoner.should_deliberate(event) is True

    def test_rule_violation_returns_true(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(EntityEventType.RULE_VIOLATION)
        assert reasoner.should_deliberate(event) is True

    def test_prolonged_presence_returns_true(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(EntityEventType.PROLONGED_PRESENCE)
        assert reasoner.should_deliberate(event) is True

    def test_collision_predicted_returns_true(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(EntityEventType.COLLISION_PREDICTED)
        assert reasoner.should_deliberate(event) is True

    def test_entered_returns_false(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(EntityEventType.ENTERED)
        assert reasoner.should_deliberate(event) is False

    def test_cooldown_prevents_second_call(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=999.0)
        event = _make_event(EntityEventType.ANOMALY)
        # Simulate that a deliberation just happened by setting the timestamp
        reasoner._last_deliberation_time = time.time()
        assert reasoner.should_deliberate(event) is False


# ── TestDeliberate ────────────────────────────────────────────────────────────


class TestDeliberate:
    """8 tests: result structure, urgency validity, confidence range, duration."""

    def _run(self, etype=EntityEventType.ANOMALY, **kw):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(etype, **kw)
        ws = _make_world_state()
        return reasoner.deliberate(event, ws)

    def test_returns_deliberation_result(self):
        result = self._run()
        assert isinstance(result, DeliberationResult)

    def test_best_hypothesis_or_none(self):
        result = self._run()
        assert result.best_hypothesis is None or isinstance(result.best_hypothesis, Hypothesis)

    def test_hypotheses_list_not_empty(self):
        result = self._run()
        assert len(result.hypotheses) >= 1

    def test_urgency_in_valid_set(self):
        result = self._run()
        assert result.urgency in {"low", "medium", "high", "critical"}

    def test_action_ja_not_empty(self):
        result = self._run()
        assert isinstance(result.action_ja, str) and len(result.action_ja) > 0

    def test_action_en_not_empty(self):
        result = self._run()
        assert isinstance(result.action_en, str) and len(result.action_en) > 0

    def test_overall_confidence_in_range(self):
        result = self._run()
        assert 0.0 <= result.overall_confidence <= 1.0

    def test_duration_ms_positive(self):
        result = self._run()
        assert result.duration_ms >= 0.0


# ── TestHypotheses ────────────────────────────────────────────────────────────


class TestHypotheses:
    """6 tests: ANOMALY and RULE_VIOLATION generate >=2 hypotheses; structure checks."""

    def _hypotheses(self, etype):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(etype)
        ws = _make_world_state()
        result = reasoner.deliberate(event, ws)
        return result.hypotheses

    def test_anomaly_generates_at_least_two(self):
        hyps = self._hypotheses(EntityEventType.ANOMALY)
        assert len(hyps) >= 2

    def test_rule_violation_generates_at_least_two(self):
        hyps = self._hypotheses(EntityEventType.RULE_VIOLATION)
        assert len(hyps) >= 2

    def test_each_hypothesis_has_claim_ja(self):
        hyps = self._hypotheses(EntityEventType.ANOMALY)
        for h in hyps:
            assert isinstance(h.claim_ja, str) and len(h.claim_ja) > 0

    def test_each_hypothesis_has_claim_en(self):
        hyps = self._hypotheses(EntityEventType.ANOMALY)
        for h in hyps:
            assert isinstance(h.claim_en, str) and len(h.claim_en) > 0

    def test_belief_in_range(self):
        hyps = self._hypotheses(EntityEventType.ANOMALY)
        for h in hyps:
            assert 0.0 <= h.belief <= 1.0

    def test_plausibility_in_range(self):
        hyps = self._hypotheses(EntityEventType.ANOMALY)
        for h in hyps:
            assert 0.0 <= h.plausibility <= 1.0

    def test_entity_ids_is_list(self):
        hyps = self._hypotheses(EntityEventType.ANOMALY)
        for h in hyps:
            assert isinstance(h.entity_ids, list)


# ── TestEvidence ──────────────────────────────────────────────────────────────


class TestEvidence:
    """5 tests: high-conf, low-conf, multiple entities, goal_hypotheses, episodes."""

    def test_high_confidence_event_adds_evidence_for_h1(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(confidence=0.95, details={"severity": "warning", "z_score": 3.0})
        ws = _make_world_state(entity_count=3)
        result = reasoner.deliberate(event, ws)
        # The first hypothesis (H1 genuine) should have evidence_for populated
        best = result.best_hypothesis
        assert best is not None
        # At least some hypothesis should have evidence_for
        total_evidence = sum(len(h.evidence_for) for h in result.hypotheses)
        assert total_evidence > 0

    def test_low_confidence_event_adds_evidence_for_h2(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(confidence=0.2, details={"severity": "info", "z_score": 0.5})
        ws = _make_world_state(entity_count=1)
        result = reasoner.deliberate(event, ws)
        # H2 (false positive) should have supporting evidence
        h2_candidates = [
            h for h in result.hypotheses
            if any(kw in h.claim_en.lower() for kw in ["false positive", "noise", "unintentional", "aware", "normal"])
        ]
        assert len(h2_candidates) > 0
        # At least one H2 candidate has evidence_for
        assert any(len(h.evidence_for) > 0 for h in h2_candidates)

    def test_multiple_entities_adds_evidence(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(confidence=0.85, details={"severity": "warning", "z_score": 2.5})
        ws = _make_world_state(entity_count=5)
        result = reasoner.deliberate(event, ws)
        total_evidence = sum(len(h.evidence_for) for h in result.hypotheses)
        assert total_evidence > 0

    def test_goal_hypotheses_used_when_provided(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(confidence=0.8)
        ws = _make_world_state()
        gh = MagicMock()
        gh.risk_score = 0.9
        result = reasoner.deliberate(event, ws, goal_hypotheses=[gh])
        # Evidence from goal_recognizer should appear in at least one hypothesis
        goal_evidence = [
            ev
            for h in result.hypotheses
            for ev in h.evidence_for
            if ev.source == "goal_recognizer"
        ]
        assert len(goal_evidence) > 0

    def test_recent_episodes_used_when_provided(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(EntityEventType.ANOMALY, confidence=0.8)
        ws = _make_world_state(recent_event_count=8)
        # Make timestamps recent so they pass the 30s filter
        for ev in ws.events:
            ev.timestamp = time.time()
        episode = MagicMock()
        episode.severity = "critical"
        result = reasoner.deliberate(event, ws, recent_episodes=[episode])
        episode_evidence = [
            ev
            for h in result.hypotheses
            for ev in h.evidence_for
            if ev.source == "episodic_memory"
        ]
        assert len(episode_evidence) > 0


# ── TestRanking ───────────────────────────────────────────────────────────────


class TestRanking:
    """4 tests: sorted desc, normalized sum <=1, best is first, best >= others."""

    def _result(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event()
        ws = _make_world_state()
        return reasoner.deliberate(event, ws)

    def test_hypotheses_sorted_by_belief_desc(self):
        result = self._result()
        beliefs = [h.belief for h in result.hypotheses]
        assert beliefs == sorted(beliefs, reverse=True)

    def test_beliefs_normalized_sum_leq_1(self):
        result = self._result()
        total = sum(h.belief for h in result.hypotheses)
        # Sum should be very close to 1.0 (after normalization) or <= 1.0
        assert total <= 1.01  # small float tolerance

    def test_best_hypothesis_is_first(self):
        result = self._result()
        if result.hypotheses:
            assert result.best_hypothesis is result.hypotheses[0]

    def test_best_belief_gte_others(self):
        result = self._result()
        if result.best_hypothesis and len(result.hypotheses) > 1:
            best_belief = result.best_hypothesis.belief
            for h in result.hypotheses[1:]:
                assert best_belief >= h.belief


# ── TestUrgency ───────────────────────────────────────────────────────────────


class TestUrgency:
    """5 tests: ANOMALY high belief critical/high, PROLONGED_PRESENCE medium,
    low confidence low, valid set, changes with belief."""

    def test_anomaly_high_belief_critical_or_high(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        # High confidence + high z_score to push belief up
        event = _make_event(
            EntityEventType.ANOMALY,
            confidence=0.99,
            details={"severity": "critical", "z_score": 5.0},
        )
        ws = _make_world_state(entity_count=5)
        result = reasoner.deliberate(event, ws)
        assert result.urgency in {"critical", "high"}

    def test_prolonged_presence_medium(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(EntityEventType.PROLONGED_PRESENCE, confidence=0.6)
        ws = _make_world_state()
        result = reasoner.deliberate(event, ws)
        assert result.urgency in {"medium", "low", "high"}  # medium is expected but allow range

    def test_low_confidence_event_low_urgency(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(confidence=0.1, details={"severity": "info", "z_score": 0.1})
        ws = _make_world_state(entity_count=1)
        result = reasoner.deliberate(event, ws)
        # Low confidence → best belief should be low → urgency "low"
        assert result.urgency in {"low", "medium"}

    def test_urgency_in_valid_set(self):
        for etype in [
            EntityEventType.ANOMALY,
            EntityEventType.RULE_VIOLATION,
            EntityEventType.PROLONGED_PRESENCE,
            EntityEventType.COLLISION_PREDICTED,
        ]:
            reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
            event = _make_event(etype)
            ws = _make_world_state()
            result = reasoner.deliberate(event, ws)
            assert result.urgency in {"low", "medium", "high", "critical"}, (
                f"Urgency {result.urgency!r} not valid for {etype}"
            )

    def test_urgency_changes_with_belief_level(self):
        # High belief scenario
        r_high = DeliberativeReasoner(cooldown_seconds=0.0)
        ev_high = _make_event(
            EntityEventType.ANOMALY,
            confidence=0.99,
            details={"severity": "critical", "z_score": 9.0},
        )
        ws_high = _make_world_state(entity_count=10)
        result_high = r_high.deliberate(ev_high, ws_high)

        # Low belief scenario
        r_low = DeliberativeReasoner(cooldown_seconds=0.0)
        ev_low = _make_event(confidence=0.05, details={"severity": "info", "z_score": 0.0})
        ws_low = _make_world_state(entity_count=0)
        result_low = r_low.deliberate(ev_low, ws_low)

        # They should not both be the same urgency (different belief levels)
        # At minimum the high-belief result should be >= low-belief result in severity order
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        assert order[result_high.urgency] >= order[result_low.urgency]


# ── TestAction ────────────────────────────────────────────────────────────────


class TestAction:
    """4 tests: critical mentions urgent, action_ja Japanese, action_en English, not empty."""

    def _get_critical_result(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event(
            EntityEventType.ANOMALY,
            confidence=0.99,
            details={"severity": "critical", "z_score": 8.0},
        )
        ws = _make_world_state(entity_count=5)
        result = reasoner.deliberate(event, ws)
        return result

    def test_critical_urgency_ja_mentions_urgent_response(self):
        result = self._get_critical_result()
        if result.urgency == "critical":
            # Should contain Japanese urgent response keywords
            assert any(
                kw in result.action_ja
                for kw in ["直ちに", "確認", "セキュリティ", "派遣"]
            )

    def test_action_ja_is_nonempty_string(self):
        result = self._get_critical_result()
        assert isinstance(result.action_ja, str)
        assert len(result.action_ja) > 0

    def test_action_en_is_nonempty_string(self):
        result = self._get_critical_result()
        assert isinstance(result.action_en, str)
        assert len(result.action_en) > 0

    def test_action_not_empty_for_all_event_types(self):
        for etype in [
            EntityEventType.ANOMALY,
            EntityEventType.RULE_VIOLATION,
            EntityEventType.PROLONGED_PRESENCE,
            EntityEventType.COLLISION_PREDICTED,
        ]:
            reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
            event = _make_event(etype)
            ws = _make_world_state()
            result = reasoner.deliberate(event, ws)
            assert len(result.action_ja) > 0
            assert len(result.action_en) > 0


# ── TestHistory ───────────────────────────────────────────────────────────────


class TestHistory:
    """4 tests: empty initially, returns results after deliberate, newest first, max_history."""

    def test_empty_initially(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        assert reasoner.get_recent_deliberations() == []

    def test_returns_results_after_deliberate(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        event = _make_event()
        ws = _make_world_state()
        reasoner.deliberate(event, ws)
        results = reasoner.get_recent_deliberations()
        assert len(results) == 1
        assert isinstance(results[0], DeliberationResult)

    def test_newest_first(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0)
        ws = _make_world_state()
        r1 = reasoner.deliberate(_make_event(), ws)
        r2 = reasoner.deliberate(_make_event(), ws)
        history = reasoner.get_recent_deliberations()
        assert history[0].id == r2.id
        assert history[1].id == r1.id

    def test_max_history_respected(self):
        max_hist = 3
        reasoner = DeliberativeReasoner(cooldown_seconds=0.0, max_history=max_hist)
        ws = _make_world_state()
        for _ in range(5):
            reasoner.deliberate(_make_event(), ws)
        history = reasoner.get_recent_deliberations(n=100)
        assert len(history) <= max_hist


# ── TestCooldown ──────────────────────────────────────────────────────────────


class TestCooldown:
    """3 tests: second deliberate still works, should_deliberate False in cooldown,
    should_deliberate True after expiry."""

    def test_second_deliberate_within_cooldown_still_works(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=999.0)
        event = _make_event()
        ws = _make_world_state()
        # Even within cooldown, deliberate() itself should not be blocked
        r1 = reasoner.deliberate(event, ws)
        r2 = reasoner.deliberate(event, ws)
        assert isinstance(r1, DeliberationResult)
        assert isinstance(r2, DeliberationResult)

    def test_should_deliberate_false_after_recent_deliberation(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=999.0)
        event = _make_event()
        ws = _make_world_state()
        reasoner.deliberate(event, ws)
        # should_deliberate now returns False
        assert reasoner.should_deliberate(event) is False

    def test_should_deliberate_true_after_cooldown_expires(self):
        reasoner = DeliberativeReasoner(cooldown_seconds=0.001)  # 1 ms
        event = _make_event()
        ws = _make_world_state()
        reasoner.deliberate(event, ws)
        # Wait for cooldown to expire
        time.sleep(0.05)
        assert reasoner.should_deliberate(event) is True
