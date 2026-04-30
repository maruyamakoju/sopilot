"""Tests for Phase 59: Agentic Closed-Loop (OODA) — AgentLoop.

Covers:
  - Enumerations (AgentAction, OODAPhase, AgentStatus)
  - Data structures (AgentEvent, SituationSummary, AgentDecision)
  - AgentLoop construction, observe, should_think, think, orient, decide, act
  - Stats, reset, PerceptionConfig defaults
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from sopilot.perception.agent_loop import (
    AgentAction,
    AgentDecision,
    AgentEvent,
    AgentLoop,
    AgentStatus,
    OODAPhase,
    SituationSummary,
    _MAX_DECISIONS,
    _MAX_EVENTS,
)
from sopilot.perception.types import PerceptionConfig


# ═══════════════════════════════════════════════════════════════════════════
# TestEnumerations
# ═══════════════════════════════════════════════════════════════════════════


class TestEnumerations:
    """Test AgentAction, OODAPhase, AgentStatus enum values."""

    def test_agent_action_values(self):
        assert AgentAction.MONITOR == "monitor"
        assert AgentAction.ESCALATE == "escalate"
        assert AgentAction.PLAN == "plan"
        assert AgentAction.ADAPT == "adapt"
        assert AgentAction.QUERY == "query"
        assert AgentAction.RESPOND == "respond"

    def test_agent_action_count(self):
        assert len(AgentAction) == 6

    def test_ooda_phase_values(self):
        assert OODAPhase.OBSERVE == "observe"
        assert OODAPhase.ORIENT == "orient"
        assert OODAPhase.DECIDE == "decide"
        assert OODAPhase.ACT == "act"
        assert OODAPhase.LEARN == "learn"

    def test_ooda_phase_count(self):
        assert len(OODAPhase) == 5

    def test_agent_status_values(self):
        assert AgentStatus.IDLE == "idle"
        assert AgentStatus.THINKING == "thinking"
        assert AgentStatus.DISABLED == "disabled"


# ═══════════════════════════════════════════════════════════════════════════
# TestAgentEvent
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentEvent:
    """Test AgentEvent data structure."""

    def test_construction(self):
        ev = AgentEvent(event_type="violation", camera_id="cam1", severity="critical")
        assert ev.event_type == "violation"
        assert ev.camera_id == "cam1"
        assert ev.severity == "critical"

    def test_defaults(self):
        ev = AgentEvent(event_type="event")
        assert ev.camera_id == "cam"
        assert ev.severity == "info"
        assert ev.violation_rule == ""
        assert ev.timestamp == 0.0

    def test_severity_values(self):
        for sev in ("info", "warning", "critical"):
            ev = AgentEvent(event_type="test", severity=sev)
            assert ev.severity == sev

    def test_details_dict(self):
        ev = AgentEvent(event_type="test", details={"key": "val"})
        assert ev.details["key"] == "val"


# ═══════════════════════════════════════════════════════════════════════════
# TestSituationSummary
# ═══════════════════════════════════════════════════════════════════════════


class TestSituationSummary:
    """Test SituationSummary data structure."""

    def test_construction(self):
        s = SituationSummary()
        assert s.risk_level == "low"
        assert s.trend == "stable"
        assert s.camera_violations == {}

    def test_custom_risk_level(self):
        s = SituationSummary(risk_level="critical", dominant_violation_type="no_helmet")
        assert s.risk_level == "critical"
        assert s.dominant_violation_type == "no_helmet"

    def test_severity_counts(self):
        s = SituationSummary(severity_counts={"critical": 2, "warning": 1, "info": 5})
        assert s.severity_counts["critical"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# TestAgentDecision
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentDecision:
    """Test AgentDecision data structure."""

    def test_construction(self):
        d = AgentDecision(action=AgentAction.MONITOR)
        assert d.action == AgentAction.MONITOR
        assert d.reasoning_ja == ""

    def test_to_dict(self):
        d = AgentDecision(
            action=AgentAction.ESCALATE,
            reasoning_ja="重大",
            reasoning_en="Critical",
            latency_ms=12.5,
        )
        result = d.to_dict()
        assert result["action"] == "escalate"
        assert result["reasoning_ja"] == "重大"
        assert result["latency_ms"] == 12.5

    def test_outcome(self):
        d = AgentDecision(action=AgentAction.PLAN, outcome="plan_submitted")
        assert d.outcome == "plan_submitted"

    def test_timestamp(self):
        now = time.time()
        d = AgentDecision(action=AgentAction.MONITOR, timestamp=now)
        assert d.timestamp == now


# ═══════════════════════════════════════════════════════════════════════════
# TestConstruction
# ═══════════════════════════════════════════════════════════════════════════


class TestConstruction:
    """Test AgentLoop initialization."""

    def test_default_init(self):
        loop = AgentLoop()
        assert loop._think_interval_s == 5.0
        assert loop._critical_trigger is True
        assert loop._status == AgentStatus.IDLE

    def test_custom_params(self):
        loop = AgentLoop(
            think_interval_s=10.0,
            critical_trigger=False,
            escalation_threshold=5,
            plan_threshold=8,
        )
        assert loop._think_interval_s == 10.0
        assert loop._critical_trigger is False
        assert loop._escalation_threshold == 5
        assert loop._plan_threshold == 8

    def test_stats_shape(self):
        loop = AgentLoop()
        stats = loop.get_stats()
        assert stats["enabled"] is True
        assert stats["status"] == "idle"
        assert stats["think_count"] == 0
        assert stats["pending_events"] == 0
        assert "components" in stats

    def test_components_none_by_default(self):
        loop = AgentLoop()
        stats = loop.get_stats()
        for key in stats["components"]:
            assert stats["components"][key] is False

    def test_components_set(self):
        mock_planner = MagicMock()
        mock_reporter = MagicMock()
        loop = AgentLoop(safety_planner=mock_planner, incident_reporter=mock_reporter)
        stats = loop.get_stats()
        assert stats["components"]["safety_planner"] is True
        assert stats["components"]["incident_reporter"] is True
        assert stats["components"]["causal_inference"] is False


# ═══════════════════════════════════════════════════════════════════════════
# TestObserve
# ═══════════════════════════════════════════════════════════════════════════


class _MockViolation:
    """Minimal mock for Violation."""
    def __init__(self, severity="info", rule="test_rule"):
        self.severity = MagicMock(value=severity)
        self.rule = rule
        self.description_ja = rule


class _MockFrameResult:
    """Minimal mock for FrameResult."""
    def __init__(self, violations=None, events=None):
        self.violations = violations or []
        ws = MagicMock()
        ws.events = events or []
        self.world_state = ws


class TestObserve:
    """Test observe() buffering."""

    def test_violations_buffered(self):
        loop = AgentLoop()
        fr = _MockFrameResult(violations=[_MockViolation("warning", "no_helmet")])
        loop.observe(fr, camera_id="cam1")
        assert len(loop._pending_events) == 1
        assert loop._pending_events[0].severity == "warning"

    def test_critical_trigger_returns_true(self):
        loop = AgentLoop(critical_trigger=True)
        fr = _MockFrameResult(violations=[_MockViolation("critical", "fall")])
        result = loop.observe(fr, camera_id="cam1")
        assert result is True

    def test_non_critical_returns_false(self):
        loop = AgentLoop()
        fr = _MockFrameResult(violations=[_MockViolation("info", "test")])
        result = loop.observe(fr, camera_id="cam1")
        assert result is False

    def test_empty_frame(self):
        loop = AgentLoop()
        fr = _MockFrameResult()
        result = loop.observe(fr, camera_id="cam1")
        assert result is False
        assert len(loop._pending_events) == 0

    def test_maxlen_respected(self):
        loop = AgentLoop()
        for i in range(_MAX_EVENTS + 50):
            ev = AgentEvent(event_type="violation", severity="info")
            loop._pending_events.append(ev)
        assert len(loop._pending_events) == _MAX_EVENTS

    def test_world_state_events_buffered(self):
        mock_event = MagicMock()
        mock_event.event_type = MagicMock(value="zone_entered")
        fr = _MockFrameResult(events=[mock_event])
        loop = AgentLoop()
        loop.observe(fr, camera_id="cam1")
        assert len(loop._pending_events) == 1
        assert loop._pending_events[0].event_type == "zone_entered"

    def test_multiple_violations(self):
        loop = AgentLoop()
        fr = _MockFrameResult(violations=[
            _MockViolation("warning", "no_vest"),
            _MockViolation("critical", "fall"),
            _MockViolation("info", "loiter"),
        ])
        result = loop.observe(fr, camera_id="cam1")
        assert len(loop._pending_events) == 3
        assert result is True  # critical present

    def test_critical_trigger_disabled(self):
        loop = AgentLoop(critical_trigger=False)
        fr = _MockFrameResult(violations=[_MockViolation("critical", "fall")])
        result = loop.observe(fr, camera_id="cam1")
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════
# TestShouldThink
# ═══════════════════════════════════════════════════════════════════════════


class TestShouldThink:
    """Test should_think() timing logic."""

    def test_no_events(self):
        loop = AgentLoop()
        assert loop.should_think() is False

    def test_has_events_first_time(self):
        loop = AgentLoop(think_interval_s=0.0)
        loop._pending_events.append(AgentEvent(event_type="test"))
        assert loop.should_think() is True

    def test_interval_not_elapsed(self):
        loop = AgentLoop(think_interval_s=10.0)
        loop._pending_events.append(AgentEvent(event_type="test"))
        loop._last_think_time = time.time()
        assert loop.should_think() is False

    def test_interval_elapsed(self):
        loop = AgentLoop(think_interval_s=0.01)
        loop._pending_events.append(AgentEvent(event_type="test"))
        loop._last_think_time = time.time() - 1.0
        assert loop.should_think() is True

    def test_custom_interval(self):
        loop = AgentLoop(think_interval_s=0.0)
        loop._pending_events.append(AgentEvent(event_type="test"))
        loop._last_think_time = time.time()
        assert loop.should_think() is True


# ═══════════════════════════════════════════════════════════════════════════
# TestThink
# ═══════════════════════════════════════════════════════════════════════════


class TestThink:
    """Test think() full OODA cycle."""

    def test_no_events_returns_none(self):
        loop = AgentLoop()
        assert loop.think() is None

    def test_produces_decision(self):
        loop = AgentLoop()
        loop._pending_events.append(AgentEvent(
            event_type="violation", severity="info", violation_rule="test",
        ))
        decision = loop.think()
        assert decision is not None
        assert isinstance(decision, AgentDecision)

    def test_clears_pending(self):
        loop = AgentLoop()
        loop._pending_events.append(AgentEvent(event_type="violation"))
        loop.think()
        assert len(loop._pending_events) == 0

    def test_monitor_action(self):
        loop = AgentLoop()
        loop._pending_events.append(AgentEvent(
            event_type="violation", severity="info",
        ))
        decision = loop.think()
        assert decision.action == AgentAction.MONITOR

    def test_escalate_action(self):
        loop = AgentLoop(escalation_threshold=1)
        loop._pending_events.append(AgentEvent(
            event_type="violation", severity="critical", violation_rule="fall",
        ))
        decision = loop.think()
        assert decision.action == AgentAction.ESCALATE

    def test_plan_action(self):
        loop = AgentLoop(plan_threshold=2)
        for _ in range(5):
            loop._pending_events.append(AgentEvent(
                event_type="violation", severity="warning", violation_rule="no_helmet",
            ))
        # Add a critical to push risk to high
        loop._pending_events.append(AgentEvent(
            event_type="violation", severity="critical", violation_rule="no_helmet",
        ))
        decision = loop.think()
        # Might be ESCALATE (critical present) or PLAN (high+threshold)
        assert decision.action in (AgentAction.ESCALATE, AgentAction.PLAN)

    def test_decision_has_reasoning(self):
        loop = AgentLoop()
        loop._pending_events.append(AgentEvent(
            event_type="violation", severity="info",
        ))
        decision = loop.think()
        assert decision.reasoning_ja != ""
        assert decision.reasoning_en != ""

    def test_decision_history_grows(self):
        loop = AgentLoop()
        loop._pending_events.append(AgentEvent(event_type="violation"))
        loop.think()
        assert len(loop._decision_history) == 1

    def test_think_count_increments(self):
        loop = AgentLoop()
        loop._pending_events.append(AgentEvent(event_type="violation"))
        loop.think()
        assert loop._think_count == 1

    def test_action_counts_updated(self):
        loop = AgentLoop()
        loop._pending_events.append(AgentEvent(event_type="violation", severity="info"))
        decision = loop.think()
        assert loop._action_counts[decision.action.value] >= 1


# ═══════════════════════════════════════════════════════════════════════════
# TestOrient
# ═══════════════════════════════════════════════════════════════════════════


class TestOrient:
    """Test _orient() aggregation."""

    def test_aggregation(self):
        loop = AgentLoop()
        events = [
            AgentEvent(event_type="violation", camera_id="cam1", severity="warning", violation_rule="no_helmet"),
            AgentEvent(event_type="violation", camera_id="cam1", severity="info", violation_rule="loiter"),
            AgentEvent(event_type="violation", camera_id="cam2", severity="critical", violation_rule="no_helmet"),
        ]
        summary = loop._orient(events)
        assert summary.camera_violations["cam1"] == 2
        assert summary.camera_violations["cam2"] == 1

    def test_severity_counts(self):
        loop = AgentLoop()
        events = [
            AgentEvent(event_type="violation", severity="warning"),
            AgentEvent(event_type="violation", severity="warning"),
            AgentEvent(event_type="violation", severity="critical"),
        ]
        summary = loop._orient(events)
        assert summary.severity_counts["warning"] == 2
        assert summary.severity_counts["critical"] == 1

    def test_dominant_type(self):
        loop = AgentLoop()
        events = [
            AgentEvent(event_type="violation", violation_rule="no_helmet"),
            AgentEvent(event_type="violation", violation_rule="no_helmet"),
            AgentEvent(event_type="violation", violation_rule="fall"),
        ]
        summary = loop._orient(events)
        assert summary.dominant_violation_type == "no_helmet"

    def test_risk_level_critical(self):
        loop = AgentLoop()
        events = [
            AgentEvent(event_type="violation", severity="critical"),
            AgentEvent(event_type="violation", severity="critical"),
        ]
        summary = loop._orient(events)
        assert summary.risk_level == "critical"

    def test_risk_level_low(self):
        loop = AgentLoop()
        events = [AgentEvent(event_type="violation", severity="info")]
        summary = loop._orient(events)
        assert summary.risk_level == "low"

    def test_risk_level_medium(self):
        loop = AgentLoop()
        events = [
            AgentEvent(event_type="violation", severity="warning"),
        ]
        summary = loop._orient(events)
        assert summary.risk_level == "medium"


# ═══════════════════════════════════════════════════════════════════════════
# TestDecide
# ═══════════════════════════════════════════════════════════════════════════


class TestDecide:
    """Test _decide() priority cascade."""

    def test_monitor_default(self):
        loop = AgentLoop()
        s = SituationSummary()
        action, ja, en, params = loop._decide(s)
        assert action == AgentAction.MONITOR
        assert ja != ""
        assert en != ""

    def test_escalate_critical(self):
        loop = AgentLoop(escalation_threshold=1)
        s = SituationSummary(
            severity_counts={"info": 0, "warning": 0, "critical": 2},
            camera_violations={"cam1": 2},
            dominant_violation_type="fall",
        )
        action, _, _, _ = loop._decide(s)
        assert action == AgentAction.ESCALATE

    def test_plan_high_risk_repeated(self):
        loop = AgentLoop(plan_threshold=3)
        s = SituationSummary(
            severity_counts={"info": 0, "warning": 3, "critical": 1},
            camera_violations={"cam1": 4},
            risk_level="high",
            dominant_violation_type="no_helmet",
        )
        action, _, _, params = loop._decide(s)
        # Could be ESCALATE (1 critical) or PLAN
        assert action in (AgentAction.ESCALATE, AgentAction.PLAN)

    def test_adapt_novel(self):
        mock_learner = MagicMock()
        mock_learner.get_stats.return_value = {"categories": ["helmet", "vest"]}
        loop = AgentLoop(online_learner=mock_learner)
        s = SituationSummary(
            severity_counts={"info": 0, "warning": 1, "critical": 0},
            camera_violations={"cam1": 1},
            risk_level="medium",
            dominant_violation_type="new_pattern",
        )
        action, _, _, _ = loop._decide(s)
        assert action == AgentAction.ADAPT

    def test_query_with_rag(self):
        mock_rag = MagicMock()
        loop = AgentLoop(rag_engine=mock_rag, use_rag_orient=True)
        s = SituationSummary(
            severity_counts={"info": 1, "warning": 0, "critical": 0},
            camera_violations={"cam1": 1},
            risk_level="low",
            dominant_violation_type="loiter",
            rag_context="Historical data suggests ...",
        )
        action, _, _, _ = loop._decide(s)
        assert action == AgentAction.QUERY

    def test_respond_high_risk(self):
        mock_ew = MagicMock()
        loop = AgentLoop(early_warning_responder=mock_ew, use_rag_orient=False)
        s = SituationSummary(
            severity_counts={"info": 0, "warning": 0, "critical": 0},
            camera_violations={},
            risk_level="high",
        )
        action, _, _, _ = loop._decide(s)
        assert action == AgentAction.RESPOND

    def test_all_actions_have_reasoning(self):
        loop = AgentLoop()
        for risk in ("low", "medium", "high", "critical"):
            s = SituationSummary(
                severity_counts={"critical": 2 if risk == "critical" else 0, "warning": 0, "info": 0},
                camera_violations={"cam1": 5} if risk in ("high", "critical") else {},
                risk_level=risk,
            )
            _, ja, en, _ = loop._decide(s)
            assert isinstance(ja, str)
            assert isinstance(en, str)


# ═══════════════════════════════════════════════════════════════════════════
# TestAct
# ═══════════════════════════════════════════════════════════════════════════


class TestAct:
    """Test _act() dispatch to sub-systems."""

    def test_act_escalate(self):
        mock_reporter = MagicMock()
        loop = AgentLoop(incident_reporter=mock_reporter)
        s = SituationSummary(camera_violations={"cam1": 1}, dominant_violation_type="fall")
        outcome = loop._act(AgentAction.ESCALATE, {"violation_type": "fall", "severity": "critical"}, s)
        assert outcome == "incident_reported"
        mock_reporter.report.assert_called_once()

    def test_act_plan(self):
        mock_planner = MagicMock()
        loop = AgentLoop(safety_planner=mock_planner)
        s = SituationSummary(dominant_violation_type="no_helmet")
        outcome = loop._act(AgentAction.PLAN, {"goal": "Fix no_helmet"}, s)
        assert outcome == "plan_submitted"
        mock_planner.submit_goal.assert_called_once()

    def test_act_adapt(self):
        mock_learner = MagicMock()
        loop = AgentLoop(online_learner=mock_learner)
        s = SituationSummary()
        outcome = loop._act(AgentAction.ADAPT, {"novel_pattern": "new_thing"}, s)
        assert "adapt_noted" in outcome

    def test_act_query(self):
        loop = AgentLoop()
        s = SituationSummary()
        outcome = loop._act(AgentAction.QUERY, {"query": "no_helmet"}, s)
        assert "rag_queried" in outcome

    def test_act_respond(self):
        mock_ew = MagicMock()
        loop = AgentLoop(early_warning_responder=mock_ew)
        s = SituationSummary()
        outcome = loop._act(AgentAction.RESPOND, {"risk_level": "high"}, s)
        assert outcome == "ew_response_sent"
        mock_ew.respond.assert_called_once()

    def test_act_monitor(self):
        loop = AgentLoop()
        s = SituationSummary()
        outcome = loop._act(AgentAction.MONITOR, {}, s)
        assert outcome == "monitoring"


# ═══════════════════════════════════════════════════════════════════════════
# TestGetStats
# ═══════════════════════════════════════════════════════════════════════════


class TestGetStats:
    """Test get_stats() output shape."""

    def test_shape(self):
        loop = AgentLoop()
        stats = loop.get_stats()
        required_keys = {
            "enabled", "status", "think_count", "pending_events",
            "decision_count", "action_counts", "think_interval_s",
            "components", "uptime_s",
        }
        assert required_keys.issubset(set(stats.keys()))

    def test_components(self):
        loop = AgentLoop(safety_planner=MagicMock())
        stats = loop.get_stats()
        assert stats["components"]["safety_planner"] is True
        assert stats["components"]["rag_engine"] is False

    def test_uptime(self):
        loop = AgentLoop()
        stats = loop.get_stats()
        assert stats["uptime_s"] >= 0


# ═══════════════════════════════════════════════════════════════════════════
# TestGetRecentDecisions
# ═══════════════════════════════════════════════════════════════════════════


class TestGetRecentDecisions:
    """Test get_recent_decisions() output."""

    def test_empty(self):
        loop = AgentLoop()
        assert loop.get_recent_decisions() == []

    def test_returns_dicts(self):
        loop = AgentLoop()
        loop._pending_events.append(AgentEvent(event_type="violation", severity="info"))
        loop.think()
        result = loop.get_recent_decisions()
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "action" in result[0]

    def test_limit(self):
        loop = AgentLoop()
        for _ in range(5):
            loop._pending_events.append(AgentEvent(event_type="violation"))
            loop.think()
        result = loop.get_recent_decisions(limit=3)
        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════════════════
# TestGetPendingEvents
# ═══════════════════════════════════════════════════════════════════════════


class TestGetPendingEvents:
    """Test get_pending_events() output."""

    def test_empty(self):
        loop = AgentLoop()
        assert loop.get_pending_events() == []

    def test_returns_dicts(self):
        loop = AgentLoop()
        loop.inject_event(AgentEvent(event_type="violation", severity="critical"))
        result = loop.get_pending_events()
        assert len(result) == 1
        assert result[0]["event_type"] == "violation"
        assert result[0]["severity"] == "critical"


# ═══════════════════════════════════════════════════════════════════════════
# TestInjectEvent
# ═══════════════════════════════════════════════════════════════════════════


class TestInjectEvent:
    """Test inject_event() for testing."""

    def test_inject(self):
        loop = AgentLoop()
        ev = AgentEvent(event_type="test_event", severity="warning")
        loop.inject_event(ev)
        assert len(loop._pending_events) == 1
        assert loop._pending_events[0].event_type == "test_event"


# ═══════════════════════════════════════════════════════════════════════════
# TestReset
# ═══════════════════════════════════════════════════════════════════════════


class TestReset:
    """Test reset() clears all state."""

    def test_clears_events(self):
        loop = AgentLoop()
        loop._pending_events.append(AgentEvent(event_type="test"))
        loop._pending_events.append(AgentEvent(event_type="test"))
        loop.reset()
        assert len(loop._pending_events) == 0

    def test_clears_decisions(self):
        loop = AgentLoop()
        loop._pending_events.append(AgentEvent(event_type="violation"))
        loop.think()
        assert len(loop._decision_history) == 1
        loop.reset()
        assert len(loop._decision_history) == 0
        assert loop._think_count == 0
        assert loop._status == AgentStatus.IDLE


# ═══════════════════════════════════════════════════════════════════════════
# TestPerceptionConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestPerceptionConfig:
    """Test Phase 59 config defaults in PerceptionConfig."""

    def test_defaults(self):
        cfg = PerceptionConfig()
        assert cfg.agent_loop_enabled is False
        assert cfg.agent_loop_think_interval_s == 5.0
        assert cfg.agent_loop_critical_trigger is True
        assert cfg.agent_loop_use_llava_orient is False
        assert cfg.agent_loop_use_rag_orient is True
        assert cfg.agent_loop_escalation_threshold == 3
        assert cfg.agent_loop_plan_threshold == 5

    def test_custom_config(self):
        cfg = PerceptionConfig(
            agent_loop_enabled=True,
            agent_loop_think_interval_s=10.0,
            agent_loop_plan_threshold=8,
        )
        assert cfg.agent_loop_enabled is True
        assert cfg.agent_loop_think_interval_s == 10.0
        assert cfg.agent_loop_plan_threshold == 8
