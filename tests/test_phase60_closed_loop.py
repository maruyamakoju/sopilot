"""Tests for Phase 60: Closed-Loop Verification.

Covers:
  - VerificationOutcome enum
  - VerificationTask / VerificationRecord data structures
  - MetaLearner EMA updates and threshold suggestions
  - ClosedLoopVerifier: schedule, check_pending, compute_outcome, meta_update
  - get_stats, get_recent_records, get_pending_tasks, reset
  - PerceptionConfig defaults
  - AgentLoop verifier integration
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from sopilot.perception.closed_loop_verifier import (
    ClosedLoopVerifier,
    MetaLearner,
    VerificationOutcome,
    VerificationRecord,
    VerificationTask,
    _EMA_ALPHA,
    _THRESHOLD_MAX_DELTA,
)
from sopilot.perception.agent_loop import AgentAction, AgentEvent, AgentLoop
from sopilot.perception.types import PerceptionConfig


# ═══════════════════════════════════════════════════════════════════════════
# TestVerificationOutcome
# ═══════════════════════════════════════════════════════════════════════════


class TestVerificationOutcome:
    def test_values(self):
        assert VerificationOutcome.SUCCESS == "success"
        assert VerificationOutcome.PARTIAL == "partial"
        assert VerificationOutcome.FAILURE == "failure"
        assert VerificationOutcome.UNKNOWN == "unknown"

    def test_count(self):
        assert len(VerificationOutcome) == 4

    def test_is_str_enum(self):
        assert isinstance(VerificationOutcome.SUCCESS, str)


# ═══════════════════════════════════════════════════════════════════════════
# TestVerificationTask
# ═══════════════════════════════════════════════════════════════════════════


class TestVerificationTask:
    def _make(self, check_after_s=30.0, **kw) -> VerificationTask:
        return VerificationTask(
            task_id="abc",
            decision_action="escalate",
            camera_id="cam1",
            baseline_violations=5,
            baseline_severity={"critical": 1, "warning": 2, "info": 2},
            baseline_risk_level="high",
            scheduled_at=time.time(),
            check_after_s=check_after_s,
            **kw,
        )

    def test_construction(self):
        t = self._make()
        assert t.task_id == "abc"
        assert t.baseline_violations == 5
        assert t.decision_action == "escalate"

    def test_is_due_false(self):
        t = self._make(check_after_s=9999.0)
        assert t.is_due is False

    def test_is_due_true(self):
        t = self._make(check_after_s=0.0)
        assert t.is_due is True

    def test_check_due_at(self):
        now = time.time()
        t = VerificationTask(
            task_id="x", decision_action="plan", camera_id="c",
            baseline_violations=3, baseline_severity={}, baseline_risk_level="low",
            scheduled_at=now, check_after_s=10.0,
        )
        assert abs(t.check_due_at - (now + 10.0)) < 0.1

    def test_decision_reasoning_default(self):
        t = self._make()
        assert t.decision_reasoning_ja == ""


# ═══════════════════════════════════════════════════════════════════════════
# TestVerificationRecord
# ═══════════════════════════════════════════════════════════════════════════


class TestVerificationRecord:
    def _make(self, outcome=VerificationOutcome.SUCCESS) -> VerificationRecord:
        return VerificationRecord(
            task_id="t1",
            decision_action="escalate",
            camera_id="cam1",
            baseline_violations=10,
            current_violations=3,
            baseline_risk_level="high",
            outcome=outcome,
            reduction_ratio=0.7,
            latency_s=32.5,
            verified_at=time.time(),
        )

    def test_construction(self):
        r = self._make()
        assert r.baseline_violations == 10
        assert r.current_violations == 3
        assert r.outcome == VerificationOutcome.SUCCESS

    def test_to_dict(self):
        r = self._make()
        d = r.to_dict()
        assert d["outcome"] == "success"
        assert d["reduction_ratio"] == 0.7
        assert d["task_id"] == "t1"

    def test_to_dict_failure(self):
        r = self._make(VerificationOutcome.FAILURE)
        d = r.to_dict()
        assert d["outcome"] == "failure"

    def test_reduction_ratio(self):
        r = self._make()
        assert r.reduction_ratio == 0.7


# ═══════════════════════════════════════════════════════════════════════════
# TestMetaLearner
# ═══════════════════════════════════════════════════════════════════════════


class TestMetaLearner:
    def test_initial_state(self):
        m = MetaLearner()
        stats = m.get_stats()
        assert stats["success_rates"] == {}
        assert stats["escalation_threshold_adj"] == 0
        assert stats["plan_threshold_adj"] == 0

    def test_update_success(self):
        m = MetaLearner()
        m.update("escalate", VerificationOutcome.SUCCESS)
        stats = m.get_stats()
        assert "escalate" in stats["success_rates"]
        assert stats["success_rates"]["escalate"] == 1.0

    def test_update_failure(self):
        m = MetaLearner()
        m.update("escalate", VerificationOutcome.FAILURE)
        stats = m.get_stats()
        assert stats["success_rates"]["escalate"] == 0.0

    def test_ema_smoothing(self):
        m = MetaLearner()
        m.update("plan", VerificationOutcome.SUCCESS)   # 1.0
        m.update("plan", VerificationOutcome.FAILURE)   # EMA alpha*0 + (1-alpha)*1
        stats = m.get_stats()
        expected = _EMA_ALPHA * 0.0 + (1 - _EMA_ALPHA) * 1.0
        assert abs(stats["success_rates"]["plan"] - expected) < 1e-6

    def test_escalation_threshold_raise_on_low_success(self):
        m = MetaLearner(original_escalation=3)
        # 3+ failures → adj should increase
        for _ in range(5):
            m.update("escalate", VerificationOutcome.FAILURE)
        assert m._escalation_adj >= 1

    def test_plan_threshold_raise_on_low_success(self):
        m = MetaLearner(original_plan=5)
        for _ in range(5):
            m.update("plan", VerificationOutcome.FAILURE)
        assert m._plan_adj >= 1

    def test_suggested_thresholds_never_below_min(self):
        m = MetaLearner(original_escalation=1, original_plan=2)
        # Force negative adj
        m._escalation_adj = -999
        m._plan_adj = -999
        assert m.suggested_escalation_threshold >= 1
        assert m.suggested_plan_threshold >= 2

    def test_reset(self):
        m = MetaLearner()
        m.update("escalate", VerificationOutcome.SUCCESS)
        m.reset()
        assert m.get_stats()["success_rates"] == {}
        assert m._escalation_adj == 0

    def test_max_delta_capped(self):
        m = MetaLearner()
        m._escalation_adj = _THRESHOLD_MAX_DELTA
        # Another failure should NOT exceed max
        m.update("escalate", VerificationOutcome.FAILURE)
        m.update("escalate", VerificationOutcome.FAILURE)
        m.update("escalate", VerificationOutcome.FAILURE)
        assert m._escalation_adj <= _THRESHOLD_MAX_DELTA


# ═══════════════════════════════════════════════════════════════════════════
# TestClosedLoopVerifierConstruction
# ═══════════════════════════════════════════════════════════════════════════


class TestClosedLoopVerifierConstruction:
    def test_defaults(self):
        v = ClosedLoopVerifier()
        assert v._verify_after_s == 30.0
        assert v._success_threshold == 0.5
        assert v._meta_learning_enabled is True

    def test_custom_params(self):
        v = ClosedLoopVerifier(
            verify_after_s=10.0,
            success_threshold=0.3,
            meta_learning_enabled=False,
        )
        assert v._verify_after_s == 10.0
        assert v._success_threshold == 0.3
        assert v._meta_learning_enabled is False

    def test_stats_shape(self):
        v = ClosedLoopVerifier()
        s = v.get_stats()
        required = {"enabled", "pending_tasks", "completed_records",
                    "outcome_counts", "meta", "uptime_s"}
        assert required.issubset(set(s.keys()))

    def test_stats_enabled(self):
        v = ClosedLoopVerifier()
        assert v.get_stats()["enabled"] is True

    def test_outcome_counts_initialized(self):
        v = ClosedLoopVerifier()
        counts = v.get_stats()["outcome_counts"]
        assert set(counts.keys()) == {"success", "partial", "failure", "unknown"}


# ═══════════════════════════════════════════════════════════════════════════
# TestSchedule
# ═══════════════════════════════════════════════════════════════════════════


class TestSchedule:
    def test_returns_task_id(self):
        v = ClosedLoopVerifier()
        tid = v.schedule("escalate", "cam1", 5, {}, "high")
        assert isinstance(tid, str)
        assert len(tid) > 0

    def test_pending_count_increases(self):
        v = ClosedLoopVerifier()
        v.schedule("escalate", "cam1", 5, {}, "high")
        assert v.get_stats()["pending_tasks"] == 1

    def test_multiple_schedules(self):
        v = ClosedLoopVerifier()
        v.schedule("escalate", "cam1", 5, {}, "high")
        v.schedule("plan", "cam2", 3, {}, "medium")
        assert v.get_stats()["pending_tasks"] == 2

    def test_unique_task_ids(self):
        v = ClosedLoopVerifier()
        ids = [v.schedule("escalate", "cam1", 1, {}, "low") for _ in range(5)]
        assert len(set(ids)) == 5

    def test_reasoning_stored(self):
        v = ClosedLoopVerifier()
        v.schedule("plan", "cam1", 3, {}, "medium", "重大違反が続いています")
        tasks = v.get_pending_tasks()
        assert any("重大" in t["decision_reasoning_ja"] for t in tasks)

    def test_max_pending_evicts_oldest(self):
        from sopilot.perception.closed_loop_verifier import _MAX_PENDING
        v = ClosedLoopVerifier()
        for i in range(_MAX_PENDING + 5):
            v.schedule("monitor", f"cam{i}", i, {}, "low")
        assert len(v._pending) == _MAX_PENDING


# ═══════════════════════════════════════════════════════════════════════════
# TestCheckPending
# ═══════════════════════════════════════════════════════════════════════════


class _MockFrame:
    def __init__(self, violation_count: int = 0):
        self.violations = [None] * violation_count


class TestCheckPending:
    def test_not_due_returns_empty(self):
        v = ClosedLoopVerifier(verify_after_s=9999.0)
        v.schedule("escalate", "cam1", 5, {}, "high")
        completed = v.check_pending(_MockFrame(3), "cam1")
        assert completed == []
        assert v.get_stats()["pending_tasks"] == 1

    def test_due_returns_record(self):
        v = ClosedLoopVerifier(verify_after_s=0.0)
        v.schedule("escalate", "cam1", 5, {}, "high")
        completed = v.check_pending(_MockFrame(1), "cam1")
        assert len(completed) == 1
        assert isinstance(completed[0], VerificationRecord)

    def test_clears_pending_after_check(self):
        v = ClosedLoopVerifier(verify_after_s=0.0)
        v.schedule("escalate", "cam1", 5, {}, "high")
        v.check_pending(_MockFrame(1), "cam1")
        assert v.get_stats()["pending_tasks"] == 0

    def test_wrong_camera_not_checked(self):
        v = ClosedLoopVerifier(verify_after_s=0.0)
        v.schedule("escalate", "cam1", 5, {}, "high")
        completed = v.check_pending(_MockFrame(1), "cam2")
        assert completed == []
        assert v.get_stats()["pending_tasks"] == 1

    def test_outcome_success(self):
        v = ClosedLoopVerifier(verify_after_s=0.0, success_threshold=0.5)
        v.schedule("escalate", "cam1", 10, {}, "high")
        completed = v.check_pending(_MockFrame(3), "cam1")  # 70% reduction
        assert completed[0].outcome == VerificationOutcome.SUCCESS

    def test_outcome_partial(self):
        v = ClosedLoopVerifier(verify_after_s=0.0, success_threshold=0.5)
        v.schedule("escalate", "cam1", 10, {}, "high")
        completed = v.check_pending(_MockFrame(8), "cam1")  # 20% reduction
        assert completed[0].outcome == VerificationOutcome.PARTIAL

    def test_outcome_failure(self):
        v = ClosedLoopVerifier(verify_after_s=0.0)
        v.schedule("escalate", "cam1", 10, {}, "high")
        completed = v.check_pending(_MockFrame(12), "cam1")  # increased
        assert completed[0].outcome == VerificationOutcome.FAILURE

    def test_outcome_unknown_zero_baseline(self):
        v = ClosedLoopVerifier(verify_after_s=0.0)
        v.schedule("escalate", "cam1", 0, {}, "low")
        completed = v.check_pending(_MockFrame(0), "cam1")
        assert completed[0].outcome == VerificationOutcome.UNKNOWN

    def test_completed_records_saved(self):
        v = ClosedLoopVerifier(verify_after_s=0.0)
        v.schedule("escalate", "cam1", 5, {}, "high")
        v.check_pending(_MockFrame(1), "cam1")
        assert v.get_stats()["completed_records"] == 1

    def test_outcome_counts_updated(self):
        v = ClosedLoopVerifier(verify_after_s=0.0, success_threshold=0.5)
        v.schedule("escalate", "cam1", 10, {}, "high")
        v.check_pending(_MockFrame(2), "cam1")
        counts = v.get_stats()["outcome_counts"]
        assert counts["success"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# TestComputeOutcome
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeOutcome:
    def _v(self, threshold=0.5) -> ClosedLoopVerifier:
        return ClosedLoopVerifier(success_threshold=threshold)

    def test_success_50_percent(self):
        v = self._v()
        outcome, ratio = v._compute_outcome(10, 5)
        assert outcome == VerificationOutcome.SUCCESS
        assert abs(ratio - 0.5) < 1e-6

    def test_success_100_percent(self):
        v = self._v()
        outcome, ratio = v._compute_outcome(10, 0)
        assert outcome == VerificationOutcome.SUCCESS
        assert ratio == 1.0

    def test_partial_30_percent(self):
        v = self._v()
        outcome, ratio = v._compute_outcome(10, 7)
        assert outcome == VerificationOutcome.PARTIAL
        assert abs(ratio - 0.3) < 1e-6

    def test_failure_no_improvement(self):
        v = self._v()
        outcome, ratio = v._compute_outcome(10, 10)
        assert outcome == VerificationOutcome.FAILURE

    def test_failure_increase(self):
        v = self._v()
        outcome, ratio = v._compute_outcome(10, 20)
        assert outcome == VerificationOutcome.FAILURE
        assert ratio < 0

    def test_unknown_zero_baseline_zero_current(self):
        v = self._v()
        outcome, ratio = v._compute_outcome(0, 0)
        assert outcome == VerificationOutcome.UNKNOWN

    def test_unknown_zero_baseline_nonzero_current(self):
        v = self._v()
        outcome, ratio = v._compute_outcome(0, 5)
        assert outcome == VerificationOutcome.FAILURE

    def test_custom_threshold(self):
        v = self._v(threshold=0.9)
        # 80% reduction but threshold is 90% → PARTIAL
        outcome, _ = v._compute_outcome(10, 2)
        assert outcome == VerificationOutcome.PARTIAL


# ═══════════════════════════════════════════════════════════════════════════
# TestMetaUpdate
# ═══════════════════════════════════════════════════════════════════════════


class TestMetaUpdate:
    def test_meta_update_on_check(self):
        v = ClosedLoopVerifier(verify_after_s=0.0, meta_learning_enabled=True)
        v.schedule("escalate", "cam1", 10, {}, "high")
        v.check_pending(_MockFrame(2), "cam1")
        meta = v.get_meta_state()
        assert "escalate" in meta["success_rates"]

    def test_meta_disabled(self):
        v = ClosedLoopVerifier(verify_after_s=0.0, meta_learning_enabled=False)
        v.schedule("escalate", "cam1", 10, {}, "high")
        v.check_pending(_MockFrame(2), "cam1")
        meta = v.get_meta_state()
        assert meta["success_rates"] == {}

    def test_meta_suggested_thresholds_present(self):
        v = ClosedLoopVerifier()
        meta = v.get_meta_state()
        assert "suggested_escalation_threshold" in meta
        assert "suggested_plan_threshold" in meta

    def test_meta_multiple_outcomes(self):
        v = ClosedLoopVerifier(verify_after_s=0.0, success_threshold=0.5)
        for violations in [2, 2, 8]:
            v.schedule("escalate", "cam1", 10, {}, "high")
        for viol_count in [2, 2, 8]:
            v.check_pending(_MockFrame(viol_count), "cam1")
        meta = v.get_meta_state()
        assert meta["update_counts"].get("escalate", 0) == 3


# ═══════════════════════════════════════════════════════════════════════════
# TestGetStats
# ═══════════════════════════════════════════════════════════════════════════


class TestGetStats:
    def test_pending_count(self):
        v = ClosedLoopVerifier()
        v.schedule("plan", "cam1", 3, {}, "medium")
        assert v.get_stats()["pending_tasks"] == 1

    def test_completed_count(self):
        v = ClosedLoopVerifier(verify_after_s=0.0)
        v.schedule("plan", "cam1", 3, {}, "medium")
        v.check_pending(_MockFrame(1), "cam1")
        assert v.get_stats()["completed_records"] == 1

    def test_uptime_positive(self):
        v = ClosedLoopVerifier()
        assert v.get_stats()["uptime_s"] >= 0

    def test_verify_after_s_in_stats(self):
        v = ClosedLoopVerifier(verify_after_s=60.0)
        assert v.get_stats()["verify_after_s"] == 60.0


# ═══════════════════════════════════════════════════════════════════════════
# TestGetRecentRecords
# ═══════════════════════════════════════════════════════════════════════════


class TestGetRecentRecords:
    def test_empty(self):
        v = ClosedLoopVerifier()
        assert v.get_recent_records() == []

    def test_returns_dicts(self):
        v = ClosedLoopVerifier(verify_after_s=0.0)
        v.schedule("escalate", "cam1", 5, {}, "high")
        v.check_pending(_MockFrame(1), "cam1")
        records = v.get_recent_records()
        assert len(records) == 1
        assert isinstance(records[0], dict)
        assert "outcome" in records[0]

    def test_limit(self):
        v = ClosedLoopVerifier(verify_after_s=0.0)
        for _ in range(5):
            v.schedule("escalate", "cam1", 5, {}, "high")
            v.check_pending(_MockFrame(1), "cam1")
        records = v.get_recent_records(limit=3)
        assert len(records) == 3

    def test_most_recent_first(self):
        v = ClosedLoopVerifier(verify_after_s=0.0, success_threshold=0.5)
        # First: SUCCESS (5→1 = 80% reduction)
        v.schedule("escalate", "cam1", 5, {}, "high")
        v.check_pending(_MockFrame(1), "cam1")
        # Second: FAILURE (5→5 = 0% reduction)
        v.schedule("escalate", "cam1", 5, {}, "high")
        v.check_pending(_MockFrame(5), "cam1")
        records = v.get_recent_records(limit=2)
        assert records[0]["outcome"] == "failure"  # most recent first


# ═══════════════════════════════════════════════════════════════════════════
# TestReset
# ═══════════════════════════════════════════════════════════════════════════


class TestReset:
    def test_clears_pending(self):
        v = ClosedLoopVerifier()
        v.schedule("plan", "cam1", 3, {}, "medium")
        v.reset()
        assert v.get_stats()["pending_tasks"] == 0

    def test_clears_records(self):
        v = ClosedLoopVerifier(verify_after_s=0.0)
        v.schedule("plan", "cam1", 3, {}, "medium")
        v.check_pending(_MockFrame(1), "cam1")
        v.reset()
        assert v.get_stats()["completed_records"] == 0

    def test_clears_outcome_counts(self):
        v = ClosedLoopVerifier(verify_after_s=0.0)
        v.schedule("plan", "cam1", 3, {}, "medium")
        v.check_pending(_MockFrame(1), "cam1")
        v.reset()
        counts = v.get_stats()["outcome_counts"]
        assert all(c == 0 for c in counts.values())


# ═══════════════════════════════════════════════════════════════════════════
# TestPerceptionConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestPerceptionConfig:
    def test_defaults(self):
        cfg = PerceptionConfig()
        assert cfg.closed_loop_enabled is False
        assert cfg.closed_loop_verify_after_s == 30.0
        assert cfg.closed_loop_success_threshold == 0.5
        assert cfg.closed_loop_meta_learning_enabled is True
        assert cfg.closed_loop_history_size == 200

    def test_custom(self):
        cfg = PerceptionConfig(
            closed_loop_enabled=True,
            closed_loop_verify_after_s=60.0,
            closed_loop_success_threshold=0.7,
        )
        assert cfg.closed_loop_enabled is True
        assert cfg.closed_loop_verify_after_s == 60.0
        assert cfg.closed_loop_success_threshold == 0.7

    def test_coexists_with_agent_loop_config(self):
        cfg = PerceptionConfig(
            agent_loop_enabled=True,
            closed_loop_enabled=True,
        )
        assert cfg.agent_loop_enabled is True
        assert cfg.closed_loop_enabled is True


# ═══════════════════════════════════════════════════════════════════════════
# TestAgentLoopVerifierIntegration
# ═══════════════════════════════════════════════════════════════════════════


class _MockViolation:
    def __init__(self, severity="warning", rule="test"):
        self.severity = MagicMock(value=severity)
        self.rule = rule
        self.description_ja = rule


class _MockFrameResult:
    def __init__(self, violations=None):
        self.violations = violations or []
        ws = MagicMock()
        ws.events = []
        self.world_state = ws


class TestAgentLoopVerifierIntegration:
    def test_verifier_accepted_in_constructor(self):
        v = ClosedLoopVerifier()
        loop = AgentLoop(verifier=v)
        stats = loop.get_stats()
        assert stats["components"]["verifier"] is True

    def test_no_verifier_by_default(self):
        loop = AgentLoop()
        assert loop.get_stats()["components"]["verifier"] is False

    def test_schedule_called_on_non_monitor_action(self):
        v = ClosedLoopVerifier()
        loop = AgentLoop(escalation_threshold=1, verifier=v)
        # Inject a critical violation to trigger ESCALATE
        loop._pending_events.append(AgentEvent(
            event_type="violation", severity="critical", violation_rule="fall",
            camera_id="cam1",
        ))
        decision = loop.think()
        assert decision is not None
        if decision.action != AgentAction.MONITOR:
            assert v.get_stats()["pending_tasks"] >= 1

    def test_monitor_does_not_schedule(self):
        v = ClosedLoopVerifier()
        loop = AgentLoop(verifier=v)
        # Only info events → MONITOR
        loop._pending_events.append(AgentEvent(
            event_type="violation", severity="info", camera_id="cam1",
        ))
        decision = loop.think()
        if decision and decision.action == AgentAction.MONITOR:
            assert v.get_stats()["pending_tasks"] == 0

    def test_verifier_check_pending_called_from_engine(self):
        """Smoke test: verifier.check_pending is callable with FrameResult."""
        v = ClosedLoopVerifier(verify_after_s=0.0)
        v.schedule("escalate", "cam1", 5, {}, "high")
        fr = _MockFrameResult(violations=[_MockViolation()])
        completed = v.check_pending(fr, "cam1")
        assert isinstance(completed, list)
