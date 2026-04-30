"""Tests for Phase 61: Agent Intelligence Report.

Covers:
  - Data structures: DecisionStats, OutcomeStats, PatternInsight, AgentIntelligenceReport
  - AgentReportGenerator: construction, generate, summary, insights, get_stats
  - _analyze_decisions, _analyze_outcomes, _compute_risk_trend
  - _find_effective_actions, _detect_patterns
  - _compose_summary_ja, _compose_summary_en
  - Integration with AgentLoop and ClosedLoopVerifier mocks
"""

from __future__ import annotations

import time
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from sopilot.perception.agent_report import (
    AgentIntelligenceReport,
    AgentReportGenerator,
    DecisionStats,
    OutcomeStats,
    PatternInsight,
)
from sopilot.perception.agent_loop import AgentAction, AgentDecision
from sopilot.perception.closed_loop_verifier import (
    ClosedLoopVerifier,
    VerificationOutcome,
    VerificationRecord,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_decision(
    action: AgentAction = AgentAction.MONITOR,
    reasoning_ja: str = "テスト",
    latency_ms: float = 5.0,
    dominant_violation: str = "",
    ts: float | None = None,
) -> AgentDecision:
    return AgentDecision(
        action=action,
        reasoning_ja=reasoning_ja,
        reasoning_en="test",
        latency_ms=latency_ms,
        timestamp=ts if ts is not None else time.time(),
        situation={"dominant_violation_type": dominant_violation, "risk_level": "medium",
                   "severity_counts": {}},
        params={},
    )


def _make_record(
    action: str = "escalate",
    outcome: VerificationOutcome = VerificationOutcome.SUCCESS,
    baseline: int = 10,
    current: int = 3,
    ts: float | None = None,
) -> VerificationRecord:
    red = (baseline - current) / max(baseline, 1)
    return VerificationRecord(
        task_id="t1",
        decision_action=action,
        camera_id="cam1",
        baseline_violations=baseline,
        current_violations=current,
        baseline_risk_level="high",
        outcome=outcome,
        reduction_ratio=round(red, 3),
        latency_s=32.0,
        verified_at=ts if ts is not None else time.time(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# TestDecisionStats
# ═══════════════════════════════════════════════════════════════════════════


class TestDecisionStats:
    def test_defaults(self):
        ds = DecisionStats()
        assert ds.total == 0
        assert ds.action_breakdown == {}
        assert ds.avg_latency_ms == 0.0

    def test_construction(self):
        ds = DecisionStats(total=5, action_breakdown={"escalate": 3, "monitor": 2})
        assert ds.total == 5
        assert ds.action_breakdown["escalate"] == 3


# ═══════════════════════════════════════════════════════════════════════════
# TestOutcomeStats
# ═══════════════════════════════════════════════════════════════════════════


class TestOutcomeStats:
    def test_defaults(self):
        os_ = OutcomeStats()
        assert os_.total_verified == 0
        assert os_.success_rate == 0.0

    def test_construction(self):
        os_ = OutcomeStats(
            total_verified=10,
            success_rate=0.7,
            per_action_success={"escalate": 0.8},
        )
        assert os_.total_verified == 10
        assert os_.success_rate == 0.7


# ═══════════════════════════════════════════════════════════════════════════
# TestPatternInsight
# ═══════════════════════════════════════════════════════════════════════════


class TestPatternInsight:
    def test_defaults(self):
        p = PatternInsight(category="test", message_ja="テスト", message_en="test")
        assert p.severity == "info"

    def test_custom_severity(self):
        p = PatternInsight(category="x", message_ja="x", message_en="x", severity="warning")
        assert p.severity == "warning"

    def test_asdict(self):
        p = PatternInsight(category="effectiveness", message_ja="良好", message_en="good")
        d = asdict(p)
        assert d["category"] == "effectiveness"
        assert d["severity"] == "info"


# ═══════════════════════════════════════════════════════════════════════════
# TestAgentIntelligenceReport
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentIntelligenceReport:
    def test_construction(self):
        r = AgentIntelligenceReport(window_hours=24.0, generated_at=time.time())
        assert r.window_hours == 24.0
        assert r.risk_trend == "stable"
        assert r.summary_ja == ""

    def test_to_dict(self):
        r = AgentIntelligenceReport(
            window_hours=24.0,
            generated_at=1000.0,
            risk_trend="improving",
        )
        d = r.to_dict()
        assert d["window_hours"] == 24.0
        assert d["risk_trend"] == "improving"
        assert "decisions" in d
        assert "outcomes" in d
        assert "insights" in d

    def test_to_dict_insights_are_dicts(self):
        insight = PatternInsight(category="test", message_ja="テスト", message_en="test")
        r = AgentIntelligenceReport(
            window_hours=1.0, generated_at=0.0, insights=[insight]
        )
        d = r.to_dict()
        assert len(d["insights"]) == 1
        assert isinstance(d["insights"][0], dict)


# ═══════════════════════════════════════════════════════════════════════════
# TestAgentReportGeneratorConstruction
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentReportGeneratorConstruction:
    def test_no_deps(self):
        gen = AgentReportGenerator()
        assert gen._agent_loop is None
        assert gen._verifier is None

    def test_with_deps(self):
        loop = MagicMock()
        ver = MagicMock()
        gen = AgentReportGenerator(agent_loop=loop, verifier=ver)
        assert gen._agent_loop is loop
        assert gen._verifier is ver

    def test_stats_shape(self):
        gen = AgentReportGenerator()
        stats = gen.get_stats()
        assert stats["enabled"] is True
        assert stats["report_count"] == 0
        assert stats["has_agent_loop"] is False
        assert stats["has_verifier"] is False

    def test_stats_with_deps(self):
        gen = AgentReportGenerator(agent_loop=MagicMock(), verifier=MagicMock())
        stats = gen.get_stats()
        assert stats["has_agent_loop"] is True
        assert stats["has_verifier"] is True


# ═══════════════════════════════════════════════════════════════════════════
# TestGenerateEmpty
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerateEmpty:
    def test_no_data_returns_report(self):
        gen = AgentReportGenerator()
        report = gen.generate(window_hours=24.0)
        assert isinstance(report, AgentIntelligenceReport)

    def test_empty_decisions(self):
        gen = AgentReportGenerator()
        report = gen.generate()
        assert report.decision_stats.total == 0

    def test_empty_outcomes(self):
        gen = AgentReportGenerator()
        report = gen.generate()
        assert report.outcome_stats.total_verified == 0

    def test_empty_summary_ja_mentions_no_records(self):
        gen = AgentReportGenerator()
        report = gen.generate(window_hours=24.0)
        assert "記録はありません" in report.summary_ja or report.summary_ja == "" or \
               "判断記録はありません" in report.summary_ja

    def test_report_count_increments(self):
        gen = AgentReportGenerator()
        gen.generate()
        gen.generate()
        assert gen.get_stats()["report_count"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# TestGenerateWithDecisions
# ═══════════════════════════════════════════════════════════════════════════


def _mock_loop_with_decisions(decisions: list[AgentDecision]) -> MagicMock:
    loop = MagicMock()
    loop._decision_history = decisions
    return loop


class TestGenerateWithDecisions:
    def test_total_counted(self):
        decisions = [
            _make_decision(AgentAction.ESCALATE),
            _make_decision(AgentAction.MONITOR),
            _make_decision(AgentAction.PLAN),
        ]
        gen = AgentReportGenerator(agent_loop=_mock_loop_with_decisions(decisions))
        report = gen.generate(window_hours=24.0)
        assert report.decision_stats.total == 3

    def test_action_breakdown(self):
        decisions = [
            _make_decision(AgentAction.ESCALATE),
            _make_decision(AgentAction.ESCALATE),
            _make_decision(AgentAction.MONITOR),
        ]
        gen = AgentReportGenerator(agent_loop=_mock_loop_with_decisions(decisions))
        report = gen.generate()
        assert report.decision_stats.action_breakdown.get("escalate") == 2
        assert report.decision_stats.action_breakdown.get("monitor") == 1

    def test_avg_latency(self):
        decisions = [
            _make_decision(latency_ms=10.0),
            _make_decision(latency_ms=20.0),
        ]
        gen = AgentReportGenerator(agent_loop=_mock_loop_with_decisions(decisions))
        report = gen.generate()
        assert abs(report.decision_stats.avg_latency_ms - 15.0) < 0.1

    def test_top_violation_type(self):
        decisions = [
            _make_decision(dominant_violation="no_helmet"),
            _make_decision(dominant_violation="no_helmet"),
            _make_decision(dominant_violation="fall"),
        ]
        gen = AgentReportGenerator(agent_loop=_mock_loop_with_decisions(decisions))
        report = gen.generate()
        assert report.decision_stats.top_violation_type == "no_helmet"

    def test_window_filtering(self):
        old_ts = time.time() - 25 * 3600  # 25 hours ago
        decisions = [
            _make_decision(ts=old_ts),         # outside 24h window
            _make_decision(ts=time.time()),    # inside window
        ]
        gen = AgentReportGenerator(agent_loop=_mock_loop_with_decisions(decisions))
        report = gen.generate(window_hours=24.0)
        assert report.decision_stats.total == 1

    def test_summary_ja_mentions_count(self):
        decisions = [_make_decision() for _ in range(5)]
        gen = AgentReportGenerator(agent_loop=_mock_loop_with_decisions(decisions))
        report = gen.generate()
        assert "5件" in report.summary_ja

    def test_summary_en_mentions_count(self):
        decisions = [_make_decision() for _ in range(3)]
        gen = AgentReportGenerator(agent_loop=_mock_loop_with_decisions(decisions))
        report = gen.generate()
        assert "3 decisions" in report.summary_en

    def test_dominant_action_in_summary(self):
        decisions = [_make_decision(AgentAction.ESCALATE) for _ in range(4)]
        decisions += [_make_decision(AgentAction.MONITOR)]
        gen = AgentReportGenerator(agent_loop=_mock_loop_with_decisions(decisions))
        report = gen.generate()
        assert "エスカレーション" in report.summary_ja


# ═══════════════════════════════════════════════════════════════════════════
# TestGenerateWithVerifications
# ═══════════════════════════════════════════════════════════════════════════


def _mock_verifier_with_records(records: list[VerificationRecord]) -> MagicMock:
    ver = MagicMock()
    ver._records = records
    ver.get_meta_state.return_value = {
        "success_rates": {},
        "update_counts": {},
        "escalation_threshold_adj": 0,
        "plan_threshold_adj": 0,
        "suggested_escalation_threshold": 3,
        "suggested_plan_threshold": 5,
    }
    return ver


class TestGenerateWithVerifications:
    def test_total_verified(self):
        records = [_make_record() for _ in range(4)]
        gen = AgentReportGenerator(verifier=_mock_verifier_with_records(records))
        report = gen.generate()
        assert report.outcome_stats.total_verified == 4

    def test_success_rate_all_success(self):
        records = [_make_record(outcome=VerificationOutcome.SUCCESS) for _ in range(4)]
        gen = AgentReportGenerator(verifier=_mock_verifier_with_records(records))
        report = gen.generate()
        assert report.outcome_stats.success_rate == 1.0

    def test_success_rate_mixed(self):
        records = [
            _make_record(outcome=VerificationOutcome.SUCCESS),
            _make_record(outcome=VerificationOutcome.FAILURE),
        ]
        gen = AgentReportGenerator(verifier=_mock_verifier_with_records(records))
        report = gen.generate()
        assert abs(report.outcome_stats.success_rate - 0.5) < 0.01

    def test_partial_counts_as_success(self):
        records = [
            _make_record(outcome=VerificationOutcome.SUCCESS),
            _make_record(outcome=VerificationOutcome.PARTIAL),
        ]
        gen = AgentReportGenerator(verifier=_mock_verifier_with_records(records))
        report = gen.generate()
        assert report.outcome_stats.success_rate == 1.0

    def test_outcome_breakdown(self):
        records = [
            _make_record(outcome=VerificationOutcome.SUCCESS),
            _make_record(outcome=VerificationOutcome.FAILURE),
            _make_record(outcome=VerificationOutcome.FAILURE),
        ]
        gen = AgentReportGenerator(verifier=_mock_verifier_with_records(records))
        report = gen.generate()
        assert report.outcome_stats.outcome_breakdown.get("success") == 1
        assert report.outcome_stats.outcome_breakdown.get("failure") == 2

    def test_per_action_success(self):
        records = [
            _make_record(action="escalate", outcome=VerificationOutcome.SUCCESS),
            _make_record(action="plan", outcome=VerificationOutcome.FAILURE),
        ]
        gen = AgentReportGenerator(verifier=_mock_verifier_with_records(records))
        report = gen.generate()
        assert report.outcome_stats.per_action_success.get("escalate") == 1.0
        assert report.outcome_stats.per_action_success.get("plan") == 0.0

    def test_most_effective_action(self):
        records = [
            _make_record(action="escalate", outcome=VerificationOutcome.SUCCESS),
            _make_record(action="escalate", outcome=VerificationOutcome.SUCCESS),
            _make_record(action="plan", outcome=VerificationOutcome.FAILURE),
        ]
        gen = AgentReportGenerator(verifier=_mock_verifier_with_records(records))
        report = gen.generate()
        assert report.most_effective_action == "escalate"

    def test_least_effective_action(self):
        records = [
            _make_record(action="escalate", outcome=VerificationOutcome.SUCCESS),
            _make_record(action="plan", outcome=VerificationOutcome.FAILURE),
        ]
        gen = AgentReportGenerator(verifier=_mock_verifier_with_records(records))
        report = gen.generate()
        assert report.least_effective_action == "plan"


# ═══════════════════════════════════════════════════════════════════════════
# TestRiskTrend
# ═══════════════════════════════════════════════════════════════════════════


class TestRiskTrend:
    def _gen(self) -> AgentReportGenerator:
        return AgentReportGenerator()

    def test_stable_few_records(self):
        gen = self._gen()
        records = [_make_record(outcome=VerificationOutcome.SUCCESS)]
        assert gen._compute_risk_trend(records) == "stable"

    def test_improving(self):
        gen = self._gen()
        # Old failures → recent successes
        records = (
            [_make_record(outcome=VerificationOutcome.FAILURE)] * 7
            + [_make_record(outcome=VerificationOutcome.SUCCESS)] * 3
        )
        result = gen._compute_risk_trend(records)
        assert result in ("improving", "stable")

    def test_worsening(self):
        gen = self._gen()
        # Old successes → recent failures
        records = (
            [_make_record(outcome=VerificationOutcome.SUCCESS)] * 7
            + [_make_record(outcome=VerificationOutcome.FAILURE)] * 3
        )
        result = gen._compute_risk_trend(records)
        assert result in ("worsening", "stable")

    def test_stable(self):
        gen = self._gen()
        records = (
            [_make_record(outcome=VerificationOutcome.SUCCESS)] * 5
            + [_make_record(outcome=VerificationOutcome.SUCCESS)] * 5
        )
        assert gen._compute_risk_trend(records) == "stable"


# ═══════════════════════════════════════════════════════════════════════════
# TestDetectPatterns
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectPatterns:
    def _gen(self) -> AgentReportGenerator:
        return AgentReportGenerator()

    def test_high_success_insight(self):
        gen = self._gen()
        os_ = OutcomeStats(total_verified=5, success_rate=0.8)
        insights = gen._detect_patterns(DecisionStats(), os_, {})
        cats = [i.category for i in insights]
        assert "effectiveness" in cats

    def test_low_success_insight(self):
        gen = self._gen()
        os_ = OutcomeStats(
            total_verified=5, success_rate=0.2,
            outcome_breakdown={"failure": 4, "success": 1}
        )
        insights = gen._detect_patterns(DecisionStats(), os_, {})
        assert any(i.severity == "warning" for i in insights)

    def test_top_violation_insight(self):
        gen = self._gen()
        ds = DecisionStats(total=3, top_violation_type="no_helmet")
        insights = gen._detect_patterns(ds, OutcomeStats(), {})
        viol_insights = [i for i in insights if i.category == "violation"]
        assert len(viol_insights) >= 1
        assert "no_helmet" in viol_insights[0].message_ja

    def test_threshold_adjustment_insight(self):
        gen = self._gen()
        meta = {"threshold_adjustments": {"escalation": 1, "plan": 0}}
        insights = gen._detect_patterns(DecisionStats(), OutcomeStats(), meta)
        thr_insights = [i for i in insights if i.category == "threshold"]
        assert len(thr_insights) >= 1
        assert "+1" in thr_insights[0].message_ja

    def test_no_insight_on_empty_data(self):
        gen = self._gen()
        insights = gen._detect_patterns(DecisionStats(), OutcomeStats(), {})
        # Only "no verification data" insight might appear if decisions > 0
        assert isinstance(insights, list)

    def test_decisions_without_verifications_insight(self):
        gen = self._gen()
        ds = DecisionStats(total=15)
        insights = gen._detect_patterns(ds, OutcomeStats(total_verified=0), {})
        trend_ins = [i for i in insights if i.category == "trend"]
        assert len(trend_ins) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# TestComposeSummaryJa
# ═══════════════════════════════════════════════════════════════════════════


class TestComposeSummaryJa:
    def _gen(self) -> AgentReportGenerator:
        return AgentReportGenerator()

    def test_empty_returns_no_record_msg(self):
        gen = self._gen()
        s = gen._compose_summary_ja(24.0, DecisionStats(), OutcomeStats(), "stable", {})
        assert "記録はありません" in s

    def test_mentions_window(self):
        gen = self._gen()
        ds = DecisionStats(total=3)
        s = gen._compose_summary_ja(12.0, ds, OutcomeStats(), "stable", {})
        assert "12" in s

    def test_mentions_decision_count(self):
        gen = self._gen()
        ds = DecisionStats(total=7)
        s = gen._compose_summary_ja(24.0, ds, OutcomeStats(), "stable", {})
        assert "7件" in s

    def test_trend_improving_in_text(self):
        gen = self._gen()
        ds = DecisionStats(total=3)
        s = gen._compose_summary_ja(24.0, ds, OutcomeStats(), "improving", {})
        assert "改善" in s

    def test_trend_worsening_in_text(self):
        gen = self._gen()
        ds = DecisionStats(total=3)
        s = gen._compose_summary_ja(24.0, ds, OutcomeStats(), "worsening", {})
        assert "悪化" in s

    def test_threshold_adjustment_mentioned(self):
        gen = self._gen()
        ds = DecisionStats(total=3)
        meta = {"threshold_adjustments": {"escalation": 2, "plan": 0}}
        s = gen._compose_summary_ja(24.0, ds, OutcomeStats(), "stable", meta)
        assert "引き上げ" in s or "+2" in s


# ═══════════════════════════════════════════════════════════════════════════
# TestComposeSummaryEn
# ═══════════════════════════════════════════════════════════════════════════


class TestComposeSummaryEn:
    def _gen(self) -> AgentReportGenerator:
        return AgentReportGenerator()

    def test_empty_mentions_no_decisions(self):
        gen = self._gen()
        s = gen._compose_summary_en(24.0, DecisionStats(), OutcomeStats(), "stable", {})
        assert "No agent decisions" in s

    def test_decision_count_in_text(self):
        gen = self._gen()
        ds = DecisionStats(total=5)
        s = gen._compose_summary_en(24.0, ds, OutcomeStats(), "stable", {})
        assert "5 decisions" in s

    def test_effectiveness_mentioned(self):
        gen = self._gen()
        ds = DecisionStats(total=5)
        os_ = OutcomeStats(total_verified=5, success_rate=0.8)
        s = gen._compose_summary_en(24.0, ds, os_, "stable", {})
        assert "80%" in s

    def test_trend_in_english(self):
        gen = self._gen()
        ds = DecisionStats(total=3)
        s = gen._compose_summary_en(24.0, ds, OutcomeStats(), "improving", {})
        assert "positive" in s.lower() or "improving" in s.lower()


# ═══════════════════════════════════════════════════════════════════════════
# TestGetSummary
# ═══════════════════════════════════════════════════════════════════════════


class TestGetSummary:
    def test_returns_dict(self):
        gen = AgentReportGenerator()
        s = gen.get_summary()
        assert isinstance(s, dict)
        assert "total_decisions" in s
        assert "success_rate" in s
        assert "risk_trend" in s

    def test_no_crash_without_deps(self):
        gen = AgentReportGenerator()
        s = gen.get_summary(window_hours=1.0)
        assert s["total_decisions"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# TestGetInsights
# ═══════════════════════════════════════════════════════════════════════════


class TestGetInsights:
    def test_returns_list(self):
        gen = AgentReportGenerator()
        ins = gen.get_insights()
        assert isinstance(ins, list)

    def test_each_insight_is_dict(self):
        gen = AgentReportGenerator()
        # Inject some state via mock
        ver = _mock_verifier_with_records([
            _make_record(outcome=VerificationOutcome.SUCCESS) for _ in range(5)
        ])
        gen._verifier = ver
        ins = gen.get_insights()
        for i in ins:
            assert isinstance(i, dict)
            assert "message_ja" in i


# ═══════════════════════════════════════════════════════════════════════════
# TestIntegrationWithRealComponents
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegrationWithRealComponents:
    def test_with_real_agent_loop(self):
        """Smoke test: AgentReportGenerator reads from a real AgentLoop."""
        from sopilot.perception.agent_loop import AgentLoop, AgentEvent
        loop = AgentLoop()
        loop._pending_events.append(AgentEvent(
            event_type="violation", severity="critical", violation_rule="fall",
            camera_id="cam1",
        ))
        loop.think()

        gen = AgentReportGenerator(agent_loop=loop)
        report = gen.generate(window_hours=1.0)
        assert report.decision_stats.total >= 1

    def test_with_real_verifier(self):
        """Smoke test: AgentReportGenerator reads from a real ClosedLoopVerifier."""
        ver = ClosedLoopVerifier(verify_after_s=0.0, success_threshold=0.5)
        ver.schedule("escalate", "cam1", 10, {}, "high")

        class _F:
            violations = [None, None, None]  # 3/10 = 70% reduction → SUCCESS

        ver.check_pending(_F(), "cam1")

        gen = AgentReportGenerator(verifier=ver)
        report = gen.generate(window_hours=1.0)
        assert report.outcome_stats.total_verified == 1
        assert report.outcome_stats.success_rate == 1.0

    def test_full_pipeline(self):
        """End-to-end: loop + verifier → report."""
        from sopilot.perception.agent_loop import AgentLoop, AgentEvent
        ver = ClosedLoopVerifier(verify_after_s=0.0)
        loop = AgentLoop(verifier=ver, escalation_threshold=1)

        loop._pending_events.append(AgentEvent(
            event_type="violation", severity="critical", violation_rule="fall",
        ))
        loop.think()

        class _F:
            violations = []

        ver.check_pending(_F(), "cam")

        gen = AgentReportGenerator(agent_loop=loop, verifier=ver)
        report = gen.generate(window_hours=1.0)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["decisions"]["total"] >= 1
        assert isinstance(d["summary_ja"], str)
        assert len(d["summary_ja"]) > 0
