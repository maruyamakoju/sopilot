"""Agent Intelligence Report — Phase 61.

Analyses the AgentLoop decision history and ClosedLoopVerifier records to
produce a structured Japanese/English intelligence report explaining what
the autonomous agent did, how effective it was, and what it recommends.

This is the "self-explanation" layer of the autonomous agent:

    AgentLoop._decision_history   ─┐
    ClosedLoopVerifier._records   ─┤─► AgentReportGenerator.generate()
    ClosedLoopVerifier._meta       ─┘         └─► AgentIntelligenceReport

Design goals:
  - Pure computation: no ML inference, sub-millisecond generation
  - Rule-based Japanese narrative (not LLaVA) for determinism
  - Interpretability: every number is traceable to a specific decision
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


# ── Supporting data structures ────────────────────────────────────────────


@dataclass
class DecisionStats:
    """Breakdown of agent decisions in the analysis window."""
    total: int = 0
    action_breakdown: dict[str, int] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    top_violation_type: str = ""
    camera_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class OutcomeStats:
    """Verification outcome analysis."""
    total_verified: int = 0
    outcome_breakdown: dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0        # SUCCESS + PARTIAL treated as success
    avg_reduction_ratio: float = 0.0
    per_action_success: dict[str, float] = field(default_factory=dict)  # action → success_rate


@dataclass
class PatternInsight:
    """A single detected pattern or anomaly."""
    category: str            # "effectiveness", "trend", "threshold", "violation"
    message_ja: str
    message_en: str
    severity: str = "info"   # "info", "warning", "critical"


@dataclass
class AgentIntelligenceReport:
    """Full intelligence report for a time window."""
    window_hours: float
    generated_at: float

    # Decision breakdown
    decision_stats: DecisionStats = field(default_factory=DecisionStats)

    # Verification outcomes
    outcome_stats: OutcomeStats = field(default_factory=OutcomeStats)

    # Meta-learning state
    threshold_adjustments: dict[str, int] = field(default_factory=dict)
    suggested_thresholds: dict[str, int] = field(default_factory=dict)

    # Risk trend
    risk_trend: str = "stable"       # "improving", "stable", "worsening"

    # Most / least effective actions (among those verified)
    most_effective_action: str = ""
    least_effective_action: str = ""

    # Narrative
    summary_ja: str = ""
    summary_en: str = ""
    insights: list[PatternInsight] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_hours": self.window_hours,
            "generated_at": self.generated_at,
            "decisions": asdict(self.decision_stats),
            "outcomes": asdict(self.outcome_stats),
            "threshold_adjustments": self.threshold_adjustments,
            "suggested_thresholds": self.suggested_thresholds,
            "risk_trend": self.risk_trend,
            "most_effective_action": self.most_effective_action,
            "least_effective_action": self.least_effective_action,
            "summary_ja": self.summary_ja,
            "summary_en": self.summary_en,
            "insights": [asdict(i) for i in self.insights],
        }


# ── Narrative templates ───────────────────────────────────────────────────

_ACTION_NAME_JA = {
    "monitor":  "監視継続",
    "escalate": "エスカレーション",
    "plan":     "安全プラン作成",
    "adapt":    "オンライン適応",
    "query":    "RAG照会",
    "respond":  "自律対応",
}
_ACTION_NAME_EN = {
    "monitor":  "Monitor",
    "escalate": "Escalate",
    "plan":     "Plan",
    "adapt":    "Adapt",
    "query":    "Query",
    "respond":  "Respond",
}
_TREND_JA = {
    "improving": "リスクレベルは改善傾向にあります",
    "stable":    "リスクレベルは安定しています",
    "worsening": "リスクレベルは悪化傾向にあります",
}
_TREND_EN = {
    "improving": "Risk level is trending positively",
    "stable":    "Risk level is stable",
    "worsening": "Risk level is trending negatively",
}


# ── AgentReportGenerator ─────────────────────────────────────────────────


class AgentReportGenerator:
    """Generates AgentIntelligenceReport from live agent state.

    Receives references to AgentLoop and ClosedLoopVerifier via duck-typing
    (``Any | None``) to avoid circular imports.
    """

    def __init__(
        self,
        agent_loop: Any | None = None,
        verifier: Any | None = None,
    ) -> None:
        self._agent_loop = agent_loop
        self._verifier = verifier
        self._last_report: AgentIntelligenceReport | None = None
        self._report_count: int = 0
        self._start_time: float = time.time()

    # ── Public API ───────────────────────────────────────────────────────

    def generate(self, window_hours: float = 24.0) -> AgentIntelligenceReport:
        """Generate a full intelligence report for the past *window_hours*."""
        cutoff = time.time() - window_hours * 3600.0

        # Collect raw data
        decisions = self._collect_decisions(cutoff)
        records = self._collect_records(cutoff)
        meta = self._collect_meta()

        # Analyse
        decision_stats = self._analyze_decisions(decisions)
        outcome_stats = self._analyze_outcomes(records)
        risk_trend = self._compute_risk_trend(records)
        most_eff, least_eff = self._find_effective_actions(outcome_stats)

        # Patterns → insights
        insights = self._detect_patterns(decision_stats, outcome_stats, meta)

        # Narratives
        summary_ja = self._compose_summary_ja(
            window_hours, decision_stats, outcome_stats, risk_trend, meta
        )
        summary_en = self._compose_summary_en(
            window_hours, decision_stats, outcome_stats, risk_trend, meta
        )

        report = AgentIntelligenceReport(
            window_hours=window_hours,
            generated_at=time.time(),
            decision_stats=decision_stats,
            outcome_stats=outcome_stats,
            threshold_adjustments=meta.get("threshold_adjustments", {}),
            suggested_thresholds=meta.get("suggested_thresholds", {}),
            risk_trend=risk_trend,
            most_effective_action=most_eff,
            least_effective_action=least_eff,
            summary_ja=summary_ja,
            summary_en=summary_en,
            insights=insights,
        )
        self._last_report = report
        self._report_count += 1
        return report

    def get_summary(self, window_hours: float = 24.0) -> dict[str, Any]:
        """Lightweight summary without full pattern detection."""
        cutoff = time.time() - window_hours * 3600.0
        decisions = self._collect_decisions(cutoff)
        records = self._collect_records(cutoff)
        meta = self._collect_meta()
        ds = self._analyze_decisions(decisions)
        os_ = self._analyze_outcomes(records)
        return {
            "window_hours": window_hours,
            "total_decisions": ds.total,
            "success_rate": round(os_.success_rate, 3),
            "top_violation_type": ds.top_violation_type,
            "risk_trend": self._compute_risk_trend(records),
            "threshold_adjustments": meta.get("threshold_adjustments", {}),
        }

    def get_insights(self, window_hours: float = 24.0) -> list[dict[str, Any]]:
        """Return only the insight bullets."""
        cutoff = time.time() - window_hours * 3600.0
        ds = self._analyze_decisions(self._collect_decisions(cutoff))
        os_ = self._analyze_outcomes(self._collect_records(cutoff))
        meta = self._collect_meta()
        return [asdict(i) for i in self._detect_patterns(ds, os_, meta)]

    def get_stats(self) -> dict[str, Any]:
        """Return generator meta-stats."""
        return {
            "enabled": True,
            "report_count": self._report_count,
            "has_agent_loop": self._agent_loop is not None,
            "has_verifier": self._verifier is not None,
            "uptime_s": round(time.time() - self._start_time, 1),
            "last_report_at": (
                self._last_report.generated_at if self._last_report else None
            ),
        }

    # ── Data collection ──────────────────────────────────────────────────

    def _collect_decisions(self, cutoff: float) -> list[Any]:
        if self._agent_loop is None:
            return []
        try:
            history = list(getattr(self._agent_loop, "_decision_history", []))
            return [d for d in history if getattr(d, "timestamp", 0.0) >= cutoff]
        except Exception:
            return []

    def _collect_records(self, cutoff: float) -> list[Any]:
        if self._verifier is None:
            return []
        try:
            records = list(getattr(self._verifier, "_records", []))
            return [r for r in records if getattr(r, "verified_at", 0.0) >= cutoff]
        except Exception:
            return []

    def _collect_meta(self) -> dict[str, Any]:
        if self._verifier is None:
            return {}
        try:
            meta = self._verifier.get_meta_state()
            return {
                "threshold_adjustments": {
                    "escalation": meta.get("escalation_threshold_adj", 0),
                    "plan": meta.get("plan_threshold_adj", 0),
                },
                "suggested_thresholds": {
                    "escalation": meta.get("suggested_escalation_threshold", 3),
                    "plan": meta.get("suggested_plan_threshold", 5),
                },
                "success_rates": meta.get("success_rates", {}),
                "update_counts": meta.get("update_counts", {}),
            }
        except Exception:
            return {}

    # ── Analysis ─────────────────────────────────────────────────────────

    def _analyze_decisions(self, decisions: list[Any]) -> DecisionStats:
        if not decisions:
            return DecisionStats()

        action_breakdown: dict[str, int] = {}
        violation_types: dict[str, int] = {}
        camera_counts: dict[str, int] = {}
        latencies: list[float] = []

        for d in decisions:
            action = getattr(d, "action", None)
            action_str = action.value if hasattr(action, "value") else str(action)
            action_breakdown[action_str] = action_breakdown.get(action_str, 0) + 1

            lat = getattr(d, "latency_ms", 0.0)
            if lat:
                latencies.append(lat)

            # Extract violation type from situation snapshot
            situation = getattr(d, "situation", {})
            vtype = situation.get("dominant_violation_type", "")
            if vtype:
                violation_types[vtype] = violation_types.get(vtype, 0) + 1

            # Extract primary camera from params
            params = getattr(d, "params", {})
            cam = params.get("camera_id", "")
            if cam:
                camera_counts[cam] = camera_counts.get(cam, 0) + 1

        top_violation = (
            max(violation_types, key=violation_types.get)  # type: ignore[arg-type]
            if violation_types else ""
        )
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return DecisionStats(
            total=len(decisions),
            action_breakdown=action_breakdown,
            avg_latency_ms=round(avg_latency, 2),
            top_violation_type=top_violation,
            camera_counts=camera_counts,
        )

    def _analyze_outcomes(self, records: list[Any]) -> OutcomeStats:
        if not records:
            return OutcomeStats()

        outcome_breakdown: dict[str, int] = {}
        per_action: dict[str, list[bool]] = {}
        reductions: list[float] = []

        for r in records:
            outcome = getattr(r, "outcome", None)
            outcome_str = outcome.value if hasattr(outcome, "value") else str(outcome)
            outcome_breakdown[outcome_str] = outcome_breakdown.get(outcome_str, 0) + 1

            action = getattr(r, "decision_action", "")
            is_success = outcome_str in ("success", "partial")
            per_action.setdefault(action, []).append(is_success)

            red = getattr(r, "reduction_ratio", 0.0)
            if isinstance(red, (int, float)):
                reductions.append(float(red))

        total = len(records)
        successes = (
            outcome_breakdown.get("success", 0)
            + outcome_breakdown.get("partial", 0)
        )
        success_rate = successes / total if total else 0.0

        per_action_success = {
            action: sum(results) / len(results)
            for action, results in per_action.items()
            if results
        }
        avg_reduction = sum(reductions) / len(reductions) if reductions else 0.0

        return OutcomeStats(
            total_verified=total,
            outcome_breakdown=outcome_breakdown,
            success_rate=round(success_rate, 3),
            avg_reduction_ratio=round(avg_reduction, 3),
            per_action_success={k: round(v, 3) for k, v in per_action_success.items()},
        )

    def _compute_risk_trend(self, records: list[Any]) -> str:
        """Compare success rate of newest 30% vs overall."""
        if len(records) < 4:
            return "stable"

        split = max(1, len(records) * 7 // 10)  # 70% = older, 30% = recent
        older = records[:split]
        recent = records[split:]

        def _sr(recs: list[Any]) -> float:
            if not recs:
                return 0.0
            s = sum(
                1 for r in recs
                if getattr(r, "outcome", None) is not None
                and getattr(r.outcome, "value", str(r.outcome)) in ("success", "partial")
            )
            return s / len(recs)

        older_sr = _sr(older)
        recent_sr = _sr(recent)

        if recent_sr >= older_sr + 0.1:
            return "improving"
        elif recent_sr <= older_sr - 0.1:
            return "worsening"
        return "stable"

    def _find_effective_actions(
        self, outcome_stats: OutcomeStats
    ) -> tuple[str, str]:
        """Return (most_effective, least_effective) action names."""
        pas = outcome_stats.per_action_success
        if not pas:
            return "", ""
        most = max(pas, key=pas.get)  # type: ignore[arg-type]
        least = min(pas, key=pas.get)  # type: ignore[arg-type]
        # Only return "least" if it differs from "most"
        return most, (least if least != most else "")

    # ── Pattern detection ────────────────────────────────────────────────

    def _detect_patterns(
        self,
        ds: DecisionStats,
        os_: OutcomeStats,
        meta: dict[str, Any],
    ) -> list[PatternInsight]:
        insights: list[PatternInsight] = []

        # 1. Overall effectiveness
        if os_.total_verified > 0:
            if os_.success_rate >= 0.7:
                insights.append(PatternInsight(
                    category="effectiveness",
                    message_ja=f"施策全体の有効率は{round(os_.success_rate * 100)}%で良好です",
                    message_en=f"Overall action effectiveness is good ({round(os_.success_rate * 100)}%)",
                    severity="info",
                ))
            elif os_.success_rate < 0.3:
                insights.append(PatternInsight(
                    category="effectiveness",
                    message_ja=f"施策有効率が{round(os_.success_rate * 100)}%と低水準です — ルール見直しを推奨",
                    message_en=f"Low action effectiveness ({round(os_.success_rate * 100)}%) — rule review recommended",
                    severity="warning",
                ))

        # 2. Per-action success rate warnings
        for action, sr in os_.per_action_success.items():
            name_ja = _ACTION_NAME_JA.get(action, action)
            name_en = _ACTION_NAME_EN.get(action, action)
            if sr < 0.3 and os_.outcome_breakdown.get("failure", 0) >= 2:
                insights.append(PatternInsight(
                    category="effectiveness",
                    message_ja=f"{name_ja}施策の成功率が{round(sr * 100)}%と低いです",
                    message_en=f"{name_en} action success rate is low ({round(sr * 100)}%)",
                    severity="warning",
                ))
            elif sr >= 0.8:
                insights.append(PatternInsight(
                    category="effectiveness",
                    message_ja=f"{name_ja}施策は非常に効果的です（成功率{round(sr * 100)}%）",
                    message_en=f"{name_en} action is highly effective ({round(sr * 100)}% success)",
                    severity="info",
                ))

        # 3. Top violation type
        if ds.top_violation_type:
            insights.append(PatternInsight(
                category="violation",
                message_ja=f"最頻出違反タイプ: {ds.top_violation_type}",
                message_en=f"Most frequent violation: {ds.top_violation_type}",
                severity="info",
            ))

        # 4. Threshold adjustments
        adj = meta.get("threshold_adjustments", {})
        esc_adj = adj.get("escalation", 0)
        plan_adj = adj.get("plan", 0)
        if esc_adj > 0:
            insights.append(PatternInsight(
                category="threshold",
                message_ja=f"エスカレーション閾値を自動引き上げました (+{esc_adj}) — 過剰反応を抑制",
                message_en=f"Escalation threshold auto-raised (+{esc_adj}) to reduce over-triggering",
                severity="info",
            ))
        elif esc_adj < 0:
            insights.append(PatternInsight(
                category="threshold",
                message_ja=f"エスカレーション閾値を自動引き下げました ({esc_adj}) — 検出感度向上",
                message_en=f"Escalation threshold auto-lowered ({esc_adj}) for higher sensitivity",
                severity="info",
            ))
        if plan_adj > 0:
            insights.append(PatternInsight(
                category="threshold",
                message_ja=f"プラン閾値を自動引き上げました (+{plan_adj})",
                message_en=f"Plan threshold auto-raised (+{plan_adj})",
                severity="info",
            ))

        # 5. High decision count without verifications
        if ds.total >= 10 and os_.total_verified == 0:
            insights.append(PatternInsight(
                category="trend",
                message_ja=f"{ds.total}件の判断を実施しましたが、検証データがまだありません",
                message_en=f"{ds.total} decisions made but no verification data yet",
                severity="info",
            ))

        return insights

    # ── Narrative composition ────────────────────────────────────────────

    def _compose_summary_ja(
        self,
        window_hours: float,
        ds: DecisionStats,
        os_: OutcomeStats,
        trend: str,
        meta: dict[str, Any],
    ) -> str:
        parts: list[str] = []

        # Opening
        window_str = f"過去{int(window_hours)}時間" if window_hours >= 1 else "直近"
        if ds.total == 0:
            return f"{window_str}に自律エージェントの判断記録はありません。"

        parts.append(f"{window_str}で{ds.total}件の判断を実施しました。")

        # Dominant action
        if ds.action_breakdown:
            dom_action = max(ds.action_breakdown, key=ds.action_breakdown.get)  # type: ignore[arg-type]
            dom_count = ds.action_breakdown[dom_action]
            dom_name = _ACTION_NAME_JA.get(dom_action, dom_action)
            parts.append(f"最多判断は「{dom_name}」({dom_count}件)でした。")

        # Effectiveness
        if os_.total_verified > 0:
            sr_pct = round(os_.success_rate * 100)
            if os_.success_rate >= 0.7:
                parts.append(f"施策有効率は{sr_pct}%と良好で、")
            elif os_.success_rate >= 0.4:
                parts.append(f"施策有効率は{sr_pct}%で概ね効果的であり、")
            else:
                parts.append(f"施策有効率は{sr_pct}%と課題があり、")

        # Trend
        parts.append(_TREND_JA.get(trend, "状況は安定しています") + "。")

        # Threshold adjustment
        adj = meta.get("threshold_adjustments", {})
        esc_adj = adj.get("escalation", 0)
        if esc_adj != 0:
            direction = "引き上げ" if esc_adj > 0 else "引き下げ"
            parts.append(
                f"メタ学習によりエスカレーション閾値を自動{direction}({esc_adj:+d})しました。"
            )

        # Violation type
        if ds.top_violation_type:
            parts.append(f"最頻出違反: {ds.top_violation_type}。")

        return " ".join(parts)

    def _compose_summary_en(
        self,
        window_hours: float,
        ds: DecisionStats,
        os_: OutcomeStats,
        trend: str,
        meta: dict[str, Any],
    ) -> str:
        if ds.total == 0:
            window_str = f"past {int(window_hours)}h" if window_hours >= 1 else "recent window"
            return f"No agent decisions recorded in the {window_str}."

        parts: list[str] = []
        window_str = f"past {int(window_hours)}h"
        parts.append(f"Agent made {ds.total} decisions in the {window_str}.")

        if ds.action_breakdown:
            dom_action = max(ds.action_breakdown, key=ds.action_breakdown.get)  # type: ignore[arg-type]
            dom_count = ds.action_breakdown[dom_action]
            dom_name = _ACTION_NAME_EN.get(dom_action, dom_action)
            parts.append(f"Most common action: {dom_name} ({dom_count}x).")

        if os_.total_verified > 0:
            sr_pct = round(os_.success_rate * 100)
            parts.append(f"Action effectiveness: {sr_pct}%.")

        parts.append(_TREND_EN.get(trend, "Situation is stable") + ".")

        adj = meta.get("threshold_adjustments", {})
        esc_adj = adj.get("escalation", 0)
        if esc_adj != 0:
            direction = "raised" if esc_adj > 0 else "lowered"
            parts.append(f"Meta-learner auto-{direction} escalation threshold ({esc_adj:+d}).")

        if ds.top_violation_type:
            parts.append(f"Top violation: {ds.top_violation_type}.")

        return " ".join(parts)
