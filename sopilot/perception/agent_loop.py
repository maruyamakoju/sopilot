"""Agentic Closed-Loop (OODA) orchestrator — Phase 59.

Connects SafetyPlanner, CausalInference, IncidentReporter, Video RAG,
LLaVA, OnlineLearner, EarlyWarningResponder, and AnomalyForecaster into
a single Observe-Orient-Decide-Act cycle.

The agent is NOT a per-frame stage (LLaVA is too expensive at 30 fps).
Instead:
  - ``observe()`` is called every frame (<0.1 ms, deque append)
  - ``think()`` runs every ~5 s or on critical triggers

Design:
    Per-frame (30 fps):
        Stage 1-6u → normal pipeline
        Stage 6v   → agent_loop.observe(frame_result)

    Every ~5 s or critical trigger:
        agent_loop.think()
            ├─ ORIENT:  aggregate events → CausalInference → optional RAG
            ├─ DECIDE:  rule-based priority cascade → AgentAction
            ├─ ACT:     SafetyPlanner / IncidentReporter / OnlineLearner
            └─ LEARN:   record AgentDecision → decision_history
"""

from __future__ import annotations

import enum
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Enumerations ──────────────────────────────────────────────────────────


class AgentAction(str, enum.Enum):
    """Actions the agent can choose after the DECIDE phase."""
    MONITOR = "monitor"
    ESCALATE = "escalate"
    PLAN = "plan"
    ADAPT = "adapt"
    QUERY = "query"
    RESPOND = "respond"


class OODAPhase(str, enum.Enum):
    """Phases of the OODA loop."""
    OBSERVE = "observe"
    ORIENT = "orient"
    DECIDE = "decide"
    ACT = "act"
    LEARN = "learn"


class AgentStatus(str, enum.Enum):
    """Runtime status of the agent loop."""
    IDLE = "idle"
    THINKING = "thinking"
    DISABLED = "disabled"


# ── Data structures ───────────────────────────────────────────────────────


@dataclass
class AgentEvent:
    """A buffered event from Stage 6v observe()."""
    event_type: str
    camera_id: str = "cam"
    severity: str = "info"          # info / warning / critical
    violation_rule: str = ""
    timestamp: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SituationSummary:
    """Output of the ORIENT phase — aggregated situational awareness."""
    camera_violations: dict[str, int] = field(default_factory=dict)
    severity_counts: dict[str, int] = field(default_factory=dict)
    dominant_violation_type: str = ""
    risk_level: str = "low"         # low / medium / high / critical
    trend: str = "stable"           # stable / rising / falling
    causal_edges: list[dict[str, Any]] = field(default_factory=list)
    rag_context: str = ""
    llava_summary: str = ""
    recalled_episodes: list[dict[str, Any]] = field(default_factory=list)  # Phase 62


@dataclass
class AgentDecision:
    """Record of a single OODA decision."""
    action: AgentAction
    reasoning_ja: str = ""
    reasoning_en: str = ""
    outcome: str = ""
    latency_ms: float = 0.0
    timestamp: float = 0.0
    situation: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        return d


# ── AgentLoop ─────────────────────────────────────────────────────────────


_MAX_EVENTS = 500
_MAX_DECISIONS = 200


class AgentLoop:
    """OODA closed-loop orchestrator.

    All sub-system references are typed ``Any | None`` to avoid circular
    imports — they are duck-typed at runtime.
    """

    def __init__(
        self,
        *,
        think_interval_s: float = 5.0,
        critical_trigger: bool = True,
        use_llava_orient: bool = False,
        use_rag_orient: bool = True,
        escalation_threshold: int = 3,
        plan_threshold: int = 5,
        # Sub-system references (all optional)
        safety_planner: Any | None = None,
        causal_inference: Any | None = None,
        incident_reporter: Any | None = None,
        rag_engine: Any | None = None,
        llava_client: Any | None = None,
        online_learner: Any | None = None,
        early_warning_responder: Any | None = None,
        anomaly_forecaster: Any | None = None,
        verifier: Any | None = None,
        episode_store: Any | None = None,
    ) -> None:
        self._think_interval_s = think_interval_s
        self._critical_trigger = critical_trigger
        self._use_llava_orient = use_llava_orient
        self._use_rag_orient = use_rag_orient
        self._escalation_threshold = escalation_threshold
        self._plan_threshold = plan_threshold

        # Sub-systems
        self._safety_planner = safety_planner
        self._causal_inference = causal_inference
        self._incident_reporter = incident_reporter
        self._rag_engine = rag_engine
        self._llava_client = llava_client
        self._online_learner = online_learner
        self._early_warning_responder = early_warning_responder
        self._anomaly_forecaster = anomaly_forecaster
        self._verifier = verifier          # ClosedLoopVerifier | None (Phase 60)
        self._episode_store = episode_store  # AgentEpisodeStore | None (Phase 62)

        # State
        self._pending_events: deque[AgentEvent] = deque(maxlen=_MAX_EVENTS)
        self._decision_history: deque[AgentDecision] = deque(maxlen=_MAX_DECISIONS)
        self._action_counts: dict[str, int] = {a.value: 0 for a in AgentAction}
        self._think_count: int = 0
        self._last_think_time: float = 0.0
        self._status: AgentStatus = AgentStatus.IDLE
        self._start_time: float = time.time()

    # ── OBSERVE (Stage 6v, per-frame) ────────────────────────────────────

    def observe(self, frame_result: Any, *, camera_id: str = "cam") -> bool:
        """Buffer events from a frame result.  Returns True if critical trigger fires."""
        has_critical = False
        try:
            violations = getattr(frame_result, "violations", None) or []
            for v in violations:
                sev = getattr(v, "severity", None)
                sev_str = sev.value if hasattr(sev, "value") else str(sev) if sev else "info"
                rule = getattr(v, "rule", "") or getattr(v, "description_ja", "") or ""
                ev = AgentEvent(
                    event_type="violation",
                    camera_id=camera_id,
                    severity=sev_str,
                    violation_rule=rule,
                    timestamp=time.time(),
                )
                self._pending_events.append(ev)
                if sev_str == "critical" and self._critical_trigger:
                    has_critical = True

            # Also observe world-state events
            ws = getattr(frame_result, "world_state", None)
            if ws is not None:
                ws_events = getattr(ws, "events", None) or []
                for e in ws_events:
                    etype = getattr(e, "event_type", None)
                    etype_str = etype.value if hasattr(etype, "value") else str(etype) if etype else "event"
                    ev = AgentEvent(
                        event_type=etype_str,
                        camera_id=camera_id,
                        severity="info",
                        timestamp=time.time(),
                    )
                    self._pending_events.append(ev)
        except Exception:
            logger.debug("AgentLoop.observe() error", exc_info=True)

        return has_critical

    # ── Timing ───────────────────────────────────────────────────────────

    def should_think(self) -> bool:
        """True if enough time has elapsed since the last think() call."""
        if not self._pending_events:
            return False
        return (time.time() - self._last_think_time) >= self._think_interval_s

    # ── THINK (full OODA cycle) ──────────────────────────────────────────

    def think(self) -> AgentDecision | None:
        """Execute one full OODA cycle.  Returns the decision or None."""
        if not self._pending_events:
            return None

        t0 = time.time()
        self._status = AgentStatus.THINKING
        try:
            # Snapshot and clear pending events
            events = list(self._pending_events)
            self._pending_events.clear()

            # ORIENT
            situation = self._orient(events)

            # DECIDE
            action, reasoning_ja, reasoning_en, params = self._decide(situation)

            # ACT
            outcome = self._act(action, params, situation)

            # LEARN — build decision record first
            elapsed_ms = (time.time() - t0) * 1000.0
            decision = AgentDecision(
                action=action,
                reasoning_ja=reasoning_ja,
                reasoning_en=reasoning_en,
                outcome=outcome,
                latency_ms=round(elapsed_ms, 2),
                timestamp=time.time(),
                situation={
                    "risk_level": situation.risk_level,
                    "dominant_violation_type": situation.dominant_violation_type,
                    "severity_counts": situation.severity_counts,
                },
                params=dict(params),
            )
            self._decision_history.append(decision)
            self._action_counts[action.value] = self._action_counts.get(action.value, 0) + 1
            self._think_count += 1
            self._last_think_time = time.time()

            # Phase 62: STORE EPISODE — get episode_id before scheduling verification
            _ep_id: str | None = None
            if self._episode_store is not None:
                try:
                    _vcam_ep = (
                        list(situation.camera_violations.keys())[0]
                        if situation.camera_violations
                        else "cam"
                    )
                    _ep_id = self._episode_store.store_decision(
                        risk_level=situation.risk_level,
                        severity_counts=dict(situation.severity_counts),
                        camera_violations=dict(situation.camera_violations),
                        dominant_violation_type=situation.dominant_violation_type,
                        decision_action=action.value,
                        reasoning_ja=reasoning_ja,
                        reasoning_en=reasoning_en,
                        camera_id=_vcam_ep,
                    )
                    decision.params["episode_id"] = _ep_id
                except Exception:
                    logger.debug("Episode store failed", exc_info=True)

            # VERIFY — schedule closed-loop verification (Phase 60), pass episode_id (Phase 62)
            if self._verifier is not None and action != AgentAction.MONITOR:
                try:
                    _vcam = (
                        list(situation.camera_violations.keys())[0]
                        if situation.camera_violations
                        else "cam"
                    )
                    self._verifier.schedule(
                        decision_action=action.value,
                        camera_id=_vcam,
                        baseline_violations=sum(situation.camera_violations.values()),
                        baseline_severity=dict(situation.severity_counts),
                        baseline_risk_level=situation.risk_level,
                        decision_reasoning_ja=reasoning_ja,
                        episode_id=_ep_id,
                    )
                except Exception:
                    logger.debug("Failed to schedule verification", exc_info=True)

            logger.debug(
                "AgentLoop.think(): action=%s latency=%.1fms outcome=%s",
                action.value, elapsed_ms, outcome,
            )
            return decision

        except Exception:
            logger.debug("AgentLoop.think() error", exc_info=True)
            return None
        finally:
            self._status = AgentStatus.IDLE

    # ── ORIENT ───────────────────────────────────────────────────────────

    def _orient(self, events: list[AgentEvent]) -> SituationSummary:
        """Aggregate events into a situational summary."""
        camera_violations: dict[str, int] = {}
        severity_counts: dict[str, int] = {"info": 0, "warning": 0, "critical": 0}
        violation_types: dict[str, int] = {}

        for ev in events:
            if ev.event_type == "violation":
                camera_violations[ev.camera_id] = camera_violations.get(ev.camera_id, 0) + 1
                sev = ev.severity if ev.severity in severity_counts else "info"
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
                if ev.violation_rule:
                    violation_types[ev.violation_rule] = violation_types.get(ev.violation_rule, 0) + 1

        # Dominant violation type
        dominant = ""
        if violation_types:
            dominant = max(violation_types, key=violation_types.get)  # type: ignore[arg-type]

        # Risk level
        total_critical = severity_counts.get("critical", 0)
        total_warning = severity_counts.get("warning", 0)
        total_violations = sum(camera_violations.values())
        if total_critical >= 2:
            risk_level = "critical"
        elif total_critical >= 1 or total_warning >= 3:
            risk_level = "high"
        elif total_warning >= 1 or total_violations >= 3:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Trend (compare with recent history)
        recent_counts = [
            sum(1 for _ in self._decision_history)
        ]  # simplified — just check if decisions are growing
        trend = "stable"
        if len(self._decision_history) >= 3:
            recent_3 = list(self._decision_history)[-3:]
            recent_actions = [d.action for d in recent_3]
            if all(a in (AgentAction.ESCALATE, AgentAction.PLAN) for a in recent_actions):
                trend = "rising"

        # Causal inference integration
        causal_edges: list[dict[str, Any]] = []
        if self._causal_inference is not None:
            try:
                dag = self._causal_inference.get_dag()
                edges = getattr(dag, "edges", None) or dag.get("edges", []) if isinstance(dag, dict) else []
                causal_edges = edges[:10] if isinstance(edges, list) else []
            except Exception:
                pass

        # RAG context (optional)
        rag_context = ""
        if self._use_rag_orient and self._rag_engine is not None and dominant:
            try:
                result = self._rag_engine.query(dominant, top_k=3)
                if hasattr(result, "answer"):
                    rag_context = result.answer
                elif isinstance(result, dict):
                    rag_context = result.get("answer", "")
            except Exception:
                pass

        # LLaVA summary (optional, expensive)
        llava_summary = ""
        if self._use_llava_orient and self._llava_client is not None and dominant:
            try:
                prompt = f"Summarize the current situation: {dominant} violations detected."
                resp = self._llava_client.answer_question(None, prompt)
                llava_summary = resp if isinstance(resp, str) else str(resp)
            except Exception:
                pass

        # Phase 62: Episode recall — find similar past situations
        recalled_episodes: list[dict[str, Any]] = []
        if self._episode_store is not None:
            try:
                recalled_episodes = self._episode_store.recall(
                    risk_level=risk_level,
                    severity_counts=dict(severity_counts),
                    camera_violations=dict(camera_violations),
                    dominant_violation_type=dominant,
                    top_k=3,
                )
            except Exception:
                pass

        return SituationSummary(
            camera_violations=camera_violations,
            severity_counts=severity_counts,
            dominant_violation_type=dominant,
            risk_level=risk_level,
            trend=trend,
            causal_edges=causal_edges,
            rag_context=rag_context,
            llava_summary=llava_summary,
            recalled_episodes=recalled_episodes,
        )

    # ── DECIDE ───────────────────────────────────────────────────────────

    def _decide(
        self, situation: SituationSummary
    ) -> tuple[AgentAction, str, str, dict[str, Any]]:
        """Priority cascade decision tree."""
        total_critical = situation.severity_counts.get("critical", 0)
        total_warning = situation.severity_counts.get("warning", 0)
        total_violations = sum(situation.camera_violations.values())
        dominant = situation.dominant_violation_type
        params: dict[str, Any] = {}

        # 1. Critical + forecaster worsening → ESCALATE
        if total_critical >= 1:
            forecaster_worsening = False
            if self._anomaly_forecaster is not None:
                try:
                    stats = self._anomaly_forecaster.get_stats()
                    alerts = stats.get("active_alerts", 0) if isinstance(stats, dict) else 0
                    forecaster_worsening = alerts > 0
                except Exception:
                    pass

            if forecaster_worsening or total_critical >= self._escalation_threshold:
                params = {"severity": "critical", "violation_type": dominant}
                return (
                    AgentAction.ESCALATE,
                    f"重大違反 ({total_critical}件) + 予知保全アラート — 即時エスカレーション",
                    f"Critical violations ({total_critical}) + forecaster alert — immediate escalation",
                    params,
                )

        # 2. High + repeated violations ≥ plan_threshold → PLAN
        if situation.risk_level in ("high", "critical") and total_violations >= self._plan_threshold:
            params = {"goal": f"Resolve {dominant}", "violation_count": total_violations}
            return (
                AgentAction.PLAN,
                f"高リスク + 反復違反 ({total_violations}件) — 安全プラン作成",
                f"High risk + repeated violations ({total_violations}) — creating safety plan",
                params,
            )

        # 3. Medium + novel pattern → ADAPT
        if situation.risk_level == "medium" and dominant:
            is_novel = False
            if self._online_learner is not None:
                try:
                    stats = self._online_learner.get_stats()
                    categories = stats.get("categories", []) if isinstance(stats, dict) else []
                    if dominant not in categories:
                        is_novel = True
                except Exception:
                    pass
            if is_novel:
                params = {"novel_pattern": dominant}
                return (
                    AgentAction.ADAPT,
                    f"新規パターン検出: {dominant} — オンライン学習に適応",
                    f"Novel pattern detected: {dominant} — adapting online learner",
                    params,
                )

        # 4. Violations + RAG available → QUERY
        if total_violations > 0 and self._use_rag_orient and self._rag_engine is not None:
            if situation.rag_context:
                params = {"query": dominant, "rag_context": situation.rag_context[:200]}
                return (
                    AgentAction.QUERY,
                    f"違反パターン '{dominant}' — RAG検索で過去事例を参照",
                    f"Violation pattern '{dominant}' — querying RAG for historical context",
                    params,
                )

        # 5. Early warning high risk → RESPOND
        if situation.risk_level in ("high", "critical") and self._early_warning_responder is not None:
            params = {"risk_level": situation.risk_level}
            return (
                AgentAction.RESPOND,
                f"予兆リスク {situation.risk_level} — 自律対応実行",
                f"Early warning risk {situation.risk_level} — executing autonomous response",
                params,
            )

        # 6. Default → MONITOR
        return (
            AgentAction.MONITOR,
            "異常なし — 監視継続",
            "No actionable anomalies — continuing monitoring",
            {},
        )

    # ── ACT ──────────────────────────────────────────────────────────────

    def _act(
        self,
        action: AgentAction,
        params: dict[str, Any],
        situation: SituationSummary,
    ) -> str:
        """Dispatch the chosen action to the relevant sub-system."""
        try:
            if action == AgentAction.ESCALATE:
                return self._act_escalate(params, situation)
            elif action == AgentAction.PLAN:
                return self._act_plan(params, situation)
            elif action == AgentAction.ADAPT:
                return self._act_adapt(params, situation)
            elif action == AgentAction.QUERY:
                return self._act_query(params)
            elif action == AgentAction.RESPOND:
                return self._act_respond(params)
            else:
                return "monitoring"
        except Exception as e:
            logger.debug("AgentLoop._act() error: %s", e, exc_info=True)
            return f"error: {e}"

    def _act_escalate(self, params: dict[str, Any], situation: SituationSummary) -> str:
        """Dispatch escalation via IncidentReporter."""
        if self._incident_reporter is not None:
            try:
                self._incident_reporter.report(
                    camera_id=list(situation.camera_violations.keys())[0] if situation.camera_violations else "cam",
                    violation_type=params.get("violation_type", "unknown"),
                    severity=params.get("severity", "critical"),
                    description=f"Agent escalation: {situation.dominant_violation_type}",
                )
                return "incident_reported"
            except Exception as e:
                return f"escalation_failed: {e}"
        return "no_incident_reporter"

    def _act_plan(self, params: dict[str, Any], situation: SituationSummary) -> str:
        """Submit a safety goal via SafetyPlanner."""
        if self._safety_planner is not None:
            try:
                goal_desc = params.get("goal", f"Resolve violations: {situation.dominant_violation_type}")
                self._safety_planner.submit_goal(
                    description=goal_desc,
                    priority=3,  # high
                )
                return "plan_submitted"
            except Exception as e:
                return f"plan_failed: {e}"
        return "no_safety_planner"

    def _act_adapt(self, params: dict[str, Any], situation: SituationSummary) -> str:
        """Trigger online learner adaptation."""
        if self._online_learner is not None:
            try:
                pattern = params.get("novel_pattern", "unknown")
                return f"adapt_noted:{pattern}"
            except Exception as e:
                return f"adapt_failed: {e}"
        return "no_online_learner"

    def _act_query(self, params: dict[str, Any]) -> str:
        """RAG query result is already in situation; just log it."""
        return f"rag_queried:{params.get('query', '')}"

    def _act_respond(self, params: dict[str, Any]) -> str:
        """Trigger early warning responder."""
        if self._early_warning_responder is not None:
            try:
                self._early_warning_responder.respond(
                    risk_level=params.get("risk_level", "high"),
                )
                return "ew_response_sent"
            except Exception as e:
                return f"respond_failed: {e}"
        return "no_early_warning_responder"

    # ── Public API ───────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return agent loop statistics."""
        return {
            "enabled": True,
            "status": self._status.value,
            "think_count": self._think_count,
            "pending_events": len(self._pending_events),
            "decision_count": len(self._decision_history),
            "action_counts": dict(self._action_counts),
            "think_interval_s": self._think_interval_s,
            "critical_trigger": self._critical_trigger,
            "use_llava_orient": self._use_llava_orient,
            "use_rag_orient": self._use_rag_orient,
            "escalation_threshold": self._escalation_threshold,
            "plan_threshold": self._plan_threshold,
            "uptime_s": round(time.time() - self._start_time, 1),
            "components": {
                "safety_planner": self._safety_planner is not None,
                "causal_inference": self._causal_inference is not None,
                "incident_reporter": self._incident_reporter is not None,
                "rag_engine": self._rag_engine is not None,
                "llava_client": self._llava_client is not None,
                "online_learner": self._online_learner is not None,
                "early_warning_responder": self._early_warning_responder is not None,
                "anomaly_forecaster": self._anomaly_forecaster is not None,
                "verifier": self._verifier is not None,
                "episode_store": self._episode_store is not None,
            },
        }

    def get_recent_decisions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent decisions as dicts."""
        decisions = list(self._decision_history)
        recent = decisions[-limit:] if limit < len(decisions) else decisions
        return [d.to_dict() for d in reversed(recent)]

    def get_pending_events(self) -> list[dict[str, Any]]:
        """Return the current pending event buffer."""
        return [
            {
                "event_type": ev.event_type,
                "camera_id": ev.camera_id,
                "severity": ev.severity,
                "violation_rule": ev.violation_rule,
                "timestamp": ev.timestamp,
            }
            for ev in self._pending_events
        ]

    def inject_event(self, event: AgentEvent) -> None:
        """Inject an event for testing purposes."""
        self._pending_events.append(event)

    def reset(self) -> None:
        """Reset all state."""
        self._pending_events.clear()
        self._decision_history.clear()
        self._action_counts = {a.value: 0 for a in AgentAction}
        self._think_count = 0
        self._last_think_time = 0.0
        self._status = AgentStatus.IDLE
        self._start_time = time.time()
