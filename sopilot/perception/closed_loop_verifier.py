"""Closed-Loop Verification — Phase 60.

After the AgentLoop ACTs, the verifier re-observes the situation N seconds
later to determine whether the action was effective.  Outcome history drives
a MetaLearner that gradually adjusts the agent's decision thresholds.

Flow:
    AgentLoop.think()
        └─ _act() → decision + situation_baseline
        └─ verifier.schedule(decision, situation, camera_id)

    Per-frame Stage 6v (same camera, ≥ verify_after_s later):
        verifier.check_pending(frame_result, camera_id)
            └─ _compute_outcome(baseline, current_metrics)
            └─ _meta_update(record)          ← adjust thresholds

Outcome determination:
    baseline_violations = total violations when action was taken
    current_violations  = violations in the verification frame

    reduction = (baseline - current) / max(baseline, 1)

    SUCCESS  : reduction ≥ success_threshold   (default 0.5 = 50% drop)
    PARTIAL  : reduction ≥ 0.1
    FAILURE  : reduction < 0.1  or  violations increased
    UNKNOWN  : no data (first observation for this camera, baseline=0)
"""

from __future__ import annotations

import enum
import logging
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Enumerations ──────────────────────────────────────────────────────────


class VerificationOutcome(str, enum.Enum):
    """Result of a closed-loop verification check."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    UNKNOWN = "unknown"


# ── Data structures ───────────────────────────────────────────────────────


@dataclass
class VerificationTask:
    """A pending verification scheduled after an agent action."""
    task_id: str
    decision_action: str         # AgentAction value
    camera_id: str
    baseline_violations: int     # total violations at ACT time
    baseline_severity: dict[str, int]  # {"critical":0,"warning":0,"info":0}
    baseline_risk_level: str
    scheduled_at: float          # time.time() when created
    check_after_s: float         # seconds to wait before checking
    decision_reasoning_ja: str = ""
    episode_id: str | None = None      # AgentEpisodeStore episode ID (Phase 62)

    @property
    def check_due_at(self) -> float:
        return self.scheduled_at + self.check_after_s

    @property
    def is_due(self) -> bool:
        return time.time() >= self.check_due_at


@dataclass
class VerificationRecord:
    """A completed verification with outcome."""
    task_id: str
    decision_action: str
    camera_id: str
    baseline_violations: int
    current_violations: int
    baseline_risk_level: str
    outcome: VerificationOutcome
    reduction_ratio: float       # (baseline - current) / max(baseline, 1)
    latency_s: float             # seconds from schedule to check
    verified_at: float
    decision_reasoning_ja: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d


# ── MetaLearner ──────────────────────────────────────────────────────────


_EMA_ALPHA = 0.2        # smoothing factor for EMA success rate
_THRESHOLD_MAX_DELTA = 2  # max adjustment ± per action type


class MetaLearner:
    """Adjusts agent decision thresholds based on outcome EMA.

    Tracks per-action success rates and suggests threshold adjustments:
      - If ESCALATE success_rate < 0.3 → suggest raising escalation_threshold
      - If PLAN success_rate > 0.7 → suggest lowering plan_threshold
      - Caps adjustments at ±_THRESHOLD_MAX_DELTA from original value
    """

    def __init__(self, original_escalation: int = 3, original_plan: int = 5) -> None:
        self._original_escalation = original_escalation
        self._original_plan = original_plan

        # EMA success rate per action (None = no data yet)
        self._success_rate: dict[str, float | None] = {}
        self._update_count: dict[str, int] = {}
        self._escalation_adj: int = 0  # +n = raise threshold (less aggressive)
        self._plan_adj: int = 0

    def update(self, action: str, outcome: VerificationOutcome) -> None:
        """Update EMA success rate and compute threshold adjustments."""
        is_success = outcome in (VerificationOutcome.SUCCESS, VerificationOutcome.PARTIAL)
        score = 1.0 if is_success else 0.0

        prev = self._success_rate.get(action)
        if prev is None:
            self._success_rate[action] = score
        else:
            self._success_rate[action] = _EMA_ALPHA * score + (1 - _EMA_ALPHA) * prev

        self._update_count[action] = self._update_count.get(action, 0) + 1

        # Adjust escalation threshold
        sr_esc = self._success_rate.get("escalate")
        if sr_esc is not None and self._update_count.get("escalate", 0) >= 3:
            if sr_esc < 0.3:
                # Too many failed escalations → be less aggressive
                self._escalation_adj = min(self._escalation_adj + 1, _THRESHOLD_MAX_DELTA)
            elif sr_esc > 0.8:
                # Very successful → can be more aggressive
                self._escalation_adj = max(self._escalation_adj - 1, -_THRESHOLD_MAX_DELTA)

        # Adjust plan threshold
        sr_plan = self._success_rate.get("plan")
        if sr_plan is not None and self._update_count.get("plan", 0) >= 3:
            if sr_plan < 0.3:
                self._plan_adj = min(self._plan_adj + 1, _THRESHOLD_MAX_DELTA)
            elif sr_plan > 0.7:
                self._plan_adj = max(self._plan_adj - 1, -_THRESHOLD_MAX_DELTA)

    @property
    def suggested_escalation_threshold(self) -> int:
        return max(1, self._original_escalation + self._escalation_adj)

    @property
    def suggested_plan_threshold(self) -> int:
        return max(2, self._original_plan + self._plan_adj)

    def get_stats(self) -> dict[str, Any]:
        return {
            "success_rates": {
                k: round(v, 3) for k, v in self._success_rate.items() if v is not None
            },
            "update_counts": dict(self._update_count),
            "escalation_threshold_adj": self._escalation_adj,
            "plan_threshold_adj": self._plan_adj,
            "suggested_escalation_threshold": self.suggested_escalation_threshold,
            "suggested_plan_threshold": self.suggested_plan_threshold,
        }

    def reset(self) -> None:
        self._success_rate.clear()
        self._update_count.clear()
        self._escalation_adj = 0
        self._plan_adj = 0


# ── ClosedLoopVerifier ───────────────────────────────────────────────────


_MAX_PENDING = 100


class ClosedLoopVerifier:
    """Schedules and evaluates closed-loop verification tasks.

    Called from:
      1. AgentLoop._act() via schedule()
      2. engine.py Stage 6v via check_pending()
    """

    def __init__(
        self,
        *,
        verify_after_s: float = 30.0,
        success_threshold: float = 0.5,
        meta_learning_enabled: bool = True,
        history_size: int = 200,
        original_escalation: int = 3,
        original_plan: int = 5,
        episode_store: Any | None = None,     # AgentEpisodeStore (Phase 62)
    ) -> None:
        self._verify_after_s = verify_after_s
        self._success_threshold = success_threshold
        self._meta_learning_enabled = meta_learning_enabled

        self._pending: dict[str, VerificationTask] = {}  # task_id → task
        self._records: deque[VerificationRecord] = deque(maxlen=history_size)
        self._outcome_counts: dict[str, int] = {o.value: 0 for o in VerificationOutcome}
        self._meta = MetaLearner(
            original_escalation=original_escalation,
            original_plan=original_plan,
        )
        self._episode_store = episode_store  # Phase 62
        self._start_time = time.time()

    # ── Public API ───────────────────────────────────────────────────────

    def schedule(
        self,
        decision_action: str,
        camera_id: str,
        baseline_violations: int,
        baseline_severity: dict[str, int],
        baseline_risk_level: str,
        decision_reasoning_ja: str = "",
        episode_id: str | None = None,
    ) -> str:
        """Schedule a verification task.  Returns task_id."""
        if len(self._pending) >= _MAX_PENDING:
            # Drop oldest pending task
            oldest = next(iter(self._pending))
            del self._pending[oldest]

        task_id = str(uuid.uuid4())[:8]
        task = VerificationTask(
            task_id=task_id,
            decision_action=decision_action,
            camera_id=camera_id,
            baseline_violations=baseline_violations,
            baseline_severity=dict(baseline_severity),
            baseline_risk_level=baseline_risk_level,
            scheduled_at=time.time(),
            check_after_s=self._verify_after_s,
            decision_reasoning_ja=decision_reasoning_ja,
            episode_id=episode_id,
        )
        self._pending[task_id] = task
        logger.debug(
            "Verification scheduled: task_id=%s action=%s camera=%s baseline=%d",
            task_id, decision_action, camera_id, baseline_violations,
        )
        return task_id

    def check_pending(
        self, frame_result: Any, camera_id: str
    ) -> list[VerificationRecord]:
        """Check due tasks for this camera.  Returns newly completed records."""
        completed: list[VerificationRecord] = []

        # Current frame metrics
        violations = getattr(frame_result, "violations", None) or []
        current_count = len(violations)

        # Find due tasks for this camera
        due_ids = [
            tid for tid, task in self._pending.items()
            if task.camera_id == camera_id and task.is_due
        ]

        for tid in due_ids:
            task = self._pending.pop(tid)
            try:
                outcome, reduction = self._compute_outcome(
                    task.baseline_violations, current_count
                )
                record = VerificationRecord(
                    task_id=tid,
                    decision_action=task.decision_action,
                    camera_id=camera_id,
                    baseline_violations=task.baseline_violations,
                    current_violations=current_count,
                    baseline_risk_level=task.baseline_risk_level,
                    outcome=outcome,
                    reduction_ratio=round(reduction, 3),
                    latency_s=round(time.time() - task.scheduled_at, 1),
                    verified_at=time.time(),
                    decision_reasoning_ja=task.decision_reasoning_ja,
                )
                self._records.append(record)
                self._outcome_counts[outcome.value] = (
                    self._outcome_counts.get(outcome.value, 0) + 1
                )
                if self._meta_learning_enabled:
                    self._meta.update(task.decision_action, outcome)

                # Phase 62: update episode outcome in AgentEpisodeStore
                if self._episode_store is not None and task.episode_id is not None:
                    try:
                        self._episode_store.update_outcome(
                            task.episode_id,
                            outcome.value,
                            reduction_ratio=reduction,
                        )
                    except Exception:
                        logger.debug("Episode outcome update failed", exc_info=True)

                completed.append(record)
                logger.debug(
                    "Verification complete: task_id=%s action=%s outcome=%s "
                    "baseline=%d current=%d reduction=%.2f",
                    tid, task.decision_action, outcome.value,
                    task.baseline_violations, current_count, reduction,
                )
            except Exception:
                logger.debug("Verification check failed for task %s", tid, exc_info=True)

        return completed

    def _compute_outcome(
        self, baseline: int, current: int
    ) -> tuple[VerificationOutcome, float]:
        """Determine outcome from violation counts."""
        if baseline == 0:
            if current == 0:
                return VerificationOutcome.UNKNOWN, 0.0
            else:
                return VerificationOutcome.FAILURE, -float(current)

        reduction = (baseline - current) / baseline
        if reduction >= self._success_threshold:
            return VerificationOutcome.SUCCESS, reduction
        elif reduction >= 0.1:
            return VerificationOutcome.PARTIAL, reduction
        else:
            return VerificationOutcome.FAILURE, reduction

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "pending_tasks": len(self._pending),
            "completed_records": len(self._records),
            "outcome_counts": dict(self._outcome_counts),
            "verify_after_s": self._verify_after_s,
            "success_threshold": self._success_threshold,
            "meta_learning_enabled": self._meta_learning_enabled,
            "meta": self._meta.get_stats(),
            "uptime_s": round(time.time() - self._start_time, 1),
        }

    def get_recent_records(self, limit: int = 10) -> list[dict[str, Any]]:
        records = list(self._records)
        recent = records[-limit:] if limit < len(records) else records
        return [r.to_dict() for r in reversed(recent)]

    def get_pending_tasks(self) -> list[dict[str, Any]]:
        now = time.time()
        return [
            {
                "task_id": t.task_id,
                "decision_action": t.decision_action,
                "camera_id": t.camera_id,
                "baseline_violations": t.baseline_violations,
                "due_in_s": round(max(0.0, t.check_due_at - now), 1),
                "decision_reasoning_ja": t.decision_reasoning_ja,
            }
            for t in self._pending.values()
        ]

    def get_meta_state(self) -> dict[str, Any]:
        return self._meta.get_stats()

    def reset(self) -> None:
        self._pending.clear()
        self._records.clear()
        self._outcome_counts = {o.value: 0 for o in VerificationOutcome}
        self._meta.reset()
        self._start_time = time.time()
