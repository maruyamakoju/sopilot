"""Autonomous action executor for perception events.

Maps EntityEvents -> ActionPlans -> ActionResults.
Provides a rule-based loop that converts perception insights into
configurable actions (alert, webhook, escalate, record, suppress).

Note: EntityEvent does not carry a severity field directly; severity is
read from event.details["severity"] with a default of "warning". Tests
may inject severity either via the details dict or by patching the event.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sopilot.perception.types import EntityEvent, EntityEventType, ViolationSeverity, WorldState

try:
    import httpx as httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


class ActionType(str, Enum):
    ALERT = "alert"          # Log a structured alert
    WEBHOOK = "webhook"      # POST to an external URL
    ESCALATE = "escalate"    # Escalate to VLM analysis
    RECORD = "record"        # Force-record a frame snapshot
    SUPPRESS = "suppress"    # Suppress downstream violations for cooldown


@dataclass
class ActionPlan:
    """A rule that maps an event condition to an action."""

    plan_id: str
    action_type: ActionType
    trigger_event_type: str                    # EntityEventType.name or "*" for any
    trigger_severity_min: str = "warning"      # min severity level
    cooldown_seconds: float = 60.0
    parameters: dict[str, Any] = field(default_factory=dict)
    description_ja: str = ""
    description_en: str = ""
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "action_type": self.action_type.value,
            "trigger_event_type": self.trigger_event_type,
            "trigger_severity_min": self.trigger_severity_min,
            "cooldown_seconds": self.cooldown_seconds,
            "parameters": self.parameters,
            "description_ja": self.description_ja,
            "description_en": self.description_en,
            "enabled": self.enabled,
        }


@dataclass
class ActionResult:
    """Result of executing an ActionPlan."""

    result_id: str
    plan_id: str
    action_type: ActionType
    executed_at: float            # Unix timestamp
    success: bool
    trigger_event_type: str
    trigger_severity: str
    entity_id: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "result_id": self.result_id,
            "plan_id": self.plan_id,
            "action_type": self.action_type.value,
            "executed_at": self.executed_at,
            "success": self.success,
            "trigger_event_type": self.trigger_event_type,
            "trigger_severity": self.trigger_severity,
            "entity_id": self.entity_id,
            "details": self.details,
        }


def _event_severity(event: EntityEvent) -> str:
    """Extract severity string from an EntityEvent.

    EntityEvent does not have a top-level severity attribute, so we look in
    event.details["severity"] and fall back to "warning".  If the caller has
    monkey-patched a .severity attribute onto the event (for testing) that is
    also accepted.
    """
    # Honour a monkey-patched / subclassed attribute first
    attr = getattr(event, "severity", None)
    if attr is not None:
        if hasattr(attr, "name"):
            return attr.name.lower()
        return str(attr).lower()
    # Fall back to details dict
    raw = event.details.get("severity", "warning")
    if hasattr(raw, "name"):
        return raw.name.lower()
    return str(raw).lower()


class ActionExecutor:
    """Evaluate EntityEvents against ActionPlans and execute matched actions.

    Thread-safe. All methods protected by a RLock.
    Max 200 result log entries (oldest dropped on overflow).
    """

    MAX_LOG_SIZE = 200

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._plans: dict[str, ActionPlan] = {}    # plan_id -> ActionPlan
        self._last_fired: dict[str, float] = {}    # plan_id -> last fire timestamp
        self._log: list[ActionResult] = []          # recent results, bounded

    # ── Plan management ───────────────────────────────────────────────────────

    def add_plan(self, plan: ActionPlan) -> str:
        """Register an action plan. Returns plan_id."""
        with self._lock:
            self._plans[plan.plan_id] = plan
            logger.info("ActionPlan added: %s (%s)", plan.plan_id, plan.action_type.value)
        return plan.plan_id

    def create_plan(
        self,
        action_type: ActionType,
        trigger_event_type: str,
        trigger_severity_min: str = "warning",
        cooldown_seconds: float = 60.0,
        parameters: dict | None = None,
        description_ja: str = "",
        description_en: str = "",
    ) -> ActionPlan:
        """Create and register a new plan, returning it."""
        plan = ActionPlan(
            plan_id=str(uuid.uuid4()),
            action_type=action_type,
            trigger_event_type=trigger_event_type,
            trigger_severity_min=trigger_severity_min,
            cooldown_seconds=cooldown_seconds,
            parameters=parameters or {},
            description_ja=description_ja,
            description_en=description_en,
        )
        self.add_plan(plan)
        return plan

    def remove_plan(self, plan_id: str) -> bool:
        """Remove a plan by ID. Returns True if found."""
        with self._lock:
            if plan_id in self._plans:
                del self._plans[plan_id]
                self._last_fired.pop(plan_id, None)
                return True
        return False

    def get_plans(self) -> list[ActionPlan]:
        """Return all registered plans."""
        with self._lock:
            return list(self._plans.values())

    def enable_plan(self, plan_id: str) -> bool:
        """Enable a plan. Returns True if found."""
        with self._lock:
            if plan_id in self._plans:
                self._plans[plan_id].enabled = True
                return True
        return False

    def disable_plan(self, plan_id: str) -> bool:
        """Disable a plan without removing it. Returns True if found."""
        with self._lock:
            if plan_id in self._plans:
                self._plans[plan_id].enabled = False
                return True
        return False

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        events: list[EntityEvent],
        world_state: WorldState | None = None,
    ) -> list[ActionResult]:
        """Check events against plans and execute matching actions.

        Returns list of ActionResults for actions that fired this call.
        Silently skips on-cooldown or disabled plans.
        """
        results: list[ActionResult] = []
        now = time.time()

        with self._lock:
            plans = list(self._plans.values())

        for plan in plans:
            if not plan.enabled:
                continue
            for event in events:
                if self._matches(plan, event, now):
                    result = self._execute(plan, event, now)
                    results.append(result)
                    with self._lock:
                        self._last_fired[plan.plan_id] = now
                        self._log.append(result)
                        if len(self._log) > self.MAX_LOG_SIZE:
                            self._log = self._log[-self.MAX_LOG_SIZE:]
                    # One fire per plan per evaluate() call (first matching event wins)
                    break

        return results

    # ── Log management ────────────────────────────────────────────────────────

    def get_log(self, n: int = 20) -> list[ActionResult]:
        """Return the most recent n action results."""
        with self._lock:
            if n <= 0:
                return []
            return list(self._log[-n:])

    def clear_log(self) -> None:
        """Clear the execution log."""
        with self._lock:
            self._log.clear()

    def get_state_dict(self) -> dict:
        """Return summary state for API."""
        with self._lock:
            return {
                "total_plans": len(self._plans),
                "enabled_plans": sum(1 for p in self._plans.values() if p.enabled),
                "total_executions": len(self._log),
                "plans": [p.to_dict() for p in self._plans.values()],
            }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _matches(self, plan: ActionPlan, event: EntityEvent, now: float) -> bool:
        """Check if event matches plan trigger conditions."""
        # Cooldown check
        last = self._last_fired.get(plan.plan_id, 0.0)
        if (now - last) < plan.cooldown_seconds:
            return False

        # Event type check
        event_type_name = (
            event.event_type.name
            if hasattr(event.event_type, "name")
            else str(event.event_type)
        )
        if plan.trigger_event_type != "*" and plan.trigger_event_type != event_type_name:
            return False

        # Severity check
        event_sev = _event_severity(event)
        min_level = _SEVERITY_ORDER.get(plan.trigger_severity_min.lower(), 1)
        event_level = _SEVERITY_ORDER.get(event_sev, 1)
        if event_level < min_level:
            return False

        return True

    def _execute(self, plan: ActionPlan, event: EntityEvent, now: float) -> ActionResult:
        """Execute the action for a matched plan+event."""
        event_type_name = (
            event.event_type.name
            if hasattr(event.event_type, "name")
            else str(event.event_type)
        )
        event_sev = _event_severity(event)
        result_id = str(uuid.uuid4())
        success = True
        details: dict = {}

        try:
            if plan.action_type == ActionType.ALERT:
                desc = event.details.get("description_ja", "")
                logger.warning(
                    "[ACTION:ALERT] plan=%s event=%s sev=%s entity=%d desc=%s",
                    plan.plan_id,
                    event_type_name,
                    event_sev,
                    event.entity_id,
                    desc,
                )
                details["message"] = f"Alert: {desc or event_type_name}"

            elif plan.action_type == ActionType.WEBHOOK:
                url = plan.parameters.get("url", "")
                details["url"] = url
                if url:
                    try:
                        httpx.post(
                            url,
                            json={
                                "plan_id": plan.plan_id,
                                "event_type": event_type_name,
                                "severity": event_sev,
                                "entity_id": event.entity_id,
                                "timestamp": event.timestamp,
                                "details": event.details,
                            },
                            timeout=5,
                        )
                        details["status"] = "sent"
                    except Exception as exc:
                        details["error"] = str(exc)
                        success = False
                else:
                    details["error"] = "no url configured"
                    success = False

            elif plan.action_type == ActionType.ESCALATE:
                logger.info(
                    "[ACTION:ESCALATE] plan=%s event=%s -> VLM escalation requested",
                    plan.plan_id,
                    event_type_name,
                )
                details["escalation_requested"] = True
                details["event_type"] = event_type_name

            elif plan.action_type == ActionType.RECORD:
                logger.info(
                    "[ACTION:RECORD] plan=%s frame=%d timestamp=%.1f",
                    plan.plan_id,
                    event.frame_number,
                    event.timestamp,
                )
                details["frame_number"] = event.frame_number
                details["timestamp"] = event.timestamp

            elif plan.action_type == ActionType.SUPPRESS:
                duration = plan.parameters.get("duration_seconds", 30.0)
                details["suppress_duration_seconds"] = duration
                logger.info(
                    "[ACTION:SUPPRESS] plan=%s suppressing for %.0fs",
                    plan.plan_id,
                    duration,
                )

        except Exception as exc:
            logger.exception("Action execution failed for plan=%s", plan.plan_id)
            success = False
            details["error"] = str(exc)

        return ActionResult(
            result_id=result_id,
            plan_id=plan.plan_id,
            action_type=plan.action_type,
            executed_at=now,
            success=success,
            trigger_event_type=event_type_name,
            trigger_severity=event_sev,
            entity_id=event.entity_id,
            details=details,
        )
