"""Tests for sopilot.perception.action_executor — Phase 7 autonomous action loop.

All 45+ tests in a single TestCase class.
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from sopilot.perception.types import EntityEvent, EntityEventType, ViolationSeverity
from sopilot.perception.action_executor import (
    ActionExecutor,
    ActionPlan,
    ActionResult,
    ActionType,
    _SEVERITY_ORDER,
    _event_severity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_type: EntityEventType = EntityEventType.ANOMALY,
    entity_id: int = 1,
    timestamp: float = 1.0,
    frame_number: int = 10,
    severity: str | None = "warning",
    **detail_kwargs,
) -> EntityEvent:
    """Create a minimal EntityEvent.  Severity is stored in details["severity"]."""
    details = {}
    if severity is not None:
        details["severity"] = severity
    details.update(detail_kwargs)
    return EntityEvent(
        event_type=event_type,
        entity_id=entity_id,
        timestamp=timestamp,
        frame_number=frame_number,
        details=details,
    )


def _make_plan(
    action_type: ActionType = ActionType.ALERT,
    trigger_event_type: str = "ANOMALY",
    trigger_severity_min: str = "warning",
    cooldown_seconds: float = 0.0,
    parameters: dict | None = None,
    enabled: bool = True,
) -> ActionPlan:
    """Create a minimal ActionPlan with zero cooldown by default."""
    import uuid
    return ActionPlan(
        plan_id=str(uuid.uuid4()),
        action_type=action_type,
        trigger_event_type=trigger_event_type,
        trigger_severity_min=trigger_severity_min,
        cooldown_seconds=cooldown_seconds,
        parameters=parameters or {},
        enabled=enabled,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestActionExecutor(unittest.TestCase):

    # ── ActionType enum ──────────────────────────────────────────────────────

    def test_action_type_alert_value(self):
        self.assertEqual(ActionType.ALERT.value, "alert")

    def test_action_type_webhook_value(self):
        self.assertEqual(ActionType.WEBHOOK.value, "webhook")

    def test_action_type_escalate_value(self):
        self.assertEqual(ActionType.ESCALATE.value, "escalate")

    def test_action_type_record_value(self):
        self.assertEqual(ActionType.RECORD.value, "record")

    def test_action_type_suppress_value(self):
        self.assertEqual(ActionType.SUPPRESS.value, "suppress")

    # ── ActionPlan ───────────────────────────────────────────────────────────

    def test_action_plan_to_dict_keys(self):
        plan = _make_plan()
        d = plan.to_dict()
        expected_keys = {
            "plan_id", "action_type", "trigger_event_type",
            "trigger_severity_min", "cooldown_seconds", "parameters",
            "description_ja", "description_en", "enabled",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_action_plan_to_dict_action_type_is_str(self):
        plan = _make_plan(action_type=ActionType.ESCALATE)
        self.assertEqual(plan.to_dict()["action_type"], "escalate")

    def test_action_plan_to_dict_enabled_true(self):
        plan = _make_plan(enabled=True)
        self.assertTrue(plan.to_dict()["enabled"])

    def test_action_plan_to_dict_enabled_false(self):
        plan = _make_plan(enabled=False)
        self.assertFalse(plan.to_dict()["enabled"])

    # ── ActionResult ─────────────────────────────────────────────────────────

    def test_action_result_to_dict_keys(self):
        result = ActionResult(
            result_id="r1",
            plan_id="p1",
            action_type=ActionType.ALERT,
            executed_at=999.0,
            success=True,
            trigger_event_type="ANOMALY",
            trigger_severity="warning",
            entity_id=42,
            details={"msg": "ok"},
        )
        d = result.to_dict()
        expected_keys = {
            "result_id", "plan_id", "action_type", "executed_at",
            "success", "trigger_event_type", "trigger_severity",
            "entity_id", "details",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_action_result_to_dict_action_type_is_str(self):
        result = ActionResult(
            result_id="r2", plan_id="p2",
            action_type=ActionType.RECORD,
            executed_at=1.0, success=True,
            trigger_event_type="ANOMALY", trigger_severity="critical",
            entity_id=7,
        )
        self.assertEqual(result.to_dict()["action_type"], "record")

    # ── ActionExecutor basic API ─────────────────────────────────────────────

    def test_add_plan_returns_plan_id(self):
        executor = ActionExecutor()
        plan = _make_plan()
        returned = executor.add_plan(plan)
        self.assertEqual(returned, plan.plan_id)

    def test_create_plan_convenience(self):
        executor = ActionExecutor()
        plan = executor.create_plan(
            action_type=ActionType.ALERT,
            trigger_event_type="ANOMALY",
        )
        self.assertIsInstance(plan, ActionPlan)
        self.assertEqual(plan.action_type, ActionType.ALERT)
        self.assertIn(plan, executor.get_plans())

    def test_get_plans_empty_initially(self):
        executor = ActionExecutor()
        self.assertEqual(executor.get_plans(), [])

    def test_get_plans_returns_registered(self):
        executor = ActionExecutor()
        p1 = _make_plan()
        p2 = _make_plan(action_type=ActionType.RECORD)
        executor.add_plan(p1)
        executor.add_plan(p2)
        ids = {p.plan_id for p in executor.get_plans()}
        self.assertIn(p1.plan_id, ids)
        self.assertIn(p2.plan_id, ids)

    def test_remove_plan_returns_true_when_found(self):
        executor = ActionExecutor()
        plan = _make_plan()
        executor.add_plan(plan)
        self.assertTrue(executor.remove_plan(plan.plan_id))

    def test_remove_plan_returns_false_when_missing(self):
        executor = ActionExecutor()
        self.assertFalse(executor.remove_plan("nonexistent-id"))

    def test_remove_plan_removes_from_get_plans(self):
        executor = ActionExecutor()
        plan = _make_plan()
        executor.add_plan(plan)
        executor.remove_plan(plan.plan_id)
        self.assertEqual(executor.get_plans(), [])

    def test_enable_plan_returns_true_when_found(self):
        executor = ActionExecutor()
        plan = _make_plan(enabled=False)
        executor.add_plan(plan)
        self.assertTrue(executor.enable_plan(plan.plan_id))

    def test_enable_plan_returns_false_when_missing(self):
        executor = ActionExecutor()
        self.assertFalse(executor.enable_plan("ghost-id"))

    def test_disable_plan_returns_true_when_found(self):
        executor = ActionExecutor()
        plan = _make_plan(enabled=True)
        executor.add_plan(plan)
        self.assertTrue(executor.disable_plan(plan.plan_id))

    def test_disable_plan_returns_false_when_missing(self):
        executor = ActionExecutor()
        self.assertFalse(executor.disable_plan("ghost-id"))

    # ── evaluate() scenarios ─────────────────────────────────────────────────

    def test_evaluate_empty_events_returns_empty(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan())
        self.assertEqual(executor.evaluate([]), [])

    def test_evaluate_matching_anomaly_event_returns_one_result(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(trigger_event_type="ANOMALY", cooldown_seconds=0.0))
        event = _make_event(event_type=EntityEventType.ANOMALY, severity="warning")
        results = executor.evaluate([event])
        self.assertEqual(len(results), 1)

    def test_evaluate_wrong_event_type_no_result(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(trigger_event_type="RULE_VIOLATION", cooldown_seconds=0.0))
        event = _make_event(event_type=EntityEventType.ANOMALY, severity="warning")
        results = executor.evaluate([event])
        self.assertEqual(results, [])

    def test_evaluate_severity_below_threshold_no_result(self):
        executor = ActionExecutor()
        # Requires "critical" but event is "info"
        executor.add_plan(_make_plan(trigger_severity_min="critical", cooldown_seconds=0.0))
        event = _make_event(severity="info")
        results = executor.evaluate([event])
        self.assertEqual(results, [])

    def test_evaluate_cooldown_blocks_second_call(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(cooldown_seconds=60.0))
        event = _make_event()
        # First call should fire
        r1 = executor.evaluate([event])
        self.assertEqual(len(r1), 1)
        # Immediate second call should be blocked by cooldown
        r2 = executor.evaluate([event])
        self.assertEqual(r2, [])

    def test_evaluate_wildcard_trigger_matches_any_event_type(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(trigger_event_type="*", cooldown_seconds=0.0))
        event = _make_event(event_type=EntityEventType.ENTERED, severity="warning")
        results = executor.evaluate([event])
        self.assertEqual(len(results), 1)

    def test_evaluate_wildcard_matches_different_types(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(trigger_event_type="*", cooldown_seconds=0.0))
        for et in [EntityEventType.ANOMALY, EntityEventType.RULE_VIOLATION, EntityEventType.EXITED]:
            executor.clear_log()
            executor._last_fired.clear()
            event = _make_event(event_type=et, severity="warning")
            results = executor.evaluate([event])
            self.assertEqual(len(results), 1, f"wildcard should match {et}")

    def test_evaluate_multiple_plans_only_matching_fires(self):
        executor = ActionExecutor()
        plan_a = _make_plan(trigger_event_type="ANOMALY", cooldown_seconds=0.0,
                            action_type=ActionType.ALERT)
        plan_b = _make_plan(trigger_event_type="RULE_VIOLATION", cooldown_seconds=0.0,
                            action_type=ActionType.RECORD)
        executor.add_plan(plan_a)
        executor.add_plan(plan_b)
        event = _make_event(event_type=EntityEventType.ANOMALY, severity="warning")
        results = executor.evaluate([event])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].plan_id, plan_a.plan_id)

    def test_evaluate_result_has_correct_plan_id(self):
        executor = ActionExecutor()
        plan = _make_plan(trigger_event_type="ANOMALY", cooldown_seconds=0.0)
        executor.add_plan(plan)
        event = _make_event()
        results = executor.evaluate([event])
        self.assertEqual(results[0].plan_id, plan.plan_id)

    # ── Action-type specific details ─────────────────────────────────────────

    def test_alert_action_success_true(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(action_type=ActionType.ALERT, cooldown_seconds=0.0))
        results = executor.evaluate([_make_event()])
        self.assertTrue(results[0].success)

    def test_escalate_action_details_escalation_requested(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(action_type=ActionType.ESCALATE, cooldown_seconds=0.0))
        results = executor.evaluate([_make_event()])
        self.assertTrue(results[0].details.get("escalation_requested"))

    def test_record_action_details_frame_number(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(action_type=ActionType.RECORD, cooldown_seconds=0.0))
        event = _make_event(frame_number=42)
        results = executor.evaluate([event])
        self.assertEqual(results[0].details.get("frame_number"), 42)

    def test_suppress_action_details_duration(self):
        executor = ActionExecutor()
        plan = _make_plan(
            action_type=ActionType.SUPPRESS,
            cooldown_seconds=0.0,
            parameters={"duration_seconds": 90.0},
        )
        executor.add_plan(plan)
        results = executor.evaluate([_make_event()])
        self.assertEqual(results[0].details.get("suppress_duration_seconds"), 90.0)

    def test_webhook_no_url_success_false(self):
        executor = ActionExecutor()
        plan = _make_plan(action_type=ActionType.WEBHOOK, cooldown_seconds=0.0,
                          parameters={})  # no url
        executor.add_plan(plan)
        results = executor.evaluate([_make_event()])
        self.assertFalse(results[0].success)
        self.assertIn("error", results[0].details)

    # ── get_log() ────────────────────────────────────────────────────────────

    def test_get_log_returns_recent_results(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(cooldown_seconds=0.0))
        for _ in range(5):
            executor._last_fired.clear()
            executor.evaluate([_make_event()])
        self.assertLessEqual(len(executor.get_log(3)), 3)

    def test_get_log_zero_returns_empty(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(cooldown_seconds=0.0))
        executor.evaluate([_make_event()])
        self.assertEqual(executor.get_log(0), [])

    def test_get_log_default_n_returns_up_to_20(self):
        executor = ActionExecutor()
        plan = _make_plan(cooldown_seconds=0.0)
        executor.add_plan(plan)
        for _ in range(25):
            executor._last_fired.clear()
            executor.evaluate([_make_event()])
        log = executor.get_log()  # default n=20
        self.assertLessEqual(len(log), 20)

    # ── clear_log() ──────────────────────────────────────────────────────────

    def test_clear_log_empties_log(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(cooldown_seconds=0.0))
        executor.evaluate([_make_event()])
        executor.clear_log()
        self.assertEqual(executor.get_log(), [])

    # ── MAX_LOG_SIZE ─────────────────────────────────────────────────────────

    def test_max_log_size_kept_at_200(self):
        executor = ActionExecutor()
        plan = _make_plan(cooldown_seconds=0.0)
        executor.add_plan(plan)
        for _ in range(201):
            executor._last_fired.clear()
            executor.evaluate([_make_event()])
        with executor._lock:
            self.assertLessEqual(len(executor._log), ActionExecutor.MAX_LOG_SIZE)

    # ── get_state_dict() ─────────────────────────────────────────────────────

    def test_get_state_dict_structure(self):
        executor = ActionExecutor()
        d = executor.get_state_dict()
        self.assertIn("total_plans", d)
        self.assertIn("enabled_plans", d)
        self.assertIn("total_executions", d)
        self.assertIn("plans", d)

    def test_get_state_dict_counts_correct(self):
        executor = ActionExecutor()
        p1 = _make_plan(enabled=True)
        p2 = _make_plan(enabled=False)
        executor.add_plan(p1)
        executor.add_plan(p2)
        d = executor.get_state_dict()
        self.assertEqual(d["total_plans"], 2)
        self.assertEqual(d["enabled_plans"], 1)

    def test_get_state_dict_plans_list(self):
        executor = ActionExecutor()
        plan = _make_plan()
        executor.add_plan(plan)
        d = executor.get_state_dict()
        self.assertIsInstance(d["plans"], list)
        self.assertEqual(len(d["plans"]), 1)

    def test_get_state_dict_total_executions_after_evaluate(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(cooldown_seconds=0.0))
        executor.evaluate([_make_event()])
        d = executor.get_state_dict()
        self.assertGreaterEqual(d["total_executions"], 1)

    # ── Thread safety ─────────────────────────────────────────────────────────

    def test_thread_safety_concurrent_evaluate(self):
        """10 threads each calling evaluate() concurrently should not crash."""
        executor = ActionExecutor()
        # Use cooldown=0 so every call can potentially fire
        executor.add_plan(_make_plan(trigger_event_type="ANOMALY", cooldown_seconds=0.0))
        event = _make_event(event_type=EntityEventType.ANOMALY, severity="warning")

        errors: list[Exception] = []
        fired_counts: list[int] = []

        def worker():
            try:
                results = executor.evaluate([event])
                fired_counts.append(len(results))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(errors, [], f"Thread errors: {errors}")
        # Total fires should be between 1 and 10 (cooldown race, but no crash)
        self.assertGreaterEqual(sum(fired_counts), 1)

    # ── Severity threshold edge cases ────────────────────────────────────────

    def test_info_event_critical_threshold_no_fire(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(trigger_severity_min="critical", cooldown_seconds=0.0))
        event = _make_event(severity="info")
        results = executor.evaluate([event])
        self.assertEqual(results, [])

    def test_critical_event_info_threshold_fires(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(trigger_severity_min="info", cooldown_seconds=0.0))
        event = _make_event(severity="critical")
        results = executor.evaluate([event])
        self.assertEqual(len(results), 1)

    def test_warning_event_warning_threshold_fires(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(trigger_severity_min="warning", cooldown_seconds=0.0))
        event = _make_event(severity="warning")
        results = executor.evaluate([event])
        self.assertEqual(len(results), 1)

    def test_info_event_warning_threshold_no_fire(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(trigger_severity_min="warning", cooldown_seconds=0.0))
        event = _make_event(severity="info")
        results = executor.evaluate([event])
        self.assertEqual(results, [])

    # ── Disabled plan ────────────────────────────────────────────────────────

    def test_disabled_plan_never_fires(self):
        executor = ActionExecutor()
        plan = _make_plan(enabled=False, cooldown_seconds=0.0)
        executor.add_plan(plan)
        event = _make_event()
        results = executor.evaluate([event])
        self.assertEqual(results, [])

    def test_re_enabling_plan_fires_again(self):
        executor = ActionExecutor()
        plan = _make_plan(enabled=True, cooldown_seconds=0.0)
        executor.add_plan(plan)
        event = _make_event()

        # Fire once
        r1 = executor.evaluate([event])
        self.assertEqual(len(r1), 1)

        # Disable — should not fire
        executor.disable_plan(plan.plan_id)
        executor._last_fired.clear()
        r2 = executor.evaluate([event])
        self.assertEqual(r2, [])

        # Re-enable — should fire again
        executor.enable_plan(plan.plan_id)
        executor._last_fired.clear()
        r3 = executor.evaluate([event])
        self.assertEqual(len(r3), 1)

    # ── _event_severity helper ────────────────────────────────────────────────

    def test_event_severity_from_details(self):
        event = _make_event(severity="critical")
        self.assertEqual(_event_severity(event), "critical")

    def test_event_severity_default_warning_when_missing(self):
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1,
            timestamp=0.0,
            frame_number=0,
            details={},  # no severity key
        )
        self.assertEqual(_event_severity(event), "warning")

    def test_event_severity_accepts_violation_severity_enum_in_details(self):
        event = EntityEvent(
            event_type=EntityEventType.ANOMALY,
            entity_id=1,
            timestamp=0.0,
            frame_number=0,
            details={"severity": ViolationSeverity.CRITICAL},
        )
        self.assertEqual(_event_severity(event), "critical")

    # ── Webhook action (with mock) ────────────────────────────────────────────

    def test_webhook_with_url_calls_httpx(self):
        """Webhook action with a URL should attempt an HTTP POST (mocked)."""
        executor = ActionExecutor()
        plan = _make_plan(
            action_type=ActionType.WEBHOOK,
            cooldown_seconds=0.0,
            parameters={"url": "http://example.com/hook"},
        )
        executor.add_plan(plan)
        with patch("sopilot.perception.action_executor.httpx") as mock_httpx:
            mock_httpx.post.return_value = None
            results = executor.evaluate([_make_event()])
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        mock_httpx.post.assert_called_once()

    # ── First-event-wins per evaluate() call ─────────────────────────────────

    def test_only_first_matching_event_fires_plan(self):
        """When multiple events match a plan, only one result is produced."""
        executor = ActionExecutor()
        executor.add_plan(_make_plan(trigger_event_type="ANOMALY", cooldown_seconds=0.0))
        events = [
            _make_event(entity_id=1),
            _make_event(entity_id=2),
            _make_event(entity_id=3),
        ]
        results = executor.evaluate(events)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].entity_id, 1)

    # ── result fields ────────────────────────────────────────────────────────

    def test_result_trigger_event_type_correct(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(trigger_event_type="ANOMALY", cooldown_seconds=0.0))
        results = executor.evaluate([_make_event(event_type=EntityEventType.ANOMALY)])
        self.assertEqual(results[0].trigger_event_type, "ANOMALY")

    def test_result_trigger_severity_correct(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(cooldown_seconds=0.0, trigger_severity_min="info"))
        results = executor.evaluate([_make_event(severity="critical")])
        self.assertEqual(results[0].trigger_severity, "critical")

    def test_result_entity_id_correct(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(cooldown_seconds=0.0))
        results = executor.evaluate([_make_event(entity_id=99)])
        self.assertEqual(results[0].entity_id, 99)

    def test_result_executed_at_is_recent(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(cooldown_seconds=0.0))
        before = time.time()
        results = executor.evaluate([_make_event()])
        after = time.time()
        self.assertGreaterEqual(results[0].executed_at, before)
        self.assertLessEqual(results[0].executed_at, after)

    def test_result_result_id_is_unique(self):
        executor = ActionExecutor()
        executor.add_plan(_make_plan(cooldown_seconds=0.0))
        r1 = executor.evaluate([_make_event()])
        executor._last_fired.clear()
        r2 = executor.evaluate([_make_event()])
        self.assertNotEqual(r1[0].result_id, r2[0].result_id)


if __name__ == "__main__":
    unittest.main()
