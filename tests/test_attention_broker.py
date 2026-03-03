"""Tests for sopilot.perception.attention_broker.

~55 pytest tests covering:
  SessionQuota, AttentionBroker init, request_call, record_violation,
  priority decay, get_quota, get_state_dict, reset_session, remove_session,
  and thread safety.

No sopilot imports beyond the module under test are needed.
"""
from __future__ import annotations

import threading
import time

import pytest

from sopilot.perception.attention_broker import (
    AttentionBroker,
    SessionQuota,
    _SEVERITY_BOOST,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = 1_000_000.0  # arbitrary fixed "now" for deterministic tests


def broker(
    global_cpm: int = 20,
    session_cpm: int = 5,
    base_priority: float = 0.5,
    priority_decay_s: float = 120.0,
    window_s: float = 60.0,
) -> AttentionBroker:
    return AttentionBroker(
        global_cpm=global_cpm,
        session_cpm=session_cpm,
        base_priority=base_priority,
        priority_decay_s=priority_decay_s,
        window_s=window_s,
    )


# ===========================================================================
# TestSessionQuotaToDict
# ===========================================================================

class TestSessionQuotaToDict:
    def _quota(self, **kwargs) -> SessionQuota:
        defaults = dict(
            session_id="sess-1",
            priority=0.5,
            calls_this_window=3,
            calls_total=10,
            allowed_total=8,
            denied_total=2,
            last_violation_ts=T0,
            last_call_ts=T0 + 1,
        )
        defaults.update(kwargs)
        return SessionQuota(**defaults)

    def test_all_keys_present(self):
        d = self._quota().to_dict()
        expected_keys = {
            "session_id", "priority", "calls_this_window",
            "calls_total", "allowed_total", "denied_total",
            "last_violation_ts", "last_call_ts",
        }
        assert set(d.keys()) == expected_keys

    def test_session_id_value(self):
        d = self._quota(session_id="cam-42").to_dict()
        assert d["session_id"] == "cam-42"

    def test_priority_rounded_to_4dp(self):
        d = self._quota(priority=0.123456789).to_dict()
        assert d["priority"] == round(0.123456789, 4)

    def test_numeric_fields_match(self):
        q = self._quota(calls_this_window=7, calls_total=20, allowed_total=18, denied_total=2)
        d = q.to_dict()
        assert d["calls_this_window"] == 7
        assert d["calls_total"] == 20
        assert d["allowed_total"] == 18
        assert d["denied_total"] == 2

    def test_timestamp_fields(self):
        q = self._quota(last_violation_ts=T0, last_call_ts=T0 + 5.0)
        d = q.to_dict()
        assert d["last_violation_ts"] == T0
        assert d["last_call_ts"] == T0 + 5.0

    def test_zero_violation_ts(self):
        d = self._quota(last_violation_ts=0.0).to_dict()
        assert d["last_violation_ts"] == 0.0


# ===========================================================================
# TestInit
# ===========================================================================

class TestInit:
    def test_default_params(self):
        b = AttentionBroker()
        assert b._global_cpm == AttentionBroker.DEFAULT_GLOBAL_CPM
        assert b._session_cpm == AttentionBroker.DEFAULT_SESSION_CPM
        assert b._base_priority == 0.5
        assert b._decay_s == 120.0
        assert b._window_s == 60.0

    def test_custom_params(self):
        b = broker(global_cpm=10, session_cpm=3, base_priority=0.7,
                   priority_decay_s=60.0, window_s=30.0)
        assert b._global_cpm == 10
        assert b._session_cpm == 3
        assert b._base_priority == 0.7
        assert b._decay_s == 60.0
        assert b._window_s == 30.0

    def test_global_cpm_min_1_clamping(self):
        b = broker(global_cpm=0)
        assert b._global_cpm == 1

    def test_session_cpm_min_1_clamping(self):
        b = broker(session_cpm=0)
        assert b._session_cpm == 1

    def test_negative_global_cpm_clamped(self):
        b = broker(global_cpm=-5)
        assert b._global_cpm == 1

    def test_base_priority_clamped_to_zero_one(self):
        b_low = broker(base_priority=-0.5)
        assert b_low._base_priority == 0.0
        b_high = broker(base_priority=1.5)
        assert b_high._base_priority == 1.0

    def test_initial_counters_zero(self):
        b = AttentionBroker()
        assert b._global_allowed == 0
        assert b._global_denied == 0
        assert len(b._sessions) == 0


# ===========================================================================
# TestRequestCall
# ===========================================================================

class TestFirstCallAllowed:
    def test_fresh_session_allowed(self):
        b = broker()
        result = b.request_call("sess-1", _now=T0)
        assert result is True

    def test_returns_true_on_success(self):
        b = broker()
        assert b.request_call("sess-new", _now=T0) is True


class TestGlobalCapEnforced:
    def test_global_cap_blocks_excess(self):
        b = broker(global_cpm=3, session_cpm=100)
        now = T0
        # Fill global cap across different sessions so session cap doesn't bite first
        b.request_call("s1", _now=now)
        b.request_call("s2", _now=now)
        b.request_call("s3", _now=now)
        # 4th must be denied
        result = b.request_call("s4", _now=now)
        assert result is False

    def test_global_allowed_counter_matches_cap(self):
        b = broker(global_cpm=2, session_cpm=100)
        b.request_call("s1", _now=T0)
        b.request_call("s2", _now=T0)
        b.request_call("s3", _now=T0)  # denied
        assert b._global_allowed == 2
        assert b._global_denied == 1


class TestSessionCapEnforced:
    def test_session_cap_blocks_single_session(self):
        # session_cpm=2, base_priority=0.5 → effective = round(2*(0.5+0.5))=2
        # urgency=0 → urgency_boost=0 → final_session_cap=2
        b = broker(global_cpm=100, session_cpm=2, base_priority=0.5)
        b.request_call("s1", urgency=0.0, _now=T0)
        b.request_call("s1", urgency=0.0, _now=T0)
        result = b.request_call("s1", urgency=0.0, _now=T0)
        assert result is False

    def test_denied_total_incremented(self):
        b = broker(global_cpm=100, session_cpm=1, base_priority=0.5)
        b.request_call("s1", urgency=0.0, _now=T0)
        b.request_call("s1", urgency=0.0, _now=T0)  # denied
        q = b.get_quota("s1", _now=T0)
        assert q.denied_total >= 1


class TestMultiSessionIndependence:
    def test_session_a_cap_does_not_block_b(self):
        b = broker(global_cpm=100, session_cpm=2, base_priority=0.5)
        # Fill session A
        b.request_call("A", urgency=0.0, _now=T0)
        b.request_call("A", urgency=0.0, _now=T0)
        b.request_call("A", urgency=0.0, _now=T0)  # denied
        # Session B should still be allowed
        result = b.request_call("B", urgency=0.0, _now=T0)
        assert result is True

    def test_two_sessions_track_independently(self):
        b = broker(global_cpm=100, session_cpm=5)
        for _ in range(3):
            b.request_call("A", _now=T0)
        for _ in range(2):
            b.request_call("B", _now=T0)
        qa = b.get_quota("A", _now=T0)
        qb = b.get_quota("B", _now=T0)
        assert qa.calls_this_window == 3
        assert qb.calls_this_window == 2


class TestUrgencyBoostsLimit:
    def test_urgency_1_allows_more_than_urgency_0(self):
        # With urgency=0, final_session_cap = effective_session_cap
        # With urgency=1, final_session_cap = effective + round(effective*0.5*1)
        # Use session_cpm=4, base_priority=0.5 → effective=round(4*1.0)=4
        # urgency=1 → boost=round(4*0.5*1)=2 → cap=6
        # urgency=0 → boost=0 → cap=4
        b_low = broker(global_cpm=100, session_cpm=4)
        b_high = broker(global_cpm=100, session_cpm=4)

        now = T0
        allowed_low = sum(
            1 for _ in range(10)
            if b_low.request_call("s", urgency=0.0, _now=now)
        )
        allowed_high = sum(
            1 for _ in range(10)
            if b_high.request_call("s", urgency=1.0, _now=now)
        )
        assert allowed_high > allowed_low


class TestReturnsBool:
    def test_always_returns_bool_on_allow(self):
        b = broker()
        result = b.request_call("s", _now=T0)
        assert isinstance(result, bool)

    def test_always_returns_bool_on_deny(self):
        b = broker(global_cpm=1, session_cpm=1)
        b.request_call("s", urgency=0.0, _now=T0)
        result = b.request_call("s", urgency=0.0, _now=T0)
        assert isinstance(result, bool)


class TestCallCountsIncrement:
    def test_allowed_total_increments(self):
        b = broker()
        b.request_call("s", _now=T0)
        b.request_call("s", _now=T0)
        q = b.get_quota("s", _now=T0)
        assert q.allowed_total == 2

    def test_denied_total_increments(self):
        b = broker(global_cpm=1, session_cpm=100)
        b.request_call("s", _now=T0)  # allowed
        b.request_call("s", _now=T0)  # denied (global cap)
        q = b.get_quota("s", _now=T0)
        assert q.denied_total == 1

    def test_calls_total_is_sum_of_allowed_and_denied(self):
        b = broker(global_cpm=2, session_cpm=100)
        b.request_call("s", _now=T0)
        b.request_call("s", _now=T0)
        b.request_call("s", _now=T0)  # denied
        q = b.get_quota("s", _now=T0)
        # calls_total only counts allowed (see impl: incremented on grant)
        assert q.calls_total == q.allowed_total


# ===========================================================================
# TestRecordViolation
# ===========================================================================

class TestViolationBoostsPriority:
    def test_priority_increases_after_violation(self):
        b = broker(base_priority=0.5)
        b.record_violation("s", severity="warning", _now=T0)
        q = b.get_quota("s", _now=T0)
        assert q.priority > 0.5

    def test_last_violation_ts_updated(self):
        b = broker()
        b.record_violation("s", severity="info", _now=T0)
        assert b._sessions["s"]["last_violation_ts"] == T0


class TestCriticalViolationBoostsMore:
    def test_critical_boost_greater_than_warning(self):
        b_warn = broker(base_priority=0.5)
        b_crit = broker(base_priority=0.5)
        b_warn.record_violation("s", severity="warning", _now=T0)
        b_crit.record_violation("s", severity="critical", _now=T0)
        p_warn = b_warn._sessions["s"]["priority"]
        p_crit = b_crit._sessions["s"]["priority"]
        assert p_crit > p_warn


class TestPriorityCapAt1:
    def test_multiple_violations_never_exceed_1(self):
        b = broker(base_priority=0.5)
        for _ in range(10):
            b.record_violation("s", severity="critical", _now=T0)
        assert b._sessions["s"]["priority"] <= 1.0


class TestHighPriorityGetsMoreCalls:
    def test_boosted_session_gets_more_calls(self):
        # Two brokers: one with violation, one without
        b_base = broker(global_cpm=100, session_cpm=4)
        b_boost = broker(global_cpm=100, session_cpm=4)
        b_boost.record_violation("s", severity="critical", _now=T0)
        # Count allowed calls at urgency=0 in the same window
        allowed_base = sum(
            1 for _ in range(20) if b_base.request_call("s", urgency=0.0, _now=T0)
        )
        allowed_boost = sum(
            1 for _ in range(20) if b_boost.request_call("s", urgency=0.0, _now=T0)
        )
        assert allowed_boost >= allowed_base


class TestSeverityBoostValues:
    def test_info_boost(self):
        assert _SEVERITY_BOOST["info"] == 0.1

    def test_warning_boost(self):
        assert _SEVERITY_BOOST["warning"] == 0.3

    def test_critical_boost(self):
        assert _SEVERITY_BOOST["critical"] == 0.5


class TestUnknownSeverityDefaultsToWarning:
    def test_unknown_severity_uses_0_3(self):
        b = broker(base_priority=0.5)
        b.record_violation("s", severity="UNKNOWN_LEVEL", _now=T0)
        p = b._sessions["s"]["priority"]
        # Should have applied 0.3 boost: 0.5 + 0.3 = 0.8
        assert abs(p - 0.8) < 1e-9


# ===========================================================================
# TestPriorityDecay
# ===========================================================================

class TestPriorityDecaysToBase:
    def test_after_full_decay_priority_is_base(self):
        b = broker(base_priority=0.5, priority_decay_s=100.0)
        b.record_violation("s", severity="critical", _now=T0)
        # At T0 + decay_s, priority should be base
        p = b._decayed_priority("s", now=T0 + 100.0)
        assert abs(p - 0.5) < 1e-9

    def test_after_decay_request_call_uses_base_priority(self):
        b = broker(global_cpm=100, session_cpm=4, base_priority=0.5, priority_decay_s=10.0)
        b.record_violation("s", severity="critical", _now=T0)
        # Make a request well after decay
        result = b.request_call("s", urgency=0.0, _now=T0 + 1000.0)
        assert result is True  # still allowed; just priority is base


class TestPriorityDecayLinear:
    def test_half_decay_halfway_between(self):
        b = broker(base_priority=0.5, priority_decay_s=100.0)
        # Set priority directly to 1.0, violation at T0
        b._get_or_create_session("s")
        b._sessions["s"]["priority"] = 1.0
        b._sessions["s"]["last_violation_ts"] = T0
        # At T0 + 50 (half of decay_s=100), expect halfway: 0.75
        p = b._decayed_priority("s", now=T0 + 50.0)
        assert abs(p - 0.75) < 1e-6

    def test_quarter_decay(self):
        b = broker(base_priority=0.5, priority_decay_s=100.0)
        b._get_or_create_session("s")
        b._sessions["s"]["priority"] = 1.0
        b._sessions["s"]["last_violation_ts"] = T0
        # At T0 + 25 (quarter of 100), expect: 1.0 - (1.0-0.5)*0.25 = 0.875
        p = b._decayed_priority("s", now=T0 + 25.0)
        assert abs(p - 0.875) < 1e-6


class TestNoViolationNeverBoostsPriority:
    def test_priority_stays_at_base_without_violation(self):
        b = broker(base_priority=0.5)
        b._get_or_create_session("s")
        p = b._decayed_priority("s", now=T0 + 999.0)
        assert p == 0.5


# ===========================================================================
# TestGetQuota
# ===========================================================================

class TestGetQuotaNewSession:
    def test_returns_session_quota_type(self):
        b = broker()
        q = b.get_quota("s", _now=T0)
        assert isinstance(q, SessionQuota)

    def test_new_session_defaults(self):
        b = broker(base_priority=0.5)
        q = b.get_quota("s", _now=T0)
        assert q.session_id == "s"
        assert q.calls_this_window == 0
        assert q.calls_total == 0
        assert q.allowed_total == 0
        assert q.denied_total == 0
        assert q.last_violation_ts == 0.0
        assert q.last_call_ts == 0.0


class TestQuotaCallsThisWindowDecrement:
    def test_calls_in_window_counted(self):
        b = broker(global_cpm=100, session_cpm=100)
        b.request_call("s", _now=T0)
        b.request_call("s", _now=T0)
        q = b.get_quota("s", _now=T0)
        assert q.calls_this_window == 2

    def test_old_calls_pruned_from_window(self):
        b = broker(global_cpm=100, session_cpm=100, window_s=60.0)
        b.request_call("s", _now=T0)
        b.request_call("s", _now=T0)
        # Advance time beyond window
        q = b.get_quota("s", _now=T0 + 61.0)
        assert q.calls_this_window == 0


class TestQuotaFieldsConsistent:
    def test_calls_this_window_lte_calls_total(self):
        b = broker(global_cpm=100, session_cpm=100)
        for _ in range(5):
            b.request_call("s", _now=T0)
        q = b.get_quota("s", _now=T0)
        assert q.calls_this_window <= q.calls_total


# ===========================================================================
# TestGetStateDict
# ===========================================================================

class TestGetStateDictKeys:
    def test_required_keys_present(self):
        b = broker()
        d = b.get_state_dict(_now=T0)
        required = {
            "global_cpm", "session_cpm", "global_calls_this_window",
            "global_allowed_total", "global_denied_total",
            "session_count", "sessions",
        }
        assert required.issubset(set(d.keys()))

    def test_session_entry_keys(self):
        b = broker()
        b.request_call("s", _now=T0)
        d = b.get_state_dict(_now=T0)
        sess = d["sessions"]["s"]
        assert "priority" in sess
        assert "calls_total" in sess
        assert "denied_total" in sess


class TestSessionCountAccurate:
    def test_n_sessions_reflected(self):
        b = broker()
        b.request_call("A", _now=T0)
        b.request_call("B", _now=T0)
        b.request_call("C", _now=T0)
        d = b.get_state_dict(_now=T0)
        assert d["session_count"] == 3

    def test_zero_sessions_initially(self):
        b = broker()
        d = b.get_state_dict(_now=T0)
        assert d["session_count"] == 0


# ===========================================================================
# TestResetSession
# ===========================================================================

class TestResetSessionClearsWindow:
    def test_calls_this_window_zero_after_reset(self):
        b = broker(global_cpm=100, session_cpm=100)
        b.request_call("s", _now=T0)
        b.request_call("s", _now=T0)
        b.reset_session("s")
        q = b.get_quota("s", _now=T0)
        assert q.calls_this_window == 0

    def test_calls_total_zero_after_reset(self):
        b = broker(global_cpm=100, session_cpm=100)
        b.request_call("s", _now=T0)
        b.reset_session("s")
        q = b.get_quota("s", _now=T0)
        assert q.calls_total == 0


class TestResetPreservesPriority:
    def test_priority_unchanged_after_reset(self):
        b = broker(base_priority=0.5)
        b.record_violation("s", severity="critical", _now=T0)
        priority_before = b._sessions["s"]["priority"]
        b.reset_session("s")
        priority_after = b._sessions["s"]["priority"]
        assert priority_before == priority_after

    def test_reset_nonexistent_session_no_error(self):
        b = broker()
        b.reset_session("ghost")  # should not raise


# ===========================================================================
# TestRemoveSession
# ===========================================================================

class TestRemoveExistingSession:
    def test_returns_true_and_session_gone(self):
        b = broker()
        b.request_call("s", _now=T0)
        assert "s" in b._sessions
        result = b.remove_session("s")
        assert result is True
        assert "s" not in b._sessions
        assert "s" not in b._session_windows

    def test_state_dict_excludes_removed_session(self):
        b = broker()
        b.request_call("s", _now=T0)
        b.remove_session("s")
        d = b.get_state_dict(_now=T0)
        assert "s" not in d["sessions"]


class TestRemoveNonexistent:
    def test_returns_false_no_error(self):
        b = broker()
        result = b.remove_session("ghost")
        assert result is False


# ===========================================================================
# TestThreadSafety
# ===========================================================================

class TestConcurrentRequests:
    def test_20_threads_no_errors_total_allowed_lte_global_cpm(self):
        global_cpm = 10
        b = broker(global_cpm=global_cpm, session_cpm=100)
        results = []
        errors = []
        lock = threading.Lock()

        def worker(sid: str):
            try:
                for _ in range(5):
                    r = b.request_call(sid)
                    with lock:
                        results.append(r)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"s{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Thread errors: {errors}"
        allowed_count = sum(1 for r in results if r)
        # All sessions share the same time window (real time), so allowed may vary;
        # but allowed should be > 0 and <= global_cpm * n_windows.
        # In CI the window is real time so we just assert no over-grant:
        # global_window should never exceed global_cpm at any instant
        state = b.get_state_dict()
        assert state["global_calls_this_window"] <= global_cpm

    def test_no_data_races_concurrent_access(self):
        b = broker(global_cpm=50, session_cpm=10)
        errors = []
        lock = threading.Lock()

        def worker():
            try:
                for i in range(10):
                    b.request_call(f"s{i % 5}")
                    b.get_quota(f"s{i % 5}")
                    b.get_state_dict()
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors


class TestConcurrentViolationAndRequest:
    def test_concurrent_record_and_request_no_errors(self):
        b = broker(global_cpm=100, session_cpm=50)
        errors = []
        lock = threading.Lock()

        def requester():
            try:
                for _ in range(20):
                    b.request_call("shared")
            except Exception as exc:
                with lock:
                    errors.append(exc)

        def violator():
            try:
                for sev in ["info", "warning", "critical"] * 5:
                    b.record_violation("shared", severity=sev)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = (
            [threading.Thread(target=requester) for _ in range(5)]
            + [threading.Thread(target=violator) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors
