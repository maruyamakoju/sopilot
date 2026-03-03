"""VLM call budget manager for multi-camera/multi-session deployments.

Implements a token-bucket style rate limiter with per-session quotas
and dynamic priority boosting when violations are detected.

Higher-priority sessions get proportionally more of the global budget.
Thread-safe. No external dependencies beyond stdlib.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_SEVERITY_BOOST = {"info": 0.1, "warning": 0.3, "critical": 0.5}


@dataclass
class SessionQuota:
    """Per-session VLM call accounting."""
    session_id: str
    priority: float          # [0.0, 1.0] — higher gets more budget
    calls_this_window: int   # calls in current 60s window
    calls_total: int
    allowed_total: int
    denied_total: int
    last_violation_ts: float  # 0.0 if never
    last_call_ts: float       # 0.0 if never

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "priority": round(self.priority, 4),
            "calls_this_window": self.calls_this_window,
            "calls_total": self.calls_total,
            "allowed_total": self.allowed_total,
            "denied_total": self.denied_total,
            "last_violation_ts": self.last_violation_ts,
            "last_call_ts": self.last_call_ts,
        }


class AttentionBroker:
    """VLM call budget manager with per-session dynamic priority.

    Uses a sliding window (deque of timestamps) to enforce:
    - Global calls-per-minute cap
    - Per-session calls-per-minute cap (adjusted by priority)

    Priority mechanics:
    - Base priority: 0.5 for all new sessions
    - Violation boosts priority by `_SEVERITY_BOOST[severity]`
    - Priority decays back to base_priority over `priority_decay_s`

    Parameters
    ----------
    global_cpm : int
        Maximum total VLM calls per minute across all sessions.
    session_cpm : int
        Base VLM calls per minute per session (before priority scaling).
    base_priority : float
        Default priority for new sessions [0,1]. Default 0.5.
    priority_decay_s : float
        Seconds for priority to decay from max back to base. Default 120.
    window_s : float
        Sliding window duration in seconds. Default 60.0.
    """

    DEFAULT_GLOBAL_CPM: int = 20
    DEFAULT_SESSION_CPM: int = 5

    def __init__(
        self,
        global_cpm: int = DEFAULT_GLOBAL_CPM,
        session_cpm: int = DEFAULT_SESSION_CPM,
        base_priority: float = 0.5,
        priority_decay_s: float = 120.0,
        window_s: float = 60.0,
    ) -> None:
        self._global_cpm = max(1, global_cpm)
        self._session_cpm = max(1, session_cpm)
        self._base_priority = max(0.0, min(1.0, base_priority))
        self._decay_s = max(1.0, priority_decay_s)
        self._window_s = max(1.0, window_s)
        self._lock = threading.RLock()
        # global sliding window of call timestamps
        self._global_window: deque[float] = deque()
        # per-session data
        self._sessions: dict[str, dict] = {}
        # per-session sliding window
        self._session_windows: dict[str, deque[float]] = {}
        self._global_denied: int = 0
        self._global_allowed: int = 0

    def _get_or_create_session(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "priority": self._base_priority,
                "calls_total": 0,
                "allowed_total": 0,
                "denied_total": 0,
                "last_violation_ts": 0.0,
                "last_call_ts": 0.0,
            }
            self._session_windows[session_id] = deque()
        return self._sessions[session_id]

    def _prune_window(self, window: deque, now: float) -> None:
        """Remove timestamps older than window_s."""
        cutoff = now - self._window_s
        while window and window[0] < cutoff:
            window.popleft()

    def _decayed_priority(self, session_id: str, now: float) -> float:
        """Priority decays linearly from boosted value back to base."""
        s = self._sessions.get(session_id)
        if s is None:
            return self._base_priority
        p = s["priority"]
        vts = s["last_violation_ts"]
        if vts == 0.0 or p <= self._base_priority:
            return self._base_priority
        elapsed = now - vts
        if elapsed >= self._decay_s:
            return self._base_priority
        # Linear interpolation: p → base over decay_s
        frac = elapsed / self._decay_s
        return p - (p - self._base_priority) * frac

    def request_call(
        self,
        session_id: str,
        urgency: float = 0.5,
        _now: float | None = None,
    ) -> bool:
        """Request permission to make one VLM call.

        Returns True if allowed, False if budget exceeded.

        Parameters
        ----------
        urgency : float
            0.0-1.0. Higher urgency gets bonus allowance (scales session limit).
        """
        now = _now if _now is not None else time.time()
        with self._lock:
            s = self._get_or_create_session(session_id)

            # Prune windows
            self._prune_window(self._global_window, now)
            sw = self._session_windows[session_id]
            self._prune_window(sw, now)

            # Update decayed priority
            s["priority"] = self._decayed_priority(session_id, now)

            # Priority-adjusted per-session cap
            priority = max(0.1, s["priority"])
            effective_session_cap = max(1, round(self._session_cpm * (0.5 + priority)))

            # Urgency boost: up to +50% of session cap
            urgency_boost = max(0, round(effective_session_cap * 0.5 * urgency))
            final_session_cap = effective_session_cap + urgency_boost

            # Check global cap
            if len(self._global_window) >= self._global_cpm:
                s["denied_total"] += 1
                self._global_denied += 1
                return False

            # Check session cap
            if len(sw) >= final_session_cap:
                s["denied_total"] += 1
                self._global_denied += 1
                return False

            # Grant
            self._global_window.append(now)
            sw.append(now)
            s["calls_total"] += 1
            s["allowed_total"] += 1
            s["last_call_ts"] = now
            self._global_allowed += 1
            return True

    def record_violation(
        self,
        session_id: str,
        severity: str = "warning",
        _now: float | None = None,
    ) -> None:
        """Boost priority for session_id after a violation is detected."""
        now = _now if _now is not None else time.time()
        boost = _SEVERITY_BOOST.get(severity.lower(), 0.3)
        with self._lock:
            s = self._get_or_create_session(session_id)
            current = self._decayed_priority(session_id, now)
            s["priority"] = min(1.0, current + boost)
            s["last_violation_ts"] = now
            logger.debug(
                "AttentionBroker: session=%s priority boosted to %.2f (sev=%s)",
                session_id, s["priority"], severity,
            )

    def get_quota(self, session_id: str, _now: float | None = None) -> SessionQuota:
        """Return current SessionQuota for a session."""
        now = _now if _now is not None else time.time()
        with self._lock:
            s = self._get_or_create_session(session_id)
            sw = self._session_windows[session_id]
            self._prune_window(sw, now)
            return SessionQuota(
                session_id=session_id,
                priority=round(self._decayed_priority(session_id, now), 4),
                calls_this_window=len(sw),
                calls_total=s["calls_total"],
                allowed_total=s["allowed_total"],
                denied_total=s["denied_total"],
                last_violation_ts=s["last_violation_ts"],
                last_call_ts=s["last_call_ts"],
            )

    def get_state_dict(self, _now: float | None = None) -> dict:
        """Return global usage summary."""
        now = _now if _now is not None else time.time()
        with self._lock:
            self._prune_window(self._global_window, now)
            return {
                "global_cpm": self._global_cpm,
                "session_cpm": self._session_cpm,
                "global_calls_this_window": len(self._global_window),
                "global_allowed_total": self._global_allowed,
                "global_denied_total": self._global_denied,
                "session_count": len(self._sessions),
                "sessions": {
                    sid: {
                        "priority": round(self._decayed_priority(sid, now), 4),
                        "calls_total": s["calls_total"],
                        "denied_total": s["denied_total"],
                    }
                    for sid, s in self._sessions.items()
                },
            }

    def reset_session(self, session_id: str) -> None:
        """Reset call history for a session (priority preserved)."""
        with self._lock:
            if session_id in self._session_windows:
                self._session_windows[session_id].clear()
            if session_id in self._sessions:
                self._sessions[session_id]["calls_total"] = 0
                self._sessions[session_id]["allowed_total"] = 0
                self._sessions[session_id]["denied_total"] = 0

    def remove_session(self, session_id: str) -> bool:
        """Remove a session entirely. Returns True if it existed."""
        with self._lock:
            existed = session_id in self._sessions
            self._sessions.pop(session_id, None)
            self._session_windows.pop(session_id, None)
            return existed
