"""Locust load test shape scenarios.

Usage — run with a specific scenario::

    locust -f tests/load/locustfile.py --host http://localhost:8000 \\
           --shape-class SpikeScenario

Scenarios:
- SpikeScenario  — ramp to 200 users in 10s, hold 30s, drop to 0
- SoakScenario   — hold 20 users for 10 minutes (memory leak detection)
- StressScenario — step up every 30s until 500 users
"""

from __future__ import annotations

from locust import LoadTestShape  # type: ignore[import-untyped]


class SpikeScenario(LoadTestShape):
    """Spike test: sudden burst to 200 users, then immediate drop."""

    def tick(self) -> tuple[int, int] | None:
        run_time = self.get_run_time()
        if run_time < 10:
            return 200, 50   # ramp: 200 users, spawn 50/s
        if run_time < 40:
            return 200, 10   # hold
        return None          # stop


class SoakScenario(LoadTestShape):
    """Soak test: sustained 20 users for 10 minutes."""

    SOAK_DURATION_SECONDS = 600

    def tick(self) -> tuple[int, int] | None:
        if self.get_run_time() < self.SOAK_DURATION_SECONDS:
            return 20, 2
        return None


class StressScenario(LoadTestShape):
    """Stress test: step up 50 users every 30 seconds until 500."""

    STEP_USERS = 50
    STEP_DURATION = 30
    MAX_USERS = 500

    def tick(self) -> tuple[int, int] | None:
        run_time = self.get_run_time()
        current_step = int(run_time // self.STEP_DURATION)
        user_count = min(self.STEP_USERS * (current_step + 1), self.MAX_USERS)
        if user_count > self.MAX_USERS:
            return None
        return user_count, self.STEP_USERS
