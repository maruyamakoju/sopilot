"""SOPilot Locust load test.

Usage::

    locust -f tests/load/locustfile.py --host http://localhost:8000 \\
           --users 50 --spawn-rate 5 --run-time 60s --headless

Task weights reflect realistic traffic distribution:
- 10%  health check
- 5%   readiness
- 20%  video list
- 15%  analytics
- 15%  vigil sessions
- 10%  early warning
- 10%  health score
- 5%   daily report
- 10%  metrics (Prometheus)
"""

from locust import HttpUser, between, task  # type: ignore[import-untyped]


class SOPilotUser(HttpUser):
    """Simulates a typical SOPilot UI/API consumer."""

    wait_time = between(0.5, 2.0)

    # ── Health & observability (15% combined) ──────────────────────────

    @task(10)
    def health_check(self) -> None:
        self.client.get("/health", name="/health")

    @task(5)
    def readiness_check(self) -> None:
        self.client.get("/readiness", name="/readiness")

    # ── Core API (35% combined) ────────────────────────────────────────

    @task(20)
    def list_videos(self) -> None:
        self.client.get("/videos?limit=20", name="/videos")

    @task(15)
    def analytics(self) -> None:
        self.client.get("/analytics", name="/analytics")

    # ── VigilPilot (25% combined) ─────────────────────────────────────

    @task(15)
    def vigil_sessions(self) -> None:
        self.client.get("/vigil/sessions?limit=10", name="/vigil/sessions")

    @task(10)
    def early_warning(self) -> None:
        self.client.get("/vigil/perception/early-warning", name="/vigil/perception/early-warning")

    # ── Perception health (15% combined) ──────────────────────────────

    @task(10)
    def health_score(self) -> None:
        self.client.get("/vigil/perception/health-score", name="/vigil/perception/health-score")

    @task(5)
    def daily_report(self) -> None:
        self.client.get("/vigil/perception/daily-report", name="/vigil/perception/daily-report")

    # ── Metrics ───────────────────────────────────────────────────────

    @task(10)
    def metrics(self) -> None:
        self.client.get("/metrics", name="/metrics")
