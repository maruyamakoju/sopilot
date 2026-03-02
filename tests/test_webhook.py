"""Tests for the global VigilPilot webhook alert notification system.

Coverage:
  - WebhookRepository: CRUD (create, list, get, delete, update_enabled, update_triggered) — 8 tests
  - WebhookDispatcher: _send, dispatch_violation severity/enabled filtering, test_webhook — 12 tests
  - API endpoints (E2E via TestClient): CRUD + enable/disable + test + 404 handling — 20 tests
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sopilot.database import Database
from sopilot.main import create_app
from sopilot.vigil.webhook_dispatcher import SEVERITY_ORDER, WebhookDispatcher
from sopilot.vigil.webhook_repository import WebhookRepository


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _init_db(db_path: Path) -> str:
    """Initialise a fresh SQLite database with all tables + migrations."""
    db = Database(db_path)
    db.close()
    return str(db_path)


def _make_repo(tmp_path: Path) -> WebhookRepository:
    db_path = tmp_path / "test_wh.db"
    _init_db(db_path)
    return WebhookRepository(db_path)


def _make_client(tmp_path: Path) -> TestClient:
    """Create a TestClient backed by a fresh in-process app."""
    db_path = tmp_path / "test_app.db"
    _init_db(db_path)
    import os
    os.environ.setdefault("SOPILOT_DB_PATH", str(db_path))
    os.environ.setdefault("SOPILOT_DATA_DIR", str(tmp_path))
    os.environ["SOPILOT_DB_PATH"] = str(db_path)
    os.environ["SOPILOT_DATA_DIR"] = str(tmp_path)
    app = create_app()
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────────────────
# Repository Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestWebhookRepository:
    """Unit tests for WebhookRepository — fresh SQLite DB per test."""

    def test_create_returns_dict(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        row = repo.create("https://example.com/wh", "Test", "", "critical")
        assert isinstance(row, dict)
        assert row["url"] == "https://example.com/wh"
        assert row["name"] == "Test"
        assert row["min_severity"] == "critical"
        assert row["enabled"] == 1
        assert row["trigger_count"] == 0
        assert row["last_triggered_at"] is None
        assert "created_at" in row

    def test_create_auto_increments_id(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        r1 = repo.create("https://a.com", "A", "", "critical")
        r2 = repo.create("https://b.com", "B", "", "warning")
        assert r2["id"] > r1["id"]

    def test_list_all_empty(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert repo.list_all() == []

    def test_list_all_returns_all(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        repo.create("https://a.com", "A", "", "critical")
        repo.create("https://b.com", "B", "", "warning")
        rows = repo.list_all()
        assert len(rows) == 2
        urls = {r["url"] for r in rows}
        assert urls == {"https://a.com", "https://b.com"}

    def test_get_existing(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        created = repo.create("https://c.com", "C", "mysecret", "info")
        fetched = repo.get(created["id"])
        assert fetched is not None
        assert fetched["url"] == "https://c.com"
        assert fetched["secret"] == "mysecret"

    def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert repo.get(9999) is None

    def test_update_enabled_disable(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        row = repo.create("https://d.com", "", "", "critical")
        wid = row["id"]
        ok = repo.update_enabled(wid, False)
        assert ok is True
        fetched = repo.get(wid)
        assert fetched is not None
        assert fetched["enabled"] == 0

    def test_update_enabled_re_enable(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        row = repo.create("https://e.com", "", "", "warning")
        wid = row["id"]
        repo.update_enabled(wid, False)
        repo.update_enabled(wid, True)
        fetched = repo.get(wid)
        assert fetched is not None
        assert fetched["enabled"] == 1

    def test_update_enabled_nonexistent_returns_false(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert repo.update_enabled(9999, True) is False

    def test_delete_existing(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        row = repo.create("https://f.com", "", "", "critical")
        wid = row["id"]
        assert repo.delete(wid) is True
        assert repo.get(wid) is None

    def test_delete_nonexistent_returns_false(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert repo.delete(9999) is False

    def test_update_triggered_increments_count(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        row = repo.create("https://g.com", "", "", "critical")
        wid = row["id"]
        repo.update_triggered(wid)
        repo.update_triggered(wid)
        fetched = repo.get(wid)
        assert fetched is not None
        assert fetched["trigger_count"] == 2
        assert fetched["last_triggered_at"] is not None


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestWebhookDispatcher:
    """Unit tests for WebhookDispatcher."""

    # ── SEVERITY_ORDER constant ───────────────────────────────────────────────

    def test_severity_order_values(self) -> None:
        assert SEVERITY_ORDER["info"] < SEVERITY_ORDER["warning"] < SEVERITY_ORDER["critical"]

    # ── _send ─────────────────────────────────────────────────────────────────

    def test_send_posts_json_payload(self) -> None:
        dispatcher = WebhookDispatcher()
        wh = {"id": 1, "url": "https://example.com/hook", "secret": "", "enabled": 1}
        mock_repo = MagicMock()
        payload = {"event": "violation", "severity": "critical"}

        with patch("sopilot.vigil.webhook_dispatcher.httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            dispatcher._send(wh, payload, mock_repo)

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs[0][0] == "https://example.com/hook"
            body = json.loads(call_kwargs[1]["content"])
            assert body["event"] == "violation"
            assert body["severity"] == "critical"
            headers = call_kwargs[1]["headers"]
            assert headers["Content-Type"] == "application/json"
            assert "SOPilot-Webhook" in headers["User-Agent"]

    def test_send_adds_hmac_header_when_secret_set(self) -> None:
        import hashlib
        import hmac as _hmac

        dispatcher = WebhookDispatcher()
        secret = "my-secret"
        wh = {"id": 2, "url": "https://example.com/secure", "secret": secret, "enabled": 1}
        payload = {"event": "test"}
        mock_repo = MagicMock()

        with patch("sopilot.vigil.webhook_dispatcher.httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            dispatcher._send(wh, payload, mock_repo)

            call_kwargs = mock_post.call_args
            headers = call_kwargs[1]["headers"]
            assert "X-SOPilot-Signature" in headers
            sig_header = headers["X-SOPilot-Signature"]
            assert sig_header.startswith("sha256=")
            body_str = call_kwargs[1]["content"]
            expected_sig = _hmac.new(
                secret.encode(), body_str.encode(), hashlib.sha256
            ).hexdigest()
            assert sig_header == f"sha256={expected_sig}"

    def test_send_no_hmac_header_when_no_secret(self) -> None:
        dispatcher = WebhookDispatcher()
        wh = {"id": 3, "url": "https://example.com/hook", "secret": "", "enabled": 1}
        mock_repo = MagicMock()

        with patch("sopilot.vigil.webhook_dispatcher.httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            dispatcher._send(wh, {"event": "test"}, mock_repo)

            headers = mock_post.call_args[1]["headers"]
            assert "X-SOPilot-Signature" not in headers

    def test_send_calls_update_triggered_on_success(self) -> None:
        dispatcher = WebhookDispatcher()
        wh = {"id": 5, "url": "https://example.com/hook", "secret": "", "enabled": 1}
        mock_repo = MagicMock()

        with patch("sopilot.vigil.webhook_dispatcher.httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            dispatcher._send(wh, {"event": "v"}, mock_repo)

        mock_repo.update_triggered.assert_called_once_with(5)

    def test_send_does_not_raise_on_network_error(self) -> None:
        dispatcher = WebhookDispatcher()
        wh = {"id": 6, "url": "https://unreachable.invalid/hook", "secret": "", "enabled": 1}
        mock_repo = MagicMock()

        with patch("sopilot.vigil.webhook_dispatcher.httpx.post", side_effect=Exception("timeout")):
            # Must not raise
            dispatcher._send(wh, {"event": "v"}, mock_repo)

        mock_repo.update_triggered.assert_not_called()

    # ── dispatch_violation ────────────────────────────────────────────────────

    def test_dispatch_skips_disabled_webhook(self) -> None:
        dispatcher = WebhookDispatcher()
        webhooks = [{"id": 10, "url": "https://x.com/h", "secret": "", "enabled": 0, "min_severity": "info"}]
        mock_repo = MagicMock()
        payload = {"event": "violation", "severity": "critical"}

        with patch("sopilot.vigil.webhook_dispatcher.httpx.post") as mock_post:
            dispatcher.dispatch_violation(payload, webhooks, mock_repo)
            time.sleep(0.05)  # give daemon threads time to run
            mock_post.assert_not_called()

    def test_dispatch_skips_below_min_severity(self) -> None:
        """A warning-threshold webhook should NOT fire for an info-level event."""
        dispatcher = WebhookDispatcher()
        webhooks = [{"id": 11, "url": "https://x.com/h", "secret": "", "enabled": 1, "min_severity": "warning"}]
        mock_repo = MagicMock()
        payload = {"event": "violation", "severity": "info"}

        with patch("sopilot.vigil.webhook_dispatcher.httpx.post") as mock_post:
            dispatcher.dispatch_violation(payload, webhooks, mock_repo)
            time.sleep(0.05)
            mock_post.assert_not_called()

    def test_dispatch_fires_when_severity_meets_threshold(self) -> None:
        """A warning-threshold webhook SHOULD fire for a critical event."""
        dispatcher = WebhookDispatcher()
        webhooks = [{"id": 12, "url": "https://x.com/h", "secret": "", "enabled": 1, "min_severity": "warning"}]
        mock_repo = MagicMock()
        payload = {"event": "violation", "severity": "critical"}
        fired = threading.Event()

        def _fake_post(*args, **kwargs):
            fired.set()
            r = MagicMock()
            r.status_code = 200
            return r

        with patch("sopilot.vigil.webhook_dispatcher.httpx.post", side_effect=_fake_post):
            dispatcher.dispatch_violation(payload, webhooks, mock_repo)
            assert fired.wait(timeout=2.0), "webhook was not fired within 2s"

    def test_dispatch_fires_for_matching_severity(self) -> None:
        """A critical-only webhook SHOULD fire for a critical event."""
        dispatcher = WebhookDispatcher()
        webhooks = [{"id": 13, "url": "https://x.com/h", "secret": "", "enabled": 1, "min_severity": "critical"}]
        mock_repo = MagicMock()
        payload = {"event": "violation", "severity": "critical"}
        fired = threading.Event()

        def _fake_post(*args, **kwargs):
            fired.set()
            r = MagicMock()
            r.status_code = 200
            return r

        with patch("sopilot.vigil.webhook_dispatcher.httpx.post", side_effect=_fake_post):
            dispatcher.dispatch_violation(payload, webhooks, mock_repo)
            assert fired.wait(timeout=2.0)

    def test_dispatch_critical_webhook_skips_warning_event(self) -> None:
        """A critical-only webhook should NOT fire for a warning event."""
        dispatcher = WebhookDispatcher()
        webhooks = [{"id": 14, "url": "https://x.com/h", "secret": "", "enabled": 1, "min_severity": "critical"}]
        mock_repo = MagicMock()
        payload = {"event": "violation", "severity": "warning"}

        with patch("sopilot.vigil.webhook_dispatcher.httpx.post") as mock_post:
            dispatcher.dispatch_violation(payload, webhooks, mock_repo)
            time.sleep(0.05)
            mock_post.assert_not_called()

    def test_dispatch_multiple_webhooks_fires_eligible_only(self) -> None:
        """Only the webhook whose min_severity is met should fire."""
        dispatcher = WebhookDispatcher()
        webhooks = [
            {"id": 20, "url": "https://a.com/h", "secret": "", "enabled": 1, "min_severity": "info"},
            {"id": 21, "url": "https://b.com/h", "secret": "", "enabled": 1, "min_severity": "critical"},
            {"id": 22, "url": "https://c.com/h", "secret": "", "enabled": 0, "min_severity": "info"},
        ]
        mock_repo = MagicMock()
        payload = {"event": "violation", "severity": "warning"}
        fired_urls: list[str] = []
        lock = threading.Lock()

        def _fake_post(url, **kwargs):
            with lock:
                fired_urls.append(url)
            r = MagicMock()
            r.status_code = 200
            return r

        with patch("sopilot.vigil.webhook_dispatcher.httpx.post", side_effect=_fake_post):
            dispatcher.dispatch_violation(payload, webhooks, mock_repo)
            time.sleep(0.2)

        # Only https://a.com/h qualifies (info≤warning, enabled=1)
        # https://b.com/h: critical > warning → skip
        # https://c.com/h: disabled → skip
        assert "https://a.com/h" in fired_urls
        assert "https://b.com/h" not in fired_urls
        assert "https://c.com/h" not in fired_urls

    # ── test_webhook ──────────────────────────────────────────────────────────

    def test_test_webhook_success(self) -> None:
        dispatcher = WebhookDispatcher()
        wh = {"id": 30, "url": "https://example.com/hook", "secret": "", "enabled": 1}

        with patch("sopilot.vigil.webhook_dispatcher.httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            result = dispatcher.test_webhook(wh)

        assert result["ok"] is True
        assert result["status_code"] == 200
        assert result["error"] is None

    def test_test_webhook_failure_status(self) -> None:
        dispatcher = WebhookDispatcher()
        wh = {"id": 31, "url": "https://example.com/hook", "secret": "", "enabled": 1}

        with patch("sopilot.vigil.webhook_dispatcher.httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_post.return_value = mock_response

            result = dispatcher.test_webhook(wh)

        assert result["ok"] is False
        assert result["status_code"] == 403
        assert result["error"] is None

    def test_test_webhook_network_error(self) -> None:
        dispatcher = WebhookDispatcher()
        wh = {"id": 32, "url": "https://unreachable.invalid/", "secret": "", "enabled": 1}

        with patch(
            "sopilot.vigil.webhook_dispatcher.httpx.post",
            side_effect=Exception("connection refused"),
        ):
            result = dispatcher.test_webhook(wh)

        assert result["ok"] is False
        assert result["status_code"] is None
        assert "connection refused" in result["error"]


# ──────────────────────────────────────────────────────────────────────────────
# API Endpoint Tests (E2E via TestClient)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return _make_client(tmp_path)


class TestWebhookAPI:
    """End-to-end API tests for /vigil/webhooks endpoints."""

    def test_create_webhook_201(self, client: TestClient) -> None:
        r = client.post("/vigil/webhooks", json={
            "url": "https://hooks.slack.com/test",
            "name": "Slack",
            "min_severity": "critical",
        })
        assert r.status_code == 201
        data = r.json()
        assert data["url"] == "https://hooks.slack.com/test"
        assert data["name"] == "Slack"
        assert data["min_severity"] == "critical"
        assert data["enabled"] is True
        assert data["trigger_count"] == 0
        assert "id" in data
        assert "created_at" in data

    def test_create_webhook_defaults(self, client: TestClient) -> None:
        r = client.post("/vigil/webhooks", json={"url": "https://example.com/wh"})
        assert r.status_code == 201
        data = r.json()
        assert data["min_severity"] == "critical"
        assert data["name"] == ""

    def test_create_webhook_invalid_url_422(self, client: TestClient) -> None:
        r = client.post("/vigil/webhooks", json={"url": "ftp://bad-scheme.com"})
        assert r.status_code == 422

    def test_create_webhook_missing_url_422(self, client: TestClient) -> None:
        r = client.post("/vigil/webhooks", json={"name": "no-url"})
        assert r.status_code == 422

    def test_list_webhooks_empty(self, client: TestClient) -> None:
        r = client.get("/vigil/webhooks")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_webhooks_after_create(self, client: TestClient) -> None:
        client.post("/vigil/webhooks", json={"url": "https://a.com/wh"})
        client.post("/vigil/webhooks", json={"url": "https://b.com/wh"})
        r = client.get("/vigil/webhooks")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        urls = {wh["url"] for wh in data}
        assert urls == {"https://a.com/wh", "https://b.com/wh"}

    def test_get_webhook_by_id(self, client: TestClient) -> None:
        created = client.post("/vigil/webhooks", json={
            "url": "https://teams.microsoft.com/wh",
            "name": "Teams",
            "min_severity": "warning",
        }).json()
        wid = created["id"]
        r = client.get(f"/vigil/webhooks/{wid}")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == wid
        assert data["url"] == "https://teams.microsoft.com/wh"
        assert data["name"] == "Teams"

    def test_get_webhook_not_found_404(self, client: TestClient) -> None:
        r = client.get("/vigil/webhooks/9999")
        assert r.status_code == 404

    def test_delete_webhook_204(self, client: TestClient) -> None:
        created = client.post("/vigil/webhooks", json={"url": "https://del.example.com/wh"}).json()
        wid = created["id"]
        r = client.delete(f"/vigil/webhooks/{wid}")
        assert r.status_code == 204
        # Confirm it is gone
        r2 = client.get(f"/vigil/webhooks/{wid}")
        assert r2.status_code == 404

    def test_delete_webhook_not_found_404(self, client: TestClient) -> None:
        r = client.delete("/vigil/webhooks/9999")
        assert r.status_code == 404

    def test_enable_disable_webhook(self, client: TestClient) -> None:
        created = client.post("/vigil/webhooks", json={"url": "https://toggle.example.com/wh"}).json()
        wid = created["id"]
        # Disable
        r = client.patch(f"/vigil/webhooks/{wid}/enable", json={"enabled": False})
        assert r.status_code == 200
        assert r.json()["enabled"] is False
        # Re-enable
        r2 = client.patch(f"/vigil/webhooks/{wid}/enable", json={"enabled": True})
        assert r2.status_code == 200
        assert r2.json()["enabled"] is True

    def test_enable_webhook_not_found_404(self, client: TestClient) -> None:
        r = client.patch("/vigil/webhooks/9999/enable", json={"enabled": True})
        assert r.status_code == 404

    def test_test_webhook_success(self, client: TestClient) -> None:
        created = client.post("/vigil/webhooks", json={"url": "https://test.example.com/wh"}).json()
        wid = created["id"]

        with patch("sopilot.vigil.webhook_dispatcher.httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            r = client.post(f"/vigil/webhooks/{wid}/test")

        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["status_code"] == 200
        assert data["error"] is None

    def test_test_webhook_not_found_404(self, client: TestClient) -> None:
        r = client.post("/vigil/webhooks/9999/test")
        assert r.status_code == 404

    def test_test_webhook_network_error(self, client: TestClient) -> None:
        created = client.post("/vigil/webhooks", json={"url": "https://fail.invalid/wh"}).json()
        wid = created["id"]

        with patch(
            "sopilot.vigil.webhook_dispatcher.httpx.post",
            side_effect=Exception("connection refused"),
        ):
            r = client.post(f"/vigil/webhooks/{wid}/test")

        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is False
        assert data["error"] is not None

    def test_webhook_response_fields(self, client: TestClient) -> None:
        """Verify all WebhookResponse fields are present in the API response."""
        r = client.post("/vigil/webhooks", json={
            "url": "https://check.example.com/wh",
            "name": "FieldCheck",
            "min_severity": "info",
        })
        data = r.json()
        for field in ("id", "url", "name", "min_severity", "enabled", "created_at",
                      "last_triggered_at", "trigger_count"):
            assert field in data, f"Missing field: {field}"

    def test_trigger_count_starts_zero(self, client: TestClient) -> None:
        r = client.post("/vigil/webhooks", json={"url": "https://tc.example.com/wh"})
        assert r.json()["trigger_count"] == 0

    def test_create_with_all_severities(self, client: TestClient) -> None:
        for sev in ("info", "warning", "critical"):
            r = client.post("/vigil/webhooks", json={
                "url": f"https://{sev}.example.com/wh",
                "min_severity": sev,
            })
            assert r.status_code == 201
            assert r.json()["min_severity"] == sev

    def test_full_lifecycle(self, client: TestClient) -> None:
        """Create → list → get → disable → re-enable → test → delete."""
        # Create
        created = client.post("/vigil/webhooks", json={
            "url": "https://lifecycle.example.com/wh",
            "name": "Lifecycle",
            "min_severity": "warning",
        }).json()
        wid = created["id"]

        # List
        lst = client.get("/vigil/webhooks").json()
        assert any(w["id"] == wid for w in lst)

        # Get
        got = client.get(f"/vigil/webhooks/{wid}").json()
        assert got["name"] == "Lifecycle"

        # Disable
        dis = client.patch(f"/vigil/webhooks/{wid}/enable", json={"enabled": False}).json()
        assert dis["enabled"] is False

        # Re-enable
        ena = client.patch(f"/vigil/webhooks/{wid}/enable", json={"enabled": True}).json()
        assert ena["enabled"] is True

        # Test
        with patch("sopilot.vigil.webhook_dispatcher.httpx.post") as mp:
            mp.return_value.status_code = 204
            tr = client.post(f"/vigil/webhooks/{wid}/test").json()
        assert tr["ok"] is True

        # Delete
        client.delete(f"/vigil/webhooks/{wid}")
        assert client.get(f"/vigil/webhooks/{wid}").status_code == 404
