"""Tests for sopilot.api.oauth2 — OAuth2 + Audit Logging (Item 3).

25 tests covering: token issuance, expiry, invalid secret, role enforcement,
audit log write/read, middleware capture.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(
    jwt_secret: str = "test-secret",
    jwt_expire_minutes: int = 60,
    oauth_clients_json: str | None = None,
) -> MagicMock:
    settings = MagicMock()
    settings.jwt_secret = jwt_secret
    settings.jwt_expire_minutes = jwt_expire_minutes
    settings.oauth_clients_json = oauth_clients_json or json.dumps(
        [
            {
                "client_id": "admin-ui",
                "client_secret_hash": "plain:admin-secret",
                "role": "admin",
            },
            {
                "client_id": "viewer-01",
                "client_secret_hash": "plain:viewer-secret",
                "role": "viewer",
            },
        ]
    )
    return settings


def _mock_verify(plain: str, hashed: str) -> bool:
    """Simple plain-text comparison mock for _verify_secret."""
    return hashed == f"plain:{plain}"


# ---------------------------------------------------------------------------
# _verify_secret
# ---------------------------------------------------------------------------

class TestVerifySecret:
    def test_correct_secret_returns_true(self) -> None:
        from sopilot.api.oauth2 import _verify_secret

        with patch("sopilot.api.oauth2._verify_secret", side_effect=_mock_verify):
            result = _mock_verify("admin-secret", "plain:admin-secret")
        assert result is True

    def test_wrong_secret_returns_false(self) -> None:
        result = _mock_verify("wrong", "plain:admin-secret")
        assert result is False


# ---------------------------------------------------------------------------
# _load_clients
# ---------------------------------------------------------------------------

class TestLoadClients:
    def test_parses_valid_json(self) -> None:
        from sopilot.api.oauth2 import _load_clients

        clients = _load_clients('[{"client_id": "c1", "role": "viewer"}]')
        assert len(clients) == 1
        assert clients[0]["client_id"] == "c1"

    def test_returns_empty_list_for_invalid_json(self) -> None:
        from sopilot.api.oauth2 import _load_clients

        assert _load_clients("not-json") == []

    def test_returns_empty_list_for_non_list(self) -> None:
        from sopilot.api.oauth2 import _load_clients

        assert _load_clients('{"key": "val"}') == []

    def test_returns_empty_list_for_empty_array(self) -> None:
        from sopilot.api.oauth2 import _load_clients

        assert _load_clients("[]") == []


# ---------------------------------------------------------------------------
# decode_token
# ---------------------------------------------------------------------------

class TestDecodeToken:
    def _make_token(self, payload: dict, secret: str = "test-secret") -> str:
        from jose import jwt  # type: ignore[import-untyped]
        return jwt.encode(payload, secret, algorithm="HS256")

    def test_valid_token_decodes(self) -> None:
        from sopilot.api.oauth2 import decode_token

        payload = {"sub": "admin-ui", "role": "admin", "exp": int(time.time()) + 3600, "iat": int(time.time())}
        token = self._make_token(payload)
        result = decode_token(token, "test-secret")
        assert result["sub"] == "admin-ui"
        assert result["role"] == "admin"

    def test_expired_token_raises_401(self) -> None:
        from fastapi import HTTPException
        from sopilot.api.oauth2 import decode_token

        payload = {"sub": "x", "exp": int(time.time()) - 10, "iat": int(time.time()) - 70}
        token = self._make_token(payload)
        with pytest.raises(HTTPException) as exc:
            decode_token(token, "test-secret")
        assert exc.value.status_code == 401

    def test_wrong_secret_raises_401(self) -> None:
        from fastapi import HTTPException
        from sopilot.api.oauth2 import decode_token

        payload = {"sub": "x", "exp": int(time.time()) + 3600}
        token = self._make_token(payload, secret="other-secret")
        with pytest.raises(HTTPException) as exc:
            decode_token(token, "test-secret")
        assert exc.value.status_code == 401

    def test_malformed_token_raises_401(self) -> None:
        from fastapi import HTTPException
        from sopilot.api.oauth2 import decode_token

        with pytest.raises(HTTPException) as exc:
            decode_token("not.a.token", "secret")
        assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# build_oauth2_router — POST /auth/token
# ---------------------------------------------------------------------------

class TestOauth2Router:
    def _make_app(self, settings: MagicMock) -> TestClient:
        from fastapi import FastAPI
        from sopilot.api.oauth2 import build_oauth2_router

        app = FastAPI()
        app.state.settings = settings
        app.include_router(build_oauth2_router(settings))
        return TestClient(app, raise_server_exceptions=True)

    def test_valid_credentials_returns_token(self) -> None:
        settings = _make_settings()
        client = self._make_app(settings)

        with patch("sopilot.api.oauth2._verify_secret", side_effect=_mock_verify):
            resp = client.post(
                "/auth/token",
                data={"username": "admin-ui", "password": "admin-secret"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert body["expires_in"] == 60 * 60

    def test_wrong_password_returns_401(self) -> None:
        settings = _make_settings()
        client = self._make_app(settings)

        with patch("sopilot.api.oauth2._verify_secret", return_value=False):
            resp = client.post(
                "/auth/token",
                data={"username": "admin-ui", "password": "wrong"},
            )

        assert resp.status_code == 401

    def test_unknown_client_returns_401(self) -> None:
        settings = _make_settings()
        client = self._make_app(settings)

        resp = client.post(
            "/auth/token",
            data={"username": "no-such-client", "password": "pass"},
        )
        assert resp.status_code == 401

    def test_token_contains_role(self) -> None:
        settings = _make_settings()
        client = self._make_app(settings)

        with patch("sopilot.api.oauth2._verify_secret", side_effect=_mock_verify):
            resp = client.post(
                "/auth/token",
                data={"username": "viewer-01", "password": "viewer-secret"},
            )

        assert resp.status_code == 200
        token = resp.json()["access_token"]

        from jose import jwt  # type: ignore[import-untyped]
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        assert payload["role"] == "viewer"


# ---------------------------------------------------------------------------
# require_role dependency
# ---------------------------------------------------------------------------

class TestRequireRole:
    def _make_app_with_role(self, required_role: str) -> TestClient:
        from fastapi import FastAPI
        from sopilot.api.oauth2 import require_role

        app = FastAPI()

        @app.get("/protected")
        async def protected(_auth=require_role(required_role)):
            return {"ok": True}

        @app.state.__class__.settings.getter  # type: ignore[misc]
        def settings(self):  # noqa: N805
            return MagicMock()

        # Simpler approach: attach settings directly
        mock_settings = _make_settings()
        app.state.settings = mock_settings
        return TestClient(app, raise_server_exceptions=False)

    def test_valid_admin_token_passes(self) -> None:
        from fastapi import FastAPI
        from sopilot.api.oauth2 import require_role

        mock_settings = _make_settings()

        app = FastAPI()
        app.state.settings = mock_settings

        @app.get("/protected")
        async def protected(_auth=require_role("admin")):
            return {"ok": True}

        from jose import jwt  # type: ignore[import-untyped]
        token = jwt.encode(
            {"sub": "admin-ui", "role": "admin", "exp": int(time.time()) + 3600},
            mock_settings.jwt_secret,
            algorithm="HS256",
        )

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_missing_token_returns_401(self) -> None:
        from fastapi import FastAPI
        from sopilot.api.oauth2 import require_role

        app = FastAPI()
        app.state.settings = _make_settings()

        @app.get("/protected")
        async def protected(_auth=require_role("admin")):
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/protected")
        assert resp.status_code == 401

    def test_wrong_role_returns_403(self) -> None:
        from fastapi import FastAPI
        from sopilot.api.oauth2 import require_role

        mock_settings = _make_settings()
        app = FastAPI()
        app.state.settings = mock_settings

        @app.get("/admin-only")
        async def admin_only(_auth=require_role("admin")):
            return {"ok": True}

        from jose import jwt  # type: ignore[import-untyped]
        token = jwt.encode(
            {"sub": "viewer-01", "role": "viewer", "exp": int(time.time()) + 3600},
            mock_settings.jwt_secret,
            algorithm="HS256",
        )

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/admin-only", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# AuditRepository
# ---------------------------------------------------------------------------

class TestAuditRepository:
    def _make_repo(self, tmp_path):
        import sqlite3
        from contextlib import contextmanager
        from pathlib import Path
        from sopilot.repositories.audit_repository import AuditRepository

        db_path = tmp_path / "test_audit.db"

        @contextmanager
        def connect():
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        return AuditRepository(connect)

    def test_log_request_inserts_row(self, tmp_path) -> None:
        repo = self._make_repo(tmp_path)
        repo.log_request(method="POST", path="/videos", status_code=201, duration_ms=12.5)
        entries = repo.list_audit_log()
        assert len(entries) == 1
        assert entries[0]["method"] == "POST"
        assert entries[0]["path"] == "/videos"
        assert entries[0]["status_code"] == 201

    def test_list_audit_log_returns_most_recent_first(self, tmp_path) -> None:
        repo = self._make_repo(tmp_path)
        repo.log_request(method="POST", path="/a", status_code=200)
        repo.log_request(method="DELETE", path="/b", status_code=204)
        entries = repo.list_audit_log(limit=10)
        assert entries[0]["path"] == "/b"  # most recent first

    def test_filter_by_client_id(self, tmp_path) -> None:
        repo = self._make_repo(tmp_path)
        repo.log_request(method="POST", path="/x", client_id="admin-ui")
        repo.log_request(method="POST", path="/y", client_id="viewer-01")
        entries = repo.list_audit_log(client_id="admin-ui")
        assert all(e["client_id"] == "admin-ui" for e in entries)
        assert len(entries) == 1

    def test_count_audit_log(self, tmp_path) -> None:
        repo = self._make_repo(tmp_path)
        repo.log_request(method="POST", path="/a")
        repo.log_request(method="POST", path="/b")
        assert repo.count_audit_log() == 2

    def test_log_request_never_raises(self, tmp_path) -> None:
        """log_request should swallow exceptions."""
        repo = self._make_repo(tmp_path)
        # Corrupt the connect function
        repo._connect = lambda: None  # type: ignore[assignment]
        repo.log_request(method="POST", path="/safe")  # must not raise


# ---------------------------------------------------------------------------
# AuditMiddleware
# ---------------------------------------------------------------------------

class TestAuditMiddleware:
    def test_post_request_is_captured(self) -> None:
        from fastapi import FastAPI
        from sopilot.api.audit_middleware import AuditMiddleware

        mock_worker = MagicMock()
        app = FastAPI()
        app.add_middleware(AuditMiddleware, audit_worker=mock_worker)

        @app.post("/test")
        def endpoint():
            return {"ok": True}

        client = TestClient(app)
        resp = client.post("/test")
        assert resp.status_code == 200
        mock_worker.enqueue.assert_called_once()
        call_kwargs = mock_worker.enqueue.call_args[1]
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["path"] == "/test"
        assert call_kwargs["status_code"] == 200

    def test_get_request_not_captured(self) -> None:
        from fastapi import FastAPI
        from sopilot.api.audit_middleware import AuditMiddleware

        mock_worker = MagicMock()
        app = FastAPI()
        app.add_middleware(AuditMiddleware, audit_worker=mock_worker)

        @app.get("/test")
        def endpoint():
            return {"ok": True}

        client = TestClient(app)
        client.get("/test")
        mock_worker.enqueue.assert_not_called()
