from __future__ import annotations

import base64
from pathlib import Path
import os
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient

from sopilot.api import (
    _has_required_role,
    _is_public_path,
    _normalize_role,
    _parse_basic,
    _parse_role_tokens,
    _resolve_identity,
    create_app,
)


def _base_env(data_dir: Path, **overrides) -> dict[str, str]:
    env = {
        "SOPILOT_DATA_DIR": str(data_dir),
        "SOPILOT_EMBEDDER_BACKEND": "heuristic",
        "SOPILOT_EMBEDDING_DEVICE": "auto",
        "SOPILOT_NIGHTLY_ENABLED": "0",
        "SOPILOT_QUEUE_BACKEND": "inline",
        "SOPILOT_SCORE_WORKERS": "1",
        "SOPILOT_TRAIN_WORKERS": "1",
        "SOPILOT_ENABLE_FEATURE_ADAPTER": "1",
        "SOPILOT_AUTH_REQUIRED": "0",
        "SOPILOT_MIN_SCORING_CLIPS": "1",
        "SOPILOT_UPLOAD_MAX_MB": "512",
        "SOPILOT_PRIVACY_MASK_ENABLED": "0",
        "SOPILOT_ADAPT_COMMAND": "",
        "SOPILOT_API_TOKEN": "",
        "SOPILOT_API_TOKEN_ROLE": "admin",
        "SOPILOT_API_ROLE_TOKENS": "",
        "SOPILOT_BASIC_USER": "",
        "SOPILOT_BASIC_PASSWORD": "",
        "SOPILOT_BASIC_ROLE": "admin",
        "SOPILOT_AUTH_DEFAULT_ROLE": "admin",
        "SOPILOT_AUDIT_SIGNING_KEY": "",
        "SOPILOT_AUDIT_SIGNING_KEY_ID": "local",
    }
    env.update(overrides)
    return env


class TestNormalizeRole:
    def test_valid_roles(self) -> None:
        assert _normalize_role("viewer") == "viewer"
        assert _normalize_role("operator") == "operator"
        assert _normalize_role("admin") == "admin"

    def test_case_insensitive(self) -> None:
        assert _normalize_role("ADMIN") == "admin"
        assert _normalize_role("Operator") == "operator"

    def test_strips_whitespace(self) -> None:
        assert _normalize_role("  admin  ") == "admin"

    def test_unknown_role_returns_default(self) -> None:
        assert _normalize_role("superuser") == "viewer"
        assert _normalize_role("superuser", default="admin") == "admin"


class TestHasRequiredRole:
    def test_admin_has_all_roles(self) -> None:
        assert _has_required_role("admin", "viewer") is True
        assert _has_required_role("admin", "operator") is True
        assert _has_required_role("admin", "admin") is True

    def test_operator_has_operator_and_viewer(self) -> None:
        assert _has_required_role("operator", "viewer") is True
        assert _has_required_role("operator", "operator") is True
        assert _has_required_role("operator", "admin") is False

    def test_viewer_only_has_viewer(self) -> None:
        assert _has_required_role("viewer", "viewer") is True
        assert _has_required_role("viewer", "operator") is False
        assert _has_required_role("viewer", "admin") is False


class TestParseBasic:
    def test_valid_basic_auth(self) -> None:
        encoded = base64.b64encode(b"user:pass").decode()
        result = _parse_basic(f"Basic {encoded}")
        assert result == ("user", "pass")

    def test_password_with_colon(self) -> None:
        encoded = base64.b64encode(b"user:pass:word").decode()
        result = _parse_basic(f"Basic {encoded}")
        assert result == ("user", "pass:word")

    def test_not_basic_prefix(self) -> None:
        assert _parse_basic("Bearer token") is None

    def test_empty_payload(self) -> None:
        assert _parse_basic("Basic ") is None

    def test_invalid_base64(self) -> None:
        assert _parse_basic("Basic not-valid-base64!!!") is None


class TestParseRoleTokens:
    def test_empty_spec(self) -> None:
        assert _parse_role_tokens("") == []

    def test_single_role_token(self) -> None:
        result = _parse_role_tokens("admin:secret123")
        assert result == [("admin", "secret123")]

    def test_multiple_role_tokens(self) -> None:
        result = _parse_role_tokens("viewer:v-tok,operator:o-tok,admin:a-tok")
        assert len(result) == 3
        assert ("viewer", "v-tok") in result
        assert ("operator", "o-tok") in result
        assert ("admin", "a-tok") in result

    def test_semicolon_separator(self) -> None:
        result = _parse_role_tokens("viewer:tok1;admin:tok2")
        assert len(result) == 2

    def test_invalid_entries_skipped(self) -> None:
        result = _parse_role_tokens("admin:valid,nocolon,badrole:,,:empty")
        assert len(result) == 1
        assert result[0] == ("admin", "valid")


class TestIsPublicPath:
    def test_public_paths(self) -> None:
        assert _is_public_path("/") is True
        assert _is_public_path("/health") is True
        assert _is_public_path("/ui") is True
        assert _is_public_path("/docs") is True

    def test_public_paths_with_trailing_slash(self) -> None:
        assert _is_public_path("/health/") is True
        assert _is_public_path("/ui/") is True

    def test_non_public_paths(self) -> None:
        assert _is_public_path("/videos") is False
        assert _is_public_path("/score") is False
        assert _is_public_path("/admin") is False


class TestResolveIdentity:
    def test_no_auth_configured_returns_anonymous(self) -> None:
        result = _resolve_identity(
            "",
            api_token="",
            api_token_role="admin",
            api_role_tokens=[],
            basic_user="",
            basic_password="",
            basic_role="admin",
            auth_default_role="admin",
        )
        assert result is not None
        assert result[0] == "anonymous"

    def test_bearer_token_match(self) -> None:
        result = _resolve_identity(
            "Bearer my-secret",
            api_token="my-secret",
            api_token_role="admin",
            api_role_tokens=[],
            basic_user="",
            basic_password="",
            basic_role="admin",
            auth_default_role="admin",
        )
        assert result is not None
        assert result[0] == "token:api"
        assert result[1] == "admin"

    def test_role_token_takes_priority(self) -> None:
        result = _resolve_identity(
            "Bearer op-token",
            api_token="api-token",
            api_token_role="admin",
            api_role_tokens=[("operator", "op-token")],
            basic_user="",
            basic_password="",
            basic_role="admin",
            auth_default_role="admin",
        )
        assert result is not None
        assert result[0] == "token:operator"
        assert result[1] == "operator"

    def test_basic_auth_match(self) -> None:
        encoded = base64.b64encode(b"admin:password").decode()
        result = _resolve_identity(
            f"Basic {encoded}",
            api_token="",
            api_token_role="admin",
            api_role_tokens=[],
            basic_user="admin",
            basic_password="password",
            basic_role="admin",
            auth_default_role="admin",
        )
        assert result is not None
        assert result[0] == "basic:admin"
        assert result[1] == "admin"

    def test_wrong_credentials_returns_none(self) -> None:
        result = _resolve_identity(
            "Bearer wrong-token",
            api_token="my-secret",
            api_token_role="admin",
            api_role_tokens=[],
            basic_user="",
            basic_password="",
            basic_role="admin",
            auth_default_role="admin",
        )
        assert result is None


class TestBasicAuthIntegration:
    def test_basic_auth_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = _base_env(
                root / "data",
                SOPILOT_BASIC_USER="testuser",
                SOPILOT_BASIC_PASSWORD="testpass",
                SOPILOT_BASIC_ROLE="admin",
            )
            with patch.dict(os.environ, env):
                app = create_app()

                encoded = base64.b64encode(b"testuser:testpass").decode()
                headers = {"Authorization": f"Basic {encoded}"}

                with TestClient(app) as client:
                    # Without auth - should be 401
                    blocked = client.get("/videos")
                    assert blocked.status_code == 401

                    # With Basic auth - should work
                    ok = client.get("/videos", headers=headers)
                    assert ok.status_code == 200

    def test_wrong_basic_credentials_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = _base_env(
                root / "data",
                SOPILOT_BASIC_USER="testuser",
                SOPILOT_BASIC_PASSWORD="testpass",
            )
            with patch.dict(os.environ, env):
                app = create_app()

                encoded = base64.b64encode(b"testuser:wrongpass").decode()
                headers = {"Authorization": f"Basic {encoded}"}

                with TestClient(app) as client:
                    res = client.get("/videos", headers=headers)
                    assert res.status_code == 401


class TestDefaultRoleWhenNoAuth:
    def test_default_role_viewer_restricts_writes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = _base_env(
                root / "data",
                SOPILOT_AUTH_REQUIRED="0",
                SOPILOT_AUTH_DEFAULT_ROLE="viewer",
            )
            with patch.dict(os.environ, env):
                app = create_app()

                with TestClient(app) as client:
                    # GET /videos should work for viewer
                    ok = client.get("/videos")
                    assert ok.status_code == 200

                    # POST /train/nightly requires admin -> should fail
                    blocked = client.post("/train/nightly")
                    assert blocked.status_code == 403
