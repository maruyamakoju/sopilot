"""API edge case tests — search endpoint, metrics, query param boundaries, error mapping."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from sopilot.api import (
    _has_required_role,
    _is_public_path,
    _map_service_error,
    _normalize_role,
    _parse_basic,
    _parse_role_tokens,
    create_app,
)


def _base_env(data_dir: Path, **overrides) -> dict[str, str]:
    env = {
        "SOPILOT_DATA_DIR": str(data_dir),
        "SOPILOT_EMBEDDER_BACKEND": "heuristic",
        "SOPILOT_EMBEDDING_DEVICE": "auto",
        "SOPILOT_VJEPA2_REPO": "facebookresearch/vjepa2",
        "SOPILOT_VJEPA2_VARIANT": "vjepa2_vit_large",
        "SOPILOT_NIGHTLY_ENABLED": "0",
        "SOPILOT_QUEUE_BACKEND": "inline",
        "SOPILOT_SCORE_WORKERS": "1",
        "SOPILOT_TRAIN_WORKERS": "1",
        "SOPILOT_ENABLE_FEATURE_ADAPTER": "1",
        "SOPILOT_AUTH_REQUIRED": "0",
        "SOPILOT_MIN_SCORING_CLIPS": "1",
        "SOPILOT_UPLOAD_MAX_MB": "512",
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
        "SOPILOT_PRIVACY_MASK_ENABLED": "",
        "SOPILOT_PRIVACY_MASK_MODE": "",
        "SOPILOT_PRIVACY_MASK_RECTS": "",
        "SOPILOT_PRIVACY_FACE_BLUR": "",
    }
    env.update(overrides)
    return env


@pytest.fixture()
def _app_client(tmp_path):
    """Create a test client with a fully initialized app."""
    env = _base_env(tmp_path)
    with patch.dict(os.environ, env, clear=False):
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client
        # Lifespan exit calls service.shutdown() which closes DB


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestNormalizeRole:
    def test_known_role(self):
        assert _normalize_role("admin") == "admin"
        assert _normalize_role("viewer") == "viewer"
        assert _normalize_role("operator") == "operator"

    def test_case_insensitive(self):
        assert _normalize_role("ADMIN") == "admin"
        assert _normalize_role("  Operator  ") == "operator"

    def test_unknown_role_returns_default(self):
        assert _normalize_role("superuser") == "viewer"
        assert _normalize_role("superuser", default="admin") == "admin"

    def test_empty_string(self):
        assert _normalize_role("") == "viewer"


class TestHasRequiredRole:
    def test_admin_has_all_roles(self):
        assert _has_required_role("admin", "admin")
        assert _has_required_role("admin", "operator")
        assert _has_required_role("admin", "viewer")

    def test_viewer_only_viewer(self):
        assert _has_required_role("viewer", "viewer")
        assert not _has_required_role("viewer", "operator")
        assert not _has_required_role("viewer", "admin")

    def test_operator_mid_level(self):
        assert _has_required_role("operator", "operator")
        assert _has_required_role("operator", "viewer")
        assert not _has_required_role("operator", "admin")


class TestIsPublicPath:
    def test_known_public_paths(self):
        assert _is_public_path("/")
        assert _is_public_path("/health")
        assert _is_public_path("/metrics")
        assert _is_public_path("/ui")
        assert _is_public_path("/docs")

    def test_trailing_slash(self):
        assert _is_public_path("/health/")
        assert _is_public_path("/metrics/")

    def test_non_public_paths(self):
        assert not _is_public_path("/videos")
        assert not _is_public_path("/score")
        assert not _is_public_path("/train/nightly")


class TestParseBasic:
    def test_valid_basic_auth(self):
        import base64

        creds = base64.b64encode(b"user:pass").decode()
        result = _parse_basic(f"Basic {creds}")
        assert result == ("user", "pass")

    def test_not_basic_prefix(self):
        assert _parse_basic("Bearer token123") is None

    def test_empty_payload(self):
        assert _parse_basic("Basic ") is None

    def test_invalid_base64(self):
        assert _parse_basic("Basic !!!invalid!!!") is None

    def test_no_colon_separator(self):
        import base64

        creds = base64.b64encode(b"useronly").decode()
        assert _parse_basic(f"Basic {creds}") is None


class TestParseRoleTokens:
    def test_valid_tokens(self):
        result = _parse_role_tokens("viewer:v-tok,operator:o-tok,admin:a-tok")
        assert len(result) == 3
        assert ("viewer", "v-tok") in result
        assert ("operator", "o-tok") in result
        assert ("admin", "a-tok") in result

    def test_semicolon_separator(self):
        result = _parse_role_tokens("viewer:v-tok;admin:a-tok")
        assert len(result) == 2

    def test_empty_string(self):
        assert _parse_role_tokens("") == []

    def test_invalid_entries_skipped(self):
        result = _parse_role_tokens("viewer:v-tok,,badentry,admin:a-tok")
        assert len(result) == 2

    def test_unknown_role_skipped(self):
        result = _parse_role_tokens("superuser:tok1,admin:tok2")
        assert len(result) == 1
        assert result[0] == ("admin", "tok2")


class TestMapServiceError:
    def test_value_error_maps_to_400(self):
        exc = ValueError("bad input")
        http_exc = _map_service_error(exc)
        assert http_exc.status_code == 400
        assert "bad input" in http_exc.detail

    def test_runtime_error_maps_to_500(self):
        exc = RuntimeError("something broke")
        http_exc = _map_service_error(exc)
        assert http_exc.status_code == 500

    def test_custom_status_codes(self):
        exc = ValueError("not found")
        http_exc = _map_service_error(exc, value_error_status=404)
        assert http_exc.status_code == 404

    def test_generic_exception(self):
        exc = TypeError("unexpected")
        http_exc = _map_service_error(exc)
        assert http_exc.status_code == 500
        assert http_exc.detail == "internal server error"

    def test_http_exception_passthrough(self):
        from fastapi import HTTPException

        original = HTTPException(status_code=418, detail="teapot")
        result = _map_service_error(original)
        assert result.status_code == 418


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    def test_root_returns_200(self, _app_client):
        resp = _app_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data


class TestHealthEndpoint:
    def test_health_returns_200(self, _app_client):
        resp = _app_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in {"ok", "degraded"}
        assert "db" in data


class TestMetricsEndpoint:
    def test_metrics_returns_response(self, _app_client):
        resp = _app_client.get("/metrics")
        # Either 200 (prometheus available) or 503 (not installed)
        assert resp.status_code in {200, 503}


class TestSearchEndpoint:
    def test_search_missing_params_returns_422(self, _app_client):
        """Search without required query params should return 422."""
        resp = _app_client.get("/search")
        assert resp.status_code == 422

    def test_search_partial_params_returns_422(self, _app_client):
        """Search with only some params should return 422."""
        resp = _app_client.get("/search?task_id=t1")
        assert resp.status_code == 422

    def test_search_k_out_of_range_high(self, _app_client):
        """k > 50 should return 422."""
        resp = _app_client.get("/search?task_id=t1&video_id=1&clip_idx=0&k=51")
        assert resp.status_code == 422

    def test_search_k_out_of_range_low(self, _app_client):
        """k < 1 should return 422."""
        resp = _app_client.get("/search?task_id=t1&video_id=1&clip_idx=0&k=0")
        assert resp.status_code == 422

    def test_search_k_boundary_valid(self, _app_client):
        """k=1 and k=50 should be valid (may fail on data but not validation)."""
        resp = _app_client.get("/search?task_id=t1&video_id=1&clip_idx=0&k=1")
        # Should not be 422 — might be 400/500 from service but not validation error
        assert resp.status_code != 422

    def test_search_invalid_video_id_type(self, _app_client):
        """Non-integer video_id should return 422."""
        resp = _app_client.get("/search?task_id=t1&video_id=abc&clip_idx=0&k=5")
        assert resp.status_code == 422


class TestVideoListBoundary:
    def test_list_videos_limit_too_high(self, _app_client):
        resp = _app_client.get("/videos?limit=501")
        assert resp.status_code == 422

    def test_list_videos_limit_too_low(self, _app_client):
        resp = _app_client.get("/videos?limit=0")
        assert resp.status_code == 422

    def test_list_videos_limit_boundary_valid(self, _app_client):
        resp = _app_client.get("/videos?limit=1")
        assert resp.status_code == 200
        resp = _app_client.get("/videos?limit=500")
        assert resp.status_code == 200


class TestAuditTrailBoundary:
    def test_audit_trail_limit_too_high(self, _app_client):
        resp = _app_client.get("/audit/trail?limit=1001")
        assert resp.status_code == 422

    def test_audit_trail_limit_too_low(self, _app_client):
        resp = _app_client.get("/audit/trail?limit=0")
        assert resp.status_code == 422

    def test_audit_trail_default(self, _app_client):
        resp = _app_client.get("/audit/trail")
        assert resp.status_code == 200


class TestNotFoundEndpoints:
    def test_ingest_job_not_found(self, _app_client):
        resp = _app_client.get("/videos/jobs/nonexistent-job-id")
        assert resp.status_code == 404

    def test_score_job_not_found(self, _app_client):
        resp = _app_client.get("/score/nonexistent-job-id")
        assert resp.status_code == 404

    def test_training_job_not_found(self, _app_client):
        resp = _app_client.get("/train/jobs/nonexistent-job-id")
        assert resp.status_code == 404

    def test_video_not_found(self, _app_client):
        resp = _app_client.get("/videos/99999")
        assert resp.status_code == 404

    def test_video_file_not_found(self, _app_client):
        resp = _app_client.get("/videos/99999/file")
        # Should be 404 (ValueError mapped to 404)
        assert resp.status_code in {404, 500}
