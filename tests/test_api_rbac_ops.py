from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from sopilot.api import create_app


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


def test_rbac_roles_enforce_sensitive_endpoints() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        env = _base_env(
            root / "data",
            SOPILOT_API_ROLE_TOKENS="viewer:view-token,operator:op-token,admin:admin-token",
        )
        with patch.dict(os.environ, env):
            app = create_app()

            with TestClient(app) as client:
                viewer = {"Authorization": "Bearer view-token"}
                operator = {"Authorization": "Bearer op-token"}
                admin = {"Authorization": "Bearer admin-token"}

                read_ok = client.get("/videos", headers=viewer)
                assert read_ok.status_code == 200, read_ok.text

                train_forbidden = client.post("/train/nightly", headers=viewer)
                assert train_forbidden.status_code == 403, train_forbidden.text

                delete_forbidden = client.delete("/videos/123", headers=operator)
                assert delete_forbidden.status_code == 403, delete_forbidden.text

                queue_ok = client.get("/ops/queue", headers=operator)
                assert queue_ok.status_code == 200, queue_ok.text

                delete_as_admin = client.delete("/videos/123", headers=admin)
                assert delete_as_admin.status_code == 404, delete_as_admin.text


def test_signed_audit_export_contains_valid_hmac_signature() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        env = _base_env(
            root / "data",
            SOPILOT_API_TOKEN="audit-admin-token",
            SOPILOT_API_TOKEN_ROLE="admin",
            SOPILOT_AUDIT_SIGNING_KEY="test-signing-secret",
            SOPILOT_AUDIT_SIGNING_KEY_ID="test-key-1",
        )
        with patch.dict(os.environ, env):
            app = create_app()

            headers = {"Authorization": "Bearer audit-admin-token"}
            with TestClient(app) as client:
                exported = client.post("/audit/export?limit=20", headers=headers)
                assert exported.status_code == 200, exported.text
                payload = exported.json()
                assert payload["item_count"] >= 0
                assert payload["signature"]["key_id"] == "test-key-1"

                file_res = client.get(f"/audit/export/{payload['export_id']}/file", headers=headers)
                assert file_res.status_code == 200, file_res.text
                signed_payload = json.loads(file_res.text)

                signature = signed_payload.get("signature")
                assert isinstance(signature, dict)
                unsigned = dict(signed_payload)
                unsigned.pop("signature", None)
                canonical = json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                expected_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                expected_sig = hmac.new(
                    b"test-signing-secret",
                    canonical.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                assert signature["payload_sha256"] == expected_sha
                assert signature["signature_hex"] == expected_sig


def test_queue_metrics_endpoint_returns_inline_stats() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with patch.dict(os.environ, _base_env(root / "data")):
            app = create_app()

            with TestClient(app) as client:
                res = client.get("/ops/queue")
                assert res.status_code == 200, res.text
                payload = res.json()
                assert payload["queue"]["backend"] == "inline"
                assert payload["runtime_mode"] == "api"
                assert set(payload["jobs"].keys()) == {"ingest", "score", "training"}
