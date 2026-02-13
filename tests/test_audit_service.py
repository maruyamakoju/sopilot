"""Tests for audit_service.py (P0-1).

Covers:
- Audit trail retrieval (delegation to db.list_audit_trail)
- Canonical JSON serialization (deterministic, sorted)
- Signed export creation (HMAC-SHA256)
- Signature verification (round-trip integrity)
- Export path resolution (safe ID, missing ID, path traversal)
- Error on missing signing key
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest

from conftest import make_test_settings
from sopilot.audit_service import AuditService
from sopilot.db import Database


@pytest.fixture()
def audit_env(tmp_path):
    """Create a fresh Database + AuditService with a signing key."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    settings = make_test_settings(
        data_dir=tmp_path,
        reports_dir=tmp_path / "reports",
        audit_signing_key="test-secret-key-1234",
        audit_signing_key_id="test-key-v1",
    )
    (tmp_path / "reports").mkdir(exist_ok=True)
    service = AuditService(settings, db)
    return service, db, settings


class TestGetAuditTrail:
    """Test audit trail retrieval."""

    def test_empty_trail(self, audit_env):
        service, db, _ = audit_env
        trail = service.get_audit_trail()
        assert trail == []

    def test_trail_with_ingest_job(self, audit_env):
        service, db, _ = audit_env
        job_id = db.create_ingest_job(
            task_id="t1",
            role="trainee",
            requested_by="user@test",
            file_name="video.mp4",
            file_path="/tmp/video.mp4",
            site_id=None,
            camera_id=None,
            operator_id_hash=None,
        )
        trail = service.get_audit_trail(limit=10)
        assert len(trail) == 1
        assert trail[0]["job_id"] == job_id
        assert trail[0]["job_type"] == "ingest"
        assert trail[0]["status"] == "queued"

    def test_trail_with_multiple_job_types(self, audit_env):
        service, db, _ = audit_env
        # Create one of each job type
        from sopilot.db import VideoCreateInput

        vid = db.create_video(VideoCreateInput(
            task_id="t1", role="gold", file_path="/tmp/v.mp4", embedding_model="heuristic",
        ))
        db.create_ingest_job(
            task_id="t1", role="trainee", requested_by=None,
            file_name="v.mp4", file_path="/tmp/v.mp4",
            site_id=None, camera_id=None, operator_id_hash=None,
        )
        db.create_score_job(vid, vid, requested_by="admin")
        db.create_training_job("nightly", requested_by=None)

        trail = service.get_audit_trail(limit=100)
        job_types = {item["job_type"] for item in trail}
        assert "ingest" in job_types
        assert "score" in job_types
        assert "training" in job_types

    def test_trail_limit(self, audit_env):
        service, db, _ = audit_env
        for i in range(5):
            db.create_training_job(f"trigger_{i}", requested_by=None)
        trail = service.get_audit_trail(limit=3)
        assert len(trail) == 3


class TestCanonicalJson:
    """Test deterministic JSON serialization."""

    def test_sorted_keys(self):
        payload = {"z": 1, "a": 2, "m": 3}
        result = AuditService._canonical_json(payload)
        parsed = json.loads(result)
        assert list(parsed.keys()) == ["a", "m", "z"]

    def test_no_whitespace(self):
        payload = {"key": "value", "num": 42}
        result = AuditService._canonical_json(payload)
        assert " " not in result
        assert "\n" not in result

    def test_ascii_only(self):
        payload = {"emoji": "テスト"}
        result = AuditService._canonical_json(payload)
        assert "\\u" in result  # Non-ASCII escaped

    def test_deterministic(self):
        payload = {"b": [1, 2, 3], "a": {"x": 1, "y": 2}}
        r1 = AuditService._canonical_json(payload)
        r2 = AuditService._canonical_json(payload)
        assert r1 == r2


class TestExportSignedAuditTrail:
    """Test HMAC-SHA256 signed export creation."""

    def test_signed_export_structure(self, audit_env):
        service, db, _ = audit_env
        db.create_training_job("test", requested_by="admin")

        result = service.export_signed_audit_trail(limit=10)

        assert "export_id" in result
        assert "generated_at" in result
        assert "item_count" in result
        assert result["item_count"] == 1
        assert "file_path" in result
        assert "signature" in result
        sig = result["signature"]
        assert sig["algorithm"] == "hmac-sha256"
        assert sig["key_id"] == "test-key-v1"
        assert "payload_sha256" in sig
        assert "signature_hex" in sig

    def test_exported_file_exists(self, audit_env):
        service, db, _ = audit_env
        result = service.export_signed_audit_trail()
        path = Path(result["file_path"])
        assert path.exists()

    def test_exported_file_is_valid_json(self, audit_env):
        service, db, _ = audit_env
        result = service.export_signed_audit_trail()
        with open(result["file_path"], encoding="utf-8") as f:
            data = json.load(f)
        assert "signature" in data
        assert "items" in data

    def test_signature_verification_roundtrip(self, audit_env):
        service, db, settings = audit_env
        db.create_ingest_job(
            task_id="t1", role="trainee", requested_by="user",
            file_name="v.mp4", file_path="/tmp/v.mp4",
            site_id=None, camera_id=None, operator_id_hash=None,
        )
        result = service.export_signed_audit_trail()

        # Read the exported file and verify signature
        with open(result["file_path"], encoding="utf-8") as f:
            signed_data = json.load(f)

        sig_block = signed_data.pop("signature")
        canonical = AuditService._canonical_json(signed_data)
        payload_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        expected_sig = hmac.new(
            settings.audit_signing_key.encode("utf-8"),
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        assert sig_block["payload_sha256"] == payload_sha
        assert sig_block["signature_hex"] == expected_sig

    def test_no_signing_key_raises(self, tmp_path):
        db = Database(tmp_path / "test.db")
        settings = make_test_settings(
            data_dir=tmp_path,
            reports_dir=tmp_path / "reports",
            audit_signing_key="",
        )
        service = AuditService(settings, db)
        with pytest.raises(RuntimeError, match="signing key"):
            service.export_signed_audit_trail()


class TestGetAuditExportPath:
    """Test export path resolution and sanitization."""

    def test_valid_export_id(self, audit_env):
        service, db, _ = audit_env
        # Create an export first
        result = service.export_signed_audit_trail()
        export_id = result["export_id"]
        path = service.get_audit_export_path(export_id)
        assert path.exists()

    def test_invalid_export_id_empty(self, audit_env):
        service, _, _ = audit_env
        with pytest.raises(ValueError, match="invalid export_id"):
            service.get_audit_export_path("")

    def test_special_chars_sanitized(self, audit_env):
        service, _, _ = audit_env
        # Slashes/dots stripped, "etcpasswd" remains → file not found
        with pytest.raises(ValueError, match="not found"):
            service.get_audit_export_path("../../etc/passwd")

    def test_only_special_chars_raises_invalid(self, audit_env):
        service, _, _ = audit_env
        # Regex keeps [A-Za-z0-9_.-], only slashes/spaces/etc stripped
        # Use chars that are all outside the allowed set
        with pytest.raises(ValueError, match="invalid export_id"):
            service.get_audit_export_path("@#$%^&*()")

    def test_nonexistent_export_id(self, audit_env):
        service, _, _ = audit_env
        with pytest.raises(ValueError, match="not found"):
            service.get_audit_export_path("nonexistent_id_12345")

    def test_path_traversal_sanitized(self, audit_env):
        service, _, _ = audit_env
        # Slashes and dots stripped, remaining chars form safe ID
        # "..%2f..%2f" → sanitized to "2f2f" (alphanumeric only)
        with pytest.raises(ValueError, match="not found"):
            service.get_audit_export_path("..%2f..%2fetc%2fpasswd")
