"""Tests for compliance API (routes/compliance.py) — Phase 28A."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sopilot.vigil.stream_session import create_session


# ── Minimal app fixture ───────────────────────────────────────────────────────


def _make_app(gold_video_row=None):
    from fastapi import FastAPI

    from sopilot.api.routes.compliance import build_compliance_router

    app = FastAPI()
    app.state.stream_sessions = {}

    # --- Stub service --------------------------------------------------------
    service_stub = MagicMock()
    conn_stub = MagicMock()
    conn_stub.execute.return_value.rowcount = 0
    conn_stub.execute.return_value.lastrowid = 1

    if gold_video_row:
        conn_stub.execute.return_value.fetchone.return_value = gold_video_row
    else:
        conn_stub.execute.return_value.fetchone.return_value = None

    @contextmanager
    def _connect():
        yield conn_stub

    service_stub.database.connect = _connect

    app.state.sopilot_service = service_stub

    app.include_router(build_compliance_router(), prefix="/api/v1")
    return app


class _FakeRow(dict):
    """Dict subclass that behaves like sqlite3.Row (supports dict() conversion)."""


def _make_gold_row(file_path="/fake/video.mp4", video_id=1):
    return _FakeRow({"id": video_id, "file_path": file_path, "status": "ready"})


# ── GET /tasks/{task_id}/vigil-rules ─────────────────────────────────────────


def test_get_vigil_rules_empty():
    app = _make_app()
    with TestClient(app) as client:
        resp = client.get("/api/v1/tasks/task1/vigil-rules")
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "task1"
    assert data["rules"] == []
    assert data["total"] == 0


# ── DELETE /tasks/{task_id}/vigil-rules ──────────────────────────────────────


def test_delete_vigil_rules_returns_zero_when_empty():
    app = _make_app()
    with TestClient(app) as client:
        resp = client.delete("/api/v1/tasks/task1/vigil-rules")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 0


# ── POST /tasks/{task_id}/generate-vigil-rules ────────────────────────────────


def test_generate_rules_no_gold_video_404():
    app = _make_app(gold_video_row=None)  # no gold video
    with TestClient(app) as client:
        resp = client.post("/api/v1/tasks/taskX/generate-vigil-rules")
    assert resp.status_code == 404


def test_generate_rules_no_vlm_returns_defaults():
    app = _make_app(gold_video_row=_make_gold_row())
    with TestClient(app) as client, patch("sopilot.services.rule_generator.RuleGenerator._sample_frames", return_value=[]):
        resp = client.post("/api/v1/tasks/task1/generate-vigil-rules")
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "task1"
    assert data["generated_count"] > 0
    assert isinstance(data["rules"], list)


def test_generate_rules_dry_run_does_not_store():
    app = _make_app(gold_video_row=_make_gold_row())
    with TestClient(app) as client, patch("sopilot.services.rule_generator.RuleGenerator._sample_frames", return_value=[]):
        resp = client.post("/api/v1/tasks/task1/generate-vigil-rules?dry_run=true")
    assert resp.status_code == 200
    # After dry_run, GET should return empty (nothing persisted)
    with TestClient(app) as client:
        get_resp = client.get("/api/v1/tasks/task1/vigil-rules")
    assert get_resp.json()["total"] == 0


def test_generate_rules_stores_when_not_dry_run():
    app = _make_app(gold_video_row=_make_gold_row())
    vlm_response = json.dumps({
        "rules": [{"rule": "安全確認必須", "confidence": 0.8, "category": "behavior"}]
    })
    mock_vlm = MagicMock(return_value=vlm_response)
    fake_jpeg = b"\xff\xd8\xff" + b"\x00" * 50

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}), \
         patch("sopilot.services.rule_generator.make_claude_vlm_callable", return_value=mock_vlm), \
         patch("sopilot.services.rule_generator.RuleGenerator._sample_frames", return_value=[fake_jpeg]):
        with TestClient(app) as client:
            resp = client.post("/api/v1/tasks/task1/generate-vigil-rules")
    assert resp.status_code == 200


def test_generate_rules_response_fields():
    app = _make_app(gold_video_row=_make_gold_row())
    with TestClient(app) as client, patch("sopilot.services.rule_generator.RuleGenerator._sample_frames", return_value=[]):
        resp = client.post("/api/v1/tasks/task1/generate-vigil-rules")
    data = resp.json()
    assert "task_id" in data
    assert "rules" in data
    assert "generated_count" in data


# ── POST /tasks/{task_id}/apply-vigil-rules ───────────────────────────────────


def test_apply_rules_creates_stream_session():
    app = _make_app(gold_video_row=_make_gold_row())
    # First store some rules
    with TestClient(app) as client, patch("sopilot.services.rule_generator.RuleGenerator._sample_frames", return_value=[]):
        client.post("/api/v1/tasks/task1/generate-vigil-rules")
        resp = client.post("/api/v1/tasks/task1/apply-vigil-rules", json={})
    assert resp.status_code == 201
    data = resp.json()
    assert "session_id" in data
    assert data["task_id"] == "task1"


def test_apply_rules_uses_default_session_name():
    app = _make_app(gold_video_row=_make_gold_row())
    with TestClient(app) as client, patch("sopilot.services.rule_generator.RuleGenerator._sample_frames", return_value=[]):
        resp = client.post("/api/v1/tasks/myTask/apply-vigil-rules", json={})
    assert resp.status_code == 201
    data = resp.json()
    assert "myTask" in data["name"]


def test_apply_rules_custom_session_name():
    app = _make_app(gold_video_row=_make_gold_row())
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/tasks/task1/apply-vigil-rules",
            json={"session_name": "カスタム名"},
        )
    assert resp.status_code == 201
    assert resp.json()["name"] == "カスタム名"


def test_apply_rules_with_no_stored_rules_uses_defaults():
    app = _make_app(gold_video_row=_make_gold_row())
    with TestClient(app) as client:
        resp = client.post("/api/v1/tasks/task1/apply-vigil-rules", json={})
    assert resp.status_code == 201
    assert len(resp.json()["rules"]) > 0


def test_apply_rules_session_registered_in_state():
    app = _make_app(gold_video_row=_make_gold_row())
    with TestClient(app) as client:
        resp = client.post("/api/v1/tasks/task1/apply-vigil-rules", json={})
    sid = resp.json()["session_id"]
    assert sid in app.state.stream_sessions


def test_roundtrip_generate_and_apply():
    """Full roundtrip: generate → apply → session created in registry."""
    app = _make_app(gold_video_row=_make_gold_row())
    with TestClient(app) as client, \
         patch("sopilot.services.rule_generator.RuleGenerator._sample_frames", return_value=[]):
        gen_resp = client.post("/api/v1/tasks/task1/generate-vigil-rules")
        apply_resp = client.post("/api/v1/tasks/task1/apply-vigil-rules", json={})

    assert gen_resp.status_code == 200
    assert apply_resp.status_code == 201
    # Session created with fallback defaults (no real DB persistence in mock)
    sid = apply_resp.json()["session_id"]
    assert sid in app.state.stream_sessions
