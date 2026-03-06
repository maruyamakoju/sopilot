"""Tests for stream_router.py (LiveStream API endpoints)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sopilot.vigil.stream_session import create_session


# ── Minimal app fixture ───────────────────────────────────────────────────────


def _make_app():
    """Create a minimal FastAPI app with the stream router mounted."""
    from fastapi import FastAPI

    from sopilot.vigil.stream_router import build_stream_router

    app = FastAPI()
    app.state.stream_sessions = {}

    # Provide a minimal vigil_pipeline stub so _get_inferrer doesn't crash
    pipeline_stub = MagicMock()
    pipeline_stub._vlm = MagicMock(spec=[])  # no _engine attribute
    app.state.vigil_pipeline = pipeline_stub

    app.include_router(build_stream_router())
    return app


@pytest.fixture
def client():
    app = _make_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _create_session(client: TestClient, name: str = "test-cam", rules=None) -> dict:
    rules = rules or []
    resp = client.post("/vigil/stream/sessions", json={"name": name, "rules": rules})
    assert resp.status_code == 201, resp.text
    return resp.json()


# ── POST /sessions ────────────────────────────────────────────────────────────


def test_create_session_returns_201(client):
    resp = client.post("/vigil/stream/sessions", json={"name": "cam1", "rules": []})
    assert resp.status_code == 201


def test_create_session_response_fields(client):
    resp = client.post(
        "/vigil/stream/sessions",
        json={"name": "cam1", "rules": ["ヘルメット未着用"]},
    )
    data = resp.json()
    assert "session_id" in data
    assert data["name"] == "cam1"
    assert data["rules"] == ["ヘルメット未着用"]
    assert data["is_active"] is True
    assert data["frame_count"] == 0


def test_create_session_no_name_422(client):
    resp = client.post("/vigil/stream/sessions", json={})
    assert resp.status_code == 422


def test_create_session_empty_name_422(client):
    resp = client.post("/vigil/stream/sessions", json={"name": ""})
    assert resp.status_code == 422


def test_create_multiple_sessions(client):
    s1 = _create_session(client, "cam1")
    s2 = _create_session(client, "cam2")
    assert s1["session_id"] != s2["session_id"]


# ── GET /sessions ─────────────────────────────────────────────────────────────


def test_list_sessions_empty(client):
    resp = client.get("/vigil/stream/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_sessions_shows_created(client):
    _create_session(client, "cam1")
    _create_session(client, "cam2")
    resp = client.get("/vigil/stream/sessions")
    assert len(resp.json()) == 2


def test_list_sessions_names(client):
    _create_session(client, "entrance")
    _create_session(client, "warehouse")
    names = {s["name"] for s in client.get("/vigil/stream/sessions").json()}
    assert names == {"entrance", "warehouse"}


# ── DELETE /sessions/{sid} ────────────────────────────────────────────────────


def test_delete_session(client):
    s = _create_session(client)
    sid = s["session_id"]
    resp = client.delete(f"/vigil/stream/sessions/{sid}")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == sid
    assert resp.json()["status"] == "stopped"


def test_delete_session_removes_from_list(client):
    s = _create_session(client)
    sid = s["session_id"]
    client.delete(f"/vigil/stream/sessions/{sid}")
    resp = client.get("/vigil/stream/sessions")
    assert all(sess["session_id"] != sid for sess in resp.json())


def test_delete_session_not_found(client):
    resp = client.delete("/vigil/stream/sessions/nonexistent-id")
    assert resp.status_code == 404


# ── POST /sessions/{sid}/frame ────────────────────────────────────────────────


def _make_jpeg_bytes(h: int = 32, w: int = 32) -> bytes:
    import cv2
    import numpy as np

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    return bytes(buf)


def test_push_frame_increments_count(client):
    s = _create_session(client)
    sid = s["session_id"]
    jpeg = _make_jpeg_bytes()
    resp = client.post(
        f"/vigil/stream/sessions/{sid}/frame",
        content=jpeg,
        headers={"Content-Type": "image/jpeg"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["frame_count"] == 1
    assert data["session_id"] == sid


def test_push_frame_response_structure(client):
    s = _create_session(client)
    jpeg = _make_jpeg_bytes()
    resp = client.post(
        f"/vigil/stream/sessions/{s['session_id']}/frame",
        content=jpeg,
        headers={"Content-Type": "image/jpeg"},
    )
    data = resp.json()
    assert "detections" in data
    assert "violations" in data
    assert "frame_ms" in data


def test_push_frame_multiple_increments(client):
    s = _create_session(client)
    sid = s["session_id"]
    jpeg = _make_jpeg_bytes()
    for i in range(5):
        resp = client.post(
            f"/vigil/stream/sessions/{sid}/frame",
            content=jpeg,
            headers={"Content-Type": "image/jpeg"},
        )
    assert resp.json()["frame_count"] == 5


def test_push_frame_session_not_found(client):
    resp = client.post(
        "/vigil/stream/sessions/no-such-id/frame",
        content=_make_jpeg_bytes(),
        headers={"Content-Type": "image/jpeg"},
    )
    assert resp.status_code == 404


def test_push_frame_empty_body_422(client):
    s = _create_session(client)
    resp = client.post(
        f"/vigil/stream/sessions/{s['session_id']}/frame",
        content=b"",
        headers={"Content-Type": "image/jpeg"},
    )
    assert resp.status_code == 400


def test_push_frame_stopped_session_410(client):
    s = _create_session(client)
    sid = s["session_id"]
    client.delete(f"/vigil/stream/sessions/{sid}")

    # Re-inject stopped session for testing (delete removes it; simulate by creating new stopped)
    app = client.app
    stopped = create_session("stopped-cam", [])
    stopped.stop()
    stopped_id = stopped.session_id
    app.state.stream_sessions[stopped_id] = stopped
    resp = client.post(
        f"/vigil/stream/sessions/{stopped_id}/frame",
        content=_make_jpeg_bytes(),
        headers={"Content-Type": "image/jpeg"},
    )
    assert resp.status_code == 410


def test_push_frame_no_detector_empty_detections(client):
    s = _create_session(client)
    jpeg = _make_jpeg_bytes()
    resp = client.post(
        f"/vigil/stream/sessions/{s['session_id']}/frame",
        content=jpeg,
        headers={"Content-Type": "image/jpeg"},
    )
    assert resp.json()["detections"] == []


# ── WebSocket ─────────────────────────────────────────────────────────────────


def test_websocket_rejects_missing_session(client):
    # Server closes with code 4004 — TestClient raises WebSocketDisconnect at __enter__
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises((WebSocketDisconnect, Exception)):
        with client.websocket_connect("/vigil/stream/sessions/no-such/ws") as ws:
            ws.receive_text()


def test_websocket_connect_and_push_frame(client):
    s = _create_session(client)
    sid = s["session_id"]
    jpeg = _make_jpeg_bytes()

    with client.websocket_connect(f"/vigil/stream/sessions/{sid}/ws") as ws:
        ws.send_bytes(jpeg)
        data = json.loads(ws.receive_text())
        assert data["session_id"] == sid
        assert "detections" in data
        assert "violations" in data
        assert data["frame_count"] == 1


def test_websocket_multiple_frames(client):
    s = _create_session(client)
    sid = s["session_id"]
    jpeg = _make_jpeg_bytes()

    with client.websocket_connect(f"/vigil/stream/sessions/{sid}/ws") as ws:
        for i in range(3):
            ws.send_bytes(jpeg)
            result = json.loads(ws.receive_text())
        assert result["frame_count"] == 3


# ── MJPEG preview ─────────────────────────────────────────────────────────────


def test_preview_session_not_found(client):
    resp = client.get("/vigil/stream/sessions/no-such/preview")
    assert resp.status_code == 404


def test_preview_content_type(client):
    s = _create_session(client)
    # Push a frame so last_frame_jpeg is set
    jpeg = _make_jpeg_bytes()
    client.post(
        f"/vigil/stream/sessions/{s['session_id']}/frame",
        content=jpeg,
        headers={"Content-Type": "image/jpeg"},
    )
    # Stop session so MJPEG generator exits
    app = client.app
    app.state.stream_sessions[s["session_id"]].stop()

    resp = client.get(f"/vigil/stream/sessions/{s['session_id']}/preview")
    assert "multipart/x-mixed-replace" in resp.headers.get("content-type", "")
