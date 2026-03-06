"""Tests for Re-ID router (vigil/reid_router.py) — Phase 28B."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sopilot.vigil.reid_gallery import ReIDGallery


# ── App fixture ───────────────────────────────────────────────────────────────


def _make_app(gallery: ReIDGallery | None = None):
    from fastapi import FastAPI

    from sopilot.vigil.reid_router import build_reid_router

    app = FastAPI()
    app.state.reid_gallery = gallery or ReIDGallery()
    app.include_router(build_reid_router())
    return app


@pytest.fixture
def client():
    app = _make_app()
    with TestClient(app) as c:
        yield c


# ── Helper stubs ──────────────────────────────────────────────────────────────


@dataclass
class _BBox:
    x1: float = 0.1
    y1: float = 0.1
    x2: float = 0.5
    y2: float = 0.9


@dataclass
class _Det:
    label: str = "person"
    confidence: float = 0.9
    bbox: _BBox = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.bbox is None:
            self.bbox = _BBox()


def _rand_emb(dim=512) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ── GET /vigil/reid/persons ───────────────────────────────────────────────────


def test_list_persons_empty(client):
    resp = client.get("/vigil/reid/persons")
    assert resp.status_code == 200
    data = resp.json()
    assert data["persons"] == []
    assert data["total"] == 0


def test_list_persons_with_data(client):
    gallery = client.app.state.reid_gallery
    gallery.update("cam1", "Camera 1", [_Det()], [_rand_emb()])
    resp = client.get("/vigil/reid/persons")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


def test_list_persons_session_filter(client):
    gallery = client.app.state.reid_gallery
    gallery.update("cam1", "Camera 1", [_Det()], [_rand_emb()])
    resp = client.get("/vigil/reid/persons?session_id=cam1")
    assert resp.json()["total"] == 1
    resp2 = client.get("/vigil/reid/persons?session_id=cam2")
    assert resp2.json()["total"] == 0


def test_list_persons_limit(client):
    gallery = ReIDGallery(match_threshold=0.99)
    for i in range(5):
        emb = np.zeros(512, dtype=np.float32)
        emb[i] = 1.0
        gallery.update("cam1", "C1", [_Det()], [emb])
    app = _make_app(gallery)
    with TestClient(app) as c:
        resp = c.get("/vigil/reid/persons?limit=3")
    assert len(resp.json()["persons"]) <= 3


# ── GET /vigil/reid/persons/{person_id} ───────────────────────────────────────


def test_get_person_not_found(client):
    resp = client.get("/vigil/reid/persons/no-such-id")
    assert resp.status_code == 404


def test_get_person_found(client):
    gallery = client.app.state.reid_gallery
    gallery.update("cam1", "Camera 1", [_Det()], [_rand_emb()])
    persons = gallery.get_persons()
    pid = persons[0]["person_id"]
    resp = client.get(f"/vigil/reid/persons/{pid}")
    assert resp.status_code == 200
    assert resp.json()["person_id"] == pid


def test_get_person_fields(client):
    gallery = client.app.state.reid_gallery
    gallery.update("cam1", "Camera 1", [_Det()], [_rand_emb()])
    pid = gallery.get_persons()[0]["person_id"]
    data = client.get(f"/vigil/reid/persons/{pid}").json()
    assert "person_id" in data
    assert "label" in data
    assert "camera_history" in data
    assert "cameras" in data


# ── GET /vigil/reid/events/recent ─────────────────────────────────────────────


def test_recent_events_empty(client):
    resp = client.get("/vigil/reid/events/recent")
    assert resp.status_code == 200
    assert resp.json()["events"] == []


def test_recent_events_populated(client):
    g = ReIDGallery(match_threshold=0.5)
    emb = _rand_emb()
    g.update("cam1", "C1", [_Det()], [emb])
    g.update("cam2", "C2", [_Det()], [emb])
    app = _make_app(g)
    with TestClient(app) as c:
        resp = c.get("/vigil/reid/events/recent")
    assert resp.json()["total"] >= 1


# ── GET /vigil/reid/stats ─────────────────────────────────────────────────────


def test_stats_empty(client):
    resp = client.get("/vigil/reid/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_persons"] == 0
    assert data["multi_camera_persons"] == 0


def test_stats_with_cross_camera(client):
    g = ReIDGallery(match_threshold=0.5)
    emb = _rand_emb()
    g.update("cam1", "C1", [_Det()], [emb])
    g.update("cam2", "C2", [_Det()], [emb])
    app = _make_app(g)
    with TestClient(app) as c:
        resp = c.get("/vigil/reid/stats")
    data = resp.json()
    assert data["total_persons"] == 1
    assert data["multi_camera_persons"] == 1
    assert data["recent_cross_camera_events"] >= 1


# ── DELETE /vigil/reid/sessions/{sid} ────────────────────────────────────────


def test_clear_session(client):
    gallery = client.app.state.reid_gallery
    gallery.update("cam1", "Camera 1", [_Det()], [_rand_emb()])
    resp = client.delete("/vigil/reid/sessions/cam1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "cam1"
    assert data["persons_removed"] >= 0


def test_clear_nonexistent_session(client):
    resp = client.delete("/vigil/reid/sessions/no-such")
    assert resp.status_code == 200
    assert resp.json()["persons_removed"] == 0


# ── No gallery 503 ────────────────────────────────────────────────────────────


def test_no_gallery_returns_503():
    from fastapi import FastAPI

    from sopilot.vigil.reid_router import build_reid_router

    app = FastAPI()
    # Intentionally no reid_gallery on app.state
    app.include_router(build_reid_router())
    with TestClient(app) as c:
        resp = c.get("/vigil/reid/persons")
    assert resp.status_code == 503
