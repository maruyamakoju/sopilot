"""Tests for CameraGroupRepository and camera-groups API endpoints.

Covers:
- Repository CRUD (create, get, list, update, delete)
- Session membership (add, remove, list)
- Cross-camera overview aggregation
- Profile apply
- API endpoints via TestClient with mocked app.state
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sopilot.vigil.camera_group_repository import CameraGroupRepository
from sopilot.vigil.camera_group_router import build_camera_group_router


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    """Create a temporary SQLite DB with the required schema."""
    path = str(tmp_path / "test_camera_groups.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS camera_groups (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            name                 TEXT    NOT NULL,
            description          TEXT    NOT NULL DEFAULT '',
            anomaly_profile_name TEXT    NOT NULL DEFAULT '',
            location             TEXT    NOT NULL DEFAULT '',
            created_at           TEXT    NOT NULL,
            updated_at           TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS vigil_sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT    NOT NULL,
            rules_json      TEXT    NOT NULL DEFAULT '[]',
            sample_fps      REAL    NOT NULL DEFAULT 1.0,
            severity_threshold TEXT NOT NULL DEFAULT 'warning',
            status          TEXT    NOT NULL DEFAULT 'idle',
            video_filename  TEXT,
            total_frames_analyzed INTEGER NOT NULL DEFAULT 0,
            violation_count INTEGER NOT NULL DEFAULT 0,
            camera_group_id INTEGER REFERENCES camera_groups(id) ON DELETE SET NULL,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS vigil_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      INTEGER NOT NULL REFERENCES vigil_sessions(id) ON DELETE CASCADE,
            timestamp_sec   REAL    NOT NULL DEFAULT 0.0,
            frame_number    INTEGER NOT NULL DEFAULT 0,
            violations_json TEXT    NOT NULL DEFAULT '[]',
            frame_path      TEXT,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture()
def repo(db_path: str) -> CameraGroupRepository:
    return CameraGroupRepository(db_path)


def _add_session(db_path: str, name: str = "cam1", group_id: int | None = None) -> int:
    """Insert a vigil_session row and return its id."""
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "INSERT INTO vigil_sessions (name, camera_group_id, created_at, updated_at) VALUES (?, ?, datetime('now'), datetime('now'))",
        (name, group_id),
    )
    sid = cur.lastrowid
    conn.commit()
    conn.close()
    return sid  # type: ignore[return-value]


def _add_event(db_path: str, session_id: int, violations: list | None = None) -> int:
    """Insert a vigil_event row and return its id."""
    conn = sqlite3.connect(db_path)
    vj = json.dumps(violations or [{"rule": "test", "severity": "warning"}])
    cur = conn.execute(
        "INSERT INTO vigil_events (session_id, timestamp_sec, frame_number, violations_json, created_at) VALUES (?, 1.0, 1, ?, datetime('now'))",
        (session_id, vj),
    )
    eid = cur.lastrowid
    conn.commit()
    conn.close()
    return eid  # type: ignore[return-value]


# ── CameraGroupRepository: CRUD ───────────────────────────────────────────────


class TestCameraGroupCreate:
    def test_creates_group_returns_dict(self, repo: CameraGroupRepository) -> None:
        g = repo.create(name="Floor 1")
        assert g["id"] >= 1
        assert g["name"] == "Floor 1"

    def test_optional_fields_default(self, repo: CameraGroupRepository) -> None:
        g = repo.create(name="Entrance")
        assert g["description"] == ""
        assert g["anomaly_profile_name"] == ""
        assert g["location"] == ""

    def test_full_fields(self, repo: CameraGroupRepository) -> None:
        g = repo.create(
            name="Rooftop",
            description="High security zone",
            anomaly_profile_name="dayshift",
            location="Building A",
        )
        assert g["description"] == "High security zone"
        assert g["anomaly_profile_name"] == "dayshift"
        assert g["location"] == "Building A"

    def test_created_at_set(self, repo: CameraGroupRepository) -> None:
        g = repo.create(name="G1")
        assert g["created_at"] != ""

    def test_updated_at_set(self, repo: CameraGroupRepository) -> None:
        g = repo.create(name="G1")
        assert g["updated_at"] != ""

    def test_auto_increment_ids(self, repo: CameraGroupRepository) -> None:
        g1 = repo.create(name="A")
        g2 = repo.create(name="B")
        assert g2["id"] == g1["id"] + 1


class TestCameraGroupGet:
    def test_get_existing(self, repo: CameraGroupRepository) -> None:
        g = repo.create(name="Lobby")
        fetched = repo.get(g["id"])
        assert fetched is not None
        assert fetched["name"] == "Lobby"

    def test_get_nonexistent_returns_none(self, repo: CameraGroupRepository) -> None:
        assert repo.get(9999) is None


class TestCameraGroupListAll:
    def test_empty(self, repo: CameraGroupRepository) -> None:
        assert repo.list_all() == []

    def test_ordered_by_name(self, repo: CameraGroupRepository) -> None:
        repo.create(name="Zulu")
        repo.create(name="Alpha")
        repo.create(name="Mike")
        names = [g["name"] for g in repo.list_all()]
        assert names == ["Alpha", "Mike", "Zulu"]

    def test_all_returned(self, repo: CameraGroupRepository) -> None:
        for i in range(5):
            repo.create(name=f"Group {i}")
        assert len(repo.list_all()) == 5


class TestCameraGroupUpdate:
    def test_update_name(self, repo: CameraGroupRepository) -> None:
        g = repo.create(name="Old")
        updated = repo.update(g["id"], name="New")
        assert updated is not None
        assert updated["name"] == "New"

    def test_update_description(self, repo: CameraGroupRepository) -> None:
        g = repo.create(name="G")
        updated = repo.update(g["id"], description="changed desc")
        assert updated is not None
        assert updated["description"] == "changed desc"

    def test_update_anomaly_profile(self, repo: CameraGroupRepository) -> None:
        g = repo.create(name="G")
        updated = repo.update(g["id"], anomaly_profile_name="nightshift")
        assert updated is not None
        assert updated["anomaly_profile_name"] == "nightshift"

    def test_update_location(self, repo: CameraGroupRepository) -> None:
        g = repo.create(name="G")
        updated = repo.update(g["id"], location="Floor 3")
        assert updated is not None
        assert updated["location"] == "Floor 3"

    def test_update_nonexistent_returns_none(self, repo: CameraGroupRepository) -> None:
        assert repo.update(9999, name="x") is None

    def test_partial_update_preserves_other_fields(self, repo: CameraGroupRepository) -> None:
        g = repo.create(name="G", description="keep this", location="here")
        repo.update(g["id"], name="NewName")
        fetched = repo.get(g["id"])
        assert fetched is not None
        assert fetched["description"] == "keep this"
        assert fetched["location"] == "here"


class TestCameraGroupDelete:
    def test_delete_existing(self, repo: CameraGroupRepository) -> None:
        g = repo.create(name="ToBe")
        assert repo.delete(g["id"]) is True
        assert repo.get(g["id"]) is None

    def test_delete_nonexistent(self, repo: CameraGroupRepository) -> None:
        assert repo.delete(9999) is False

    def test_delete_unlinks_sessions(self, repo: CameraGroupRepository, db_path: str) -> None:
        g = repo.create(name="G")
        sid = _add_session(db_path, group_id=g["id"])
        repo.delete(g["id"])
        # session camera_group_id should be NULL now
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT camera_group_id FROM vigil_sessions WHERE id = ?", (sid,)).fetchone()
        conn.close()
        assert row[0] is None


# ── Session membership ────────────────────────────────────────────────────────


class TestSessionMembership:
    def test_add_session_success(self, repo: CameraGroupRepository, db_path: str) -> None:
        g = repo.create(name="G")
        sid = _add_session(db_path)
        assert repo.add_session(g["id"], sid) is True

    def test_add_session_nonexistent_returns_false(self, repo: CameraGroupRepository) -> None:
        g = repo.create(name="G")
        assert repo.add_session(g["id"], 9999) is False

    def test_remove_session_success(self, repo: CameraGroupRepository, db_path: str) -> None:
        g = repo.create(name="G")
        sid = _add_session(db_path, group_id=g["id"])
        assert repo.remove_session(sid) is True

    def test_remove_session_nonexistent_returns_false(self, repo: CameraGroupRepository) -> None:
        assert repo.remove_session(9999) is False

    def test_list_sessions_in_group(self, repo: CameraGroupRepository, db_path: str) -> None:
        g = repo.create(name="G")
        sid1 = _add_session(db_path, name="cam1", group_id=g["id"])
        sid2 = _add_session(db_path, name="cam2", group_id=g["id"])
        sessions = repo.list_sessions_in_group(g["id"])
        ids = {s["id"] for s in sessions}
        assert sid1 in ids
        assert sid2 in ids

    def test_list_sessions_excludes_other_groups(self, repo: CameraGroupRepository, db_path: str) -> None:
        g1 = repo.create(name="G1")
        g2 = repo.create(name="G2")
        sid1 = _add_session(db_path, name="camA", group_id=g1["id"])
        _add_session(db_path, name="camB", group_id=g2["id"])
        sessions = repo.list_sessions_in_group(g1["id"])
        assert len(sessions) == 1
        assert sessions[0]["id"] == sid1

    def test_session_includes_group_name(self, repo: CameraGroupRepository, db_path: str) -> None:
        g = repo.create(name="SpecialGroup")
        sid = _add_session(db_path, name="cam1", group_id=g["id"])
        sessions = repo.list_sessions_in_group(g["id"])
        assert sessions[0]["group_name"] == "SpecialGroup"


# ── Cross-camera overview ─────────────────────────────────────────────────────


class TestGroupOverview:
    def test_overview_nonexistent_group_returns_empty(self, repo: CameraGroupRepository) -> None:
        result = repo.get_group_overview(9999)
        assert result == {}

    def test_overview_counts(self, repo: CameraGroupRepository, db_path: str) -> None:
        g = repo.create(name="G")
        _add_session(db_path, name="cam1", group_id=g["id"])
        _add_session(db_path, name="cam2", group_id=g["id"])
        overview = repo.get_group_overview(g["id"])
        assert overview["total_sessions"] == 2
        assert overview["group"]["id"] == g["id"]

    def test_overview_includes_recent_events(self, repo: CameraGroupRepository, db_path: str) -> None:
        g = repo.create(name="G")
        sid = _add_session(db_path, name="cam1", group_id=g["id"])
        _add_event(db_path, sid)
        overview = repo.get_group_overview(g["id"])
        assert len(overview["recent_events"]) == 1

    def test_overview_max_10_events(self, repo: CameraGroupRepository, db_path: str) -> None:
        g = repo.create(name="G")
        sid = _add_session(db_path, name="cam1", group_id=g["id"])
        for _ in range(15):
            _add_event(db_path, sid)
        overview = repo.get_group_overview(g["id"])
        assert len(overview["recent_events"]) <= 10

    def test_overview_empty_group(self, repo: CameraGroupRepository) -> None:
        g = repo.create(name="EmptyG")
        overview = repo.get_group_overview(g["id"])
        assert overview["total_sessions"] == 0
        assert overview["recent_events"] == []

    def test_overview_idle_count(self, repo: CameraGroupRepository, db_path: str) -> None:
        g = repo.create(name="G")
        _add_session(db_path, name="idle_cam", group_id=g["id"])
        overview = repo.get_group_overview(g["id"])
        assert overview["idle_sessions"] == 1
        assert overview["active_sessions"] == 0


# ── API endpoints ─────────────────────────────────────────────────────────────


@pytest.fixture()
def api_client(repo: CameraGroupRepository) -> TestClient:
    """TestClient wired to a real CameraGroupRepository."""
    app = FastAPI()
    app.state.camera_group_repo = repo
    app.include_router(build_camera_group_router())
    return TestClient(app, raise_server_exceptions=True)


class TestCameraGroupAPI:
    def test_create_group_201(self, api_client: TestClient) -> None:
        resp = api_client.post("/vigil/camera-groups", json={"name": "Floor 1"})
        assert resp.status_code == 201
        assert resp.json()["name"] == "Floor 1"

    def test_create_group_validation_empty_name(self, api_client: TestClient) -> None:
        resp = api_client.post("/vigil/camera-groups", json={"name": ""})
        assert resp.status_code == 422

    def test_list_groups_empty(self, api_client: TestClient) -> None:
        resp = api_client.get("/vigil/camera-groups")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_groups_after_create(self, api_client: TestClient) -> None:
        api_client.post("/vigil/camera-groups", json={"name": "A"})
        api_client.post("/vigil/camera-groups", json={"name": "B"})
        resp = api_client.get("/vigil/camera-groups")
        assert len(resp.json()) == 2

    def test_get_group(self, api_client: TestClient) -> None:
        created = api_client.post("/vigil/camera-groups", json={"name": "G"}).json()
        resp = api_client.get(f"/vigil/camera-groups/{created['id']}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "G"

    def test_get_group_not_found(self, api_client: TestClient) -> None:
        resp = api_client.get("/vigil/camera-groups/9999")
        assert resp.status_code == 404

    def test_update_group(self, api_client: TestClient) -> None:
        created = api_client.post("/vigil/camera-groups", json={"name": "Old"}).json()
        resp = api_client.put(
            f"/vigil/camera-groups/{created['id']}",
            json={"name": "New"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "New"

    def test_update_group_not_found(self, api_client: TestClient) -> None:
        resp = api_client.put("/vigil/camera-groups/9999", json={"name": "x"})
        assert resp.status_code == 404

    def test_delete_group(self, api_client: TestClient) -> None:
        created = api_client.post("/vigil/camera-groups", json={"name": "G"}).json()
        resp = api_client.delete(f"/vigil/camera-groups/{created['id']}")
        assert resp.status_code == 200
        assert resp.json()["deleted_group_id"] == created["id"]

    def test_delete_group_not_found(self, api_client: TestClient) -> None:
        resp = api_client.delete("/vigil/camera-groups/9999")
        assert resp.status_code == 404

    def test_get_sessions_empty(self, api_client: TestClient) -> None:
        created = api_client.post("/vigil/camera-groups", json={"name": "G"}).json()
        resp = api_client.get(f"/vigil/camera-groups/{created['id']}/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_add_session_not_found(self, api_client: TestClient) -> None:
        created = api_client.post("/vigil/camera-groups", json={"name": "G"}).json()
        resp = api_client.post(
            f"/vigil/camera-groups/{created['id']}/sessions",
            json={"session_id": 9999},
        )
        assert resp.status_code == 404

    def test_remove_session_not_found(self, api_client: TestClient) -> None:
        created = api_client.post("/vigil/camera-groups", json={"name": "G"}).json()
        resp = api_client.delete(f"/vigil/camera-groups/{created['id']}/sessions/9999")
        assert resp.status_code == 404

    def test_overview_not_found(self, api_client: TestClient) -> None:
        resp = api_client.get("/vigil/camera-groups/9999/overview")
        assert resp.status_code == 404

    def test_overview_existing(self, api_client: TestClient) -> None:
        created = api_client.post("/vigil/camera-groups", json={"name": "G"}).json()
        resp = api_client.get(f"/vigil/camera-groups/{created['id']}/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_sessions"] == 0
        assert "group" in data

    def test_apply_profile(self, api_client: TestClient) -> None:
        created = api_client.post("/vigil/camera-groups", json={"name": "G"}).json()
        resp = api_client.post(
            f"/vigil/camera-groups/{created['id']}/apply-profile",
            json={"profile_name": "dayshift"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["profile_name"] == "dayshift"
        assert data["group_id"] == created["id"]

    def test_apply_profile_not_found(self, api_client: TestClient) -> None:
        resp = api_client.post(
            "/vigil/camera-groups/9999/apply-profile",
            json={"profile_name": "p"},
        )
        assert resp.status_code == 404

    def test_apply_profile_validation_empty(self, api_client: TestClient) -> None:
        created = api_client.post("/vigil/camera-groups", json={"name": "G"}).json()
        resp = api_client.post(
            f"/vigil/camera-groups/{created['id']}/apply-profile",
            json={"profile_name": ""},
        )
        assert resp.status_code == 422

    def test_no_repo_returns_503(self) -> None:
        app = FastAPI()
        # No camera_group_repo in state
        app.include_router(build_camera_group_router())
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/vigil/camera-groups")
        assert resp.status_code == 503
