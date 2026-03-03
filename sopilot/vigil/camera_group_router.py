"""Camera group management API endpoints.

Allows operators to organise VigilPilot sessions (cameras) into logical
groups, apply shared anomaly normality profiles, and view cross-camera
violation overviews.

Mounted at ``/vigil/camera-groups``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


# ── Pydantic schemas ────────────────────────────────────────────────────────


class CameraGroupCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""
    anomaly_profile_name: str = ""
    location: str = ""


class CameraGroupUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    anomaly_profile_name: str | None = None
    location: str | None = None


class SessionMembershipRequest(BaseModel):
    session_id: int


class ApplyProfileRequest(BaseModel):
    profile_name: str = Field(..., min_length=1, max_length=100)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _get_repo(request: Request):
    """Get the CameraGroupRepository from app state."""
    repo = getattr(request.app.state, "camera_group_repo", None)
    if repo is None:
        raise HTTPException(
            status_code=503,
            detail="Camera group repository not initialised",
        )
    return repo


# ── Router factory ───────────────────────────────────────────────────────────


def build_camera_group_router() -> APIRouter:
    """Build and return the camera group management router."""
    router = APIRouter(prefix="/vigil/camera-groups", tags=["camera-groups"])

    @router.post("", status_code=201)
    async def create_group(body: CameraGroupCreate, request: Request) -> dict:
        """新しいカメラグループを作成。"""
        repo = _get_repo(request)

        def _create():
            return repo.create(
                name=body.name,
                description=body.description,
                anomaly_profile_name=body.anomaly_profile_name,
                location=body.location,
            )

        return await run_in_threadpool(_create)

    @router.get("")
    async def list_groups(request: Request) -> list[dict]:
        """全カメラグループ一覧。"""
        repo = _get_repo(request)

        def _list():
            return repo.list_all()

        return await run_in_threadpool(_list)

    @router.get("/{group_id}")
    async def get_group(group_id: int, request: Request) -> dict:
        """カメラグループの詳細。"""
        repo = _get_repo(request)

        def _get():
            return repo.get(group_id)

        group = await run_in_threadpool(_get)
        if group is None:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
        return group

    @router.put("/{group_id}")
    async def update_group(
        group_id: int, body: CameraGroupUpdate, request: Request
    ) -> dict:
        """カメラグループ情報を更新。"""
        repo = _get_repo(request)

        def _update():
            return repo.update(
                group_id=group_id,
                name=body.name,
                description=body.description,
                anomaly_profile_name=body.anomaly_profile_name,
                location=body.location,
            )

        result = await run_in_threadpool(_update)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
        return result

    @router.delete("/{group_id}")
    async def delete_group(group_id: int, request: Request) -> dict:
        """カメラグループを削除（セッションはグループ未割当に戻す）。"""
        repo = _get_repo(request)

        def _delete():
            return repo.delete(group_id)

        deleted = await run_in_threadpool(_delete)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
        return {"status": "ok", "deleted_group_id": group_id}

    # ── Session membership ─────────────────────────────────────────────

    @router.get("/{group_id}/sessions")
    async def get_group_sessions(group_id: int, request: Request) -> list[dict]:
        """グループに属するセッション一覧。"""
        repo = _get_repo(request)

        def _list():
            return repo.list_sessions_in_group(group_id)

        return await run_in_threadpool(_list)

    @router.post("/{group_id}/sessions")
    async def add_session_to_group(
        group_id: int, body: SessionMembershipRequest, request: Request
    ) -> dict:
        """セッションをグループに追加。"""
        repo = _get_repo(request)

        def _add():
            return repo.add_session(group_id, body.session_id)

        ok = await run_in_threadpool(_add)
        if not ok:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "ok", "group_id": group_id, "session_id": body.session_id}

    @router.delete("/{group_id}/sessions/{session_id}")
    async def remove_session_from_group(
        group_id: int, session_id: int, request: Request
    ) -> dict:
        """セッションをグループから外す。"""
        repo = _get_repo(request)

        def _remove():
            return repo.remove_session(session_id)

        ok = await run_in_threadpool(_remove)
        if not ok:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "ok", "session_id": session_id}

    # ── Cross-camera overview ──────────────────────────────────────────

    @router.get("/{group_id}/overview")
    async def get_group_overview(group_id: int, request: Request) -> dict:
        """グループ内全カメラの違反状況をクロスカメラ集計。

        Returns:
            - group metadata
            - session count (total / active / idle)
            - total violations and frames analyzed
            - per-session summary
            - 10 most recent events across all cameras
        """
        repo = _get_repo(request)

        def _overview():
            return repo.get_group_overview(group_id)

        result = await run_in_threadpool(_overview)
        if not result:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
        return result

    # ── Profile apply ──────────────────────────────────────────────────

    @router.post("/{group_id}/apply-profile")
    async def apply_profile_to_group(
        group_id: int, body: ApplyProfileRequest, request: Request
    ) -> dict:
        """グループ内の全セッションに異常検知プロファイルを紐付け。

        Updates the group's ``anomaly_profile_name`` field and records
        which profile was applied.  Operators can then load it per-session.
        """
        repo = _get_repo(request)

        def _apply():
            result = repo.update(
                group_id=group_id,
                anomaly_profile_name=body.profile_name,
            )
            if result is None:
                return None
            sessions = repo.list_sessions_in_group(group_id)
            return {
                "status": "ok",
                "group_id": group_id,
                "profile_name": body.profile_name,
                "sessions_updated": len(sessions),
                "session_ids": [s["id"] for s in sessions],
            }

        result = await run_in_threadpool(_apply)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
        return result

    return router
