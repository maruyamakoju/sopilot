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


def _get_learning_store(request: Request):
    """Get the GroupLearningStore from app state. Returns None if not available."""
    return getattr(request.app.state, "group_learning_store", None)


def _get_engine_optional(request: Request):
    """Return PerceptionEngine if perception backend is active, else None."""
    try:
        vlm = request.app.state.vigil_pipeline._vlm
        if hasattr(vlm, "_engine"):
            return vlm._engine
    except Exception:
        pass
    return None


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

    # ── Phase 14B: Cross-Camera Federated Learning ─────────────────────

    @router.get("/learning/compare")
    async def compare_group_learning(request: Request) -> dict:
        """全グループのσ値を比較し、フェデレーテッドラーニング推奨を返す (Phase 14B)。

        グループ間でσ値の差が大きい detector に対して推奨インサイトを生成する。
        """
        store = _get_learning_store(request)
        if store is None:
            return {"groups": [], "recommendations": [], "note": "learning store not available"}

        def _compare():
            return store.compare()

        return await run_in_threadpool(_compare)

    @router.get("/{group_id}/learning")
    async def get_group_learning(group_id: int, request: Request) -> dict:
        """グループの保存済み学習スナップショットを返す (Phase 14B)。

        スナップショットがなければ 404 を返す。
        """
        store = _get_learning_store(request)
        if store is None:
            raise HTTPException(status_code=503, detail="Group learning store not available")

        def _load():
            return store.load(group_id)

        snap = await run_in_threadpool(_load)
        if snap is None:
            raise HTTPException(
                status_code=404,
                detail=f"No learning snapshot for group {group_id}. Use /export to create one.",
            )
        return snap

    @router.post("/{group_id}/learning/export")
    async def export_group_learning(group_id: int, request: Request) -> dict:
        """現在のエンジン学習状態をグループにエクスポートする (Phase 14B)。

        SigmaTuner.get_state() + AnomalyTuner.get_stats() を取得してグループに保存する。
        エンジンが起動していない場合はベースライン値 (sigma=2.0) で保存する。
        """
        repo = _get_repo(request)
        store = _get_learning_store(request)
        if store is None:
            raise HTTPException(status_code=503, detail="Group learning store not available")

        def _export():
            group = repo.get(group_id)
            if group is None:
                return None
            engine = _get_engine_optional(request)
            # Sigma state
            sigma_state: dict[str, Any] = {"base_sigma": 2.0, "total_adjustments": 0,
                                            "detector_sigmas": {}, "recent_adjustments": []}
            if engine is not None:
                st = getattr(engine, "_sigma_tuner", None)
                if st is not None:
                    sigma_state = st.get_state()
            # Tuner stats
            tuner_stats: dict[str, Any] = {"total_feedback": 0, "overall_confirm_rate": 0.0,
                                            "suppressed_pairs": [], "trusted_pairs": []}
            if engine is not None:
                at = getattr(engine, "_anomaly_tuner", None)
                if at is not None:
                    tuner_stats = at.get_stats()
            snap = store.save(
                group_id=group_id,
                group_name=group.get("name", f"group_{group_id}"),
                sigma_state=sigma_state,
                tuner_stats=tuner_stats,
            )
            return snap

        result = await run_in_threadpool(_export)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
        return {"status": "ok", "snapshot": result}

    @router.post("/{group_id}/learning/import")
    async def import_group_learning(group_id: int, request: Request) -> dict:
        """グループの保存済みσ値を現在のエンジンに適用する (Phase 14B)。

        保存されたスナップショットの detector_sigmas を SigmaTuner に一括適用する。
        エンジンが起動していない場合は 400 を返す。
        """
        store = _get_learning_store(request)
        if store is None:
            raise HTTPException(status_code=503, detail="Group learning store not available")

        engine = _get_engine_optional(request)
        if engine is None:
            raise HTTPException(
                status_code=400,
                detail="Perception engine not active. Set VIGIL_VLM_BACKEND=perception",
            )

        def _import():
            snap = store.load(group_id)
            if snap is None:
                return None
            sigma_tuner = getattr(engine, "_sigma_tuner", None)
            if sigma_tuner is None:
                return {"applied": [], "note": "SigmaTuner not initialized"}
            applied = sigma_tuner.apply_detector_sigmas(
                snap.get("detector_sigmas", {})
            )
            return {
                "status": "ok",
                "group_id": group_id,
                "group_name": snap.get("group_name", ""),
                "applied_detectors": applied,
                "detector_sigmas": snap.get("detector_sigmas", {}),
                "source_saved_at": snap.get("saved_at"),
            }

        result = await run_in_threadpool(_import)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"No learning snapshot for group {group_id}. Use /export first.",
            )
        return result

    return router
