"""SOP×Vigil compliance bridge endpoints.

Phase 28A — Auto Rule Generation.

Analyzes a task's gold video with a VLM and extracts a list of surveillance
monitoring rules that can be applied to a VigilPilot streaming session.

Endpoints
---------
POST   /tasks/{task_id}/generate-vigil-rules   Run VLM analysis → store rules
GET    /tasks/{task_id}/vigil-rules            Retrieve stored rules
DELETE /tasks/{task_id}/vigil-rules            Clear stored rules
POST   /tasks/{task_id}/apply-vigil-rules      Create a live streaming session
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi import Body as _Body
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from sopilot.api.error_handling import service_errors
from sopilot.api.routes._helpers import get_service
from sopilot.repositories.rule_repository import RuleRepository
from sopilot.services.rule_generator import RuleGenerator, _make_default_rules

logger = logging.getLogger(__name__)


# ── Schemas ────────────────────────────────────────────────────────────────────


class RuleItem(BaseModel):
    rule: str
    confidence: float
    category: str
    source: str = "auto"


class GenerateRulesResponse(BaseModel):
    task_id: str
    rules: list[RuleItem]
    source_video_id: int | None = None
    generated_count: int


class RulesListResponse(BaseModel):
    task_id: str
    rules: list[dict]
    total: int


class ApplyRulesResponse(BaseModel):
    session_id: str
    name: str
    rules: list[str]
    task_id: str


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get_rule_repo(request: Request) -> RuleRepository:
    service = get_service(request)
    return RuleRepository(service.database.connect)


def _get_gold_video(service: Any, task_id: str) -> dict[str, Any]:
    """Return the most recent ready gold video for *task_id*, or raise 404."""
    with service.database.connect() as conn:
        row = conn.execute(
            """
            SELECT id, file_path, status FROM videos
            WHERE task_id = ? AND is_gold = 1 AND status = 'ready'
            ORDER BY created_at DESC LIMIT 1
            """,
            (task_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No ready gold video found for task '{task_id}'",
        )
    return dict(row)


# ── Router ─────────────────────────────────────────────────────────────────────


def build_compliance_router() -> APIRouter:
    router = APIRouter(tags=["Compliance"])

    # ── POST /tasks/{task_id}/generate-vigil-rules ─────────────────────────────

    @router.post(
        "/tasks/{task_id}/generate-vigil-rules",
        response_model=GenerateRulesResponse,
        summary="Generate VigilPilot rules from gold video (VLM analysis)",
        description=(
            "Samples frames from the task's gold video and uses a VLM to extract "
            "safety monitoring rules. Stores rules in the database and returns them. "
            "Use ``dry_run=true`` to preview without saving."
        ),
    )
    @service_errors
    async def generate_vigil_rules(
        task_id: str,
        request: Request,
        dry_run: bool = Query(False, description="Preview only — do not store rules"),
        n_frames: int = Query(5, ge=1, le=20, description="Number of frames to sample"),
    ) -> GenerateRulesResponse:
        service = get_service(request)
        video = _get_gold_video(service, task_id)
        file_path = video.get("file_path", "")

        if not file_path:
            raise HTTPException(status_code=422, detail="Gold video has no file_path")

        # Build VLM callable if API key is configured
        vlm_callable = None
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("VIGIL_VLM_API_KEY", "")
        if api_key and api_key not in ("not-configured", ""):
            from sopilot.services.rule_generator import make_claude_vlm_callable

            vlm_callable = make_claude_vlm_callable(api_key)

        generator = RuleGenerator(vlm_callable=vlm_callable, n_frames=n_frames)

        rule_dicts = await run_in_threadpool(generator.generate, file_path)

        rule_strings = [r["rule"] for r in rule_dicts]
        confidences = [r["confidence"] for r in rule_dicts]

        if not dry_run:
            repo = _get_rule_repo(request)
            await run_in_threadpool(
                repo.save_rules,
                task_id,
                rule_strings,
                "auto",
                confidences,
            )

        return GenerateRulesResponse(
            task_id=task_id,
            rules=[
                RuleItem(
                    rule=r["rule"],
                    confidence=r["confidence"],
                    category=r.get("category", "behavior"),
                    source="auto",
                )
                for r in rule_dicts
            ],
            source_video_id=video["id"],
            generated_count=len(rule_dicts),
        )

    # ── GET /tasks/{task_id}/vigil-rules ───────────────────────────────────────

    @router.get(
        "/tasks/{task_id}/vigil-rules",
        response_model=RulesListResponse,
        summary="Get stored VigilPilot rules for a task",
    )
    @service_errors
    async def get_vigil_rules(
        task_id: str, request: Request
    ) -> RulesListResponse:
        repo = _get_rule_repo(request)
        rules = await run_in_threadpool(repo.get_rules, task_id)
        return RulesListResponse(task_id=task_id, rules=rules, total=len(rules))

    # ── DELETE /tasks/{task_id}/vigil-rules ────────────────────────────────────

    @router.delete(
        "/tasks/{task_id}/vigil-rules",
        summary="Clear stored VigilPilot rules for a task",
    )
    @service_errors
    async def delete_vigil_rules(
        task_id: str, request: Request
    ) -> dict[str, Any]:
        repo = _get_rule_repo(request)
        count = await run_in_threadpool(repo.delete_rules, task_id)
        return {"task_id": task_id, "deleted": count}

    # ── POST /tasks/{task_id}/apply-vigil-rules ────────────────────────────────

    @router.post(
        "/tasks/{task_id}/apply-vigil-rules",
        response_model=ApplyRulesResponse,
        status_code=201,
        summary="Create a VigilPilot live stream session with auto-generated rules",
        description=(
            "Creates a new streaming session (Phase 27) populated with the "
            "stored auto-generated rules for the task. "
            "If no rules are stored, falls back to defaults."
        ),
    )
    @service_errors
    async def apply_vigil_rules(
        task_id: str,
        request: Request,
        session_name: str = _Body(default="", embed=True, description="Override session name"),
    ) -> ApplyRulesResponse:
        repo = _get_rule_repo(request)
        rule_strings = await run_in_threadpool(repo.get_rule_strings, task_id)

        if not rule_strings:
            # Fallback to defaults so the session is always usable
            rule_strings = [r["rule"] for r in _make_default_rules()]

        name = session_name.strip() or f"{task_id} 自動監視"

        # Register in the live stream session registry (app.state.stream_sessions)
        from sopilot.vigil.stream_session import create_session

        if not hasattr(request.app.state, "stream_sessions"):
            request.app.state.stream_sessions = {}

        session = create_session(name=name, rules=rule_strings)
        request.app.state.stream_sessions[session.session_id] = session

        logger.info(
            "apply-vigil-rules: task=%s → stream session=%s (%d rules)",
            task_id,
            session.session_id,
            len(rule_strings),
        )
        return ApplyRulesResponse(
            session_id=session.session_id,
            name=name,
            rules=rule_strings,
            task_id=task_id,
        )

    return router
