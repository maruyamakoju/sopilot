"""Pydantic schemas for the VigilPilot API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────


class SessionCreateRequest(BaseModel):
    name: str = Field(..., description="監視セッション名")
    rules: list[str] = Field(
        ...,
        min_length=1,
        description="監視ルール一覧（自然言語で記述）",
        examples=[["ヘルメット未着用の作業者を検出", "安全ベルトなしで高所作業している人を検出"]],
    )
    sample_fps: float = Field(default=1.0, ge=0.1, le=5.0, description="解析フレームレート (fps)")
    severity_threshold: str = Field(
        default="warning",
        pattern="^(info|warning|critical)$",
        description="通知する最低重大度",
    )


# ── Response models ────────────────────────────────────────────────────────


class ViolationDetail(BaseModel):
    rule_index: int
    rule: str
    description_ja: str
    severity: str  # info | warning | critical
    confidence: float


class ViolationEvent(BaseModel):
    event_id: int
    session_id: int
    timestamp_sec: float
    frame_number: int
    violations: list[ViolationDetail]
    frame_url: str | None = None
    created_at: str


class SessionResponse(BaseModel):
    session_id: int
    name: str
    rules: list[str]
    sample_fps: float
    severity_threshold: str
    status: str  # idle | processing | completed | failed
    video_filename: str | None = None
    total_frames_analyzed: int
    violation_count: int
    created_at: str
    updated_at: str


class SessionListItem(BaseModel):
    session_id: int
    name: str
    status: str
    violation_count: int
    total_frames_analyzed: int
    created_at: str


class AnalyzeResponse(BaseModel):
    session_id: int
    status: str
    message: str


class SessionReport(BaseModel):
    session_id: int
    name: str
    rules: list[str]
    status: str
    video_filename: str | None
    total_frames_analyzed: int
    violation_count: int
    severity_breakdown: dict[str, int]
    rule_breakdown: dict[str, int]
    events: list[ViolationEvent]
    created_at: str


class VLMResult(BaseModel):
    """Parsed response from VLM for a single frame."""

    has_violation: bool
    violations: list[dict[str, Any]] = Field(default_factory=list)
    raw_text: str = ""
