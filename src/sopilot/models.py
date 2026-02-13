"""SQLAlchemy models for VIGIL-RAG (PostgreSQL schema).

This module defines the database schema for both:
1. SOPilot (short-form SOP scoring) - backward compatible
2. VIGIL-RAG (long-form video understanding + events)

Design principles:
- Additive, not destructive (extends existing tables)
- Backward compatible with SQLite-based SOPilot
- Supports multi-scale chunking (shot/micro/meso/macro)
- Event detection with confidence + evidence links
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


def utc_now():
    """Generate timezone-aware UTC timestamp."""
    return datetime.now(tz=timezone.utc)


class Video(Base):
    """Video metadata table (superset of SOPilot + VIGIL-RAG fields)."""

    __tablename__ = "videos"

    # Primary key
    video_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # SOPilot compatibility fields
    file_path = Column(Text, nullable=False, unique=True)
    task_id = Column(String(200), index=True)
    role = Column(String(20))  # gold / trainee / audit
    site_id = Column(String(100), index=True)
    camera_id = Column(String(100), index=True)
    operator_id_hash = Column(String(100))
    num_clips = Column(Integer)
    embedding_model = Column(String(100))
    created_at = Column(Text, default=lambda: utc_now().isoformat())

    # VIGIL-RAG new fields
    uri = Column(Text)  # S3/HTTP URI (optional)
    storage_path = Column(Text)  # Object storage path
    duration_sec = Column(Float)
    fps = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    domain = Column(String(50), index=True)  # factory / surveillance / sports
    checksum = Column(String(64))  # SHA-256
    ingest_time = Column(Text, default=lambda: utc_now().isoformat())

    # Relationships
    clips = relationship("Clip", back_populates="video", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="video", cascade="all, delete-orphan")

    __table_args__ = (Index("idx_videos_domain_camera", "domain", "camera_id"),)


class Clip(Base):
    """Video clip table (multi-level: shot/micro/meso/macro)."""

    __tablename__ = "clips"

    clip_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.video_id"), nullable=False, index=True)

    # Chunk metadata
    level = Column(String(20), nullable=False, index=True)  # shot / micro / meso / macro
    start_sec = Column(Float, nullable=False)
    end_sec = Column(Float, nullable=False)

    # Keyframes and features
    keyframe_paths = Column(JSONB)  # ["/path/frame_001.jpg", ...]
    transcript_text = Column(Text)  # Whisper transcription (optional)
    embedding_id = Column(UUID(as_uuid=True), ForeignKey("embeddings.embedding_id"), index=True)

    # Legacy SOPilot compatibility
    clip_idx = Column(Integer)  # Sequential index (for backward compat)

    # Relationships
    video = relationship("Video", back_populates="clips")
    embedding = relationship("Embedding", back_populates="clips")

    __table_args__ = (
        Index("idx_clips_video_level", "video_id", "level"),
        Index("idx_clips_time_range", "video_id", "start_sec", "end_sec"),
        CheckConstraint("end_sec > start_sec", name="check_clip_time_valid"),
    )


class Embedding(Base):
    """Embedding metadata table (tracks model versions)."""

    __tablename__ = "embeddings"

    embedding_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    model_name = Column(String(100), nullable=False)  # internvideo2 / vjepa2 / heuristic
    model_version = Column(String(50))  # v1.0 / v2.5-chat-8b
    dimension = Column(Integer, nullable=False)
    vector_ref = Column(Text)  # Qdrant point ID or FAISS index key

    created_at = Column(Text, default=lambda: utc_now().isoformat())

    # Relationships
    clips = relationship("Clip", back_populates="embedding")

    __table_args__ = (Index("idx_embeddings_model", "model_name", "model_version"),)


class Event(Base):
    """Event detection table (zero-shot / supervised / anomaly)."""

    __tablename__ = "events"

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.video_id"), nullable=False, index=True)

    # Event metadata
    event_type = Column(String(100), nullable=False, index=True)  # intrusion / ppe_violation / goal / ...
    start_sec = Column(Float, nullable=False)
    end_sec = Column(Float, nullable=False)

    # Detection info
    confidence = Column(Float, nullable=False)  # 0.0 - 1.0
    method = Column(String(50), nullable=False)  # zero_shot / supervised / anomaly
    evidence_clip_ids = Column(ARRAY(UUID(as_uuid=True)))  # Array of clip UUIDs

    # Status tracking
    status = Column(String(20), default="new")  # new / confirmed / false_alarm / reviewed
    reviewed_by = Column(String(100))  # User who reviewed
    reviewed_at = Column(Text)

    # Additional metadata (flexible JSON)
    # Note: Using 'event_metadata' attribute name to avoid SQLAlchemy reserved word 'metadata'
    event_metadata = Column("metadata", JSONB)  # {"description": "...", "detector_version": "...", ...}

    created_at = Column(Text, default=lambda: utc_now().isoformat())

    # Relationships
    video = relationship("Video", back_populates="events")

    __table_args__ = (
        Index("idx_events_type_status", "event_type", "status"),
        Index("idx_events_video_time", "video_id", "start_sec", "end_sec"),
        CheckConstraint("end_sec > start_sec", name="check_event_time_valid"),
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="check_confidence_range"),
    )


class Query(Base):
    """Query history table (for RAG analytics and improvement)."""

    __tablename__ = "queries"

    query_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.video_id"), index=True)

    # Query content
    question = Column(Text, nullable=False)
    answer = Column(Text)
    confidence = Column(Float)

    # Retrieval metadata
    retrieved_clip_ids = Column(ARRAY(UUID(as_uuid=True)))
    retrieval_scores = Column(JSONB)  # [{"clip_id": "...", "score": 0.85}, ...]

    # Model info
    embedding_model = Column(String(100))
    llm_model = Column(String(100))

    # User feedback (for active learning)
    feedback = Column(String(20))  # helpful / unhelpful / incorrect / unknown
    feedback_comment = Column(Text)

    # Timestamps
    query_time = Column(Text, default=lambda: utc_now().isoformat())
    response_latency_sec = Column(Float)

    __table_args__ = (Index("idx_queries_video_time", "video_id", "query_time"),)


# Legacy SOPilot tables (for backward compatibility tracking)
# These will be maintained during migration but gradually deprecated


class IngestJob(Base):
    """SOPilot ingest job table (backward compat)."""

    __tablename__ = "ingest_jobs"

    job_id = Column(String(100), primary_key=True)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.video_id"))
    status = Column(String(20))  # queued / processing / completed / failed
    task_id = Column(String(200))
    role = Column(String(20))
    error_message = Column(Text)
    queued_at = Column(Text)
    started_at = Column(Text)
    finished_at = Column(Text)


class ScoreJob(Base):
    """SOPilot scoring job table (backward compat)."""

    __tablename__ = "score_jobs"

    job_id = Column(String(100), primary_key=True)
    gold_video_id = Column(UUID(as_uuid=True), ForeignKey("videos.video_id"))
    trainee_video_id = Column(UUID(as_uuid=True), ForeignKey("videos.video_id"))
    status = Column(String(20))
    score = Column(Float)
    result_json = Column(JSONB)
    error_message = Column(Text)
    queued_at = Column(Text)
    started_at = Column(Text)
    finished_at = Column(Text)


class TrainingJob(Base):
    """SOPilot training job table (backward compat)."""

    __tablename__ = "training_jobs"

    job_id = Column(String(100), primary_key=True)
    status = Column(String(20))
    trigger = Column(String(50))  # manual / nightly / api
    result_json = Column(JSONB)
    error_message = Column(Text)
    queued_at = Column(Text)
    started_at = Column(Text)
    finished_at = Column(Text)
