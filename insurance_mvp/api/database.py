"""Database Layer

SQLAlchemy ORM models and database operations.
Supports both SQLite (development) and PostgreSQL (production).
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
    Index,
    Enum as SQLEnum,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool

from insurance_mvp.api.models import ClaimStatus, EventType


Base = declarative_base()


# ORM Models

class Claim(Base):
    """Claims table - stores uploaded videos and processing status"""
    __tablename__ = "claims"

    id = Column(String(64), primary_key=True)
    claim_number = Column(String(128), nullable=True, index=True)
    claimant_id = Column(String(128), nullable=True, index=True)
    video_path = Column(String(512), nullable=False)
    video_hash = Column(String(64), nullable=True, index=True)  # SHA256 for dedup

    status = Column(SQLEnum(ClaimStatus), nullable=False, default=ClaimStatus.UPLOADED, index=True)
    upload_time = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    processing_started = Column(DateTime, nullable=True)
    processing_completed = Column(DateTime, nullable=True)

    progress_percent = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    metadata_json = Column(Text, nullable=True)  # JSON string for flexible metadata

    # Relationships
    assessment = relationship("Assessment", back_populates="claim", uselist=False, cascade="all, delete-orphan")
    reviews = relationship("Review", back_populates="claim", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="claim", cascade="all, delete-orphan")

    # Indexes for common queries
    __table_args__ = (
        Index('idx_status_upload_time', 'status', 'upload_time'),
        Index('idx_upload_time_desc', upload_time.desc()),
    )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Parse metadata JSON"""
        if self.metadata_json:
            return json.loads(self.metadata_json)
        return {}

    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        """Store metadata as JSON"""
        self.metadata_json = json.dumps(value) if value else None


class Assessment(Base):
    """Assessments table - stores AI evaluation results"""
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(String(64), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)

    # Core assessment
    severity = Column(String(16), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    prediction_set = Column(Text, nullable=False)  # JSON array
    review_priority = Column(String(16), nullable=False, index=True)

    # Fault assessment
    fault_ratio = Column(Float, nullable=False)
    fault_reasoning = Column(Text, nullable=False)
    applicable_rules = Column(Text, nullable=True)  # JSON array
    scenario_type = Column(String(64), nullable=True)
    traffic_signal = Column(String(16), nullable=True)
    right_of_way = Column(String(64), nullable=True)

    # Fraud detection
    fraud_risk_score = Column(Float, nullable=False, index=True)
    fraud_indicators = Column(Text, nullable=True)  # JSON array
    fraud_reasoning = Column(Text, nullable=False)

    # Reasoning
    causal_reasoning = Column(Text, nullable=False)
    recommended_action = Column(String(32), nullable=False)

    # Evidence (stored as JSON for flexibility)
    hazards_json = Column(Text, nullable=True)
    evidence_json = Column(Text, nullable=True)

    # Metadata
    processing_time_sec = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    model_version = Column(String(64), nullable=True)

    # Relationships
    claim = relationship("Claim", back_populates="assessment")

    # Indexes
    __table_args__ = (
        Index('idx_review_priority_timestamp', 'review_priority', 'timestamp'),
        Index('idx_fraud_risk_desc', fraud_risk_score.desc()),
    )


class Review(Base):
    """Reviews table - stores human review decisions"""
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(String(64), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True)
    reviewer_id = Column(String(128), nullable=False, index=True)

    decision = Column(String(32), nullable=False, index=True)
    reasoning = Column(Text, nullable=False)
    comments = Column(Text, nullable=True)

    # Overrides
    severity_override = Column(String(16), nullable=True)
    fault_ratio_override = Column(Float, nullable=True)
    fraud_override = Column(Boolean, nullable=True)

    # Metadata
    review_time_sec = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Relationships
    claim = relationship("Claim", back_populates="reviews")

    # Indexes
    __table_args__ = (
        Index('idx_reviewer_timestamp', 'reviewer_id', 'timestamp'),
    )


class AuditLog(Base):
    """Audit log table - immutable log of all claim events"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(String(64), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True)

    event_type = Column(SQLEnum(EventType), nullable=False, index=True)
    actor_type = Column(String(16), nullable=False)  # AI or HUMAN
    actor_id = Column(String(128), nullable=False)

    explanation = Column(Text, nullable=False)
    before_state = Column(Text, nullable=True)  # JSON
    after_state = Column(Text, nullable=True)   # JSON

    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Relationships
    claim = relationship("Claim", back_populates="audit_logs")

    # Indexes
    __table_args__ = (
        Index('idx_claim_timestamp', 'claim_id', 'timestamp'),
        Index('idx_event_type_timestamp', 'event_type', 'timestamp'),
    )


# Database Manager

class DatabaseManager:
    """Database connection and session management"""

    def __init__(self, database_url: str, echo: bool = False):
        """
        Initialize database manager.

        Args:
            database_url: SQLAlchemy database URL
                - SQLite: "sqlite:///./insurance.db"
                - PostgreSQL: "postgresql://user:pass@localhost/insurance"
            echo: Enable SQL query logging
        """
        # Special handling for in-memory SQLite (testing)
        if database_url == "sqlite:///:memory:":
            self.engine = create_engine(
                database_url,
                echo=echo,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            self.engine = create_engine(database_url, echo=echo)

        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self):
        """Create all tables (idempotent)"""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """Drop all tables (DANGER: data loss!)"""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self):
        """
        Get database session context manager.

        Usage:
            with db_manager.get_session() as session:
                claim = session.query(Claim).first()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session_dependency(self):
        """
        FastAPI dependency for database sessions.

        Usage in route:
            @app.get("/claims/{claim_id}")
            def get_claim(claim_id: str, db: Session = Depends(db_manager.get_session_dependency)):
                return db.query(Claim).filter(Claim.id == claim_id).first()
        """
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()


# Database Operations (Repository Pattern)

class ClaimRepository:
    """Repository for claim operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        claim_id: str,
        video_path: str,
        claim_number: Optional[str] = None,
        claimant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        video_hash: Optional[str] = None,
    ) -> Claim:
        """Create new claim"""
        claim = Claim(
            id=claim_id,
            claim_number=claim_number,
            claimant_id=claimant_id,
            video_path=video_path,
            video_hash=video_hash,
            status=ClaimStatus.UPLOADED,
            upload_time=datetime.utcnow(),
        )
        if metadata:
            claim.metadata = metadata

        self.session.add(claim)
        self.session.commit()
        self.session.refresh(claim)
        return claim

    def get_by_id(self, claim_id: str) -> Optional[Claim]:
        """Get claim by ID"""
        return self.session.query(Claim).filter(Claim.id == claim_id).first()

    def get_by_hash(self, video_hash: str) -> Optional[Claim]:
        """Get claim by video hash (for deduplication)"""
        return self.session.query(Claim).filter(Claim.video_hash == video_hash).first()

    def update_status(
        self,
        claim_id: str,
        status: ClaimStatus,
        progress_percent: Optional[float] = None,
        error_message: Optional[str] = None,
    ):
        """Update claim status"""
        claim = self.get_by_id(claim_id)
        if not claim:
            raise ValueError(f"Claim {claim_id} not found")

        claim.status = status
        if progress_percent is not None:
            claim.progress_percent = progress_percent
        if error_message is not None:
            claim.error_message = error_message

        if status == ClaimStatus.PROCESSING and not claim.processing_started:
            claim.processing_started = datetime.utcnow()
        elif status in [ClaimStatus.ASSESSED, ClaimStatus.FAILED]:
            claim.processing_completed = datetime.utcnow()

        self.session.commit()
        self.session.refresh(claim)
        return claim

    def get_queue(
        self,
        status: Optional[ClaimStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Claim]:
        """Get claims for review queue"""
        query = self.session.query(Claim)

        if status:
            query = query.filter(Claim.status == status)
        else:
            # Default: claims awaiting review
            query = query.filter(Claim.status.in_([ClaimStatus.ASSESSED, ClaimStatus.UNDER_REVIEW]))

        query = query.order_by(Claim.upload_time.desc())
        return query.limit(limit).offset(offset).all()

    def count_by_status(self, status: ClaimStatus) -> int:
        """Count claims by status"""
        return self.session.query(Claim).filter(Claim.status == status).count()

    def count_today(self) -> int:
        """Count claims uploaded today"""
        from datetime import date
        today = datetime.combine(date.today(), datetime.min.time())
        return self.session.query(Claim).filter(Claim.upload_time >= today).count()


class AssessmentRepository:
    """Repository for assessment operations"""

    def __init__(self, session: Session):
        self.session = session

    def create_from_dict(self, claim_id: str, assessment_data: Dict[str, Any]) -> Assessment:
        """Create assessment from dictionary (from ClaimAssessment model)"""
        assessment = Assessment(
            claim_id=claim_id,
            severity=assessment_data.get("severity"),
            confidence=assessment_data.get("confidence"),
            prediction_set=json.dumps(list(assessment_data.get("prediction_set", []))),
            review_priority=assessment_data.get("review_priority"),
            fault_ratio=assessment_data.get("fault_assessment", {}).get("fault_ratio"),
            fault_reasoning=assessment_data.get("fault_assessment", {}).get("reasoning", ""),
            applicable_rules=json.dumps(assessment_data.get("fault_assessment", {}).get("applicable_rules", [])),
            scenario_type=assessment_data.get("fault_assessment", {}).get("scenario_type"),
            traffic_signal=assessment_data.get("fault_assessment", {}).get("traffic_signal"),
            right_of_way=assessment_data.get("fault_assessment", {}).get("right_of_way"),
            fraud_risk_score=assessment_data.get("fraud_risk", {}).get("risk_score"),
            fraud_indicators=json.dumps(assessment_data.get("fraud_risk", {}).get("indicators", [])),
            fraud_reasoning=assessment_data.get("fraud_risk", {}).get("reasoning", ""),
            causal_reasoning=assessment_data.get("causal_reasoning", ""),
            recommended_action=assessment_data.get("recommended_action", "REVIEW"),
            hazards_json=json.dumps([h.dict() for h in assessment_data.get("hazards", [])]),
            evidence_json=json.dumps([e.dict() for e in assessment_data.get("evidence", [])]),
            processing_time_sec=assessment_data.get("processing_time_sec", 0.0),
            model_version=assessment_data.get("model_version"),
        )

        self.session.add(assessment)
        self.session.commit()
        self.session.refresh(assessment)
        return assessment

    def get_by_claim_id(self, claim_id: str) -> Optional[Assessment]:
        """Get assessment for claim"""
        return self.session.query(Assessment).filter(Assessment.claim_id == claim_id).first()


class ReviewRepository:
    """Repository for review operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        claim_id: str,
        reviewer_id: str,
        decision: str,
        reasoning: str,
        review_time_sec: float,
        severity_override: Optional[str] = None,
        fault_ratio_override: Optional[float] = None,
        fraud_override: Optional[bool] = None,
        comments: Optional[str] = None,
    ) -> Review:
        """Create review"""
        review = Review(
            claim_id=claim_id,
            reviewer_id=reviewer_id,
            decision=decision,
            reasoning=reasoning,
            comments=comments,
            severity_override=severity_override,
            fault_ratio_override=fault_ratio_override,
            fraud_override=fraud_override,
            review_time_sec=review_time_sec,
        )

        self.session.add(review)
        self.session.commit()
        self.session.refresh(review)
        return review

    def get_by_claim_id(self, claim_id: str) -> List[Review]:
        """Get all reviews for claim"""
        return self.session.query(Review).filter(Review.claim_id == claim_id).order_by(Review.timestamp.desc()).all()

    def count_today(self) -> int:
        """Count reviews today"""
        from datetime import date
        today = datetime.combine(date.today(), datetime.min.time())
        return self.session.query(Review).filter(Review.timestamp >= today).count()


class AuditLogRepository:
    """Repository for audit log operations"""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        claim_id: str,
        event_type: EventType,
        actor_type: str,
        actor_id: str,
        explanation: str,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """Create audit log entry"""
        log = AuditLog(
            claim_id=claim_id,
            event_type=event_type,
            actor_type=actor_type,
            actor_id=actor_id,
            explanation=explanation,
            before_state=json.dumps(before_state) if before_state else None,
            after_state=json.dumps(after_state) if after_state else None,
        )

        self.session.add(log)
        self.session.commit()
        self.session.refresh(log)
        return log

    def get_by_claim_id(self, claim_id: str, limit: int = 100) -> List[AuditLog]:
        """Get audit logs for claim"""
        return (
            self.session.query(AuditLog)
            .filter(AuditLog.claim_id == claim_id)
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
            .all()
        )
