"""Insurance MVP FastAPI Application

Production-ready REST API for insurance claim review system.
"""

import os
import logging
import hashlib
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from insurance_mvp.api.models import (
    # Requests
    UploadRequest,
    ReviewDecisionRequest,
    # Responses
    UploadResponse,
    StatusResponse,
    AssessmentResponse,
    QueueResponse,
    QueueItem,
    ReviewDecisionResponse,
    AuditHistoryResponse,
    AuditLogEntry,
    MetricsResponse,
    ErrorResponse,
    HealthResponse,
    # Enums
    ClaimStatus,
    ReviewPriority,
    Severity,
    EventType,
    # Supporting models
    EvidenceItem,
    HazardItem,
    FaultAssessmentResponse,
    FraudRiskResponse,
)
from insurance_mvp.api.database import (
    DatabaseManager,
    ClaimRepository,
    AssessmentRepository,
    ReviewRepository,
    AuditLogRepository,
    Claim,
)
from insurance_mvp.api.auth import (
    get_api_key,
    get_api_key_optional,
    require_permission,
    check_rate_limit,
    initialize_dev_keys,
    initialize_dev_reviewers,
    APIKey,
)
from insurance_mvp.api.background import (
    initialize_worker,
    get_worker,
    shutdown_worker,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Application Configuration

class Config:
    """Application configuration"""

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./insurance.db")
    DATABASE_ECHO: bool = os.getenv("DATABASE_ECHO", "false").lower() == "true"

    # Storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
    ALLOWED_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv"]

    # Worker
    WORKER_MAX_THREADS: int = int(os.getenv("WORKER_MAX_THREADS", "4"))
    USE_PIPELINE: bool = os.getenv("USE_PIPELINE", "false").lower() == "true"

    # API
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    API_VERSION: str = "1.0.0"
    API_TITLE: str = "Insurance MVP API"

    # Development
    DEV_MODE: bool = os.getenv("DEV_MODE", "true").lower() == "true"


config = Config()


# FastAPI Application

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="Production-ready API for insurance claim review system with AI assessment",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Database and Worker Initialization

db_manager: Optional[DatabaseManager] = None
app_start_time: datetime = datetime.utcnow()


@app.on_event("startup")
async def startup_event():
    """Initialize database and background worker on startup"""
    global db_manager

    logger.info(f"Starting {config.API_TITLE} v{config.API_VERSION}")

    # Initialize database
    db_manager = DatabaseManager(config.DATABASE_URL, echo=config.DATABASE_ECHO)
    db_manager.create_tables()
    logger.info(f"Database initialized: {config.DATABASE_URL}")

    # Initialize background worker
    initialize_worker(
        db_manager=db_manager,
        max_workers=config.WORKER_MAX_THREADS,
        use_pipeline=config.USE_PIPELINE,
    )
    logger.info(f"Background worker initialized with {config.WORKER_MAX_THREADS} threads")

    # Create upload directory
    Path(config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory: {config.UPLOAD_DIR}")

    # Development mode: initialize test keys
    if config.DEV_MODE:
        logger.warning("!!! DEVELOPMENT MODE - Initializing test API keys !!!")
        dev_keys = initialize_dev_keys()
        initialize_dev_reviewers()
        logger.info("Test credentials initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")
    shutdown_worker(wait=True)
    logger.info("Shutdown complete")


# Dependency: Database Session

def get_db():
    """FastAPI dependency for database session"""
    session = db_manager.SessionLocal()
    try:
        yield session
    finally:
        session.close()


# Exception Handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Standardized error response for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
        ).model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Catch-all error handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail=str(exc) if config.DEV_MODE else None,
        ).model_dump(mode="json"),
    )


# Health Check

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.

    Returns system health status and basic metrics.
    """
    uptime = (datetime.utcnow() - app_start_time).total_seconds()

    # Check database connection
    db_connected = False
    try:
        from sqlalchemy import text
        with db_manager.get_session() as session:
            session.execute(text("SELECT 1"))
        db_connected = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")

    # Check background worker
    worker = get_worker()
    worker_active = worker.is_active() if worker else False

    # Overall status
    overall_status = "healthy" if db_connected else "unhealthy"

    return HealthResponse(
        status=overall_status,
        version=config.API_VERSION,
        uptime_seconds=uptime,
        database_connected=db_connected,
        background_worker_active=worker_active,
        timestamp=datetime.utcnow(),
    )


# Claims Endpoints

@app.post(
    "/claims/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Claims"],
    summary="Upload dashcam video",
    dependencies=[Depends(check_rate_limit)],
)
async def upload_claim(
    video: UploadFile = File(..., description="Dashcam video file"),
    claim_number: Optional[str] = Query(None, description="Optional claim reference number"),
    claimant_id: Optional[str] = Query(None, description="Optional claimant ID"),
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
):
    """
    Upload dashcam video for AI assessment.

    **Process:**
    1. Validate file (size, format)
    2. Save to storage
    3. Create claim record
    4. Queue for background processing
    5. Return claim ID

    **Authentication:** Requires valid API key with write permission.

    **Rate Limit:** 60 requests/minute per API key.
    """
    # Check write permission
    if not api_key.permissions.get("write", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Write permission required",
        )

    # Validate file extension
    file_ext = Path(video.filename).suffix.lower()
    if file_ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}",
        )

    # Generate unique claim ID
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    claim_id = f"claim_{timestamp}_{hashlib.sha256(video.filename.encode()).hexdigest()[:12]}"

    # Save video file
    video_path = Path(config.UPLOAD_DIR) / f"{claim_id}{file_ext}"

    try:
        # Stream to disk (memory efficient for large files)
        with video_path.open("wb") as f:
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await video.read(chunk_size):
                f.write(chunk)

        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        logger.info(f"Video saved: {video_path} ({file_size_mb:.2f} MB)")

        # Check size limit
        if file_size_mb > config.MAX_UPLOAD_SIZE_MB:
            video_path.unlink()  # Delete oversized file
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large ({file_size_mb:.2f} MB). Maximum: {config.MAX_UPLOAD_SIZE_MB} MB",
            )

        # Calculate video hash for deduplication
        video_hash = hashlib.sha256()
        with video_path.open("rb") as f:
            while chunk := f.read(8192):
                video_hash.update(chunk)
        video_hash_hex = video_hash.hexdigest()

        # Check for duplicate
        claim_repo = ClaimRepository(db)
        existing = claim_repo.get_by_hash(video_hash_hex)
        if existing:
            logger.warning(f"Duplicate video detected: {video_hash_hex} -> {existing.id}")
            video_path.unlink()  # Delete duplicate
            return UploadResponse(
                claim_id=existing.id,
                status=existing.status,
                message=f"Duplicate video detected. Existing claim: {existing.id}",
                upload_time=existing.upload_time,
            )

        # Create claim record
        metadata = {}
        if claim_number:
            metadata["claim_number"] = claim_number
        if claimant_id:
            metadata["claimant_id"] = claimant_id

        claim = claim_repo.create(
            claim_id=claim_id,
            video_path=str(video_path),
            claim_number=claim_number,
            claimant_id=claimant_id,
            metadata=metadata,
            video_hash=video_hash_hex,
        )

        # Audit log
        audit_repo = AuditLogRepository(db)
        audit_repo.create(
            claim_id=claim_id,
            event_type=EventType.CLAIM_UPLOADED,
            actor_type="API",
            actor_id=api_key.name,
            explanation=f"Video uploaded: {video.filename} ({file_size_mb:.2f} MB)",
            after_state={"status": claim.status.value},
        )

        # Submit for background processing
        worker = get_worker()
        worker.submit_claim(claim_id)

        return UploadResponse(
            claim_id=claim_id,
            status=ClaimStatus.QUEUED,
            message="Video uploaded successfully and queued for processing",
            upload_time=claim.upload_time,
        )

    except Exception as e:
        # Cleanup on error
        if video_path.exists():
            video_path.unlink()
        logger.error(f"Upload failed: {e}")
        raise


@app.get(
    "/claims/{claim_id}/status",
    response_model=StatusResponse,
    tags=["Claims"],
    summary="Check processing status",
)
async def get_claim_status(
    claim_id: str,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
):
    """
    Check claim processing status.

    **Returns:**
    - Current status (uploaded, queued, processing, assessed, etc.)
    - Processing progress (0-100%)
    - Error message (if failed)

    **Authentication:** Requires valid API key.
    """
    claim_repo = ClaimRepository(db)
    claim = claim_repo.get_by_id(claim_id)

    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim {claim_id} not found",
        )

    # Generate status message
    status_messages = {
        ClaimStatus.UPLOADED: "Video uploaded, awaiting processing",
        ClaimStatus.QUEUED: "Queued for processing",
        ClaimStatus.PROCESSING: "AI assessment in progress",
        ClaimStatus.ASSESSED: "AI assessment complete, ready for review",
        ClaimStatus.UNDER_REVIEW: "Under human review",
        ClaimStatus.REVIEWED: "Human review complete",
        ClaimStatus.APPROVED: "Claim approved",
        ClaimStatus.REJECTED: "Claim rejected",
        ClaimStatus.PENDING_INFO: "Additional information requested",
        ClaimStatus.FAILED: "Processing failed",
    }

    return StatusResponse(
        claim_id=claim.id,
        status=claim.status,
        message=status_messages.get(claim.status, "Unknown status"),
        upload_time=claim.upload_time,
        processing_started=claim.processing_started,
        processing_completed=claim.processing_completed,
        progress_percent=claim.progress_percent,
        error_message=claim.error_message,
    )


@app.get(
    "/claims/{claim_id}/assessment",
    response_model=AssessmentResponse,
    tags=["Claims"],
    summary="Get AI assessment",
)
async def get_claim_assessment(
    claim_id: str,
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
):
    """
    Get AI assessment results for claim.

    **Returns:**
    - Severity level (NONE, LOW, MEDIUM, HIGH)
    - Confidence score
    - Fault assessment (ratio, reasoning)
    - Fraud risk (score, indicators)
    - Evidence (hazards, timestamps, descriptions)

    **Authentication:** Requires valid API key.
    """
    # Get assessment
    assessment_repo = AssessmentRepository(db)
    assessment = assessment_repo.get_by_claim_id(claim_id)

    if not assessment:
        # Check if claim exists
        claim_repo = ClaimRepository(db)
        claim = claim_repo.get_by_id(claim_id)

        if not claim:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Claim {claim_id} not found",
            )

        if claim.status == ClaimStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Processing failed: {claim.error_message}",
            )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment not yet available. Current status: {claim.status.value}",
        )

    # Parse JSON fields
    import json
    prediction_set = json.loads(assessment.prediction_set) if assessment.prediction_set else []
    applicable_rules = json.loads(assessment.applicable_rules) if assessment.applicable_rules else []
    fraud_indicators = json.loads(assessment.fraud_indicators) if assessment.fraud_indicators else []
    hazards_raw = json.loads(assessment.hazards_json) if assessment.hazards_json else []
    evidence_raw = json.loads(assessment.evidence_json) if assessment.evidence_json else []

    return AssessmentResponse(
        claim_id=claim_id,
        severity=Severity(assessment.severity),
        confidence=assessment.confidence,
        prediction_set=[Severity(s) for s in prediction_set],
        review_priority=ReviewPriority(assessment.review_priority),
        fault_assessment=FaultAssessmentResponse(
            fault_ratio=assessment.fault_ratio,
            reasoning=assessment.fault_reasoning,
            applicable_rules=applicable_rules,
            scenario_type=assessment.scenario_type or "unknown",
            traffic_signal=assessment.traffic_signal,
            right_of_way=assessment.right_of_way,
        ),
        fraud_risk=FraudRiskResponse(
            risk_score=assessment.fraud_risk_score,
            indicators=fraud_indicators,
            reasoning=assessment.fraud_reasoning,
        ),
        hazards=[HazardItem(**h) for h in hazards_raw],
        evidence=[EvidenceItem(**e) for e in evidence_raw],
        causal_reasoning=assessment.causal_reasoning,
        recommended_action=assessment.recommended_action,
        processing_time_sec=assessment.processing_time_sec,
        timestamp=assessment.timestamp,
    )


# Review Endpoints

@app.get(
    "/reviews/queue",
    response_model=QueueResponse,
    tags=["Reviews"],
    summary="Get review queue",
)
async def get_review_queue(
    priority: Optional[ReviewPriority] = Query(None, description="Filter by priority"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum items to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
):
    """
    Get review queue sorted by priority.

    **Sorting:**
    1. URGENT (high fraud risk, low confidence, large prediction sets)
    2. STANDARD
    3. LOW_PRIORITY (high confidence, small prediction sets)

    **Authentication:** Requires valid API key.
    """
    # Get assessed claims
    claim_repo = ClaimRepository(db)
    claims = claim_repo.get_queue(status=ClaimStatus.ASSESSED, limit=limit, offset=offset)

    # Build queue items
    queue_items = []
    for claim in claims:
        assessment = claim.assessment
        if not assessment:
            continue

        import json
        prediction_set = json.loads(assessment.prediction_set) if assessment.prediction_set else []
        hazards = json.loads(assessment.hazards_json) if assessment.hazards_json else []

        queue_items.append(QueueItem(
            claim_id=claim.id,
            upload_time=claim.upload_time,
            review_priority=ReviewPriority(assessment.review_priority),
            severity=Severity(assessment.severity),
            confidence=assessment.confidence,
            fraud_risk_score=assessment.fraud_risk_score,
            prediction_set_size=len(prediction_set),
            hazard_count=len(hazards),
            fault_ratio=assessment.fault_ratio,
            recommended_action=assessment.recommended_action,
        ))

    # Sort by priority
    priority_order = {ReviewPriority.URGENT: 0, ReviewPriority.STANDARD: 1, ReviewPriority.LOW_PRIORITY: 2}
    queue_items.sort(key=lambda x: (priority_order[x.review_priority], x.upload_time))

    # Filter by priority if requested
    if priority:
        queue_items = [item for item in queue_items if item.review_priority == priority]

    # Count by priority
    urgent_count = sum(1 for item in queue_items if item.review_priority == ReviewPriority.URGENT)
    standard_count = sum(1 for item in queue_items if item.review_priority == ReviewPriority.STANDARD)
    low_priority_count = sum(1 for item in queue_items if item.review_priority == ReviewPriority.LOW_PRIORITY)

    return QueueResponse(
        total_count=len(queue_items),
        items=queue_items,
        urgent_count=urgent_count,
        standard_count=standard_count,
        low_priority_count=low_priority_count,
    )


@app.post(
    "/reviews/{claim_id}/decision",
    response_model=ReviewDecisionResponse,
    tags=["Reviews"],
    summary="Submit human review decision",
    dependencies=[Depends(check_rate_limit)],
)
async def submit_review_decision(
    claim_id: str,
    decision: ReviewDecisionRequest,
    reviewer_id: str = Query(..., description="Reviewer identifier"),
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
):
    """
    Submit human review decision.

    **Process:**
    1. Validate claim exists and is assessed
    2. Record review decision
    3. Apply overrides (if any)
    4. Update claim status
    5. Create audit log

    **Authentication:** Requires valid API key with write permission.
    """
    # Check write permission
    if not api_key.permissions.get("write", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Write permission required",
        )

    # Get claim
    claim_repo = ClaimRepository(db)
    claim = claim_repo.get_by_id(claim_id)

    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim {claim_id} not found",
        )

    if claim.status not in [ClaimStatus.ASSESSED, ClaimStatus.UNDER_REVIEW]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Claim cannot be reviewed in current status: {claim.status.value}",
        )

    # Get original assessment
    assessment_repo = AssessmentRepository(db)
    assessment = assessment_repo.get_by_claim_id(claim_id)

    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="AI assessment not found",
        )

    # Record review
    review_repo = ReviewRepository(db)
    review_start = time.time()

    review = review_repo.create(
        claim_id=claim_id,
        reviewer_id=reviewer_id,
        decision=decision.decision.value,
        reasoning=decision.reasoning,
        review_time_sec=time.time() - review_start,
        severity_override=decision.severity_override.value if decision.severity_override else None,
        fault_ratio_override=decision.fault_ratio_override,
        fraud_override=decision.fraud_override,
        comments=decision.comments,
    )

    # Update claim status based on decision
    new_status = {
        "APPROVE": ClaimStatus.APPROVED,
        "REJECT": ClaimStatus.REJECTED,
        "REQUEST_MORE_INFO": ClaimStatus.PENDING_INFO,
    }[decision.decision.value]

    # Capture state for audit
    before_state = {
        "status": claim.status.value,
        "severity": assessment.severity,
        "fault_ratio": assessment.fault_ratio,
    }

    claim_repo.update_status(claim_id, new_status)

    after_state = {
        "status": new_status.value,
        "severity": decision.severity_override.value if decision.severity_override else assessment.severity,
        "fault_ratio": decision.fault_ratio_override if decision.fault_ratio_override else assessment.fault_ratio,
    }

    # Audit log
    audit_repo = AuditLogRepository(db)
    audit_repo.create(
        claim_id=claim_id,
        event_type=EventType.HUMAN_REVIEW,
        actor_type="HUMAN",
        actor_id=reviewer_id,
        explanation=f"Review decision: {decision.decision.value}",
        before_state=before_state,
        after_state=after_state,
    )

    return ReviewDecisionResponse(
        claim_id=claim_id,
        status=new_status,
        message=f"Review decision recorded: {decision.decision.value}",
        review_time=review.timestamp,
    )


@app.get(
    "/reviews/{claim_id}/history",
    response_model=AuditHistoryResponse,
    tags=["Reviews"],
    summary="Get audit log for claim",
)
async def get_claim_history(
    claim_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum events to return"),
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_api_key),
):
    """
    Get complete audit log for claim.

    **Returns:**
    - All events (uploads, assessments, reviews, status changes)
    - Actor information (AI model, reviewer ID)
    - Before/after state changes
    - Timestamps

    **Authentication:** Requires valid API key.
    """
    # Verify claim exists
    claim_repo = ClaimRepository(db)
    claim = claim_repo.get_by_id(claim_id)

    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim {claim_id} not found",
        )

    # Get audit logs
    audit_repo = AuditLogRepository(db)
    logs = audit_repo.get_by_claim_id(claim_id, limit=limit)

    # Convert to response format
    import json
    events = []
    for log in logs:
        before_state = json.loads(log.before_state) if log.before_state else None
        after_state = json.loads(log.after_state) if log.after_state else None

        events.append(AuditLogEntry(
            log_id=log.id,
            claim_id=log.claim_id,
            event_type=log.event_type,
            actor_type=log.actor_type,
            actor_id=log.actor_id,
            explanation=log.explanation,
            before_state=before_state,
            after_state=after_state,
            timestamp=log.timestamp,
        ))

    return AuditHistoryResponse(
        claim_id=claim_id,
        total_events=len(events),
        events=events,
    )


# Metrics Endpoint

@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["System"],
    summary="System metrics",
)
async def get_metrics(
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_api_key_optional),
):
    """
    Get system metrics and statistics.

    **Metrics:**
    - Processing rate (claims/hour)
    - Queue depth by priority
    - Review statistics
    - Approval/rejection rates
    - Error rates

    **Authentication:** Optional (public metrics available without auth).
    """
    claim_repo = ClaimRepository(db)
    review_repo = ReviewRepository(db)

    # Total claims
    total_claims = db.query(Claim).count()
    claims_today = claim_repo.count_today()

    # Processing metrics
    from datetime import datetime, timedelta
    hour_ago = datetime.utcnow() - timedelta(hours=1)
    claims_last_hour = db.query(Claim).filter(Claim.upload_time >= hour_ago).count()
    processing_rate = claims_last_hour  # claims per hour

    # Average processing time (completed claims only)
    completed_claims = db.query(Claim).filter(
        Claim.processing_completed.isnot(None),
        Claim.processing_started.isnot(None),
    ).all()

    if completed_claims:
        processing_times = [
            (c.processing_completed - c.processing_started).total_seconds()
            for c in completed_claims
        ]
        avg_processing_time = sum(processing_times) / len(processing_times)
    else:
        avg_processing_time = 0.0

    # Queue metrics
    queue_depth = claim_repo.count_by_status(ClaimStatus.ASSESSED)
    queue_depth_by_priority = {
        "URGENT": 0,
        "STANDARD": 0,
        "LOW_PRIORITY": 0,
    }

    assessed_claims = claim_repo.get_queue(status=ClaimStatus.ASSESSED, limit=1000)
    for claim in assessed_claims:
        if claim.assessment:
            priority = claim.assessment.review_priority
            queue_depth_by_priority[priority] = queue_depth_by_priority.get(priority, 0) + 1

    # Review metrics
    pending_review = queue_depth
    reviewed_today = review_repo.count_today()

    # Average review time
    from insurance_mvp.api.database import Review
    recent_reviews = db.query(Review).order_by(Review.timestamp.desc()).limit(100).all()
    if recent_reviews:
        avg_review_time = sum(r.review_time_sec for r in recent_reviews) / len(recent_reviews)
    else:
        avg_review_time = 0.0

    # Decision metrics
    approved_count = claim_repo.count_by_status(ClaimStatus.APPROVED)
    rejected_count = claim_repo.count_by_status(ClaimStatus.REJECTED)
    total_decisions = approved_count + rejected_count

    if total_decisions > 0:
        approval_rate = approved_count / total_decisions
        rejection_rate = rejected_count / total_decisions
    else:
        approval_rate = 0.0
        rejection_rate = 0.0

    # AI metrics (average across all assessments)
    from insurance_mvp.api.database import Assessment
    assessments = db.query(Assessment).limit(1000).all()

    if assessments:
        avg_confidence = sum(a.confidence for a in assessments) / len(assessments)
        avg_fraud_risk = sum(a.fraud_risk_score for a in assessments) / len(assessments)
    else:
        avg_confidence = 0.0
        avg_fraud_risk = 0.0

    # Error metrics
    failed_count = claim_repo.count_by_status(ClaimStatus.FAILED)
    error_rate = failed_count / total_claims if total_claims > 0 else 0.0

    return MetricsResponse(
        total_claims=total_claims,
        claims_today=claims_today,
        processing_rate_per_hour=processing_rate,
        average_processing_time_sec=avg_processing_time,
        queue_depth=queue_depth,
        queue_depth_by_priority=queue_depth_by_priority,
        pending_review_count=pending_review,
        reviewed_today=reviewed_today,
        average_review_time_sec=avg_review_time,
        approval_rate=approval_rate,
        rejection_rate=rejection_rate,
        average_ai_confidence=avg_confidence,
        average_fraud_risk=avg_fraud_risk,
        failed_processing_count=failed_count,
        error_rate=error_rate,
        timestamp=datetime.utcnow(),
    )


# Development Endpoints (only in dev mode)

if config.DEV_MODE:
    @app.post("/dev/reset", tags=["Development"], summary="Reset database (dev only)")
    async def dev_reset_database(api_key: APIKey = Depends(require_permission("admin"))):
        """
        Reset database (DANGER: deletes all data).

        Only available in development mode.
        """
        logger.warning("!!! DATABASE RESET REQUESTED !!!")
        db_manager.drop_tables()
        db_manager.create_tables()
        return {"message": "Database reset complete"}


# Main Entry Point

def main():
    """Run development server"""
    uvicorn.run(
        "insurance_mvp.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
