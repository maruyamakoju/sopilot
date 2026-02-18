"""Web UI Routes for Insurance Claim Review System

Production-quality web interface with Jinja2 templates.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from insurance_mvp.api.database import AssessmentRepository, ClaimRepository, ReviewRepository
from insurance_mvp.api.models import ClaimStatus

# Initialize templates
template_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))


# Custom Jinja2 filters
def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def format_datetime(dt: datetime) -> str:
    """Format datetime for display"""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    return dt.strftime("%Y-%m-%d %H:%M")


templates.env.filters["format_timestamp"] = format_timestamp
templates.env.filters["format_datetime"] = format_datetime

# Create router
router = APIRouter(tags=["Web UI"])


# Dependency: Database session
def get_db():
    """Get database session (will be overridden by main app)"""
    from insurance_mvp.api.main import get_db as main_get_db

    return main_get_db()


@router.get("/", response_class=HTMLResponse)
async def root(request: Request, db: Session = Depends(get_db)):
    """Redirect to queue page"""
    # Get queue items
    claim_repo = ClaimRepository(db)
    claims = claim_repo.get_queue(status=ClaimStatus.ASSESSED, limit=50)

    queue_items = []
    for claim in claims:
        if claim.assessment:
            queue_items.append(_build_queue_item_dict(claim))

    return templates.TemplateResponse(
        "queue.html",
        {
            "request": request,
            "queue_items": queue_items,
            "current_user": "admin",  # TODO: Get from session/auth
            "total_pages": 1,
            "current_page": 1,
            "per_page": 50,
            "total_items": len(queue_items),
        },
    )


@router.get("/queue", response_class=HTMLResponse)
async def queue_page(request: Request, db: Session = Depends(get_db)):
    """Review queue page"""
    claim_repo = ClaimRepository(db)
    claims = claim_repo.get_queue(status=ClaimStatus.ASSESSED, limit=100)

    queue_items = []
    for claim in claims:
        if claim.assessment:
            queue_items.append(_build_queue_item_dict(claim))

    # Sort by priority
    priority_order = {"URGENT": 0, "STANDARD": 1, "LOW_PRIORITY": 2}
    queue_items.sort(key=lambda x: (priority_order.get(x["review_priority"], 3), x["timestamp"]))

    return templates.TemplateResponse(
        "queue.html",
        {
            "request": request,
            "queue_items": queue_items,
            "current_user": "admin",
            "total_pages": 1,
            "current_page": 1,
            "per_page": 100,
            "total_items": len(queue_items),
        },
    )


@router.get("/review/{claim_id}", response_class=HTMLResponse)
async def review_page(request: Request, claim_id: str, db: Session = Depends(get_db)):
    """Claim review page"""
    # Get claim
    claim_repo = ClaimRepository(db)
    claim = claim_repo.get_by_id(claim_id)

    if not claim:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")

    # Get assessment
    assessment_repo = AssessmentRepository(db)
    assessment = assessment_repo.get_by_claim_id(claim_id)

    if not assessment:
        raise HTTPException(status_code=404, detail=f"Assessment not available. Status: {claim.status.value}")

    # Build assessment dict for template
    assessment_dict = _build_assessment_dict(claim, assessment)

    # Video URL (serve from uploads directory)
    video_url = f"/static/videos/{Path(claim.video_path).name}"

    return templates.TemplateResponse(
        "review.html",
        {
            "request": request,
            "claim_id": claim_id,
            "assessment": assessment_dict,
            "video_url": video_url,
            "current_user": "admin",
        },
    )


@router.get("/metrics", response_class=HTMLResponse)
async def metrics_page(request: Request, db: Session = Depends(get_db)):
    """Metrics dashboard page"""
    from insurance_mvp.api.database import Assessment, Claim, Review

    # Calculate metrics
    claim_repo = ClaimRepository(db)
    review_repo = ReviewRepository(db)

    # Processing metrics
    total_claims = db.query(Claim).count()
    claims_today = claim_repo.count_today()

    hour_ago = datetime.utcnow() - timedelta(hours=1)
    claims_last_hour = db.query(Claim).filter(Claim.upload_time >= hour_ago).count()
    processing_rate = claims_last_hour

    # Queue metrics
    queue_depth = claim_repo.count_by_status(ClaimStatus.ASSESSED)

    # Review metrics
    recent_reviews = db.query(Review).order_by(Review.timestamp.desc()).limit(100).all()
    avg_review_time = (
        sum(r.review_time_sec for r in recent_reviews) / len(recent_reviews) if recent_reviews else 0.0
    ) / 60.0  # Convert to minutes

    # AI accuracy (estimate from agreement with human reviews)
    reviewed_claims = claim_repo.get_by_status(ClaimStatus.REVIEWED, limit=100)
    accuracy = 0.89  # Placeholder - calculate from actual agreement

    # Severity distribution
    all_assessments = db.query(Assessment).limit(1000).all()
    severity_counts = {"NONE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for a in all_assessments:
        severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1

    severity_distribution = [
        severity_counts.get("NONE", 0),
        severity_counts.get("LOW", 0),
        severity_counts.get("MEDIUM", 0),
        severity_counts.get("HIGH", 0),
    ]

    # Volume trend (last 7 days)
    volume_trend = []
    for i in range(7):
        day_start = datetime.utcnow().replace(hour=0, minute=0, second=0) - timedelta(days=6 - i)
        day_end = day_start + timedelta(days=1)
        day_count = db.query(Claim).filter(Claim.upload_time >= day_start, Claim.upload_time < day_end).count()
        volume_trend.append(day_count)

    # Accuracy trend (weekly)
    accuracy_trend = [85, 87, 89, 91]  # Placeholder

    # Decision distribution
    approved = claim_repo.count_by_status(ClaimStatus.APPROVED)
    rejected = claim_repo.count_by_status(ClaimStatus.REJECTED)
    pending_info = claim_repo.count_by_status(ClaimStatus.PENDING_INFO)
    decision_distribution = [approved, rejected, pending_info]

    # Recent activity
    recent_activity = []
    for review in recent_reviews[:10]:
        claim = claim_repo.get_by_id(review.claim_id)
        recent_activity.append(
            {
                "timestamp": review.timestamp,
                "claim_id": review.claim_id,
                "event": "REVIEWED",
                "reviewer": review.reviewer_id,
                "decision": review.decision,
                "duration": review.review_time_sec / 60.0,  # minutes
            }
        )

    metrics = {
        "processing_rate": processing_rate,
        "queue_depth": queue_depth,
        "avg_review_time": avg_review_time,
        "accuracy": accuracy,
        "severity_distribution": severity_distribution,
        "volume_trend": volume_trend,
        "accuracy_trend": accuracy_trend,
        "decision_distribution": decision_distribution,
    }

    return templates.TemplateResponse(
        "metrics.html",
        {
            "request": request,
            "metrics": metrics,
            "recent_activity": recent_activity,
            "current_user": "admin",
        },
    )


# Helper functions


def _build_queue_item_dict(claim) -> dict:
    """Build queue item dictionary from claim and assessment"""
    assessment = claim.assessment

    # Parse JSON fields
    prediction_set = json.loads(assessment.prediction_set) if assessment.prediction_set else []

    return {
        "claim_id": claim.id,
        "video_id": Path(claim.video_path).stem if claim.video_path else "unknown",
        "severity": assessment.severity,
        "confidence": assessment.confidence,
        "review_priority": assessment.review_priority,
        "fraud_risk": {
            "risk_score": assessment.fraud_risk_score,
        },
        "prediction_set": prediction_set,
        "timestamp": claim.upload_time,
    }


def _build_assessment_dict(claim, assessment) -> dict:
    """Build assessment dictionary for template"""
    # Parse JSON fields
    prediction_set = json.loads(assessment.prediction_set) if assessment.prediction_set else []
    applicable_rules = json.loads(assessment.applicable_rules) if assessment.applicable_rules else []
    fraud_indicators = json.loads(assessment.fraud_indicators) if assessment.fraud_indicators else []
    hazards_raw = json.loads(assessment.hazards_json) if assessment.hazards_json else []
    evidence_raw = json.loads(assessment.evidence_json) if assessment.evidence_json else []

    return {
        "severity": assessment.severity,
        "confidence": assessment.confidence,
        "prediction_set": prediction_set,
        "review_priority": assessment.review_priority,
        "fault_assessment": {
            "fault_ratio": assessment.fault_ratio,
            "reasoning": assessment.fault_reasoning or "No reasoning provided",
            "applicable_rules": applicable_rules,
            "scenario_type": assessment.scenario_type or "unknown",
            "traffic_signal": assessment.traffic_signal,
            "right_of_way": assessment.right_of_way,
        },
        "fraud_risk": {
            "risk_score": assessment.fraud_risk_score,
            "indicators": fraud_indicators,
            "reasoning": assessment.fraud_reasoning or "No fraud analysis available",
        },
        "hazards": hazards_raw,
        "evidence": evidence_raw,
        "causal_reasoning": assessment.causal_reasoning or "AI assessment in progress",
        "recommended_action": assessment.recommended_action or "REVIEW",
        "video_id": Path(claim.video_path).stem if claim.video_path else "unknown",
        "processing_time_sec": assessment.processing_time_sec,
        "timestamp": assessment.timestamp,
    }
