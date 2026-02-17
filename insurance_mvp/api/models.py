"""API Request/Response Models

Pydantic models for API endpoints.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


# Enums for type safety

class ClaimStatus(str, Enum):
    """Claim processing status"""
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    ASSESSED = "assessed"
    UNDER_REVIEW = "under_review"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING_INFO = "pending_info"
    FAILED = "failed"


class Severity(str, Enum):
    """Severity levels"""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ReviewPriority(str, Enum):
    """Review priority levels"""
    URGENT = "URGENT"
    STANDARD = "STANDARD"
    LOW_PRIORITY = "LOW_PRIORITY"


class ReviewDecisionType(str, Enum):
    """Review decision types"""
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_MORE_INFO = "REQUEST_MORE_INFO"


class EventType(str, Enum):
    """Audit log event types"""
    CLAIM_UPLOADED = "CLAIM_UPLOADED"
    AI_ASSESSMENT = "AI_ASSESSMENT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    DECISION_CHANGE = "DECISION_CHANGE"
    STATUS_UPDATE = "STATUS_UPDATE"
    FRAUD_FLAG = "FRAUD_FLAG"


# Request Models

class UploadRequest(BaseModel):
    """Video upload request metadata"""
    claim_number: Optional[str] = Field(None, description="Optional claim reference number")
    claimant_id: Optional[str] = Field(None, description="Optional claimant identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "claim_number": "CLM-2026-001234",
                "claimant_id": "CUSTOMER-5678",
                "metadata": {
                    "incident_date": "2026-02-15",
                    "location": "Tokyo, Shibuya"
                }
            }
        }


class ReviewDecisionRequest(BaseModel):
    """Human review decision request"""
    decision: ReviewDecisionType = Field(description="Review decision")
    reasoning: str = Field(min_length=10, description="Detailed reasoning (min 10 characters)")

    # Optional overrides
    severity_override: Optional[Severity] = Field(None, description="Override AI severity assessment")
    fault_ratio_override: Optional[float] = Field(None, ge=0.0, le=100.0, description="Override fault ratio (0-100%)")
    fraud_override: Optional[bool] = Field(None, description="Override fraud risk flag")

    comments: Optional[str] = Field(None, description="Additional comments")

    @validator('reasoning')
    def validate_reasoning(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Reasoning must be at least 10 characters after stripping whitespace')
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "decision": "APPROVE",
                "reasoning": "AI assessment is accurate. Clear evidence of no-fault scenario with proper defensive driving.",
                "severity_override": None,
                "fault_ratio_override": None,
                "fraud_override": False,
                "comments": "Recommend using this case for training data."
            }
        }


# Response Models

class UploadResponse(BaseModel):
    """Video upload response"""
    claim_id: str = Field(description="Unique claim identifier")
    status: ClaimStatus = Field(description="Current processing status")
    message: str = Field(description="Human-readable status message")
    upload_time: datetime = Field(description="Upload timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "claim_id": "claim_abc123def456",
                "status": "queued",
                "message": "Video uploaded successfully and queued for processing",
                "upload_time": "2026-02-17T10:30:00Z"
            }
        }


class StatusResponse(BaseModel):
    """Claim status response"""
    claim_id: str
    status: ClaimStatus
    message: str
    upload_time: datetime
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None
    progress_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    error_message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "claim_id": "claim_abc123def456",
                "status": "processing",
                "message": "AI assessment in progress",
                "upload_time": "2026-02-17T10:30:00Z",
                "processing_started": "2026-02-17T10:30:15Z",
                "processing_completed": None,
                "progress_percent": 45.0,
                "error_message": None
            }
        }


class EvidenceItem(BaseModel):
    """Evidence from video"""
    timestamp_sec: float
    description: str
    frame_path: Optional[str] = None


class HazardItem(BaseModel):
    """Hazard detection"""
    type: str
    actors: List[str]
    spatial_relation: str
    timestamp_sec: float


class FaultAssessmentResponse(BaseModel):
    """Fault assessment response"""
    fault_ratio: float = Field(ge=0.0, le=100.0)
    reasoning: str
    applicable_rules: List[str]
    scenario_type: str
    traffic_signal: Optional[str] = None
    right_of_way: Optional[str] = None


class FraudRiskResponse(BaseModel):
    """Fraud risk response"""
    risk_score: float = Field(ge=0.0, le=1.0)
    indicators: List[str]
    reasoning: str


class AssessmentResponse(BaseModel):
    """Complete AI assessment response"""
    claim_id: str

    # Core assessment
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    prediction_set: List[Severity] = Field(description="Conformal prediction set")
    review_priority: ReviewPriority

    # Detailed assessments
    fault_assessment: FaultAssessmentResponse
    fraud_risk: FraudRiskResponse

    # Evidence
    hazards: List[HazardItem]
    evidence: List[EvidenceItem]

    # Reasoning
    causal_reasoning: str
    recommended_action: str

    # Metadata
    processing_time_sec: float
    timestamp: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "claim_id": "claim_abc123def456",
                "severity": "MEDIUM",
                "confidence": 0.87,
                "prediction_set": ["MEDIUM", "HIGH"],
                "review_priority": "STANDARD",
                "fault_assessment": {
                    "fault_ratio": 30.0,
                    "reasoning": "Driver braked appropriately but slightly delayed reaction",
                    "applicable_rules": ["Following distance rule", "Defensive driving"],
                    "scenario_type": "rear_end",
                    "traffic_signal": None,
                    "right_of_way": None
                },
                "fraud_risk": {
                    "risk_score": 0.12,
                    "indicators": [],
                    "reasoning": "No suspicious patterns detected"
                },
                "hazards": [
                    {
                        "type": "near_miss",
                        "actors": ["car", "pedestrian"],
                        "spatial_relation": "front",
                        "timestamp_sec": 12.5
                    }
                ],
                "evidence": [
                    {
                        "timestamp_sec": 12.5,
                        "description": "Pedestrian suddenly entered crosswalk",
                        "frame_path": "/data/frames/claim_abc123def456_frame_0125.jpg"
                    }
                ],
                "causal_reasoning": "Near-miss incident caused by pedestrian crossing without looking. Driver response time was adequate.",
                "recommended_action": "REVIEW",
                "processing_time_sec": 23.4,
                "timestamp": "2026-02-17T10:30:45Z"
            }
        }


class QueueItem(BaseModel):
    """Single item in review queue"""
    claim_id: str
    upload_time: datetime
    review_priority: ReviewPriority
    severity: Severity
    confidence: float
    fraud_risk_score: float
    prediction_set_size: int = Field(description="Number of possible severity levels (uncertainty)")

    # Preview data
    hazard_count: int
    fault_ratio: float
    recommended_action: str

    class Config:
        json_schema_extra = {
            "example": {
                "claim_id": "claim_abc123def456",
                "upload_time": "2026-02-17T10:30:00Z",
                "review_priority": "URGENT",
                "severity": "HIGH",
                "confidence": 0.65,
                "fraud_risk_score": 0.78,
                "prediction_set_size": 3,
                "hazard_count": 2,
                "fault_ratio": 75.0,
                "recommended_action": "REVIEW"
            }
        }


class QueueResponse(BaseModel):
    """Review queue response"""
    total_count: int
    items: List[QueueItem]

    # Queue statistics
    urgent_count: int
    standard_count: int
    low_priority_count: int

    class Config:
        json_schema_extra = {
            "example": {
                "total_count": 42,
                "items": [],  # List of QueueItem objects
                "urgent_count": 5,
                "standard_count": 28,
                "low_priority_count": 9
            }
        }


class ReviewDecisionResponse(BaseModel):
    """Review decision response"""
    claim_id: str
    status: ClaimStatus
    message: str
    review_time: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "claim_id": "claim_abc123def456",
                "status": "approved",
                "message": "Claim approved by reviewer",
                "review_time": "2026-02-17T11:45:00Z"
            }
        }


class AuditLogEntry(BaseModel):
    """Single audit log entry"""
    log_id: int
    claim_id: str
    event_type: EventType
    actor_type: str = Field(description="AI or HUMAN")
    actor_id: str
    explanation: str
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    timestamp: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "log_id": 1234,
                "claim_id": "claim_abc123def456",
                "event_type": "HUMAN_REVIEW",
                "actor_type": "HUMAN",
                "actor_id": "reviewer_john_doe",
                "explanation": "Reviewed and approved claim with minor severity override",
                "before_state": {"status": "under_review", "severity": "MEDIUM"},
                "after_state": {"status": "approved", "severity": "LOW"},
                "timestamp": "2026-02-17T11:45:00Z"
            }
        }


class AuditHistoryResponse(BaseModel):
    """Audit history response"""
    claim_id: str
    total_events: int
    events: List[AuditLogEntry]

    class Config:
        json_schema_extra = {
            "example": {
                "claim_id": "claim_abc123def456",
                "total_events": 4,
                "events": []  # List of AuditLogEntry objects
            }
        }


class MetricsResponse(BaseModel):
    """System metrics response"""

    # Processing metrics
    total_claims: int
    claims_today: int
    processing_rate_per_hour: float
    average_processing_time_sec: float

    # Queue metrics
    queue_depth: int
    queue_depth_by_priority: Dict[str, int]

    # Review metrics
    pending_review_count: int
    reviewed_today: int
    average_review_time_sec: float

    # Decision metrics
    approval_rate: float = Field(ge=0.0, le=1.0)
    rejection_rate: float = Field(ge=0.0, le=1.0)

    # AI metrics
    average_ai_confidence: float = Field(ge=0.0, le=1.0)
    average_fraud_risk: float = Field(ge=0.0, le=1.0)

    # Error metrics
    failed_processing_count: int
    error_rate: float = Field(ge=0.0, le=1.0)

    timestamp: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "total_claims": 15847,
                "claims_today": 234,
                "processing_rate_per_hour": 52.3,
                "average_processing_time_sec": 18.7,
                "queue_depth": 42,
                "queue_depth_by_priority": {
                    "URGENT": 5,
                    "STANDARD": 28,
                    "LOW_PRIORITY": 9
                },
                "pending_review_count": 42,
                "reviewed_today": 178,
                "average_review_time_sec": 145.2,
                "approval_rate": 0.72,
                "rejection_rate": 0.18,
                "average_ai_confidence": 0.84,
                "average_fraud_risk": 0.23,
                "failed_processing_count": 12,
                "error_rate": 0.05,
                "timestamp": "2026-02-17T12:00:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request data",
                "detail": "Field 'reasoning' must be at least 10 characters",
                "timestamp": "2026-02-17T12:00:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(description="healthy or unhealthy")
    version: str
    uptime_seconds: float
    database_connected: bool
    background_worker_active: bool
    timestamp: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 86400.5,
                "database_connected": True,
                "background_worker_active": True,
                "timestamp": "2026-02-17T12:00:00Z"
            }
        }
