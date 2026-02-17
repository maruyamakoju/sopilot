# Insurance MVP API

Production-ready FastAPI backend for insurance claim review system with AI assessment.

## Features

- **Video Upload**: Dashcam video upload with validation and deduplication
- **AI Assessment**: Automated claim evaluation (severity, fault ratio, fraud risk)
- **Review Queue**: Priority-based review queue for human oversight
- **Audit Trail**: Complete audit log for regulatory compliance
- **Metrics**: Real-time system metrics and statistics
- **Security**: API key authentication with permission-based access control
- **Rate Limiting**: 60 requests/minute per API key
- **Background Processing**: Asynchronous video processing with progress tracking

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install fastapi uvicorn sqlalchemy pydantic python-multipart

# Or install entire project
pip install -e ".[api]"
```

### 2. Run Server

```bash
# Development mode (auto-reload, test API keys)
python -m insurance_mvp.api.main

# Production mode
uvicorn insurance_mvp.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Claims

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/claims/upload` | Upload dashcam video | Required (write) |
| GET | `/claims/{id}/status` | Check processing status | Required |
| GET | `/claims/{id}/assessment` | Get AI assessment | Required |

### Reviews

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/reviews/queue` | Get review queue (sorted by priority) | Required |
| POST | `/reviews/{id}/decision` | Submit human review decision | Required (write) |
| GET | `/reviews/{id}/history` | Get audit log for claim | Required |

### System

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/health` | Health check | None |
| GET | `/metrics` | System metrics | Optional |

## Authentication

### API Keys

The API uses API key authentication via `X-API-Key` header or Bearer token.

**Development Mode** (auto-generated on startup):
```bash
# Full access key (printed on startup)
X-API-Key: ins_<random_key>

# Or Bearer token
Authorization: Bearer ins_<random_key>
```

**Production Mode**:
```python
from insurance_mvp.api.auth import api_key_store

# Generate production key
api_key = api_key_store.generate_key(
    name="production-frontend",
    permissions={"read": True, "write": True, "admin": False}
)
print(f"API Key: {api_key}")  # Store securely!
```

### Permissions

- **read**: Access GET endpoints
- **write**: Access POST/PUT/DELETE endpoints
- **admin**: Access admin endpoints (dev mode only)

## Usage Examples

### 1. Upload Video

```bash
curl -X POST "http://localhost:8000/claims/upload" \
  -H "X-API-Key: ins_your_api_key_here" \
  -F "video=@dashcam001.mp4" \
  -F "claim_number=CLM-2026-001234" \
  -F "claimant_id=CUSTOMER-5678"
```

Response:
```json
{
  "claim_id": "claim_20260217103000_abc123def456",
  "status": "queued",
  "message": "Video uploaded successfully and queued for processing",
  "upload_time": "2026-02-17T10:30:00Z"
}
```

### 2. Check Status

```bash
curl -X GET "http://localhost:8000/claims/claim_20260217103000_abc123def456/status" \
  -H "X-API-Key: ins_your_api_key_here"
```

Response:
```json
{
  "claim_id": "claim_20260217103000_abc123def456",
  "status": "processing",
  "message": "AI assessment in progress",
  "upload_time": "2026-02-17T10:30:00Z",
  "processing_started": "2026-02-17T10:30:15Z",
  "progress_percent": 65.0
}
```

### 3. Get AI Assessment

```bash
curl -X GET "http://localhost:8000/claims/claim_20260217103000_abc123def456/assessment" \
  -H "X-API-Key: ins_your_api_key_here"
```

Response:
```json
{
  "claim_id": "claim_20260217103000_abc123def456",
  "severity": "MEDIUM",
  "confidence": 0.87,
  "prediction_set": ["MEDIUM", "HIGH"],
  "review_priority": "STANDARD",
  "fault_assessment": {
    "fault_ratio": 30.0,
    "reasoning": "Driver braked appropriately but slightly delayed reaction",
    "applicable_rules": ["Following distance rule", "Defensive driving"],
    "scenario_type": "rear_end"
  },
  "fraud_risk": {
    "risk_score": 0.12,
    "indicators": [],
    "reasoning": "No suspicious patterns detected"
  },
  "hazards": [...],
  "evidence": [...],
  "causal_reasoning": "...",
  "recommended_action": "REVIEW"
}
```

### 4. Get Review Queue

```bash
curl -X GET "http://localhost:8000/reviews/queue?priority=URGENT&limit=50" \
  -H "X-API-Key: ins_your_api_key_here"
```

### 5. Submit Review Decision

```bash
curl -X POST "http://localhost:8000/reviews/claim_abc123/decision?reviewer_id=reviewer_alice" \
  -H "X-API-Key: ins_your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "decision": "APPROVE",
    "reasoning": "AI assessment is accurate. Clear evidence of no-fault scenario.",
    "severity_override": null,
    "fault_ratio_override": null,
    "fraud_override": false,
    "comments": "Good training example"
  }'
```

### 6. Get Audit History

```bash
curl -X GET "http://localhost:8000/reviews/claim_abc123/history" \
  -H "X-API-Key: ins_your_api_key_here"
```

### 7. System Metrics

```bash
curl -X GET "http://localhost:8000/metrics" \
  -H "X-API-Key: ins_your_api_key_here"
```

Response:
```json
{
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
  "approval_rate": 0.72,
  "average_ai_confidence": 0.84,
  "error_rate": 0.05
}
```

## Python Client Example

```python
import requests
from pathlib import Path

API_BASE = "http://localhost:8000"
API_KEY = "ins_your_api_key_here"

class InsuranceClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}

    def upload_video(self, video_path: Path, claim_number: str = None):
        """Upload dashcam video"""
        with open(video_path, "rb") as f:
            files = {"video": f}
            params = {}
            if claim_number:
                params["claim_number"] = claim_number

            response = requests.post(
                f"{self.base_url}/claims/upload",
                headers=self.headers,
                files=files,
                params=params,
            )
            response.raise_for_status()
            return response.json()

    def get_status(self, claim_id: str):
        """Get claim status"""
        response = requests.get(
            f"{self.base_url}/claims/{claim_id}/status",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def get_assessment(self, claim_id: str):
        """Get AI assessment"""
        response = requests.get(
            f"{self.base_url}/claims/{claim_id}/assessment",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def get_queue(self, priority: str = None, limit: int = 100):
        """Get review queue"""
        params = {"limit": limit}
        if priority:
            params["priority"] = priority

        response = requests.get(
            f"{self.base_url}/reviews/queue",
            headers=self.headers,
            params=params,
        )
        response.raise_for_status()
        return response.json()

    def submit_review(self, claim_id: str, reviewer_id: str, decision: dict):
        """Submit review decision"""
        response = requests.post(
            f"{self.base_url}/reviews/{claim_id}/decision",
            headers=self.headers,
            params={"reviewer_id": reviewer_id},
            json=decision,
        )
        response.raise_for_status()
        return response.json()

# Usage
client = InsuranceClient(API_BASE, API_KEY)

# Upload video
result = client.upload_video(Path("dashcam001.mp4"), claim_number="CLM-001")
claim_id = result["claim_id"]

# Wait for processing...
import time
while True:
    status = client.get_status(claim_id)
    if status["status"] == "assessed":
        break
    time.sleep(2)

# Get assessment
assessment = client.get_assessment(claim_id)
print(f"Severity: {assessment['severity']}")
print(f"Confidence: {assessment['confidence']}")
print(f"Fault ratio: {assessment['fault_assessment']['fault_ratio']}%")
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=sqlite:///./insurance.db  # or postgresql://user:pass@localhost/insurance
DATABASE_ECHO=false

# Storage
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE_MB=500

# Worker
WORKER_MAX_THREADS=4
USE_PIPELINE=false  # Set to true for actual pipeline integration

# API
CORS_ORIGINS=*  # Comma-separated origins
DEV_MODE=true   # Set to false in production

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

### Production Deployment

```bash
# 1. Set production environment
export DEV_MODE=false
export DATABASE_URL=postgresql://user:pass@prod-db/insurance
export UPLOAD_DIR=/mnt/storage/uploads

# 2. Generate production API keys
python -c "
from insurance_mvp.api.auth import api_key_store
key = api_key_store.generate_key('prod-frontend', {'read': True, 'write': True})
print(f'API Key: {key}')
"

# 3. Run with gunicorn (production WSGI server)
gunicorn insurance_mvp.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile /var/log/insurance-api/access.log \
  --error-logfile /var/log/insurance-api/error.log
```

## Database Schema

### Tables

1. **claims** - Uploaded videos and processing status
2. **assessments** - AI evaluation results
3. **reviews** - Human review decisions
4. **audit_logs** - Immutable audit trail

### Migration (Production)

For production, use Alembic for database migrations:

```bash
# Initialize migrations
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Initial schema"

# Apply migration
alembic upgrade head
```

## Testing

```bash
# Run tests
pytest insurance_mvp/tests/test_api.py -v

# With coverage
pytest insurance_mvp/tests/test_api.py --cov=insurance_mvp.api --cov-report=html
```

## Monitoring

### Health Check

```bash
# Basic health check
curl http://localhost:8000/health

# With monitoring tool (e.g., Prometheus)
# Configure /metrics endpoint for Prometheus scraping
```

### Logging

Logs are written to stdout in JSON format for structured logging:

```json
{
  "timestamp": "2026-02-17T10:30:00Z",
  "level": "INFO",
  "logger": "insurance_mvp.api.main",
  "message": "Video uploaded: dashcam001.mp4 (125.4 MB)",
  "claim_id": "claim_abc123"
}
```

## Security Best Practices

1. **API Keys**: Generate strong keys, store in environment variables
2. **HTTPS**: Always use HTTPS in production (configure reverse proxy)
3. **Rate Limiting**: Adjust limits based on expected load
4. **File Upload**: Validate file types and sizes
5. **Database**: Use connection pooling and prepared statements
6. **CORS**: Restrict origins in production
7. **Audit Logging**: Enable for all state changes

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP/HTTPS
       ▼
┌─────────────────────────────┐
│      FastAPI App            │
│  ┌──────────────────────┐   │
│  │   Auth Middleware    │   │
│  └──────────────────────┘   │
│  ┌──────────────────────┐   │
│  │   Rate Limiter       │   │
│  └──────────────────────┘   │
│  ┌──────────────────────┐   │
│  │   API Endpoints      │   │
│  └──────────────────────┘   │
└──────┬─────────────┬────────┘
       │             │
       ▼             ▼
┌─────────────┐ ┌──────────────┐
│  Database   │ │  Background  │
│  (SQLite/   │ │   Worker     │
│  PostgreSQL)│ │ (ThreadPool) │
└─────────────┘ └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │   Pipeline   │
                │  (AI Models) │
                └──────────────┘
```

## Troubleshooting

### Common Issues

**1. "Database not initialized"**
```bash
# Manually initialize database
python -c "
from insurance_mvp.api.database import DatabaseManager
db = DatabaseManager('sqlite:///./insurance.db')
db.create_tables()
"
```

**2. "Background worker not active"**
- Check worker initialization in startup event
- Verify max_workers setting
- Check for processing errors in logs

**3. "Rate limit exceeded"**
- Wait 60 seconds
- Reduce request frequency
- Request increased rate limit

## Contributing

See main project README for contribution guidelines.

## License

Proprietary - Internal use only
