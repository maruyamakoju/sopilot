# Insurance MVP API - Quick Start Guide

Get up and running in 5 minutes!

## 1. Install Dependencies

```bash
# Navigate to project root
cd C:/Users/07013/Desktop/02081

# Install FastAPI and dependencies
pip install fastapi uvicorn sqlalchemy pydantic python-multipart
```

## 2. Start Server

```bash
# Run development server
python -m insurance_mvp.api.main
```

You should see:
```
[DEV] Full access API key: ins_AbCd1234EfGh5678...
[DEV] Read-only API key: ins_XyZ9876WvUt5432...
[DEV] Reviewers initialized: reviewer_alice, reviewer_bob
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**IMPORTANT**: Copy the API keys! You'll need them.

## 3. Test the API

### Option A: Browser (Interactive)

Open http://localhost:8000/docs in your browser.

- Click "Authorize" button
- Enter your API key: `ins_AbCd1234...`
- Try endpoints interactively

### Option B: cURL (Command Line)

```bash
# Set your API key
export API_KEY="ins_AbCd1234EfGh5678..."  # Replace with your key

# 1. Health check
curl http://localhost:8000/health

# 2. Get metrics
curl -H "X-API-Key: $API_KEY" http://localhost:8000/metrics

# 3. Get review queue (empty initially)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/reviews/queue

# 4. Upload video (requires actual video file)
curl -X POST http://localhost:8000/claims/upload \
  -H "X-API-Key: $API_KEY" \
  -F "video=@path/to/dashcam.mp4" \
  -F "claim_number=TEST-001"
```

### Option C: Python Client

```python
from insurance_mvp.api.example_client import InsuranceAPIClient

# Initialize client
client = InsuranceAPIClient(
    base_url="http://localhost:8000",
    api_key="ins_AbCd1234..."  # Your API key
)

# Health check
health = client.health_check()
print(f"Status: {health['status']}")

# Get metrics
metrics = client.get_metrics()
print(f"Total Claims: {metrics['total_claims']}")
print(f"Queue Depth: {metrics['queue_depth']}")

# Upload video (requires actual file)
# claim_id = client.upload_video("dashcam.mp4", claim_number="TEST-001")
# assessment = client.wait_for_assessment(claim_id)
# print(f"Severity: {assessment['severity']}")
```

## 4. Complete Workflow Example

Here's a full workflow from upload to review:

```python
from insurance_mvp.api.example_client import InsuranceAPIClient
import time

# Initialize
client = InsuranceAPIClient("http://localhost:8000", "ins_your_api_key")

# 1. Upload video
print("1. Uploading video...")
claim_id = client.upload_video(
    video_path="dashcam.mp4",
    claim_number="DEMO-001",
    claimant_id="CUST-123"
)
print(f"   Claim ID: {claim_id}")

# 2. Wait for processing
print("2. Waiting for AI assessment...")
assessment = client.wait_for_assessment(claim_id, timeout=120)
print(f"   Severity: {assessment['severity']}")
print(f"   Confidence: {assessment['confidence']:.1%}")
print(f"   Fault Ratio: {assessment['fault_assessment']['fault_ratio']:.1f}%")
print(f"   Fraud Risk: {assessment['fraud_risk']['risk_score']:.1%}")

# 3. Review queue
print("3. Checking review queue...")
queue = client.get_queue(priority="URGENT")
print(f"   Urgent claims: {queue['urgent_count']}")

# 4. Submit human review
print("4. Submitting review decision...")
result = client.submit_review(
    claim_id=claim_id,
    reviewer_id="reviewer_alice",
    decision="APPROVE",
    reasoning="AI assessment verified. Video evidence is clear. No fraud indicators.",
    fraud_override=False
)
print(f"   Status: {result['status']}")

# 5. Audit history
print("5. Viewing audit history...")
history = client.get_history(claim_id)
print(f"   Total events: {history['total_events']}")
for event in history['events'][:3]:
    print(f"   - {event['event_type']}: {event['explanation']}")

print("\n✓ Complete workflow finished!")
```

## 5. API Endpoints Reference

### Claims
- `POST /claims/upload` - Upload video
- `GET /claims/{id}/status` - Check status
- `GET /claims/{id}/assessment` - Get AI results

### Reviews
- `GET /reviews/queue` - Get review queue
- `POST /reviews/{id}/decision` - Submit review
- `GET /reviews/{id}/history` - Audit log

### System
- `GET /health` - Health check
- `GET /metrics` - System metrics

## 6. Common Tasks

### Check Server Status
```bash
curl http://localhost:8000/health
```

### View API Documentation
```bash
# Open in browser
open http://localhost:8000/docs
```

### Stop Server
```bash
# Press Ctrl+C in terminal where server is running
```

### Reset Database (Development)
```bash
# With admin API key
curl -X POST http://localhost:8000/dev/reset \
  -H "X-API-Key: $ADMIN_API_KEY"
```

## 7. Troubleshooting

### "Module not found"
```bash
# Make sure you're in project root
cd C:/Users/07013/Desktop/02081

# Reinstall in development mode
pip install -e .
```

### "Port 8000 already in use"
```bash
# Kill existing process
# Windows:
netstat -ano | findstr :8000
taskkill /PID <process_id> /F

# Linux/Mac:
lsof -ti:8000 | xargs kill -9
```

### "Database locked"
```bash
# Remove SQLite lock file
rm insurance.db-journal

# Or use different database
export DATABASE_URL="sqlite:///./test.db"
```

### "API key invalid"
```bash
# Generate new key in Python:
python -c "
from insurance_mvp.api.auth import api_key_store
key = api_key_store.generate_key('my-key', {'read': True, 'write': True})
print(f'API Key: {key}')
"
```

## 8. Next Steps

Once you're comfortable with the basics:

1. **Integration**: Connect to actual ML pipeline
   - Edit `background.py` → `PipelineProcessor.process_claim()`
   - Replace mock data with actual inference

2. **Production**: Deploy to server
   - See [SETUP.md](SETUP.md) for deployment guide
   - Configure PostgreSQL
   - Set up Nginx reverse proxy

3. **Frontend**: Build web dashboard
   - Use API client from `example_client.py`
   - React/Vue/Svelte + API
   - Or use included templates in `/templates`

4. **Testing**: Run test suite
   ```bash
   pytest insurance_mvp/tests/test_api_basic.py -v
   ```

## Need Help?

- **API Docs**: http://localhost:8000/docs
- **README**: [README.md](README.md) - Full API documentation
- **Setup Guide**: [SETUP.md](SETUP.md) - Production deployment
- **Summary**: [API_IMPLEMENTATION_SUMMARY.md](../API_IMPLEMENTATION_SUMMARY.md)

## Quick Reference Card

```
╔══════════════════════════════════════════════════════════════╗
║                   INSURANCE MVP API                          ║
╠══════════════════════════════════════════════════════════════╣
║  Start Server:  python -m insurance_mvp.api.main             ║
║  Docs:          http://localhost:8000/docs                   ║
║  Health:        http://localhost:8000/health                 ║
║                                                              ║
║  Auth Header:   X-API-Key: ins_your_key_here                ║
║  Or Bearer:     Authorization: Bearer ins_your_key_here      ║
╠══════════════════════════════════════════════════════════════╣
║  Upload:        POST /claims/upload                          ║
║  Status:        GET  /claims/{id}/status                     ║
║  Assessment:    GET  /claims/{id}/assessment                 ║
║  Queue:         GET  /reviews/queue                          ║
║  Review:        POST /reviews/{id}/decision                  ║
║  History:       GET  /reviews/{id}/history                   ║
║  Metrics:       GET  /metrics                                ║
╚══════════════════════════════════════════════════════════════╝
```

Happy coding! 🚀
