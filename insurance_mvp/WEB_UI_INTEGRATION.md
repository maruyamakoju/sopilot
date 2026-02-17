# Web UI Integration Guide

This guide shows how to integrate the production-quality web UI with the existing Insurance MVP API.

## Quick Start

### 1. Mount Web Routes

Add these lines to `insurance_mvp/api/main.py`:

```python
# At the top with other imports
from insurance_mvp.api.web_routes import router as web_router
from fastapi.staticfiles import StaticFiles

# After creating the FastAPI app
app = FastAPI(...)

# Mount web UI routes
app.include_router(web_router)

# Mount static files (for videos, images, etc.)
app.mount("/static/videos", StaticFiles(directory="data/uploads"), name="videos")
app.mount("/static/frames", StaticFiles(directory="data/frames"), name="frames")
```

### 2. Run the Server

```bash
# Start the API server
python -m insurance_mvp.api.main

# Or with uvicorn directly
uvicorn insurance_mvp.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access the Web UI

Open your browser and navigate to:

- **Review Queue**: http://localhost:8000/queue
- **Metrics Dashboard**: http://localhost:8000/metrics
- **Specific Claim Review**: http://localhost:8000/review/{claim_id}

## Integration Details

### Template Location

Templates are located in: `insurance_mvp/templates/`

- `base.html` - Base template with navigation and common elements
- `queue.html` - Review queue page
- `review.html` - Claim review page
- `metrics.html` - Metrics dashboard

### Database Integration

The web routes use the existing database repositories:

```python
from insurance_mvp.api.database import (
    ClaimRepository,
    AssessmentRepository,
    ReviewRepository,
)
```

All data is fetched from the same SQLite/PostgreSQL database used by the API.

### Authentication (Optional)

To add authentication, modify the routes in `web_routes.py`:

```python
from insurance_mvp.api.auth import get_current_user, require_permission

@router.get("/queue", response_class=HTMLResponse)
async def queue_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),  # Add authentication
):
    # ... route logic
```

### Static File Serving

For production, serve static files through a reverse proxy (nginx, Caddy):

```nginx
# nginx configuration
location /static/ {
    alias /var/www/insurance_mvp/data/;
    expires 1d;
    add_header Cache-Control "public, immutable";
}

location / {
    proxy_pass http://localhost:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

## API Endpoints for Web UI

The web UI consumes these JSON API endpoints:

### 1. Get Queue Items
```
GET /reviews/queue?priority=URGENT&limit=100
```

Returns list of claims ready for review, sorted by priority.

### 2. Get Claim Assessment
```
GET /claims/{claim_id}/assessment
```

Returns AI assessment with severity, fault ratio, fraud risk, and evidence.

### 3. Submit Review Decision
```
POST /api/reviews
Content-Type: application/json

{
  "claim_id": "claim_...",
  "reviewer_id": "admin",
  "decision": "APPROVE",
  "reasoning": "Assessment is accurate...",
  "review_time_sec": 180.5,
  "severity_override": null,
  "fault_ratio_override": null,
  "fraud_override": null,
  "comments": null
}
```

### 4. Get Metrics
```
GET /api/metrics
```

Returns system metrics for dashboard charts.

## Customization

### Branding

Edit `templates/base.html`:

```html
<!-- Change logo and title -->
<span class="ml-3 text-xl font-semibold text-gray-900">
    Your Company Name
</span>

<!-- Update footer -->
<p>&copy; 2026 Your Company. All rights reserved.</p>
```

### Colors

Modify TailwindCSS classes in templates:

```html
<!-- Change primary color from blue to your brand color -->
<button class="bg-blue-600 hover:bg-blue-700">
<!-- To -->
<button class="bg-purple-600 hover:bg-purple-700">
```

### Thresholds

Edit risk/confidence thresholds in templates:

```python
# In review.html
{% if assessment.fraud_risk.risk_score > 0.7 %}
    <span class="status-indicator status-error">HIGH RISK</span>
{% elif assessment.fraud_risk.risk_score > 0.4 %}
    <span class="status-indicator status-warning">MEDIUM RISK</span>
{% else %}
    <span class="status-indicator status-healthy">LOW RISK</span>
{% endif %}
```

## Testing

### 1. Populate Test Data

Use the API to create test claims:

```bash
# Upload a test video
curl -X POST http://localhost:8000/claims/upload \
  -H "X-API-Key: dev_key_write" \
  -F "video=@test_dashcam.mp4" \
  -F "claim_number=TEST-001"

# Check processing status
curl http://localhost:8000/claims/{claim_id}/status \
  -H "X-API-Key: dev_key_read"
```

### 2. Access Web UI

Once claims are processed (status = ASSESSED), they will appear in the queue:

```
http://localhost:8000/queue
```

### 3. Submit Test Review

1. Click on a claim in the queue
2. Review the AI assessment
3. Select decision (APPROVE/REJECT/REQUEST_MORE_INFO)
4. Fill in reasoning
5. Click "Submit Review"

## Production Deployment

### Environment Variables

```bash
# .env file
DATABASE_URL=postgresql://user:pass@localhost/insurance_db
UPLOAD_DIR=/var/www/insurance_mvp/uploads
MAX_UPLOAD_SIZE_MB=1000
DEV_MODE=false
CORS_ORIGINS=https://claims.yourcompany.com
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/insurance
      - DEV_MODE=false
    volumes:
      - ./data/uploads:/app/data/uploads
      - ./data/frames:/app/data/frames
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: insurance
      POSTGRES_PASSWORD: password
    volumes:
      - pgdata:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./data/uploads:/var/www/static/videos
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api

volumes:
  pgdata:
```

### SSL/TLS Setup

Use Let's Encrypt with Certbot:

```bash
# Install certbot
apt-get install certbot python3-certbot-nginx

# Get certificate
certbot --nginx -d claims.yourcompany.com

# Auto-renewal
certbot renew --dry-run
```

### Reverse Proxy (nginx)

```nginx
upstream api_backend {
    server localhost:8000;
}

server {
    listen 80;
    server_name claims.yourcompany.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name claims.yourcompany.com;

    ssl_certificate /etc/letsencrypt/live/claims.yourcompany.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/claims.yourcompany.com/privkey.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Static files
    location /static/ {
        alias /var/www/insurance_mvp/data/;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }

    # API and Web UI
    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Monitoring

### Application Logs

```bash
# View logs
tail -f logs/insurance_mvp.log

# Search for errors
grep ERROR logs/insurance_mvp.log
```

### Health Check

```bash
# Check API health
curl http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "database_connected": true,
  "background_worker_active": true,
  "timestamp": "2026-02-17T12:00:00Z"
}
```

### Metrics Dashboard

The `/metrics` page shows real-time system metrics:
- Processing rate
- Queue depth
- Average review time
- AI accuracy
- Error rates

## Troubleshooting

### Issue: Templates not found

**Solution**: Check template directory path in `web_routes.py`:

```python
template_dir = Path(__file__).parent.parent / "templates"
print(f"Template directory: {template_dir}")
print(f"Exists: {template_dir.exists()}")
```

### Issue: Video files not loading

**Solution**: Ensure static file mounting in `main.py`:

```python
app.mount("/static/videos", StaticFiles(directory="data/uploads"), name="videos")
```

And verify files exist:

```bash
ls -la data/uploads/
```

### Issue: Database connection error

**Solution**: Check `DATABASE_URL` environment variable:

```bash
echo $DATABASE_URL

# Or in Python
import os
print(os.getenv("DATABASE_URL"))
```

### Issue: Slow page load

**Solution**:
1. Enable template caching in production
2. Add database indexes
3. Optimize queries in repositories
4. Use CDN for assets

## Next Steps

1. **Add Authentication**: Integrate OAuth2 or SAML
2. **Enable HTTPS**: Set up SSL certificates
3. **Add Monitoring**: Integrate Sentry, DataDog, or New Relic
4. **Optimize Performance**: Enable caching, CDN
5. **User Management**: Add role-based access control
6. **Custom Workflows**: Extend review states
7. **Reporting**: Add PDF export for audit reports

## Support

For questions or issues:
- Check logs: `logs/insurance_mvp.log`
- Review API docs: http://localhost:8000/docs
- Contact: [your-team@company.com]

---

**Last Updated**: 2026-02-17
