# Insurance MVP API - Setup Guide

Complete setup and deployment guide for the production-ready FastAPI backend.

## Quick Start (Development)

### 1. Install Dependencies

```bash
# Install FastAPI and core dependencies
pip install fastapi uvicorn sqlalchemy pydantic python-multipart

# Or install from project
cd /path/to/project
pip install -e .
```

### 2. Run Development Server

```bash
# Navigate to project root
cd C:/Users/07013/Desktop/02081

# Run with auto-reload
python -m insurance_mvp.api.main
```

Output:
```
[DEV] Full access API key: ins_<generated_key>
[DEV] Read-only API key: ins_<generated_key>
[DEV] Reviewers initialized: reviewer_alice, reviewer_bob
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Save the API keys!** You'll need them for authentication.

### 3. Access API Documentation

Open browser to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### 4. Test with Example Client

```bash
# Edit example_client.py to add your API key
python insurance_mvp/api/example_client.py
```

## Production Deployment

### Step 1: Environment Configuration

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql://insurance_user:secure_password@localhost:5432/insurance_prod
DATABASE_ECHO=false

# Storage
UPLOAD_DIR=/mnt/storage/insurance/uploads
MAX_UPLOAD_SIZE_MB=1000

# Worker
WORKER_MAX_THREADS=8
USE_PIPELINE=true

# API
CORS_ORIGINS=https://insurance.company.com,https://api.company.com
DEV_MODE=false

# Logging
LOG_LEVEL=INFO
```

### Step 2: Database Setup

#### PostgreSQL (Recommended for Production)

```bash
# 1. Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# 2. Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE insurance_prod;
CREATE USER insurance_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE insurance_prod TO insurance_user;
EOF

# 3. Test connection
psql postgresql://insurance_user:secure_password@localhost/insurance_prod -c "SELECT 1"

# 4. Initialize tables
python -c "
from insurance_mvp.api.database import DatabaseManager
db = DatabaseManager('postgresql://insurance_user:secure_password@localhost/insurance_prod')
db.create_tables()
print('Tables created successfully')
"
```

#### SQLite (Development/Small Deployments)

```bash
# Tables are auto-created on startup
# Database file: ./insurance.db
```

### Step 3: Generate Production API Keys

```bash
python << EOF
from insurance_mvp.api.auth import api_key_store

# Frontend application key
frontend_key = api_key_store.generate_key(
    name="production-frontend",
    permissions={"read": True, "write": True, "admin": False}
)
print(f"Frontend API Key: {frontend_key}")

# Admin key
admin_key = api_key_store.generate_key(
    name="production-admin",
    permissions={"read": True, "write": True, "admin": True}
)
print(f"Admin API Key: {admin_key}")

# Monitoring key (read-only)
monitor_key = api_key_store.generate_key(
    name="production-monitoring",
    permissions={"read": True, "write": False, "admin": False}
)
print(f"Monitoring API Key: {monitor_key}")
EOF
```

**IMPORTANT**: Store keys securely! Use environment variables or secret management service.

### Step 4: Deploy with Gunicorn

```bash
# Install gunicorn
pip install gunicorn uvicorn[standard]

# Create systemd service
sudo tee /etc/systemd/system/insurance-api.service << EOF
[Unit]
Description=Insurance MVP API
After=network.target postgresql.service

[Service]
Type=notify
User=insurance
Group=insurance
WorkingDirectory=/opt/insurance-mvp
Environment="DATABASE_URL=postgresql://insurance_user:password@localhost/insurance_prod"
Environment="DEV_MODE=false"
ExecStart=/opt/insurance-mvp/venv/bin/gunicorn insurance_mvp.api.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile /var/log/insurance-api/access.log \
    --error-logfile /var/log/insurance-api/error.log
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create log directory
sudo mkdir -p /var/log/insurance-api
sudo chown insurance:insurance /var/log/insurance-api

# Enable and start service
sudo systemctl enable insurance-api
sudo systemctl start insurance-api
sudo systemctl status insurance-api
```

### Step 5: Nginx Reverse Proxy

```bash
# Install nginx
sudo apt-get install nginx

# Configure reverse proxy
sudo tee /etc/nginx/sites-available/insurance-api << 'EOF'
upstream insurance_backend {
    server 127.0.0.1:8000 fail_timeout=0;
}

server {
    listen 80;
    server_name api.insurance.company.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.insurance.company.com;

    # SSL certificates
    ssl_certificate /etc/ssl/certs/insurance-api.crt;
    ssl_certificate_key /etc/ssl/private/insurance-api.key;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # File upload size
    client_max_body_size 1000M;

    # Proxy settings
    location / {
        proxy_pass http://insurance_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long uploads
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
        send_timeout 300;
    }

    # Static files (if needed)
    location /static {
        alias /opt/insurance-mvp/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Health check (bypass auth)
    location /health {
        proxy_pass http://insurance_backend;
        access_log off;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/insurance-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Step 6: SSL Certificate (Let's Encrypt)

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d api.insurance.company.com

# Auto-renewal is configured automatically
sudo certbot renew --dry-run
```

## Monitoring

### Systemd Journal Logs

```bash
# View logs
sudo journalctl -u insurance-api -f

# Last 100 lines
sudo journalctl -u insurance-api -n 100

# Today's logs
sudo journalctl -u insurance-api --since today
```

### Application Logs

```bash
# Access logs
tail -f /var/log/insurance-api/access.log

# Error logs
tail -f /var/log/insurance-api/error.log

# Filter by claim ID
grep "claim_abc123" /var/log/insurance-api/error.log
```

### Health Monitoring

```bash
# Simple health check
curl https://api.insurance.company.com/health

# Detailed metrics
curl -H "X-API-Key: $MONITOR_API_KEY" https://api.insurance.company.com/metrics
```

### Prometheus Integration (Optional)

Create `/etc/prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'insurance-api'
    scrape_interval: 30s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    authorization:
      credentials: 'ins_monitoring_api_key'
```

## Backup and Recovery

### Database Backup

```bash
# PostgreSQL backup
pg_dump insurance_prod > backup_$(date +%Y%m%d).sql

# Automated daily backups
cat > /etc/cron.daily/insurance-backup << 'EOF'
#!/bin/bash
pg_dump insurance_prod | gzip > /backup/insurance_$(date +%Y%m%d).sql.gz
find /backup -name "insurance_*.sql.gz" -mtime +30 -delete
EOF
chmod +x /etc/cron.daily/insurance-backup
```

### Database Restore

```bash
# PostgreSQL restore
psql insurance_prod < backup_20260217.sql
```

### Video Storage Backup

```bash
# Rsync to backup server
rsync -avz /mnt/storage/insurance/uploads/ backup-server:/backup/videos/

# S3 backup (AWS CLI)
aws s3 sync /mnt/storage/insurance/uploads/ s3://insurance-backup/videos/
```

## Scaling

### Horizontal Scaling

```bash
# Run multiple workers behind load balancer
# Server 1
gunicorn insurance_mvp.api.main:app --bind 0.0.0.0:8001

# Server 2
gunicorn insurance_mvp.api.main:app --bind 0.0.0.0:8002

# Load balancer (nginx)
upstream insurance_cluster {
    server server1:8001;
    server server2:8002;
    server server3:8003;
}
```

### Database Connection Pooling

```python
# In database.py, add pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    database_url,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_timeout=30,
    pool_recycle=3600,
)
```

### Background Worker Scaling

Replace ThreadPoolExecutor with Celery + Redis:

```bash
# Install Celery
pip install celery redis

# Run Celery worker
celery -A insurance_mvp.api.background worker --loglevel=info --concurrency=8

# Run multiple workers
celery multi start worker1 worker2 worker3 -A insurance_mvp.api.background
```

## Security Hardening

### 1. API Key Rotation

```python
# Revoke old key
from insurance_mvp.api.auth import api_key_store
api_key_store.revoke_key("old_api_key")

# Generate new key
new_key = api_key_store.generate_key("production-frontend", {...})
```

### 2. Rate Limiting

Adjust in `auth.py`:

```python
rate_limiter = RateLimiter(requests_per_minute=120)  # Increase limit
```

Or use Redis-based rate limiting:

```bash
pip install slowapi redis
```

### 3. Input Validation

Already implemented via Pydantic models. Add custom validators as needed.

### 4. Database Security

```sql
-- Restrict permissions
REVOKE ALL ON DATABASE insurance_prod FROM PUBLIC;
GRANT CONNECT ON DATABASE insurance_prod TO insurance_user;

-- Row-level security (if needed)
CREATE POLICY claim_access_policy ON claims
    USING (claimant_id = current_user);
```

### 5. Audit Logging

All state changes are logged to `audit_logs` table. Enable database audit trail:

```sql
-- PostgreSQL audit extension
CREATE EXTENSION IF NOT EXISTS "pgaudit";
```

## Troubleshooting

### Issue: "Database not initialized"

```bash
# Manually create tables
python -c "
from insurance_mvp.api.database import DatabaseManager
from insurance_mvp.api.main import config
db = DatabaseManager(config.DATABASE_URL)
db.create_tables()
"
```

### Issue: "Background worker not active"

Check logs:
```bash
sudo journalctl -u insurance-api | grep "worker"
```

Restart service:
```bash
sudo systemctl restart insurance-api
```

### Issue: "Too many database connections"

Increase connection pool:
```python
# In database.py
engine = create_engine(url, pool_size=50, max_overflow=100)
```

### Issue: "Video upload fails"

Check disk space:
```bash
df -h /mnt/storage/insurance/uploads
```

Check file permissions:
```bash
ls -la /mnt/storage/insurance/uploads
```

Fix permissions:
```bash
sudo chown -R insurance:insurance /mnt/storage/insurance/uploads
sudo chmod -R 755 /mnt/storage/insurance/uploads
```

### Issue: "Rate limit errors"

Temporary override:
```python
# In auth.py
rate_limiter = RateLimiter(requests_per_minute=1000)  # Increase
```

Or disable for specific endpoint:
```python
@app.post("/claims/upload")  # Remove dependencies=[Depends(check_rate_limit)]
async def upload_claim(...):
    ...
```

## Performance Tuning

### Database Indexes

```sql
-- Already created in schema, verify with:
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'claims';

-- Add custom index if needed
CREATE INDEX idx_claims_custom ON claims(status, upload_time) WHERE status = 'assessed';
```

### Query Optimization

```python
# Use eager loading to avoid N+1 queries
from sqlalchemy.orm import joinedload

claims = session.query(Claim).options(
    joinedload(Claim.assessment),
    joinedload(Claim.reviews)
).all()
```

### Caching

```bash
# Install Redis
pip install redis

# Add caching layer
from redis import Redis
cache = Redis(host='localhost', port=6379, db=0)

# Cache assessment results
cache.setex(f"assessment:{claim_id}", 3600, json.dumps(assessment))
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest insurance_mvp/tests/test_api_basic.py -v

# With coverage
pytest insurance_mvp/tests/test_api_basic.py --cov=insurance_mvp.api --cov-report=html
```

### Integration Tests

```bash
# Start test server
python -m insurance_mvp.api.main &
SERVER_PID=$!

# Run integration tests
pytest insurance_mvp/tests/test_api_integration.py -v

# Cleanup
kill $SERVER_PID
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load_test.py --host=http://localhost:8000
```

## Maintenance

### Database Vacuum (PostgreSQL)

```bash
# Manual vacuum
psql insurance_prod -c "VACUUM ANALYZE;"

# Auto-vacuum (already enabled by default)
```

### Log Rotation

```bash
# Configure logrotate
sudo tee /etc/logrotate.d/insurance-api << EOF
/var/log/insurance-api/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 insurance insurance
    sharedscripts
    postrotate
        systemctl reload insurance-api > /dev/null 2>&1 || true
    endscript
}
EOF
```

### Cleanup Old Data

```sql
-- Archive old claims (>1 year)
INSERT INTO claims_archive SELECT * FROM claims WHERE upload_time < NOW() - INTERVAL '1 year';
DELETE FROM claims WHERE upload_time < NOW() - INTERVAL '1 year';
```

## Support

For issues and questions:
- Check logs: `/var/log/insurance-api/`
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

## Next Steps

1. **Integration**: Connect to actual ML pipeline (see `background.py` PipelineProcessor)
2. **Frontend**: Build web dashboard using API (React, Vue, or use included templates)
3. **Authentication**: Upgrade to OAuth2/JWT for multi-user support
4. **Monitoring**: Set up Prometheus + Grafana dashboards
5. **CI/CD**: Automated deployment pipeline (GitHub Actions, GitLab CI)
