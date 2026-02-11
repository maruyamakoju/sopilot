# SOPilot Deployment Guide

Complete guide for deploying SOPilot in production environments.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Bare Metal Deployment](#bare-metal-deployment)
5. [GPU Configuration](#gpu-configuration)
6. [High Availability Setup](#high-availability-setup)
7. [Monitoring & Observability](#monitoring--observability)
8. [Security Hardening](#security-hardening)
9. [Backup & Recovery](#backup--recovery)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites
- Python 3.10+
- Redis 5.0+ (for production queue backend)
- (Optional) NVIDIA GPU with CUDA 12.x for V-JEPA2 + GPU DTW

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/sopilot.git
cd sopilot

# Install dependencies
pip install -e ".[ml,gpu]"

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Initialize database
export SOPILOT_DATA_DIR=./data
python -c "from sopilot.db import Database; from sopilot.config import get_settings; s = get_settings(); Database(s.db_path)"

# Start services
sopilot-api &          # API server (port 8000)
sopilot-worker &       # Background worker
sopilot-watch &        # Watch folder daemon (optional)
```

---

## Docker Deployment

### Single Container (Development)

```bash
# Build image
docker build -t sopilot:latest -f docker/Dockerfile.cpu .

# Run container
docker run -d \
  --name sopilot-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e SOPILOT_QUEUE_BACKEND=inline \
  -e SOPILOT_EMBEDDER_BACKEND=heuristic \
  sopilot:latest
```

### Docker Compose (Production)

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  api:
    image: sopilot:latest
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - SOPILOT_QUEUE_BACKEND=rq
      - SOPILOT_REDIS_URL=redis://redis:6379/0
      - SOPILOT_EMBEDDER_BACKEND=heuristic
      - SOPILOT_API_TOKEN=${API_TOKEN}
    depends_on:
      - redis
    command: sopilot-api

  worker:
    image: sopilot:latest
    volumes:
      - ./data:/app/data
    environment:
      - SOPILOT_QUEUE_BACKEND=rq
      - SOPILOT_REDIS_URL=redis://redis:6379/0
      - SOPILOT_EMBEDDER_BACKEND=heuristic
    depends_on:
      - redis
    command: sopilot-worker
    deploy:
      replicas: 2  # Scale workers as needed

volumes:
  redis-data:
```

**Start services:**
```bash
docker-compose up -d
```

### GPU-Enabled Docker

```dockerfile
# docker/Dockerfile.gpu
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -e ".[ml,gpu]"

EXPOSE 8000
CMD ["sopilot-api"]
```

**Build & run:**
```bash
docker build -t sopilot-gpu:latest -f docker/Dockerfile.gpu .

docker run -d \
  --gpus all \
  --name sopilot-gpu \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e SOPILOT_EMBEDDER_BACKEND=vjepa2 \
  -e SOPILOT_DTW_USE_GPU=true \
  sopilot-gpu:latest
```

---

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster 1.24+
- kubectl configured
- Persistent storage (NFS/EBS/GCE PD)
- (Optional) GPU nodes with NVIDIA device plugin

### Namespace Setup

```bash
kubectl create namespace sopilot
```

### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sopilot-config
  namespace: sopilot
data:
  SOPILOT_QUEUE_BACKEND: "rq"
  SOPILOT_REDIS_URL: "redis://redis:6379/0"
  SOPILOT_EMBEDDER_BACKEND: "heuristic"
  SOPILOT_TARGET_FPS: "4"
  SOPILOT_CLIP_SECONDS: "4.0"
  SOPILOT_MIN_SCORING_CLIPS: "4"
```

### Secrets

```bash
kubectl create secret generic sopilot-secrets \
  --from-literal=api-token=YOUR_SECRET_TOKEN \
  --from-literal=audit-signing-key=YOUR_SIGNING_KEY \
  -n sopilot
```

### PersistentVolumeClaim

```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: sopilot-data
  namespace: sopilot
spec:
  accessModes:
    - ReadWriteMany  # Required for multi-pod access
  resources:
    requests:
      storage: 100Gi
  storageClassName: nfs-storage  # Adjust to your storage class
```

### Redis Deployment

```yaml
# k8s/redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: sopilot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: sopilot
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

### API Deployment

```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sopilot-api
  namespace: sopilot
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sopilot-api
  template:
    metadata:
      labels:
        app: sopilot-api
    spec:
      containers:
      - name: sopilot-api
        image: sopilot:latest
        command: ["sopilot-api"]
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: sopilot-config
        env:
        - name: SOPILOT_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: sopilot-secrets
              key: api-token
        - name: SOPILOT_DATA_DIR
          value: "/data"
        volumeMounts:
        - name: data
          mountPath: /data
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: sopilot-data
---
apiVersion: v1
kind: Service
metadata:
  name: sopilot-api
  namespace: sopilot
spec:
  type: LoadBalancer
  selector:
    app: sopilot-api
  ports:
  - port: 80
    targetPort: 8000
```

### Worker Deployment

```yaml
# k8s/worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sopilot-worker
  namespace: sopilot
spec:
  replicas: 4  # Scale based on workload
  selector:
    matchLabels:
      app: sopilot-worker
  template:
    metadata:
      labels:
        app: sopilot-worker
    spec:
      containers:
      - name: sopilot-worker
        image: sopilot:latest
        command: ["sopilot-worker"]
        envFrom:
        - configMapRef:
            name: sopilot-config
        env:
        - name: SOPILOT_DATA_DIR
          value: "/data"
        volumeMounts:
        - name: data
          mountPath: /data
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: sopilot-data
```

### GPU Worker Deployment

```yaml
# k8s/gpu-worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sopilot-gpu-worker
  namespace: sopilot
spec:
  replicas: 1  # One per GPU
  selector:
    matchLabels:
      app: sopilot-gpu-worker
  template:
    metadata:
      labels:
        app: sopilot-gpu-worker
    spec:
      nodeSelector:
        nvidia.com/gpu: "true"
      containers:
      - name: sopilot-gpu-worker
        image: sopilot-gpu:latest
        command: ["sopilot-worker"]
        envFrom:
        - configMapRef:
            name: sopilot-config
        env:
        - name: SOPILOT_EMBEDDER_BACKEND
          value: "vjepa2"
        - name: SOPILOT_DTW_USE_GPU
          value: "true"
        - name: SOPILOT_DATA_DIR
          value: "/data"
        volumeMounts:
        - name: data
          mountPath: /data
        resources:
          limits:
            nvidia.com/gpu: 1  # Request 1 GPU
            memory: "16Gi"
            cpu: "4000m"
```

### Deploy All Resources

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/worker-deployment.yaml
kubectl apply -f k8s/gpu-worker-deployment.yaml  # If GPU available
```

### Verify Deployment

```bash
kubectl get pods -n sopilot
kubectl logs -f deployment/sopilot-api -n sopilot
kubectl get svc sopilot-api -n sopilot  # Get LoadBalancer IP
```

---

## Bare Metal Deployment

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Disk: 50GB SSD
- OS: Ubuntu 22.04 LTS

**Recommended (with GPU):**
- CPU: 16+ cores
- RAM: 128GB
- Disk: 1TB NVMe SSD
- GPU: NVIDIA RTX 3090 / 4090 / 5090 (24GB+ VRAM)
- OS: Ubuntu 22.04 LTS with CUDA 12.1+

### Step-by-Step Installation

#### 1. System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
  python3.10 python3-pip python3-venv \
  redis-server \
  libgl1-mesa-glx libglib2.0-0 \
  git curl

# Install NVIDIA drivers (if GPU)
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit
nvidia-smi  # Verify GPU detected
```

#### 2. Application Installation

```bash
# Create user
sudo useradd -m -s /bin/bash sopilot
sudo su - sopilot

# Create directories
mkdir -p /home/sopilot/{app,data,logs}
cd /home/sopilot/app

# Clone & install
git clone https://github.com/your-org/sopilot.git .
python3 -m venv venv
source venv/bin/activate
pip install -e ".[ml,gpu]"
```

#### 3. Configuration

```bash
# Create .env file
cat > /home/sopilot/app/.env <<EOF
SOPILOT_DATA_DIR=/home/sopilot/data
SOPILOT_QUEUE_BACKEND=rq
SOPILOT_REDIS_URL=redis://127.0.0.1:6379/0
SOPILOT_EMBEDDER_BACKEND=vjepa2
SOPILOT_DTW_USE_GPU=true
SOPILOT_API_TOKEN=changeme_production_token
SOPILOT_AUTH_REQUIRED=true
SOPILOT_AUDIT_SIGNING_KEY=changeme_signing_key
EOF

# Secure .env
chmod 600 /home/sopilot/app/.env
```

#### 4. Systemd Services

**API Service:**
```ini
# /etc/systemd/system/sopilot-api.service
[Unit]
Description=SOPilot API Server
After=network.target redis.service

[Service]
Type=simple
User=sopilot
WorkingDirectory=/home/sopilot/app
EnvironmentFile=/home/sopilot/app/.env
ExecStart=/home/sopilot/app/venv/bin/sopilot-api
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Worker Service:**
```ini
# /etc/systemd/system/sopilot-worker@.service
[Unit]
Description=SOPilot Worker %i
After=network.target redis.service

[Service]
Type=simple
User=sopilot
WorkingDirectory=/home/sopilot/app
EnvironmentFile=/home/sopilot/app/.env
ExecStart=/home/sopilot/app/venv/bin/sopilot-worker
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable & Start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable sopilot-api
sudo systemctl enable sopilot-worker@{1..4}  # 4 workers
sudo systemctl start sopilot-api
sudo systemctl start sopilot-worker@{1..4}
```

#### 5. Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/sopilot
upstream sopilot_api {
    least_conn;
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name sopilot.example.com;

    client_max_body_size 1024M;

    location / {
        proxy_pass http://sopilot_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 600s;
    }

    location /metrics {
        proxy_pass http://sopilot_api;
        allow 10.0.0.0/8;  # Restrict to monitoring network
        deny all;
    }
}
```

**Enable & reload:**
```bash
sudo ln -s /etc/nginx/sites-available/sopilot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## GPU Configuration

### CUDA Setup

```bash
# Verify CUDA
nvcc --version

# Install CuPy for GPU DTW
pip install cupy-cuda12x

# Test GPU availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "from sopilot.dtw_gpu import is_gpu_available; print(is_gpu_available())"
```

### Environment Variables

```bash
# Force GPU usage
export SOPILOT_EMBEDDER_BACKEND=vjepa2
export SOPILOT_EMBEDDING_DEVICE=cuda
export SOPILOT_DTW_USE_GPU=true

# Optimize batch sizes for your GPU
export SOPILOT_VJEPA2_BATCH_SIZE=16  # RTX 3090/4090: 8-16, RTX 5090: 16-32
export SOPILOT_INGEST_EMBED_BATCH_SIZE=16
```

### Multi-GPU Setup

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Or distribute workers across GPUs
# Worker 1: GPU 0
CUDA_VISIBLE_DEVICES=0 sopilot-worker &

# Worker 2: GPU 1
CUDA_VISIBLE_DEVICES=1 sopilot-worker &
```

---

## High Availability Setup

### Load Balancing

Use HAProxy or cloud load balancer to distribute traffic across multiple API instances.

**HAProxy config:**
```
frontend sopilot_front
    bind *:80
    default_backend sopilot_back

backend sopilot_back
    balance roundrobin
    option httpchk GET /health
    server api1 10.0.1.10:8000 check
    server api2 10.0.1.11:8000 check
    server api3 10.0.1.12:8000 check
```

### Redis Sentinel (HA Queue)

```bash
# redis-sentinel.conf
sentinel monitor sopilot-redis 127.0.0.1 6379 2
sentinel down-after-milliseconds sopilot-redis 5000
sentinel parallel-syncs sopilot-redis 1
sentinel failover-timeout sopilot-redis 10000
```

### Shared Storage

Use NFS, GlusterFS, or cloud storage (EFS/GCS/Azure Files) for `/data` directory.

---

## Monitoring & Observability

### Prometheus Scraping

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'sopilot'
    static_configs:
      - targets: ['sopilot-api:8000']
    metrics_path: /metrics
    scrape_interval: 15s
```

### Grafana Dashboard

Import `monitoring/grafana-dashboard.json` for pre-built SOPilot dashboard.

**Key metrics:**
- Job throughput (ingest/score/training)
- Queue depths
- Job latencies (p50, p95, p99)
- GPU memory usage
- Error rates

### Structured Logging

```bash
# Enable JSON logs for log aggregators
export SOPILOT_LOG_FORMAT=json
export SOPILOT_LOG_LEVEL=INFO

# Ship logs to ELK/Loki
# Example: Filebeat configuration
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /home/sopilot/logs/*.log
  json.keys_under_root: true
```

---

## Security Hardening

### Authentication

```bash
# Generate strong tokens
openssl rand -hex 32  # API token

# Configure role-based tokens
export SOPILOT_API_ROLE_TOKENS="admin:$(openssl rand -hex 32),operator:$(openssl rand -hex 32)"
```

### Network Security

- Expose only port 80/443 (API) to internet
- Restrict `/metrics` to monitoring network
- Keep Redis on private network only
- Use TLS/SSL (Let's Encrypt)

### File Permissions

```bash
chmod 700 /home/sopilot/data
chmod 600 /home/sopilot/app/.env
chown -R sopilot:sopilot /home/sopilot
```

### Audit Logging

```bash
# Enable signed audit exports
export SOPILOT_AUDIT_SIGNING_KEY=$(openssl rand -hex 32)
export SOPILOT_AUDIT_SIGNING_KEY_ID=prod

# Export audit trail periodically
curl -X POST http://localhost:8000/audit/export \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"limit": 5000}' \
  -o audit_$(date +%Y%m%d).json
```

---

## Backup & Recovery

### Database Backup

```bash
# Backup SQLite database
cp /home/sopilot/data/sopilot.db /backup/sopilot_$(date +%Y%m%d).db

# Automate with cron
0 2 * * * cp /home/sopilot/data/sopilot.db /backup/sopilot_$(date +\%Y\%m\%d).db
```

### Data Backup

```bash
# Backup all data
tar czf /backup/sopilot_data_$(date +%Y%m%d).tar.gz \
  /home/sopilot/data
```

### Recovery

```bash
# Restore database
cp /backup/sopilot_20260208.db /home/sopilot/data/sopilot.db

# Rebuild index
python -c "from sopilot.service import SopilotService; from sopilot.config import get_settings; from sopilot.db import Database; s = get_settings(); db = Database(s.db_path); svc = SopilotService(s, db); svc._rebuild_task_index('task_id')"
```

---

## Troubleshooting

### API won't start
```bash
# Check logs
sudo journalctl -u sopilot-api -n 50

# Common issues:
# 1. Port 8000 already in use
sudo lsof -i :8000

# 2. Database locked
rm /home/sopilot/data/sopilot.db-wal
```

### Worker not processing jobs
```bash
# Check Redis connection
redis-cli ping

# Check worker logs
sudo journalctl -u sopilot-worker@1 -n 50

# Verify queue
python -c "from redis import Redis; r = Redis(); print(r.llen('sopilot_ingest'))"
```

### GPU not detected
```bash
# Verify CUDA
nvidia-smi
nvcc --version

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Check CuPy
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

### Out of memory
```bash
# Reduce batch sizes
export SOPILOT_VJEPA2_BATCH_SIZE=2
export SOPILOT_INGEST_EMBED_BATCH_SIZE=4

# Monitor memory
nvidia-smi -l 1  # GPU
htop  # CPU/RAM
```

---

## Performance Tuning

### Worker Scaling
- **CPU workers:** 1-2x CPU cores
- **GPU workers:** 1 per GPU
- **Queue backend:** Use `rq` for production (not `inline`)

### Database Optimization
```sql
-- Enable WAL mode
PRAGMA journal_mode=WAL;

-- Add indices
CREATE INDEX IF NOT EXISTS idx_videos_task_role ON videos(task_id, role);
```

### Redis Tuning
```conf
# redis.conf
maxmemory 4gb
maxmemory-policy allkeys-lru
save ""  # Disable RDB if using Redis as pure cache
```

---

## Next Steps

1. Configure monitoring: [Monitoring Guide](MONITORING.md)
2. Review security: [Security Checklist](SECURITY.md)
3. API usage: [API Reference](API_REFERENCE.md)
4. Configuration: [Configuration Reference](CONFIGURATION.md)

---

**Support:** For deployment issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
