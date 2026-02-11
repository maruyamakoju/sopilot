# SOPilot Troubleshooting Guide

**Last Updated:** 2026-02-08
**Version:** 0.1.0

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [GPU Problems](#gpu-problems)
3. [Queue & Worker Issues](#queue--worker-issues)
4. [Database Errors](#database-errors)
5. [API Errors](#api-errors)
6. [Video Processing Failures](#video-processing-failures)
7. [Performance Issues](#performance-issues)
8. [Authentication Problems](#authentication-problems)
9. [Deployment Issues](#deployment-issues)
10. [Common Error Messages](#common-error-messages)

---

## Installation Issues

### Problem: `pip install` fails with dependency conflicts

**Symptoms:**
```
ERROR: Cannot install sopilot because these package versions have conflicts:
  - torch==2.5.0 requires numpy<2.0,>=1.23.5
  - numpy==2.1.0 is incompatible
```

**Solutions:**
1. **Use a clean virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -e .
   ```

2. **Force compatible numpy version:**
   ```bash
   pip install "numpy<2.0" torch torchvision
   pip install -e .
   ```

3. **Use conda (recommended for GPU):**
   ```bash
   conda create -n sopilot python=3.10
   conda activate sopilot
   conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install -e .
   ```

---

### Problem: V-JEPA2 model download fails

**Symptoms:**
```
Failed to download model from https://dl.fbaipublicfiles.com/vjepa2/...
URLError: <urlopen error [Errno 110] Connection timed out>
```

**Solutions:**
1. **Manual download:**
   ```bash
   cd ~/.cache/torch/hub/checkpoints/
   wget https://dl.fbaipublicfiles.com/vjepa2/vit_large_pt.pth
   ```

2. **Set proxy (if behind corporate firewall):**
   ```bash
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

3. **Use local model path:**
   ```bash
   export SOPILOT_EMBEDDER_MODEL="vjepa2:vit_large:pt"
   export SOPILOT_EMBEDDER_CHECKPOINT_PATH="/path/to/vit_large_pt.pth"
   ```

---

### Problem: `ModuleNotFoundError: No module named 'cupy'`

**Symptoms:**
```
ModuleNotFoundError: No module named 'cupy'
```

**Solutions:**
1. **Install CuPy for your CUDA version:**
   ```bash
   # For CUDA 12.x (RTX 5090, RTX 40-series)
   pip install cupy-cuda12x

   # For CUDA 11.x (older GPUs)
   pip install cupy-cuda11x
   ```

2. **Check CUDA installation:**
   ```bash
   nvidia-smi  # Should show CUDA version
   nvcc --version  # Should match nvidia-smi
   ```

3. **CPU-only mode (fallback):**
   ```bash
   export SOPILOT_DTW_USE_GPU=false
   # GPU DTW will be skipped, falls back to CPU
   ```

---

## GPU Problems

### Problem: CUDA out of memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 32.00 GiB total capacity; 28.50 GiB already allocated)
```

**Solutions:**
1. **Reduce batch size:**
   ```bash
   export SOPILOT_EMBEDDER_BATCH_SIZE=8  # Default: auto-detect (16-24)
   ```

2. **Clear GPU cache before processing:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Restart worker:**
   ```bash
   # Kill worker process
   pkill -f sopilot-worker
   # Restart
   sopilot-worker
   ```

4. **Check for memory leaks:**
   ```bash
   watch -n 1 nvidia-smi  # Monitor GPU memory usage
   ```

---

### Problem: GPU not detected

**Symptoms:**
```
INFO: GPU not available, falling back to CPU
```

**Solutions:**
1. **Verify CUDA installation:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Check PyTorch CUDA version:**
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

3. **Reinstall PyTorch with CUDA:**
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Docker: Add GPU runtime:**
   ```bash
   docker run --gpus all ...  # Add --gpus all flag
   ```

---

### Problem: torch.compile fails with "unsupported backend"

**Symptoms:**
```
RuntimeError: Unsupported backend: reduce-overhead
```

**Solutions:**
1. **Disable torch.compile:**
   ```bash
   export SOPILOT_EMBEDDER_COMPILE=false
   ```

2. **Upgrade PyTorch:**
   ```bash
   pip install --upgrade torch  # Requires PyTorch 2.0+
   ```

3. **Use CPU mode for old GPUs:**
   ```bash
   export CUDA_VISIBLE_DEVICES=""  # Force CPU
   ```

---

## Queue & Worker Issues

### Problem: Jobs stuck in "queued" status

**Symptoms:**
```
GET /videos/jobs/{job_id}
{"status": "queued", "queued_at": "2026-02-08T10:00:00Z"}
# Status never changes to "running"
```

**Solutions:**
1. **Check if workers are running:**
   ```bash
   ps aux | grep sopilot-worker
   ```

2. **Check Redis connection:**
   ```bash
   redis-cli ping  # Should return PONG
   echo $SOPILOT_REDIS_URL  # Should be redis://localhost:6379/0
   ```

3. **Restart workers:**
   ```bash
   # If using systemd
   sudo systemctl restart sopilot-worker

   # If using docker-compose
   docker-compose restart worker
   ```

4. **Check worker logs:**
   ```bash
   journalctl -u sopilot-worker -f  # systemd
   docker logs sopilot_worker_1 -f  # docker
   ```

---

### Problem: Jobs fail immediately with "ModuleNotFoundError"

**Symptoms:**
```
ERROR: run_ingest_job failed: ModuleNotFoundError: No module named 'sopilot'
```

**Solutions:**
1. **Ensure package is installed in worker environment:**
   ```bash
   # Activate same venv as worker
   source /path/to/venv/bin/activate
   pip install -e .
   ```

2. **Check worker PYTHONPATH:**
   ```bash
   # In worker systemd service
   Environment="PYTHONPATH=/app"
   ```

3. **Use absolute imports:**
   ```python
   from sopilot.service import ingest_video  # ✅ Good
   from service import ingest_video  # ❌ Bad
   ```

---

### Problem: Redis connection refused

**Symptoms:**
```
redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379. Connection refused.
```

**Solutions:**
1. **Start Redis:**
   ```bash
   # Ubuntu/Debian
   sudo systemctl start redis

   # macOS (Homebrew)
   brew services start redis

   # Docker
   docker run -d -p 6379:6379 redis:7-alpine
   ```

2. **Check Redis is listening:**
   ```bash
   netstat -tlnp | grep 6379
   ```

3. **Update connection URL:**
   ```bash
   export SOPILOT_REDIS_URL=redis://hostname:6379/0
   ```

---

## Database Errors

### Problem: "database is locked"

**Symptoms:**
```
sqlite3.OperationalError: database is locked
```

**Solutions:**
1. **Enable WAL mode (Write-Ahead Logging):**
   ```python
   # In db.py or startup script
   conn = sqlite3.connect("sopilot.db")
   conn.execute("PRAGMA journal_mode=WAL")
   ```

2. **Increase timeout:**
   ```python
   conn = sqlite3.connect("sopilot.db", timeout=30.0)
   ```

3. **Close connections properly:**
   ```python
   try:
       conn = get_connection()
       # ... use connection
   finally:
       conn.close()
   ```

4. **Migrate to PostgreSQL (production):**
   See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md#postgresql-setup)

---

### Problem: Foreign key constraint violations

**Symptoms:**
```
sqlite3.IntegrityError: FOREIGN KEY constraint failed
```

**Solutions:**
1. **Enable foreign keys:**
   ```python
   conn.execute("PRAGMA foreign_keys=ON")
   ```

2. **Check insertion order:**
   ```python
   # ✅ Correct order
   insert_video(...)  # Insert parent first
   insert_clips(...)  # Then children

   # ❌ Wrong order
   insert_clips(...)  # Fails if video doesn't exist
   ```

3. **Delete cascade:**
   ```sql
   CREATE TABLE clips (
       video_id INTEGER REFERENCES videos(video_id) ON DELETE CASCADE
   );
   ```

---

### Problem: Database file corrupted

**Symptoms:**
```
sqlite3.DatabaseError: database disk image is malformed
```

**Solutions:**
1. **Check disk integrity:**
   ```bash
   sqlite3 sopilot.db "PRAGMA integrity_check;"
   ```

2. **Restore from backup:**
   ```bash
   cp data/backups/sopilot_20260208.db data/sopilot.db
   ```

3. **Dump and restore:**
   ```bash
   sqlite3 sopilot.db .dump > dump.sql
   sqlite3 sopilot_new.db < dump.sql
   mv sopilot_new.db sopilot.db
   ```

---

## API Errors

### Problem: 401 Unauthorized

**Symptoms:**
```
{"detail": "Invalid or missing authentication token"}
```

**Solutions:**
1. **Check Authorization header:**
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/videos
   ```

2. **Verify token configuration:**
   ```bash
   echo $SOPILOT_API_ROLE_TOKENS
   # Should be: admin:secret123,operator:secret456,viewer:secret789
   ```

3. **Use correct token:**
   ```bash
   # For admin role
   curl -H "Authorization: Bearer secret123" ...
   ```

4. **Disable auth for testing:**
   ```bash
   export SOPILOT_API_NO_AUTH=true  # ⚠️ Development only!
   ```

---

### Problem: 413 Payload Too Large

**Symptoms:**
```
{"detail": "Upload exceeds maximum size of 500 MB"}
```

**Solutions:**
1. **Increase upload limit:**
   ```bash
   export SOPILOT_UPLOAD_MAX_MB=2000  # Default: 500
   ```

2. **Check nginx limits (if using reverse proxy):**
   ```nginx
   client_max_body_size 2000M;
   ```

3. **Compress video before upload:**
   ```bash
   ffmpeg -i input.mp4 -vcodec h264 -crf 28 output.mp4
   ```

---

### Problem: 500 Internal Server Error

**Symptoms:**
```
{"detail": "Internal server error"}
```

**Solutions:**
1. **Check API logs:**
   ```bash
   journalctl -u sopilot-api -f  # systemd
   docker logs sopilot_api_1 -f  # docker
   ```

2. **Enable debug logging:**
   ```bash
   export SOPILOT_LOG_LEVEL=DEBUG
   ```

3. **Test endpoint manually:**
   ```bash
   curl -v http://localhost:8000/health
   ```

4. **Check database connection:**
   ```bash
   sqlite3 data/sopilot.db "SELECT COUNT(*) FROM videos;"
   ```

---

## Video Processing Failures

### Problem: "Unsupported video codec"

**Symptoms:**
```
ERROR: cv2.VideoCapture failed to open video: unsupported codec
```

**Solutions:**
1. **Check codec:**
   ```bash
   ffprobe input.mp4 | grep Video:
   # Should show: Video: h264 (High) ...
   ```

2. **Re-encode to H.264:**
   ```bash
   ffmpeg -i input.mov -vcodec libx264 -acodec aac output.mp4
   ```

3. **Install OpenCV with FFMPEG:**
   ```bash
   pip uninstall opencv-python opencv-python-headless
   pip install opencv-python-headless  # Headless includes FFMPEG
   ```

---

### Problem: "Video too short"

**Symptoms:**
```
ERROR: Video duration (2.3s) is below minimum (5.0s)
```

**Solutions:**
1. **Adjust minimum duration:**
   ```bash
   export SOPILOT_VIDEO_MIN_DURATION=1.0  # Default: 5.0
   ```

2. **Pad video to minimum length:**
   ```bash
   ffmpeg -i short.mp4 -vf "tpad=stop_mode=clone:stop_duration=5" padded.mp4
   ```

---

### Problem: Frame extraction fails

**Symptoms:**
```
ERROR: Failed to extract frames from video
```

**Solutions:**
1. **Check video file integrity:**
   ```bash
   ffmpeg -v error -i input.mp4 -f null -
   # Should show no errors
   ```

2. **Reduce target FPS:**
   ```bash
   export SOPILOT_TARGET_FPS=2  # Default: 4
   ```

3. **Test with known-good video:**
   ```bash
   ffmpeg -f lavfi -i testsrc=duration=10:size=640x480:rate=30 test.mp4
   curl -F "file=@test.mp4" -F "task_id=test" http://localhost:8000/videos
   ```

---

## Performance Issues

### Problem: Slow embedding generation

**Symptoms:**
- Ingest jobs take >60s for 10s videos
- GPU utilization <30%

**Solutions:**
1. **Enable GPU acceleration:**
   ```bash
   export SOPILOT_DTW_USE_GPU=true
   nvidia-smi  # Verify GPU is detected
   ```

2. **Increase batch size:**
   ```bash
   export SOPILOT_EMBEDDER_BATCH_SIZE=24  # RTX 5090: 16-24
   ```

3. **Enable torch.compile:**
   ```bash
   export SOPILOT_EMBEDDER_COMPILE=true
   ```

4. **Use heuristic embedder for testing:**
   ```bash
   export SOPILOT_EMBEDDER_MODEL=heuristic-v1  # No GPU needed
   ```

---

### Problem: DTW alignment is slow

**Symptoms:**
- Score jobs take >30s
- CPU at 100%, GPU idle

**Solutions:**
1. **Enable GPU DTW:**
   ```bash
   export SOPILOT_DTW_USE_GPU=true
   pip install cupy-cuda12x  # Install CuPy if missing
   ```

2. **Verify GPU DTW is active:**
   ```bash
   python -c "from sopilot.dtw_gpu import is_gpu_available; print(is_gpu_available())"
   ```

3. **Check alignment matrix size:**
   ```bash
   # If gold video has 2000+ clips, consider downsampling
   export SOPILOT_TARGET_FPS=2  # Reduce clip count
   ```

---

### Problem: High memory usage

**Symptoms:**
- Worker OOM killed
- System swap usage increasing

**Solutions:**
1. **Reduce concurrent workers:**
   ```bash
   # docker-compose.yml
   deploy:
     replicas: 2  # Reduce from 4
   ```

2. **Limit batch size:**
   ```bash
   export SOPILOT_EMBEDDER_BATCH_SIZE=8
   ```

3. **Enable embedding caching:**
   ```python
   # Add to config
   export SOPILOT_CACHE_EMBEDDINGS=true
   ```

4. **Monitor memory:**
   ```bash
   watch -n 1 'free -h && docker stats --no-stream'
   ```

---

## Authentication Problems

### Problem: Token rejected after rotation

**Symptoms:**
```
{"detail": "Invalid or missing authentication token"}
# (worked before, suddenly fails)
```

**Solutions:**
1. **Check token configuration:**
   ```bash
   echo $SOPILOT_API_ROLE_TOKENS
   # admin:NEW_TOKEN,operator:op_token,viewer:view_token
   ```

2. **Update client token:**
   ```bash
   export API_TOKEN="NEW_TOKEN"
   curl -H "Authorization: Bearer $API_TOKEN" ...
   ```

3. **Verify token format:**
   ```bash
   # Tokens must not contain commas or colons
   # Bad: admin:abc,def:123
   # Good: admin:abc_def_123
   ```

---

### Problem: Role hierarchy not working

**Symptoms:**
- Admin cannot access operator endpoints
- Viewer has write access

**Solutions:**
1. **Check role assignment:**
   ```bash
   # List configured roles
   python -c "from sopilot.config import get_settings; print(get_settings().api_role_tokens)"
   ```

2. **Verify endpoint decorators:**
   ```python
   @router.delete("/videos/{video_id}", dependencies=[Depends(require_admin)])
   # Must use correct role decorator
   ```

3. **Test with highest role:**
   ```bash
   # Use admin token for debugging
   curl -H "Authorization: Bearer admin_secret" ...
   ```

---

## Deployment Issues

### Problem: Docker container exits immediately

**Symptoms:**
```
$ docker ps -a
CONTAINER ID   STATUS
abc123         Exited (1) 5 seconds ago
```

**Solutions:**
1. **Check logs:**
   ```bash
   docker logs abc123
   ```

2. **Common startup errors:**
   ```bash
   # Missing data directory
   docker run -v /data:/data ...

   # Port already in use
   netstat -tlnp | grep 8000
   docker run -p 8001:8000 ...

   # Missing environment variables
   docker run -e SOPILOT_REDIS_URL=redis://redis:6379/0 ...
   ```

3. **Run interactively for debugging:**
   ```bash
   docker run -it --entrypoint /bin/bash sopilot:latest
   ```

---

### Problem: Kubernetes pod in CrashLoopBackOff

**Symptoms:**
```
$ kubectl get pods
NAME                    READY   STATUS             RESTARTS
sopilot-worker-abc123   0/1     CrashLoopBackOff   5
```

**Solutions:**
1. **Check pod logs:**
   ```bash
   kubectl logs sopilot-worker-abc123
   kubectl describe pod sopilot-worker-abc123
   ```

2. **Common issues:**
   ```yaml
   # Missing ConfigMap
   kubectl get configmap sopilot-config

   # Missing PVC
   kubectl get pvc sopilot-data

   # Insufficient resources
   resources:
     limits:
       memory: "16Gi"  # Increase if OOM killed
   ```

3. **Debug with ephemeral container:**
   ```bash
   kubectl debug -it sopilot-worker-abc123 --image=busybox
   ```

---

### Problem: Nginx 502 Bad Gateway

**Symptoms:**
```
{"error": "502 Bad Gateway"}
```

**Solutions:**
1. **Check API is running:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Verify nginx upstream:**
   ```nginx
   upstream sopilot_api {
       server localhost:8000;  # Should match API port
   }
   ```

3. **Check nginx error logs:**
   ```bash
   tail -f /var/log/nginx/error.log
   ```

4. **Increase timeout:**
   ```nginx
   proxy_read_timeout 300s;  # For long-running requests
   ```

---

## Common Error Messages

### "ValueError: Settings validation failed"

**Cause:** Invalid environment variable configuration

**Solution:**
```bash
# Check which validation failed
export SOPILOT_TARGET_FPS=4  # Must be >= 1
export SOPILOT_STEP_BOUNDARY_THRESHOLD_FACTOR=2.0  # Must be >= 0
# See docs/CONFIGURATION.md for all constraints
```

---

### "FileNotFoundError: /data/raw/video_123.mp4"

**Cause:** Video file missing (deleted or moved)

**Solution:**
```bash
# Check file exists
ls -lh /data/raw/video_123.mp4

# Restore from backup
cp /backup/video_123.mp4 /data/raw/

# Or delete video record
DELETE FROM videos WHERE video_id=123;
```

---

### "JSONDecodeError: Expecting value: line 1 column 1 (char 0)"

**Cause:** Empty or malformed JSON response

**Solution:**
```bash
# Check API endpoint returns valid JSON
curl -s http://localhost:8000/videos | jq .

# If API returns HTML error page, check logs
journalctl -u sopilot-api -n 50
```

---

### "RuntimeError: CUDA error: device-side assert triggered"

**Cause:** GPU kernel assertion failure (usually array out of bounds)

**Solution:**
```bash
# Enable CUDA error checking
export CUDA_LAUNCH_BLOCKING=1

# Re-run to get detailed error
sopilot-worker

# Check PyTorch version compatibility
python -c "import torch; print(torch.version.cuda)"
```

---

### "MemoryError: Unable to allocate array"

**Cause:** Out of RAM (not GPU memory)

**Solution:**
```bash
# Reduce embedding cache size
export SOPILOT_CACHE_MAX_SIZE=100  # Default: 1000

# Limit concurrent jobs
# In docker-compose.yml
deploy:
  replicas: 1  # Reduce workers

# Add swap space (Linux)
sudo fallocate -l 16G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Diagnostic Commands

### Check System Health

```bash
# API health
curl http://localhost:8000/health

# Database connectivity
sqlite3 data/sopilot.db "SELECT COUNT(*) FROM videos;"

# Redis connectivity
redis-cli ping

# GPU status
nvidia-smi

# Worker processes
ps aux | grep sopilot-worker

# Queue depth
curl http://localhost:8000/ops/queue
```

---

### Collect Debug Information

```bash
#!/bin/bash
# debug_info.sh - Collect system diagnostics

echo "=== SOPilot Debug Report ===" > debug_report.txt
echo "Date: $(date)" >> debug_report.txt

echo -e "\n=== Environment ===" >> debug_report.txt
env | grep SOPILOT >> debug_report.txt

echo -e "\n=== GPU Info ===" >> debug_report.txt
nvidia-smi >> debug_report.txt 2>&1

echo -e "\n=== Processes ===" >> debug_report.txt
ps aux | grep sopilot >> debug_report.txt

echo -e "\n=== API Health ===" >> debug_report.txt
curl -s http://localhost:8000/health >> debug_report.txt

echo -e "\n=== Queue Status ===" >> debug_report.txt
curl -s http://localhost:8000/ops/queue >> debug_report.txt

echo -e "\n=== Recent Logs ===" >> debug_report.txt
journalctl -u sopilot-api -n 100 >> debug_report.txt

echo "Debug report saved to debug_report.txt"
```

---

## Getting Help

### Before Opening an Issue

1. **Check logs:**
   ```bash
   journalctl -u sopilot-api -n 100
   journalctl -u sopilot-worker -n 100
   ```

2. **Run diagnostics:**
   ```bash
   ./debug_info.sh
   ```

3. **Test with minimal config:**
   ```bash
   export SOPILOT_EMBEDDER_MODEL=heuristic-v1  # CPU-only
   export SOPILOT_QUEUE_BACKEND=inline  # No Redis
   ```

4. **Check for known issues:**
   - [GitHub Issues](https://github.com/your-org/sopilot/issues)
   - [Documentation](docs/)

### Include in Bug Report

- SOPilot version: `pip show sopilot`
- Python version: `python --version`
- OS: `uname -a` or `systeminfo` (Windows)
- GPU: `nvidia-smi` output
- Error logs (last 50 lines)
- Steps to reproduce
- Expected vs actual behavior

---

## Performance Benchmarks

### Expected Performance (RTX 5090)

| Operation | Duration | Notes |
|-----------|----------|-------|
| Ingest (10s video) | 2-5s | With V-JEPA2, batch=16 |
| Score (500x500 DTW) | 0.5-1s | GPU DTW enabled |
| Training (100 videos) | 5-10min | Full reindex |

**If significantly slower:**
1. Check GPU is being used: `nvidia-smi`
2. Verify batch size: `echo $SOPILOT_EMBEDDER_BATCH_SIZE`
3. Enable torch.compile: `export SOPILOT_EMBEDDER_COMPILE=true`
4. Check DTW GPU: `python -c "from sopilot.dtw_gpu import is_gpu_available; print(is_gpu_available())"`

---

## References

- [API Reference](API_REFERENCE.md)
- [Configuration Guide](CONFIGURATION.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Architecture Overview](ARCHITECTURE.md)

---

**For urgent production issues, escalate to:**
- On-call engineer: [Contact Info]
- Slack: #sopilot-ops
- Email: sopilot-support@example.com
