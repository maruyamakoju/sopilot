# How to Mount Web UI Routes in main.py

Add these lines to `insurance_mvp/api/main.py` to enable the web UI:

## Step 1: Add Imports

Add at the top of `main.py`, after the existing imports:

```python
# Add these imports after line 68 (after background imports)
from insurance_mvp.api.web_routes import router as web_router
from fastapi.staticfiles import StaticFiles
from pathlib import Path
```

## Step 2: Mount Routes

Add after the FastAPI app initialization (around line 116):

```python
# After: app.add_middleware(CORSMiddleware, ...)

# Mount Web UI routes
app.include_router(web_router)
logger.info("Web UI routes mounted")
```

## Step 3: Mount Static Files

Add in the `startup_event()` function (around line 156):

```python
# After: Path(config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Mount static file directories for web UI
try:
    upload_dir = Path(config.UPLOAD_DIR)
    frames_dir = Path(config.UPLOAD_DIR).parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    app.mount("/static/videos", StaticFiles(directory=str(upload_dir)), name="videos")
    app.mount("/static/frames", StaticFiles(directory=str(frames_dir)), name="frames")
    logger.info("Static file serving enabled")
except Exception as e:
    logger.warning(f"Static file mounting failed: {e}")
```

## Step 4: Override get_db in web_routes

The web_routes.py module has a placeholder `get_db()`. To fix this, add to `web_routes.py`:

```python
# At the top of web_routes.py, change the get_db dependency:

def get_db():
    """Get database session - imported from main app"""
    # This will be overridden when mounting in main.py
    # But we need a working implementation for testing
    try:
        from insurance_mvp.api.main import db_manager
        return next(db_manager.get_session_dependency())
    except ImportError:
        raise RuntimeError("Database not initialized. Start the API server first.")
```

## Complete Example

Here's what the modified sections look like:

### main.py additions:

```python
# At top (after line 68)
from insurance_mvp.api.web_routes import router as web_router
from fastapi.staticfiles import StaticFiles

# After app.add_middleware(...) around line 126
app.include_router(web_router)
logger.info("Web UI routes mounted")

# In startup_event() around line 156, after creating upload directory
try:
    upload_dir = Path(config.UPLOAD_DIR)
    frames_dir = Path(config.UPLOAD_DIR).parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    app.mount("/static/videos", StaticFiles(directory=str(upload_dir)), name="videos")
    app.mount("/static/frames", StaticFiles(directory=str(frames_dir)), name="frames")
    logger.info("Static file serving enabled for web UI")
except Exception as e:
    logger.warning(f"Static file mounting failed: {e}")
```

## Verify Installation

1. Start the server:
   ```bash
   python -m insurance_mvp.api.main
   ```

2. Check logs for:
   ```
   INFO: Web UI routes mounted
   INFO: Static file serving enabled for web UI
   ```

3. Access the web UI:
   ```
   http://localhost:8000/queue
   http://localhost:8000/metrics
   ```

4. Check available routes:
   ```bash
   curl http://localhost:8000/docs
   ```

   You should see web UI routes listed:
   - GET `/`
   - GET `/queue`
   - GET `/review/{claim_id}`
   - GET `/metrics`

## Testing

Test the integration:

```bash
# 1. Upload a test video via API
curl -X POST http://localhost:8000/claims/upload \
  -H "X-API-Key: dev_key_write" \
  -F "video=@test.mp4"

# 2. Get the claim_id from response
# {
#   "claim_id": "claim_20260217120000_abc123",
#   ...
# }

# 3. Wait for processing to complete
curl http://localhost:8000/claims/claim_20260217120000_abc123/status \
  -H "X-API-Key: dev_key_read"

# 4. Once status is "assessed", visit web UI
# http://localhost:8000/queue
# Click on the claim to review
```

## Troubleshooting

### Error: "Template not found"

Check template directory exists:
```bash
ls -la insurance_mvp/templates/
# Should show: base.html, queue.html, review.html, metrics.html
```

### Error: "StaticFiles not found"

Install FastAPI with static file support:
```bash
pip install "fastapi[all]"
# or
pip install aiofiles
```

### Error: "Database not initialized"

Ensure the API server is fully started before accessing web routes.

### Videos not loading

Check:
1. Upload directory exists: `ls -la data/uploads/`
2. Static mount is correct: Check startup logs
3. Video file path in database is correct

## Production Notes

For production deployment:

1. **Use a reverse proxy** (nginx, Caddy) to serve static files
2. **Enable HTTPS** with SSL certificates
3. **Add authentication** to protect web routes
4. **Enable template caching** in Jinja2
5. **Use CDN** for TailwindCSS, Video.js, Chart.js

Example nginx config:

```nginx
location /static/ {
    alias /var/www/insurance_mvp/data/;
    expires 1d;
}

location / {
    proxy_pass http://localhost:8000;
    proxy_set_header Host $host;
}
```

---

That's it! The web UI is now integrated with your API.
