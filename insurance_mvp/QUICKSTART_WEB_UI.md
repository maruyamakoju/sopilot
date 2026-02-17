# Quick Start - Web UI

Get the insurance claim review web UI running in 5 minutes.

## Prerequisites

- Python 3.10+
- FastAPI project already set up
- Database initialized

## 1. Verify Files (30 seconds)

```bash
# Check templates exist
ls insurance_mvp/templates/
# Should show: base.html, queue.html, review.html, metrics.html

# Check web routes
ls insurance_mvp/api/web_routes.py
```

## 2. Install Dependencies (1 minute)

```bash
# If not already installed
pip install fastapi[all] jinja2 aiofiles
```

## 3. Mount Web Routes (2 minutes)

Edit `insurance_mvp/api/main.py`:

### Add imports at top:
```python
from insurance_mvp.api.web_routes import router as web_router
from fastapi.staticfiles import StaticFiles
from pathlib import Path
```

### Mount routes after app creation:
```python
# After: app.add_middleware(CORSMiddleware, ...)

app.include_router(web_router)
```

### Add static files in startup_event():
```python
# In startup_event(), after creating upload directory:

frames_dir = Path(config.UPLOAD_DIR).parent / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static/videos", StaticFiles(directory=config.UPLOAD_DIR), name="videos")
app.mount("/static/frames", StaticFiles(directory=str(frames_dir)), name="frames")
```

## 4. Start Server (30 seconds)

```bash
python -m insurance_mvp.api.main
```

Look for these log messages:
```
INFO: Web UI routes mounted
INFO: Static file serving enabled
```

## 5. Access Web UI (30 seconds)

Open browser:

- **Queue**: http://localhost:8000/queue
- **Metrics**: http://localhost:8000/metrics

## 6. Test with Sample Data (1 minute)

Upload a test video:

```bash
curl -X POST http://localhost:8000/claims/upload \
  -H "X-API-Key: dev_key_write" \
  -F "video=@test_video.mp4"
```

Wait for processing, then refresh queue page to see the claim.

## That's it!

You now have a production-ready web UI running.

## Next Steps

- Read full documentation: `WEB_UI_INTEGRATION.md`
- Customize branding: Edit `templates/base.html`
- Add authentication: See integration guide
- Deploy to production: Follow deployment checklist

## Troubleshooting

### Templates not found
```bash
# Check path
python -c "from pathlib import Path; print(Path('insurance_mvp/templates').resolve())"
```

### Static files not loading
```bash
# Create directories
mkdir -p data/uploads data/frames
```

### Port already in use
```bash
# Change port in main.py or use:
uvicorn insurance_mvp.api.main:app --port 8001
```

## Support

For detailed help, see:
- `WEB_UI_INTEGRATION.md` - Full integration guide
- `templates/README.md` - Template documentation
- `api/MOUNT_WEB_UI.md` - Step-by-step mounting

---

**Total Setup Time**: 5 minutes
**Difficulty**: Easy
**Status**: Production-ready
