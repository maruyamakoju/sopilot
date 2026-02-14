# SOPilot MVP

SOPilot is an on-prem MVP for SOP execution scoring from work videos.
This repository implements a practical first slice:

- Video ingestion (`/videos`, `/gold`)
- Async ingest jobs (`/videos/jobs/{id}`)
- Clip embedding extraction (`V-JEPA2` hub/local + fallback)
- **GPU-accelerated inference** (V-JEPA2 with torch.compile, dynamic batching)
- **GPU-accelerated DTW** (CuPy-based, 10-30x speedup)
- Step boundary detection (change points)
- Gold vs trainee alignment (DTW)
- Async scoring queue with explainable deviations
- Similar failure retrieval (`/search`)
- Nightly domain adaptation job (`/train/nightly`)
- Field-facing UI (`/ui`) with synced Gold/Trainee playback and deviation timeline
- Audit PDF export (`/score/{id}/report.pdf`)
- Audit trail API (`/audit/trail`)
- Signed audit export (`/audit/export`)
- Queue metrics API (`/ops/queue`)
- **Prometheus metrics** (`/metrics` endpoint for monitoring)
- **Structured logging** (JSON output with structlog)
- Role-based access control (`viewer` / `operator` / `admin`)
- Optional privacy masking in ingest preprocessing (static regions and optional face blur)
- Video deletion API (`DELETE /videos/{video_id}`)

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies.
3. Run the API.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
set SOPILOT_QUEUE_BACKEND=inline
uvicorn sopilot.main:app --reload
```

API docs: `http://127.0.0.1:8000/docs`
Field UI: `http://127.0.0.1:8000/ui`
Prometheus metrics: `http://127.0.0.1:8000/metrics`

RQ mode (durable jobs):

```powershell
set SOPILOT_QUEUE_BACKEND=rq
set SOPILOT_REDIS_URL=redis://127.0.0.1:6379/0
uvicorn sopilot.main:app --reload
sopilot-worker --queues ingest,score,training
```

## GPU Acceleration (Optional, Recommended)

SOPilot supports **GPU acceleration** for 10-100x performance improvements.

### Requirements

- **NVIDIA GPU** with CUDA 12.x support (RTX 30/40/50 series, A100, H100, etc.)
- **CUDA Toolkit** 12.1+ installed
- **16+ GB GPU memory** (recommended for V-JEPA2 ViT-Large with batch=16-24)

### Installation

```powershell
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install CuPy for GPU-accelerated DTW
pip install cupy-cuda12x

# Verify GPU is detected
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from sopilot.dtw_gpu import is_gpu_available; print(f'GPU DTW: {is_gpu_available()}')"
```

### Configuration

```powershell
# Enable GPU features (default: auto-detect)
set SOPILOT_DTW_USE_GPU=true                 # GPU-accelerated DTW (10-30x faster)
set SOPILOT_EMBEDDER_COMPILE=true            # torch.compile for 20-30% speedup
set SOPILOT_EMBEDDER_BATCH_SIZE=16           # Auto-detect if not set (RTX 5090: 16-24)

# Run worker with GPU
sopilot-worker --queues ingest,score,training
```

### Performance Expectations

| Operation | CPU (NumPy) | GPU (CuPy/CUDA) | Speedup |
|-----------|-------------|-----------------|---------|
| **DTW (2000×2000)** | 2-3s | 0.1-0.3s | **10-30x** |
| **V-JEPA2 (batch=16)** | 8-12s | 0.5-1s | **8-12x** |
| **Ingest (10s video)** | 15-25s | 2-5s | **5-8x** |

### CPU Fallback

All GPU code has **automatic CPU fallback** if CUDA is unavailable:

```powershell
# Disable GPU (testing, CI/CD)
set SOPILOT_DTW_USE_GPU=false
set SOPILOT_EMBEDDER_COMPILE=false
set CUDA_VISIBLE_DEVICES=""  # Hide GPU from PyTorch
```

## Minimal Workflow

1. Register a gold video (`POST /gold`).
2. Poll ingest job (`GET /videos/jobs/{id}`) and obtain `video_id`.
3. Upload trainee video (`POST /videos` with `role=trainee`).
4. Poll ingest job (`GET /videos/jobs/{id}`) and obtain `video_id`.
5. Enqueue scoring (`POST /score`).
6. Poll detailed result (`GET /score/{id}`).
7. Export audit PDF (`GET /score/{id}/report.pdf`).
8. Query nearest failure examples (`GET /search`).
9. Trigger nightly adaptation manually if needed (`POST /train/nightly`).
10. Poll training result (`GET /train/jobs/{id}`).
11. Inspect job audit trail (`GET /audit/trail`).
12. Export signed audit artifact (`POST /audit/export`).
13. Check queue backlog (`GET /ops/queue`).

## Docker (On-Prem Friendly)

```powershell
docker compose up --build
```

- API: `http://127.0.0.1:8000`
- UI: `http://127.0.0.1:8000/ui`
- Redis queue: `redis://127.0.0.1:6379/0`
- Persistent data: `./data`
- Offline model assets mount: `./models` -> `/models` (read only)
- Detailed offline notes: `docs/offline_deploy.md`

## Monitoring & Observability

SOPilot exposes **Prometheus metrics** for production monitoring:

```powershell
# Scrape metrics endpoint
curl http://localhost:8000/metrics
```

**Key Metrics:**
- `sopilot_ingest_jobs_total` - Counter by status (queued, running, completed, failed)
- `sopilot_score_jobs_total` - Counter by status
- `sopilot_job_duration_seconds` - Histogram with percentiles (p50, p95, p99)
- `sopilot_dtw_execution_seconds` - DTW performance tracking (CPU vs GPU)
- `sopilot_embedding_generation_seconds` - V-JEPA2 throughput
- `sopilot_queue_depth` - Current queue backlog by queue name
- `sopilot_gpu_memory_bytes` - GPU memory usage (allocated, reserved, total)

**Grafana Dashboard:** See `monitoring/grafana-dashboard.json` (included in deployment artifacts)

**Structured Logging:**
```powershell
# Enable JSON logging for log aggregation (ELK, Splunk, Loki)
set SOPILOT_LOG_FORMAT=json
uvicorn sopilot.main:app
```

## Important Env Vars

### Core Configuration
- `SOPILOT_EMBEDDER_BACKEND=auto|vjepa2|heuristic` (default: `auto`)
- `SOPILOT_INGEST_EMBED_BATCH_SIZE=8` clip embed batch size during ingest (deprecated, use `SOPILOT_EMBEDDER_BATCH_SIZE`)
- `SOPILOT_EMBEDDER_BATCH_SIZE=16` V-JEPA2 batch size (auto-detect if not set)
- `SOPILOT_UPLOAD_MAX_MB=1024` upload size limit
- `SOPILOT_MIN_SCORING_CLIPS=4` minimum clips required for scoring

### GPU Acceleration (New)
- `SOPILOT_DTW_USE_GPU=auto|true|false` enable GPU-accelerated DTW (default: `auto`)
- `SOPILOT_EMBEDDER_COMPILE=true|false` enable torch.compile for 20-30% speedup (default: `false`)

### Monitoring & Logging (New)
- `SOPILOT_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR` logging level (default: `INFO`)
- `SOPILOT_LOG_FORMAT=json|console` structured logging format (default: `console`)
- `SOPILOT_QUEUE_BACKEND=rq|inline` (default: `rq`)
- `SOPILOT_REDIS_URL=redis://127.0.0.1:6379/0`
- `SOPILOT_RQ_QUEUE_PREFIX=sopilot`
- `SOPILOT_API_TOKEN=<secret>` bearer token auth
- `SOPILOT_AUTH_REQUIRED=true|false` require configured credentials for non-public APIs (default: `true`)
- `SOPILOT_API_TOKEN_ROLE=admin|operator|viewer` role for single bearer token
- `SOPILOT_API_ROLE_TOKENS=admin:tokenA,operator:tokenB,viewer:tokenC`
- `SOPILOT_BASIC_USER=<user>` basic auth user
- `SOPILOT_BASIC_PASSWORD=<password>` basic auth password
- `SOPILOT_BASIC_ROLE=admin|operator|viewer`
- `SOPILOT_AUTH_DEFAULT_ROLE=admin|operator|viewer` role when auth is disabled
- `SOPILOT_AUDIT_SIGNING_KEY=<secret>` enables signed audit export
- `SOPILOT_AUDIT_SIGNING_KEY_ID=<key-id>`
- `SOPILOT_PRIVACY_MASK_ENABLED=true|false`
- `SOPILOT_PRIVACY_MASK_MODE=black|blur`
- `SOPILOT_PRIVACY_MASK_RECTS="0:0:1:0.15;0.70:0.70:1:1"` normalized regions
- `SOPILOT_PRIVACY_FACE_BLUR=true|false`
- `SOPILOT_VJEPA2_VARIANT=vjepa2_vit_large|vjepa2_vit_huge|vjepa2_vit_giant`
- `SOPILOT_VJEPA2_PRETRAINED=true|false` (default: `true`)
- `SOPILOT_VJEPA2_SOURCE=hub|local` (default: `hub`)
- `SOPILOT_VJEPA2_LOCAL_REPO=/models/vjepa2` local repo path for offline load
- `SOPILOT_VJEPA2_LOCAL_CHECKPOINT=/models/checkpoints/vitl.pt` local weights path
- `SOPILOT_NIGHTLY_ENABLED=true|false` nightly scheduler switch
- `SOPILOT_NIGHTLY_HOUR_LOCAL=2` nightly run hour (local time)
- `SOPILOT_NIGHTLY_MIN_NEW_VIDEOS=10` minimum new videos for nightly run
- `SOPILOT_ADAPT_COMMAND=\"...\"` optional external pretraining command
- `SOPILOT_ENABLE_FEATURE_ADAPTER=true|false` apply built-in adapter + reindex

External adaptation command example (bundled):

```powershell
set SOPILOT_ADAPT_COMMAND=torchrun --standalone --nproc_per_node=1 -m sopilot.adapt.train_domain_adapter --job-id {job_id} --data-dir {data_dir} --models-dir {models_dir} --reports-dir {reports_dir} --since "{since}"
```

## Notes

- V-JEPA2 loading follows official torch hub entrypoints (`facebookresearch/vjepa2`), and supports local-repo/local-checkpoint mode for network-isolated environments.
- If V-JEPA2 loading fails in `auto` mode, service falls back to heuristic embedder.
- Upload and ingest are decoupled; `/videos` and `/gold` return job ids and processing continues in worker queues.
- Ingest is clip-streaming based, so long videos do not require full-frame materialization in memory.
- Scoring includes structural penalties (order violations, duplicate alignment pressure, temporal drift, local similarity gap) in addition to DTW cost, so obvious failure patterns are less likely to be over-scored.
- Reindex during nightly adaptation uses index versioning (staging build then atomic pointer swap).
- If task-level index dimensions drift (for example after embedder/model changes), ingest auto-recovers by rebuilding that task index with the current embedding dimension.
- UI file download and playback use authenticated fetch, so token/basic auth deployments remain usable.
- Signed audit export uses HMAC-SHA256 for tamper-evident archival.
- External adaptation command is executed with argument parsing (`shell=False`) for safer operation.
- In production, run API and workers as separate processes using Redis-backed RQ.
- Score and training are asynchronous; use result endpoints to inspect final state.
- All artifacts are stored locally under `data/` (on-prem friendly).

## Go-To-Market Docs

- Concept one-pager: `docs/sopilot_concept_onepager.md`
- Fixed paid PoC package: `docs/paid_poc_fixed_package.md`
- Filled paid PoC example: `docs/paid_poc_offer_example_plant_a.md`
- InfoSec one-pager: `docs/infosec_onepager.md`
- Equipment maintenance demo plan: `docs/equipment_maintenance_demo_plan.md`
- 3-minute recording script: `docs/demo_recording_script_3min.md`
- Paid PoC package template: `docs/paid_poc_package_template.md`
- Field capture guide: `docs/field_capture_guide.md`
- Task definition sheet template: `docs/task_definition_sheet_template.md`
- Data handling spec: `docs/data_handling_spec.md`
- InfoSec quickpack: `docs/infosec_quickpack.md`
- RBAC and signed export notes: `docs/rbac_and_audit_export.md`

## Demo Video Generator

Generate `Gold 1 + failure 5` sample videos for the recommended maintenance demo:

```powershell
python scripts/generate_equipment_maintenance_demo.py --out-dir demo_videos/maintenance_filter_swap
```

Run the full demo flow (`ingest -> score -> PDF -> audit trail`) and export artifacts:

```powershell
python scripts/run_demo_flow.py --base-url http://127.0.0.1:8000 --token demo-token --task-id maintenance_filter_swap --gold demo_videos/maintenance_filter_swap/gold.mp4 --trainee demo_videos/maintenance_filter_swap/missing.mp4 --out-dir demo_artifacts
```

`run_demo_flow.py` also captures:
- `queue_metrics_latest.json` (`GET /ops/queue`)
- `audit_export_<id>.json` (`POST /audit/export` + download) when admin role is available

Run release gate before customer delivery:

```powershell
python scripts/run_release_gate.py --base-url http://127.0.0.1:8000 --token admin-token --task-id maintenance_filter_swap --gold demo_videos/maintenance_filter_swap/gold.mp4 --variants missing,swap,deviation,time_over,mixed --out-json demo_artifacts/release_gate_report.json
```

Build a send-ready sales pack:

```powershell
python scripts/build_sales_pack.py --company "Target Company" --source-offer docs/paid_poc_offer_example_plant_a.md --demo-artifacts-dir demo_artifacts --out-root sales_pack
```

By default the pack includes only the latest score/audit artifacts to keep customer attachments clean.
Use `--include-all-artifacts` to bundle every historical artifact.

Verify signed audit export offline:

```powershell
python scripts/verify_audit_export.py --file data/reports/audit_export_<id>.json --key <secret>
```

## Folder Watch Ingest (Optional)

Shared-folder style ingest for field operations:

```powershell
set SOPILOT_WATCH_ENABLED=true
set SOPILOT_WATCH_DIR=watch_inbox
set SOPILOT_WATCH_TASK_ID=maintenance_filter_swap
set SOPILOT_QUEUE_BACKEND=rq
sopilot-watch
```

Directory conventions:
- `watch_inbox/<task_id>/<role>/*.mp4` (role: `gold|trainee|audit`)
- or use `SOPILOT_WATCH_TASK_ID` with flat drop directory

## Comprehensive Documentation

**For production deployments and detailed configuration:**

- **[API Reference](docs/API_REFERENCE.md)** - Complete REST API documentation with examples
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Docker, Kubernetes, bare metal instructions
- **[Configuration Reference](docs/CONFIGURATION.md)** - All 76+ environment variables explained
- **[Architecture Overview](docs/ARCHITECTURE.md)** - System design with Mermaid diagrams
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Research Background](docs/RESEARCH_BACKGROUND.md)** - V-JEPA2, DTW, and algorithm details

## Performance Benchmarks

**With GPU acceleration (RTX 5090):**

| Benchmark | CPU Baseline | GPU Accelerated | Improvement |
|-----------|--------------|-----------------|-------------|
| **V-JEPA2 Throughput** | 50 clips/s | 150-300 clips/s | **3-6x** |
| **DTW (2000×2000)** | 2800 ms | 120 ms | **23x** |
| **Ingest (10s video)** | 15-25s | 2-5s | **5-8x** |
| **Score Job (500×500)** | 5-8s | 0.5-1s | **8-10x** |

**Run benchmarks locally:**
```powershell
# DTW benchmark (CPU vs GPU)
python benchmarks/benchmark_dtw.py

# V-JEPA2 throughput test
python benchmarks/benchmark_embeddings.py

# End-to-end pipeline latency
python benchmarks/benchmark_end_to_end.py
```

Results saved to `benchmarks/results/*.json`

## Technology Stack

- **API Framework:** FastAPI 0.115+
- **Queue:** Redis Queue (RQ) 1.16+
- **Database:** SQLite 3.35+ (PostgreSQL for production)
- **ML Framework:** PyTorch 2.5+ with CUDA 12.1+
- **GPU Compute:** CuPy 13.0+ for custom kernels
- **Video Processing:** OpenCV 4.10+ (headless)
- **Metrics:** Prometheus Client 0.20+
- **Logging:** structlog 25.0+
- **Validation:** Pydantic 2.10+
- **PDF Generation:** ReportLab 4.2+
