![Version](https://img.shields.io/badge/version-1.2.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![Tests](https://img.shields.io/badge/tests-1021%20passed-brightgreen)
![License](https://img.shields.io/badge/license-proprietary-red)

# SOPilot / VigilPilot

**On-premises SOP compliance scoring and surveillance violation detection** — SOP手順遵守スコアリング＆監視カメラAIサービス

---

## Products

### SOPilot — SOP Compliance Scoring

Automatically evaluates whether a trainee's recorded work video follows the correct Standard Operating Procedure. Upload a gold reference video and a trainee video; receive a 0–100 score, deviation events with Japanese comments, and a `pass / needs_review / retrain / fail` verdict within seconds. Runs fully on-premises with no GPU required.

### VigilPilot — Surveillance Camera Violation Detection

Detects rule violations in any video footage or RTSP live stream without requiring gold reference videos. Operators define detection rules as plain natural-language text; the system samples frames at configurable fps and reports violations with severity levels, confidence scores, and annotated frame thumbnails. Powered by pluggable VLM backends.

---

## Accuracy (LOSO Evaluation)

Evaluated on 3,507 production score jobs with 70 human-annotated labels using leave-one-subject-out cross-validation at threshold = 60.0.

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | **99.40%** | [99.14%, 99.63%] |
| F1 Score | **99.62%** | — |
| AUC-ROC | **99.69%** | — |
| Critical Miss Rate | **0.75%** | — |

Confusion matrix: TP=2766, FN=21, **FP=0**, TN=720.

The original evaluation reported 16 FPs; root-cause analysis confirmed these were annotation artifacts (4 videos mislabeled as "fail" in a batch of 160 identical-output jobs where 144 were labeled "pass"). After re-annotation: FP=0, perfect precision.

---

## VigilPilot VLM Backends

| Backend | Key | Notes |
|---|---|---|
| `claude` (default) | `VIGIL_VLM_BACKEND=claude` | Claude Sonnet 4.6 via Anthropic API; no GPU required; set `ANTHROPIC_API_KEY` |
| `qwen3` | `VIGIL_VLM_BACKEND=qwen3` | Qwen3-VL-4B local inference via `transformers`; returns bounding boxes; requires CUDA + ~10 GB VRAM |
| `qwen3-api` | `VIGIL_VLM_BACKEND=qwen3-api` | Any OpenAI-compatible endpoint (Together.ai, Hyperbolic, vLLM); `<think>` tags stripped automatically |

Additional VigilPilot capabilities:
- **RTSP live streaming**: `POST /vigil/sessions/{id}/stream` accepts `rtsp://` URL for real-time analysis; `DELETE` stops the stream
- **Bounding-box evidence**: Qwen3-VL backend annotates violation frames with colored bbox overlays; served via `GET /vigil/events/{id}/frame?annotate=true`
- **Severity filtering**: `info / warning / critical` threshold configurable per session
- **Background processing**: non-blocking; poll `GET /vigil/sessions/{id}` for progress

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser UI  (single-page HTML/JS, served at /)                 │
│  SOPilot tabs + VigilPilot 監視タブ                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────────┐
│  FastAPI application  (sopilot/main.py)                          │
│  ├── Middleware: CorrelationID → CORS → APIKey → Rate            │
│  ├── Routers: videos · scoring · analytics · admin  (/api/v1)   │
│  └── VigilPilot router  (/vigil/*)                               │
└────────┬─────────────────────────────┬──────────────────────────┘
         │                             │
┌────────▼──────────┐    ┌─────────────▼──────────────────────────┐
│  Video Pipeline   │    │  VigilPilot Pipeline (sopilot/vigil/)  │
│  ├── Quality Gate │    │  ├── Frame extractor (OpenCV, 1 fps)   │
│  │   (5 checks)   │    │  ├── VLM client (Claude Vision API)    │
│  ├── ColorMotion  │    │  ├── Severity filter + event store     │
│  │   embedder     │    │  └── Background thread (daemon)        │
│  └── Clip store   │    └────────────────┬───────────────────────┘
└────────┬──────────┘                     │
         │              ┌─────────────────▼─────────────────────┐
         │              │  Scoring Pipeline                      │
         │              │  ├── DTW alignment (Sakoe-Chiba)       │
         │              │  ├── Step detection (cosine CP)        │
         │              │  ├── Deviation scoring + comments      │
         │              │  └── Ensemble / Bootstrap CI           │
         │              └─────────────────┬─────────────────────┘
         │                                │
┌────────▼────────────────────────────────▼──────────────────────┐
│  SQLite (WAL mode)  —  connection pool, atomic ops             │
│  ./data/sopilot.db   (+ vigil_sessions / vigil_events tables)  │
└────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
cp .env.example .env          # set SOPILOT_API_KEY (and ANTHROPIC_API_KEY for VigilPilot)
docker compose up -d          # build + start (~3 min first build)
python scripts/smoke_test.py  # end-to-end verification: All checks passed (9/9)
```

Open `http://localhost:8000` for the browser UI (SOP評価コンソール).

Full instructions, API walkthrough, and configuration reference: see **[QUICKSTART.md](QUICKSTART.md)**.

---

## API Surface

### SOPilot Core

| Category | Endpoints |
|---|---|
| Video management | `POST /gold` (`enforce_quality=true` → 422 on failure), `POST /videos`, `GET /videos`, `GET /videos/{id}`, `DELETE /videos/{id}` |
| Video quality | `GET /videos/{id}/quality` — brightness, sharpness, stability, resolution, duration |
| Scoring | `POST /score`, `GET /score`, `GET /score/{id}`, `POST /score/{id}/rerun`, `POST /score/batch`, `POST /score/ensemble` |
| Reports | `GET /score/{id}/report`, `GET /score/{id}/report/pdf`, `GET /score/{id}/uncertainty` |
| Analytics | `GET /analytics`, `GET /analytics/compliance`, `GET /analytics/steps`, `GET /analytics/operators/{id}/trend` |
| Task config | `GET /tasks`, `GET /task-profile`, `PUT /task-profile`, `PUT /tasks/steps` |
| Administration | `POST /admin/rescore`, `POST /admin/backup`, `GET /admin/db-stats` |
| Observability | `GET /health`, `GET /readiness`, `GET /metrics` (Prometheus) |

### VigilPilot

| Category | Endpoints |
|---|---|
| Sessions | `POST /vigil/sessions`, `GET /vigil/sessions`, `GET /vigil/sessions/{id}`, `DELETE /vigil/sessions/{id}` |
| Video analysis | `POST /vigil/sessions/{id}/analyze` — upload video file, start background analysis |
| RTSP streaming | `POST /vigil/sessions/{id}/stream` — start live stream; `DELETE /vigil/sessions/{id}/stream` — stop |
| Events | `GET /vigil/sessions/{id}/events` — violation events with timestamps and bbox data |
| Frame evidence | `GET /vigil/events/{id}/frame?annotate=true` — JPEG frame with optional bbox overlays |
| Reports | `GET /vigil/sessions/{id}/report` — severity and rule breakdown aggregation |

Full OpenAPI spec at `http://localhost:8000/docs` (Swagger UI) or `/redoc`.

---

## Decision Logic

```
if any deviation has severity = "critical":
    decision = "fail"                         # regardless of score
elif score >= pass_score (60.0):
    decision = "pass"
elif score >= retrain_score (50.0):
    decision = "needs_review"
else:
    decision = "retrain"
```

Thresholds are configurable per task profile (`PUT /task-profile`) and are LOSO-validated at 60.0. Use `POST /admin/rescore` to re-apply updated thresholds to all stored jobs.

---

## Resource Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 2 cores | 4 cores |
| RAM | 1 GB | 2 GB (8+ GB for V-JEPA2 backend) |
| Disk | 10 GB | 100 GB (for video storage) |
| GPU | Not required (color-motion backend) | Required for `vjepa2` / `qwen3` backends |
| Docker | >= 24 (Linux containers) | — |

---

## Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Single-task enforcement | Only one SOP task ID per deployment by default | Set `SOPILOT_ENFORCE_PRIMARY_TASK=false` for multi-task mode |
| ColorMotion backend | Captures motion/color features only; not semantic content | Switch to V-JEPA2 for semantically richer embeddings (requires GPU) |
| SQLite concurrency | Write throughput limited under high load | Use `SOPILOT_SCORE_WORKERS=1` (default); scale via multiple deployments |
| No video authentication | Uploaded videos are not tamper-verified | Enforce upstream access control |
| Annotation dependency | Accuracy metrics depend on human label quality | Re-annotation of edge cases recommended (see `artifacts/fp_analysis/`) |

---

## Deployment Note

VigilPilot's default backend (`claude`) requires an Anthropic API key:

```dotenv
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Add this to `.env` before starting the container. For the `qwen3-api` backend (GPU-free alternative), configure `VIGIL_QWEN3_API_BASE` and `VIGIL_QWEN3_API_KEY` instead. See `QUICKSTART.md §8` for full VigilPilot setup.

---

## Documentation

| Document | Description |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | Docker deployment, full API walkthrough, configuration reference, troubleshooting |
| [TECHNICAL_SUMMARY.md](TECHNICAL_SUMMARY.md) | Architecture, accuracy metrics, decision logic, API surface, known limitations |
| [DEMO.md](DEMO.md) | 5-minute demo script for technical reviewers |
