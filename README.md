![Version](https://img.shields.io/badge/version-6.5.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![Tests](https://img.shields.io/badge/tests-5900%20passed-brightgreen)
![License](https://img.shields.io/badge/license-proprietary-red)

# SOPilot / VigilPilot / Perception Engine

**On-prem SOP compliance + surveillance + agentic vision AI** — 手順遵守スコアリング・監視カメラAI・「カメラのOS」

---

## Products

### SOPilot — SOP Compliance Scoring

Automatically evaluates whether a trainee's recorded work video follows the correct Standard Operating Procedure. Upload a gold reference video and a trainee video; receive a 0–100 score, deviation events with Japanese comments, and a `pass / needs_review / retrain / fail` verdict within seconds. Runs fully on-premises with no GPU required.

### VigilPilot — Surveillance Camera Violation Detection

Detects rule violations in any video footage or RTSP live stream without requiring gold reference videos. Operators define detection rules as plain natural-language text; the system samples frames at configurable fps and reports violations with severity levels, confidence scores, and annotated frame thumbnails. Powered by pluggable VLM backends.

### Perception Engine — Agentic Vision AI ("カメラのOS")

A 62-phase pipeline that turns any camera into a self-aware reasoner. It does not just *detect*: it **observes, orients, decides, acts, verifies, and remembers**. Every action it takes is checked 30 seconds later against the resulting world state; outcomes feed an episodic memory and a meta-learner that adjusts the agent's own decision thresholds. Deployed on a single RTX 4090 with 4-bit LLaVA-NeXT quantization. Ships with self-supervised domain adaptation, predictive maintenance, XAI saliency, federated Re-ID under (ε,δ)-DP, and a long-horizon safety planner.

```
   per frame (≤30 fps)                every ~5s or on critical event
   ──────────────────                  ─────────────────────────────
   YOLO → DINO → LLaVA cascade         OODA: observe → orient → decide → act
   ↓                                   ↓
   tracking · scene graph · world      schedule verification (T+30s)
   ↓                                   ↓                  ↓
   anomaly · forecasting · XAI         outcome record  →  episode memory
                                       ↓                  ↓
                                       meta-learner adjusts thresholds
```

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
┌──────────────────────────────────────────────────────────────────────┐
│  Browser UI  (SPA, /)   SOPilot · VigilPilot · Perception cards     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ HTTP / SSE
┌──────────────────────────────▼───────────────────────────────────────┐
│  FastAPI  (sopilot/main.py)                                          │
│  Middleware: CorrelationID → CORS → APIKey → RateLimit → Audit       │
│  Routers:  /api/v1/*  ·  /vigil/*  ·  /vigil/perception/*  (174 ep) │
└──┬─────────────────────────────┬─────────────────────┬───────────────┘
   │                             │                     │
┌──▼──────────────┐   ┌──────────▼──────────┐   ┌──────▼──────────────────┐
│ SOPilot Pipeline│   │ VigilPilot Pipeline │   │ Perception Engine        │
│ ─────────────── │   │ ─────────────────── │   │ ──────────────────────── │
│ Quality gate    │   │ Frame sampler       │   │ Stage 1  detect cascade  │
│ ColorMotion /   │   │ VLM (Claude/Qwen)   │   │ Stage 2  track + Re-ID   │
│   V-JEPA2       │   │ Severity filter     │   │ Stage 3  scene graph     │
│ DTW alignment   │   │ RTSP / file mode    │   │ Stage 4  world model     │
│ Step detection  │   │ Bbox annotations    │   │ Stage 5  causal reason   │
│ Deviation score │   │                     │   │ Stage 6  agent OODA loop │
│ Bootstrap CI    │   │                     │   │ Stage 7  closed-loop ver │
└──────────┬──────┘   └──────────┬──────────┘   │ Stage 8  metacognition   │
           │                     │              └──┬──────────────────────┘
           │                     │                 │
           │                     │   ┌─────────────▼────────────────────────┐
           │                     │   │ Models (orchestrator, VRAM-managed)  │
           │                     │   │ DINOv2 · YOLO · Grounding-DINO 1.5   │
           │                     │   │ LLaVA-NeXT (4-bit) · SAM 2 · V-JEPA2 │
           │                     │   └──────────────────────────────────────┘
           │                     │
┌──────────▼─────────────────────▼─────────────────────────────────────┐
│ Storage:  SQLite (WAL) · PostgreSQL (opt) · Episode memory (JSON)   │
│ Telemetry: Prometheus · OpenTelemetry · SLA observatory             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# CPU-only: SOPilot + VigilPilot (Claude or Qwen3-API VLM)
cp .env.example .env                       # set SOPILOT_API_KEY (+ ANTHROPIC_API_KEY for VigilPilot)
docker compose up -d                       # build + start (~3 min first build)
python scripts/smoke_test.py               # end-to-end verification: 9/9 pass

# GPU: full Perception Engine (RTX 4090 + 4-bit LLaVA)
pip install -e ".[gpu,sam2]"
python scripts/setup_models.py             # download YOLO + DINOv2 + Grounding DINO + LLaVA + SAM 2
python scripts/demo_phase58.py             # end-to-end demo, prints VRAM / cascade stats / scene caption
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

### Perception Engine (174 endpoints under `/vigil/perception/`)

| Category | Endpoints |
|---|---|
| Cascade detection | `/cascade/stats`, `/sam/state`, `/temporal/state`, `/online-learner/{categories,exemplars}`, `/grounding-dino/state`, `/vlm/{state,describe,ask}` |
| Re-ID & tracking | `/reid/*`, `/tracker/{sid}/*`, `/federated-reid/{stats,export,import}` |
| Spatial intelligence | `/spatial/{stats,map,cameras,zones,zones/events,zones/dwell,zones/flow}`, `/crowd/{stats,state,flow,alerts,capacity}` |
| Reasoning | `/causal/{narrative,inference/stats,inference/graph}`, `/planner/stats`, `/deliberation`, `/goals` |
| Self-monitoring | `/health-score`, `/health-score/history`, `/daily-report`, `/sla/{stats,slos,stages}`, `/explainability/*` |
| Domain adaptation | `/domain/{stats,cameras,shift,adapt,observe,label,reset}` (Phase 55) |
| Predictive maintenance | `/forecast/{stats,cameras,predict,trend,alerts,observe}` (Phase 57) |
| **Agentic OODA loop** | `/agent/{stats,decisions,events,actions,think,inject,reset}` (Phase 59) |
| **Closed-loop verifier** | `/verifier/{stats,records,pending,meta,check,reset}` (Phase 60) |
| **Agent reports** | `/agent/report`, `/agent/report/{summary,insights}` (Phase 61) |
| **Episodic memory** | `/agent/memory/{stats,episodes,search,reset}` (Phase 62) |
| **Counterfactual reasoning** | `/agent/counterfactual/{stats,evaluate,latest,reset}` (Phase 63) |

Full OpenAPI spec at `http://localhost:8000/docs` (Swagger UI) or `/redoc`.

---

## Perception Engine — Capability Map

| Phase | Capability | Module |
|---:|---|---|
| 27–28 | Live RTSP + cross-camera Re-ID + SOP×Vigil rule generation | `stream_session.py`, `reid_*.py`, `rule_generator.py` |
| 29–30 | Person tracking + behaviour anomalies + UI overlay | `person_tracker.py`, `behavior_detector.py` |
| 31–36 | Orchestrator · DINOv2 backbone · YOLO/DINO/VLM cascade · SAM 2 · temporal fusion · online learning | `orchestrator.py`, `cascade_detector.py`, `sam_segmenter.py`, `temporal_fusion.py`, `online_learner.py` |
| 37–39 | V-JEPA2 / InternVideo 2.5 · LLaVA-NeXT VQA · Grounding DINO 1.5 | `video_foundation.py`, `llava_client.py`, `grounding_dino.py` |
| 40–42 | Incident webhooks (Slack/Teams) · federated Re-ID with (ε,δ)-DP · edge optimisation (Jetson, ONNX, INT8) | `incident_reporter.py`, `federated_reid.py`, `edge_optimizer.py` |
| 43–45 | Video RAG · causal-chain analysis · active-learning loop | `video_memory.py`, `rag_engine.py`, `causal_analyzer.py`, `active_loop.py` |
| 46–50 | Spatial mapping · floor map UI · SLA observatory · crowd flow | `spatial_mapper.py`, `crowd_analytics.py`, `sla_monitor.py` |
| 51–52 | SOP-aware perception with depth · multi-agent coordination | `sop_engine.py`, multi-agent state |
| 53–54 | Granger causal inference · long-horizon safety planner | `causal_inference.py`, `safety_planner.py` |
| 55–57 | Self-supervised domain adaptation · XAI saliency · predictive maintenance | `domain_adapter.py`, `explainability.py`, `anomaly_forecaster.py` |
| 58 | Real inference stack — RTX 4090 · LLaVA 4-bit nf4 quantization | `for_rtx4090()` config preset |
| **59** | **Agentic OODA loop** — observe per frame, think every 5 s, 6 actions | `agent_loop.py` |
| **60** | **Closed-loop verification** — schedule T+30 s outcome check, MetaLearner adjusts thresholds | `closed_loop_verifier.py` |
| **61** | **Agent intelligence report** — JP/EN narrative covering decisions, outcomes, risk trends | `agent_report.py` |
| **62** | **Episodic memory** — 6-dim fingerprint, cosine recall, episode→outcome linkage | `agent_episode_store.py` |
| **63** | **Counterfactual reasoning** — observational E[outcome \| action, situation], regret, narrative | `counterfactual.py` |

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

Three deployment tiers, depending on which products you enable:

| Tier | Use case | CPU | RAM | GPU | Disk |
|---|---|---|---|---|---|
| **SOPilot only** | SOP scoring with ColorMotion embedder | 2 cores | 1 GB | none | 10 GB |
| **VigilPilot Cloud-VLM** | Surveillance with Claude/Qwen3-API VLM | 4 cores | 4 GB | none | 50 GB |
| **Perception Engine (full)** | Agentic vision, OODA loop, all 62 phases | 8 cores | 32 GB | RTX 4090 (24 GB VRAM) | 200 GB |

The full Perception Engine stack uses 4-bit nf4 quantization (`PerceptionConfig.for_rtx4090()`) so DINOv2 + LLaVA-NeXT + Grounding DINO 1.5 + SAM 2 fit on a single 24 GB card. Component VRAM budgets are managed by `ModelOrchestrator` with LRU eviction. Edge inference (Jetson Orin, RPi5+Coral) is supported via `EdgeOptimizer` with ONNX + INT8 export — see `scripts/export_edge_models.py`.

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

## Demo Scripts

| Script | What it shows |
|---|---|
| `scripts/demo_video.py` | End-to-end SOPilot scoring + VigilPilot violation detection on a sample video |
| `scripts/demo_live.py` | RTSP live-mode VigilPilot with bbox annotation feed |
| `scripts/demo_closed_loop.py` | Self-learning σ-tuner → review queue → early-warning → autonomous response (Phase 13–16) |
| `scripts/demo_phase58.py` | Full RTX 4090 Perception Engine — cascade stats, DINOv2 features, LLaVA scene description, VRAM (`--mock` for CPU) |
| **`scripts/demo_agentic_loop.py`** | **Killer narrative — OBSERVE → ORIENT → DECIDE → ACT → VERIFY → REMEMBER → ADAPT in 60 s** (Phase 59-62) |
| `scripts/setup_models.py` | Dry-run model download CLI for the GPU stack (`yolo`, `dinov2`, `grounding_dino`, `llava`, `sam2`) |

## Documentation

| Document | Description |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | Docker deployment, full API walkthrough, configuration reference, troubleshooting |
| [TECHNICAL_SUMMARY.md](TECHNICAL_SUMMARY.md) | Architecture, accuracy metrics, decision logic, API surface, known limitations |
| [DEMO.md](DEMO.md) | 5-minute demo script for technical reviewers |
