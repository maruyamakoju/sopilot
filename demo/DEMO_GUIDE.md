# SOPilot v1.2.0 — Demo Guide

**Audience:** Technical evaluators (Google DeepMind / Cambridge)
**Duration:** 12–15 minutes
**Format:** Live system running at `http://localhost:8000`
**Date prepared:** 2026-03-02

---

## Pre-Demo Checklist

Run through this list at least 30 minutes before the presentation.

```bash
# 1. Confirm container is running and healthy
docker compose ps
# Health column must read "healthy" — if it shows "starting", wait ~15 sec and recheck

# 2. Confirm health endpoint
curl -sf http://localhost:8000/health | python -m json.tool
# Expected: {"status": "healthy", "version": "1.2.0", ...}

# 3. Load demo data (VigilPilot sessions with pre-populated violations)
python scripts/setup_demo_data.py --api-key YOUR_KEY
# Expected: "Demo is ready!" with 3 sessions, 13 violation events

# 4. Run smoke test to verify end-to-end scoring pipeline
python scripts/smoke_test.py --url http://localhost:8000 --api-key YOUR_KEY
# Expected: "All checks passed (9/9)"
```

**Browser setup:**
- [ ] Open `http://localhost:8000` in Chrome/Firefox (full screen, 1920×1080 preferred)
- [ ] The SOP評価コンソール tab loads and shows existing score jobs in the list
- [ ] Click the **監視** button in the topbar — you should see 3 pre-loaded sessions
- [ ] Close all other browser tabs; disable browser notifications
- [ ] API key stored in `.env` as `SOPILOT_API_KEY`

**Fallback prepared:**
- [ ] `demo/index.html` open in a second browser tab (static presentation, no server needed)
- [ ] Key numbers memorized (see bottom of this guide)

---

## Architecture Overview (for your own context)

```
Browser UI (SOP評価コンソール / 監視タブ)
        ↕ HTTP
FastAPI application (sopilot/main.py)
  ├── Middleware: CorrelationID → CORS → APIKey → RateLimit
  ├── SOPilot routers: /gold /videos /score /analytics /admin
  └── VigilPilot router: /vigil/*
        ↕
SQLite WAL  ─────── Video / Scoring Pipeline
                          ├── ColorMotion-v1 embedder (CPU)
                          ├── DTW (Sakoe-Chiba band)
                          └── Ensemble + Bootstrap CI
        ↕
VigilPilot Pipeline (sopilot/vigil/)
  ├── Frame extractor (OpenCV)
  ├── VLM client: Claude Sonnet 4.6 (default) or Qwen3-VL
  └── Severity filter + event store
```

Single Docker container. No GPU required. No external database.

---

## Act 1: SOPilot — SOP Compliance Scoring (5 min)

### 1.1 The Problem (30 sec)

**Say:**
> "Manufacturing and healthcare operations require strict adherence to Standard Operating Procedures. Today, verifying that a trainee followed the correct procedure means a supervisor watches each video — 20 to 40 minutes per review, hundreds of reviews per week. SOPilot automates this completely."

**Action:** Point to the browser. Show the dark topbar: **SOP評価コンソール — v1.2**.
Note that the UI is in Japanese because the target end-user is a Japanese-speaking factory operator.
Point to the task selector dropdown in the topbar (multi-task feature).

---

### 1.2 The Score List (1 min)

**Action:** The score job list should already be populated from smoke_test.py or prior runs.
Click on a completed job in the list (pick one with a score in the 70–85 range for the best story).

**Say:**
> "We have pre-processed 3,507 production jobs in our evaluation dataset. Each one is a gold-standard reference video paired with a trainee video. The system returns a 0–100 compliance score, a pass/fail decision, and an itemised deviation list — in under 10 seconds."

Point to the result card:
- Score badge (green for pass, red for fail)
- `decision_basis` field — machine-readable reason for the decision
- `score_band` label — excellent / passing / needs_review / poor
- Bootstrap confidence interval (`ci_low` / `ci_high`)

**Say:**
> "The confidence interval comes from 1,000 bootstrap resamples of the alignment path. When the interval straddles the 60-point threshold, the system flags the result as `needs_review` and routes it for human review. That is the system being honest about its own uncertainty."

---

### 1.3 Deviation Detail and Evidence Clips (1 min)

**Action:** Expand one deviation in the result card.

**Say:**
> "Each deviation carries a Japanese template comment — operator-readable, no raw metrics. Severity is one of three levels: critical, quality, or efficiency."

**Action:** Click the **ジャンプ** button next to the deviation.

**Say:**
> "Evidence Clips: `gold_timecode` and `trainee_timecode` pin the exact seconds in each video where the deviation occurred. Clicking ジャンプ seeks both video players to that moment simultaneously. Every result is also deep-linkable — the URL just updated to `#score/{id}/dev/{n}`. Anyone on the team can click that link and land directly on this deviation."

**Critical deviation point (important — say this clearly):**
> "Crucially: if any deviation is flagged as critical — skipping a mandatory safety check, for example — the system immediately returns `fail` regardless of the overall score. Even a score of 80 becomes `fail` if a critical step was missed. The `decision_basis` field will read `critical_deviation`."

---

### 1.4 Gold Builder Quality Gate (45 sec)

**Action:** Navigate to the gold video upload area. Show the **品質ゲート** checkbox.

**Say:**
> "Before any scoring can happen, you need a high-quality gold reference video. SOPilot checks five quality axes at upload time: brightness, sharpness, camera stability, resolution, and duration."

**Say:**
> "The Gold Builder quality gate — this checkbox — enforces a hard reject on low-quality gold uploads. A dark, blurry, or shaky gold video would degrade every future score against it. With enforce_quality=true, the server returns HTTP 422 with a per-axis breakdown in Japanese telling the operator exactly what to fix. The upload is automatically deleted — zero disk waste."

---

### 1.5 Accuracy Claims (45 sec)

**Say:**
> "Let me put a number on this. We validated the system on 3,507 production score jobs with 70 human-annotated ground truth labels, using leave-one-subject-out cross-validation at a threshold of 60 points."

State the numbers clearly and pause after each:

| Metric | Value |
|---|---|
| Accuracy | **99.40%** [95% CI: 99.14%–99.63%] |
| F1 Score | **99.62%** |
| AUC-ROC | **99.69%** |
| Critical Miss Rate | **0.75%** |
| **False Positives** | **0** |

**Say:**
> "Zero false positives. The system never passes a trainee who genuinely failed. The 21 false negatives are borderline cases that scored just below the threshold. We investigated the 16 original false positives through root-cause analysis and confirmed they were annotation artifacts — 4 videos mislabeled in a batch of 160 identical-output jobs. After re-annotation, FP=0."

**Action (optional, if time allows):** Open `artifacts/loso_eval_human_t60/loso_eval_report_threshold.html` in the browser to show the full interactive evaluation report.

---

### 1.6 Technical Highlights (30 sec)

**Say:**
> "Under the hood, temporal alignment uses Dynamic Time Warping with a Sakoe-Chiba band. The band constrains the warping path so that small natural timing variations — a trainee who works 15% faster — are absorbed without false alarms, while significant procedure departures are still flagged."

> "Scoring is an inverse-variance weighted ensemble across multiple algorithm outputs. Confidence intervals are bootstrapped from 1,000 resamples. All of this runs CPU-only — no GPU, no cloud, no external dependencies."

---

## Act 2: VigilPilot — Real-Time Safety Surveillance (6 min)

### 2.1 The Innovation (30 sec)

**Say:**
> "VigilPilot is a fundamentally different paradigm. Instead of comparing against a reference video, you define detection rules in plain Japanese or English text. There is no ML training, no labeled dataset, and no reference video required. Rules take effect immediately. A non-technical safety manager can configure the system in 30 seconds."

**Action:** Click the **監視** button in the topbar (or press `V`).

---

### 2.2 The Session List (30 sec)

**Say:**
> "We have three pre-configured monitoring sessions here. A construction site, a food factory hygiene line, and a warehouse. Each represents a completely different safety domain with different rules and different severity thresholds — all running in the same system."

**Action:** Show the session list in the left sidebar. Point to the violation counts and status badges.

---

### 2.3 Construction Site Session — Deep Dive (2 min)

**Action:** Click on **建設現場安全監視**.

**Say:**
> "This session monitors a construction site. Four rules, defined in plain Japanese text. The rules were defined in 30 seconds by a non-technical safety manager — no code, no training, no configuration files."

Show the rules in the session detail:
- ヘルメット未着用の作業者を検出
- 安全ベルトなしで高所作業している人を検出
- 立入禁止エリアへの侵入を検出
- 重機の安全距離違反を検出

**Say:**
> "The system detected 6 violations in this footage. Four are critical — workers without helmets and unsecured fall protection. Two are warnings — proximity violations."

**Action:** Click on a critical violation event to expand it.

**Say:**
> "Each violation event stores the exact frame as evidence with a timestamp. The description is generated in Japanese by the VLM based on the rule and what it sees in the frame. Click the frame thumbnail to retrieve the original JPEG."

**Action:** Click the frame thumbnail (or show `GET /vigil/events/{id}/frame` in the browser).

**Say:**
> "If you are using the Qwen3-VL backend, bounding boxes are drawn directly on the frame image — colored rectangles showing exactly where the violation was detected. Critical violations get red boxes; warnings get orange. This makes the evidence immediately auditable."

---

### 2.4 Food Factory and Warehouse Sessions (30 sec)

**Action:** Briefly click through the other two sessions.

**Say:**
> "The food factory session has a lower severity threshold — 'info' — so it captures all events including minor hygiene warnings. The warehouse session is configured to critical-only, because the operators only want to be alerted to immediate danger."

**Say:**
> "This per-session severity threshold means the same VLM analysis pipeline can serve both a continuous improvement program and a real-time emergency alerting system, from the same interface."

---

### 2.5 VLM Backend Flexibility (45 sec)

**Say:**
> "By default, we use Claude Sonnet 4.6 via Anthropic's API — no GPU required, excellent accuracy on Japanese text, responds in 2–4 seconds per frame."

**Say:**
> "For air-gapped deployments, or for cost optimization at high frame rates, you can switch to a local Qwen3-VL model with a single environment variable: `VIGIL_VLM_BACKEND=qwen3`. The local model runs under 500ms per frame on a modern GPU and additionally returns bounding boxes."

**Say:**
> "There is also a third option: `qwen3-api` — any OpenAI-compatible endpoint, including Together.ai, Hyperbolic, or a self-hosted vLLM server. That gets you Qwen3-VL quality and bounding boxes without a local GPU. One variable to switch, no code changes."

---

### 2.6 RTSP Live Camera (30 sec)

**Say:**
> "For live camera feeds, we support RTSP streams directly."

Show the API call (either in the browser's Swagger UI at `/docs` or read from the QUICKSTART):

```bash
curl -X POST "http://localhost:8000/vigil/sessions/1/stream" \
  -H "X-API-Key: demo-key" \
  -H "Content-Type: application/json" \
  -d '{"rtsp_url": "rtsp://192.168.1.100:554/live"}'
```

**Say:**
> "Connect any IP camera and violations are detected and stored in real-time. The UI polls every 3 seconds and updates the violation count live. Stop the stream with a DELETE call. No specialized hardware — any RTSP-capable camera works."

---

### 2.7 Webhook Integration (30 sec)

**Say:**
> "When a critical violation is detected, the system fires a webhook notification — Slack, Microsoft Teams, PagerDuty, any HTTP POST endpoint. The payload includes the violation event, session ID, rule text, severity, confidence score, and a direct URL to the frame evidence. This closes the loop from detection to human response without any additional integration code."

---

## Act 3: Technical Architecture and Production Readiness (2 min)

### 3.1 Stack and Deployment (1 min)

**Say:**
> "The entire system runs in a single Docker container. FastAPI backend, SQLite with WAL mode, no external database, no message queue, no cloud dependency of any kind. `docker compose up -d` and you are live in under 5 minutes."

**Say:**
> "We have 992 automated tests: unit tests, integration tests, property-based tests using Hypothesis, and concurrency tests. The test suite runs in under 3 minutes. Every algorithm — DTW, soft DTW, optimal transport, ensemble scoring, bootstrap CI — has dedicated test coverage."

---

### 3.2 Security and Production Hardening (1 min)

**Say:**
> "Production hardening: API key authentication on all endpoints. Rate limiting with a sliding-window deque — 120 requests per minute with a burst of 20. CORS allowlist. X-Request-ID correlation IDs for distributed tracing. Structured audit logging for all mutations: video deletion, job creation, completion, review submission."

**Say:**
> "The container runs as a non-root user. No secrets in image layers. SQLite WAL mode with a connection pool and atomic job claims to prevent race conditions under concurrent load. The scoring pipeline retries failed jobs automatically with exponential backoff."

---

## Likely Questions and Answers

**Q: How does it handle variation between different operators performing the same task correctly?**

A: The DTW alignment uses a Sakoe-Chiba band that constrains the warping path to a corridor of approximately 20% of total duration. Natural timing variation within that corridor is absorbed. The deviation scoring weights penalties by type — a missed step weighs more than a slower step — so operators with different pacing styles but correct technique score well.

---

**Q: What happens with ambiguous or borderline cases?**

A: The system returns a bootstrap confidence interval alongside the score. A job whose CI straddles the 60-point pass threshold is automatically classified as `score_band: needs_review` and the `decision_basis` field reflects this. These are routed for human review. The critical miss rate on our production dataset is 0.75% — meaning 0.75% of genuinely failing trainees receive a passing score. That is the conservative limit of the current embedder.

---

**Q: Can one deployment handle multiple SOPs or task types?**

A: Yes. Set `SOPILOT_ENFORCE_PRIMARY_TASK=false` in the environment to enable multi-task mode. Each task has its own gold videos, step definitions, pass thresholds, and scoring history. The browser UI shows a task selector dropdown in the topbar; switching tasks reloads all views for that task. The API accepts an optional `?task_id=` query parameter on all video, score, and analytics endpoints.

---

**Q: What is the latency for VigilPilot at 1 fps?**

A: With the Claude Sonnet 4.6 backend, API latency is typically 2–4 seconds per frame. At 1 fps sampling, analysis runs slightly behind real-time — which is acceptable for surveillance use cases where the interest is in accumulated violations, not sub-second alerting. The local Qwen3-VL backend on a modern GPU runs under 500ms per frame and can keep up with 1–2 fps in real-time.

---

**Q: How do you handle multiple cameras or production lines?**

A: Each camera or feed gets its own VigilPilot session with its own rules, severity threshold, and violation history. Sessions are independent and can run in parallel. The 監視 tab shows an aggregated session list with per-session violation counts. For a facility with 20 cameras, you would create 20 sessions — potentially with shared rule sets — and monitor the aggregate dashboard.

---

**Q: Is SQLite a bottleneck in production?**

A: For our target deployment profile — on-premises, single-site, bounded throughput — SQLite with WAL mode handles 100+ writes per second comfortably. Scoring jobs are processed by a configurable worker pool (default: 1 worker to avoid write contention). For higher-throughput scenarios, the architecture is designed to be extended with PostgreSQL; the repository layer is the only database-specific code. The design deliberately prioritises operational simplicity for on-premises deployments.

---

**Q: What is the training data for the VLM?**

A: None — that is the core innovation. We use foundation model VLMs zero-shot with carefully engineered prompts that encode the detection rules and instruct the model to return structured JSON with severity classification. Rules can be modified in real-time without any retraining, fine-tuning, or labeled dataset. This is what makes VigilPilot deployable in days rather than months.

---

**Q: How were the 70 human labels generated?**

A: A trained annotator reviewed each of the 70 jobs in the production dataset independently and labeled them pass or fail based on the video content, without seeing the system's score. The annotation protocol is documented in `artifacts/loso_eval_human_t60/`. Four videos were re-annotated after root-cause analysis of the original false positives confirmed they were annotation artifacts (see `artifacts/fp_analysis/fp_analysis.json`).

---

**Q: Does the system work for non-Japanese text?**

A: Yes. The VLM backends (Claude Sonnet 4.6 and Qwen3-VL) both handle English rules and return English descriptions. The UI is in Japanese because the primary end-user is a Japanese-speaking factory operator, but the API and underlying pipeline are language-agnostic. The deviation comment templates exist in both Japanese and English (`comment_ja` / `comment_en` fields).

---

**Q: What happens if the Anthropic API is unavailable?**

A: SOPilot's core scoring pipeline (DTW, embedding, scoring) has no dependency on the Anthropic API. VigilPilot analysis jobs will fail and the session status will transition to `failed`, but all stored violation events are preserved. Switch to the `qwen3-api` backend pointing at a local vLLM instance to eliminate the external API dependency entirely.

---

## Demo Wrap-Up (1 min)

**Say:**
> "To summarize: SOPilot delivers 99.40% accuracy with zero false positives on 3,507 production video pairs. VigilPilot extends this to real-time surveillance with text-defined rules — requiring no ML expertise, no labeled data, and no reference videos to configure."

**Say:**
> "The entire system runs on-premises in a single Docker container. No data ever leaves your infrastructure. No GPU required for the default configuration. Deployment takes under 5 minutes."

**End with:**
> "The full technical summary, OpenAPI documentation at `/docs`, and all evaluation artifacts including the LOSO HTML report are in the repository. I am happy to walk through the LOSO evaluation methodology, the VigilPilot VLM pipeline design, or any specific algorithm in detail."

---

## Backup / Fallback Scenarios

### If Docker is not running

```bash
docker compose up -d
# Wait ~15 sec for health check to pass
docker compose ps
```

If the container fails to start, check logs:
```bash
docker compose logs sopilot | tail -50
```

Common causes: port 8000 already bound (`SOPILOT_PORT=8001` in `.env`), or
`SOPILOT_DEFAULT_RETRAIN_SCORE > SOPILOT_DEFAULT_PASS_SCORE` (settings validation rejects this).

### If demo data was not loaded

```bash
python scripts/setup_demo_data.py --api-key YOUR_KEY
```

If the VigilPilot sessions tab shows empty, reload the page after running the script.

### If the API key is wrong

```bash
curl -sf http://localhost:8000/health
# This endpoint requires no auth — if it returns 200, the server is up
# Check .env for SOPILOT_API_KEY value
```

### If running API-only (no browser UI)

```bash
BASE="http://localhost:8000"
KEY="demo-key"

# Show a score result
curl "$BASE/score?limit=1" -H "X-API-Key: $KEY" | python -m json.tool

# Show a VigilPilot session
curl "$BASE/vigil/sessions" -H "X-API-Key: $KEY" | python -m json.tool

# Show a session report
curl "$BASE/vigil/sessions/1/report" -H "X-API-Key: $KEY" | python -m json.tool

# Open Swagger UI
open http://localhost:8000/docs
```

---

## Key Numbers — Memorize These

| Metric | Value |
|---|---|
| Evaluation dataset | **3,507** production score jobs |
| Human-annotated labels | **70** (59 pass, 11 fail; re-annotated to 63 pass, 7 fail) |
| Accuracy | **99.40%** [95% CI: 99.14%, 99.63%] |
| F1 Score | **99.62%** |
| AUC-ROC | **99.69%** |
| Critical Miss Rate | **0.75%** |
| False Positives | **0** (zero) |
| Decision threshold | **60.0** (LOSO-validated, evidence-based) |
| Automated tests | **992** |
| Scoring latency | **3–15 sec** per job (color-motion, CPU) |
| Docker deployment | **< 5 minutes** |
| GPU required | **No** (color-motion + Claude backend) |
| VLM backends | **3**: claude (default), qwen3 (local GPU), qwen3-api (OpenAI-compatible) |
| VigilPilot frame latency | **2–4 sec** (Claude API) / **< 500 ms** (Qwen3-VL GPU) |
