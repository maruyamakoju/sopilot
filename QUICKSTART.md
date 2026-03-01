# SOPilot Quickstart — v1.0.0

On-prem SOP video scoring service. Evaluates trainee videos against gold-standard references
using DTW-based alignment of video embeddings. Produces a 0–100 score and a
`pass / needs_review / retrain / fail` decision. Target: backend / MLOps engineers.

---

## Prerequisites

- Docker Desktop ≥ 24 (Linux containers mode)
- `curl` for API calls
- 2 GB RAM available for the container (`color-motion` backend; 8+ GB for `vjepa2`)
- No GPU required for the default `color-motion` backend

---

## 1. Environment Setup

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```dotenv
SOPILOT_API_KEY=your-secret-key-here
```

All other values have production-safe defaults. Leave `SOPILOT_EMBEDDER_BACKEND=color-motion`
unless a CUDA-capable GPU host is available.

---

## 2. Quick Start

```bash
cp .env.example .env                    # 1. copy template, then set SOPILOT_API_KEY
docker compose up -d                    # 2. build image and start container (first build ~3 min)
docker compose ps                       # 3. confirm Health column reads "healthy"
curl -sf http://localhost:8000/health   # 4. liveness check
```

Expected health response:

```json
{
  "status": "healthy",
  "checks": {
    "database": {"status": "up", "latency_ms": 0.1},
    "disk":     {"status": "up", "free_gb": 120.0, "total_gb": 500.0, "writable": true},
    "embedder": {"status": "up", "name": "color-motion-v1", "failed_over": false}
  },
  "version": "1.0.0"
}
```

The container health check has a 15 s `start_period`. Wait for `docker compose ps` to show
`healthy` before submitting scoring requests.

---

## 3. Step-by-Step: Score a Trainee Video

Set shell variables for convenience:

```bash
KEY="your-secret-key-here"
BASE="http://localhost:8000"
TASK="pilot_task"
```

### 3.1 Upload a Gold (Reference) Video

```bash
curl -X POST "$BASE/gold" \
  -H "X-API-Key: $KEY" \
  -F "file=@/path/to/gold_video.mp4" \
  -F "task_id=$TASK"
```

Response (`VideoIngestResponse`):

```json
{
  "video_id": 1,
  "task_id": "pilot_task",
  "is_gold": true,
  "status": "ready",
  "clip_count": 12,
  "step_boundaries": [0, 3, 7, 12],
  "original_filename": "gold_video.mp4",
  "gold_version": 1,
  "quality": {
    "overall_pass": true,
    "checks": [
      {"name": "brightness",  "passed": true, "value": 118.4},
      {"name": "sharpness",   "passed": true, "value": 142.1},
      {"name": "stability",   "passed": true, "value": 12.3},
      {"name": "resolution",  "passed": true, "value": 720},
      {"name": "duration",    "passed": true, "value": 47.2}
    ],
    "recommendations_ja": [],
    "recommendations_en": []
  }
}
```

`gold_version` is the 1-based upload sequence number for gold videos within the task
(v1 = first gold, v2 = replacement gold, etc.). Note the `video_id` — you will need it
when submitting a score job.

### 3.1.1 Gold Builder Quality Gate (optional strict mode)

By default, quality check is **informational** — a quality failure does not block gold ingest.
To enforce the quality gate and reject low-quality gold videos with an error:

```bash
curl -X POST "$BASE/gold" \
  -H "X-API-Key: $KEY" \
  -F "file=@/path/to/dark_video.mp4" \
  -F "task_id=$TASK" \
  -F "enforce_quality=true"
```

If quality fails, the server returns **HTTP 422** and cleans up the upload automatically:

```json
{
  "error": {
    "code": "QUALITY_GATE_FAILED",
    "message": "Gold動画が品質基準を満たしていません",
    "details": {
      "quality": {
        "overall_pass": false,
        "checks": [
          {"name": "brightness", "passed": false, "value": 28.4,
           "message_ja": "照度不足（平均輝度: 28 < 40）"}
        ],
        "recommendations_ja": ["照明を追加するか、カメラの露出を上げてください"]
      }
    }
  }
}
```

The UI "品質ゲート" checkbox enables this mode for gold uploads via the browser interface.

### 3.2 Upload a Trainee Video

```bash
curl -X POST "$BASE/videos" \
  -H "X-API-Key: $KEY" \
  -F "file=@/path/to/trainee_video.mp4" \
  -F "task_id=$TASK"
```

Response (`VideoIngestResponse`):

```json
{
  "video_id": 2,
  "task_id": "pilot_task",
  "is_gold": false,
  "status": "ready",
  "clip_count": 11,
  "step_boundaries": [0, 4, 8, 11],
  "original_filename": "trainee_video.mp4",
  "quality": {
    "overall_pass": true,
    "checks": [
      {"name": "duration",    "passed": true, "value": 47.2},
      {"name": "resolution",  "passed": true, "value": 720},
      {"name": "brightness",  "passed": true, "value": 118.4},
      {"name": "sharpness",   "passed": true, "value": 142.1},
      {"name": "stability",   "passed": true, "value": 12.3}
    ],
    "recommendations_ja": [],
    "recommendations_en": []
  }
}
```

Videos are processed synchronously at upload time. `status` is `ready` immediately on
success, or `failed` with an `error` field if processing encountered an error.

The `quality` field summarises 5 video quality axes checked at ingest time:
brightness, sharpness, stability (camera shake), resolution, and duration.
If `overall_pass` is `false`, `recommendations_ja` / `recommendations_en` contain
actionable operator instructions (e.g. "照明を追加するか、カメラの露出を上げてください").

### 3.2.1 Detailed Video Quality Report (optional)

```bash
curl "$BASE/videos/2/quality" -H "X-API-Key: $KEY"
```

Returns the full `VideoQualityReport` including per-check messages and thresholds:

```json
{
  "overall_pass": false,
  "checks": [
    {
      "name": "brightness",
      "passed": false,
      "value": 28.4,
      "threshold": 40.0,
      "message_ja": "照度不足（平均輝度: 28 < 40）",
      "message_en": "Too dark (mean brightness: 28)"
    }
  ],
  "recommendations_ja": ["照明を追加するか、カメラの露出を上げてください"],
  "recommendations_en": ["Add more lighting or increase camera exposure"],
  "frame_count_sampled": 20,
  "duration_sec": 47.2,
  "resolution": [1280, 720]
}
```

Quality check is **informational** for trainee videos — a failure does not block scoring.
Re-upload a higher-quality video for more reliable scores. For gold videos, use
`enforce_quality=true` to block low-quality uploads at the gate (see §3.1.1).

### 3.3 Submit a Score Job

```bash
curl -X POST "$BASE/score" \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"gold_video_id": 1, "trainee_video_id": 2}'
```

Response (`ScoreJobResponse`):

```json
{
  "job_id": 1,
  "status": "queued",
  "result": null,
  "weights": null,
  "error": null,
  "created_at": "2026-02-28T09:00:00+00:00",
  "started_at": null,
  "finished_at": null
}
```

### 3.4 Poll for Result

```bash
curl "$BASE/score/1" -H "X-API-Key: $KEY"
```

Poll until `status` is `completed` or `failed`. Typical latency on the `color-motion`
backend: 3–10 s per minute of video on a modern CPU core.

Each deviation in `result.deviations` includes **Evidence Clips** fields:

| Field | Type | Description |
|---|---|---|
| `gold_timecode` | `[start_sec, end_sec]` | Time range in the gold reference video for this deviation |
| `trainee_timecode` | `[start_sec, end_sec]` | Time range in the trainee video for this deviation |

These allow operators to jump directly to the relevant moment in each video. The browser
UI renders these as clickable "ジャンプ" links next to each deviation.

```json
{
  "job_id": 1,
  "status": "completed",
  "result": {
    "score": 87.5,
    "summary": {
      "decision": "pass",
      "decision_reason": "score >= pass_score (60.0)",
      "decision_basis": "score_above_threshold",
      "score_band": "excellent",
      "comment_ja": "スコア 87.5点 — 合格ですが、品質逸脱1件の改善を推奨します。",
      "comment_en": "Score 87.5 — Pass, but 1 quality deviation(s) noted for improvement.",
      "severity_counts": {"critical": 0, "quality": 1, "efficiency": 0},
      "pass_score": 60.0,
      "retrain_score": 50.0
    },
    "metrics": {
      "miss_steps": 0,
      "swap_steps": 0,
      "deviation_steps": 1,
      "over_time_ratio": 0.05,
      "dtw_normalized_cost": 0.123
    },
    "deviations": [
      {
        "type": "step_deviation",
        "step_index": 3,
        "severity": "quality",
        "severity_ja": "品質",
        "comment_ja": "手順4（部品取付）の動作が基準と異なります",
        "comment_en": "Step 4 (Part Assembly) deviates from the standard",
        "severity_description_ja": "品質に影響する可能性がある逸脱です。改善を推奨します。",
        "severity_description_en": "Quality-impacting deviation. Improvement recommended.",
        "gold_timecode": [12.0, 16.0],
        "trainee_timecode": [14.5, 19.0]
      }
    ],
    "confidence": {
      "ci_low": 84.1,
      "ci_high": 90.9,
      "stability": "high"
    }
  },
  "error": null,
  "created_at": "2026-02-28T09:00:00+00:00",
  "started_at": "2026-02-28T09:00:01+00:00",
  "finished_at": "2026-02-28T09:00:04+00:00"
}
```

---

## 4. Score Result Interpretation

### Decision fields

| Field | Type | Description |
|---|---|---|
| `score` | float 0–100 | DTW alignment quality vs gold. Higher = better. |
| `decision` | string | `pass` / `needs_review` / `retrain` / `fail` |
| `decision_basis` | string | Machine-readable reason for the decision (see below) |
| `score_band` | string | Threshold-relative performance label (see below) |
| `severity_counts` | object | Count of deviations by severity: `critical`, `quality`, `efficiency` |
| `confidence.ci_low/ci_high` | float | 95% bootstrap confidence interval |

### `decision_basis` values

| Value | Meaning |
|---|---|
| `critical_deviation` | One or more critical severity deviations detected — always `fail` regardless of score |
| `score_above_threshold` | `score >= pass_score` → `pass` |
| `score_between_thresholds` | `retrain_score <= score < pass_score` → `needs_review` |
| `score_below_retrain` | `score < retrain_score` → `retrain` |

### `score_band` values

| Value | Condition |
|---|---|
| `excellent` | `score >= pass_score × 1.2` |
| `passing` | `pass_score <= score < pass_score × 1.2` |
| `needs_review` | `retrain_score <= score < pass_score` |
| `poor` | `score < retrain_score` |

Default thresholds: `pass_score = 60.0`, `retrain_score = 50.0`.

---

## 5. Key API Calls

All authenticated endpoints require `X-API-Key: <value>`. Public paths (no auth required):
`/health`, `/readiness`, `/status`, `/metrics`.

### List score jobs

```bash
curl "$BASE/score?limit=50" -H "X-API-Key: $KEY"
```

### Re-run a score job

```bash
curl -X POST "$BASE/score/1/rerun" -H "X-API-Key: $KEY"
```

### Ensemble score (multiple gold videos)

```bash
curl -X POST "$BASE/score/ensemble" \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"gold_video_ids": [1, 2, 3], "trainee_video_id": 4}'
```

### Re-apply decision thresholds to all stored jobs

Useful after changing `pass_score` or `retrain_score`:

```bash
# Dry run first (preview changes without writing):
curl -X POST "$BASE/admin/rescore?dry_run=true" -H "X-API-Key: $KEY"

# Apply:
curl -X POST "$BASE/admin/rescore" -H "X-API-Key: $KEY"
```

### Analytics

```bash
curl "$BASE/analytics"                                    -H "X-API-Key: $KEY"
curl "$BASE/analytics/compliance"                         -H "X-API-Key: $KEY"
curl "$BASE/analytics/steps"                              -H "X-API-Key: $KEY"
curl "$BASE/analytics/operators/OPERATOR_HASH/trend"      -H "X-API-Key: $KEY"
curl "$BASE/analytics/recommendations/OPERATOR_HASH"      -H "X-API-Key: $KEY"
```

### Video quality check

```bash
# Quick check (included in upload response)
curl -X POST "$BASE/videos" -H "X-API-Key: $KEY" -F "file=@trainee.mp4" -F "task_id=$TASK" \
  | python -m json.tool | grep -A5 '"quality"'

# Detailed per-check report with Japanese operator guidance
curl "$BASE/videos/2/quality" -H "X-API-Key: $KEY"
```

### Score reports

```bash
curl "$BASE/score/1/report"     -H "X-API-Key: $KEY" -o report.html
curl "$BASE/score/1/report/pdf" -H "X-API-Key: $KEY" -o report.pdf
curl "$BASE/score/1/uncertainty" -H "X-API-Key: $KEY"
```

### Prometheus metrics

```bash
curl "$BASE/metrics"   # no auth required
```

### Database backup

```bash
curl -X POST "$BASE/admin/backup"   -H "X-API-Key: $KEY"
curl -X POST "$BASE/admin/optimize" -H "X-API-Key: $KEY"
curl       "$BASE/admin/db-stats"   -H "X-API-Key: $KEY"
```

---

## 6. Smoke Test

After deploying, run the automated smoke test to verify the full pipeline end-to-end:

```bash
# Against Docker container (with API key):
python scripts/smoke_test.py --url http://localhost:8000 --api-key your-secret-key-here

# Against local uvicorn (no auth):
python scripts/smoke_test.py --url http://localhost:8000
```

Expected output: `All checks passed (9/9)`. The script uploads synthetic videos, runs a
score job, and verifies `score_band`, `decision_basis`, admin rescore, and all key fields
are present in responses.

---

## 7. Configuration Reference

All variables are read from the process environment. Docker Compose forwards them from `.env`.
Restart the container after any change.

### Core

| Variable | Default | Description |
|---|---|---|
| `SOPILOT_API_KEY` | _(empty)_ | API key for `X-API-Key` auth. Empty = auth disabled (dev only). |
| `SOPILOT_DATA_DIR` | `/app/data` | Root path for the SQLite DB and raw video files. |
| `SOPILOT_PORT` | `8000` | Host port mapped to the container (docker-compose only). |
| `SOPILOT_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `SOPILOT_LOG_JSON` | `true` | `true` for structured JSON lines; `false` for plain text. |

### Scoring Pipeline

| Variable | Default | Description |
|---|---|---|
| `SOPILOT_DEFAULT_PASS_SCORE` | `60.0` | Minimum score for a `pass` decision (0–100). |
| `SOPILOT_DEFAULT_RETRAIN_SCORE` | `50.0` | Score below which decision is `retrain`. Must be ≤ pass score. |
| `SOPILOT_SCORE_WORKERS` | `1` | Concurrent scoring threads. |
| `SOPILOT_SCORE_JOB_MAX_RETRIES` | `2` | Automatic retries before marking a job `failed`. |

### Video Processing

| Variable | Default | Description |
|---|---|---|
| `SOPILOT_EMBEDDER_BACKEND` | `color-motion` | `color-motion` (CPU) or `vjepa2` (GPU). |
| `SOPILOT_ALLOW_EMBEDDER_FALLBACK` | `true` | Fall back to `color-motion` if `vjepa2` fails to init. |
| `SOPILOT_MAX_UPLOAD_MB` | `500` | Maximum video upload size in MB. |
| `SOPILOT_SAMPLE_FPS` | `4` | Frames per second sampled for embedding extraction. |
| `SOPILOT_CLIP_SECONDS` | `4` | Clip window length for temporal segmentation (seconds). |

### Security / Network

| Variable | Default | Description |
|---|---|---|
| `SOPILOT_CORS_ORIGINS` | `http://localhost:8000,...` | Comma-separated allowed CORS origins. |
| `SOPILOT_RATE_LIMIT_RPM` | `120` | Requests per minute per IP. `0` = disabled. |
| `SOPILOT_RATE_LIMIT_BURST` | `20` | Maximum burst above the RPM window. |
| `SOPILOT_WEBHOOK_URL` | _(empty)_ | HTTP endpoint to POST score-completion notifications. |

### V-JEPA2 (only when `SOPILOT_EMBEDDER_BACKEND=vjepa2`)

| Variable | Default | Description |
|---|---|---|
| `SOPILOT_VJEPA2_VARIANT` | `vjepa2_vit_large` | Model variant identifier. |
| `SOPILOT_VJEPA2_DEVICE` | `auto` | `auto`, `cuda`, or `cpu`. |
| `SOPILOT_VJEPA2_POOLING` | `mean_tokens` | Token pooling strategy. |
| `SOPILOT_VJEPA2_USE_AMP` | `true` | Enable automatic mixed precision (requires CUDA). |

---

## 8. Troubleshooting

### Container not starting

```bash
docker compose logs sopilot
```

Common causes:

- `SOPILOT_DEFAULT_RETRAIN_SCORE > SOPILOT_DEFAULT_PASS_SCORE` — settings validation
  rejects this at startup. Fix values in `.env`, then `docker compose up -d`.
- Port 8000 already bound — set `SOPILOT_PORT=8001` in `.env` and restart.

### Health check failing

```bash
curl -v http://localhost:8000/health
docker compose ps
```

`/health` bypasses API-key auth. A non-200 response means the app has crashed — check logs.
`/readiness` additionally verifies DB connectivity.

### Data persistence

```
./data/
  sopilot.db        # SQLite database (WAL mode)
  sopilot.db-wal    # WAL file — present during normal operation
  sopilot.db-shm    # Shared memory — present during normal operation
  raw/              # Uploaded video files
```

Never delete the WAL file while the container is running. Online backup:

```bash
curl -X POST "$BASE/admin/backup" -H "X-API-Key: $KEY"
```

### Upload rejected with 413

```dotenv
SOPILOT_MAX_UPLOAD_MB=1024
```

Restart: `docker compose up -d`.

### Switching to the V-JEPA2 backend

In `.env`:

```dotenv
SOPILOT_EMBEDDER_BACKEND=vjepa2
SOPILOT_VJEPA2_DEVICE=cuda
```

Also required in `docker-compose.yml`:

1. Increase memory limit from `2G` to at least `8G`.
2. Add NVIDIA GPU passthrough under `deploy.resources.reservations.devices`.
3. Rebuild: `docker compose up -d --build`.

### Rate limiting

Default: 120 requests/min with a burst of 20. For high-throughput batch pipelines:

```dotenv
SOPILOT_RATE_LIMIT_RPM=0
```
