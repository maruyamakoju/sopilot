# SOPilot

**On-premises SOP compliance scoring service — v1.0.0**

SOPilot automatically evaluates whether a trainee's recorded work video follows the correct
Standard Operating Procedure — comparing against a gold-standard reference video — and returns
a numeric score plus a pass/fail decision within seconds.

---

## At a Glance

| Property | Value |
|---|---|
| Version | v1.0.0 |
| Evaluation dataset | 3,507 scored video pairs (96-hour production run) |
| Accuracy | **99.40%** [99.14%, 99.63%] |
| F1 Score | **99.62%** |
| AUC-ROC | **99.69%** |
| Decision threshold | 60.0 (LOSO-validated) |
| Deployment | Docker Compose, single container, no GPU required |
| Test coverage | 895 automated tests |

---

## Quick Start

```bash
cp .env.example .env          # set SOPILOT_API_KEY
docker compose up -d          # build + start (~3 min first build)
python scripts/smoke_test.py  # verify: All checks passed (9/9)
```

See **[QUICKSTART.md](QUICKSTART.md)** for full step-by-step instructions, API usage examples,
and configuration reference.

---

## How It Works

```
Upload gold video  ──┐
                     ├──▶  POST /score  ──▶  0–100 score + pass/fail decision
Upload trainee video─┘
```

1. **Quality Gate**: Uploaded videos are checked for brightness, sharpness, stability,
   resolution, and duration before scoring. Low-quality videos receive actionable
   Japanese-language feedback via `GET /videos/{id}/quality`.
2. **Ingest**: Videos are sampled at 4 fps, split into 4-second clips, and embedded using
   the ColorMotion-v1 embedder (CPU, no GPU required).
3. **Align**: Dynamic Time Warping aligns the trainee clip sequence against the gold reference.
4. **Score**: Weighted penalties for missed steps, swapped steps, deviations, and overtime
   are combined into a 0–100 score with a bootstrap confidence interval.
5. **Decide**: `pass` / `needs_review` / `retrain` / `fail` based on thresholds and critical
   deviation flags. Each deviation carries a localized Japanese template comment.

---

## v1.0 New Features

- **Gold Builder**: `POST /gold` accepts `enforce_quality=true` to reject low-quality gold videos
  at upload time (HTTP 422 with per-axis quality breakdown). UI: "品質ゲート" checkbox +
  detailed quality wizard card after every gold upload.
- **SOP Versioning**: Each gold video receives a sequential `gold_version` number per task
  (v1, v2, …). Returned in all video list, detail, and ingest responses. UI: version badge
  in gold list + version prefix in video pane.
- **Evidence Clips**: Every deviation includes `gold_timecode` and `trainee_timecode`
  (`[start_sec, end_sec]`) pinpointing the exact moment in each video. UI: "ジャンプ" button
  seeks both video players to the deviation timestamp.

---

## Documentation

| Document | Description |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | Docker deployment, API walkthrough, configuration reference, troubleshooting |
| [TECHNICAL_SUMMARY.md](TECHNICAL_SUMMARY.md) | Architecture, accuracy metrics, decision logic, API surface, known limitations |
| [DEMO.md](DEMO.md) | 5-minute demo script for technical reviewers |

---

## API Overview

```bash
BASE="http://localhost:8000"
KEY="your-api-key"

# Upload reference and trainee videos
curl -X POST "$BASE/gold"   -H "X-API-Key: $KEY" -F "file=@gold.mp4"    -F "task_id=pilot_task"
curl -X POST "$BASE/videos" -H "X-API-Key: $KEY" -F "file=@trainee.mp4" -F "task_id=pilot_task"

# Check video quality (brightness, sharpness, stability, resolution, duration)
curl "$BASE/videos/2/quality" -H "X-API-Key: $KEY"

# Score
curl -X POST "$BASE/score" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"gold_video_id": 1, "trainee_video_id": 2}'

# Poll result (includes comment_ja, severity_ja per deviation)
curl "$BASE/score/1" -H "X-API-Key: $KEY"
```

Full OpenAPI spec: `http://localhost:8000/docs`

---

## Accuracy

Evaluated with leave-one-subject-out cross-validation on 3,507 production score jobs
(70 human-annotated labels) at decision threshold = 60.0.

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | **99.40%** | [99.14%, 99.63%] |
| F1 Score | **99.62%** | — |
| AUC-ROC | **99.69%** | — |
| Critical Miss Rate | **0.75%** | — |

Confusion matrix: TP=2766, FN=21, FP=0, TN=720.
Zero false positives after re-annotation of 4 mislabeled videos (annotation quality artifacts).

Evaluation artifacts: `artifacts/loso_eval_human_t60/`

---

## Resource Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 2 cores | 4 cores |
| RAM | 1 GB | 2 GB |
| Disk | 10 GB | 100 GB (for video storage) |
| GPU | Not required | — |
| Docker | ≥ 24 (Linux containers) | — |

---

## Project Structure

```
sopilot/              # Application package
  api/routes/         # FastAPI routers (videos, scoring, analytics, admin)
  core/               # Algorithms: DTW, scoring, ensemble, uncertainty
  services/           # Business logic: video ingest, scoring pipeline
  repositories/       # SQLite data layer
  ui/                 # Browser UI (single-page HTML/JS)
scripts/
  smoke_test.py       # End-to-end deployment verification (9 checks)
  run_loso_evaluation.py
tests/                # 895 automated tests
artifacts/
  loso_eval_human_t60/  # LOSO evaluation results (N=3507)
  fp_analysis/          # False positive root-cause analysis
data/                 # SQLite DB + raw video files (bind-mounted in Docker)
```
