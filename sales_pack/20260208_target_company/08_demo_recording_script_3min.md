# SOPilot 3-Min Demo Script (Sales)

## Goal
- Show value in 3 minutes: `deviation jump -> PDF -> audit trail`.

## Preconditions
- System is running: `docker compose up -d`
- Token auth is enabled (recommended): `SOPILOT_API_TOKEN`
- Demo videos exist in `demo_videos/maintenance_filter_swap/`

## 0:00-0:20 Open + Trust
1. Open `http://127.0.0.1:8000/ui`.
2. Enter token in the auth field.
3. Say: "On-prem, authenticated, and auditable."

## 0:20-1:10 Ingest
1. Upload Gold (`gold.mp4`) with `task_id=maintenance_filter_swap`.
2. Upload one failure video (`missing.mp4` or `deviation.mp4`).
3. Wait for ingest completion in UI status.

## 1:10-2:00 Score + Explain
1. Click `Create Score Job`.
2. Click `Poll Score Job` until completed.
3. Click deviation markers to jump playback.
4. Say: "This is evidence-linked deviation, not generic CV detection."

## 2:00-2:30 Report
1. Click `Download PDF`.
2. Open downloaded report and point to:
   - `Score Job`
   - `Requested By`
   - `Embedding Model`

## 2:30-3:00 Audit
1. Click `Refresh Audit`.
2. Show `requested_by` and `job_id` rows.
3. Close with PoC Done criteria:
   - 30-min video stable processing
   - deviation jump
   - PDF output
   - audit trail
