# SOPilot On-Prem Runbook (PoC)

## Runtime constraint

- Queue is currently in-process.
- Use:
  - `uvicorn sopilot.main:create_app --factory --host 0.0.0.0 --port 8000 --workers 1`
- Do not scale to multi-worker until external queue is introduced.

## Pre-flight

1. Set PoC env vars (task fixed, one-worker).
2. Optionally preload model cache:
   - `python scripts/preload_vjepa2.py --variant vjepa2_vit_large --pretrained true --crop-size 256`
3. Start app and open `/`.

## Lock profile

Run once before data collection:

```cmd
python scripts\update_task_profile.py ^
  --db-path data\sopilot.db ^
  --task-id filter_change ^
  --create-if-missing ^
  --pass-score 90 ^
  --retrain-score 80
```

## Data collection operation

- Capture guide: `docs/poc_capture_guide.md`
- Target: Gold 20 / Trainee 50
- Optional internet seed pull (Wikimedia Commons, for pipeline smoke test):
  - `python scripts/fetch_commons_videos.py --output-dir poc_videos/candidates --max-files 20 --manifest data/commons_seed_manifest.json`
  - `python scripts/fetch_commons_videos.py --layout split --gold-count 5 --output-dir poc_videos --max-files 25 --manifest data/commons_seed_manifest.json`
  - Keep `data/commons_seed_manifest.json` for attribution/license trace.
- Bulk upload (recommended):
  - `python scripts/bulk_upload_folder.py --base-dir poc_videos --task-id %SOPILOT_PRIMARY_TASK_ID% --recursive --output data/upload_results.json`
- Create intentional bad examples for critical-deviation check (recommended in PoC):
  - `python scripts/generate_bad_examples.py --input-dir poc_videos_tesda/trainee --output-dir poc_videos_tesda/trainee_bad --max-source 6`
- Batch score:
  - `python scripts/score_batch.py --only-ready --wait --output data/score_batch_results.json`
- If API process/port operation is constrained, run local one-shot:
  - `python scripts/poc_local_pipeline.py --base-dir poc_videos_tesda --recursive --task-id filter_change --site-id site-a --embedder-backend color-motion --data-dir data_tesda_cm --labels-output data_tesda_cm/labels_template.json --eval-output data_tesda_cm/eval_report.json --output data_tesda_cm/local_pipeline_summary.json`
- For repeated operation, use incremental pipeline:
  - `python scripts/poc_incremental_pipeline.py --base-dir poc_videos_tesda --recursive --task-id filter_change --task-name "Filter Change" --site-id site-a --embedder-backend color-motion --data-dir data_tesda_steady --gold-id 2 --score-scope unscored --labels-output data_tesda_steady/labels_template.json --eval-output data_tesda_steady/eval_report.json --output data_tesda_steady/incremental_summary.json`
- Progress:
  - `python scripts/poc_status.py --db-path data/sopilot.db --task-id filter_change`

## Evaluation operation

1. Build labels template:
   - `python scripts/generate_labels_template.py --db-path data/sopilot.db --task-id filter_change --limit 20 --output data/labels_template.json`
2. Fill `critical_expected` manually.
   - PoC default labeling rule for this repo:
     - freeze-only variants: `critical_expected=true`
     - cut-tail / skip-start variants: `critical_expected=false` (quality/efficiency lane)
   - Optional prefill helper:
     - `python scripts/prefill_critical_labels.py --summary data_tesda_cm_v2/local_pipeline_summary.json --labels data_tesda_cm_v2/labels_template.json --critical-pattern _bad_freeze`
3. Evaluate and gate:
   - `python scripts/evaluate_poc.py --db-path data/sopilot.db --task-id filter_change --labels data/labels_template.json --max-critical-miss-rate 0.10 --max-critical-fpr 0.30 --max-rescore-jitter 5.0 --max-dtw-p90 0.60 --fail-on-gate`
   - research profile (partner-facing evidence):
     - `python scripts/evaluate_poc.py --db-path data/sopilot.db --task-id filter_change --labels data/labels_template.json --gate-profile research_v1 --fail-on-gate`
   - research_v2 threshold sweep:
     - `python scripts/evaluate_poc.py --db-path data/sopilot.db --task-id filter_change --labels data/labels_template.json --gate-profile research_v2 --critical-scoring-mode continuous_v1 --critical-threshold 0.50 --critical-sweep-auto`
   - research_v2 guarded detector:
     - `python scripts/evaluate_poc.py --db-path data/sopilot.db --task-id filter_change --labels data/labels_template.json --gate-profile research_v2 --critical-scoring-mode guarded_binary_v1 --fail-on-gate`
   - research_v2 guarded detector v2 (extra Stage-B):
     - `python scripts/evaluate_poc.py --db-path data/sopilot.db --task-id filter_change --labels data/labels_template.json --gate-profile research_v2 --critical-scoring-mode guarded_binary_v2 --fail-on-gate`
   - Dev/Test/Challenge candidate evaluation:
     - `python scripts/evaluate_split_profiles.py --db-path data/sopilot.db --labels data/labels_template.json --task-id filter_change --gate-profile research_v2 --candidate-mode guarded_binary_v1 --candidate-mode guarded_binary_v2 --output-dir artifacts/split_eval --write-split-labels`
   - leak-resilient split manifest (group-stratified):
     - `python scripts/evaluate_split_profiles.py --db-path data/sopilot.db --labels data/labels_template.json --task-id filter_change --gate-profile research_v2 --split-strategy group_trainee --candidate-mode guarded_binary_v2 --output-dir artifacts/split_eval_group --write-split-labels`
   - fit policy on Dev only:
     - `python scripts/fit_critical_policy.py --db-path data/sopilot.db --labels data/labels_template.json --split-manifest artifacts/split_eval_group/split_manifest.json --task-id filter_change --gate-profile research_v2 --output-policy artifacts/policy/critical_policy_guarded_v2_devfit.json --output-report artifacts/policy/critical_policy_guarded_v2_devfit_report.json`
   - evaluate fixed policy on manifest (Dev/Test/Challenge):
     - `python scripts/evaluate_split_profiles.py --db-path data/sopilot.db --labels data/labels_template.json --task-id filter_change --gate-profile research_v2 --split-manifest artifacts/split_eval_group/split_manifest.json --critical-policy artifacts/policy/critical_policy_guarded_v2_devfit.json --candidate-mode guarded_binary_v2 --output-dir artifacts/split_eval_policy_locked --write-split-labels`
   - ops profile (labelなし運用監視):
     - `python scripts/evaluate_poc.py --db-path data/sopilot.db --task-id filter_change --gate-profile ops_v1`
   - shadow compare (baseline vs candidate):
     - `python scripts/evaluate_shadow_candidate.py --db-path data/sopilot.db --task-id filter_change --labels data/labels_template.json --baseline-mode guarded_binary_v1 --candidate-policy artifacts/policy/critical_policy_guarded_v2_devfit.json --output artifacts/policy/shadow_guarded_v1_vs_devfit_v2.json`
   - build evidence dossier:
     - `python scripts/build_evidence_dossier.py --split-report artifacts/split_eval_policy_locked/split_evaluation_report.json --policy artifacts/policy/critical_policy_guarded_v2_devfit.json --gate-report data/gate_report_research_v2_policy_devfit.json --shadow-report artifacts/policy/shadow_guarded_v1_vs_devfit_v2.json --fp-breakdown artifacts/errors/fp_breakdown.json --out-dir artifacts/evidence_dossier`
   - note: `research_v1` is lock-protected; threshold overrides require `--allow-profile-overrides`.
4. Extract FP/FN error packet for root-cause loop:
   - `python scripts/extract_error_cases.py --db-path data/sopilot.db --labels data/labels_template.json --task-id filter_change --critical-scoring-mode legacy_binary --critical-threshold 0.5 --output-dir artifacts/errors`
5. If `rescore_jitter` is missing, run repeated scoring on same pairs:
   - `python scripts/rescore_existing_pairs.py --data-dir data_tesda_cm_v2 --task-id filter_change --task-name Filter_Change --gold-id 2 --backend color-motion --repeat 1 --output data_tesda_cm_v2/rescore_results.json`
6. One-command orchestrator:
   - `python scripts/run_poc_critical_gate_flow.py --task-id filter_change --task-name "Filter Change" --base-dir poc_videos_tesda --trainee-dir poc_videos_tesda/trainee --trainee-bad-dir poc_videos_tesda/trainee_bad --data-dir data_tesda_cm_v2 --site-id site-a --gold-id 2 --backend color-motion --reset-data-dir`
7. 96h unattended autopilot:
   - `python scripts/poc_autopilot_runner.py --task-id filter_change --task-name "Filter Change" --base-dir poc_videos_tesda --trainee-dir poc_videos_tesda/trainee --trainee-bad-dir poc_videos_tesda/trainee_bad --data-dir data_tesda_96h --site-id site-a --backend color-motion --gold-id 2 --duration-hours 96 --interval-minutes 60 --bad-example-every 6 --rescore-every 6 --rescore-repeat 1 --backup-every 24 --critical-pattern _bad_freeze --step-timeout-minutes 30`
   - for research-grade gate in unattended runs, add: `--gate-profile research_v1`
8. Detached start/stop (recommended for overnight / travel):
   - start:
     - `python scripts/start_autopilot_detached.py --task-id filter_change --task-name "Filter Change" --base-dir poc_videos_tesda --trainee-dir poc_videos_tesda/trainee --trainee-bad-dir poc_videos_tesda/trainee_bad --data-dir data_tesda_96h --site-id site-a --backend color-motion --gold-id 2 --duration-hours 96 --interval-minutes 60 --bad-example-every 6 --rescore-every 6 --rescore-repeat 1 --backup-every 24 --critical-pattern _bad_freeze`
     - add `--gate-profile research_v1` when collecting partner-facing evidence
   - stop:
     - `python scripts/stop_autopilot.py --data-dir data_tesda_96h --require-run-id --run-id <RUN_ID> --grace-seconds 10`
     - legacy (still supported): `python scripts/stop_autopilot.py --pid-file data_tesda_96h/autopilot/runner.json --force`
   - status:
     - `python scripts/autopilot_status.py --data-dir data_tesda_96h`
     - stale-step detection: `python scripts/autopilot_status.py --data-dir data_tesda_96h --stale-seconds 1800`

## Daily command

```cmd
scripts\poc_daily_eval.cmd data\labels_template.json
```

## Backup and restore

- Backup:
  - `python scripts/backup_onprem.py --data-dir data --out-dir backups`
- Restore:
  - `python scripts/restore_onprem.py --backup backups/<file>.zip --data-dir data --force`

## Troubleshooting checklist

- `task_id` mismatch rejected:
  - confirm `SOPILOT_PRIMARY_TASK_ID` and upload form `task_id`.
- model cache/download issues:
  - preload with `scripts/preload_vjepa2.py`.
- scoring stuck:
  - check logs for queue enqueue/start/fail transitions.
- disk pressure:
  - inspect `data/raw` size and rotate backups.
