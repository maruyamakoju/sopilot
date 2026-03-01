# SOPilot 90-day execution roadmap

## Day 0-14: Baseline that can be demoed

- Settle one pilot task (`task_id`) with clear step boundaries.
- Capture at least 20 gold videos and 50 trainee videos.
- Run MVP API in one on-prem node and validate ingest-to-score latency.
- Define pass/fail policy and score thresholds with QA stakeholders.

## Day 15-45: Pilot in one site

- Integrate score workflow into trainer/QA review routine.
- Track KPI delta:
  - review time per video
  - missed critical deviation rate
  - time-to-qualification for new operators
- Tune scoring weights per task (`w_miss`, `w_swap`, `w_dev`, `w_time`).
- Add role-based access and export reports (JSON/PDF) if required by audit.

## Day 46-90: Make it production-ready

- Replace brute-force search with FAISS/Qdrant for larger datasets.
- Add nightly model update pipeline and model registry (`model_v{n}`).
- Add model rollout guardrails:
  - shadow scoring before promotion
  - score drift alerts
  - one-click rollback
- Prepare packaged on-prem deployment:
  - offline installation bundle
  - backup and restore scripts
  - security hardening checklist

## Definition of success

- Minimum 50% reduction in evaluator review time.
- Minimum 30% reduction in serious SOP deviations after training loop.
- Stable rescoring variance after model updates on the same input set.

