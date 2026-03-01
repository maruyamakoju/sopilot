# SOPilot MVP implementation notes

## Product boundary

This MVP intentionally excludes broad surveillance features and focuses on:

- SOP execution quality
- training feedback loops
- audit evidence generation

## PoC-first constraints

- Primary task is fixed (`SOPILOT_PRIMARY_TASK_ID`) during PoC.
- Score policy is task-profile driven (weights, thresholds, severity mapping).
- UI flow is operator-first: select videos from list, not by manual IDs.

## Core model strategy

1. Start from V-JEPA2 pretrained encoder embeddings (`torch.hub`).
2. Align SOP trajectories with step boundaries + DTW.
3. Accumulate customer video data on-prem for domain adaptation.

## Upgrade path

- Add nightly domain-adaptive training jobs over customer video archives.
- Add action-conditioned input streams (PLC/POS/event logs).
- Move search index from brute-force cosine to FAISS/Qdrant when volume grows.

## Operational notes

- Scoring is asynchronous (`queued/running/completed/failed`) via in-process worker queue.
- App startup requeues unfinished jobs (`queued` and `running`) from SQLite.
- UI is intentionally minimal and focused on side-by-side review and deviation timecodes.
- Score review can be persisted (`pass/retrain/fail/needs_review`) with notes.
- PoC evaluation metrics are available via `scripts/evaluate_poc.py`.
