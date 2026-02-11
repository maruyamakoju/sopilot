# Offline Deployment Notes

## Goal

Run SOPilot in a network-isolated environment.

## Required local assets

- V-JEPA2 repository mirror under `./models/vjepa2`
- Model checkpoint under `./models/checkpoints/vitl.pt` (or another path)

## Recommended environment

Set these variables (already included in `docker-compose.yml` defaults):

- `SOPILOT_EMBEDDER_BACKEND=auto`
- `SOPILOT_VJEPA2_SOURCE=local`
- `SOPILOT_VJEPA2_LOCAL_REPO=/models/vjepa2`
- `SOPILOT_VJEPA2_LOCAL_CHECKPOINT=/models/checkpoints/vitl.pt`
- `SOPILOT_VJEPA2_PRETRAINED=false`
- `SOPILOT_AUTH_REQUIRED=true`
- `SOPILOT_API_TOKEN=<secret>` (推奨)
- `SOPILOT_API_TOKEN_ROLE=admin`
- `SOPILOT_AUDIT_SIGNING_KEY=<secret>` (監査署名を使う場合)
- `SOPILOT_PRIVACY_MASK_ENABLED=true` (必要時)

`PRETRAINED=false` avoids URL weight fetch and enforces checkpoint loading from local disk.

## Start

```powershell
docker compose up --build
```

## Validate

1. Open `http://127.0.0.1:8000/health` and verify status is ok.
2. Confirm worker logs show queues `sopilot_ingest`, `sopilot_score`, `sopilot_training`.
3. Open `http://127.0.0.1:8000/ui` and run upload -> score flow.
4. Export `report.pdf` from score detail.

## Troubleshooting

- If V-JEPA2 load fails, `auto` backend falls back to heuristic embeddings.
- If you need strict fail-fast behavior, set `SOPILOT_EMBEDDER_BACKEND=vjepa2`.
- API and workers must share the same mounted `data` directory.
