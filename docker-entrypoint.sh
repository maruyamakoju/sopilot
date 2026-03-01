#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SOPilot Docker Entrypoint
#
# 1. On first boot (no DB file yet), run the seed script to create demo data.
# 2. Start uvicorn via exec so it becomes PID 1 and receives signals properly.
# =============================================================================

DATA_DIR="${SOPILOT_DATA_DIR:-/app/data}"
DB_FILE="${DATA_DIR}/sopilot.db"

# ---- First-boot seed -------------------------------------------------------
if [ ! -f "${DB_FILE}" ]; then
    echo "[entrypoint] Database not found at ${DB_FILE} -- running seed..."
    python -m sopilot.seed
    echo "[entrypoint] Seed complete."
else
    echo "[entrypoint] Database exists at ${DB_FILE} -- skipping seed."
fi

# ---- Start the application --------------------------------------------------
echo "[entrypoint] Starting SOPilot on port ${SOPILOT_PORT:-8000}..."
exec uvicorn sopilot.main:create_app \
    --factory \
    --host 0.0.0.0 \
    --port "${SOPILOT_PORT:-8000}" \
    --workers "${SOPILOT_UVICORN_WORKERS:-1}" \
    --log-level "${SOPILOT_UVICORN_LOG_LEVEL:-info}" \
    "$@"
