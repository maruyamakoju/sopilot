#!/usr/bin/env bash
# =============================================================================
# SOPilot load test runner
#
# Starts the server, waits for /health, runs a 60-second Locust test,
# saves an HTML report, then shuts down the server.
#
# Usage:
#   bash scripts/run_load_test.sh [--host <url>] [--users <n>] [--time <Xs>]
# =============================================================================

set -euo pipefail

HOST="${SOPILOT_HOST:-http://localhost:8000}"
USERS="${LOAD_USERS:-50}"
SPAWN_RATE="${LOAD_SPAWN_RATE:-5}"
RUN_TIME="${LOAD_RUN_TIME:-60s}"
REPORT_FILE="${LOAD_REPORT:-load-report.html}"

SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ------------------------------------------------------------------
# Start the server
# ------------------------------------------------------------------
echo "Starting SOPilot server..."
SOPILOT_LOG_JSON=false uvicorn sopilot.main:create_app \
    --factory --host 0.0.0.0 --port 8000 \
    --no-access-log &
SERVER_PID=$!

# ------------------------------------------------------------------
# Wait for /health to return 200 (max 30 seconds)
# ------------------------------------------------------------------
echo "Waiting for server to be ready..."
for i in $(seq 1 30); do
    if curl -sf "${HOST}/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
    if [ "$i" -eq 30 ]; then
        echo "Server did not start within 30 seconds" >&2
        exit 1
    fi
done

# ------------------------------------------------------------------
# Run Locust
# ------------------------------------------------------------------
echo "Running load test (users=${USERS}, spawn-rate=${SPAWN_RATE}, time=${RUN_TIME})..."
locust \
    -f tests/load/locustfile.py \
    --host "${HOST}" \
    --users "${USERS}" \
    --spawn-rate "${SPAWN_RATE}" \
    --run-time "${RUN_TIME}" \
    --headless \
    --html "${REPORT_FILE}"

echo "Load test complete. Report saved to: ${REPORT_FILE}"
