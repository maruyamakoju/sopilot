#!/usr/bin/env bash
# =============================================================================
# SOPilot -- Production health check script
# =============================================================================
# Usage:
#   ./scripts/health_check.sh
#
# Environment variables (all optional):
#   SOPILOT_URL   Base URL for the SOPilot API  (http://localhost:8000)
#   DB_PATH       Path to the SQLite database    (./data/sopilot.db)
#   MIN_DISK_GB   Minimum free disk space in GB  (1)
#
# Exit codes:
#   0  — all checks passed (healthy)
#   1  — one or more checks failed (unhealthy)
# =============================================================================

set -uo pipefail
# NOTE: 'set -e' is intentionally omitted so that individual checks can fail
# without aborting the full report.  We collect results and exit at the end.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOPILOT_URL="${SOPILOT_URL:-http://localhost:8000}"
DB_PATH="${DB_PATH:-./data/sopilot.db}"
MIN_DISK_GB="${MIN_DISK_GB:-1}"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
OVERALL_STATUS=0   # 0 = healthy, 1 = unhealthy
declare -a REPORT=()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ts() {
    date '+%Y-%m-%d %H:%M:%S'
}

pass() {
    REPORT+=("  [PASS] $*")
}

fail() {
    REPORT+=("  [FAIL] $*")
    OVERALL_STATUS=1
}

separator() {
    REPORT+=("--------------------------------------------------------------")
}

# ---------------------------------------------------------------------------
# Check 1: /health endpoint returns HTTP 200
# ---------------------------------------------------------------------------
HEALTH_URL="${SOPILOT_URL}/health"
REPORT+=("")
REPORT+=("Check 1: HTTP /health endpoint (${HEALTH_URL})")

if ! command -v curl &>/dev/null; then
    fail "curl is not installed — cannot perform HTTP health check."
else
    HTTP_STATUS=""
    if HTTP_STATUS="$(curl -sf -o /dev/null -w '%{http_code}' \
            --connect-timeout 5 --max-time 10 "${HEALTH_URL}" 2>/dev/null)"; then
        if [[ "${HTTP_STATUS}" == "200" ]]; then
            pass "GET ${HEALTH_URL} → HTTP ${HTTP_STATUS}"
        else
            fail "GET ${HEALTH_URL} → HTTP ${HTTP_STATUS} (expected 200)"
        fi
    else
        fail "GET ${HEALTH_URL} → connection refused or timeout"
    fi
fi

# ---------------------------------------------------------------------------
# Check 2: Disk space — at least MIN_DISK_GB GB free
# ---------------------------------------------------------------------------
REPORT+=("")
REPORT+=("Check 2: Disk space (minimum ${MIN_DISK_GB} GB free)")

if ! command -v df &>/dev/null; then
    fail "df is not installed — cannot check disk space."
else
    # Get free space in kilobytes for the filesystem containing DB_PATH or '.'
    CHECK_PATH="."
    if [[ -e "${DB_PATH}" ]]; then
        CHECK_PATH="${DB_PATH}"
    fi

    FREE_KB="$(df -k "${CHECK_PATH}" | awk 'NR==2 {print $4}')"
    if [[ -z "${FREE_KB}" ]]; then
        fail "Could not determine free disk space for ${CHECK_PATH}."
    else
        FREE_GB="$(awk -v kb="${FREE_KB}" 'BEGIN { printf "%.2f", kb/1048576 }')"
        if awk -v free="${FREE_GB}" -v min="${MIN_DISK_GB}" 'BEGIN { exit !(free >= min) }'; then
            pass "Free disk space: ${FREE_GB} GB (>= ${MIN_DISK_GB} GB required)"
        else
            fail "Free disk space: ${FREE_GB} GB — below minimum ${MIN_DISK_GB} GB"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Check 3: Database file exists and is readable
# ---------------------------------------------------------------------------
REPORT+=("")
REPORT+=("Check 3: Database file (${DB_PATH})")

if [[ ! -e "${DB_PATH}" ]]; then
    fail "Database file not found: ${DB_PATH}"
elif [[ ! -f "${DB_PATH}" ]]; then
    fail "Database path exists but is not a regular file: ${DB_PATH}"
elif [[ ! -r "${DB_PATH}" ]]; then
    fail "Database file exists but is not readable: ${DB_PATH}"
else
    DB_SIZE="$(du -sh "${DB_PATH}" 2>/dev/null | cut -f1)"
    pass "Database file found and readable: ${DB_PATH} (${DB_SIZE})"

    # Optional: quick integrity check if sqlite3 is available
    if command -v sqlite3 &>/dev/null; then
        INTEGRITY="$(sqlite3 "${DB_PATH}" 'PRAGMA quick_check;' 2>/dev/null || true)"
        if [[ "${INTEGRITY}" == "ok" ]]; then
            pass "Database quick_check: ok"
        else
            fail "Database quick_check returned: ${INTEGRITY:-<error>}"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------
separator
printf '\n'
printf '=== SOPilot Health Check Report (%s) ===\n' "$(ts)"
for LINE in "${REPORT[@]}"; do
    printf '%s\n' "${LINE}"
done
printf '\n'

if [[ "${OVERALL_STATUS}" -eq 0 ]]; then
    printf '[%s] Overall status: HEALTHY\n' "$(ts)"
else
    printf '[%s] Overall status: UNHEALTHY\n' "$(ts)" >&2
fi

exit "${OVERALL_STATUS}"
