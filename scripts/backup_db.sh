#!/usr/bin/env bash
# =============================================================================
# SOPilot -- SQLite WAL database backup script
# =============================================================================
# Usage:
#   ./scripts/backup_db.sh
#
# Environment variables (all optional — sensible defaults shown):
#   BACKUP_DIR   Directory where timestamped backups are written  (./backups)
#   DB_PATH      Path to the live SOPilot SQLite database         (./data/sopilot.db)
#   KEEP_DAYS    Number of backups to retain (by count, newest N) (30)
#
# The script uses `sqlite3 .backup` which is safe for WAL-mode databases:
# it acquires a shared lock, checkpoints the WAL, and produces a clean,
# consistent snapshot without interrupting running transactions.
#
# Exit codes:
#   0  — backup completed successfully
#   1  — backup failed (reason printed to stderr)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKUP_DIR="${BACKUP_DIR:-./backups}"
DB_PATH="${DB_PATH:-./data/sopilot.db}"
KEEP_BACKUPS="${KEEP_BACKUPS:-30}"   # keep the N most recent backup files

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

err() {
    printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if ! command -v sqlite3 &>/dev/null; then
    err "sqlite3 is not installed or not on PATH. Aborting."
    exit 1
fi

if [[ ! -f "${DB_PATH}" ]]; then
    err "Database file not found: ${DB_PATH}"
    exit 1
fi

if [[ ! -r "${DB_PATH}" ]]; then
    err "Database file is not readable: ${DB_PATH}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Prepare backup directory
# ---------------------------------------------------------------------------
if [[ ! -d "${BACKUP_DIR}" ]]; then
    log "Creating backup directory: ${BACKUP_DIR}"
    mkdir -p "${BACKUP_DIR}"
fi

# ---------------------------------------------------------------------------
# Perform backup using sqlite3 .backup (WAL-safe hot backup)
# ---------------------------------------------------------------------------
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
BACKUP_FILE="${BACKUP_DIR}/sopilot_${TIMESTAMP}.db"

log "Starting backup: ${DB_PATH} → ${BACKUP_FILE}"

if sqlite3 "${DB_PATH}" ".backup '${BACKUP_FILE}'"; then
    BACKUP_SIZE="$(du -sh "${BACKUP_FILE}" | cut -f1)"
    log "Backup complete: ${BACKUP_FILE} (${BACKUP_SIZE})"
else
    err "sqlite3 .backup command failed. Removing incomplete file (if any)."
    rm -f "${BACKUP_FILE}"
    exit 1
fi

# Sanity check: the backup file must be non-empty
if [[ ! -s "${BACKUP_FILE}" ]]; then
    err "Backup file is empty — this should not happen. Removing."
    rm -f "${BACKUP_FILE}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Prune old backups — keep only the N most recent files
# ---------------------------------------------------------------------------
BACKUP_COUNT="$(find "${BACKUP_DIR}" -maxdepth 1 -name 'sopilot_*.db' | wc -l)"
log "Current backup count: ${BACKUP_COUNT} (retention limit: ${KEEP_BACKUPS})"

if (( BACKUP_COUNT > KEEP_BACKUPS )); then
    DELETE_COUNT=$(( BACKUP_COUNT - KEEP_BACKUPS ))
    log "Removing ${DELETE_COUNT} old backup(s)..."

    # List all backups sorted oldest-first, take the ones to delete
    find "${BACKUP_DIR}" -maxdepth 1 -name 'sopilot_*.db' \
        | sort \
        | head -n "${DELETE_COUNT}" \
        | while IFS= read -r OLD_FILE; do
            log "  Removing: ${OLD_FILE}"
            rm -f "${OLD_FILE}"
        done
fi

FINAL_COUNT="$(find "${BACKUP_DIR}" -maxdepth 1 -name 'sopilot_*.db' | wc -l)"
log "Backup rotation complete. Stored backups: ${FINAL_COUNT}"
log "Done."
exit 0
