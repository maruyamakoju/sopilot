from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .db import Database
from .utils import now_tag, write_json

logger = logging.getLogger(__name__)


class AuditService:
    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db

    def get_audit_trail(self, limit: int = 100) -> list[dict]:
        return self.db.list_audit_trail(limit=limit)

    @staticmethod
    def _canonical_json(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def export_signed_audit_trail(self, *, limit: int = 500) -> dict:
        secret = self.settings.audit_signing_key.strip()
        if not secret:
            raise RuntimeError("audit signing key is not configured")

        generated_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        items = self.get_audit_trail(limit=limit)
        unsigned = {
            "generated_at": generated_at,
            "limit": int(limit),
            "item_count": int(len(items)),
            "items": items,
        }
        canonical = self._canonical_json(unsigned)
        payload_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        signature = hmac.new(
            secret.encode("utf-8"),
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        signed_payload = dict(unsigned)
        signed_payload["signature"] = {
            "algorithm": "hmac-sha256",
            "key_id": self.settings.audit_signing_key_id,
            "payload_sha256": payload_sha,
            "signature_hex": signature,
        }
        export_id = now_tag()
        out = self.settings.reports_dir / f"audit_export_{export_id}.json"
        write_json(out, signed_payload)
        return {
            "export_id": export_id,
            "generated_at": generated_at,
            "item_count": int(len(items)),
            "file_path": str(out),
            "signature": signed_payload["signature"],
        }

    def get_audit_export_path(self, export_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "", export_id.strip())
        if not safe:
            raise ValueError("invalid export_id")
        path = self.settings.reports_dir / f"audit_export_{safe}.json"
        if not path.exists():
            raise ValueError("audit export not found")
        return path
