from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path


def _canonical(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SOPilot signed audit export (HMAC-SHA256)")
    parser.add_argument("--file", required=True, help="Path to audit_export_*.json")
    parser.add_argument("--key", required=True, help="Signing secret")
    args = parser.parse_args()

    path = Path(args.file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        raise ValueError("signature block is missing")

    unsigned = dict(payload)
    unsigned.pop("signature", None)
    canonical = _canonical(unsigned)
    expected_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    expected_sig = hmac.new(
        args.key.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    payload_sha = str(signature.get("payload_sha256", ""))
    signature_hex = str(signature.get("signature_hex", ""))
    ok = payload_sha == expected_sha and hmac.compare_digest(signature_hex, expected_sig)
    result = {
        "file": str(path),
        "algorithm": signature.get("algorithm"),
        "key_id": signature.get("key_id"),
        "payload_sha256_match": payload_sha == expected_sha,
        "signature_match": hmac.compare_digest(signature_hex, expected_sig),
        "verified": ok,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
