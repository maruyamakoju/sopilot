from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sopilot.eval.integrity import verify_payload_hash


def _extra_exclusions(payload: dict[str, Any]) -> set[str]:
    version = str(payload.get("version") or "")
    exclusions: set[str] = set()
    if "policy" in version and "policy_id" in payload:
        exclusions.add("policy_id")
    method = str(payload.get("artifact_hash_method") or "")
    if "policy_id" in method:
        exclusions.add("policy_id")
    return exclusions


def _verify_one(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ok = verify_payload_hash(payload, exclude_extra_keys=_extra_exclusions(payload))
    return {
        "path": str(path),
        "has_hash": bool(payload.get("artifact_hash_sha256")),
        "verified": bool(ok),
        "artifact_hash_sha256": payload.get("artifact_hash_sha256"),
        "artifact_hash_method": payload.get("artifact_hash_method"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify artifact hash fields on JSON files.")
    parser.add_argument("paths", nargs="+", help="JSON files")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = [_verify_one(Path(p).resolve()) for p in args.paths]
    payload = {
        "files": results,
        "all_verified": all(bool(row.get("verified")) for row in results),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    if not payload["all_verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
