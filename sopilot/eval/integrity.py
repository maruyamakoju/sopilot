from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

_DEFAULT_HASH_KEY = "artifact_hash_sha256"
_DEFAULT_METHOD_KEY = "artifact_hash_method"
_DEFAULT_METHOD = "sha256(canonical_json,exclude=artifact_hash_sha256|artifact_hash_method)"


def canonicalize_payload(
    payload: Any,
    *,
    exclude_keys: set[str] | None = None,
) -> Any:
    excluded = set(exclude_keys or set())
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for key in sorted(payload.keys(), key=lambda item: str(item)):
            if key in excluded:
                continue
            out[str(key)] = canonicalize_payload(payload[key], exclude_keys=excluded)
        return out
    if isinstance(payload, list):
        return [canonicalize_payload(item, exclude_keys=excluded) for item in payload]
    return payload


def canonical_json_bytes(
    payload: Any,
    *,
    exclude_keys: set[str] | None = None,
) -> bytes:
    canonical = canonicalize_payload(payload, exclude_keys=exclude_keys)
    return json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def payload_sha256(
    payload: Any,
    *,
    exclude_keys: set[str] | None = None,
) -> str:
    digest = hashlib.sha256(canonical_json_bytes(payload, exclude_keys=exclude_keys)).hexdigest()
    return str(digest)


def attach_payload_hash(
    payload: dict[str, Any],
    *,
    hash_key: str = _DEFAULT_HASH_KEY,
    method_key: str = _DEFAULT_METHOD_KEY,
    method: str = _DEFAULT_METHOD,
    exclude_extra_keys: set[str] | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    excluded = {str(hash_key), str(method_key)}
    if exclude_extra_keys:
        excluded.update(str(key) for key in exclude_extra_keys)
    out[method_key] = str(method)
    out[hash_key] = payload_sha256(out, exclude_keys=excluded)
    return out


def verify_payload_hash(
    payload: dict[str, Any],
    *,
    hash_key: str = _DEFAULT_HASH_KEY,
    method_key: str = _DEFAULT_METHOD_KEY,
    exclude_extra_keys: set[str] | None = None,
) -> bool:
    expected = payload.get(hash_key)
    if not expected:
        return False
    excluded = {str(hash_key), str(method_key)}
    if exclude_extra_keys:
        excluded.update(str(key) for key in exclude_extra_keys)
    actual = payload_sha256(payload, exclude_keys=excluded)
    return str(expected) == str(actual)
