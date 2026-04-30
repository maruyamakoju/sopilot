from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

try:
    from artifacts.submission_utils import (
        latest_subdir,
        read_json_dict,
        read_sha256,
        read_text,
        resolve_path,
        sha256_file,
        zip_entries,
    )
except ImportError:  # pragma: no cover
    from submission_utils import (
        latest_subdir,
        read_json_dict,
        read_sha256,
        read_text,
        resolve_path,
        sha256_file,
        zip_entries,
    )

def _path_from_arg(workspace: Path, value: str | None) -> Path | None:
    return resolve_path(workspace, value)


def _read_sha(path: Path) -> str | None:
    return read_sha256(path)


def _contents_entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for line in read_text(path).splitlines():
        value = line.strip()
        if value:
            out.append(value)
    return sorted(out)


def _latest_snapshot_manifest(observations_root: Path) -> Path | None:
    latest = latest_subdir(observations_root)
    if latest is None:
        return None
    manifest = latest / "snapshot_manifest.json"
    if manifest.exists():
        return manifest
    return None


def _path_like_tokens_from_markdown(text: str) -> list[str]:
    tokens = re.findall(r"`([^`]+)`", text)
    out: list[str] = []
    for token in tokens:
        value = token.strip()
        if not value:
            continue
        # Skip inline formulas/commands; only keep likely file paths.
        if re.search(r"\s", value):
            continue
        if "://" in value:
            continue
        if value.startswith("--"):
            continue
        if "/" in value or "\\" in value:
            out.append(value)
            continue
        if re.search(r"\.(json|md|txt|zip|sha256|db|jsonl)$", value, re.IGNORECASE):
            out.append(value)
    return sorted(dict.fromkeys(out))


def _record(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify submission bundle consistency.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument(
        "--official-zip",
        default="artifacts/official_run_artifacts_trip96h_official_20260212.zip",
    )
    parser.add_argument(
        "--official-sha",
        default="artifacts/official_run_artifacts_trip96h_official_20260212.sha256",
    )
    parser.add_argument(
        "--official-contents",
        default="artifacts/official_run_artifacts_trip96h_official_20260212.contents.txt",
    )
    parser.add_argument("--env-freeze", default="artifacts/env_pip_freeze.txt")
    parser.add_argument("--submission-packet", default="artifacts/SUBMISSION_PACKET_20260216.md")
    parser.add_argument("--public-packet", default="artifacts/SUBMISSION_PACKET_PUBLIC_20260216.md")
    parser.add_argument(
        "--final-summary",
        default="data_trip_96h_official_20260212/autopilot/final_summary.json",
    )
    parser.add_argument("--snapshot-manifest", default=None)
    parser.add_argument("--observations-root", default="artifacts/observations")
    args = parser.parse_args()

    ws = Path(args.workspace).resolve()
    official_zip = _path_from_arg(ws, args.official_zip)
    official_sha = _path_from_arg(ws, args.official_sha)
    official_contents = _path_from_arg(ws, args.official_contents)
    env_freeze = _path_from_arg(ws, args.env_freeze)
    packet = _path_from_arg(ws, args.submission_packet)
    public_packet = _path_from_arg(ws, args.public_packet)
    final_summary = _path_from_arg(ws, args.final_summary)
    snapshot_manifest = _path_from_arg(ws, args.snapshot_manifest)
    observations_root = _path_from_arg(ws, args.observations_root)

    checks: list[dict[str, Any]] = []

    for name, path in [
        ("official_zip_exists", official_zip),
        ("official_sha_exists", official_sha),
        ("official_contents_exists", official_contents),
        ("env_freeze_exists", env_freeze),
        ("submission_packet_exists", packet),
        ("public_packet_exists", public_packet),
        ("final_summary_exists", final_summary),
    ]:
        _record(checks, name, bool(path and path.exists()), str(path))

    if official_zip and official_zip.exists() and official_sha and official_sha.exists():
        expected = _read_sha(official_sha)
        actual = sha256_file(official_zip)
        _record(checks, "official_zip_hash_matches", bool(expected == actual), f"expected={expected} actual={actual}")

    if official_zip and official_zip.exists() and official_contents and official_contents.exists():
        zip_list = zip_entries(official_zip)
        contents_list = _contents_entries(official_contents)
        _record(
            checks,
            "official_contents_match_zip",
            bool(zip_list == contents_list),
            f"zip_entries={len(zip_list)} contents_entries={len(contents_list)}",
        )

    if env_freeze and env_freeze.exists():
        text = read_text(env_freeze)
        _record(checks, "env_freeze_non_empty", bool(text.strip()), f"len={len(text)}")

    if snapshot_manifest is None:
        direct = ws / "snapshot_manifest.json"
        if direct.exists():
            snapshot_manifest = direct
    if snapshot_manifest is None and observations_root is not None:
        snapshot_manifest = _latest_snapshot_manifest(observations_root)

    if snapshot_manifest is None:
        _record(checks, "snapshot_manifest_found", False, "snapshot manifest not specified and latest not found")
    else:
        _record(checks, "snapshot_manifest_exists", bool(snapshot_manifest.exists()), str(snapshot_manifest))
        manifest = read_json_dict(snapshot_manifest)
        if manifest is None:
            _record(checks, "snapshot_manifest_parse", False, str(snapshot_manifest))
        else:
            targets = manifest.get("copied_targets")
            if isinstance(targets, list):
                missing = []
                marked_false = []
                for row in targets:
                    if not isinstance(row, dict):
                        continue
                    rel = row.get("path")
                    exists_mark = bool(row.get("exists"))
                    if not exists_mark:
                        marked_false.append(str(rel))
                        continue
                    if rel is None:
                        continue
                    path = _path_from_arg(ws, str(rel))
                    if path is None or not path.exists():
                        missing.append(str(rel))
                _record(checks, "snapshot_targets_marked_exists", len(marked_false) == 0, f"count={len(marked_false)}")
                _record(checks, "snapshot_targets_exist_on_disk", len(missing) == 0, f"count={len(missing)}")
            else:
                _record(checks, "snapshot_manifest_targets_present", False, "copied_targets missing")

    if packet and packet.exists():
        packet_text = read_text(packet)
        tokens = _path_like_tokens_from_markdown(packet_text)
        missing_refs = []
        for token in tokens:
            path = _path_from_arg(ws, token)
            if path is None or not path.exists():
                missing_refs.append(token)
        sample = ",".join(missing_refs[:5]) if missing_refs else "-"
        _record(
            checks,
            "submission_packet_references_exist",
            len(missing_refs) == 0,
            f"missing={len(missing_refs)} sample={sample}",
        )

        packet_cycles_match = re.search(r"cycles:\s*`(\d+)`", packet_text)
        final_payload = read_json_dict(final_summary) if final_summary is not None else None
        final_cycles = None
        if isinstance(final_payload, dict):
            try:
                final_cycles = int(final_payload.get("cycles"))
            except Exception:
                final_cycles = None
        packet_cycles = int(packet_cycles_match.group(1)) if packet_cycles_match else None
        _record(
            checks,
            "packet_cycles_match_final_summary",
            bool(packet_cycles is not None and final_cycles is not None and packet_cycles == final_cycles),
            f"packet={packet_cycles} final={final_cycles}",
        )

        packet_hashes = [value.lower() for value in re.findall(r"`([0-9a-fA-F]{64})`", packet_text)]
        zip_hash = _read_sha(official_sha) if official_sha is not None else None
        _record(
            checks,
            "packet_contains_official_zip_hash",
            bool(zip_hash is not None and zip_hash in packet_hashes),
            f"hash_found={zip_hash in packet_hashes if zip_hash else False}",
        )

    if public_packet and public_packet.exists():
        public_text = read_text(public_packet)
        tokens = _path_like_tokens_from_markdown(public_text)
        missing_refs = []
        for token in tokens:
            path = _path_from_arg(ws, token)
            if path is None or not path.exists():
                missing_refs.append(token)
        sample = ",".join(missing_refs[:5]) if missing_refs else "-"
        _record(
            checks,
            "public_packet_references_exist",
            len(missing_refs) == 0,
            f"missing={len(missing_refs)} sample={sample}",
        )

    ok = all(bool(item.get("ok")) for item in checks)
    result = {
        "ok": bool(ok),
        "checks": checks,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
