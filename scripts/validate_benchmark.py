#!/usr/bin/env python3
"""Validate VIGIL-RAG benchmark files for common issues.

This script checks:
- All queries have GT specified (relevant_clip_ids or relevant_time_ranges)
- GT time ranges are not too wide (> 60s warns)
- video_id references exist in video_paths mapping
- No duplicate query_ids

Usage:
    python scripts/validate_benchmark.py --benchmark benchmarks/manufacturing_v1.jsonl
    python scripts/validate_benchmark.py --benchmark benchmarks/real_v2.jsonl --video-map benchmarks/video_paths.local.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def validate_benchmark(
    benchmark_path: Path,
    video_map_path: Path | None = None,
    max_time_range_sec: float = 60.0,
    min_overlap_sec: float = 0.0,
) -> list[str]:
    """Validate benchmark file.

    Args:
        benchmark_path: Path to benchmark JSONL file
        video_map_path: Optional path to video_paths.local.json
        max_time_range_sec: Maximum allowed GT time range (warns if exceeded)
        min_overlap_sec: Minimum overlap for temporal matching (informational)

    Returns:
        List of validation errors/warnings (empty if valid)
    """
    errors = []
    warnings = []

    # Load benchmark
    try:
        with open(benchmark_path, encoding="utf-8") as f:
            queries = [json.loads(line) for line in f]
    except Exception as e:
        return [f"Failed to load benchmark: {e}"]

    if not queries:
        return ["Benchmark is empty"]

    # Load video map if provided
    video_map = {}
    if video_map_path and video_map_path.exists():
        try:
            with open(video_map_path, encoding="utf-8") as f:
                video_map = json.load(f)
        except Exception as e:
            warnings.append(f"Failed to load video map: {e}")

    # Track query_ids for duplicates
    query_ids = set()

    for i, q in enumerate(queries, start=1):
        query_id = q.get("query_id", f"query_{i}")

        # Check duplicate query_id
        if query_id in query_ids:
            errors.append(f"{query_id}: Duplicate query_id")
        query_ids.add(query_id)

        # Check required fields
        if "query_text" not in q:
            errors.append(f"{query_id}: Missing query_text")
        if "video_id" not in q:
            errors.append(f"{query_id}: Missing video_id")

        # Check GT specified
        has_clip_ids = q.get("relevant_clip_ids") and len(q["relevant_clip_ids"]) > 0
        has_time_ranges = q.get("relevant_time_ranges") and len(q["relevant_time_ranges"]) > 0

        if not has_clip_ids and not has_time_ranges:
            # This might be intentional for "no GT" queries (e.g., skipped steps)
            # Warn instead of error
            warnings.append(f"{query_id}: No GT specified (relevant_clip_ids and relevant_time_ranges both empty)")

        # Check time ranges are not too wide
        if has_time_ranges:
            for r in q["relevant_time_ranges"]:
                start = r.get("start_sec", 0.0)
                end = r.get("end_sec", 0.0)
                duration = end - start

                if duration > max_time_range_sec:
                    warnings.append(
                        f"{query_id}: GT time range too wide ({duration:.1f}s > {max_time_range_sec:.1f}s). "
                        "This may cause R@1=1.0 saturation."
                    )

                if duration < 0:
                    errors.append(f"{query_id}: Invalid time range (start > end): {start:.1f}s > {end:.1f}s")

        # Check video_id exists in video_map
        if video_map and q.get("video_id") not in video_map:
            warnings.append(f"{query_id}: video_id '{q['video_id']}' not found in video_map")

    # Summary
    summary = []
    summary.append(f"Total queries: {len(queries)}")
    summary.append(f"Errors: {len(errors)}")
    summary.append(f"Warnings: {len(warnings)}")

    if min_overlap_sec > 0:
        summary.append(f"Note: min_overlap_sec={min_overlap_sec:.1f}s will be used for temporal matching")

    return summary + errors + warnings


def main():
    parser = argparse.ArgumentParser(description="Validate VIGIL-RAG benchmark files")
    parser.add_argument("--benchmark", type=Path, required=True, help="Path to benchmark JSONL file")
    parser.add_argument("--video-map", type=Path, help="Path to video_paths.local.json (optional)")
    parser.add_argument(
        "--max-time-range", type=float, default=60.0, help="Maximum allowed GT time range in seconds (default: 60)"
    )
    parser.add_argument(
        "--min-overlap", type=float, default=0.0, help="Minimum overlap for temporal matching (informational)"
    )

    args = parser.parse_args()

    if not args.benchmark.exists():
        print(f"Error: Benchmark file not found: {args.benchmark}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating benchmark: {args.benchmark}", file=sys.stderr)
    if args.video_map:
        print(f"Video map: {args.video_map}", file=sys.stderr)
    print(f"Max time range: {args.max_time_range:.1f}s", file=sys.stderr)
    print()

    # Validate
    results = validate_benchmark(
        args.benchmark,
        video_map_path=args.video_map,
        max_time_range_sec=args.max_time_range,
        min_overlap_sec=args.min_overlap,
    )

    # Print results
    has_errors = any("Error:" in r or "Duplicate" in r or "Missing" in r or "Invalid" in r for r in results)

    for line in results:
        if "Error:" in line or "Duplicate" in line or "Missing" in line or "Invalid" in line:
            print(f"❌ {line}", file=sys.stderr)
        elif "Warning:" in line or "too wide" in line or "not found" in line:
            print(f"⚠️  {line}", file=sys.stderr)
        else:
            print(f"ℹ️  {line}", file=sys.stderr)

    # Exit with error code if validation failed
    if has_errors:
        print("\n❌ Validation FAILED", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n✅ Validation PASSED", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
