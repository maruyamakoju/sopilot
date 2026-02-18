#!/usr/bin/env python3
"""List all chunks for a video from Qdrant/FAISS vector store.

This script queries the vector database to retrieve all chunks for a given video
and outputs them as JSON for GT creation workflows.

Usage:
    python scripts/list_video_chunks.py \
        --video-id oilchange-gold \
        --level micro \
        --out chunks/oilchange_gold_chunks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sopilot.config import get_settings
from sopilot.qdrant_service import QdrantService


def list_chunks(video_id: str, level: str, qdrant: QdrantService) -> list[dict[str, Any]]:
    """List all chunks for a video at a given level.

    Args:
        video_id: Video identifier
        level: Chunk level (micro, meso, macro)
        qdrant: QdrantService instance

    Returns:
        List of chunk dictionaries with clip_id, start_sec, end_sec, etc.
    """
    # Use search with dummy vector to get all chunks
    # This is a workaround - ideally we'd have a get_all_clips() method
    try:
        import numpy as np

        # Create a dummy vector (all zeros)
        # For OpenCLIP ViT-B-32: 512-dim
        # For ViT-L-14: 768-dim
        # For ViT-H-14: 1024-dim
        # We'll try 512 first and catch dimension mismatch errors
        dummy_vector = np.zeros(512).tolist()

        # Search with large top_k to get all chunks
        results = qdrant.search(
            level=level,
            query_vector=np.array(dummy_vector),
            video_id=video_id,
            top_k=1000,  # Large enough to get all chunks (adjust if needed)
        )

        # Sort by start_sec
        results.sort(key=lambda x: x.get("start_sec", 0))

        return results

    except Exception as e:
        print(f"Error listing chunks: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return []


def generate_chunk_listing(video_id: str, level: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate chunk listing JSON.

    Args:
        video_id: Video identifier
        level: Chunk level
        chunks: List of chunk dictionaries

    Returns:
        Listing dictionary
    """
    listing = {
        "video_id": video_id,
        "level": level,
        "total_chunks": len(chunks),
        "chunks": [],
    }

    for chunk in chunks:
        chunk_info = {
            "clip_id": chunk.get("clip_id", "unknown"),
            "start_sec": chunk.get("start_sec", 0.0),
            "end_sec": chunk.get("end_sec", 0.0),
            "duration_sec": chunk.get("end_sec", 0.0) - chunk.get("start_sec", 0.0),
        }
        # Add keyframe path if available
        if "keyframe_path" in chunk:
            chunk_info["keyframe_path"] = chunk["keyframe_path"]

        listing["chunks"].append(chunk_info)

    return listing


def main():
    parser = argparse.ArgumentParser(description="List all chunks for a video from vector store")
    parser.add_argument("--video-id", required=True, help="Video identifier")
    parser.add_argument("--level", choices=["micro", "meso", "macro"], default="micro", help="Chunk level")
    parser.add_argument("--out", type=Path, help="Output JSON path (default: stdout)")

    args = parser.parse_args()

    # Initialize QdrantService
    settings = get_settings()
    qdrant = QdrantService(settings)

    print(f"Listing {args.level} chunks for video: {args.video_id}", file=sys.stderr)

    # List chunks
    chunks = list_chunks(args.video_id, args.level, qdrant)

    if not chunks:
        print(f"Warning: No chunks found for video {args.video_id} at level {args.level}", file=sys.stderr)
        print("Possible reasons:", file=sys.stderr)
        print("  - Video not indexed yet", file=sys.stderr)
        print("  - Wrong video_id", file=sys.stderr)
        print("  - Wrong level", file=sys.stderr)

    # Generate listing
    listing = generate_chunk_listing(args.video_id, args.level, chunks)

    print(f"Found {listing['total_chunks']} chunks", file=sys.stderr)

    # Write output
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(listing, f, indent=2)
        print(f"Listing written to {args.out}", file=sys.stderr)
    else:
        print(json.dumps(listing, indent=2))


if __name__ == "__main__":
    main()
