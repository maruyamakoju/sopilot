#!/usr/bin/env python3
"""Index partner-provided SOP video into vector store.

This is a convenience wrapper around vigil_helpers.index_video_all_levels()
with sensible defaults for Manufacturing-v1 workflow.

Usage:
    # Basic indexing (micro only)
    python scripts/index_partner_video.py \
        --video demo_videos/partner/oilchange_gold_202602.mp4 \
        --video-id oilchange-gold

    # Hierarchical indexing (micro + meso + macro)
    python scripts/index_partner_video.py \
        --video demo_videos/partner/oilchange_gold_202602.mp4 \
        --video-id oilchange-gold \
        --hierarchical \
        --embedding-model ViT-H-14

    # With transcription (audio)
    python scripts/index_partner_video.py \
        --video demo_videos/partner/oilchange_gold_202602.mp4 \
        --video-id oilchange-gold \
        --hierarchical \
        --transcribe
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sopilot.chunking_service import ChunkingService
from sopilot.config import get_settings
from sopilot.qdrant_service import QdrantService
from sopilot.retrieval_embeddings import RetrievalConfig, RetrievalEmbedder
from sopilot.transcription_service import TranscriptionService
from sopilot.vigil_helpers import index_video_all_levels, index_video_micro


def save_chunk_manifests(video_id: str, index_result: dict, output_dir: Path = Path("chunks")):
    """Save chunk manifests to JSON files for GT creation.

    This creates {video_id}.{level}.json files with clip_id, start_sec, end_sec
    for each chunk level. These files enable chunk-based GT creation without
    querying the vector database.

    Args:
        video_id: Video identifier
        index_result: Return value from index_video_all_levels() or index_video_micro()
        output_dir: Directory to save manifest files (default: chunks/)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save micro chunks
    if "micro_metadata" in index_result:
        micro_chunks = []
        for meta in index_result["micro_metadata"]:
            chunk = {
                "clip_id": meta.get("clip_id"),
                "start_sec": meta.get("start_sec"),
                "end_sec": meta.get("end_sec"),
                "duration_sec": meta.get("end_sec", 0) - meta.get("start_sec", 0),
            }
            # Add transcript if available
            if "transcript_text" in meta:
                chunk["transcript_text"] = meta["transcript_text"]
            micro_chunks.append(chunk)

        micro_manifest = {
            "video_id": video_id,
            "level": "micro",
            "total_chunks": len(micro_chunks),
            "chunks": micro_chunks,
        }
        micro_path = output_dir / f"{video_id}.micro.json"
        with open(micro_path, "w", encoding="utf-8") as f:
            json.dump(micro_manifest, f, indent=2)
        print(f"Saved micro manifest: {micro_path} ({len(micro_chunks)} chunks)", file=sys.stderr)

    # Save meso chunks (if hierarchical)
    if "meso_metadata" in index_result:
        meso_chunks = []
        for meta in index_result["meso_metadata"]:
            chunk = {
                "clip_id": meta.get("clip_id"),
                "start_sec": meta.get("start_sec"),
                "end_sec": meta.get("end_sec"),
                "duration_sec": meta.get("end_sec", 0) - meta.get("start_sec", 0),
            }
            meso_chunks.append(chunk)

        meso_manifest = {
            "video_id": video_id,
            "level": "meso",
            "total_chunks": len(meso_chunks),
            "chunks": meso_chunks,
        }
        meso_path = output_dir / f"{video_id}.meso.json"
        with open(meso_path, "w", encoding="utf-8") as f:
            json.dump(meso_manifest, f, indent=2)
        print(f"Saved meso manifest: {meso_path} ({len(meso_chunks)} chunks)", file=sys.stderr)

    # Save macro chunks (if hierarchical)
    if "macro_metadata" in index_result:
        macro_chunks = []
        for meta in index_result["macro_metadata"]:
            chunk = {
                "clip_id": meta.get("clip_id"),
                "start_sec": meta.get("start_sec"),
                "end_sec": meta.get("end_sec"),
                "duration_sec": meta.get("end_sec", 0) - meta.get("start_sec", 0),
            }
            macro_chunks.append(chunk)

        macro_manifest = {
            "video_id": video_id,
            "level": "macro",
            "total_chunks": len(macro_chunks),
            "chunks": macro_chunks,
        }
        macro_path = output_dir / f"{video_id}.macro.json"
        with open(macro_path, "w", encoding="utf-8") as f:
            json.dump(macro_manifest, f, indent=2)
        print(f"Saved macro manifest: {macro_path} ({len(macro_chunks)} chunks)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Index partner SOP video into vector store")
    parser.add_argument("--video", type=Path, required=True, help="Path to video file")
    parser.add_argument("--video-id", required=True, help="Video identifier (e.g., oilchange-gold)")
    parser.add_argument("--hierarchical", action="store_true", help="Index all levels (micro/meso/macro)")
    parser.add_argument(
        "--embedding-model",
        default="ViT-B-32",
        choices=["ViT-B-32", "ViT-L-14", "ViT-H-14"],
        help="OpenCLIP model to use",
    )
    parser.add_argument("--transcribe", action="store_true", help="Enable transcription (requires whisper)")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size")
    parser.add_argument("--reindex", action="store_true", help="Force reindexing (delete existing)")

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    # Initialize services
    settings = get_settings()

    # Set embedding model
    print(f"Using embedding model: {args.embedding_model}", file=sys.stderr)

    # Initialize services
    chunker = ChunkingService()
    qdrant = QdrantService(settings)

    # Create OpenCLIP retrieval embedder
    retrieval_config = RetrievalConfig.for_model(args.embedding_model)
    retrieval_config.device = "cpu"  # Can add --device arg later if needed
    embedder = RetrievalEmbedder(retrieval_config)

    transcription_service = None
    if args.transcribe:
        print("Initializing transcription service...", file=sys.stderr)
        transcription_service = TranscriptionService(
            backend="openai-whisper",
            model_name=args.whisper_model,
        )

    # Check if video already indexed
    if not args.reindex:
        try:
            # Try to search for existing clips
            import numpy as np

            dummy_vector = np.zeros(512).tolist()
            existing = qdrant.search(
                query_vector=dummy_vector,
                level="micro",
                video_id=args.video_id,
                k=1,
            )
            if existing:
                print(
                    f"Warning: Video {args.video_id} already indexed. Use --reindex to force reindexing.",
                    file=sys.stderr,
                )
                response = input("Continue anyway? [y/N]: ")
                if response.lower() != "y":
                    sys.exit(0)
        except Exception:
            pass  # Video not indexed yet

    # Index video
    print(f"Indexing video: {args.video}", file=sys.stderr)
    print(f"Video ID: {args.video_id}", file=sys.stderr)
    print(f"Hierarchical: {args.hierarchical}", file=sys.stderr)
    print(f"Transcription: {args.transcribe}", file=sys.stderr)

    start_time = time.time()

    try:
        if args.hierarchical:
            # Index all levels
            print("Indexing micro + meso + macro levels...", file=sys.stderr)
            index_result = index_video_all_levels(
                video_path=Path(args.video),
                video_id=args.video_id,
                chunker=chunker,
                embedder=embedder,
                qdrant_service=qdrant,
                transcription_service=transcription_service,
            )
        else:
            # Index micro only
            print("Indexing micro level only...", file=sys.stderr)
            index_result = index_video_micro(
                video_path=Path(args.video),
                video_id=args.video_id,
                chunker=chunker,
                embedder=embedder,
                qdrant_service=qdrant,
                transcription_service=transcription_service,
            )

        elapsed = time.time() - start_time
        print(f"\n✅ Indexing complete in {elapsed:.1f}s", file=sys.stderr)

        # Save chunk manifests for GT creation
        save_chunk_manifests(args.video_id, index_result)

        # Print summary
        print("\nNext steps:", file=sys.stderr)
        print(f"  1. View chunks: cat chunks/{args.video_id}.micro.json", file=sys.stderr)
        print(
            "  2. Create GT: Edit benchmarks/manufacturing_v1.jsonl with relevant_clip_ids from chunk manifest",
            file=sys.stderr,
        )
        print("  3. Validate: python scripts/validate_benchmark.py --benchmark manufacturing_v1.jsonl", file=sys.stderr)
        print(
            "  4. Evaluate: python scripts/evaluate_vigil_real.py --benchmark manufacturing_v1.jsonl --hierarchical",
            file=sys.stderr,
        )

    except Exception as e:
        print(f"\n❌ Indexing failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
