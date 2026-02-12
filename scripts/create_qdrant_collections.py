"""Create Qdrant collections for VIGIL-RAG multi-level retrieval.

This script creates 4 collections:
- vigil_shot: Shot-level clips
- vigil_micro: Micro-level clips (2-4s)
- vigil_meso: Meso-level clips (8-16s)
- vigil_macro: Macro-level clips (32-64s)

Usage:
    python scripts/create_qdrant_collections.py --embedding-dim 768
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_collections(embedding_dim: int = 768, force: bool = False):
    """Create Qdrant collections for all levels.

    Args:
        embedding_dim: Embedding dimension (768 for OpenCLIP/InternVideo, 4096 for LLaVA)
        force: If True, recreate collections even if they exist
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
    except ImportError:
        print("❌ qdrant-client not installed. Run: pip install -e '.[vigil]'")
        return False

    host = os.getenv("VIGIL_QDRANT_HOST", "localhost")
    port = int(os.getenv("VIGIL_QDRANT_PORT", "6333"))

    try:
        client = QdrantClient(host=host, port=port)
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        print(f"   URL: http://{host}:{port}")
        print("   Check: docker-compose ps qdrant")
        return False

    levels = ["shot", "micro", "meso", "macro"]
    created = []
    skipped = []

    for level in levels:
        collection_name = f"vigil_{level}"

        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if exists and not force:
            print(f"⏭️  Collection '{collection_name}' already exists (skipping)")
            skipped.append(collection_name)
            continue

        if exists and force:
            print(f"🗑️  Deleting existing collection '{collection_name}'")
            client.delete_collection(collection_name)

        # Create collection
        print(f"✨ Creating collection '{collection_name}' (dim={embedding_dim})")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        created.append(collection_name)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✅ Created: {len(created)} collections")
    if created:
        for name in created:
            print(f"   - {name}")

    if skipped:
        print(f"⏭️  Skipped: {len(skipped)} collections (already exist)")
        for name in skipped:
            print(f"   - {name}")

    print()
    print("Next step:")
    print("  python scripts/vigil_smoke_e2e.py --video test.mp4")

    return True


def main():
    parser = argparse.ArgumentParser(description="Create Qdrant collections for VIGIL-RAG")
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=768,
        help="Embedding dimension (768 for OpenCLIP/InternVideo, 4096 for LLaVA)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate collections even if they exist (WARNING: deletes data)",
    )

    args = parser.parse_args()

    if args.force:
        confirm = input("⚠️  This will DELETE existing collections. Continue? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            return 1

    success = create_collections(embedding_dim=args.embedding_dim, force=args.force)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
