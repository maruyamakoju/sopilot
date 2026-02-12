"""VIGIL-RAG End-to-End Smoke Test.

This script demonstrates the full VIGIL-RAG pipeline:
1. Chunk video (shot/micro/meso/macro)
2. Extract retrieval embeddings (OpenCLIP)
3. Store in Qdrant
4. Query → Retrieve → Answer

Usage:
    python scripts/vigil_smoke_e2e.py --video test.mp4 --question "What happened?"
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="VIGIL-RAG E2E Smoke Test")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument(
        "--question",
        type=str,
        default="What is happening in the video?",
        help="Question to ask",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="generic",
        choices=["factory", "surveillance", "sports", "generic"],
        help="Video domain",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model (ViT-B-32, ViT-L-14, ViT-H-14)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for embeddings",
    )
    parser.add_argument(
        "--keyframe-dir",
        type=str,
        default=None,
        help="Directory to save keyframes (optional)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="mock",
        choices=["mock", "qwen2.5-vl-7b"],
        help="Video-LLM model for answer generation",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        return 1

    logger.info("=" * 60)
    logger.info("VIGIL-RAG E2E Smoke Test")
    logger.info("=" * 60)
    logger.info("Video: %s", video_path)
    logger.info("Question: %s", args.question)
    logger.info("Domain: %s", args.domain)
    logger.info("Embedding Model: %s", args.embedding_model)
    logger.info("Device: %s", args.device)
    logger.info("")

    try:
        # Step 1: Multi-scale chunking
        logger.info("Step 1: Multi-scale video chunking")
        logger.info("-" * 60)
        from sopilot.chunking_service import ChunkingService

        chunker = ChunkingService()
        keyframe_dir = Path(args.keyframe_dir) if args.keyframe_dir else None

        result = chunker.chunk_video(
            video_path,
            domain=args.domain,
            keyframe_dir=keyframe_dir,
        )

        logger.info("✅ Chunking complete:")
        logger.info("   Shots: %d", len(result.shots))
        logger.info("   Micro: %d", len(result.micro))
        logger.info("   Meso: %d", len(result.meso))
        logger.info("   Macro: %d", len(result.macro))
        logger.info("   FPS: %.2f", result.video_fps)
        logger.info("   Duration: %.2f sec", result.video_duration_sec)
        logger.info("")

        # Step 2: Extract retrieval embeddings
        logger.info("Step 2: Encode query text (OpenCLIP)")
        logger.info("-" * 60)
        from sopilot.retrieval_embeddings import create_embedder

        embedder = create_embedder(
            model_name=args.embedding_model,
            device=args.device,
        )

        query_embedding = embedder.encode_text([args.question])
        logger.info("✅ Query encoded: shape=%s", query_embedding.shape)
        logger.info("")

        # Step 3: Extract and encode micro chunk keyframes
        logger.info("Step 3: Extract and encode micro chunk keyframes")
        logger.info("-" * 60)

        import uuid
        import cv2
        from PIL import Image

        video_id = str(uuid.uuid4())  # Generate unique video ID

        # Extract keyframes and compute embeddings for each micro chunk
        micro_metadata = []
        micro_embeddings = []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open video for keyframe extraction")
            return 1

        for idx, chunk in enumerate(result.micro):
            keyframes = []
            for frame_idx in chunk.keyframe_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    keyframes.append(Image.fromarray(frame_rgb))

            if keyframes:
                # Encode all keyframes for this chunk
                keyframe_embeddings = embedder.encode_images(keyframes)

                # Average pooling + L2 normalize
                avg_embedding = np.mean(keyframe_embeddings, axis=0)
                avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-9)

                micro_embeddings.append(avg_embedding)
                micro_metadata.append({
                    "clip_id": str(uuid.uuid4()),
                    "video_id": video_id,
                    "start_sec": chunk.start_sec,
                    "end_sec": chunk.end_sec,
                    "chunk_index": idx,
                })

        cap.release()

        micro_embeddings_array = np.array(micro_embeddings, dtype=np.float32)
        logger.info("✅ Encoded %d micro chunks", len(micro_embeddings_array))
        logger.info("   Embedding dimension: %d", micro_embeddings_array.shape[1])
        logger.info("")

        # Step 4: Store in Qdrant
        logger.info("Step 4: Store embeddings in Qdrant")
        logger.info("-" * 60)

        from sopilot.qdrant_service import QdrantService, QdrantConfig

        qdrant_config = QdrantConfig(
            host="localhost",
            port=6333,
            embedding_dim=embedder.config.embedding_dim,
        )

        qdrant_service = QdrantService(qdrant_config, use_faiss_fallback=True)
        qdrant_service.ensure_collections(levels=["micro"], embedding_dim=embedder.config.embedding_dim)

        num_added = qdrant_service.add_embeddings(
            level="micro",
            embeddings=micro_embeddings_array,
            metadata=micro_metadata,
        )

        logger.info("✅ Stored %d embeddings in Qdrant/FAISS", num_added)
        logger.info("")

        # Step 5: Search and retrieve
        logger.info("Step 5: Search for top-5 relevant clips")
        logger.info("-" * 60)

        search_results = qdrant_service.search(
            level="micro",
            query_vector=query_embedding[0],  # Extract single vector from batch
            top_k=5,
            video_id=video_id,
        )

        logger.info("✅ Found %d results", len(search_results))
        logger.info("")

        # Step 6: Output results with artifacts
        logger.info("Step 6: Output results with keyframe artifacts")
        logger.info("-" * 60)

        artifacts_dir = Path("./artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        if not search_results:
            logger.warning("No search results found!")
        else:
            for rank, search_result in enumerate(search_results, 1):
                logger.info(
                    "Rank %d: [%.2f-%.2f sec] score=%.3f",
                    rank,
                    search_result.start_sec,
                    search_result.end_sec,
                    search_result.score,
                )

                # Find the chunk to get keyframes
                chunk_idx = next(
                    i for i, meta in enumerate(micro_metadata)
                    if meta["clip_id"] == search_result.clip_id
                )
                chunk = result.micro[chunk_idx]

                # Extract representative keyframe (middle one)
                mid_keyframe_idx = chunk.keyframe_indices[len(chunk.keyframe_indices) // 2]

                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_keyframe_idx)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    artifact_path = (
                        artifacts_dir
                        / f"rank{rank:02d}_t{search_result.start_sec:.1f}s_score{search_result.score:.3f}.jpg"
                    )
                    cv2.imwrite(str(artifact_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    logger.info("   Saved keyframe: %s", artifact_path.name)

        logger.info("")

        # Step 7: Generate answer with Video-LLM
        logger.info("Step 7: Generate answer with Video-LLM")
        logger.info("-" * 60)

        from sopilot.video_llm_service import VideoLLMService, get_default_config

        # Use mock mode for now (can upgrade to Qwen2.5-VL with --use-llm flag)
        llm_model = getattr(args, "llm_model", "mock")
        llm_config = get_default_config(llm_model)
        llm_config.device = args.device

        llm_service = VideoLLMService(llm_config)

        # Generate answer from top-1 clip if available
        final_answer = None
        if search_results:
            top_result = search_results[0]
            logger.info(
                "Answering based on top clip: [%.2f-%.2f sec]",
                top_result.start_sec,
                top_result.end_sec,
            )

            qa_result = llm_service.answer_question(
                video_path,
                args.question,
                start_sec=top_result.start_sec,
                end_sec=top_result.end_sec,
                enable_cot=False,
            )

            final_answer = qa_result.answer
            logger.info("✅ Answer generated: %s", final_answer[:200])
        else:
            final_answer = "No relevant clips found to answer this question."
            logger.warning("✗ No clips found, cannot generate answer")

        logger.info("")
        logger.info("=" * 60)
        logger.info("🎉 E2E smoke test PASSED (micro-only + Video-LLM)")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Results:")
        logger.info("  - Video: %s (%.1f sec, %d micro chunks)",
                    video_path.name, result.video_duration_sec, len(result.micro))
        logger.info("  - Question: %s", args.question)
        logger.info("  - Retrieved: %d clips", len(search_results))
        logger.info("  - Answer: %s", final_answer)
        logger.info("  - Artifacts: ./artifacts/")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Review artifacts to validate relevance")
        logger.info("  2. Install Qwen2.5-VL for real answer generation (pip install qwen-vl-utils)")
        logger.info("  3. Add coarse-to-fine retrieval (meso/macro levels)")

        return 0

    except Exception as e:
        logger.exception("Smoke test failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
