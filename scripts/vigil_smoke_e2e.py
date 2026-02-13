"""VIGIL-RAG End-to-End Smoke Test.

This script demonstrates the full VIGIL-RAG pipeline:
1. Chunk video (shot/micro/meso/macro)
2. Extract retrieval embeddings (OpenCLIP)
3. Store in Qdrant
4. Query → Retrieve → Answer

Index/Query separation: If the video is already indexed (same sha256),
steps 1-4 are skipped and only query/answer steps run.

Usage:
    python scripts/vigil_smoke_e2e.py --video test.mp4 --question "What happened?"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _index_video(video_path, video_id, chunker, embedder, qdrant_service, domain, keyframe_dir):
    """Index a video: chunk → encode keyframes → store in vector DB.

    Returns:
        (micro_metadata, chunk_result) — metadata list and ChunkResult.
    """
    from sopilot.vigil_helpers import index_video_micro

    index_result = index_video_micro(
        video_path, video_id, chunker, embedder, qdrant_service,
        domain=domain, keyframe_dir=keyframe_dir,
    )
    chunk_result = index_result["chunk_result"]
    logger.info("   FPS: %.2f, Duration: %.2f sec", chunk_result.video_fps, chunk_result.video_duration_sec)
    return index_result["micro_metadata"], chunk_result


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
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-K composite answer (0 = disabled, use top-1 only)",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force re-indexing even if video already indexed",
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
    logger.info("Device: %s", args.device)
    logger.info("")

    try:
        # Step 1: Compute stable video_id (sha256)
        logger.info("Step 1: Compute stable video_id")
        logger.info("-" * 60)

        from sopilot.rag_service import compute_video_id

        video_id = compute_video_id(video_path)
        logger.info("   video_id = sha256:%s", video_id[:16])

        # Step 2: Create embedder (always needed for query encoding)
        logger.info("")
        logger.info("Step 2: Create retrieval embedder (OpenCLIP)")
        logger.info("-" * 60)

        from sopilot.retrieval_embeddings import create_embedder

        embedder = create_embedder(
            model_name=args.embedding_model,
            device=args.device,
        )
        logger.info("   Model: %s, dim=%d", args.embedding_model, embedder.config.embedding_dim)

        # Step 3: Connect to vector DB
        logger.info("")
        logger.info("Step 3: Connect to vector DB")
        logger.info("-" * 60)

        from sopilot.qdrant_service import QdrantConfig, QdrantService

        qdrant_config = QdrantConfig(
            host="localhost",
            port=6333,
            embedding_dim=embedder.config.embedding_dim,
        )
        qdrant_service = QdrantService(qdrant_config, use_faiss_fallback=True)

        # Step 4: Index video (skip if already indexed)
        logger.info("")
        logger.info("Step 4: Index video")
        logger.info("-" * 60)

        existing_count = qdrant_service.count_by_video("micro", video_id)

        if existing_count > 0 and not args.force_reindex:
            logger.info("   Already indexed: %d micro chunks (skipping)", existing_count)
            micro_metadata = None  # Not needed for query path
        else:
            if existing_count > 0:
                logger.info("   Force re-index requested (overwriting %d chunks)", existing_count)
            else:
                logger.info("   New video, indexing...")

            from sopilot.chunking_service import ChunkingService

            chunker = ChunkingService()
            keyframe_dir = Path(args.keyframe_dir) if args.keyframe_dir else None

            micro_metadata, chunk_result = _index_video(
                video_path, video_id, chunker, embedder, qdrant_service,
                args.domain, keyframe_dir,
            )
            logger.info("   ✅ Indexing complete")

        # ===== QUERY PATH (always runs) =====

        # Step 5: Encode query and search
        logger.info("")
        logger.info("Step 5: Search for relevant clips")
        logger.info("-" * 60)

        query_embedding = embedder.encode_text([args.question])
        search_results = qdrant_service.search(
            level="micro",
            query_vector=query_embedding[0],
            top_k=5,
            video_id=video_id,
        )
        logger.info("   Found %d results", len(search_results))

        # Step 6: Save keyframe artifacts
        logger.info("")
        logger.info("Step 6: Save keyframe artifacts")
        logger.info("-" * 60)

        artifacts_dir = Path("./artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        if search_results:
            import cv2

            for rank, sr in enumerate(search_results, 1):
                logger.info(
                    "   Rank %d: [%.2f-%.2f sec] score=%.3f",
                    rank, sr.start_sec, sr.end_sec, sr.score,
                )

                # Extract mid-point keyframe
                mid_sec = (sr.start_sec + sr.end_sec) / 2.0
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid_sec * fps))
                ret, frame = cap.read()
                cap.release()

                if ret:
                    artifact_path = (
                        artifacts_dir
                        / f"rank{rank:02d}_t{sr.start_sec:.1f}s_score{sr.score:.3f}.jpg"
                    )
                    cv2.imwrite(str(artifact_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    logger.info("   Saved: %s", artifact_path.name)
        else:
            logger.warning("   No search results found!")

        # Step 7: Generate answer with Video-LLM
        logger.info("")
        logger.info("Step 7: Generate answer with Video-LLM")
        logger.info("-" * 60)

        from sopilot.video_llm_service import VideoLLMService, get_default_config

        llm_model = args.llm_model
        llm_config = get_default_config(llm_model)
        llm_config.device = args.device

        logger.info("   Video-LLM mode: %s", llm_model)

        llm_service = VideoLLMService(llm_config)

        if llm_model != "mock":
            if llm_service._model is None:
                logger.error("   Model failed to load! Install: pip install -e '.[vigil]'")
            else:
                logger.info("   Model loaded: %s", type(llm_service._model).__name__)

        final_answer = None
        clip_observations = None

        if args.top_k > 0 and search_results:
            # Top-K composite answer
            logger.info("   Top-K composite mode: k=%d", args.top_k)

            from sopilot.rag_service import RAGService, RetrievalConfig

            # Use existing_count or len(micro_metadata) for micro_k
            micro_k = existing_count if micro_metadata is None else len(micro_metadata)

            rag_service = RAGService(
                vector_service=qdrant_service,
                llm_service=llm_service,
                retrieval_config=RetrievalConfig(micro_k=max(micro_k, 20)),
                retrieval_embedder=embedder,
            )

            rag_result = rag_service.answer_question_topk(
                video_path,
                args.question,
                video_id=video_id,
                top_k=args.top_k,
            )

            final_answer = rag_result.answer
            clip_observations = rag_result.clip_observations

            logger.info("   ✅ Composite answer from %d clips", len(rag_result.evidence))

            # Save artifacts
            if clip_observations:
                obs_data = [
                    {
                        "clip_id": o.clip_id,
                        "start_sec": o.start_sec,
                        "end_sec": o.end_sec,
                        "relevance": o.relevance,
                        "observation": o.observation,
                        "answer_candidate": o.answer_candidate,
                        "confidence": o.confidence,
                    }
                    for o in clip_observations
                ]
                obs_path = artifacts_dir / "clip_observations.json"
                with open(obs_path, "w", encoding="utf-8") as f:
                    json.dump(obs_data, f, indent=2, ensure_ascii=False)
                logger.info("   Saved: %s", obs_path)

                answer_path = artifacts_dir / "answer.md"
                with open(answer_path, "w", encoding="utf-8") as f:
                    f.write(f"# Question\n\n{args.question}\n\n")
                    f.write(f"# Answer\n\n{final_answer}\n\n")
                    f.write(f"# Evidence ({len(clip_observations)} clips)\n\n")
                    for o in clip_observations:
                        f.write(f"- **[{o.start_sec:.1f}-{o.end_sec:.1f}s]** "
                                f"(relevance={o.relevance:.1f}, confidence={o.confidence:.1f}): "
                                f"{o.observation}\n")
                logger.info("   Saved: %s", answer_path)

        elif search_results:
            top_result = search_results[0]
            logger.info(
                "   Answering from top clip: [%.2f-%.2f sec]",
                top_result.start_sec, top_result.end_sec,
            )

            qa_result = llm_service.answer_question(
                video_path,
                args.question,
                start_sec=top_result.start_sec,
                end_sec=top_result.end_sec,
                enable_cot=False,
            )

            final_answer = qa_result.answer
            logger.info("   ✅ Answer: %s", final_answer[:200])
        else:
            final_answer = "No relevant clips found to answer this question."
            logger.warning("   No clips found, cannot generate answer")

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("E2E smoke test PASSED")
        logger.info("=" * 60)
        logger.info("  video_id: sha256:%s", video_id[:16])
        logger.info("  Question: %s", args.question)
        logger.info("  Retrieved: %d clips", len(search_results))
        if args.top_k > 0:
            logger.info("  Mode: Top-%d composite", args.top_k)
        logger.info("  Answer: %s", final_answer)
        logger.info("  Artifacts: ./artifacts/")

        return 0

    except Exception as e:
        logger.exception("Smoke test failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
