"""Event Detection Demo for VIGIL-RAG.

2-stage event detection pipeline:
  Stage 1: Retrieve candidate clips via text-to-video similarity
  Stage 2: LLM verification with confidence scoring

Index/query separation: if the video is already indexed (same sha256),
indexing is skipped and only detection runs.

Usage:
    # Mock LLM (no GPU required):
    python scripts/event_detection_demo.py \\
        --video test_video.mp4 \\
        --event-types "color change,scene transition" \\
        --llm-model mock --device cpu

    # Real Qwen2.5-VL with ground truth evaluation:
    python scripts/event_detection_demo.py \\
        --video surveillance.mp4 \\
        --event-types "intrusion,PPE violation" \\
        --llm-model qwen2.5-vl-7b --device cuda \\
        --ground-truth gt.json
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


def _index_video(video_path, video_id, chunker, embedder, qdrant_service, keyframe_dir):
    """Index a video: chunk -> encode keyframes -> store in vector DB."""
    from sopilot.vigil_helpers import index_video_micro

    index_result = index_video_micro(
        video_path,
        video_id,
        chunker,
        embedder,
        qdrant_service,
        keyframe_dir=keyframe_dir,
    )
    return index_result["num_added"]


def main():
    parser = argparse.ArgumentParser(description="VIGIL-RAG Event Detection Demo")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument(
        "--event-types",
        type=str,
        required=True,
        help="Comma-separated event type descriptions (e.g. 'intrusion,PPE violation')",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to ground truth JSON (list of {start_sec, end_sec, event_type})",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument(
        "--llm-model",
        type=str,
        default="mock",
        choices=["mock", "qwen2.5-vl-7b"],
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model (ViT-B-32, ViT-L-14, ViT-H-14)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Proposals per event type")
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--force-reindex", action="store_true")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        return 1

    event_types = [et.strip() for et in args.event_types.split(",") if et.strip()]
    if not event_types:
        logger.error("No event types provided")
        return 1

    logger.info("=" * 60)
    logger.info("VIGIL-RAG Event Detection Demo")
    logger.info("=" * 60)
    logger.info("Video: %s", video_path)
    logger.info("Event types: %s", event_types)
    logger.info("LLM: %s, Device: %s", args.llm_model, args.device)
    logger.info("")

    try:
        # Step 1: Compute video ID
        logger.info("Step 1: Compute video ID")
        logger.info("-" * 40)

        from sopilot.rag_service import compute_video_id

        video_id = compute_video_id(video_path)
        logger.info("  video_id = sha256:%s", video_id[:16])

        # Step 2: Create embedder
        logger.info("")
        logger.info("Step 2: Create retrieval embedder")
        logger.info("-" * 40)

        from sopilot.retrieval_embeddings import create_embedder

        embedder = create_embedder(model_name=args.embedding_model, device=args.device)
        logger.info("  Model: %s, dim=%d", args.embedding_model, embedder.config.embedding_dim)

        # Step 3: Connect to vector DB
        logger.info("")
        logger.info("Step 3: Connect to vector DB")
        logger.info("-" * 40)

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
        logger.info("-" * 40)

        existing_count = qdrant_service.count_by_video("micro", video_id)

        if existing_count > 0 and not args.force_reindex:
            logger.info("  Already indexed: %d micro chunks (skipping)", existing_count)
        else:
            from sopilot.chunking_service import ChunkingService

            chunker = ChunkingService()
            _index_video(video_path, video_id, chunker, embedder, qdrant_service, keyframe_dir=None)

        # Step 5: Create LLM service
        logger.info("")
        logger.info("Step 5: Create Video-LLM service")
        logger.info("-" * 40)

        from sopilot.video_llm_service import VideoLLMService, get_default_config

        llm_config = get_default_config(args.llm_model)
        llm_config.device = args.device
        llm_service = VideoLLMService(llm_config)
        logger.info("  LLM mode: %s", args.llm_model)

        # Step 6: Run event detection
        logger.info("")
        logger.info("Step 6: Run 2-stage event detection")
        logger.info("-" * 40)

        from sopilot.event_detection_service import EventDetectionService

        detector = EventDetectionService(
            vector_service=qdrant_service,
            llm_service=llm_service,
            retrieval_embedder=embedder,
        )

        detection_result = detector.detect_events(
            video_path,
            event_types,
            video_id=video_id,
            top_k=args.top_k,
            confidence_threshold=args.confidence_threshold,
        )

        logger.info("  Proposals: %d", detection_result.num_proposals)
        logger.info("  Verified:  %d", detection_result.num_verified)

        for i, evt in enumerate(detection_result.events, 1):
            logger.info(
                "  Event %d: [%.1f-%.1f s] type=%s confidence=%.2f",
                i,
                evt.start_sec,
                evt.end_sec,
                evt.event_type,
                evt.confidence,
            )
            logger.info("    Observation: %s", evt.observation[:120])

        # Step 7: Save artifacts
        logger.info("")
        logger.info("Step 7: Save artifacts")
        logger.info("-" * 40)

        artifacts_dir = Path("./artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        events_data = [
            {
                "event_type": evt.event_type,
                "start_sec": evt.start_sec,
                "end_sec": evt.end_sec,
                "confidence": evt.confidence,
                "clip_id": evt.clip_id,
                "observation": evt.observation,
            }
            for evt in detection_result.events
        ]

        output_path = artifacts_dir / "detected_events.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "video_id": detection_result.video_id,
                    "num_proposals": detection_result.num_proposals,
                    "num_verified": detection_result.num_verified,
                    "events": events_data,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("  Saved: %s", output_path)

        # Step 8: Evaluate against ground truth (optional)
        if args.ground_truth:
            logger.info("")
            logger.info("Step 8: Evaluate against ground truth")
            logger.info("-" * 40)

            with open(args.ground_truth, encoding="utf-8") as f:
                gt_events = json.load(f)

            from sopilot.evaluation.vigil_metrics import event_detection_metrics

            predictions = [
                {
                    "start_sec": evt.start_sec,
                    "end_sec": evt.end_sec,
                    "event_type": evt.event_type,
                    "confidence": evt.confidence,
                }
                for evt in detection_result.events
            ]

            # Estimate video duration for FAH calculation
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            duration_hours = (total_frames / fps) / 3600.0

            metrics = event_detection_metrics(
                predictions,
                gt_events,
                iou_threshold=0.5,
                video_duration_hours=max(duration_hours, 1e-6),
            )

            logger.info("  Precision: %.3f", metrics.precision)
            logger.info("  Recall:    %.3f", metrics.recall)
            logger.info("  F1 Score:  %.3f", metrics.f1_score)
            logger.info("  AP (mAP):  %.3f", metrics.ap)
            logger.info("  FAH:       %.1f", metrics.fah)

            # Save metrics
            metrics_path = artifacts_dir / "detection_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1_score": metrics.f1_score,
                        "ap": metrics.ap,
                        "fah": metrics.fah,
                    },
                    f,
                    indent=2,
                )
            logger.info("  Saved: %s", metrics_path)

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Event detection complete")
        logger.info("=" * 60)
        logger.info("  Video: sha256:%s", video_id[:16])
        logger.info("  Event types: %s", event_types)
        logger.info("  Proposals: %d -> Verified: %d", detection_result.num_proposals, detection_result.num_verified)
        logger.info("  Artifacts: ./artifacts/")

        return 0

    except Exception as e:
        logger.exception("Event detection failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
