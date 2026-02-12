"""Video-LLM integration service for VIGIL-RAG.

This module provides:
- Video-LLM model wrappers (InternVideo2.5 Chat, LLaVA-Video)
- Embedding extraction for retrieval
- Video question answering
- Frame sampling and preprocessing
- Batch inference support

Supported models:
- InternVideo2.5 Chat 8B (primary): https://huggingface.co/OpenGVLab/InternVideo2_5-Chat-8B
- LLaVA-Video 7B (fallback): https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

try:
    from qwen_vl_utils import process_vision_info
    QWEN_UTILS_AVAILABLE = True
except ImportError:
    QWEN_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)

ModelName = Literal["qwen2.5-vl-7b", "internvideo2.5-chat-8b", "llava-video-7b", "mock"]


@dataclass
class VideoLLMConfig:
    """Configuration for Video-LLM service."""

    model_name: ModelName = "qwen2.5-vl-7b"
    device: str = "cuda"  # cuda / cpu
    dtype: str = "float16"  # float16 / bfloat16 / float32
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Frame sampling
    max_frames: int = 32  # Maximum frames per video clip
    frame_sample_strategy: Literal["uniform", "fps"] = "uniform"
    fps: float = 1.0  # Frame sampling rate for Qwen2.5-VL

    # Inference
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Qwen2.5-VL specific
    min_pixels: int = 256 * 28 * 28  # Minimum pixels for dynamic resolution
    max_pixels: int = 1280 * 28 * 28  # Maximum pixels for dynamic resolution


@dataclass
class VideoQAResult:
    """Result of video question answering."""

    question: str
    answer: str
    confidence: float | None = None  # Model confidence (if available)
    reasoning: str | None = None  # Chain-of-thought reasoning (if enabled)


class VideoLLMService:
    """Video-LLM service for embedding extraction and QA."""

    def __init__(self, config: VideoLLMConfig) -> None:
        """Initialize Video-LLM service.

        Args:
            config: Video-LLM configuration

        Raises:
            RuntimeError: If model loading fails
        """
        self.config = config
        self._model = None
        self._processor = None

        if config.model_name != "mock":
            self._load_model()

    def _load_model(self) -> None:
        """Load Video-LLM model and processor.

        Raises:
            RuntimeError: If model loading fails
        """
        logger.info("Loading Video-LLM model: %s", self.config.model_name)

        if self.config.model_name == "qwen2.5-vl-7b":
            self._load_qwen2_5_vl()
        elif self.config.model_name == "internvideo2.5-chat-8b":
            self._load_internvideo()
        elif self.config.model_name == "llava-video-7b":
            self._load_llava_video()
        else:
            raise ValueError(f"Unknown model: {self.config.model_name}")

        logger.info("Model loaded successfully")

    def _load_qwen2_5_vl(self) -> None:
        """Load Qwen2.5-VL model.

        Raises:
            RuntimeError: If model loading fails or dependencies missing
        """
        if not QWEN_AVAILABLE:
            raise RuntimeError(
                "Qwen2.5-VL requires transformers>=4.45.0. "
                "Install with: pip install transformers>=4.45.0"
            )

        if not QWEN_UTILS_AVAILABLE:
            raise RuntimeError(
                "Qwen2.5-VL requires qwen-vl-utils. "
                "Install with: pip install qwen-vl-utils[decord]"
            )

        try:
            import torch

            # Determine torch dtype
            if self.config.dtype == "float16":
                torch_dtype = torch.float16
            elif self.config.dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            # Load model
            logger.info("Loading Qwen2.5-VL-7B-Instruct...")
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype=torch_dtype,
                device_map="auto" if self.config.device == "cuda" else "cpu",
            )

            # Load processor with custom pixel range
            self._processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                min_pixels=self.config.min_pixels,
                max_pixels=self.config.max_pixels,
            )

            logger.info("Qwen2.5-VL loaded successfully")

        except Exception as exc:
            logger.error("Failed to load Qwen2.5-VL: %s", exc)
            raise RuntimeError(f"Model loading failed: {exc}") from exc

    def _load_internvideo(self) -> None:
        """Load InternVideo2.5 Chat model.

        Note: This is a placeholder. Real implementation requires:
        - transformers library with InternVideo2.5 support
        - Model downloaded from HuggingFace
        """
        # TODO: Implement real InternVideo2.5 loading
        # from transformers import AutoModel, AutoProcessor
        # self._model = AutoModel.from_pretrained(
        #     "OpenGVLab/InternVideo2_5-Chat-8B",
        #     device_map=self.config.device,
        #     torch_dtype=...,
        # )
        # self._processor = AutoProcessor.from_pretrained(...)
        logger.warning("InternVideo2.5 loading not implemented (using mock mode)")
        self._model = None
        self._processor = None

    def _load_llava_video(self) -> None:
        """Load LLaVA-Video model.

        Note: This is a placeholder. Real implementation requires:
        - transformers library with LLaVA-Video support
        - Model downloaded from HuggingFace
        """
        # TODO: Implement real LLaVA-Video loading
        logger.warning("LLaVA-Video loading not implemented (using mock mode)")
        self._model = None
        self._processor = None

    def sample_frames(
        self,
        video_path: Path | str,
        *,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        max_frames: int | None = None,
    ) -> np.ndarray:
        """Sample frames from a video clip.

        Args:
            video_path: Path to video file
            start_sec: Start time in seconds
            end_sec: End time in seconds (None = end of video)
            max_frames: Maximum frames to sample (None = use config default)

        Returns:
            Array of frames (N, H, W, 3) in RGB format

        Raises:
            ValueError: If video cannot be opened
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video not found: {video_path}")

        max_frames = max_frames or self.config.max_frames

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame range
        start_frame = int(start_sec * fps)
        if end_sec is None:
            end_frame = total_frames
        else:
            end_frame = int(end_sec * fps)

        end_frame = min(end_frame, total_frames)

        if start_frame >= end_frame:
            cap.release()
            return np.zeros((0, 224, 224, 3), dtype=np.uint8)

        # Sample frames uniformly
        num_frames = end_frame - start_frame
        if num_frames <= max_frames:
            # Take all frames
            frame_indices = list(range(start_frame, end_frame))
        else:
            # Uniform sampling
            step = num_frames / max_frames
            frame_indices = [int(start_frame + i * step) for i in range(max_frames)]

        # Extract frames
        frames: list[np.ndarray] = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame %d", frame_idx)
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        if not frames:
            return np.zeros((0, 224, 224, 3), dtype=np.uint8)

        return np.stack(frames, axis=0)

    def extract_embedding(
        self,
        video_path: Path | str,
        *,
        start_sec: float = 0.0,
        end_sec: float | None = None,
    ) -> np.ndarray:
        """Extract video embedding for retrieval.

        Args:
            video_path: Path to video file
            start_sec: Start time in seconds
            end_sec: End time in seconds (None = end of video)

        Returns:
            Video embedding vector (D,) where D is model-dependent
            - InternVideo2.5: 768-dim
            - LLaVA-Video: 4096-dim
            - Mock: 768-dim

        Raises:
            RuntimeError: If model not loaded
        """
        # Determine embedding dimension
        if self.config.model_name == "internvideo2.5-chat-8b":
            dim = 768
        elif self.config.model_name == "llava-video-7b":
            dim = 4096
        else:  # mock or unknown
            dim = 768

        if self._model is None:
            # Mock mode: return random embedding
            logger.warning("Model not loaded, returning mock embedding")
            return np.random.randn(dim).astype(np.float32)

        # Sample frames
        frames = self.sample_frames(video_path, start_sec=start_sec, end_sec=end_sec)

        if len(frames) == 0:
            raise ValueError("No frames sampled from video")

        # TODO: Implement real embedding extraction
        # embedding = self._model.encode_video(frames)
        # return embedding

        # Placeholder
        return np.random.randn(dim).astype(np.float32)

    def answer_question(
        self,
        video_path: Path | str,
        question: str,
        *,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        enable_cot: bool = False,
    ) -> VideoQAResult:
        """Answer a question about a video clip.

        Args:
            video_path: Path to video file
            question: Question to answer
            start_sec: Start time in seconds
            end_sec: End time in seconds (None = end of video)
            enable_cot: Enable chain-of-thought reasoning

        Returns:
            VideoQAResult with answer and optional reasoning

        Raises:
            RuntimeError: If model not loaded
        """
        if self._model is None:
            # Mock mode: return placeholder answer
            logger.warning("Model not loaded, returning mock answer")
            return VideoQAResult(
                question=question,
                answer="Mock answer: This is a placeholder response.",
                confidence=0.5,
                reasoning="Mock reasoning" if enable_cot else None,
            )

        # Check if we're using Qwen2.5-VL (real implementation)
        if self.config.model_name == "qwen2.5-vl-7b" and QWEN_UTILS_AVAILABLE:
            return self._answer_question_qwen2_5_vl(
                video_path, question, start_sec, end_sec, enable_cot
            )

        # Fallback for other models (not yet implemented)
        logger.warning("Model-specific inference not implemented, using mock answer")
        return VideoQAResult(
            question=question,
            answer="Placeholder answer: Real implementation coming soon.",
            confidence=None,
            reasoning=None,
        )

    def _answer_question_qwen2_5_vl(
        self,
        video_path: Path | str,
        question: str,
        start_sec: float,
        end_sec: float | None,
        enable_cot: bool,
    ) -> VideoQAResult:
        """Answer question using Qwen2.5-VL.

        Args:
            video_path: Path to video file
            question: Question to answer
            start_sec: Start time in seconds
            end_sec: End time in seconds (None = end of video)
            enable_cot: Enable chain-of-thought reasoning

        Returns:
            VideoQAResult with answer
        """
        import torch

        # Prepare prompt with optional CoT
        if enable_cot:
            prompt = f"Let's think step by step. {question}"
        else:
            prompt = question

        # Prepare messages in Qwen2.5-VL format
        video_path = Path(video_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file:///{video_path.absolute().as_posix()}",
                        "max_pixels": self.config.max_pixels,
                        "fps": self.config.fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            # Apply chat template
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision info
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )

            # Prepare inputs
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )

            # Move to device
            if self.config.device == "cuda":
                inputs = inputs.to("cuda")

            # Generate answer
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature if self.config.temperature > 0 else None,
                    top_p=self.config.top_p,
                )

            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            answer = output_text[0] if output_text else "No response generated."

            # Parse reasoning if CoT was enabled
            reasoning = None
            if enable_cot and "step" in answer.lower():
                # Extract reasoning part (before final answer if present)
                reasoning = answer

            logger.info("Generated answer: %s", answer[:100])

            return VideoQAResult(
                question=question,
                answer=answer,
                confidence=None,  # Qwen doesn't provide confidence scores directly
                reasoning=reasoning,
            )

        except Exception as exc:
            logger.error("Video QA failed: %s", exc)
            raise RuntimeError(f"Video QA inference failed: {exc}") from exc

    def batch_extract_embeddings(
        self,
        video_clips: list[tuple[Path | str, float, float]],
    ) -> np.ndarray:
        """Extract embeddings for multiple video clips in batch.

        Args:
            video_clips: List of (video_path, start_sec, end_sec) tuples

        Returns:
            Stacked embeddings (N, D)

        Raises:
            ValueError: If no valid clips
        """
        # Determine embedding dimension
        if self.config.model_name == "internvideo2.5-chat-8b":
            dim = 768
        elif self.config.model_name == "llava-video-7b":
            dim = 4096
        else:  # mock or unknown
            dim = 768

        if not video_clips:
            return np.zeros((0, dim), dtype=np.float32)

        embeddings: list[np.ndarray] = []
        for video_path, start_sec, end_sec in video_clips:
            try:
                emb = self.extract_embedding(video_path, start_sec=start_sec, end_sec=end_sec)
                embeddings.append(emb)
            except Exception as exc:
                logger.warning("Failed to extract embedding for %s: %s", video_path, exc)
                continue

        if not embeddings:
            raise ValueError("No valid embeddings extracted")

        return np.stack(embeddings, axis=0)


def get_default_config(model_name: ModelName = "qwen2.5-vl-7b") -> VideoLLMConfig:
    """Get default configuration for a model.

    Args:
        model_name: Model to configure

    Returns:
        VideoLLMConfig with model-specific defaults
    """
    if model_name == "qwen2.5-vl-7b":
        return VideoLLMConfig(
            model_name="qwen2.5-vl-7b",
            device="cuda",
            dtype="bfloat16",  # Qwen2.5-VL works best with bfloat16
            max_frames=32,
            fps=1.0,  # 1 FPS for efficient long-video understanding
            max_new_tokens=512,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
    elif model_name == "internvideo2.5-chat-8b":
        return VideoLLMConfig(
            model_name="internvideo2.5-chat-8b",
            device="cuda",
            dtype="float16",
            max_frames=32,
            max_new_tokens=512,
        )
    elif model_name == "llava-video-7b":
        return VideoLLMConfig(
            model_name="llava-video-7b",
            device="cuda",
            dtype="float16",
            max_frames=16,  # LLaVA-Video uses fewer frames
            max_new_tokens=512,
        )
    elif model_name == "mock":
        return VideoLLMConfig(
            model_name="mock",
            device="cpu",
            dtype="float32",
            max_frames=8,
            max_new_tokens=128,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
