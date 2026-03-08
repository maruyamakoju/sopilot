"""LLaVA-NeXT VLM Client — visual language model for cascade Stage 3.

Phase 38: Replaces the stub Stage 3 of CascadeDetector with a real
vision-language model (LLaVA-NeXT / llava-v1.6-mistral-7b) that:

    1. Verifies uncertain detections from cascade Stage 2
    2. Provides holistic scene descriptions in Japanese + English
    3. Answers visual questions for the perception API
    4. Annotates SOP violations with natural-language explanations

Integrates with the ModelOrchestrator for VRAM-safe lifecycle:
    orchestrator.register("llava", lambda: ..., vram=8.0, priority=3)

The client implements the VLM interface expected by CascadeDetector:
    client.analyze_detections(frame, detections, prompts) -> list[Detection]

Usage via cascade Stage 3::

    llava = build_llava_client("llava-hf/llava-v1.6-mistral-7b-hf", orchestrator)
    cascade = CascadeDetector(..., vlm_client=llava)

Usage for scene description::

    result = llava.describe_scene(frame, language="ja")
    # {"description_ja": "...", "description_en": "...", ...}

Usage for visual Q&A::

    answer = llava.answer_question(frame, "ヘルメットを着用していない人は何人？")
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from sopilot.perception.types import BBox, Detection

logger = logging.getLogger(__name__)

# Model defaults
_DEFAULT_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
_MODEL_NAME = "llava"
_VRAM_GB = 8.0
_PRIORITY = 3  # on-demand; lower priority than YOLO (9) and Grounding DINO (6)

# Scene description prompts
_DESCRIBE_PROMPT_JA = (
    "この映像を詳しく説明してください。"
    "人物の数、行動、安全装備（ヘルメット、ベスト）、危険なエリアへの侵入を報告してください。"
    "日本語で簡潔に回答してください。"
)

_DESCRIBE_PROMPT_EN = (
    "Describe this image in detail. "
    "Report the number of people, their actions, safety equipment (helmets, vests), "
    "and any entry into dangerous areas. Answer concisely in English."
)

_SYSTEM_PROMPT_JA = (
    "あなたは産業安全監視システムのAIアシスタントです。"
    "カメラ映像を解析して安全違反を検出し、日本語で簡潔に報告してください。"
)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class LLaVAStats:
    """Usage statistics for the LLaVA client."""

    total_calls: int = 0
    describe_calls: int = 0
    analyze_calls: int = 0
    qa_calls: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.total_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "describe_calls": self.describe_calls,
            "analyze_calls": self.analyze_calls,
            "qa_calls": self.qa_calls,
            "errors": self.errors,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class VLMClient(ABC):
    """Abstract base for vision-language model clients."""

    @abstractmethod
    def analyze_detections(
        self,
        frame: np.ndarray,
        uncertain: list[Detection],
        prompts: list[str],
    ) -> list[Detection]:
        """Verify uncertain detections from cascade Stage 2.

        Parameters
        ----------
        frame:
            BGR image array (H, W, 3).
        uncertain:
            Low-confidence detections to verify.
        prompts:
            Natural-language class descriptions from the cascade.

        Returns
        -------
        list[Detection]
            Verified/adjusted detections (may drop or boost confidence).
        """
        ...

    @abstractmethod
    def describe_scene(
        self,
        frame: np.ndarray,
        language: str = "ja",
    ) -> dict[str, Any]:
        """Generate a holistic scene description.

        Parameters
        ----------
        frame:
            BGR image array.
        language:
            "ja" (Japanese), "en" (English), or "both".

        Returns
        -------
        dict
            Keys: ``description_ja``, ``description_en``, ``objects``, etc.
        """
        ...

    @abstractmethod
    def answer_question(
        self,
        frame: np.ndarray,
        question: str,
    ) -> str:
        """Answer a visual question about the frame.

        Parameters
        ----------
        frame:
            BGR image array.
        question:
            Natural language question (Japanese or English).

        Returns
        -------
        str
            Answer in the same language as the question.
        """
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Return usage statistics."""
        ...


# ---------------------------------------------------------------------------
# LLaVA-NeXT client (HuggingFace transformers)
# ---------------------------------------------------------------------------


class LLaVAClient(VLMClient):
    """LLaVA-NeXT vision-language model client.

    Uses HuggingFace ``transformers`` + ``LlavaNextProcessor`` to run
    LLaVA-NeXT (llava-v1.6-mistral-7b or llava-v1.6-vicuna-7b) locally
    on RTX 4090 in 8-bit quantized mode (~8 GB VRAM).

    Integrates with :class:`ModelOrchestrator` for VRAM lifecycle:
    the underlying model is loaded lazily on first call and can be
    evicted when VRAM pressure is high (priority=3).

    Parameters
    ----------
    model_id:
        HuggingFace model ID (default: ``"llava-hf/llava-v1.6-mistral-7b-hf"``).
    orchestrator:
        :class:`ModelOrchestrator` for VRAM-aware lifecycle. Optional.
    device:
        Torch device (``"auto"``, ``"cuda"``, ``"cpu"``).
    max_new_tokens:
        Maximum tokens to generate per response.
    load_in_8bit:
        Whether to load model in 8-bit quantization (saves ~50% VRAM).
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        orchestrator: Any = None,
        device: str = "auto",
        max_new_tokens: int = 256,
        load_in_8bit: bool = True,
    ) -> None:
        self._model_id = model_id
        self._orchestrator = orchestrator
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._load_in_8bit = load_in_8bit
        self._stats = LLaVAStats()

        # Lazy-loaded components
        self._processor: Any = None
        self._model: Any = None
        self._loaded = False

        # Register with orchestrator
        if orchestrator is not None:
            try:
                orchestrator.register(
                    _MODEL_NAME,
                    self._load_model,
                    vram_gb=_VRAM_GB,
                    priority=_PRIORITY,
                    description=f"LLaVA-NeXT ({model_id})",
                )
                logger.info(
                    "LLaVAClient registered with orchestrator (vram=%.1f GB, priority=%d)",
                    _VRAM_GB,
                    _PRIORITY,
                )
            except Exception:
                logger.debug("Failed to register LLaVA with orchestrator", exc_info=True)

        logger.info(
            "LLaVAClient created (model_id=%s, 8bit=%s, max_tokens=%d)",
            model_id,
            load_in_8bit,
            max_new_tokens,
        )

    # ── Model lifecycle ────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load model and processor (called lazily on first use)."""
        if self._loaded:
            return

        import torch
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        logger.info("Loading LLaVA model %s ...", self._model_id)

        self._processor = LlavaNextProcessor.from_pretrained(self._model_id)

        load_kwargs: dict[str, Any] = {"device_map": "auto"}
        if self._load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            self._model_id,
            **load_kwargs,
        )
        self._model.eval()

        self._loaded = True
        logger.info("LLaVA model loaded successfully on %s", self._device)

    def _ensure_loaded(self) -> bool:
        """Ensure model is loaded. Returns True if ready."""
        if not self._loaded:
            try:
                self._load_model()
            except Exception:
                logger.debug("LLaVA model load failed", exc_info=True)
                return False
        return self._loaded

    def _frame_to_pil(self, frame: np.ndarray):
        """Convert BGR numpy frame to PIL Image."""
        import cv2
        from PIL import Image as PILImage

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb)

    def _generate(self, image: Any, prompt: str) -> str:
        """Run VLM inference and return generated text."""
        import torch

        inputs = self._processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        # Decode only the generated portion (not the prompt)
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()

    # ── VLMClient interface ────────────────────────────────────────────

    def analyze_detections(
        self,
        frame: np.ndarray,
        uncertain: list[Detection],
        prompts: list[str],
    ) -> list[Detection]:
        """Stage 3 cascade: verify uncertain detections with LLaVA-NeXT."""
        if not uncertain or frame is None or frame.size == 0:
            return []

        t0 = time.monotonic()
        self._stats.analyze_calls += 1
        self._stats.total_calls += 1

        if not self._ensure_loaded():
            self._stats.errors += 1
            return [
                Detection(
                    bbox=d.bbox,
                    label=d.label,
                    confidence=d.confidence * 0.7,
                    embedding=d.embedding,
                    attributes={**d.attributes, "vlm_verified": False, "vlm_error": True},
                )
                for d in uncertain
            ]

        try:
            image = self._frame_to_pil(frame)
            detection_strs = [
                f"{d.label} (conf={d.confidence:.2f})" for d in uncertain
            ]
            prompt = (
                f"[INST] <image>\n"
                f"Verify these detections: {', '.join(detection_strs)}.\n"
                f"Looking for: {', '.join(prompts)}.\n"
                f"For each, respond: <label>: CONFIRMED or REJECTED\n"
                f"[/INST]"
            )

            response = self._generate(image, prompt)
            logger.debug("LLaVA verification: %s", response[:200])

            # Parse response to confirm/reject each detection
            verified: list[Detection] = []
            for det in uncertain:
                label_lower = det.label.lower()
                confirmed = True
                confidence_adjust = 0.9

                for line in response.lower().split("\n"):
                    if label_lower in line:
                        if "rejected" in line or "not " in line:
                            confirmed = False
                        elif "confirmed" in line:
                            confirmed = True
                            confidence_adjust = 1.1
                        break

                if confirmed:
                    verified.append(
                        Detection(
                            bbox=det.bbox,
                            label=det.label,
                            confidence=min(1.0, det.confidence * confidence_adjust),
                            embedding=det.embedding,
                            attributes={**det.attributes, "vlm_verified": True},
                        )
                    )

            elapsed = (time.monotonic() - t0) * 1000
            self._stats.total_latency_ms += elapsed
            logger.debug(
                "LLaVA analyze_detections: %d → %d verified (%.1f ms)",
                len(uncertain),
                len(verified),
                elapsed,
            )
            return verified

        except Exception:
            self._stats.errors += 1
            logger.debug("LLaVA analyze_detections failed", exc_info=True)
            return uncertain  # return as-is on failure

    def analyze_clip(
        self,
        frames: list[np.ndarray],
        prompt_ja: str = "これらはビデオから抽出された一連のフレームです。この映像で何が起きたか、特に異常やルールの違反があれば詳細に説明してください。",
    ) -> str:
        """Analyze a video clip (sequence of frames) by creating a grid image."""
        if not frames:
            return "No frames provided."
            
        t0 = time.monotonic()
        self._stats.total_calls += 1
        
        if not self._ensure_loaded():
            self._stats.errors += 1
            return "VLM not loaded."
            
        try:
            import cv2
            import math
            
            # Create a grid image from the frames
            n = len(frames)
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            
            h, w = frames[0].shape[:2]
            target_w = 320
            target_h = int(h * (target_w / w))
            
            grid_img = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)
            
            for i, f in enumerate(frames):
                r = i // cols
                c = i % cols
                resized = cv2.resize(f, (target_w, target_h))
                grid_img[r*target_h:(r+1)*target_h, c*target_w:(c+1)*target_w] = resized
                
            image = self._frame_to_pil(grid_img)
            result = self._generate(
                image, f"[INST] <image>\n{prompt_ja}\n[/INST]"
            )
            
            elapsed = (time.monotonic() - t0) * 1000
            self._stats.total_latency_ms += elapsed
            return result
        except Exception:
            self._stats.errors += 1
            logger.debug("LLaVA analyze_clip failed", exc_info=True)
            return "Failed to analyze clip."

    def describe_scene(
        self,
        frame: np.ndarray,
        language: str = "ja",
    ) -> dict[str, Any]:
        """Generate a holistic scene description in Japanese/English/both."""
        if frame is None or frame.size == 0:
            return {"error": "empty frame"}

        t0 = time.monotonic()
        self._stats.describe_calls += 1
        self._stats.total_calls += 1

        if not self._ensure_loaded():
            self._stats.errors += 1
            return {
                "description_ja": "モデル未ロード",
                "description_en": "Model not loaded",
                "objects": [],
                "person_count": 0,
                "safety_issues": [],
            }

        try:
            image = self._frame_to_pil(frame)
            results: dict[str, Any] = {"objects": [], "safety_issues": []}

            if language in ("ja", "both"):
                results["description_ja"] = self._generate(
                    image, f"[INST] <image>\n{_DESCRIBE_PROMPT_JA}\n[/INST]"
                )
            else:
                results["description_ja"] = ""

            if language in ("en", "both"):
                results["description_en"] = self._generate(
                    image, f"[INST] <image>\n{_DESCRIBE_PROMPT_EN}\n[/INST]"
                )
            else:
                results["description_en"] = ""

            elapsed = (time.monotonic() - t0) * 1000
            self._stats.total_latency_ms += elapsed
            results["latency_ms"] = round(elapsed, 1)
            results["language"] = language
            return results

        except Exception:
            self._stats.errors += 1
            logger.debug("LLaVA describe_scene failed", exc_info=True)
            return {
                "description_ja": "解析エラー",
                "description_en": "Analysis error",
                "objects": [],
                "person_count": 0,
                "safety_issues": [],
            }

    def answer_question(
        self,
        frame: np.ndarray,
        question: str,
    ) -> str:
        """Answer a visual question about the frame."""
        if frame is None or frame.size == 0 or not question:
            return ""

        t0 = time.monotonic()
        self._stats.qa_calls += 1
        self._stats.total_calls += 1

        if not self._ensure_loaded():
            self._stats.errors += 1
            return "モデル未ロード / Model not loaded"

        try:
            image = self._frame_to_pil(frame)
            prompt = f"[INST] <image>\n{question}\n[/INST]"
            answer = self._generate(image, prompt)
            elapsed = (time.monotonic() - t0) * 1000
            self._stats.total_latency_ms += elapsed
            return answer

        except Exception:
            self._stats.errors += 1
            logger.debug("LLaVA answer_question failed", exc_info=True)
            return "解析エラー / Analysis error"

    def get_stats(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "loaded": self._loaded,
            "load_in_8bit": self._load_in_8bit,
            "max_new_tokens": self._max_new_tokens,
            **self._stats.to_dict(),
        }

    def close(self) -> None:
        """Release model resources and free VRAM."""
        if self._loaded:
            try:
                import torch

                del self._model
                del self._processor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._model = None
                self._processor = None
                self._loaded = False
                logger.info("LLaVA model released")
            except Exception:
                logger.debug("LLaVA close failed", exc_info=True)


# ---------------------------------------------------------------------------
# Mock VLM client (testing / CPU-only environments)
# ---------------------------------------------------------------------------


class MockLLaVAClient(VLMClient):
    """Mock LLaVA client for testing without GPU.

    Returns deterministic, plausible responses based on detection labels.
    Thread-safe: all state uses only atomic counters.

    Parameters
    ----------
    confirm_all:
        If True (default), confirms all uncertain detections.
        If False, rejects all (for testing rejection paths).
    confidence_boost:
        Confidence delta applied to confirmed detections.
    latency_ms:
        Simulated inference latency for stat tracking.
    """

    def __init__(
        self,
        confirm_all: bool = True,
        confidence_boost: float = 0.05,
        latency_ms: float = 50.0,
    ) -> None:
        self._confirm_all = confirm_all
        self._confidence_boost = confidence_boost
        self._latency_ms = latency_ms
        self._stats = LLaVAStats()

    def analyze_detections(
        self,
        frame: np.ndarray,
        uncertain: list[Detection],
        prompts: list[str],
    ) -> list[Detection]:
        self._stats.analyze_calls += 1
        self._stats.total_calls += 1
        self._stats.total_latency_ms += self._latency_ms

        if not self._confirm_all:
            return []

        return [
            Detection(
                bbox=d.bbox,
                label=d.label,
                confidence=min(1.0, d.confidence + self._confidence_boost),
                embedding=d.embedding,
                attributes={**d.attributes, "vlm_verified": True, "vlm_mock": True},
            )
            for d in uncertain
        ]

    def describe_scene(
        self,
        frame: np.ndarray,
        language: str = "ja",
    ) -> dict[str, Any]:
        self._stats.describe_calls += 1
        self._stats.total_calls += 1
        self._stats.total_latency_ms += self._latency_ms

        h, w = frame.shape[:2] if frame is not None and frame.size > 0 else (480, 640)
        return {
            "description_ja": (
                f"作業エリア内に人物が確認されました。"
                f"フレームサイズ: {w}×{h}。安全装備を確認してください。"
            ),
            "description_en": (
                f"Personnel detected in work area. Frame: {w}×{h}. "
                f"Check safety equipment compliance."
            ),
            "objects": ["person", "work_area"],
            "person_count": 1,
            "safety_issues": [],
            "latency_ms": self._latency_ms,
            "language": language,
            "mock": True,
        }

    def answer_question(
        self,
        frame: np.ndarray,
        question: str,
    ) -> str:
        self._stats.qa_calls += 1
        self._stats.total_calls += 1
        self._stats.total_latency_ms += self._latency_ms

        # Produce contextually appropriate mock answers
        if "何人" in question or "how many" in question.lower():
            return "1名確認されました。 / 1 person detected."
        if "ヘルメット" in question or "helmet" in question.lower():
            return "ヘルメット着用状況を確認中です。 / Checking helmet compliance."
        if "違反" in question or "violation" in question.lower():
            return "現在の映像では安全違反は確認されていません。 / No violations detected."
        if "ベスト" in question or "vest" in question.lower():
            return "安全ベストを着用しています。 / Safety vest detected."
        return "解析完了。詳細はシステムログを確認してください。 / Analysis complete."

    def get_stats(self) -> dict[str, Any]:
        return {
            "model_id": "mock",
            "loaded": True,
            "mock": True,
            "confirm_all": self._confirm_all,
            **self._stats.to_dict(),
        }

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_llava_client(
    model_id: str = _DEFAULT_MODEL_ID,
    orchestrator: Any = None,
    device: str = "auto",
    max_new_tokens: int = 256,
    backend: str = "auto",
) -> VLMClient:
    """Build a LLaVA VLM client.

    Parameters
    ----------
    model_id:
        HuggingFace model ID.
    orchestrator:
        Optional :class:`ModelOrchestrator`.
    device:
        Torch device.
    max_new_tokens:
        Max generation tokens.
    backend:
        ``"llava"`` — real LLaVA-NeXT model.
        ``"mock"`` — MockLLaVAClient (no GPU required).
        ``"auto"`` — tries LLaVA, falls back to mock if CUDA unavailable.

    Returns
    -------
    VLMClient
        :class:`LLaVAClient` or :class:`MockLLaVAClient`.
    """
    if backend == "mock":
        logger.info("build_llava_client: using MockLLaVAClient")
        return MockLLaVAClient()

    if backend == "auto":
        try:
            import torch

            if not torch.cuda.is_available():
                logger.info("build_llava_client: CUDA not available → MockLLaVAClient")
                return MockLLaVAClient()
        except ImportError:
            logger.info("build_llava_client: torch not available → MockLLaVAClient")
            return MockLLaVAClient()

    return LLaVAClient(
        model_id=model_id,
        orchestrator=orchestrator,
        device=device,
        max_new_tokens=max_new_tokens,
    )
