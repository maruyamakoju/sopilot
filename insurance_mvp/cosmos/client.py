"""Production-ready Video-LLM client for insurance claim assessment.

Supports:
- NVIDIA Cosmos Reason 2 (when available)
- Qwen2.5-VL-7B-Instruct (fallback)
- Mock mode (for testing)

Features:
- Robust JSON parsing with 7-step repair pipeline
- Model caching (don't reload every inference)
- GPU memory management
- Timeout handling (1200 seconds)
- Graceful degradation on failures
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import cv2
from pydantic import ValidationError

from .prompt import get_claim_assessment_prompt, get_system_prompt
from .schema import ClaimAssessment, create_default_claim_assessment

# Optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from qwen_vl_utils import process_vision_info

    QWEN_UTILS_AVAILABLE = True
except ImportError:
    QWEN_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)

ModelName = Literal["nvidia-cosmos-reason-2", "qwen2.5-vl-7b", "mock"]


class VLMConfig:
    """Configuration for Video-LLM client."""

    def __init__(
        self,
        model_name: ModelName = "qwen2.5-vl-7b",
        device: str = "cuda",
        dtype: str = "bfloat16",
        fps: float = 2.0,
        max_frames: int = 48,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        timeout_sec: float = 300.0,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 512 * 28 * 28,
        quantize: str | None = None,
    ):
        """Initialize VLM configuration.

        Args:
            model_name: Model to use (nvidia-cosmos-reason-2, qwen2.5-vl-7b, mock)
            device: Device to run on (cuda, cpu)
            dtype: Data type (float16, bfloat16, float32)
            fps: Frame sampling rate (frames per second)
            max_frames: Maximum frames to sample per video clip
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more conservative)
            timeout_sec: Inference timeout in seconds
            min_pixels: Minimum pixels for dynamic resolution (Qwen only)
            max_pixels: Maximum pixels for dynamic resolution (Qwen only)
            quantize: Quantization mode (None, "int4", "int8"). Requires bitsandbytes.
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.fps = fps
        self.max_frames = max_frames
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.quantize = quantize


class VideoLLMClient:
    """Production-ready Video-LLM client for insurance claim assessment."""

    # Class-level cache for models (singleton pattern)
    _model_cache: dict[str, tuple[Any, Any]] = {}

    def __init__(self, config: VLMConfig):
        """Initialize Video-LLM client.

        Args:
            config: VLM configuration

        Raises:
            RuntimeError: If required dependencies are missing
        """
        self.config = config
        self._model = None
        self._processor = None

        if config.model_name != "mock":
            self._ensure_model_loaded()

    def health_check(self) -> dict:
        """Check model readiness and return status.

        Returns:
            Dict with status, model_name, model_loaded, and backend info.
        """
        import os

        backend = os.getenv("INSURANCE_VLM_BACKEND", self.config.model_name)
        result = {
            "status": "ok" if self.config.model_name == "mock" or self._model is not None else "not_loaded",
            "model_name": self.config.model_name,
            "model_loaded": self._model is not None,
            "backend": backend,
            "device": self.config.device,
            "is_mock": self.config.model_name == "mock",
        }

        if self._model is not None:
            result["status"] = "ok"
            cache_key = f"{self.config.model_name}_{self.config.device}_{self.config.dtype}"
            result["cached"] = cache_key in self._model_cache

        logger.info("Health check: %s", result["status"])
        return result

    def _ensure_model_loaded(self) -> None:
        """Load model and processor (with caching).

        Raises:
            RuntimeError: If model loading fails
        """
        cache_key = f"{self.config.model_name}_{self.config.device}_{self.config.dtype}"

        # Check cache first
        if cache_key in self._model_cache:
            logger.info("Using cached model: %s", self.config.model_name)
            self._model, self._processor = self._model_cache[cache_key]
            return

        # Load model
        logger.info("Loading Video-LLM model: %s", self.config.model_name)
        start_time = time.time()

        try:
            if self.config.model_name == "nvidia-cosmos-reason-2":
                self._load_cosmos()
            elif self.config.model_name == "qwen2.5-vl-7b":
                self._load_qwen2_5_vl()
            else:
                raise ValueError(f"Unknown model: {self.config.model_name}")

            # Cache the loaded model
            self._model_cache[cache_key] = (self._model, self._processor)

            elapsed = time.time() - start_time
            logger.info("Model loaded successfully in %.2f seconds", elapsed)

        except Exception as exc:
            logger.error("Failed to load model %s: %s", self.config.model_name, exc)
            raise RuntimeError(f"Model loading failed: {exc}") from exc

    def _load_cosmos(self) -> None:
        """Load NVIDIA Cosmos Reason 2 model.

        Raises:
            RuntimeError: Not yet implemented
        """
        raise RuntimeError(
            "NVIDIA Cosmos Reason 2 support not yet implemented. "
            "Use 'qwen2.5-vl-7b' or 'mock' instead. "
            "Coming soon after NVIDIA releases the model."
        )

    def _load_qwen2_5_vl(self) -> None:
        """Load Qwen2.5-VL-7B model.

        Raises:
            RuntimeError: If dependencies missing or loading fails
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "Qwen2.5-VL requires transformers>=4.45.0. Install with: pip install transformers>=4.45.0"
            )

        if not QWEN_UTILS_AVAILABLE:
            raise RuntimeError("Qwen2.5-VL requires qwen-vl-utils. Install with: pip install qwen-vl-utils")

        if not TORCH_AVAILABLE:
            raise RuntimeError("Qwen2.5-VL requires torch. Install with: pip install torch")

        try:
            # Determine torch dtype
            if self.config.dtype == "float16":
                torch_dtype = torch.float16
            elif self.config.dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            # Build quantization config if requested
            quantization_config = None
            if self.config.quantize in ("int4", "int8"):
                try:
                    from transformers import BitsAndBytesConfig

                    if self.config.quantize == "int4":
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch_dtype,
                            bnb_4bit_quant_type="nf4",
                        )
                        logger.info("Using int4 quantization (NF4)")
                    else:
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        logger.info("Using int8 quantization")
                except ImportError:
                    logger.warning("bitsandbytes not installed, skipping quantization")

            # Load model
            logger.info("Loading Qwen2.5-VL-7B-Instruct...")
            load_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": "auto" if self.config.device == "cuda" else "cpu",
            }
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config

            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                **load_kwargs,
            )

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                min_pixels=self.config.min_pixels,
                max_pixels=self.config.max_pixels,
            )

            logger.info("Qwen2.5-VL loaded successfully")

        except Exception as exc:
            raise RuntimeError(f"Qwen2.5-VL loading failed: {exc}") from exc

    def _sample_frames(
        self, video_path: Path | str, start_sec: float = 0.0, end_sec: float | None = None
    ) -> list[Path]:
        """Sample frames from video clip at configured FPS.

        Args:
            video_path: Path to video file
            start_sec: Start time in seconds
            end_sec: End time in seconds (None = entire video)

        Returns:
            List of paths to sampled frame images (saved in temp directory)

        Raises:
            ValueError: If video cannot be opened or no frames extracted
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video not found: {video_path}")

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
            raise ValueError(f"Invalid frame range: [{start_frame}, {end_frame})")

        # Calculate sampling interval
        duration_sec = (end_frame - start_frame) / fps
        target_frames = min(self.config.max_frames, max(8, int(duration_sec * self.config.fps)))

        # Uniform sampling
        num_frames = end_frame - start_frame
        if num_frames <= target_frames:
            frame_indices = list(range(start_frame, end_frame))
        else:
            step = num_frames / target_frames
            frame_indices = [int(start_frame + i * step) for i in range(target_frames)]

        # Create temp directory for frames
        temp_dir = Path(tempfile.mkdtemp(prefix="vlm_frames_"))
        frame_paths: list[Path] = []

        try:
            for idx, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame %d", frame_idx)
                    continue

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Save frame
                frame_path = temp_dir / f"frame_{idx:04d}.jpg"
                from PIL import Image

                Image.fromarray(frame_rgb).save(frame_path, quality=95)
                frame_paths.append(frame_path)

            cap.release()

            if not frame_paths:
                raise ValueError("No frames extracted from video clip")

            logger.info(
                "Sampled %d frames from [%.2f-%.2f sec] at %.1f FPS",
                len(frame_paths),
                start_sec,
                end_sec or (total_frames / fps),
                self.config.fps,
            )

            return frame_paths

        except Exception:
            # Cleanup on error
            cap.release()
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _run_inference(self, frame_paths: list[Path], prompt: str) -> str:
        """Run Video-LLM inference on sampled frames.

        Args:
            frame_paths: List of paths to sampled frames
            prompt: Text prompt for the model

        Returns:
            Raw model output (JSON string)

        Raises:
            TimeoutError: If inference exceeds timeout
            RuntimeError: If inference fails
        """
        if self.config.model_name == "mock":
            return self._mock_inference(prompt)

        if self.config.model_name == "qwen2.5-vl-7b":
            return self._run_qwen2_5_vl_inference(frame_paths, prompt)

        raise RuntimeError(f"Inference not implemented for {self.config.model_name}")

    def _mock_inference(self, prompt: str) -> str:
        """Generate smart mock inference output with realistic variation.

        Args:
            prompt: Input prompt (analyzed for keywords to generate context-aware response)

        Returns:
            Realistic JSON response based on prompt analysis
        """
        import hashlib
        import random

        # Analyze prompt for scenario hints
        prompt_lower = prompt.lower()

        # Determine scenario from keywords in prompt
        is_collision = any(word in prompt_lower for word in ["collision", "crash", "impact", "hit"])
        is_near_miss = any(word in prompt_lower for word in ["near-miss", "near miss", "avoid", "brake"])
        is_pedestrian = any(word in prompt_lower for word in ["pedestrian", "person", "walker"])
        has_danger = any(word in prompt_lower for word in ["danger", "hazard", "risk", "emergency"])

        # Use prompt hash for deterministic randomness (same video = same output)
        seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Scenario-based severity and reasoning
        if is_collision and not is_near_miss:
            severity = rng.choice(["HIGH", "HIGH", "MEDIUM"])  # Bias toward HIGH
            confidence = rng.uniform(0.82, 0.94)
            prediction_set = ["HIGH", "MEDIUM"] if severity == "HIGH" else ["MEDIUM", "HIGH", "LOW"]
            review_priority = "URGENT"

            causal_reasoning = rng.choice(
                [
                    "Video analysis reveals rear-end collision scenario. The dashcam footage shows the ego vehicle approaching a slowing vehicle ahead. At approximately 18-20 seconds, brake lights are visible on the lead vehicle, followed by emergency braking. Impact occurs at the 20-second mark with visible forward jolt. The collision appears to be caused by insufficient following distance combined with delayed reaction time.",
                    "Analysis of the video indicates a high-severity rear-end collision. The footage clearly shows the lead vehicle's brake lights activating around the 15-18 second mark. Despite this warning, the ego vehicle continues at speed until the last moment, resulting in significant impact force. Weather and road conditions appear clear, suggesting the collision was preventable with proper attention and following distance.",
                    "The video evidence demonstrates a classic rear-end collision scenario. Frame-by-frame analysis shows: (1) Lead vehicle begins decelerating at 15s mark, (2) Brake lights clearly visible from 16-20s, (3) Ego vehicle maintains speed until 19s, (4) Emergency braking initiated too late at 19.5s, (5) Impact at 20s with substantial force. The primary causal factor appears to be inadequate following distance and possible driver distraction.",
                ]
            )

            recommended_action = "REVIEW"

        elif is_near_miss or (has_danger and is_pedestrian):
            severity = rng.choice(["MEDIUM", "MEDIUM", "LOW"])  # Bias toward MEDIUM
            confidence = rng.uniform(0.75, 0.88)
            prediction_set = ["MEDIUM", "LOW"] if severity == "MEDIUM" else ["LOW", "MEDIUM", "NONE"]
            review_priority = "STANDARD"

            causal_reasoning = rng.choice(
                [
                    "The dashcam footage captures a near-miss incident involving a pedestrian. At the 14-15 second mark, a pedestrian enters the vehicle's path from the right side. The driver demonstrates appropriate defensive driving by applying emergency brakes, bringing the vehicle to a stop approximately 2-3 meters before the pedestrian's position. While no collision occurred, the incident represents a moderate hazard that warrants documentation.",
                    "Video analysis shows a pedestrian avoidance scenario. The footage indicates good situational awareness by the driver, who identified the pedestrian hazard early and responded with controlled braking. Deceleration begins at 14s, with the vehicle coming to a complete stop well before any collision risk. This represents effective defensive driving in response to an unexpected pedestrian crossing.",
                    "The video demonstrates successful hazard avoidance involving a pedestrian crossing. Analysis reveals: (1) Pedestrian enters frame at 13s, (2) Driver initiates braking response at 14s, (3) Vehicle decelerates smoothly from approximately 45 km/h to full stop, (4) Final distance to pedestrian approximately 3 meters. The incident classification is near-miss with no fault attributed to the driver.",
                ]
            )

            recommended_action = "DOCUMENT"

        else:  # Normal driving
            severity = rng.choice(["NONE", "NONE", "LOW"])  # Bias toward NONE
            confidence = rng.uniform(0.88, 0.96)
            prediction_set = ["NONE"] if severity == "NONE" else ["LOW", "NONE"]
            review_priority = "LOW"

            causal_reasoning = rng.choice(
                [
                    "The dashcam footage shows standard highway driving with no incidents or violations observed. The vehicle maintains consistent speed, appropriate lane position, and safe following distances throughout the recorded period. Traffic conditions are normal with moderate density. No hazardous situations, sudden maneuvers, or safety concerns are evident in the video. This represents routine, safe driving behavior.",
                    "Analysis of the video reveals normal driving conditions with no noteworthy events. The driver maintains steady speed appropriate for highway conditions, executes smooth lane changes when necessary, and demonstrates proper following distance. No aggressive driving, traffic violations, or hazardous situations are detected. The footage is consistent with standard safe driving practices.",
                    "The video documentation shows routine highway travel. Frame analysis indicates: (1) Consistent speed of 80-90 km/h appropriate for highway, (2) Proper lane discipline maintained, (3) Safe following distances observed, (4) No sudden braking or evasive maneuvers, (5) Clear weather and good visibility throughout. No incidents or concerns identified.",
                ]
            )

            recommended_action = "APPROVE"

        # Generate scenario-appropriate hazards
        hazards = []
        if is_collision:
            hazards = [
                {
                    "type": "collision",
                    "actors": ["insured_vehicle", "lead_vehicle"],
                    "spatial_relation": "front",
                    "timestamp_sec": rng.uniform(19.0, 21.0),
                }
            ]
        elif is_near_miss:
            hazards = [
                {
                    "type": "near_miss",
                    "actors": ["insured_vehicle", "pedestrian"],
                    "spatial_relation": "front",
                    "timestamp_sec": rng.uniform(14.0, 16.0),
                }
            ]

        return json.dumps(
            {
                "severity": severity,
                "confidence": round(confidence, 2),
                "prediction_set": prediction_set,
                "review_priority": review_priority,
                "fault_assessment": {
                    "fault_ratio": 50.0,  # Will be overridden by FaultAssessmentEngine
                    "reasoning": "VLM analysis complete - see causal reasoning",
                    "applicable_rules": [],
                    "scenario_type": "auto_detected",
                    "traffic_signal": None,
                    "right_of_way": None,
                },
                "fraud_risk": {
                    "risk_score": 0.0,  # Will be overridden by FraudDetectionEngine
                    "indicators": [],
                    "reasoning": "Video evidence consistent with described scenario",
                },
                "hazards": hazards,
                "evidence": [],
                "causal_reasoning": causal_reasoning,
                "recommended_action": recommended_action,
            }
        )

    def _run_qwen2_5_vl_inference(self, frame_paths: list[Path], prompt: str) -> str:
        """Run Qwen2.5-VL inference.

        Args:
            frame_paths: List of paths to sampled frames
            prompt: Text prompt

        Returns:
            Raw model output

        Raises:
            TimeoutError: If inference exceeds timeout
            RuntimeError: If inference fails
        """
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Inference exceeded {self.config.timeout_sec} seconds")

        # Set timeout (Unix only - Windows will skip)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config.timeout_sec))
        except AttributeError:
            # Windows doesn't have SIGALRM
            logger.warning("Timeout not supported on Windows - proceeding without timeout")

        try:
            # Convert Path objects to strings
            frame_strs = [str(p.absolute()) for p in frame_paths]

            # Prepare messages with system prompt for better role grounding
            system_prompt = get_system_prompt()
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frame_strs,
                            "max_pixels": self.config.max_pixels,
                            "fps": self.config.fps,
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            # Apply chat template
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Process vision info
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            # Fix video_kwargs: unwrap single-element lists
            fixed_kwargs = {}
            for k, v in video_kwargs.items():
                if isinstance(v, list) and len(v) == 1:
                    fixed_kwargs[k] = v[0]
                else:
                    fixed_kwargs[k] = v

            # Prepare inputs
            inputs = self._processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", **fixed_kwargs
            )

            # Move to device
            if self.config.device == "cuda":
                inputs = inputs.to("cuda")

            # Generate
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature if self.config.temperature > 0 else None,
                    do_sample=self.config.temperature > 0,
                    repetition_penalty=1.1,
                )

            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            result = output_text[0] if output_text else ""
            logger.info("Generated output length: %d chars", len(result))

            return result

        except TimeoutError:
            logger.error("Inference timeout after %.1f seconds", self.config.timeout_sec)
            raise

        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            raise RuntimeError(f"Qwen2.5-VL inference failed: {exc}") from exc

        finally:
            # Cancel timeout
            with contextlib.suppress(AttributeError):
                signal.alarm(0)

    def _parse_json_response(self, raw_output: str, video_id: str, processing_time: float) -> ClaimAssessment:
        """Parse Video-LLM JSON output with 7-step repair pipeline.

        Pipeline steps:
        1. Direct JSON parse
        2. Remove markdown fences (```json ... ```)
        3. Truncation repair (add closing braces)
        4. Brace extraction (find first { to last })
        5. Missing comma insertion (between fields)
        6. Orphaned key removal (keys without values)
        7. Markdown field parser (extract individual fields)

        Args:
            raw_output: Raw model output text
            video_id: Video ID for metadata
            processing_time: Processing time in seconds

        Returns:
            ClaimAssessment object (with default fallback on total failure)
        """
        logger.info("Raw VLM output (%d chars): %s", len(raw_output), raw_output[:1000])

        # Step 1: Direct parse
        try:
            data = json.loads(raw_output)
            return self._validate_and_construct(data, video_id, processing_time)
        except json.JSONDecodeError:
            pass

        # Step 2: Remove markdown fences
        cleaned = raw_output.strip()
        if "```" in cleaned:
            # Remove ```json ... ``` blocks
            cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
            cleaned = cleaned.strip()
            try:
                data = json.loads(cleaned)
                return self._validate_and_construct(data, video_id, processing_time)
            except json.JSONDecodeError:
                pass

        # Step 3: Truncation repair (add missing closing braces)
        if cleaned.count("{") > cleaned.count("}"):
            repaired = cleaned + ("}" * (cleaned.count("{") - cleaned.count("}")))
            try:
                data = json.loads(repaired)
                return self._validate_and_construct(data, video_id, processing_time)
            except json.JSONDecodeError:
                pass

        # Step 4: Brace extraction (find JSON boundaries)
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            extracted = cleaned[start:end]
            try:
                data = json.loads(extracted)
                return self._validate_and_construct(data, video_id, processing_time)
            except json.JSONDecodeError:
                pass

        # Step 5: Missing comma insertion (heuristic: newline between fields)
        if start >= 0 and end > start:
            extracted = cleaned[start:end]
            # Insert commas before newlines followed by field names
            comma_fixed = re.sub(r'(["}])\s*\n\s*"', r'\1,\n"', extracted)
            try:
                data = json.loads(comma_fixed)
                return self._validate_and_construct(data, video_id, processing_time)
            except json.JSONDecodeError:
                pass

        # Step 6: Orphaned key removal (remove incomplete fields at end)
        if start >= 0 and end > start:
            # Remove trailing incomplete fields like `"key":` or `"key": `
            truncated = re.sub(r',\s*"[^"]+"\s*:\s*$', "", extracted.rstrip("}"))
            truncated += "}"
            try:
                data = json.loads(truncated)
                return self._validate_and_construct(data, video_id, processing_time)
            except json.JSONDecodeError:
                pass

        # Step 7: Markdown field parser (last resort - extract individual fields)
        logger.warning("All JSON parse attempts failed, using field extraction")
        data = self._extract_fields_from_text(raw_output)
        return self._validate_and_construct(data, video_id, processing_time)

    def _extract_fields_from_text(self, text: str) -> dict[str, Any]:
        """Extract structured fields from free-form text (last resort).

        Args:
            text: Raw text output

        Returns:
            Dictionary with extracted fields (best effort)
        """
        data: dict[str, Any] = {}

        # Extract severity
        severity_match = re.search(r'"severity"\s*:\s*"(NONE|LOW|MEDIUM|HIGH)"', text, re.IGNORECASE)
        if severity_match:
            data["severity"] = severity_match.group(1).upper()

        # Extract confidence
        confidence_match = re.search(r'"confidence"\s*:\s*(0?\.\d+|1\.0)', text)
        if confidence_match:
            data["confidence"] = float(confidence_match.group(1))

        # Extract fault_ratio
        fault_match = re.search(r'"fault_ratio"\s*:\s*(\d+\.?\d*)', text)
        if fault_match:
            data.setdefault("fault_assessment", {})["fault_ratio"] = float(fault_match.group(1))

        # Extract reasoning (first quoted string after "reasoning")
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
        if reasoning_match:
            data["causal_reasoning"] = reasoning_match.group(1)

        logger.info("Extracted fields from text: %s", list(data.keys()))
        return data

    def _validate_and_construct(self, data: dict, video_id: str, processing_time: float) -> ClaimAssessment:
        """Validate parsed JSON and construct ClaimAssessment.

        Args:
            data: Parsed JSON dictionary
            video_id: Video ID for metadata
            processing_time: Processing time in seconds

        Returns:
            ClaimAssessment object (with default fallback on validation failure)
        """
        # Add metadata
        data["video_id"] = video_id
        data["processing_time_sec"] = processing_time

        # Fill defaults for missing sub-model fields (VLM often omits these)
        fa = data.get("fault_assessment")
        if isinstance(fa, dict):
            fa.setdefault("fault_ratio", 50.0)
            fa.setdefault("reasoning", "Not specified by model")
            fa.setdefault("scenario_type", "unknown")
        elif fa is None:
            data["fault_assessment"] = {
                "fault_ratio": 50.0,
                "reasoning": "Not specified by model",
                "scenario_type": "unknown",
            }

        fr = data.get("fraud_risk")
        if isinstance(fr, dict):
            fr.setdefault("risk_score", 0.0)
            fr.setdefault("reasoning", "Not evaluated")
        elif fr is None:
            data["fraud_risk"] = {
                "risk_score": 0.0,
                "indicators": [],
                "reasoning": "Not evaluated",
            }

        try:
            # Validate with Pydantic
            assessment = ClaimAssessment(**data)
            logger.info("Successfully parsed claim assessment: severity=%s", assessment.severity)
            return assessment

        except ValidationError as exc:
            logger.error("Validation failed: %s", exc)
            logger.warning("Returning default assessment due to validation failure")
            default = create_default_claim_assessment(video_id)
            # Preserve severity/confidence if they were parsed
            if "severity" in data and data["severity"] in ("NONE", "LOW", "MEDIUM", "HIGH"):
                default.severity = data["severity"]
            if "confidence" in data and isinstance(data["confidence"], (int, float)):
                default.confidence = max(0.0, min(1.0, float(data["confidence"])))
            if "causal_reasoning" in data and isinstance(data["causal_reasoning"], str):
                default.causal_reasoning = data["causal_reasoning"]
            return default

        except Exception as exc:
            logger.error("Unexpected error during validation: %s", exc)
            return create_default_claim_assessment(video_id)

    def assess_claim(
        self, video_path: Path | str, video_id: str, start_sec: float = 0.0, end_sec: float | None = None
    ) -> ClaimAssessment:
        """Assess insurance claim from dashcam video clip.

        This is the main entry point for claim assessment.

        Args:
            video_path: Path to dashcam video file
            video_id: Unique identifier for this video
            start_sec: Start time in seconds (default: 0.0)
            end_sec: End time in seconds (None = entire video)

        Returns:
            ClaimAssessment object with structured evaluation

        Raises:
            ValueError: If video cannot be opened
            RuntimeError: If inference fails catastrophically
        """
        start_time = time.time()

        try:
            # Step 1: Sample frames
            logger.info("Assessing claim for video: %s [%.2f-%.2f sec]", video_id, start_sec, end_sec or 0)
            frame_paths = self._sample_frames(video_path, start_sec, end_sec)

            # Step 2: Prepare prompt
            prompt = get_claim_assessment_prompt(include_calibration=True)

            # Step 3: Run inference
            raw_output = self._run_inference(frame_paths, prompt)
            logger.debug("Raw output: %s", raw_output[:500])

            # Step 4: Parse JSON
            processing_time = time.time() - start_time
            assessment = self._parse_json_response(raw_output, video_id, processing_time)

            logger.info("Claim assessment complete: %s (%.2f sec)", assessment.severity, processing_time)
            return assessment

        except Exception as exc:
            logger.error("Claim assessment failed for %s: %s", video_id, exc)
            processing_time = time.time() - start_time

            # Return safe default assessment
            default = create_default_claim_assessment(video_id)
            default.processing_time_sec = processing_time
            default.causal_reasoning = f"Assessment failed: {str(exc)[:200]}"
            return default

        finally:
            # Cleanup temp frames
            if "frame_paths" in locals():
                temp_dir = frame_paths[0].parent if frame_paths else None
                if temp_dir and temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.debug("Cleaned up temp frames: %s", temp_dir)


def create_client(model_name: ModelName = "qwen2.5-vl-7b", device: str = "cuda") -> VideoLLMClient:
    """Factory function to create VLM client with default config.

    Args:
        model_name: Model to use (nvidia-cosmos-reason-2, qwen2.5-vl-7b, mock)
        device: Device to run on (cuda, cpu)

    Returns:
        Configured VideoLLMClient instance
    """
    config = VLMConfig(
        model_name=model_name,
        device=device,
        dtype="bfloat16" if model_name == "qwen2.5-vl-7b" else "float16",
    )
    return VideoLLMClient(config)
