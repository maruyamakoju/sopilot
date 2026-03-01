import contextlib
import json
import logging
import os
import traceback
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

import numpy as np

from sopilot.config import Settings
from sopilot.core.math_utils import l2_normalize

logger = logging.getLogger(__name__)
_CAPSULE_IO_LOCK = Lock()


def _capsule_enabled() -> bool:
    value = os.getenv("SOPILOT_ENABLE_FAILURE_CAPSULE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _immediate_retry_enabled() -> bool:
    value = os.getenv("SOPILOT_EMBEDDER_IMMEDIATE_RETRY", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _capsule_path() -> Path:
    explicit = os.getenv("SOPILOT_EMBEDDER_FAILURE_CAPSULE_PATH")
    if explicit:
        return Path(explicit).resolve()
    data_dir = Path(os.getenv("SOPILOT_DATA_DIR", "data")).resolve()
    return data_dir / "embedder_failure_capsules.jsonl"


def _append_failure_capsule(payload: dict[str, object]) -> bool:
    if not _capsule_enabled():
        return False
    path = _capsule_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with _CAPSULE_IO_LOCK, path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return True


def _build_frame_info(frames: list[np.ndarray]) -> dict[str, object]:
    frame_count = int(len(frames))
    if frame_count <= 0:
        return {"frame_count": 0}
    first = np.asarray(frames[0])
    shape = list(first.shape)
    height = int(shape[0]) if len(shape) >= 2 else None
    width = int(shape[1]) if len(shape) >= 2 else None
    channels = int(shape[2]) if len(shape) >= 3 else None
    return {
        "frame_count": frame_count,
        "frame_height": height,
        "frame_width": width,
        "frame_channels": channels,
        "frame_dtype": str(first.dtype),
    }


def _build_failure_capsule(
    *,
    backend_name: str,
    backend_type: str,
    exception: Exception,
    frames: list[np.ndarray],
    runtime_context: dict[str, object] | None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "ts_utc": datetime.now(UTC).isoformat(),
        "task_id": os.getenv("SOPILOT_PRIMARY_TASK_ID"),
        "task_name": os.getenv("SOPILOT_PRIMARY_TASK_NAME"),
        "run_id": os.getenv("SOPILOT_RUN_ID"),
        "run_seed": os.getenv("SOPILOT_RUN_SEED"),
        "backend_name": backend_name,
        "backend_type": backend_type,
        "allow_embedder_fallback": os.getenv("SOPILOT_ALLOW_EMBEDDER_FALLBACK"),
        "frame_info": _build_frame_info(frames),
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "traceback_tail": traceback.format_exc()[-4000:],
    }
    if runtime_context:
        payload["runtime_context"] = runtime_context
    if metadata:
        payload["metadata"] = metadata
    return payload


class VideoEmbedder(Protocol):
    name: str

    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        ...


class ColorMotionEmbedder:
    """
    Lightweight deterministic embedder used as fallback and for tests.
    """

    name = "color-motion-v1"

    def embed_batch(self, clips: list[list[np.ndarray]]) -> np.ndarray:
        """Embed N clips; returns (N, D) float32 array."""
        return np.stack([self.embed(frames) for frames in clips])

    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            raise ValueError("frames must not be empty")

        video = np.asarray(frames, dtype=np.float32) / 255.0

        mean_rgb = video.mean(axis=(0, 1, 2))
        std_rgb = video.std(axis=(0, 1, 2))
        gray = 0.2989 * video[..., 0] + 0.5870 * video[..., 1] + 0.1140 * video[..., 2]
        brightness = np.array([gray.mean(), gray.std()], dtype=np.float32)

        if gray.shape[0] > 1:
            frame_diff = np.abs(np.diff(gray, axis=0))
            motion = np.array(
                [float(frame_diff.mean()), float(frame_diff.std()), float(np.percentile(frame_diff, 90))],
                dtype=np.float32,
            )
            time_curve = gray.mean(axis=(1, 2))
            if len(time_curve) > 2:
                slope = float(np.polyfit(np.arange(len(time_curve), dtype=np.float32), time_curve, 1)[0])
            else:
                slope = float(time_curve[-1] - time_curve[0])
        else:
            motion = np.zeros(3, dtype=np.float32)
            slope = 0.0

        vector = np.concatenate(
            [
                mean_rgb.astype(np.float32),
                std_rgb.astype(np.float32),
                motion,
                np.array([slope], dtype=np.float32),
                brightness,
            ],
            axis=0,
        )
        return l2_normalize(vector.astype(np.float32))


@dataclass(frozen=True)
class VJEPA2Config:
    variant: str
    pretrained: bool
    crop_size: int
    device: str
    use_amp: bool
    pooling: str = "mean_tokens"


class VJEPA2HubEmbedder:
    """
    V-JEPA2 embedder loaded from PyTorch Hub.

    Notes:
    - `torch.hub.load(..., "vjepa2_vit_large")` returns `(encoder, predictor)`.
    - We only need the encoder for representation extraction.
    """

    def __init__(self, config: VJEPA2Config) -> None:
        self.config = config
        self.name = f"{config.variant}-hub"
        self._lock = Lock()
        self._context_lock = Lock()
        self._runtime_context: dict[str, object] = {}
        self._ready = False
        self._torch: Any = None
        self._encoder: Any = None
        self._processor: Any = None
        self._device: Any = None

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            import torch

            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
            self._device = torch.device(device)

            # ── GPU VRAM safety pre-flight ──────────────────────────────────
            # V-JEPA2 ViT-Large requires ≈6–12 GB VRAM; warn early and block
            # if free memory is critically low to prevent system freeze.
            if self._device.type == "cuda":
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    free_gb  = free_bytes  / 1024 ** 3
                    total_gb = total_bytes / 1024 ** 3
                    logger.warning(
                        "[V-JEPA2] VRAM: %.1f GB free / %.1f GB total. "
                        "ViT-Large requires ≈6–8 GB (fp16) or ≈12 GB (fp32). "
                        "Proceeding with model load…",
                        free_gb, total_gb,
                    )
                    if free_gb < 4.0:
                        raise RuntimeError(
                            f"[V-JEPA2] Insufficient VRAM: {free_gb:.1f} GB free, "
                            "need ≥4 GB (fp16). "
                            "Set SOPILOT_EMBEDDER_BACKEND=color-motion to use CPU embedder."
                        )
                except RuntimeError:
                    raise
                except Exception as _vram_exc:
                    logger.warning("[V-JEPA2] VRAM check skipped: %s", _vram_exc)

            model_bundle = torch.hub.load(
                "facebookresearch/vjepa2",
                self.config.variant,
                pretrained=self.config.pretrained,
            )
            encoder = model_bundle[0] if isinstance(model_bundle, tuple) else model_bundle
            encoder = encoder.to(self._device)
            encoder.eval()

            # Cast model weights to fp16 when amp is enabled — halves VRAM usage
            # (~12 GB fp32 → ~6 GB fp16), reducing freeze risk on consumer GPUs.
            if self.config.use_amp and self._device.type == "cuda":
                encoder = encoder.half()
                logger.info("[V-JEPA2] Model weights cast to float16 (use_amp=True). VRAM ~halved.")

            processor = torch.hub.load(
                "facebookresearch/vjepa2",
                "vjepa2_preprocessor",
                crop_size=self.config.crop_size,
            )

            self._torch = torch
            self._encoder = encoder
            self._processor = processor
            self._ready = True

    def set_runtime_context(self, **context: object) -> None:
        with self._context_lock:
            self._runtime_context = {key: value for key, value in context.items() if value is not None}

    def _snapshot_runtime_context(self) -> dict[str, object]:
        with self._context_lock:
            return dict(self._runtime_context)

    def _pool_tokens(self, token_tensor: Any) -> Any:
        mode = str(self.config.pooling or "mean_tokens").lower()
        if mode == "flatten":
            return token_tensor.reshape(token_tensor.shape[0], -1)
        if mode == "first_token":
            if token_tensor.ndim == 2:
                return token_tensor
            if token_tensor.ndim == 3:
                return token_tensor[:, 0, :]
            if token_tensor.ndim == 4:
                return token_tensor[:, 0, 0, :]
            return token_tensor.reshape(token_tensor.shape[0], -1)

        # Default "mean_tokens": average over token-like axes while keeping feature dim.
        if token_tensor.ndim == 2:
            return token_tensor
        if token_tensor.ndim >= 3:
            reduce_dims = tuple(range(1, token_tensor.ndim - 1))
            if reduce_dims:
                return token_tensor.mean(dim=reduce_dims)
        return token_tensor.reshape(token_tensor.shape[0], -1)

    def embed_batch(self, clips: list[list[np.ndarray]]) -> np.ndarray:
        """
        Embed N clips in a single encoder forward pass.

        All clips are preprocessed individually (to handle variable frame counts
        gracefully), then batched along dim-0 before encoder inference.
        Falls back to per-clip embed() when shapes are incompatible.

        Returns
        -------
        np.ndarray of shape (N, D) float32, each row L2-normalised.
        """
        if not clips:
            raise ValueError("clips must not be empty")
        if len(clips) == 1:
            return np.expand_dims(self.embed(clips[0]), 0)

        try:
            self._ensure_ready()
            torch = self._torch
            if torch is None or self._encoder is None or self._processor is None or self._device is None:
                raise RuntimeError("V-JEPA2 embedder not initialised; call _ensure_ready() first")

            processed_list: list = []
            for frames in clips:
                clip_frames = frames if len(frames) >= 2 else [frames[0], frames[0]]
                clip_np = np.asarray(clip_frames, dtype=np.uint8)
                clip_tensor = (
                    torch.from_numpy(clip_np)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                result = self._processor(clip_tensor)
                x = result[0] if isinstance(result, list) else result
                if x.ndim == 4:
                    x = x.unsqueeze(0)   # (1, C, T, H, W)
                processed_list.append(x)

            # Stack along batch dim — requires identical (C, T, H, W) across clips.
            batch = torch.cat(processed_list, dim=0).to(self._device)

            if self._device.type == "cuda" and self.config.use_amp:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                        tokens = self._encoder(batch)
            else:
                with torch.no_grad():
                    tokens = self._encoder(batch)

            token_tensor = tokens[-1] if isinstance(tokens, list | tuple) else tokens

            embeddings = self._pool_tokens(token_tensor)   # (B, D)
            vectors = embeddings.detach().float().cpu().numpy().astype(np.float32)

            if self._device.type == "cuda":
                self._torch.cuda.empty_cache()

            return np.stack([l2_normalize(v) for v in vectors])

        except Exception as exc:
            logger.warning(
                "[V-JEPA2] embed_batch failed (%s: %s), falling back to per-clip embed",
                type(exc).__name__, exc,
            )
            # Per-clip fallback preserves failure-capsule logging in embed().
            return np.stack([self.embed(frames) for frames in clips])

    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        try:
            if not frames:
                raise ValueError("frames must not be empty")
            self._ensure_ready()

            torch = self._torch
            if torch is None or self._encoder is None or self._processor is None or self._device is None:
                raise RuntimeError("V-JEPA2 embedder not initialised; call _ensure_ready() first")

            # V-JEPA2 patch embedding requires temporal depth >= 2; duplicate single-frame clips.
            clip_frames = frames if len(frames) >= 2 else [frames[0], frames[0]]
            # V-JEPA2 preprocessor expects a (T, C, H, W) torch tensor (uint8, RGB).
            # VideoProcessor delivers frames as (H, W, C) numpy arrays, so we
            # stack to (T, H, W, C) then permute axes before handing off.
            clip_np = np.asarray(clip_frames, dtype=np.uint8)          # (T, H, W, C)
            clip_tensor = (
                torch.from_numpy(clip_np)
                .permute(0, 3, 1, 2)   # → (T, C, H, W)
                .contiguous()
            )
            processed = self._processor(clip_tensor)
            if isinstance(processed, list):
                if not processed:
                    raise ValueError("V-JEPA2 preprocessor returned empty output")
                model_input = processed[0]
            else:
                model_input = processed

            if model_input.ndim == 4:
                model_input = model_input.unsqueeze(0)
            model_input = model_input.to(self._device)

            if self._device.type == "cuda" and self.config.use_amp:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                        tokens = self._encoder(model_input)
            else:
                with torch.no_grad():
                    tokens = self._encoder(model_input)

            token_tensor = tokens[-1] if isinstance(tokens, list | tuple) else tokens

            embedding = self._pool_tokens(token_tensor)

            vector = embedding[0].detach().float().cpu().numpy().astype(np.float32)

            # Release GPU memory promptly after each clip to avoid fragmentation.
            if self._device is not None and self._device.type == "cuda":
                self._torch.cuda.empty_cache()

            return l2_normalize(vector)
        except Exception as exc:
            _append_failure_capsule(
                _build_failure_capsule(
                    backend_name=self.name,
                    backend_type=type(self).__name__,
                    exception=exc,
                    frames=frames,
                    runtime_context=self._snapshot_runtime_context(),
                )
            )
            raise


class FallbackEmbedder:
    BASE_RETRY_INTERVAL = 60.0   # seconds before first retry
    MAX_RETRY_INTERVAL = 900.0   # 15 minutes cap
    MAX_CONSECUTIVE_FAILURES = 10  # give up on primary after this many

    def __init__(self, primary: VideoEmbedder, fallback: VideoEmbedder) -> None:
        self.primary = primary
        self.fallback = fallback
        self._stats_lock = Lock()
        self._context_lock = Lock()
        self._runtime_context: dict[str, object] = {}
        self._failed_over = False
        self._failed_at: float = 0.0
        self._consecutive_failures: int = 0
        self._current_interval: float = self.BASE_RETRY_INTERVAL
        self._permanently_failed = False
        self._expected_dim: int | None = None
        self._stats: dict[str, int] = {
            "total_requests": 0,
            "primary_attempts": 0,
            "primary_successes": 0,
            "primary_failures": 0,
            "fallback_uses": 0,
            "fallback_successes": 0,
            "fallback_dim_coercions": 0,
            "recoveries": 0,
            "failure_capsules_written": 0,
            "primary_retry_attempts": 0,
            "primary_retry_successes": 0,
            "primary_retry_failures": 0,
        }
        self.name = primary.name

    def set_runtime_context(self, **context: object) -> None:
        filtered = {key: value for key, value in context.items() if value is not None}
        with self._context_lock:
            self._runtime_context = filtered
        set_primary = getattr(self.primary, "set_runtime_context", None)
        if callable(set_primary):
            with contextlib.suppress(Exception):
                set_primary(**filtered)
        set_fallback = getattr(self.fallback, "set_runtime_context", None)
        if callable(set_fallback):
            with contextlib.suppress(Exception):
                set_fallback(**filtered)

    def _snapshot_runtime_context(self) -> dict[str, object]:
        with self._context_lock:
            return dict(self._runtime_context)

    # Keep RETRY_INTERVAL as a property for backward compat with tests
    @property
    def RETRY_INTERVAL(self) -> float:
        return self._current_interval

    def _should_retry(self) -> bool:
        if self._permanently_failed:
            return False
        import time
        return (time.monotonic() - self._failed_at) >= self._current_interval

    def _record_failure(self) -> None:
        import time
        self._inc_stat("primary_failures")
        self._failed_over = True
        self._failed_at = time.monotonic()
        self._consecutive_failures += 1
        # Exponential backoff: 60s, 120s, 240s, ..., capped at 900s
        self._current_interval = min(
            self.BASE_RETRY_INTERVAL * (2 ** (self._consecutive_failures - 1)),
            self.MAX_RETRY_INTERVAL,
        )
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            self._permanently_failed = True
        self.name = f"{self.primary.name}->fallback:{self.fallback.name}"

    def _recover(self) -> None:
        self._inc_stat("recoveries")
        self._failed_over = False
        self._failed_at = 0.0
        self._consecutive_failures = 0
        self._current_interval = self.BASE_RETRY_INTERVAL
        self._permanently_failed = False
        self.name = self.primary.name

    def _enforce_dim(self, vector: np.ndarray, *, source: str) -> np.ndarray:
        flat = np.asarray(vector, dtype=np.float32).reshape(-1)
        if flat.size <= 0:
            raise ValueError(f"{source} embedder returned empty embedding")

        if self._expected_dim is None:
            self._expected_dim = int(flat.size)
            return flat

        if int(flat.size) == int(self._expected_dim):
            return flat

        if source == "fallback":
            target = int(self._expected_dim)
            coerced = np.zeros(target, dtype=np.float32)
            keep = min(target, int(flat.size))
            coerced[:keep] = flat[:keep]
            self._inc_stat("fallback_dim_coercions")
            logger.warning(
                "fallback embedding dim mismatch: expected=%s got=%s; coercing to expected dimension",
                target,
                int(flat.size),
            )
            return l2_normalize(coerced)

        raise ValueError(
            f"primary embedding dim mismatch: expected={self._expected_dim} got={int(flat.size)}"
        )

    def _inc_stat(self, key: str, delta: int = 1) -> None:
        with self._stats_lock:
            self._stats[key] = int(self._stats.get(key, 0) + delta)

    def _snapshot_stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {key: int(value) for key, value in self._stats.items()}

    def get_stats(self) -> dict[str, object]:
        stats = self._snapshot_stats()
        total_requests = int(stats.get("total_requests", 0))
        fallback_uses = int(stats.get("fallback_uses", 0))
        return {
            **stats,
            "fallback_usage_rate": (float(fallback_uses) / float(total_requests)) if total_requests > 0 else 0.0,
            "failed_over": bool(self._failed_over),
            "permanently_failed": bool(self._permanently_failed),
            "current_retry_interval_sec": float(self._current_interval),
            "expected_embedding_dim": int(self._expected_dim) if self._expected_dim is not None else None,
            "primary_name": self.primary.name,
            "fallback_name": self.fallback.name,
        }

    def _write_failure_capsule(
        self,
        *,
        exception: Exception,
        frames: list[np.ndarray],
        retry_attempt: int | None,
    ) -> None:
        wrote = _append_failure_capsule(
            _build_failure_capsule(
                backend_name=self.name,
                backend_type=type(self).__name__,
                exception=exception,
                frames=frames,
                runtime_context=self._snapshot_runtime_context(),
                metadata={
                    "primary_name": self.primary.name,
                    "fallback_name": self.fallback.name,
                    "retry_attempt": retry_attempt,
                    "failed_over": bool(self._failed_over),
                    "permanently_failed": bool(self._permanently_failed),
                    "consecutive_failures": int(self._consecutive_failures),
                    "retry_interval_sec": float(self._current_interval),
                },
            )
        )
        if wrote:
            self._inc_stat("failure_capsules_written")

    def embed_batch(self, clips: list[list[np.ndarray]]) -> np.ndarray:
        """
        Batch embed with fallback.  Delegates to primary.embed_batch() when
        the primary is healthy; otherwise falls through to per-clip embed()
        which contains the full failover / retry / capsule logic.
        """
        primary_batch = getattr(self.primary, "embed_batch", None)
        if callable(primary_batch) and not self._failed_over and not self._permanently_failed:
            try:
                return np.asarray(primary_batch(clips))
            except Exception:
                # embed_batch already logged / will fall through to per-clip path
                pass
        # Per-clip path: each call goes through the full failover state machine.
        return np.stack([self.embed(frames) for frames in clips])

    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        self._inc_stat("total_requests")
        if self._failed_over and not self._should_retry():
            self._inc_stat("fallback_uses")
            vector = self._enforce_dim(self.fallback.embed(frames), source="fallback")
            self._inc_stat("fallback_successes")
            return vector
        try:
            self._inc_stat("primary_attempts")
            result = self._enforce_dim(self.primary.embed(frames), source="primary")
            self._inc_stat("primary_successes")
            if self._failed_over:
                self._recover()
            return result
        except Exception as first_exc:
            self._write_failure_capsule(exception=first_exc, frames=frames, retry_attempt=0)
            if _immediate_retry_enabled():
                # Immediate retry is opt-in to preserve historical fallback semantics.
                self._inc_stat("primary_retry_attempts")
                try:
                    self._inc_stat("primary_attempts")
                    result = self._enforce_dim(self.primary.embed(frames), source="primary")
                    self._inc_stat("primary_successes")
                    self._inc_stat("primary_retry_successes")
                    if self._failed_over:
                        self._recover()
                    return result
                except Exception as retry_exc:
                    self._inc_stat("primary_retry_failures")
                    self._write_failure_capsule(exception=retry_exc, frames=frames, retry_attempt=1)

            self._record_failure()
            self._inc_stat("fallback_uses")
            vector = self._enforce_dim(self.fallback.embed(frames), source="fallback")
            self._inc_stat("fallback_successes")
            return vector


def build_embedder(settings: Settings) -> VideoEmbedder:
    if settings.embedder_backend == "color-motion":
        return ColorMotionEmbedder()

    primary = VJEPA2HubEmbedder(
        VJEPA2Config(
            variant=settings.vjepa2_variant,
            pretrained=settings.vjepa2_pretrained,
            crop_size=settings.vjepa2_crop_size,
            device=settings.vjepa2_device,
            use_amp=settings.vjepa2_use_amp,
            pooling=settings.vjepa2_pooling,
        )
    )
    if not settings.allow_embedder_fallback:
        return primary
    return FallbackEmbedder(primary=primary, fallback=ColorMotionEmbedder())
