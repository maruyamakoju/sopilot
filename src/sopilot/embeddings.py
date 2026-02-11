from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np

from .utils import normalize_rows
from .video import ClipWindow

if TYPE_CHECKING:
    from .config import Settings


logger = logging.getLogger(__name__)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


_l2_normalize_rows = normalize_rows


class ClipEmbedder(Protocol):
    name: str

    def embed_clips(self, clips: list[ClipWindow]) -> np.ndarray:
        ...

    def active_name(self) -> str:
        ...


@dataclass
class HeuristicClipEmbedder:
    hist_bins: int = 8
    name: str = "heuristic-v1"

    def embed_clip(self, frames: np.ndarray) -> np.ndarray:
        x = frames.astype(np.float32) / 255.0
        mean_rgb = x.mean(axis=(0, 1, 2))
        std_rgb = x.std(axis=(0, 1, 2))

        first_half = x[: max(1, len(x) // 2)].mean(axis=(0, 1, 2))
        last_half = x[len(x) // 2 :].mean(axis=(0, 1, 2))
        temporal_delta = last_half - first_half

        gray = np.mean(x, axis=3)
        gray_mean = np.array([gray.mean(), gray.std()], dtype=np.float32)

        if len(gray) > 1:
            motion = np.abs(gray[1:] - gray[:-1])
            motion_stats = np.array(
                [motion.mean(), motion.std(), motion.max()], dtype=np.float32
            )
        else:
            motion_stats = np.zeros(3, dtype=np.float32)

        edge_values = []
        for frame in frames:
            edges = cv2.Canny(frame, threshold1=80, threshold2=160)
            edge_values.append(float(edges.mean() / 255.0))
        edge_stats = np.array(
            [np.mean(edge_values), np.std(edge_values)], dtype=np.float32
        )

        hists = []
        for channel in range(3):
            hist, _ = np.histogram(
                x[:, :, :, channel],
                bins=self.hist_bins,
                range=(0.0, 1.0),
                density=True,
            )
            hists.append(hist.astype(np.float32))
        color_hist = np.concatenate(hists, axis=0)

        # Spatially-aware low-resolution descriptors improve robustness when
        # global color histograms are too similar across different steps.
        mean_frame = x.mean(axis=0)
        first_frame = x[0]
        last_frame = x[-1]
        low_size = (16, 9)
        low_mean = cv2.resize(mean_frame, low_size, interpolation=cv2.INTER_AREA).reshape(-1)
        low_first = cv2.resize(first_frame, low_size, interpolation=cv2.INTER_AREA).reshape(-1)
        low_last = cv2.resize(last_frame, low_size, interpolation=cv2.INTER_AREA).reshape(-1)
        low_delta = (low_last - low_first).astype(np.float32)

        feature = np.concatenate(
            [
                mean_rgb,
                std_rgb,
                temporal_delta,
                gray_mean,
                motion_stats,
                edge_stats,
                color_hist,
                low_mean.astype(np.float32),
                low_delta.astype(np.float32),
            ],
            axis=0,
        )
        return _l2_normalize(feature)

    def embed_clips(self, clips: list[ClipWindow]) -> np.ndarray:
        vectors = [self.embed_clip(clip.frames) for clip in clips]
        return np.stack(vectors, axis=0).astype(np.float32)

    def active_name(self) -> str:
        return self.name


class VJepa2Embedder:
    """
    V-JEPA2 encoder wrapper with RTX 5090 optimizations.

    Performance optimizations:
    - Dynamic batch sizing based on available GPU memory
    - Pinned memory for async transfers
    - Compilation with torch.compile for 20-30% speedup
    - Gradient checkpointing disabled (inference only)

    Expected model source:
    - torch.hub repository: facebookresearch/vjepa2
    - entrypoint examples: vjepa2_vit_large / vjepa2_vit_huge / vjepa2_vit_giant

    References:
    - V-JEPA2: https://ai.meta.com/research/publications/v-jepa-revisited/
    - Mixed precision: https://pytorch.org/docs/stable/amp.html
    """

    def __init__(
        self,
        *,
        repo: str,
        source: str,
        local_repo: str,
        local_checkpoint: str,
        variant: str,
        pretrained: bool,
        device: str,
        num_frames: int,
        image_size: int,
        batch_size: int,
    ) -> None:
        self.repo = repo
        self.source = source.strip().lower()
        self.local_repo = local_repo.strip()
        self.local_checkpoint = local_checkpoint.strip()
        self.variant = variant
        self.pretrained = pretrained
        self.device_pref = device
        self.num_frames = num_frames
        self.image_size = image_size
        self.batch_size = max(1, int(batch_size))

        self.name = f"vjepa2:{variant}:{'pt' if pretrained else 'scratch'}"
        self._torch = None
        self._encoder = None
        self._device = None
        self._load_error: Exception | None = None
        self._optimal_batch_size: int | None = None
        self._compile_enabled = False

    def _resolve_device(self, torch_mod):
        if self.device_pref == "auto":
            return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
        return torch_mod.device(self.device_pref)

    @staticmethod
    def _clean_backbone_key(state_dict: dict) -> dict:
        cleaned: dict = {}
        for key, val in state_dict.items():
            nk = key.replace("module.", "").replace("backbone.", "")
            cleaned[nk] = val
        return cleaned

    def _load_local_checkpoint(self, encoder) -> None:
        if not self.local_checkpoint:
            return
        ckpt_path = Path(self.local_checkpoint)
        if not ckpt_path.exists():
            raise RuntimeError(f"local checkpoint not found: {ckpt_path}")
        torch = self._torch
        assert torch is not None
        blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "encoder" in blob:
            state = blob["encoder"]
        else:
            state = blob
        if not isinstance(state, dict):
            raise RuntimeError("unsupported local checkpoint format for encoder")
        state = self._clean_backbone_key(state)
        encoder.load_state_dict(state, strict=False)

    def _auto_detect_optimal_batch_size(self) -> int:
        """Detect optimal batch size for available GPU memory."""
        torch = self._torch
        assert torch is not None and self._device is not None

        if self._device.type != "cuda":
            return self.batch_size

        try:
            # RTX 5090 has 32GB VRAM - can handle much larger batches
            total_memory = torch.cuda.get_device_properties(0).total_memory
            total_gb = total_memory / (1024 ** 3)

            # Conservative estimate: 1GB per batch item for ViT-Large
            # RTX 5090: can do 16-24 batch size safely
            if "giant" in self.variant:
                batch_limit = max(4, int(total_gb / 2.5))  # 2.5GB per giant batch
            elif "huge" in self.variant:
                batch_limit = max(8, int(total_gb / 1.5))  # 1.5GB per huge batch
            else:  # large
                batch_limit = max(16, int(total_gb / 1.0))  # 1GB per large batch

            optimal = min(batch_limit, 32)  # Cap at 32 for numerical stability
            logger.info(
                "Auto-detected optimal batch size: %d (GPU: %.1fGB, variant: %s)",
                optimal, total_gb, self.variant
            )
            return optimal
        except Exception as e:
            logger.warning("Failed to auto-detect batch size: %s. Using default %d", e, self.batch_size)
            return self.batch_size

    def _load_model_if_needed(self) -> None:
        if self._encoder is not None:
            return
        if self._load_error is not None:
            logger.info("Retrying V-JEPA2 load after previous error: %s", self._load_error)
            self._load_error = None

        try:
            import torch

            self._torch = torch
            self._device = self._resolve_device(torch)
            load_pretrained = self.pretrained
            if self.local_checkpoint:
                # Load weights explicitly from local checkpoint to stay offline/reproducible.
                load_pretrained = False

            if self.source == "local":
                if not self.local_repo:
                    raise RuntimeError("SOPILOT_VJEPA2_LOCAL_REPO is required when source=local")
                repo_or_dir = self.local_repo
                bundle = torch.hub.load(
                    repo_or_dir,
                    self.variant,
                    pretrained=load_pretrained,
                    source="local",
                    trust_repo=True,
                )
            elif self.source == "hub":
                repo_or_dir = self.repo
                bundle = torch.hub.load(
                    repo_or_dir,
                    self.variant,
                    pretrained=load_pretrained,
                    trust_repo=True,
                )
            else:
                raise RuntimeError(f"unsupported V-JEPA2 source: {self.source}")

            if isinstance(bundle, tuple):
                encoder = bundle[0]
            else:
                encoder = bundle

            self._load_local_checkpoint(encoder)
            encoder = encoder.to(self._device).eval()

            # RTX 5090 optimization: torch.compile for 20-30% speedup
            if self._device.type == "cuda" and hasattr(torch, "compile"):
                try:
                    logger.info("Compiling V-JEPA2 encoder with torch.compile...")
                    encoder = torch.compile(encoder, mode="reduce-overhead")
                    self._compile_enabled = True
                    logger.info("Torch compile enabled successfully")
                except Exception as e:
                    logger.warning("Torch compile failed (fallback to eager): %s", e)

            self._encoder = encoder

            # Auto-detect optimal batch size for this GPU
            self._optimal_batch_size = self._auto_detect_optimal_batch_size()

            logger.info(
                "Loaded V-JEPA2 encoder variant=%s source=%s pretrained=%s local_checkpoint=%s device=%s compile=%s optimal_batch=%d",
                self.variant,
                self.source,
                self.pretrained,
                bool(self.local_checkpoint),
                self._device,
                self._compile_enabled,
                self._optimal_batch_size or self.batch_size,
            )
        except Exception as exc:  # pragma: no cover - runtime/network dependent
            self._load_error = exc
            raise RuntimeError(f"unable to load V-JEPA2: {exc}") from exc

    def _sample_temporal(self, frames: np.ndarray) -> np.ndarray:
        total = int(frames.shape[0])
        if total == self.num_frames:
            return frames
        if total > self.num_frames:
            idx = np.linspace(0, total - 1, num=self.num_frames, dtype=np.int64)
            return frames[idx]
        idx = np.linspace(0, max(total - 1, 0), num=self.num_frames, dtype=np.int64)
        return frames[idx]

    def _preprocess_one(self, frames: np.ndarray):
        torch = self._torch
        assert torch is not None

        sampled = self._sample_temporal(frames)
        x = torch.from_numpy(sampled.astype(np.float32) / 255.0)

        # Pin memory for faster CPU->GPU transfer on RTX 5090
        if self._device is not None and self._device.type == "cuda":
            x = x.pin_memory()

        # T H W C -> T C H W
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.nn.functional.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        # T C H W -> C T H W
        x = x.permute(1, 0, 2, 3).contiguous()
        return x

    @staticmethod
    def _pool_encoder_output(output) -> np.ndarray:
        # Common V-JEPA2 encoder output is B x Tokens x Dim.
        if isinstance(output, (tuple, list)):
            output = output[0]
        if hasattr(output, "detach"):
            tensor = output.detach()
        else:
            raise ValueError("unexpected encoder output type")

        if tensor.ndim == 3:
            pooled = tensor.mean(dim=1)  # B D
        elif tensor.ndim == 2:
            pooled = tensor
        elif tensor.ndim == 4:
            pooled = tensor.mean(dim=(1, 2))
        else:
            raise ValueError(f"unsupported V-JEPA2 output shape: {tuple(tensor.shape)}")
        return pooled.cpu().float().numpy()

    def embed_clips(self, clips: list[ClipWindow]) -> np.ndarray:
        if not clips:
            return np.zeros((0, 0), dtype=np.float32)

        self._load_model_if_needed()
        torch = self._torch
        assert torch is not None and self._encoder is not None and self._device is not None

        # Use optimal batch size detected for this GPU
        effective_batch_size = self._optimal_batch_size or self.batch_size

        vectors: list[np.ndarray] = []
        for start in range(0, len(clips), effective_batch_size):
            batch = clips[start : start + effective_batch_size]

            # Preprocess on CPU with pinned memory
            preprocessed = [self._preprocess_one(c.frames) for c in batch]
            x = torch.stack(preprocessed, dim=0)

            # Async transfer to GPU if CUDA
            if self._device.type == "cuda":
                x = x.to(self._device, non_blocking=True)
            else:
                x = x.to(self._device)

            with torch.inference_mode():
                if self._device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self._encoder(x)
                else:
                    out = self._encoder(x)
            pooled = self._pool_encoder_output(out)
            vectors.append(pooled.astype(np.float32))

        mat = np.concatenate(vectors, axis=0).astype(np.float32)
        return _l2_normalize_rows(mat)

    def active_name(self) -> str:
        return self.name


class AutoEmbedder:
    _RETRY_INTERVAL_SEC = 300  # Re-probe primary every 5 minutes after fallback

    def __init__(self, primary: ClipEmbedder, fallback: ClipEmbedder) -> None:
        self.primary = primary
        self.fallback = fallback
        self._using_fallback = False
        self._fallback_since: float = 0.0
        self.name = f"{primary.name}|fallback:{fallback.name}"

    def embed_clips(self, clips: list[ClipWindow]) -> np.ndarray:
        if self._using_fallback:
            if time.monotonic() - self._fallback_since >= self._RETRY_INTERVAL_SEC:
                try:
                    result = self.primary.embed_clips(clips)
                    logger.info("Primary embedder recovered; switching back from fallback")
                    self._using_fallback = False
                    return result
                except Exception:
                    self._fallback_since = time.monotonic()
            return self.fallback.embed_clips(clips)
        try:
            return self.primary.embed_clips(clips)
        except Exception as exc:  # pragma: no cover - runtime dependent
            logger.warning("Primary embedder failed; switching to fallback: %s", exc)
            self._using_fallback = True
            self._fallback_since = time.monotonic()
            return self.fallback.embed_clips(clips)

    def active_name(self) -> str:
        if self._using_fallback:
            return self.fallback.active_name()
        return self.primary.active_name()


def build_embedder(settings: Settings) -> ClipEmbedder:
    backend = settings.embedder_backend.strip().lower()
    heuristic = HeuristicClipEmbedder()

    if backend == "heuristic":
        return heuristic

    vjepa2 = VJepa2Embedder(
        repo=settings.vjepa2_repo,
        source=settings.vjepa2_source,
        local_repo=settings.vjepa2_local_repo,
        local_checkpoint=settings.vjepa2_local_checkpoint,
        variant=settings.vjepa2_variant,
        pretrained=settings.vjepa2_pretrained,
        device=settings.embedding_device,
        num_frames=settings.vjepa2_num_frames,
        image_size=settings.vjepa2_image_size,
        batch_size=settings.vjepa2_batch_size,
    )

    if backend in {"vjepa2", "vjepa2_pt"}:
        return vjepa2
    if backend == "auto":
        if settings.embedder_fallback_enabled:
            return AutoEmbedder(primary=vjepa2, fallback=heuristic)
        return vjepa2

    raise ValueError(f"unknown embedder backend: {settings.embedder_backend}")
