"""Model Orchestrator — VRAM-aware lifecycle manager for ML models.

Phase 31: Manages multiple ML models on constrained GPU VRAM (e.g. RTX 4090
24 GB).  Provides a register → acquire → release lifecycle with LRU eviction
when the VRAM budget is exceeded.

Design principles:
    - Zero-dependency registration: name + load/unload callables + VRAM estimate
    - Thread-safe acquire/release with RLock
    - LRU eviction: lowest-priority, least-recently-used models evicted first
    - Graceful fallback: CPU-only mode when CUDA is unavailable
    - Observability: status dict for API/debug

Example::

    orch = ModelOrchestrator(vram_budget_gb=22.0)
    orch.register("yolo", load_fn=load_yolo, unload_fn=unload_yolo,
                  vram_estimate_gb=0.3, priority=9)
    model = orch.acquire("yolo")   # loads if needed; may evict others
    result = model.predict(frame)
    orch.release("yolo")           # marks as not in active use
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Handle — metadata for a registered model
# ---------------------------------------------------------------------------


@dataclass
class ModelHandle:
    """Metadata and lifecycle callbacks for a registered model."""

    name: str
    load_fn: Callable[[], Any]
    unload_fn: Callable[[Any], None]
    vram_estimate_bytes: int
    priority: int  # higher = keep loaded longer
    device: str = "cuda"  # "cuda" or "cpu"

    # Runtime state (managed by orchestrator)
    instance: Any = field(default=None, repr=False)
    loaded: bool = False
    last_used: float = 0.0
    load_count: int = 0
    active_users: int = 0  # number of concurrent acquire() holders


# ---------------------------------------------------------------------------
# ModelOrchestrator
# ---------------------------------------------------------------------------


class ModelOrchestrator:
    """VRAM-aware model lifecycle manager.

    Manages multiple ML models on a single GPU with a fixed VRAM budget.
    Models are registered with load/unload callables and a VRAM estimate.
    When ``acquire()`` would exceed the budget, the orchestrator evicts
    the lowest-priority, least-recently-used models to make room.

    Parameters
    ----------
    vram_budget_gb:
        Maximum VRAM (in GB) that all loaded models may consume.
        Default 22.0 leaves ~2 GB headroom on a 24 GB GPU.
    device:
        Target device.  ``"auto"`` selects CUDA if available.
    """

    def __init__(
        self,
        vram_budget_gb: float = 22.0,
        device: str = "auto",
    ) -> None:
        self._budget_bytes = int(vram_budget_gb * 1024 ** 3)
        self._device = self._resolve_device(device)
        self._lock = threading.RLock()

        # name → ModelHandle
        self._registry: dict[str, ModelHandle] = {}

        # Stats
        self._total_loads: int = 0
        self._total_evictions: int = 0

        logger.info(
            "ModelOrchestrator created: budget=%.1f GB, device=%s",
            vram_budget_gb,
            self._device,
        )

    # ── Device resolution ─────────────────────────────────────────────

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def device(self) -> str:
        return self._device

    @property
    def has_gpu(self) -> bool:
        return self._device in ("cuda", "mps")

    # ── Registration ──────────────────────────────────────────────────

    def register(
        self,
        name: str,
        load_fn: Callable[[], Any],
        unload_fn: Callable[[Any], None],
        vram_estimate_gb: float = 0.0,
        priority: int = 5,
        device: str | None = None,
    ) -> None:
        """Register a model without loading it.

        Parameters
        ----------
        name:
            Unique model identifier (e.g. ``"yolo"``, ``"dinov2"``).
        load_fn:
            Callable that loads the model and returns the instance.
        unload_fn:
            Callable that receives the model instance and frees resources.
        vram_estimate_gb:
            Estimated VRAM usage in GB.
        priority:
            Keep-loaded priority (0–10).  Higher = evicted last.
        device:
            Override device for this model.  None → orchestrator default.
        """
        with self._lock:
            if name in self._registry:
                logger.warning("Model '%s' already registered; overwriting", name)
                existing = self._registry[name]
                if existing.loaded:
                    self._do_unload(existing)

            self._registry[name] = ModelHandle(
                name=name,
                load_fn=load_fn,
                unload_fn=unload_fn,
                vram_estimate_bytes=int(vram_estimate_gb * 1024 ** 3),
                priority=priority,
                device=device or self._device,
            )
            logger.debug(
                "Registered model '%s': vram=%.2f GB, priority=%d",
                name,
                vram_estimate_gb,
                priority,
            )

    def unregister(self, name: str) -> None:
        """Remove a model from the registry, unloading if necessary."""
        with self._lock:
            handle = self._registry.pop(name, None)
            if handle and handle.loaded:
                self._do_unload(handle)
            logger.debug("Unregistered model '%s'", name)

    # ── Acquire / Release ─────────────────────────────────────────────

    def acquire(self, name: str) -> Any:
        """Get a model instance, loading it if necessary.

        If the model is not loaded and loading it would exceed the VRAM
        budget, lower-priority or least-recently-used models are evicted.

        Parameters
        ----------
        name:
            The registered model name.

        Returns
        -------
        Any
            The loaded model instance.

        Raises
        ------
        KeyError
            If *name* is not registered.
        RuntimeError
            If the model cannot be loaded even after eviction.
        """
        with self._lock:
            handle = self._registry.get(name)
            if handle is None:
                raise KeyError(f"Model '{name}' is not registered")

            if handle.loaded:
                handle.last_used = time.monotonic()
                handle.active_users += 1
                return handle.instance

            # Ensure VRAM budget allows loading
            self._ensure_budget(handle.vram_estimate_bytes, exclude=name)

            # Load the model
            self._do_load(handle)
            handle.active_users += 1
            return handle.instance

    def release(self, name: str) -> None:
        """Signal that the caller is done using the model.

        The model remains loaded (for reuse) but becomes eligible for
        eviction if VRAM pressure increases.
        """
        with self._lock:
            handle = self._registry.get(name)
            if handle and handle.active_users > 0:
                handle.active_users -= 1

    def evict(self, name: str) -> None:
        """Force-unload a specific model."""
        with self._lock:
            handle = self._registry.get(name)
            if handle and handle.loaded:
                self._do_unload(handle)

    # ── Context manager for acquire/release ───────────────────────────

    class _ModelContext:
        """Context manager returned by ``ModelOrchestrator.use()``."""

        def __init__(self, orchestrator: ModelOrchestrator, name: str) -> None:
            self._orch = orchestrator
            self._name = name
            self._instance: Any = None

        def __enter__(self) -> Any:
            self._instance = self._orch.acquire(self._name)
            return self._instance

        def __exit__(self, *exc: Any) -> None:
            self._orch.release(self._name)

    def use(self, name: str) -> _ModelContext:
        """Context manager for acquire/release.

        Example::

            with orch.use("yolo") as model:
                detections = model.predict(frame)
        """
        return self._ModelContext(self, name)

    # ── VRAM bookkeeping ──────────────────────────────────────────────

    @property
    def vram_used_bytes(self) -> int:
        """Total estimated VRAM used by currently loaded models."""
        return sum(
            h.vram_estimate_bytes
            for h in self._registry.values()
            if h.loaded
        )

    @property
    def vram_available_bytes(self) -> int:
        return max(0, self._budget_bytes - self.vram_used_bytes)

    def _ensure_budget(self, needed_bytes: int, exclude: str = "") -> None:
        """Evict models until *needed_bytes* fits in the budget.

        Eviction order: lowest priority first, then least recently used.
        Models with ``active_users > 0`` are never evicted.
        """
        while self.vram_used_bytes + needed_bytes > self._budget_bytes:
            victim = self._pick_eviction_victim(exclude)
            if victim is None:
                # Nothing to evict — proceed anyway (may OOM)
                logger.warning(
                    "Cannot free enough VRAM: need %d MB but only %d MB available "
                    "(all loaded models are in active use)",
                    needed_bytes // (1024 * 1024),
                    self.vram_available_bytes // (1024 * 1024),
                )
                break
            logger.info("Evicting model '%s' to free VRAM", victim.name)
            self._do_unload(victim)
            self._total_evictions += 1

    def _pick_eviction_victim(self, exclude: str) -> ModelHandle | None:
        """Select the best model to evict.

        Criteria (in order):
            1. Skip models that are not loaded
            2. Skip models with active_users > 0
            3. Skip the model we're trying to load (*exclude*)
            4. Among candidates: lowest priority first, then oldest last_used
        """
        candidates: list[ModelHandle] = []
        for h in self._registry.values():
            if not h.loaded:
                continue
            if h.active_users > 0:
                continue
            if h.name == exclude:
                continue
            candidates.append(h)

        if not candidates:
            return None

        # Sort: lowest priority first, then oldest last_used
        candidates.sort(key=lambda h: (h.priority, h.last_used))
        return candidates[0]

    # ── Internal load / unload ────────────────────────────────────────

    def _do_load(self, handle: ModelHandle) -> None:
        """Load a model via its registered callback."""
        t0 = time.monotonic()
        try:
            handle.instance = handle.load_fn()
        except Exception:
            logger.exception("Failed to load model '%s'", handle.name)
            raise RuntimeError(f"Failed to load model '{handle.name}'") from None

        handle.loaded = True
        handle.last_used = time.monotonic()
        handle.load_count += 1
        self._total_loads += 1
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Loaded model '%s' in %.0f ms (vram_estimate=%.2f GB, total_loads=%d)",
            handle.name,
            elapsed_ms,
            handle.vram_estimate_bytes / (1024 ** 3),
            handle.load_count,
        )

    def _do_unload(self, handle: ModelHandle) -> None:
        """Unload a model via its registered callback."""
        if not handle.loaded:
            return
        try:
            handle.unload_fn(handle.instance)
        except Exception:
            logger.exception("Error unloading model '%s'", handle.name)
        handle.instance = None
        handle.loaded = False
        handle.active_users = 0
        logger.info("Unloaded model '%s'", handle.name)

    # ── Status / observability ────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Return current orchestrator status for API/debug."""
        with self._lock:
            models: list[dict[str, Any]] = []
            for h in self._registry.values():
                models.append(
                    {
                        "name": h.name,
                        "loaded": h.loaded,
                        "priority": h.priority,
                        "vram_estimate_mb": round(h.vram_estimate_bytes / (1024 * 1024)),
                        "active_users": h.active_users,
                        "load_count": h.load_count,
                        "device": h.device,
                    }
                )

            return {
                "device": self._device,
                "vram_budget_mb": round(self._budget_bytes / (1024 * 1024)),
                "vram_used_mb": round(self.vram_used_bytes / (1024 * 1024)),
                "vram_available_mb": round(self.vram_available_bytes / (1024 * 1024)),
                "total_loads": self._total_loads,
                "total_evictions": self._total_evictions,
                "models": models,
            }

    def is_loaded(self, name: str) -> bool:
        """Check whether a model is currently loaded."""
        with self._lock:
            h = self._registry.get(name)
            return h.loaded if h else False

    @property
    def loaded_models(self) -> list[str]:
        """Names of all currently loaded models."""
        with self._lock:
            return [h.name for h in self._registry.values() if h.loaded]

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Unload all models and clear the registry."""
        with self._lock:
            for h in list(self._registry.values()):
                if h.loaded:
                    self._do_unload(h)
            self._registry.clear()
            logger.info("ModelOrchestrator closed — all models unloaded")

    def __repr__(self) -> str:
        loaded = [h.name for h in self._registry.values() if h.loaded]
        return (
            f"ModelOrchestrator(device={self._device!r}, "
            f"loaded={loaded}, "
            f"vram_used={self.vram_used_bytes // (1024 * 1024)} MB)"
        )
