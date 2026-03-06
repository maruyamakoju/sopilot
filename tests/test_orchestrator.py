"""Tests for ModelOrchestrator — Phase 31.

Comprehensive tests for VRAM-aware model lifecycle management.
All tests use mock load/unload functions (no GPU required).
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from sopilot.perception.orchestrator import ModelHandle, ModelOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GB = 1024 ** 3


def _make_model(name: str = "test_model"):
    """Return a simple mock object as a 'loaded model'."""
    m = MagicMock(name=name)
    m.predict = MagicMock(return_value=[{"label": "person", "score": 0.95}])
    return m


def _load_fn():
    """Default mock load function."""
    return _make_model()


def _unload_fn(instance):
    """Default mock unload function."""
    pass


def _make_orch(budget_gb: float = 2.0) -> ModelOrchestrator:
    """Create a small-budget orchestrator on CPU for testing."""
    return ModelOrchestrator(vram_budget_gb=budget_gb, device="cpu")


def _register_simple(orch: ModelOrchestrator, name: str, vram_gb: float = 0.5,
                      priority: int = 5, load_fn=None, unload_fn=None):
    """Register a model with sensible defaults."""
    orch.register(
        name=name,
        load_fn=load_fn or _load_fn,
        unload_fn=unload_fn or _unload_fn,
        vram_estimate_gb=vram_gb,
        priority=priority,
    )


# ===========================================================================
# 1. ModelHandle dataclass
# ===========================================================================


class TestModelHandle:
    """Tests for the ModelHandle dataclass."""

    def test_defaults(self):
        h = ModelHandle(
            name="yolo",
            load_fn=_load_fn,
            unload_fn=_unload_fn,
            vram_estimate_bytes=300 * 1024 * 1024,
            priority=9,
        )
        assert h.name == "yolo"
        assert h.device == "cuda"
        assert h.instance is None
        assert h.loaded is False
        assert h.last_used == 0.0
        assert h.load_count == 0
        assert h.active_users == 0

    def test_custom_device(self):
        h = ModelHandle(
            name="cpu_model",
            load_fn=_load_fn,
            unload_fn=_unload_fn,
            vram_estimate_bytes=0,
            priority=1,
            device="cpu",
        )
        assert h.device == "cpu"


# ===========================================================================
# 2. Registration
# ===========================================================================


class TestRegistration:
    """Tests for model registration."""

    def test_register_single(self):
        orch = _make_orch()
        _register_simple(orch, "yolo")
        status = orch.get_status()
        assert len(status["models"]) == 1
        assert status["models"][0]["name"] == "yolo"

    def test_register_multiple(self):
        orch = _make_orch()
        for name in ("yolo", "dinov2", "clip"):
            _register_simple(orch, name)
        assert len(orch.get_status()["models"]) == 3

    def test_double_register_overwrites(self):
        """Re-registering the same name replaces the old handle."""
        orch = _make_orch()
        _register_simple(orch, "yolo", priority=5)
        _register_simple(orch, "yolo", priority=9)
        models = orch.get_status()["models"]
        assert len(models) == 1
        assert models[0]["priority"] == 9

    def test_double_register_unloads_previous(self):
        """Re-registering a loaded model unloads the old instance."""
        orch = _make_orch()
        unload = MagicMock()
        _register_simple(orch, "yolo", unload_fn=unload)
        orch.acquire("yolo")
        orch.release("yolo")
        # Re-register -> should call unload on old instance
        _register_simple(orch, "yolo", unload_fn=MagicMock())
        unload.assert_called_once()

    def test_register_with_explicit_device(self):
        orch = _make_orch()
        orch.register("model", _load_fn, _unload_fn, vram_estimate_gb=0.1,
                       priority=5, device="cuda")
        models = orch.get_status()["models"]
        assert models[0]["device"] == "cuda"

    def test_register_inherits_orchestrator_device(self):
        orch = _make_orch()
        _register_simple(orch, "model")
        models = orch.get_status()["models"]
        assert models[0]["device"] == "cpu"

    def test_unregister_known(self):
        orch = _make_orch()
        _register_simple(orch, "yolo")
        orch.acquire("yolo")
        orch.release("yolo")
        orch.unregister("yolo")
        assert len(orch.get_status()["models"]) == 0

    def test_unregister_unknown(self):
        """Unregistering a non-existent model does not raise."""
        orch = _make_orch()
        orch.unregister("nonexistent")  # should not raise

    def test_unregister_unloads_loaded_model(self):
        orch = _make_orch()
        unload = MagicMock()
        _register_simple(orch, "yolo", unload_fn=unload)
        orch.acquire("yolo")
        orch.release("yolo")
        orch.unregister("yolo")
        unload.assert_called_once()


# ===========================================================================
# 3. Acquire / Release
# ===========================================================================


class TestAcquireRelease:
    """Tests for acquire/release lifecycle."""

    def test_acquire_loads_on_first_call(self):
        orch = _make_orch()
        load = MagicMock(return_value="model_instance")
        _register_simple(orch, "yolo", load_fn=load)
        instance = orch.acquire("yolo")
        assert instance == "model_instance"
        load.assert_called_once()

    def test_acquire_returns_cached_on_second_call(self):
        orch = _make_orch()
        call_count = 0

        def counting_load():
            nonlocal call_count
            call_count += 1
            return f"instance_{call_count}"

        _register_simple(orch, "yolo", load_fn=counting_load)
        inst1 = orch.acquire("yolo")
        inst2 = orch.acquire("yolo")
        assert inst1 == inst2 == "instance_1"
        assert call_count == 1

    def test_acquire_unknown_raises_keyerror(self):
        orch = _make_orch()
        with pytest.raises(KeyError, match="not registered"):
            orch.acquire("nonexistent")

    def test_release_decrements_active_users(self):
        orch = _make_orch()
        _register_simple(orch, "yolo")
        orch.acquire("yolo")
        status_before = orch.get_status()
        assert status_before["models"][0]["active_users"] == 1
        orch.release("yolo")
        status_after = orch.get_status()
        assert status_after["models"][0]["active_users"] == 0

    def test_release_unknown_no_error(self):
        """Releasing a model that was never registered does not raise."""
        orch = _make_orch()
        orch.release("ghost")  # should not raise

    def test_release_already_released(self):
        """Releasing when active_users is already 0 does not go negative."""
        orch = _make_orch()
        _register_simple(orch, "yolo")
        orch.acquire("yolo")
        orch.release("yolo")
        orch.release("yolo")  # second release
        status = orch.get_status()
        assert status["models"][0]["active_users"] == 0

    def test_acquire_increments_load_count(self):
        orch = _make_orch()
        _register_simple(orch, "yolo")
        orch.acquire("yolo")
        orch.release("yolo")
        orch.evict("yolo")
        # Re-acquire should trigger another load
        orch.acquire("yolo")
        status = orch.get_status()
        assert status["models"][0]["load_count"] == 2

    def test_acquire_updates_last_used(self):
        orch = _make_orch()
        _register_simple(orch, "yolo")
        t0 = time.monotonic()
        orch.acquire("yolo")
        handle = orch._registry["yolo"]
        assert handle.last_used >= t0

    def test_acquire_concurrent_users(self):
        """Multiple acquires increment active_users correctly."""
        orch = _make_orch()
        _register_simple(orch, "yolo")
        orch.acquire("yolo")
        orch.acquire("yolo")
        orch.acquire("yolo")
        status = orch.get_status()
        assert status["models"][0]["active_users"] == 3
        orch.release("yolo")
        orch.release("yolo")
        status2 = orch.get_status()
        assert status2["models"][0]["active_users"] == 1

    def test_load_fn_exception_raises_runtime_error(self):
        """If load_fn raises, acquire wraps it in RuntimeError."""
        orch = _make_orch()

        def bad_load():
            raise ValueError("GPU exploded")

        _register_simple(orch, "broken", load_fn=bad_load)
        with pytest.raises(RuntimeError, match="Failed to load"):
            orch.acquire("broken")


# ===========================================================================
# 4. VRAM budget enforcement & eviction
# ===========================================================================


class TestVRAMBudget:
    """Tests for VRAM budget enforcement and eviction."""

    def test_vram_used_initially_zero(self):
        orch = _make_orch(budget_gb=2.0)
        assert orch.vram_used_bytes == 0

    def test_vram_used_after_load(self):
        orch = _make_orch(budget_gb=2.0)
        _register_simple(orch, "yolo", vram_gb=0.5)
        orch.acquire("yolo")
        expected = int(0.5 * _GB)
        assert orch.vram_used_bytes == expected

    def test_vram_available_decreases(self):
        orch = _make_orch(budget_gb=2.0)
        initial_avail = orch.vram_available_bytes
        _register_simple(orch, "yolo", vram_gb=0.5)
        orch.acquire("yolo")
        assert orch.vram_available_bytes == initial_avail - int(0.5 * _GB)

    def test_evict_frees_vram(self):
        orch = _make_orch(budget_gb=2.0)
        _register_simple(orch, "yolo", vram_gb=0.5)
        orch.acquire("yolo")
        orch.release("yolo")
        orch.evict("yolo")
        assert orch.vram_used_bytes == 0
        assert "yolo" not in orch.loaded_models

    def test_evict_unloaded_model_no_op(self):
        orch = _make_orch(budget_gb=2.0)
        _register_simple(orch, "yolo")
        orch.evict("yolo")  # not loaded -> no-op

    def test_evict_unknown_model_no_op(self):
        orch = _make_orch(budget_gb=2.0)
        orch.evict("ghost")  # not registered -> no-op

    def test_auto_eviction_on_budget_pressure(self):
        """When loading a model would exceed budget, existing models are evicted."""
        orch = _make_orch(budget_gb=1.0)
        _register_simple(orch, "model_a", vram_gb=0.6, priority=3)
        _register_simple(orch, "model_b", vram_gb=0.6, priority=7)

        orch.acquire("model_a")
        orch.release("model_a")

        # Loading model_b (0.6 GB) would push total to 1.2 GB > 1.0 GB budget
        # model_a (lower priority, not active) should be evicted
        orch.acquire("model_b")
        assert "model_a" not in orch.loaded_models
        assert "model_b" in orch.loaded_models

    def test_eviction_respects_priority_order(self):
        """Lower-priority models are evicted before higher-priority ones."""
        orch = _make_orch(budget_gb=1.5)
        _register_simple(orch, "low_pri", vram_gb=0.5, priority=1)
        _register_simple(orch, "high_pri", vram_gb=0.5, priority=9)
        _register_simple(orch, "new_model", vram_gb=0.8, priority=5)

        orch.acquire("low_pri")
        orch.release("low_pri")
        orch.acquire("high_pri")
        orch.release("high_pri")

        # Loading new_model (0.8 GB): total would be 1.8 GB > 1.5 GB
        # low_pri (priority=1) should be evicted first
        orch.acquire("new_model")
        assert "low_pri" not in orch.loaded_models
        assert "high_pri" in orch.loaded_models
        assert "new_model" in orch.loaded_models

    def test_eviction_lru_tiebreak(self):
        """Among same-priority models, least recently used is evicted first."""
        orch = _make_orch(budget_gb=1.5)
        _register_simple(orch, "old_model", vram_gb=0.5, priority=5)
        _register_simple(orch, "new_model", vram_gb=0.5, priority=5)
        _register_simple(orch, "trigger", vram_gb=0.8, priority=5)

        orch.acquire("old_model")
        orch.release("old_model")
        # Small delay to differentiate last_used timestamps
        time.sleep(0.01)
        orch.acquire("new_model")
        orch.release("new_model")

        # Loading trigger (0.8 GB): total would be 1.8 GB > 1.5 GB
        # old_model has older last_used, so it should be evicted first
        orch.acquire("trigger")
        assert "old_model" not in orch.loaded_models
        assert "new_model" in orch.loaded_models

    def test_active_users_protected_from_eviction(self):
        """Models with active_users > 0 are never evicted."""
        orch = _make_orch(budget_gb=1.0)
        _register_simple(orch, "active_model", vram_gb=0.5, priority=1)
        _register_simple(orch, "idle_model", vram_gb=0.3, priority=5)
        _register_simple(orch, "incoming", vram_gb=0.5, priority=9)

        orch.acquire("active_model")  # active_users = 1, NOT released
        orch.acquire("idle_model")
        orch.release("idle_model")  # active_users = 0, evictable

        # Loading incoming: total would be 1.3 GB > 1.0 GB
        # active_model has active_users=1, so idle_model should be evicted instead
        orch.acquire("incoming")
        assert "active_model" in orch.loaded_models
        assert "idle_model" not in orch.loaded_models
        assert "incoming" in orch.loaded_models

        orch.release("active_model")
        orch.release("incoming")

    def test_all_models_active_budget_exceeded_proceeds(self):
        """If all models are active and budget is exceeded, proceeds anyway (logs warning)."""
        orch = _make_orch(budget_gb=0.8)
        _register_simple(orch, "model_a", vram_gb=0.5, priority=5)
        _register_simple(orch, "model_b", vram_gb=0.5, priority=5)

        orch.acquire("model_a")  # NOT released -> active

        # Loading model_b: total 1.0 GB > 0.8 GB, but model_a is active
        # Should proceed with warning, not raise
        instance = orch.acquire("model_b")
        assert instance is not None
        assert "model_a" in orch.loaded_models
        assert "model_b" in orch.loaded_models

        orch.release("model_a")
        orch.release("model_b")

    def test_multiple_evictions_to_fit(self):
        """Multiple models may be evicted to fit a single large model."""
        orch = _make_orch(budget_gb=1.0)
        _register_simple(orch, "small_1", vram_gb=0.3, priority=1)
        _register_simple(orch, "small_2", vram_gb=0.3, priority=2)
        _register_simple(orch, "big", vram_gb=0.8, priority=9)

        orch.acquire("small_1")
        orch.release("small_1")
        orch.acquire("small_2")
        orch.release("small_2")

        # big (0.8 GB): total would be 1.4 GB > 1.0 GB
        # Need to evict both small models (priority 1, then 2)
        orch.acquire("big")
        assert "small_1" not in orch.loaded_models
        assert "small_2" not in orch.loaded_models
        assert "big" in orch.loaded_models

    def test_total_evictions_counter(self):
        orch = _make_orch(budget_gb=0.5)
        _register_simple(orch, "model_a", vram_gb=0.4, priority=1)
        _register_simple(orch, "model_b", vram_gb=0.4, priority=5)

        orch.acquire("model_a")
        orch.release("model_a")
        orch.acquire("model_b")  # evicts model_a

        status = orch.get_status()
        assert status["total_evictions"] == 1


# ===========================================================================
# 5. Context manager (use)
# ===========================================================================


class TestContextManager:
    """Tests for the `use()` context manager."""

    def test_use_basic(self):
        orch = _make_orch()
        _register_simple(orch, "yolo")
        with orch.use("yolo") as model:
            assert model is not None
            status = orch.get_status()
            assert status["models"][0]["active_users"] == 1
        # After exit, active_users should be 0
        status = orch.get_status()
        assert status["models"][0]["active_users"] == 0

    def test_use_releases_on_exception(self):
        """active_users is decremented even if an exception occurs inside the block."""
        orch = _make_orch()
        _register_simple(orch, "yolo")
        try:
            with orch.use("yolo"):
                raise ValueError("boom")
        except ValueError:
            pass
        status = orch.get_status()
        assert status["models"][0]["active_users"] == 0

    def test_use_unknown_raises(self):
        orch = _make_orch()
        with pytest.raises(KeyError):
            with orch.use("nonexistent"):
                pass

    def test_nested_use(self):
        """Nested `use()` for different models works correctly."""
        orch = _make_orch(budget_gb=2.0)
        _register_simple(orch, "yolo", vram_gb=0.5)
        _register_simple(orch, "clip", vram_gb=0.5)
        with orch.use("yolo") as m1:
            with orch.use("clip") as m2:
                assert m1 is not None
                assert m2 is not None
                assert len(orch.loaded_models) == 2
        assert orch.get_status()["models"][0]["active_users"] == 0
        assert orch.get_status()["models"][1]["active_users"] == 0


# ===========================================================================
# 6. Status / Observability
# ===========================================================================


class TestStatus:
    """Tests for get_status() and observability properties."""

    def test_get_status_structure(self):
        orch = _make_orch(budget_gb=2.0)
        _register_simple(orch, "yolo", vram_gb=0.5, priority=9)
        status = orch.get_status()
        assert "device" in status
        assert "vram_budget_mb" in status
        assert "vram_used_mb" in status
        assert "vram_available_mb" in status
        assert "total_loads" in status
        assert "total_evictions" in status
        assert "models" in status

    def test_get_status_model_fields(self):
        orch = _make_orch()
        _register_simple(orch, "yolo", vram_gb=0.5, priority=9)
        orch.acquire("yolo")
        model_info = orch.get_status()["models"][0]
        assert model_info["name"] == "yolo"
        assert model_info["loaded"] is True
        assert model_info["priority"] == 9
        assert model_info["active_users"] == 1
        assert model_info["load_count"] == 1
        assert model_info["vram_estimate_mb"] == round(0.5 * 1024)
        orch.release("yolo")

    def test_is_loaded(self):
        orch = _make_orch()
        _register_simple(orch, "yolo")
        assert orch.is_loaded("yolo") is False
        orch.acquire("yolo")
        assert orch.is_loaded("yolo") is True
        orch.release("yolo")
        orch.evict("yolo")
        assert orch.is_loaded("yolo") is False

    def test_is_loaded_unknown(self):
        orch = _make_orch()
        assert orch.is_loaded("ghost") is False

    def test_loaded_models_property(self):
        orch = _make_orch(budget_gb=2.0)
        _register_simple(orch, "yolo", vram_gb=0.3)
        _register_simple(orch, "clip", vram_gb=0.3)
        assert orch.loaded_models == []
        orch.acquire("yolo")
        assert orch.loaded_models == ["yolo"]
        orch.acquire("clip")
        assert set(orch.loaded_models) == {"yolo", "clip"}
        orch.release("yolo")
        orch.release("clip")

    def test_total_loads_counter(self):
        orch = _make_orch()
        _register_simple(orch, "yolo")
        _register_simple(orch, "clip")
        orch.acquire("yolo")
        orch.acquire("clip")
        status = orch.get_status()
        assert status["total_loads"] == 2
        orch.release("yolo")
        orch.release("clip")

    def test_device_property(self):
        orch = _make_orch()
        assert orch.device == "cpu"

    def test_has_gpu_cpu(self):
        orch = ModelOrchestrator(vram_budget_gb=1.0, device="cpu")
        assert orch.has_gpu is False

    def test_has_gpu_cuda(self):
        orch = ModelOrchestrator(vram_budget_gb=1.0, device="cuda")
        assert orch.has_gpu is True

    def test_has_gpu_mps(self):
        orch = ModelOrchestrator(vram_budget_gb=1.0, device="mps")
        assert orch.has_gpu is True

    def test_repr(self):
        orch = _make_orch()
        _register_simple(orch, "yolo")
        r = repr(orch)
        assert "ModelOrchestrator" in r
        assert "cpu" in r


# ===========================================================================
# 7. Close
# ===========================================================================


class TestClose:
    """Tests for close() lifecycle method."""

    def test_close_unloads_all(self):
        orch = _make_orch(budget_gb=2.0)
        unload_a = MagicMock()
        unload_b = MagicMock()
        _register_simple(orch, "model_a", unload_fn=unload_a)
        _register_simple(orch, "model_b", unload_fn=unload_b)
        orch.acquire("model_a")
        orch.acquire("model_b")
        orch.release("model_a")
        orch.release("model_b")
        orch.close()
        unload_a.assert_called_once()
        unload_b.assert_called_once()

    def test_close_clears_registry(self):
        orch = _make_orch()
        _register_simple(orch, "yolo")
        orch.close()
        assert len(orch.get_status()["models"]) == 0

    def test_close_on_empty_orchestrator(self):
        orch = _make_orch()
        orch.close()  # should not raise


# ===========================================================================
# 8. Thread safety
# ===========================================================================


class TestThreadSafety:
    """Tests for concurrent acquire/release from multiple threads."""

    def test_concurrent_acquire_release(self):
        """Multiple threads can acquire/release without data corruption."""
        orch = _make_orch(budget_gb=2.0)
        _register_simple(orch, "shared_model", vram_gb=0.5)
        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def worker():
            try:
                barrier.wait(timeout=5)
                with orch.use("shared_model") as model:
                    assert model is not None
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []
        status = orch.get_status()
        assert status["models"][0]["active_users"] == 0
        orch.close()

    def test_concurrent_different_models(self):
        """Multiple threads can use different models concurrently."""
        orch = _make_orch(budget_gb=3.0)
        model_names = ["model_a", "model_b", "model_c"]
        for name in model_names:
            _register_simple(orch, name, vram_gb=0.5)

        errors: list[Exception] = []
        barrier = threading.Barrier(3)

        def worker(name):
            try:
                barrier.wait(timeout=5)
                with orch.use(name) as model:
                    assert model is not None
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(n,)) for n in model_names]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []
        orch.close()


# ===========================================================================
# 9. Multi-model orchestration scenarios
# ===========================================================================


class TestMultiModelOrchestration:
    """End-to-end scenarios with 3+ models exercising eviction order."""

    def test_three_model_rotation(self):
        """Register 3 models on a budget that fits only 2; verify rotation."""
        orch = _make_orch(budget_gb=1.0)  # fits two 0.5 GB models
        _register_simple(orch, "yolo", vram_gb=0.5, priority=5)
        _register_simple(orch, "clip", vram_gb=0.5, priority=7)
        _register_simple(orch, "dino", vram_gb=0.5, priority=3)

        # Load yolo and clip (total 1.0 GB = budget)
        orch.acquire("yolo")
        orch.release("yolo")
        orch.acquire("clip")
        orch.release("clip")
        assert set(orch.loaded_models) == {"yolo", "clip"}

        # Loading dino should evict the lowest priority: dino is p=3 but we're
        # loading dino itself. Among loaded: yolo(p=5) vs clip(p=7).
        # yolo has lower priority, so yolo is evicted.
        orch.acquire("dino")
        orch.release("dino")
        assert "yolo" not in orch.loaded_models
        assert set(orch.loaded_models) == {"clip", "dino"}

    def test_four_model_cascade_eviction(self):
        """Large model requires evicting multiple small models."""
        orch = _make_orch(budget_gb=1.0)
        _register_simple(orch, "tiny_1", vram_gb=0.2, priority=1)
        _register_simple(orch, "tiny_2", vram_gb=0.2, priority=2)
        _register_simple(orch, "tiny_3", vram_gb=0.2, priority=3)
        _register_simple(orch, "large", vram_gb=0.9, priority=10)

        for name in ("tiny_1", "tiny_2", "tiny_3"):
            orch.acquire(name)
            orch.release(name)
        assert len(orch.loaded_models) == 3

        # large (0.9 GB): total would be 1.5 GB > 1.0 GB
        # Must evict at least 0.5 GB worth of tiny models
        # Eviction order by priority: tiny_1 (p=1), then tiny_2 (p=2), then tiny_3 (p=3)
        orch.acquire("large")
        orch.release("large")
        assert "large" in orch.loaded_models
        # At least tiny_1 and tiny_2 must be evicted (0.4 GB freed)
        # But we need 0.5 GB freed, so tiny_3 must also be evicted
        assert "tiny_1" not in orch.loaded_models
        assert "tiny_2" not in orch.loaded_models
        assert "tiny_3" not in orch.loaded_models

    def test_priority_inversion_protection(self):
        """High-priority active model survives over low-priority idle models."""
        orch = _make_orch(budget_gb=1.0)
        _register_simple(orch, "critical", vram_gb=0.4, priority=1)  # low prio but active
        _register_simple(orch, "optional_a", vram_gb=0.3, priority=8)  # high prio, idle
        _register_simple(orch, "optional_b", vram_gb=0.3, priority=9)  # highest prio, idle
        _register_simple(orch, "incoming", vram_gb=0.5, priority=5)

        orch.acquire("critical")  # NOT released -> active_users=1
        orch.acquire("optional_a")
        orch.release("optional_a")
        orch.acquire("optional_b")
        orch.release("optional_b")

        # Total loaded: 1.0 GB. Loading incoming (0.5 GB) needs 0.5 GB freed.
        # critical is active -> cannot evict. optional_a(p=8) < optional_b(p=9),
        # so optional_a evicted first.
        orch.acquire("incoming")
        assert "critical" in orch.loaded_models
        assert "incoming" in orch.loaded_models
        # optional_a should be evicted (lower priority among idle)
        assert "optional_a" not in orch.loaded_models

        orch.release("critical")
        orch.release("incoming")

    def test_reload_after_eviction(self):
        """Model can be re-acquired after being evicted."""
        orch = _make_orch(budget_gb=0.5)
        load_count = {"n": 0}

        def counting_load():
            load_count["n"] += 1
            return f"instance_{load_count['n']}"

        _register_simple(orch, "model_a", vram_gb=0.4, priority=3, load_fn=counting_load)
        _register_simple(orch, "model_b", vram_gb=0.4, priority=5)

        inst1 = orch.acquire("model_a")
        assert inst1 == "instance_1"
        orch.release("model_a")

        # Acquiring model_b evicts model_a
        orch.acquire("model_b")
        orch.release("model_b")
        assert "model_a" not in orch.loaded_models

        # Re-acquire model_a -> triggers new load
        inst2 = orch.acquire("model_a")
        assert inst2 == "instance_2"
        assert load_count["n"] == 2
        orch.release("model_a")


# ===========================================================================
# 10. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_vram_budget(self):
        """With zero budget, every acquire triggers eviction of others."""
        orch = _make_orch(budget_gb=0.0)
        _register_simple(orch, "model_a", vram_gb=0.1, priority=5)
        # Should proceed (with warning) since there's nothing to evict
        instance = orch.acquire("model_a")
        assert instance is not None
        orch.release("model_a")

    def test_zero_vram_estimate_model(self):
        """Model with 0 VRAM estimate is always loadable."""
        orch = _make_orch(budget_gb=0.01)
        _register_simple(orch, "lightweight", vram_gb=0.0, priority=5)
        instance = orch.acquire("lightweight")
        assert instance is not None
        orch.release("lightweight")

    def test_unload_fn_exception_handled(self):
        """Exceptions in unload_fn are caught and do not crash."""
        orch = _make_orch(budget_gb=0.5)

        def bad_unload(inst):
            raise RuntimeError("unload crash")

        _register_simple(orch, "fragile", vram_gb=0.4, priority=1, unload_fn=bad_unload)
        orch.acquire("fragile")
        orch.release("fragile")
        # evict should catch the exception
        orch.evict("fragile")
        assert not orch.is_loaded("fragile")

    def test_vram_available_never_negative(self):
        """vram_available_bytes clamps to 0, never goes negative."""
        orch = _make_orch(budget_gb=0.1)
        _register_simple(orch, "big", vram_gb=0.5, priority=5)
        # Force-load bypassing budget (all active scenario)
        orch.acquire("big")
        assert orch.vram_available_bytes == 0
        orch.release("big")

    def test_large_budget(self):
        """Orchestrator works with large budgets (e.g. 80 GB)."""
        orch = ModelOrchestrator(vram_budget_gb=80.0, device="cpu")
        _register_simple(orch, "huge", vram_gb=40.0, priority=5)
        orch.acquire("huge")
        assert orch.vram_used_bytes == int(40.0 * _GB)
        assert orch.vram_available_bytes == int(40.0 * _GB)
        orch.release("huge")

    def test_close_then_acquire_raises(self):
        """After close(), acquiring a model raises KeyError (registry cleared)."""
        orch = _make_orch()
        _register_simple(orch, "yolo")
        orch.close()
        with pytest.raises(KeyError):
            orch.acquire("yolo")

    def test_model_instance_available_in_context(self):
        """The object yielded by use() is the actual model instance."""
        sentinel = object()
        orch = _make_orch()
        orch.register("sentinel_model", lambda: sentinel, _unload_fn,
                       vram_estimate_gb=0.1, priority=5)
        with orch.use("sentinel_model") as model:
            assert model is sentinel

    def test_evict_clears_active_users(self):
        """_do_unload resets active_users to 0."""
        orch = _make_orch()
        _register_simple(orch, "yolo")
        orch.acquire("yolo")
        # Force evict even though active_users > 0
        orch.evict("yolo")
        handle = orch._registry["yolo"]
        assert handle.active_users == 0
        assert handle.loaded is False

    def test_status_budget_matches_init(self):
        orch = _make_orch(budget_gb=4.0)
        status = orch.get_status()
        assert status["vram_budget_mb"] == round(4.0 * 1024)
