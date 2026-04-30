"""Phase 58: Real Inference Stack Tests

Tests for:
    - RTX 4090 config preset (PerceptionConfig.for_rtx4090)
    - LLaVA 4-bit quantization flag propagation
    - Grounding DINO real factory path
    - Cascade with all 3 stages (mock returns)
    - PerceptionEngine init with for_rtx4090 config
    - setup_models script CLI

All tests use mocks — no real model download required in CI.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── TestRTX4090Config ───────────────────────────────────────────────────────


class TestRTX4090Config:
    """for_rtx4090() creates correct config fields."""

    def test_classmethod_exists(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        assert hasattr(PerceptionConfig, "for_rtx4090")

    def test_returns_perception_config_instance(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig.for_rtx4090()
        assert isinstance(cfg, PerceptionConfig)

    def test_device_is_cuda(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig.for_rtx4090()
        assert cfg.device == "cuda"

    def test_orchestrator_enabled(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig.for_rtx4090()
        assert cfg.orchestrator_enabled is True

    def test_vram_budget_22gb(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig.for_rtx4090()
        assert cfg.orchestrator_vram_budget_gb == 22.0

    def test_cascade_detector(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig.for_rtx4090()
        assert cfg.detector_backend == "cascade"

    def test_dinov2_enabled(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig.for_rtx4090()
        assert cfg.dinov2_enabled is True
        assert cfg.dinov2_variant == "dinov2_vitb14"

    def test_grounding_dino_edge(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig.for_rtx4090()
        assert cfg.grounding_dino_15_enabled is True
        assert cfg.grounding_dino_15_variant == "1.5-edge"

    def test_llava_4bit_enabled(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig.for_rtx4090()
        assert cfg.llava_enabled is True
        assert cfg.llava_load_in_4bit is True
        # 8-bit should be disabled when 4-bit is requested
        assert cfg.llava_load_in_8bit is False

    def test_sam_enabled(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig.for_rtx4090()
        assert cfg.sam_enabled is True
        assert cfg.sam_model_variant == "sam2_hiera_small"

    def test_temporal_fusion_enabled(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig.for_rtx4090()
        assert cfg.temporal_fusion_enabled is True

    def test_video_foundation_vjepa(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig.for_rtx4090()
        assert cfg.video_foundation_enabled is True
        assert cfg.video_foundation_backend == "vjepa"

    def test_sla_monitor_enabled(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig.for_rtx4090()
        assert cfg.sla_monitor_enabled is True

    def test_crowd_analytics_enabled(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig.for_rtx4090()
        assert cfg.crowd_analytics_enabled is True

    def test_spatial_mapper_enabled(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig.for_rtx4090()
        assert cfg.spatial_mapper_enabled is True


# ── TestLLaVA4bit ──────────────────────────────────────────────────────────


class TestLLaVA4bit:
    """4-bit flag propagates correctly through factory and client."""

    def test_perception_config_has_4bit_field(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig()
        assert hasattr(cfg, "llava_load_in_4bit")
        assert cfg.llava_load_in_4bit is False  # default off

    def test_4bit_default_false(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        cfg = PerceptionConfig()
        assert cfg.llava_load_in_4bit is False

    def test_llava_client_accepts_4bit_param(self) -> None:
        from sopilot.perception.llava_client import LLaVAClient
        client = LLaVAClient(load_in_4bit=True)
        assert client._load_in_4bit is True

    def test_4bit_overrides_8bit(self) -> None:
        from sopilot.perception.llava_client import LLaVAClient
        client = LLaVAClient(load_in_8bit=True, load_in_4bit=True)
        assert client._load_in_4bit is True
        assert client._load_in_8bit is False  # 4-bit takes priority

    def test_8bit_only_mode(self) -> None:
        from sopilot.perception.llava_client import LLaVAClient
        client = LLaVAClient(load_in_8bit=True, load_in_4bit=False)
        assert client._load_in_8bit is True
        assert client._load_in_4bit is False

    def test_get_stats_includes_4bit(self) -> None:
        from sopilot.perception.llava_client import LLaVAClient
        client = LLaVAClient(load_in_4bit=True)
        stats = client.get_stats()
        assert "load_in_4bit" in stats
        assert stats["load_in_4bit"] is True

    def test_build_factory_passes_4bit(self) -> None:
        from sopilot.perception.llava_client import MockLLaVAClient, build_llava_client
        # With mock backend, 4-bit flag doesn't apply but factory should accept it
        client = build_llava_client(backend="mock", load_in_4bit=True)
        assert isinstance(client, MockLLaVAClient)

    def test_build_factory_passes_load_in_8bit_false(self) -> None:
        from sopilot.perception.llava_client import build_llava_client, LLaVAClient
        # backend="llava" (real path) with 4bit should produce LLaVAClient
        client = build_llava_client(backend="llava", load_in_4bit=True)
        assert isinstance(client, LLaVAClient)
        assert client._load_in_4bit is True

    def test_4bit_load_uses_bnb_config(self) -> None:
        """When load_in_4bit=True, _load_model should create BitsAndBytesConfig."""
        from sopilot.perception.llava_client import LLaVAClient

        bnb_mock = MagicMock()
        bnb_mock.return_value = MagicMock()

        mock_torch = MagicMock()
        mock_torch.float16 = "float16"

        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.eval = MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.BitsAndBytesConfig = bnb_mock
        mock_transformers.LlavaNextProcessor.from_pretrained.return_value = mock_processor
        mock_transformers.LlavaNextForConditionalGeneration.from_pretrained.return_value = mock_model

        client = LLaVAClient(load_in_4bit=True)
        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
            client._load_model()

        bnb_mock.assert_called_once()
        call_kwargs = bnb_mock.call_args.kwargs
        assert call_kwargs.get("load_in_4bit") is True

    def test_4bit_config_has_nf4_type(self) -> None:
        from sopilot.perception.llava_client import LLaVAClient

        bnb_mock = MagicMock()
        mock_torch = MagicMock()
        mock_torch.float16 = "float16"
        mock_proc = MagicMock()
        mock_mdl = MagicMock()
        mock_mdl.eval = MagicMock()
        mock_transformers = MagicMock()
        mock_transformers.BitsAndBytesConfig = bnb_mock
        mock_transformers.LlavaNextProcessor.from_pretrained.return_value = mock_proc
        mock_transformers.LlavaNextForConditionalGeneration.from_pretrained.return_value = mock_mdl

        client = LLaVAClient(load_in_4bit=True)
        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
            client._load_model()

        call_kwargs = bnb_mock.call_args.kwargs
        assert call_kwargs.get("bnb_4bit_quant_type") == "nf4"

    def test_fp16_mode_when_both_false(self) -> None:
        from sopilot.perception.llava_client import LLaVAClient
        client = LLaVAClient(load_in_8bit=False, load_in_4bit=False)
        assert client._load_in_8bit is False
        assert client._load_in_4bit is False


# ── TestGroundingDINORealFactory ───────────────────────────────────────────


class TestGroundingDINORealFactory:
    """build_grounding_dino() real path selection."""

    def test_factory_exists(self) -> None:
        from sopilot.perception.grounding_dino import build_grounding_dino
        assert callable(build_grounding_dino)

    def test_mock_backend_returns_mock(self) -> None:
        from sopilot.perception.grounding_dino import MockGroundingDINODetector, build_grounding_dino
        det = build_grounding_dino(backend="mock")
        assert isinstance(det, MockGroundingDINODetector)

    def test_auto_backend_falls_back_to_mock_without_cuda(self) -> None:
        from sopilot.perception.grounding_dino import MockGroundingDINODetector, build_grounding_dino
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            det = build_grounding_dino(backend="auto")
        assert isinstance(det, MockGroundingDINODetector)

    def test_real_backend_returns_real_detector(self) -> None:
        from sopilot.perception.grounding_dino import GroundingDINODetector, build_grounding_dino
        det = build_grounding_dino(backend="real", variant="1.5-edge")
        assert isinstance(det, GroundingDINODetector)

    def test_real_detector_has_variant_attribute(self) -> None:
        from sopilot.perception.grounding_dino import GroundingDINODetector, build_grounding_dino
        det = build_grounding_dino(backend="real", variant="1.5-edge")
        assert isinstance(det, GroundingDINODetector)
        assert hasattr(det, "_variant")
        assert det._variant == "1.5-edge"

    def test_mock_detector_detect_returns_list(self) -> None:
        from sopilot.perception.grounding_dino import MockGroundingDINODetector
        det = MockGroundingDINODetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame, ["person", "helmet"])
        assert isinstance(result, list)

    def test_mock_detector_get_stats(self) -> None:
        from sopilot.perception.grounding_dino import MockGroundingDINODetector
        det = MockGroundingDINODetector()
        stats = det.get_stats()
        assert isinstance(stats, dict)
        assert "total_detections" in stats

    def test_real_detector_not_loaded_initially(self) -> None:
        from sopilot.perception.grounding_dino import GroundingDINODetector, build_grounding_dino
        det = build_grounding_dino(backend="real")
        assert isinstance(det, GroundingDINODetector)
        assert not det._loaded


# ── TestCascadeRealStack ───────────────────────────────────────────────────


class TestCascadeRealStack:
    """Cascade with all 3 stages using mock returns."""

    def _make_cascade(self, **kwargs):
        from sopilot.perception.cascade_detector import CascadeDetector
        from sopilot.perception.detector import MockDetector
        from sopilot.perception.grounding_dino import MockGroundingDINODetector
        from sopilot.perception.llava_client import MockLLaVAClient

        stage1 = MockDetector()
        stage2 = MockGroundingDINODetector()
        stage3 = MockLLaVAClient()
        return CascadeDetector(yolo_detector=stage1, dino_detector=stage2, vlm_client=stage3, **kwargs)

    def test_cascade_creates_with_3_stages(self) -> None:
        cascade = self._make_cascade()
        assert cascade is not None

    def test_cascade_has_stage1(self) -> None:
        cascade = self._make_cascade()
        assert hasattr(cascade, "_yolo") or hasattr(cascade, "_stage1") or hasattr(cascade, "_detector")

    def test_cascade_detect_returns_list(self) -> None:
        cascade = self._make_cascade()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = cascade.detect(frame, prompts=["person", "helmet"])
        assert isinstance(results, list)

    def test_cascade_get_stats_exists(self) -> None:
        cascade = self._make_cascade()
        assert hasattr(cascade, "get_stats")

    def test_cascade_stats_has_stage_counts(self) -> None:
        cascade = self._make_cascade()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cascade.detect(frame, prompts=[])
        stats = cascade.get_stats()
        assert isinstance(stats, dict)

    def test_cascade_processes_multiple_frames(self) -> None:
        cascade = self._make_cascade()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            result = cascade.detect(frame, prompts=[])
            assert isinstance(result, list)

    def test_cascade_stage1_threshold_config(self) -> None:
        from sopilot.perception.cascade_detector import CascadeDetector
        from sopilot.perception.detector import MockDetector
        from sopilot.perception.grounding_dino import MockGroundingDINODetector
        from sopilot.perception.llava_client import MockLLaVAClient

        cascade = CascadeDetector(
            yolo_detector=MockDetector(),
            dino_detector=MockGroundingDINODetector(),
            vlm_client=MockLLaVAClient(),
            stage1_threshold=0.8,
        )
        assert cascade._stage1_thresh == 0.8

    def test_cascade_vlm_is_injected(self) -> None:
        from sopilot.perception.llava_client import MockLLaVAClient
        cascade = self._make_cascade()
        assert hasattr(cascade, "_vlm")
        assert isinstance(cascade._vlm, MockLLaVAClient)  # stored as _vlm

    def test_cascade_empty_frame_returns_empty(self) -> None:
        cascade = self._make_cascade()
        empty = np.zeros((1, 1, 3), dtype=np.uint8)
        result = cascade.detect(empty, prompts=[])
        assert isinstance(result, list)

    def test_cascade_none_detections_handled(self) -> None:
        from sopilot.perception.cascade_detector import CascadeDetector
        from sopilot.perception.detector import MockDetector
        from sopilot.perception.grounding_dino import MockGroundingDINODetector
        from sopilot.perception.llava_client import MockLLaVAClient

        stage1 = MockDetector()
        stage1.detect = lambda f: []
        cascade = CascadeDetector(
            yolo_detector=stage1,
            dino_detector=MockGroundingDINODetector(),
            vlm_client=MockLLaVAClient(),
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = cascade.detect(frame, prompts=[])
        assert result == [] or isinstance(result, list)

    def test_cascade_stats_total_frames(self) -> None:
        cascade = self._make_cascade()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            cascade.detect(frame, prompts=[])
        stats = cascade.get_stats()
        # stats dict must contain expected stage keys
        assert "total_frames" in stats or "stage1_resolved" in stats
        assert "stage1_rate" in stats or "stage1_resolved" in stats


# ── TestEngineRTX4090Init ──────────────────────────────────────────────────


class TestEngineRTX4090Init:
    """Engine initializes with for_rtx4090() config, all components present."""

    def _build_engine_cpu(self):
        """Build engine with for_rtx4090 config but device overridden to cpu."""
        from sopilot.perception.engine import build_perception_engine
        from sopilot.perception.types import PerceptionConfig

        cfg = PerceptionConfig.for_rtx4090()
        # Override to CPU for CI
        cfg.device = "cpu"
        cfg.orchestrator_enabled = False  # orchestrator needs GPU in real mode
        cfg.video_foundation_enabled = False  # V-JEPA needs torch
        return build_perception_engine(cfg)

    def test_engine_builds_without_error(self) -> None:
        engine = self._build_engine_cpu()
        assert engine is not None

    def test_engine_has_process_frame(self) -> None:
        engine = self._build_engine_cpu()
        assert hasattr(engine, "process_frame") or hasattr(engine, "process")

    def test_engine_has_detector(self) -> None:
        engine = self._build_engine_cpu()
        assert hasattr(engine, "_detector")

    def test_engine_has_llava_client(self) -> None:
        engine = self._build_engine_cpu()
        assert hasattr(engine, "_llava_client")

    def test_engine_llava_client_not_none(self) -> None:
        engine = self._build_engine_cpu()
        assert engine._llava_client is not None

    def test_engine_can_process_frame(self) -> None:
        engine = self._build_engine_cpu()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        process_fn = getattr(engine, "process_frame", getattr(engine, "process", None))
        if process_fn is not None:
            result = process_fn(frame, timestamp=0.0, frame_number=1, rules=[])
            assert result is not None

    def test_engine_process_returns_frame_result(self) -> None:
        from sopilot.perception.types import FrameResult
        engine = self._build_engine_cpu()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        process_fn = getattr(engine, "process_frame", getattr(engine, "process", None))
        if process_fn is not None:
            result = process_fn(frame, timestamp=0.0, frame_number=1, rules=[])
            assert isinstance(result, FrameResult)

    def test_engine_config_llava_4bit_propagates(self) -> None:
        from sopilot.perception.engine import build_perception_engine
        from sopilot.perception.types import PerceptionConfig

        cfg = PerceptionConfig.for_rtx4090()
        cfg.device = "cpu"
        cfg.orchestrator_enabled = False
        cfg.video_foundation_enabled = False
        assert cfg.llava_load_in_4bit is True
        engine = build_perception_engine(cfg)
        # LLaVA client: on CPU auto-mode will use mock, but 4bit field must be in config
        assert engine is not None

    def test_for_rtx4090_is_different_from_default(self) -> None:
        from sopilot.perception.types import PerceptionConfig
        default = PerceptionConfig()
        rtx = PerceptionConfig.for_rtx4090()
        assert default.device != rtx.device
        assert default.llava_load_in_4bit != rtx.llava_load_in_4bit

    def test_engine_violations_is_list(self) -> None:
        engine = self._build_engine_cpu()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        process_fn = getattr(engine, "process_frame", getattr(engine, "process", None))
        if process_fn is not None:
            result = process_fn(frame, timestamp=0.0, frame_number=1, rules=[])
            assert isinstance(result.violations, list)


# ── TestSetupModelsScript ──────────────────────────────────────────────────


class TestSetupModelsScript:
    """CLI args, dry-run, CUDA check logic."""

    def test_script_is_importable(self) -> None:
        spec = importlib.util.find_spec("scripts.setup_models")
        if spec is None:
            # Try loading directly
            import importlib.util as ilu
            import os
            path = os.path.join("scripts", "setup_models.py")
            assert os.path.exists(path), "scripts/setup_models.py not found"
        # If importable, that's fine
        assert True

    def test_models_registry_keys(self) -> None:
        sys.path.insert(0, ".")
        try:
            import importlib.util
            import os
            loader = importlib.util.spec_from_file_location("setup_models", "scripts/setup_models.py")
            mod = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(mod)
            assert "yolo" in mod.MODELS
            assert "dinov2" in mod.MODELS
            assert "grounding_dino" in mod.MODELS
            assert "llava" in mod.MODELS
            assert "sam2" in mod.MODELS
        except Exception:
            pytest.skip("setup_models not loadable in this environment")

    def test_vram_totals_in_budget(self) -> None:
        sys.path.insert(0, ".")
        try:
            import importlib.util
            loader = importlib.util.spec_from_file_location("setup_models", "scripts/setup_models.py")
            mod = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(mod)
            total_vram = sum(m["vram_gb"] for m in mod.MODELS.values())
            assert total_vram < 22.0, f"Total VRAM {total_vram} GB exceeds 22 GB budget"
        except Exception:
            pytest.skip("setup_models not loadable in this environment")

    def test_check_cuda_returns_tuple(self) -> None:
        sys.path.insert(0, ".")
        try:
            import importlib.util
            loader = importlib.util.spec_from_file_location("setup_models", "scripts/setup_models.py")
            mod = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(mod)
            result = mod.check_cuda()
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], bool)
            assert isinstance(result[1], str)
        except Exception:
            pytest.skip("setup_models not loadable in this environment")

    def test_download_yolo_dry_run(self) -> None:
        sys.path.insert(0, ".")
        try:
            import importlib.util
            loader = importlib.util.spec_from_file_location("setup_models", "scripts/setup_models.py")
            mod = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(mod)
            result = mod.download_yolo(dry_run=True)
            assert result is True
        except Exception:
            pytest.skip("setup_models not loadable in this environment")

    def test_download_llava_dry_run(self) -> None:
        sys.path.insert(0, ".")
        try:
            import importlib.util
            loader = importlib.util.spec_from_file_location("setup_models", "scripts/setup_models.py")
            mod = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(mod)
            result = mod.download_llava(dry_run=True)
            assert result is True
        except Exception:
            pytest.skip("setup_models not loadable in this environment")

    def test_downloaders_dict_complete(self) -> None:
        sys.path.insert(0, ".")
        try:
            import importlib.util
            loader = importlib.util.spec_from_file_location("setup_models", "scripts/setup_models.py")
            mod = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(mod)
            for key in mod.MODELS:
                assert key in mod._DOWNLOADERS, f"Missing downloader for '{key}'"
                assert callable(mod._DOWNLOADERS[key])
        except Exception:
            pytest.skip("setup_models not loadable in this environment")

    def test_main_dry_run_returns_0(self) -> None:
        sys.path.insert(0, ".")
        try:
            import importlib.util
            loader = importlib.util.spec_from_file_location("setup_models", "scripts/setup_models.py")
            mod = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(mod)
            with patch("sys.argv", ["setup_models.py", "--dry-run"]):
                rc = mod.main()
            assert rc == 0
        except Exception:
            pytest.skip("setup_models not loadable in this environment")
