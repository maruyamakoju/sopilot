"""Tests for sopilot.perception.llava_client (Phase 38 — LLaVA-NeXT VLM Client).

Covers:
    - VLMClient ABC enforcement
    - MockLLaVAClient: analyze_detections, describe_scene, answer_question, stats
    - LLaVAStats: arithmetic and to_dict
    - build_llava_client factory: mock/auto backends
    - Integration with Detection dataclass
    - Edge cases: empty frame, empty detections, empty question
    - Thread-safety of MockLLaVAClient
    - Rejection path (confirm_all=False)
    - Confidence clamping
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sopilot.perception.llava_client import (
    LLaVAStats,
    MockLLaVAClient,
    VLMClient,
    build_llava_client,
)
from sopilot.perception.types import BBox, Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_detection(label: str, confidence: float = 0.4) -> Detection:
    return Detection(
        bbox=BBox(x1=0.1, y1=0.1, x2=0.4, y2=0.6),
        label=label,
        confidence=confidence,
    )


def make_frame(h: int = 120, w: int = 160) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# LLaVAStats
# ---------------------------------------------------------------------------


class TestLLaVAStats:
    def test_initial_state(self):
        s = LLaVAStats()
        assert s.total_calls == 0
        assert s.describe_calls == 0
        assert s.analyze_calls == 0
        assert s.qa_calls == 0
        assert s.errors == 0
        assert s.total_latency_ms == 0.0

    def test_avg_latency_zero_calls(self):
        s = LLaVAStats()
        assert s.avg_latency_ms == 0.0

    def test_avg_latency_with_calls(self):
        s = LLaVAStats(total_calls=4, total_latency_ms=200.0)
        assert s.avg_latency_ms == 50.0

    def test_to_dict_keys(self):
        s = LLaVAStats()
        d = s.to_dict()
        assert "total_calls" in d
        assert "describe_calls" in d
        assert "analyze_calls" in d
        assert "qa_calls" in d
        assert "errors" in d
        assert "avg_latency_ms" in d

    def test_to_dict_values(self):
        s = LLaVAStats(total_calls=2, qa_calls=2, total_latency_ms=100.0)
        d = s.to_dict()
        assert d["total_calls"] == 2
        assert d["avg_latency_ms"] == 50.0

    def test_to_dict_rounded(self):
        s = LLaVAStats(total_calls=3, total_latency_ms=100.0)
        d = s.to_dict()
        assert isinstance(d["avg_latency_ms"], float)


# ---------------------------------------------------------------------------
# VLMClient ABC
# ---------------------------------------------------------------------------


class TestVLMClientABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            VLMClient()  # type: ignore[abstract]

    def test_must_implement_analyze_detections(self):
        class Partial(VLMClient):
            def describe_scene(self, frame, language="ja"):
                return {}

            def answer_question(self, frame, question):
                return ""

            def get_stats(self):
                return {}

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_must_implement_describe_scene(self):
        class Partial(VLMClient):
            def analyze_detections(self, frame, uncertain, prompts):
                return []

            def answer_question(self, frame, question):
                return ""

            def get_stats(self):
                return {}

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_must_implement_answer_question(self):
        class Partial(VLMClient):
            def analyze_detections(self, frame, uncertain, prompts):
                return []

            def describe_scene(self, frame, language="ja"):
                return {}

            def get_stats(self):
                return {}

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_full_implementation_works(self):
        class Full(VLMClient):
            def analyze_detections(self, frame, uncertain, prompts):
                return []

            def describe_scene(self, frame, language="ja"):
                return {}

            def answer_question(self, frame, question):
                return "ok"

            def get_stats(self):
                return {}

        c = Full()
        assert c.answer_question(None, "q") == "ok"


# ---------------------------------------------------------------------------
# MockLLaVAClient — analyze_detections
# ---------------------------------------------------------------------------


class TestMockLLaVAClientAnalyze:
    def test_confirms_all_detections(self):
        client = MockLLaVAClient(confirm_all=True, confidence_boost=0.05)
        frame = make_frame()
        dets = [make_detection("person", 0.4), make_detection("helmet", 0.35)]
        result = client.analyze_detections(frame, dets, ["person", "helmet"])
        assert len(result) == 2

    def test_rejects_all_when_confirm_false(self):
        client = MockLLaVAClient(confirm_all=False)
        frame = make_frame()
        dets = [make_detection("person"), make_detection("helmet")]
        result = client.analyze_detections(frame, dets, ["person", "helmet"])
        assert result == []

    def test_confidence_boosted(self):
        client = MockLLaVAClient(confirm_all=True, confidence_boost=0.1)
        frame = make_frame()
        det = make_detection("person", confidence=0.5)
        result = client.analyze_detections(frame, [det], ["person"])
        assert result[0].confidence == pytest.approx(0.6, abs=1e-6)

    def test_confidence_clamped_to_1(self):
        client = MockLLaVAClient(confirm_all=True, confidence_boost=0.5)
        frame = make_frame()
        det = make_detection("person", confidence=0.9)
        result = client.analyze_detections(frame, [det], ["person"])
        assert result[0].confidence <= 1.0

    def test_vlm_mock_attribute_set(self):
        client = MockLLaVAClient(confirm_all=True)
        frame = make_frame()
        det = make_detection("vest")
        result = client.analyze_detections(frame, [det], ["vest"])
        assert result[0].attributes.get("vlm_mock") is True
        assert result[0].attributes.get("vlm_verified") is True

    def test_empty_detections(self):
        client = MockLLaVAClient()
        frame = make_frame()
        result = client.analyze_detections(frame, [], ["person"])
        assert result == []

    def test_stats_incremented(self):
        client = MockLLaVAClient()
        frame = make_frame()
        client.analyze_detections(frame, [make_detection("person")], ["person"])
        s = client.get_stats()
        assert s["analyze_calls"] == 1
        assert s["total_calls"] == 1

    def test_latency_tracked(self):
        client = MockLLaVAClient(latency_ms=100.0)
        frame = make_frame()
        client.analyze_detections(frame, [make_detection("person")], ["person"])
        s = client.get_stats()
        assert s["avg_latency_ms"] == pytest.approx(100.0, abs=1.0)

    def test_label_preserved(self):
        client = MockLLaVAClient(confirm_all=True)
        frame = make_frame()
        det = make_detection("safety_cone", confidence=0.3)
        result = client.analyze_detections(frame, [det], ["safety_cone"])
        assert result[0].label == "safety_cone"

    def test_bbox_preserved(self):
        client = MockLLaVAClient(confirm_all=True)
        frame = make_frame()
        det = make_detection("person", confidence=0.4)
        result = client.analyze_detections(frame, [det], ["person"])
        assert result[0].bbox == det.bbox


# ---------------------------------------------------------------------------
# MockLLaVAClient — describe_scene
# ---------------------------------------------------------------------------


class TestMockLLaVAClientDescribe:
    def test_returns_dict(self):
        client = MockLLaVAClient()
        frame = make_frame()
        result = client.describe_scene(frame, language="ja")
        assert isinstance(result, dict)

    def test_has_description_ja(self):
        client = MockLLaVAClient()
        frame = make_frame()
        result = client.describe_scene(frame, language="ja")
        assert "description_ja" in result
        assert len(result["description_ja"]) > 0

    def test_has_description_en(self):
        client = MockLLaVAClient()
        frame = make_frame()
        result = client.describe_scene(frame, language="en")
        assert "description_en" in result

    def test_mock_flag_set(self):
        client = MockLLaVAClient()
        frame = make_frame()
        result = client.describe_scene(frame)
        assert result.get("mock") is True

    def test_frame_size_in_description(self):
        client = MockLLaVAClient()
        frame = make_frame(h=240, w=320)
        result = client.describe_scene(frame, language="both")
        assert "320" in result["description_en"] or "320" in result["description_ja"]

    def test_stats_incremented(self):
        client = MockLLaVAClient()
        frame = make_frame()
        client.describe_scene(frame)
        client.describe_scene(frame)
        s = client.get_stats()
        assert s["describe_calls"] == 2
        assert s["total_calls"] == 2

    def test_language_field_present(self):
        client = MockLLaVAClient()
        frame = make_frame()
        result = client.describe_scene(frame, language="both")
        assert result.get("language") == "both"

    def test_objects_list_present(self):
        client = MockLLaVAClient()
        frame = make_frame()
        result = client.describe_scene(frame)
        assert "objects" in result
        assert isinstance(result["objects"], list)

    def test_safety_issues_list(self):
        client = MockLLaVAClient()
        frame = make_frame()
        result = client.describe_scene(frame)
        assert "safety_issues" in result
        assert isinstance(result["safety_issues"], list)

    def test_latency_ms_field(self):
        client = MockLLaVAClient(latency_ms=80.0)
        frame = make_frame()
        result = client.describe_scene(frame)
        assert "latency_ms" in result


# ---------------------------------------------------------------------------
# MockLLaVAClient — answer_question
# ---------------------------------------------------------------------------


class TestMockLLaVAClientQA:
    def test_returns_string(self):
        client = MockLLaVAClient()
        frame = make_frame()
        answer = client.answer_question(frame, "何人いる？")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_person_count_question(self):
        client = MockLLaVAClient()
        frame = make_frame()
        answer = client.answer_question(frame, "何人いる？")
        assert "1" in answer

    def test_helmet_question(self):
        client = MockLLaVAClient()
        frame = make_frame()
        answer = client.answer_question(frame, "ヘルメットを確認してください")
        assert len(answer) > 0

    def test_violation_question(self):
        client = MockLLaVAClient()
        frame = make_frame()
        answer = client.answer_question(frame, "違反はありますか？")
        assert len(answer) > 0

    def test_vest_question(self):
        client = MockLLaVAClient()
        frame = make_frame()
        answer = client.answer_question(frame, "ベストを着用していますか？")
        assert len(answer) > 0

    def test_generic_question(self):
        client = MockLLaVAClient()
        frame = make_frame()
        answer = client.answer_question(frame, "What is happening?")
        assert len(answer) > 0

    def test_stats_incremented(self):
        client = MockLLaVAClient()
        frame = make_frame()
        client.answer_question(frame, "question 1")
        client.answer_question(frame, "question 2")
        s = client.get_stats()
        assert s["qa_calls"] == 2
        assert s["total_calls"] == 2

    def test_how_many_english(self):
        client = MockLLaVAClient()
        frame = make_frame()
        answer = client.answer_question(frame, "how many people are there?")
        assert "1" in answer


# ---------------------------------------------------------------------------
# MockLLaVAClient — get_stats / close
# ---------------------------------------------------------------------------


class TestMockLLaVAClientStats:
    def test_initial_stats(self):
        client = MockLLaVAClient()
        s = client.get_stats()
        assert s["model_id"] == "mock"
        assert s["loaded"] is True
        assert s["mock"] is True
        assert s["total_calls"] == 0

    def test_confirm_all_in_stats(self):
        client = MockLLaVAClient(confirm_all=False)
        s = client.get_stats()
        assert s["confirm_all"] is False

    def test_close_is_noop(self):
        client = MockLLaVAClient()
        client.close()  # should not raise
        assert client.get_stats()["loaded"] is True

    def test_cumulative_stats(self):
        client = MockLLaVAClient(latency_ms=20.0)
        frame = make_frame()
        det = make_detection("person")
        client.analyze_detections(frame, [det], ["person"])
        client.describe_scene(frame)
        client.answer_question(frame, "?")
        s = client.get_stats()
        assert s["total_calls"] == 3
        assert s["analyze_calls"] == 1
        assert s["describe_calls"] == 1
        assert s["qa_calls"] == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestMockLLaVAThreadSafety:
    def test_concurrent_analyze(self):
        client = MockLLaVAClient(confirm_all=True)
        frame = make_frame()
        results = []
        errors = []

        def worker():
            try:
                det = make_detection("person", 0.4)
                r = client.analyze_detections(frame, [det], ["person"])
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 20

    def test_concurrent_describe(self):
        client = MockLLaVAClient()
        frame = make_frame()
        results = []
        errors = []

        def worker():
            try:
                results.append(client.describe_scene(frame))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all("description_ja" in r for r in results)


# ---------------------------------------------------------------------------
# build_llava_client factory
# ---------------------------------------------------------------------------


class TestBuildLLaVAClient:
    def test_mock_backend(self):
        client = build_llava_client(backend="mock")
        assert isinstance(client, MockLLaVAClient)

    def test_auto_no_cuda_returns_mock(self):
        with patch("torch.cuda.is_available", return_value=False):
            client = build_llava_client(backend="auto")
        assert isinstance(client, MockLLaVAClient)

    def test_auto_no_torch_returns_mock(self):
        import sys
        original = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            client = build_llava_client(backend="auto")
            assert isinstance(client, MockLLaVAClient)
        finally:
            if original is not None:
                sys.modules["torch"] = original
            else:
                del sys.modules["torch"]

    def test_mock_client_is_vlm_client(self):
        client = build_llava_client(backend="mock")
        assert isinstance(client, VLMClient)

    def test_mock_client_functional(self):
        client = build_llava_client(backend="mock")
        frame = make_frame()
        result = client.describe_scene(frame, language="ja")
        assert "description_ja" in result

    def test_custom_model_id_passed_to_real(self):
        """With backend='llava' and no CUDA, should still construct LLaVAClient."""
        from sopilot.perception.llava_client import LLaVAClient
        client = build_llava_client(
            model_id="llava-hf/llava-v1.6-vicuna-7b-hf",
            backend="llava",
        )
        assert isinstance(client, LLaVAClient)
        assert client._model_id == "llava-hf/llava-v1.6-vicuna-7b-hf"

    def test_max_new_tokens_forwarded(self):
        from sopilot.perception.llava_client import LLaVAClient
        client = build_llava_client(max_new_tokens=128, backend="llava")
        assert isinstance(client, LLaVAClient)
        assert client._max_new_tokens == 128

    def test_mock_analyze_returns_list(self):
        client = build_llava_client(backend="mock")
        frame = make_frame()
        result = client.analyze_detections(frame, [make_detection("person")], ["person"])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# LLaVAClient (real) — unloaded behavior
# ---------------------------------------------------------------------------


class TestLLaVAClientUnloaded:
    def test_not_loaded_by_default(self):
        from sopilot.perception.llava_client import LLaVAClient

        client = LLaVAClient(model_id="llava-hf/llava-v1.6-mistral-7b-hf")
        assert client._loaded is False

    def test_stats_show_not_loaded(self):
        from sopilot.perception.llava_client import LLaVAClient

        client = LLaVAClient()
        s = client.get_stats()
        assert s["loaded"] is False

    def test_analyze_with_load_failure_penalizes_confidence(self):
        from sopilot.perception.llava_client import LLaVAClient

        client = LLaVAClient()
        # Patch _ensure_loaded to simulate permanent failure
        client._ensure_loaded = lambda: False  # type: ignore[method-assign]

        frame = make_frame()
        det = make_detection("person", confidence=0.5)
        result = client.analyze_detections(frame, [det], ["person"])
        # Should return penalized detections
        assert len(result) == 1
        assert result[0].confidence < 0.5
        assert result[0].attributes.get("vlm_error") is True

    def test_analyze_empty_frame_returns_empty(self):
        from sopilot.perception.llava_client import LLaVAClient

        client = LLaVAClient()
        empty = np.array([])
        result = client.analyze_detections(empty, [make_detection("x")], ["x"])
        assert result == []

    def test_describe_scene_model_not_loaded(self):
        from sopilot.perception.llava_client import LLaVAClient

        client = LLaVAClient()
        client._ensure_loaded = lambda: False  # type: ignore[method-assign]
        frame = make_frame()
        result = client.describe_scene(frame)
        assert "description_ja" in result
        # Should return fallback
        assert result["description_ja"] != ""

    def test_answer_question_model_not_loaded(self):
        from sopilot.perception.llava_client import LLaVAClient

        client = LLaVAClient()
        client._ensure_loaded = lambda: False  # type: ignore[method-assign]
        frame = make_frame()
        answer = client.answer_question(frame, "test?")
        assert len(answer) > 0

    def test_close_on_unloaded_noop(self):
        from sopilot.perception.llava_client import LLaVAClient

        client = LLaVAClient()
        client.close()  # should not raise

    def test_describe_empty_frame_returns_error(self):
        from sopilot.perception.llava_client import LLaVAClient

        client = LLaVAClient()
        empty = np.array([])
        result = client.describe_scene(empty)
        assert "error" in result

    def test_answer_question_empty_question(self):
        from sopilot.perception.llava_client import LLaVAClient

        client = LLaVAClient()
        frame = make_frame()
        answer = client.answer_question(frame, "")
        assert answer == ""

    def test_orchestrator_registration(self):
        from sopilot.perception.llava_client import LLaVAClient

        mock_orch = MagicMock()
        client = LLaVAClient(orchestrator=mock_orch)
        mock_orch.register.assert_called_once()
        call_kwargs = mock_orch.register.call_args
        assert call_kwargs[0][0] == "llava"
