"""Tests for insurance_mvp.exceptions module."""

from insurance_mvp.exceptions import (
    ConfigurationError,
    DependencyMissingError,
    FrameExtractionError,
    InsuranceMVPError,
    PipelineStageError,
    VLMInferenceError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        for exc_cls in [
            PipelineStageError,
            VLMInferenceError,
            FrameExtractionError,
            ConfigurationError,
            DependencyMissingError,
        ]:
            assert issubclass(exc_cls, InsuranceMVPError)

    def test_catch_base_catches_all(self):
        exceptions = [
            PipelineStageError("mining", "timeout"),
            VLMInferenceError("failed"),
            FrameExtractionError("corrupt"),
            ConfigurationError("bad config"),
            DependencyMissingError("torch"),
        ]
        for exc in exceptions:
            try:
                raise exc
            except InsuranceMVPError:
                pass  # Should be caught


class TestPipelineStageError:
    def test_attributes(self):
        cause = RuntimeError("original")
        exc = PipelineStageError("vlm_inference", "GPU OOM", cause=cause)
        assert exc.stage == "vlm_inference"
        assert exc.detail == "GPU OOM"
        assert exc.cause is cause
        assert "vlm_inference" in str(exc)
        assert "GPU OOM" in str(exc)

    def test_no_cause(self):
        exc = PipelineStageError("mining", "no clips")
        assert exc.cause is None


class TestVLMInferenceError:
    def test_attributes(self):
        exc = VLMInferenceError("CUDA error", attempts=3)
        assert exc.attempts == 3
        assert "CUDA error" in str(exc)

    def test_with_cause(self):
        cause = RuntimeError("device reset")
        exc = VLMInferenceError("failed", attempts=2, cause=cause)
        assert exc.cause is cause


class TestDependencyMissingError:
    def test_message_with_feature(self):
        exc = DependencyMissingError("torch", feature="GPU inference")
        assert "torch" in str(exc)
        assert "GPU inference" in str(exc)
        assert "pip install torch" in str(exc)

    def test_message_without_feature(self):
        exc = DependencyMissingError("qwen-vl-utils")
        assert "qwen-vl-utils" in str(exc)
        assert exc.package == "qwen-vl-utils"
        assert exc.feature == ""
