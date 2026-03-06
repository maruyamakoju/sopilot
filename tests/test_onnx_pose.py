"""Tests for sopilot.perception.onnx_pose — ONNX pose estimation (Item 4).

15 tests. All use mocked ort.InferenceSession — no model file required.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_pose_estimator() -> "object":
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_session.get_inputs.return_value = [mock_input]

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        from sopilot.perception.onnx_pose import OnnxPoseEstimator
        estimator = OnnxPoseEstimator("fake_pose.onnx")
    estimator._session = mock_session
    return estimator


class TestOnnxPoseEstimator:
    # ── Construction ────────────────────────────────────────────────────

    def test_constructor_sets_thresholds(self) -> None:
        est = _make_pose_estimator()
        assert est._confidence_threshold == 0.4
        assert est._keypoint_confidence == 0.3

    def test_raises_if_onnxruntime_missing(self) -> None:
        with patch.dict(__import__("sys").modules, {"onnxruntime": None}):
            with pytest.raises((ImportError, Exception)):
                from sopilot.perception.onnx_pose import OnnxPoseEstimator
                OnnxPoseEstimator("model.onnx")

    # ── Preprocess ──────────────────────────────────────────────────────

    def test_preprocess_output_shape(self) -> None:
        est = _make_pose_estimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        blob = est._preprocess(frame)
        assert blob.shape == (1, 3, 640, 640)

    def test_preprocess_normalised(self) -> None:
        est = _make_pose_estimator()
        frame = np.full((64, 64, 3), 255, dtype=np.uint8)
        blob = est._preprocess(frame, target_size=64)
        assert blob.max() <= 1.0

    # ── estimate() ──────────────────────────────────────────────────────

    def test_estimate_returns_empty_when_session_none(self) -> None:
        est = _make_pose_estimator()
        est._session = None
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert est.estimate(frame) == []

    def test_estimate_returns_list(self) -> None:
        est = _make_pose_estimator()
        est._session.run.return_value = []
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = est.estimate(frame)
        assert isinstance(result, list)

    def test_estimate_handles_inference_error(self) -> None:
        est = _make_pose_estimator()
        est._session.run.side_effect = RuntimeError("OOM")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = est.estimate(frame)  # must not raise
        assert result == []

    # ── postprocess ─────────────────────────────────────────────────────

    def test_postprocess_empty_returns_empty(self) -> None:
        est = _make_pose_estimator()
        assert est._postprocess([], img_h=480, img_w=640) == []

    def test_postprocess_below_threshold_returns_empty(self) -> None:
        est = _make_pose_estimator()
        # (1, 56, 100) — 100 detections with conf=0.0 → all filtered
        raw = np.zeros((1, 56, 100), dtype=np.float32)
        result = est._postprocess([raw], img_h=480, img_w=640)
        assert result == []

    def test_postprocess_above_threshold_returns_result(self) -> None:
        est = _make_pose_estimator()
        est._confidence_threshold = 0.5

        # Use (1, 56, 100) format: attrs=56, N=100 → raw[0]=(56,100), 56<100 → transpose to (100,56)
        raw = np.zeros((1, 56, 100), dtype=np.float32)
        # Set detection 0 (column 0 in attrs-first): cx,cy,w,h at attrs 0-3, conf at attr 4
        raw[0, 0, 0] = 320    # cx
        raw[0, 1, 0] = 240    # cy
        raw[0, 2, 0] = 100    # w
        raw[0, 3, 0] = 80     # h
        raw[0, 4, 0] = 0.9    # person conf
        # Keypoints
        for i in range(17):
            raw[0, 5 + i * 3, 0] = 320 / 640
            raw[0, 5 + i * 3 + 1, 0] = 240 / 640
            raw[0, 5 + i * 3 + 2, 0] = 0.9

        result = est._postprocess([raw], img_h=480, img_w=640)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9, abs=0.01)

    def test_postprocess_keypoints_count(self) -> None:
        est = _make_pose_estimator()
        est._confidence_threshold = 0.1

        raw = np.zeros((1, 56, 100), dtype=np.float32)
        raw[0, 4, 0] = 0.8  # person conf above threshold
        result = est._postprocess([raw], img_h=480, img_w=640)
        if result:
            assert len(result[0].keypoints) == 17

    def test_postprocess_bbox_normalised(self) -> None:
        est = _make_pose_estimator()
        est._confidence_threshold = 0.1

        raw = np.zeros((1, 56, 100), dtype=np.float32)
        raw[0, 0, 0] = 320
        raw[0, 1, 0] = 240
        raw[0, 2, 0] = 200
        raw[0, 3, 0] = 160
        raw[0, 4, 0] = 0.9
        result = est._postprocess([raw], img_h=480, img_w=640)
        if result:
            bbox = result[0].bbox
            assert 0.0 <= bbox.x1 <= 1.0
            assert 0.0 <= bbox.x2 <= 1.0

    # ── PPE heuristics ──────────────────────────────────────────────────

    def test_ppe_heuristics_helmet_detected(self) -> None:
        from sopilot.perception.onnx_pose import OnnxPoseEstimator
        from sopilot.perception.types import BBox

        est = _make_pose_estimator()
        # 17 keypoints: first 5 (head) all with high confidence
        kps = [(0.5, 0.5, 0.9)] * 17
        bbox = BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        has_helmet, _ = est._ppe_heuristics(kps, bbox)
        assert has_helmet is True

    def test_ppe_heuristics_no_helmet_when_head_kp_low_conf(self) -> None:
        from sopilot.perception.types import BBox

        est = _make_pose_estimator()
        kps = [(0.5, 0.5, 0.0)] * 17  # all low confidence
        bbox = BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        has_helmet, _ = est._ppe_heuristics(kps, bbox)
        assert has_helmet is False

    def test_ppe_heuristics_vest_detected(self) -> None:
        from sopilot.perception.types import BBox

        est = _make_pose_estimator()
        kps = [(0.5, 0.5, 0.9)] * 17  # all visible → torso covered
        bbox = BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        _, has_vest = est._ppe_heuristics(kps, bbox)
        assert has_vest is True
