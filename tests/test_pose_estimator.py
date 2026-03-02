"""Comprehensive tests for PoseEstimator (Task B: Pose × PPE Safety Detection).

Tests cover:
- PoseKeypoint, PPEStatus, PoseResult types
- PoseEstimator.__init__ (lazy load, default model, close)
- PoseEstimator.estimate() with mocked ultralytics
- PerceptionConfig pose fields
- PerceptionEngine integration with pose estimator
- Color analysis helpers (_infer_helmet, _infer_vest)

All ultralytics interactions are mocked — no model download required.

Run:  python -m pytest tests/test_pose_estimator.py -v
"""

from __future__ import annotations

import sys
import unittest

import numpy as np

# ── Import types under test ─────────────────────────────────────────────────
from sopilot.perception.types import (
    BBox,
    FrameResult,
    PerceptionConfig,
    PPEStatus,
    PoseKeypoint,
    PoseResult,
    SceneGraph,
    ViolationSeverity,
    WorldState,
)


# ---------------------------------------------------------------------------
# Helpers / Mock ultralytics structures
# ---------------------------------------------------------------------------


def _make_frame(h: int = 480, w: int = 640, color: tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """Create a synthetic BGR frame filled with a constant color."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _make_yellow_frame(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a frame filled with bright yellow (helmet-like) pixels."""
    # BGR: yellow = (0, 255, 255)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (0, 255, 255)
    return frame


def _make_orange_vest_frame(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a frame filled with bright orange-yellow (vest-like) pixels."""
    # BGR: hi-vis orange ~ (0, 200, 255)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (0, 200, 255)
    return frame


def _make_dark_frame(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a frame filled with dark pixels — no PPE colors."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_blue_frame(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a frame filled with blue pixels — not a vest color."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (200, 50, 50)  # BGR blue
    return frame


# ── Mock ultralytics objects ─────────────────────────────────────────────────


class MockBox:
    """Mimics a single ultralytics box for pose results."""

    def __init__(self, x1, y1, x2, y2, conf):
        self.xyxy = [np.array([x1, y1, x2, y2], dtype=np.float32)]
        self.conf = [conf]
        self.cls = [0]


class MockKeypoints:
    """Mimics ultralytics Keypoints object with .data attribute."""

    def __init__(self, kp_array: np.ndarray):
        # kp_array shape: (N, 17, 3)
        self.data = kp_array


class MockBoxes:
    """Mimics ultralytics Boxes container — supports indexing."""

    def __init__(self, boxes_list):
        self._boxes = boxes_list

    def __len__(self):
        return len(self._boxes)

    def __getitem__(self, idx):
        return self._boxes[idx]


class MockPoseResult:
    """Mimics a single ultralytics Results object for pose model."""

    def __init__(self, boxes_list=None, kp_array=None):
        if boxes_list is not None:
            self.boxes = MockBoxes(boxes_list)
        else:
            self.boxes = None

        if kp_array is not None:
            self.keypoints = MockKeypoints(kp_array)
        else:
            self.keypoints = None


class MockYOLOPoseModel:
    """Mimics ultralytics.YOLO for pose estimation."""

    def __init__(self, model_name="yolov8s-pose.pt"):
        self._model_name = model_name
        self._predict_results = None
        self.predict_calls = 0

    def predict(self, frame, **kwargs):
        self.predict_calls += 1
        if self._predict_results is not None:
            return self._predict_results
        # Default: one person in center
        kp_array = np.zeros((1, 17, 3), dtype=np.float32)
        # Set visible keypoints: nose at center
        h, w = frame.shape[:2]
        kp_array[0, 0] = [w / 2, h / 4, 0.9]   # nose
        kp_array[0, 5] = [w / 3, h / 2, 0.9]   # left_shoulder
        kp_array[0, 6] = [2 * w / 3, h / 2, 0.9]  # right_shoulder
        kp_array[0, 11] = [w / 3, 3 * h / 4, 0.9]  # left_hip
        kp_array[0, 12] = [2 * w / 3, 3 * h / 4, 0.9]  # right_hip
        box = MockBox(w * 0.2, h * 0.1, w * 0.8, h * 0.9, 0.85)
        r = MockPoseResult(boxes_list=[box], kp_array=kp_array)
        return [r]


def _build_pose_estimator_with_mock(model_name=None, **kwargs):
    """Build a PoseEstimator with mocked ultralytics.YOLO."""
    from sopilot.perception.pose import PoseEstimator

    mock_model = MockYOLOPoseModel(model_name or "yolov8s-pose.pt")
    est = PoseEstimator(model_name=model_name, **kwargs)

    # Inject mock directly
    def fake_load():
        est._model = mock_model
        est._loaded = True

    est._load_model = fake_load
    return est, mock_model


# ---------------------------------------------------------------------------
# 1. Type tests: PoseKeypoint, PPEStatus, PoseResult
# ---------------------------------------------------------------------------


class TestPoseKeypoint(unittest.TestCase):
    """Tests for the PoseKeypoint frozen dataclass."""

    def test_creation_stores_fields(self):
        kp = PoseKeypoint(x=0.5, y=0.3, confidence=0.9)
        self.assertAlmostEqual(kp.x, 0.5)
        self.assertAlmostEqual(kp.y, 0.3)
        self.assertAlmostEqual(kp.confidence, 0.9)

    def test_frozen_immutable(self):
        kp = PoseKeypoint(x=0.1, y=0.2, confidence=0.8)
        with self.assertRaises((AttributeError, TypeError)):
            kp.x = 0.9  # type: ignore[misc]

    def test_zero_confidence_keypoint(self):
        kp = PoseKeypoint(x=0.0, y=0.0, confidence=0.0)
        self.assertEqual(kp.confidence, 0.0)

    def test_boundary_normalized_coords(self):
        kp = PoseKeypoint(x=1.0, y=1.0, confidence=1.0)
        self.assertEqual(kp.x, 1.0)
        self.assertEqual(kp.y, 1.0)


class TestPPEStatus(unittest.TestCase):
    """Tests for the PPEStatus mutable dataclass."""

    def test_defaults(self):
        ppe = PPEStatus()
        self.assertFalse(ppe.has_helmet)
        self.assertEqual(ppe.helmet_confidence, 0.0)
        self.assertFalse(ppe.has_vest)
        self.assertEqual(ppe.vest_confidence, 0.0)

    def test_can_set_helmet(self):
        ppe = PPEStatus(has_helmet=True, helmet_confidence=0.9)
        self.assertTrue(ppe.has_helmet)
        self.assertAlmostEqual(ppe.helmet_confidence, 0.9)

    def test_can_set_vest(self):
        ppe = PPEStatus(has_vest=True, vest_confidence=0.75)
        self.assertTrue(ppe.has_vest)
        self.assertAlmostEqual(ppe.vest_confidence, 0.75)

    def test_mutable(self):
        ppe = PPEStatus()
        ppe.has_helmet = True
        self.assertTrue(ppe.has_helmet)


class TestPoseResult(unittest.TestCase):
    """Tests for the PoseResult dataclass."""

    def _make_result(self):
        bbox = BBox(0.1, 0.1, 0.5, 0.9)
        keypoints = [PoseKeypoint(x=0.3, y=0.2, confidence=0.9)] + [
            PoseKeypoint(x=0.0, y=0.0, confidence=0.0) for _ in range(16)
        ]
        ppe = PPEStatus()
        return PoseResult(
            person_bbox=bbox,
            keypoints=keypoints,
            ppe=ppe,
            pose_confidence=0.85,
        )

    def test_creation(self):
        r = self._make_result()
        self.assertIsInstance(r.person_bbox, BBox)
        self.assertEqual(len(r.keypoints), 17)
        self.assertIsInstance(r.ppe, PPEStatus)
        self.assertAlmostEqual(r.pose_confidence, 0.85)

    def test_keypoints_length(self):
        r = self._make_result()
        self.assertEqual(len(r.keypoints), 17)

    def test_bbox_stored(self):
        r = self._make_result()
        self.assertAlmostEqual(r.person_bbox.x1, 0.1)
        self.assertAlmostEqual(r.person_bbox.x2, 0.5)


# ---------------------------------------------------------------------------
# 2. PoseEstimator.__init__ and basic properties
# ---------------------------------------------------------------------------


class TestPoseEstimatorInit(unittest.TestCase):
    """Tests for PoseEstimator construction."""

    def test_default_model_name(self):
        from sopilot.perception.pose import PoseEstimator
        est = PoseEstimator()
        self.assertEqual(est._model_name, "yolov8s-pose.pt")

    def test_custom_model_name(self):
        from sopilot.perception.pose import PoseEstimator
        est = PoseEstimator(model_name="custom_pose.pt")
        self.assertEqual(est._model_name, "custom_pose.pt")

    def test_not_loaded_at_construction(self):
        from sopilot.perception.pose import PoseEstimator
        est = PoseEstimator()
        self.assertFalse(est._loaded)
        self.assertIsNone(est._model)

    def test_confidence_threshold_default(self):
        from sopilot.perception.pose import PoseEstimator
        est = PoseEstimator()
        self.assertAlmostEqual(est._confidence_threshold, 0.4)

    def test_close_resets_state(self):
        est, mock_model = _build_pose_estimator_with_mock()
        est._load_model()  # force load
        self.assertTrue(est._loaded)
        est.close()
        self.assertFalse(est._loaded)
        self.assertIsNone(est._model)

    def test_default_model_constant(self):
        from sopilot.perception.pose import PoseEstimator
        self.assertEqual(PoseEstimator.DEFAULT_MODEL, "yolov8s-pose.pt")

    def test_coco_keypoints_count(self):
        from sopilot.perception.pose import PoseEstimator
        self.assertEqual(len(PoseEstimator.COCO_KEYPOINTS), 17)

    def test_head_kp_indices(self):
        from sopilot.perception.pose import PoseEstimator
        self.assertEqual(PoseEstimator.HEAD_KP_INDICES, [0, 1, 2, 3, 4])

    def test_shoulder_hip_kp_indices(self):
        from sopilot.perception.pose import PoseEstimator
        self.assertIn(5, PoseEstimator.SHOULDER_KP_INDICES)
        self.assertIn(6, PoseEstimator.SHOULDER_KP_INDICES)
        self.assertIn(11, PoseEstimator.HIP_KP_INDICES)
        self.assertIn(12, PoseEstimator.HIP_KP_INDICES)


# ---------------------------------------------------------------------------
# 3. PoseEstimator.estimate() with mocked ultralytics
# ---------------------------------------------------------------------------


class TestPoseEstimatorEstimate(unittest.TestCase):
    """Tests for PoseEstimator.estimate() with fully mocked ultralytics."""

    def test_empty_frame_returns_empty(self):
        est, _ = _build_pose_estimator_with_mock()
        result = est.estimate(np.array([]))
        self.assertEqual(result, [])

    def test_zero_size_frame_returns_empty(self):
        est, _ = _build_pose_estimator_with_mock()
        result = est.estimate(np.zeros((0, 0, 3), dtype=np.uint8))
        self.assertEqual(result, [])

    def test_one_person_returns_one_result(self):
        est, mock_model = _build_pose_estimator_with_mock()
        est._load_model()
        frame = _make_frame(480, 640)
        results = est.estimate(frame)
        self.assertEqual(len(results), 1)

    def test_result_is_pose_result_instance(self):
        est, _ = _build_pose_estimator_with_mock()
        est._load_model()
        frame = _make_frame(480, 640)
        results = est.estimate(frame)
        self.assertIsInstance(results[0], PoseResult)

    def test_keypoints_are_17(self):
        est, _ = _build_pose_estimator_with_mock()
        est._load_model()
        frame = _make_frame(480, 640)
        results = est.estimate(frame)
        self.assertEqual(len(results[0].keypoints), 17)

    def test_keypoints_normalized_between_0_and_1(self):
        est, _ = _build_pose_estimator_with_mock()
        est._load_model()
        frame = _make_frame(480, 640)
        results = est.estimate(frame)
        for kp in results[0].keypoints:
            self.assertGreaterEqual(kp.x, 0.0)
            self.assertLessEqual(kp.x, 1.0)
            self.assertGreaterEqual(kp.y, 0.0)
            self.assertLessEqual(kp.y, 1.0)

    def test_person_bbox_normalized(self):
        est, _ = _build_pose_estimator_with_mock()
        est._load_model()
        frame = _make_frame(480, 640)
        results = est.estimate(frame)
        bbox = results[0].person_bbox
        self.assertGreaterEqual(bbox.x1, 0.0)
        self.assertLessEqual(bbox.x2, 1.0)
        self.assertGreater(bbox.x2, bbox.x1)

    def test_no_persons_returns_empty(self):
        est, mock_model = _build_pose_estimator_with_mock()
        est._load_model()
        # Set model to return empty results
        mock_model._predict_results = [MockPoseResult(boxes_list=[], kp_array=np.zeros((0, 17, 3)))]
        frame = _make_frame(480, 640)
        results = est.estimate(frame)
        self.assertEqual(results, [])

    def test_low_confidence_person_filtered(self):
        est, mock_model = _build_pose_estimator_with_mock(confidence_threshold=0.9)
        est._load_model()
        # Person with confidence 0.4 (below 0.9 threshold)
        kp_array = np.zeros((1, 17, 3), dtype=np.float32)
        kp_array[0, 0] = [320, 120, 0.9]
        box = MockBox(100, 50, 540, 430, 0.4)
        mock_model._predict_results = [MockPoseResult(boxes_list=[box], kp_array=kp_array)]
        frame = _make_frame(480, 640)
        results = est.estimate(frame)
        self.assertEqual(results, [])

    def test_model_predict_exception_returns_empty(self):
        est, mock_model = _build_pose_estimator_with_mock()
        est._load_model()
        def raise_exc(frame, **kwargs):
            raise RuntimeError("inference error")
        mock_model.predict = raise_exc
        frame = _make_frame(480, 640)
        results = est.estimate(frame)
        self.assertEqual(results, [])

    def test_ppe_status_populated(self):
        est, _ = _build_pose_estimator_with_mock()
        est._load_model()
        frame = _make_frame(480, 640)
        results = est.estimate(frame)
        self.assertIsInstance(results[0].ppe, PPEStatus)

    def test_pose_confidence_stored(self):
        est, _ = _build_pose_estimator_with_mock()
        est._load_model()
        frame = _make_frame(480, 640)
        results = est.estimate(frame)
        self.assertAlmostEqual(results[0].pose_confidence, 0.85, places=2)

    def test_ultralytics_import_error_propagated(self):
        from sopilot.perception.pose import PoseEstimator
        est = PoseEstimator()
        # Simulate ultralytics not installed
        original = sys.modules.get("ultralytics")
        sys.modules["ultralytics"] = None  # type: ignore[assignment]
        try:
            with self.assertRaises((ImportError, TypeError)):
                est._load_model()
        finally:
            if original is None:
                sys.modules.pop("ultralytics", None)
            else:
                sys.modules["ultralytics"] = original

    def test_none_keypoints_in_result_handled(self):
        est, mock_model = _build_pose_estimator_with_mock()
        est._load_model()
        # Result with None keypoints
        mock_model._predict_results = [MockPoseResult(boxes_list=None, kp_array=None)]
        frame = _make_frame(480, 640)
        results = est.estimate(frame)
        self.assertEqual(results, [])

    def test_multiple_persons_all_returned(self):
        est, mock_model = _build_pose_estimator_with_mock()
        est._load_model()
        h, w = 480, 640
        kp_array = np.zeros((2, 17, 3), dtype=np.float32)
        # Person 0 keypoints
        kp_array[0, 0] = [160, 100, 0.9]
        kp_array[0, 5] = [120, 200, 0.9]
        kp_array[0, 6] = [200, 200, 0.9]
        kp_array[0, 11] = [120, 350, 0.9]
        kp_array[0, 12] = [200, 350, 0.9]
        # Person 1 keypoints
        kp_array[1, 0] = [480, 100, 0.9]
        kp_array[1, 5] = [440, 200, 0.9]
        kp_array[1, 6] = [520, 200, 0.9]
        kp_array[1, 11] = [440, 350, 0.9]
        kp_array[1, 12] = [520, 350, 0.9]
        box0 = MockBox(80, 50, 240, 420, 0.9)
        box1 = MockBox(400, 50, 560, 420, 0.88)
        mock_model._predict_results = [MockPoseResult(boxes_list=[box0, box1], kp_array=kp_array)]
        frame = _make_frame(h, w)
        results = est.estimate(frame)
        self.assertEqual(len(results), 2)


# ---------------------------------------------------------------------------
# 4. PerceptionConfig pose fields
# ---------------------------------------------------------------------------


class TestPerceptionConfigPoseFields(unittest.TestCase):
    """Tests for new pose estimation fields in PerceptionConfig."""

    def test_pose_enabled_default_false(self):
        cfg = PerceptionConfig()
        self.assertFalse(cfg.pose_enabled)

    def test_pose_model_default(self):
        cfg = PerceptionConfig()
        self.assertEqual(cfg.pose_model, "yolov8s-pose.pt")

    def test_pose_confidence_threshold_default(self):
        cfg = PerceptionConfig()
        self.assertAlmostEqual(cfg.pose_confidence_threshold, 0.4)

    def test_pose_keypoint_confidence_default(self):
        cfg = PerceptionConfig()
        self.assertAlmostEqual(cfg.pose_keypoint_confidence, 0.3)

    def test_pose_enabled_can_be_set(self):
        cfg = PerceptionConfig(pose_enabled=True)
        self.assertTrue(cfg.pose_enabled)

    def test_pose_model_can_be_set(self):
        cfg = PerceptionConfig(pose_model="custom.pt")
        self.assertEqual(cfg.pose_model, "custom.pt")


# ---------------------------------------------------------------------------
# 5. PerceptionEngine integration tests
# ---------------------------------------------------------------------------


class MockPoseEstimator:
    """A controllable mock PoseEstimator for engine integration tests."""

    def __init__(self, results=None):
        self._results = results or []
        self.estimate_calls = 0
        self.close_called = False

    def estimate(self, frame):
        self.estimate_calls += 1
        return self._results

    def close(self):
        self.close_called = True


def _make_pose_result(
    has_helmet: bool = True,
    helmet_conf: float = 0.9,
    has_vest: bool = True,
    vest_conf: float = 0.9,
    bbox: BBox | None = None,
) -> PoseResult:
    """Helper to build a PoseResult with configurable PPE status."""
    bbox = bbox or BBox(0.1, 0.1, 0.5, 0.9)
    keypoints = [PoseKeypoint(x=0.3, y=0.2 + 0.04 * i, confidence=0.9) for i in range(17)]
    ppe = PPEStatus(
        has_helmet=has_helmet,
        helmet_confidence=helmet_conf,
        has_vest=has_vest,
        vest_confidence=vest_conf,
    )
    return PoseResult(
        person_bbox=bbox,
        keypoints=keypoints,
        ppe=ppe,
        pose_confidence=0.85,
    )


class TestPerceptionEngineWithPose(unittest.TestCase):
    """Integration tests for PerceptionEngine + PoseEstimator."""

    def _build_engine(self, pose_estimator=None):
        from sopilot.perception.engine import PerceptionEngine
        return PerceptionEngine(
            config=PerceptionConfig(),
            pose_estimator=pose_estimator,
        )

    def _make_frame(self):
        return _make_frame()

    def test_no_pose_estimator_gives_empty_pose_results(self):
        engine = self._build_engine(pose_estimator=None)
        frame = self._make_frame()
        result = engine.process_frame(frame, timestamp=0.0, frame_number=0, rules=[])
        self.assertEqual(result.pose_results, [])

    def test_with_mock_pose_estimator_populates_pose_results(self):
        pr = _make_pose_result(has_helmet=True, has_vest=True)
        mock_pe = MockPoseEstimator(results=[pr])
        engine = self._build_engine(pose_estimator=mock_pe)
        frame = self._make_frame()
        result = engine.process_frame(frame, timestamp=0.0, frame_number=0, rules=[])
        self.assertEqual(len(result.pose_results), 1)

    def test_missing_helmet_generates_warning_violation(self):
        # helmet_confidence=0.0 → absence_conf=1.0 > 0.5 → violation
        pr = _make_pose_result(has_helmet=False, helmet_conf=0.0, has_vest=True, vest_conf=0.9)
        mock_pe = MockPoseEstimator(results=[pr])
        engine = self._build_engine(pose_estimator=mock_pe)
        frame = self._make_frame()
        result = engine.process_frame(frame, timestamp=0.0, frame_number=0, rules=[])
        helmet_violations = [v for v in result.violations if "ヘルメット" in v.rule]
        self.assertGreater(len(helmet_violations), 0)
        self.assertEqual(helmet_violations[0].severity, ViolationSeverity.WARNING)
        self.assertEqual(helmet_violations[0].source, "pose")

    def test_has_helmet_no_helmet_violation(self):
        pr = _make_pose_result(has_helmet=True, helmet_conf=0.9, has_vest=True, vest_conf=0.9)
        mock_pe = MockPoseEstimator(results=[pr])
        engine = self._build_engine(pose_estimator=mock_pe)
        frame = self._make_frame()
        result = engine.process_frame(frame, timestamp=0.0, frame_number=0, rules=[])
        helmet_violations = [v for v in result.violations if "ヘルメット" in v.rule]
        self.assertEqual(len(helmet_violations), 0)

    def test_missing_vest_generates_warning_violation(self):
        pr = _make_pose_result(has_helmet=True, helmet_conf=0.9, has_vest=False, vest_conf=0.0)
        mock_pe = MockPoseEstimator(results=[pr])
        engine = self._build_engine(pose_estimator=mock_pe)
        frame = self._make_frame()
        result = engine.process_frame(frame, timestamp=0.0, frame_number=0, rules=[])
        vest_violations = [v for v in result.violations if "安全ベスト" in v.rule]
        self.assertGreater(len(vest_violations), 0)
        self.assertEqual(vest_violations[0].severity, ViolationSeverity.WARNING)

    def test_has_vest_no_vest_violation(self):
        pr = _make_pose_result(has_helmet=True, helmet_conf=0.9, has_vest=True, vest_conf=0.9)
        mock_pe = MockPoseEstimator(results=[pr])
        engine = self._build_engine(pose_estimator=mock_pe)
        frame = self._make_frame()
        result = engine.process_frame(frame, timestamp=0.0, frame_number=0, rules=[])
        vest_violations = [v for v in result.violations if "安全ベスト" in v.rule]
        self.assertEqual(len(vest_violations), 0)

    def test_pose_estimator_called_on_each_frame(self):
        mock_pe = MockPoseEstimator(results=[])
        engine = self._build_engine(pose_estimator=mock_pe)
        frame = self._make_frame()
        engine.process_frame(frame, timestamp=0.0, frame_number=0, rules=[])
        engine.process_frame(frame, timestamp=1.0, frame_number=1, rules=[])
        self.assertEqual(mock_pe.estimate_calls, 2)

    def test_close_releases_pose_estimator(self):
        mock_pe = MockPoseEstimator()
        engine = self._build_engine(pose_estimator=mock_pe)
        engine.close()
        self.assertTrue(mock_pe.close_called)

    def test_violation_rule_text_helmet(self):
        pr = _make_pose_result(has_helmet=False, helmet_conf=0.0, has_vest=True, vest_conf=0.9)
        mock_pe = MockPoseEstimator(results=[pr])
        engine = self._build_engine(pose_estimator=mock_pe)
        frame = self._make_frame()
        result = engine.process_frame(frame, timestamp=0.0, frame_number=0, rules=[])
        helmet_violations = [v for v in result.violations if "ヘルメット" in v.rule]
        self.assertEqual(helmet_violations[0].rule, "ヘルメット未着用を検出")

    def test_violation_rule_text_vest(self):
        pr = _make_pose_result(has_helmet=True, helmet_conf=0.9, has_vest=False, vest_conf=0.0)
        mock_pe = MockPoseEstimator(results=[pr])
        engine = self._build_engine(pose_estimator=mock_pe)
        frame = self._make_frame()
        result = engine.process_frame(frame, timestamp=0.0, frame_number=0, rules=[])
        vest_violations = [v for v in result.violations if "安全ベスト" in v.rule]
        self.assertEqual(vest_violations[0].rule, "安全ベスト未着用を検出")

    def test_low_confidence_absence_no_violation(self):
        # helmet_conf=0.6 → absence_conf=0.4, which is ≤ 0.5, so NO violation
        pr = _make_pose_result(has_helmet=False, helmet_conf=0.6, has_vest=True, vest_conf=0.9)
        mock_pe = MockPoseEstimator(results=[pr])
        engine = self._build_engine(pose_estimator=mock_pe)
        frame = self._make_frame()
        result = engine.process_frame(frame, timestamp=0.0, frame_number=0, rules=[])
        helmet_violations = [v for v in result.violations if "ヘルメット" in v.rule]
        self.assertEqual(len(helmet_violations), 0)


class TestBuildPerceptionEngineWithPose(unittest.TestCase):
    """Tests for build_perception_engine() factory with pose config."""

    def test_pose_disabled_by_default_no_estimator(self):
        from sopilot.perception.engine import build_perception_engine
        cfg = PerceptionConfig(detector_backend="mock")
        engine = build_perception_engine(config=cfg)
        self.assertIsNone(engine._pose_estimator)

    def test_pose_enabled_true_creates_estimator(self):
        from sopilot.perception.engine import build_perception_engine
        from sopilot.perception.pose import PoseEstimator

        # PoseEstimator.__init__ is lightweight (lazy load), no ultralytics import at construction
        cfg = PerceptionConfig(detector_backend="mock", pose_enabled=True)
        engine = build_perception_engine(config=cfg)
        self.assertIsNotNone(engine._pose_estimator)
        self.assertIsInstance(engine._pose_estimator, PoseEstimator)

    def test_pose_disabled_explicit_no_estimator(self):
        from sopilot.perception.engine import build_perception_engine
        cfg = PerceptionConfig(detector_backend="mock", pose_enabled=False)
        engine = build_perception_engine(config=cfg)
        self.assertIsNone(engine._pose_estimator)


# ---------------------------------------------------------------------------
# 6. Color analysis helper tests
# ---------------------------------------------------------------------------


class TestColorAnalysisHelpers(unittest.TestCase):
    """Tests for _infer_helmet and _infer_vest color analysis."""

    def setUp(self):
        self.est, _ = _build_pose_estimator_with_mock()
        self.est._load_model()

    def _full_frame_bbox(self) -> BBox:
        """BBox covering entire frame."""
        return BBox(0.0, 0.0, 1.0, 1.0)

    def test_infer_helmet_yellow_region_true(self):
        frame = _make_yellow_frame(100, 100)
        has_helmet, conf = self.est._infer_helmet(frame, self._full_frame_bbox())
        self.assertTrue(has_helmet, "Yellow frame should infer helmet present")
        self.assertGreater(conf, 0.5)

    def test_infer_helmet_dark_region_false(self):
        frame = _make_dark_frame(100, 100)
        has_helmet, conf = self.est._infer_helmet(frame, self._full_frame_bbox())
        self.assertFalse(has_helmet, "Dark frame should not infer helmet")

    def test_infer_vest_orange_region_true(self):
        frame = _make_orange_vest_frame(100, 100)
        has_vest, conf = self.est._infer_vest(frame, self._full_frame_bbox())
        self.assertTrue(has_vest, "Orange-yellow frame should infer vest present")
        self.assertGreater(conf, 0.5)

    def test_infer_vest_blue_region_false(self):
        frame = _make_blue_frame(100, 100)
        has_vest, conf = self.est._infer_vest(frame, self._full_frame_bbox())
        self.assertFalse(has_vest, "Blue frame should not infer vest")

    def test_infer_helmet_none_bbox_returns_false(self):
        frame = _make_frame(100, 100)
        has_helmet, conf = self.est._infer_helmet(frame, None)
        self.assertFalse(has_helmet)
        self.assertEqual(conf, 0.0)

    def test_infer_vest_none_bbox_returns_false(self):
        frame = _make_frame(100, 100)
        has_vest, conf = self.est._infer_vest(frame, None)
        self.assertFalse(has_vest)
        self.assertEqual(conf, 0.0)

    def test_infer_helmet_degenerate_tiny_bbox_returns_false(self):
        frame = _make_frame(480, 640)
        # Degenerate bbox: 1px wide
        tiny_bbox = BBox(0.5, 0.5, 0.501, 0.501)
        has_helmet, conf = self.est._infer_helmet(frame, tiny_bbox)
        # Should return False gracefully (region too small)
        self.assertFalse(has_helmet)

    def test_infer_vest_degenerate_tiny_bbox_returns_false(self):
        frame = _make_frame(480, 640)
        tiny_bbox = BBox(0.5, 0.5, 0.501, 0.501)
        has_vest, conf = self.est._infer_vest(frame, tiny_bbox)
        self.assertFalse(has_vest)

    def test_white_region_infers_helmet(self):
        # White pixels: low saturation, high value → should trigger helmet
        frame = np.full((100, 100, 3), 240, dtype=np.uint8)  # near-white BGR
        has_helmet, conf = self.est._infer_helmet(frame, self._full_frame_bbox())
        self.assertTrue(has_helmet, "White region should infer helmet (white hard hat)")

    def test_confidence_clamped_to_1(self):
        # A fully yellow frame should give confidence of exactly 1.0 (clamped)
        frame = _make_yellow_frame(100, 100)
        has_helmet, conf = self.est._infer_helmet(frame, self._full_frame_bbox())
        self.assertLessEqual(conf, 1.0)
        self.assertGreaterEqual(conf, 0.0)


# ---------------------------------------------------------------------------
# 7. Additional edge-case tests
# ---------------------------------------------------------------------------


class TestPoseEstimatorEdgeCases(unittest.TestCase):
    """Additional edge-case tests for PoseEstimator."""

    def test_frame_result_has_pose_results_field(self):
        """FrameResult must have a pose_results field defaulting to []."""
        sg = SceneGraph(timestamp=0.0, frame_number=0, entities=[], relations=[])
        ws = WorldState(
            timestamp=0.0, frame_number=0, scene_graph=sg,
            active_tracks={}, events=[], zone_occupancy={},
        )
        fr = FrameResult(
            timestamp=0.0, frame_number=0, world_state=ws,
            violations=[], processing_time_ms=0.0,
        )
        self.assertEqual(fr.pose_results, [])

    def test_frame_result_pose_results_can_be_set(self):
        sg = SceneGraph(timestamp=0.0, frame_number=0, entities=[], relations=[])
        ws = WorldState(
            timestamp=0.0, frame_number=0, scene_graph=sg,
            active_tracks={}, events=[], zone_occupancy={},
        )
        pr = _make_pose_result()
        fr = FrameResult(
            timestamp=0.0, frame_number=0, world_state=ws,
            violations=[], processing_time_ms=0.0, pose_results=[pr],
        )
        self.assertEqual(len(fr.pose_results), 1)

    def test_bgr_to_hsv_pure_numpy_no_cv2(self):
        """_bgr_to_hsv should not require cv2."""
        est, _ = _build_pose_estimator_with_mock()
        frame = _make_yellow_frame(10, 10)
        hsv = est._bgr_to_hsv(frame)
        self.assertEqual(hsv.shape, (10, 10, 3))
        # Yellow in BGR (0,255,255): H should be in ~30 range [OpenCV ~15–45]
        h = hsv[0, 0, 0]
        self.assertGreater(h, 10)
        self.assertLess(h, 60)

    def test_extract_region_valid_bbox(self):
        est, _ = _build_pose_estimator_with_mock()
        frame = _make_frame(480, 640)
        bbox = BBox(0.1, 0.1, 0.5, 0.5)
        region = est._extract_region(frame, bbox)
        self.assertIsNotNone(region)
        self.assertGreater(region.shape[0], 0)

    def test_extract_region_none_bbox_returns_none(self):
        est, _ = _build_pose_estimator_with_mock()
        frame = _make_frame(480, 640)
        result = est._extract_region(frame, None)
        self.assertIsNone(result)

    def test_pose_ppe_violations_both_missing(self):
        """Engine should generate two violations when both helmet and vest absent."""
        from sopilot.perception.engine import PerceptionEngine
        pr = _make_pose_result(
            has_helmet=False, helmet_conf=0.0,
            has_vest=False, vest_conf=0.0,
        )
        mock_pe = MockPoseEstimator(results=[pr])
        engine = PerceptionEngine(config=PerceptionConfig(), pose_estimator=mock_pe)
        frame = _make_frame()
        result = engine.process_frame(frame, 0.0, 0, [])
        pose_violations = [v for v in result.violations if v.source == "pose"]
        self.assertEqual(len(pose_violations), 2)

    def test_violation_evidence_contains_source_pose(self):
        from sopilot.perception.engine import PerceptionEngine
        pr = _make_pose_result(has_helmet=False, helmet_conf=0.0, has_vest=True, vest_conf=0.9)
        mock_pe = MockPoseEstimator(results=[pr])
        engine = PerceptionEngine(config=PerceptionConfig(), pose_estimator=mock_pe)
        frame = _make_frame()
        result = engine.process_frame(frame, 0.0, 0, [])
        helmet_v = next(v for v in result.violations if "ヘルメット" in v.rule)
        self.assertEqual(helmet_v.evidence.get("source"), "pose")

    def test_compute_head_bbox_from_keypoints(self):
        est, _ = _build_pose_estimator_with_mock()
        keypoints = [PoseKeypoint(x=0.0, y=0.0, confidence=0.0)] * 17
        # Set nose visible at (0.5, 0.2)
        keypoints[0] = PoseKeypoint(x=0.5, y=0.2, confidence=0.9)
        bbox = est._compute_head_bbox(keypoints, 640, 480)
        self.assertIsNotNone(bbox)
        # Head bbox should encompass the nose keypoint area
        self.assertLessEqual(bbox.x1, 0.5)
        self.assertGreaterEqual(bbox.x2, 0.5)

    def test_compute_torso_bbox_requires_two_keypoints(self):
        est, _ = _build_pose_estimator_with_mock()
        # Only one torso keypoint visible
        keypoints = [PoseKeypoint(x=0.0, y=0.0, confidence=0.0)] * 17
        keypoints[5] = PoseKeypoint(x=0.4, y=0.5, confidence=0.9)
        bbox = est._compute_torso_bbox(keypoints, 640, 480)
        self.assertIsNone(bbox)

    def test_compute_torso_bbox_with_enough_keypoints(self):
        est, _ = _build_pose_estimator_with_mock()
        keypoints = [PoseKeypoint(x=0.0, y=0.0, confidence=0.0)] * 17
        keypoints[5] = PoseKeypoint(x=0.35, y=0.5, confidence=0.9)   # left shoulder
        keypoints[6] = PoseKeypoint(x=0.65, y=0.5, confidence=0.9)   # right shoulder
        keypoints[11] = PoseKeypoint(x=0.35, y=0.75, confidence=0.9)  # left hip
        keypoints[12] = PoseKeypoint(x=0.65, y=0.75, confidence=0.9)  # right hip
        bbox = est._compute_torso_bbox(keypoints, 640, 480)
        self.assertIsNotNone(bbox)
        self.assertAlmostEqual(bbox.x1, max(0.0, 0.35 - 0.02), places=2)
        self.assertAlmostEqual(bbox.x2, min(1.0, 0.65 + 0.02), places=2)


if __name__ == "__main__":
    unittest.main()
