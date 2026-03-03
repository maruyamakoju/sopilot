"""Tests for sopilot/perception/depth.py — MonocularDepthEstimator.

All BBox/Detection objects are mocked locally; no imports from sopilot.perception.types.
"""
from __future__ import annotations
import math
import threading
from dataclasses import dataclass, field
from typing import Optional

import pytest

from sopilot.perception.depth import DepthEstimate, MonocularDepthEstimator


# ---------------------------------------------------------------------------
# Mock objects (simulate BBox / Detection without importing types)
# ---------------------------------------------------------------------------

@dataclass
class MockBBox:
    x: float
    y: float
    w: float
    h: float

    def area(self) -> float:
        return self.w * self.h

    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


@dataclass
class MockDetection:
    entity_id: int
    label: str
    bbox: object  # MockBBox or list
    confidence: float = 0.9


def make_det(
    entity_id: int = 1,
    label: str = "person",
    x: float = 0.1,
    y: float = 0.5,
    w: float = 0.2,
    h: float = 0.4,
    confidence: float = 0.9,
    bbox_as_list: bool = False,
) -> MockDetection:
    if bbox_as_list:
        bbox = [x, y, w, h]
    else:
        bbox = MockBBox(x=x, y=y, w=w, h=h)
    return MockDetection(entity_id=entity_id, label=label, bbox=bbox, confidence=confidence)


# ---------------------------------------------------------------------------
# TestDepthEstimate
# ---------------------------------------------------------------------------

class TestDepthEstimate:
    def test_fields_populated(self):
        de = DepthEstimate(
            entity_id=1,
            label="person",
            bbox=[0.1, 0.2, 0.3, 0.4],
            depth_relative=0.5,
            depth_metric_m=3.0,
            confidence=0.8,
        )
        assert de.entity_id == 1
        assert de.label == "person"
        assert de.bbox == [0.1, 0.2, 0.3, 0.4]
        assert de.depth_relative == 0.5
        assert de.depth_metric_m == 3.0
        assert de.confidence == 0.8

    def test_to_dict_keys(self):
        de = DepthEstimate(1, "car", [0.0, 0.0, 0.5, 0.5], 0.3, 5.0, 0.9)
        d = de.to_dict()
        assert set(d.keys()) == {"entity_id", "label", "bbox", "depth_relative", "depth_metric_m", "confidence"}

    def test_to_dict_values(self):
        de = DepthEstimate(2, "truck", [0.1, 0.1, 0.4, 0.4], 0.75, 10.123, 0.65)
        d = de.to_dict()
        assert d["entity_id"] == 2
        assert d["label"] == "truck"
        assert d["bbox"] == [0.1, 0.1, 0.4, 0.4]
        assert d["depth_relative"] == round(0.75, 4)
        assert d["depth_metric_m"] == round(10.123, 2)
        assert d["confidence"] == round(0.65, 4)

    def test_to_dict_depth_metric_none(self):
        de = DepthEstimate(3, "unknown", [0.0, 0.0, 0.1, 0.1], 0.9, None, 0.3)
        d = de.to_dict()
        assert d["depth_metric_m"] is None

    def test_depth_relative_in_range(self):
        de = DepthEstimate(1, "x", [0, 0, 0.1, 0.1], 0.42, None, 0.5)
        assert 0.0 <= de.depth_relative <= 1.0

    def test_confidence_in_range(self):
        de = DepthEstimate(1, "x", [0, 0, 0.1, 0.1], 0.5, None, 0.77)
        assert 0.0 <= de.confidence <= 1.0

    def test_to_dict_rounding(self):
        de = DepthEstimate(1, "x", [0.0, 0.0, 0.2, 0.2], 0.123456789, 1.23456, 0.987654)
        d = de.to_dict()
        assert d["depth_relative"] == round(0.123456789, 4)
        assert d["depth_metric_m"] == round(1.23456, 2)
        assert d["confidence"] == round(0.987654, 4)


# ---------------------------------------------------------------------------
# TestMonocularDepthEstimatorInit
# ---------------------------------------------------------------------------

class TestMonocularDepthEstimatorInit:
    def test_default_params(self):
        est = MonocularDepthEstimator()
        cfg = est.get_config()
        assert cfg["camera_height_m"] == 2.5
        assert cfg["tilt_angle_deg"] == pytest.approx(15.0)
        assert cfg["focal_length_px"] is None
        assert cfg["frame_width_px"] == 1280
        assert cfg["frame_height_px"] == 720
        assert cfg["area_weight"] == pytest.approx(0.6)
        assert cfg["y_weight"] == pytest.approx(0.4)

    def test_custom_params(self):
        est = MonocularDepthEstimator(
            camera_height_m=3.0,
            tilt_angle_deg=30.0,
            focal_length_px=800.0,
            frame_width_px=1920,
            frame_height_px=1080,
            area_weight=0.7,
            y_weight=0.3,
        )
        cfg = est.get_config()
        assert cfg["camera_height_m"] == 3.0
        assert cfg["tilt_angle_deg"] == pytest.approx(30.0)
        assert cfg["focal_length_px"] == 800.0
        assert cfg["frame_width_px"] == 1920
        assert cfg["frame_height_px"] == 1080
        # area_weight is clamped and y_weight is derived
        assert cfg["area_weight"] == pytest.approx(0.7)
        assert cfg["y_weight"] == pytest.approx(0.3)

    def test_area_weight_clamped_above_1(self):
        est = MonocularDepthEstimator(area_weight=1.5)
        cfg = est.get_config()
        assert cfg["area_weight"] == pytest.approx(1.0)
        assert cfg["y_weight"] == pytest.approx(0.0)

    def test_area_weight_clamped_below_0(self):
        est = MonocularDepthEstimator(area_weight=-0.5)
        cfg = est.get_config()
        assert cfg["area_weight"] == pytest.approx(0.0)
        assert cfg["y_weight"] == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        for w in [0.0, 0.3, 0.5, 0.8, 1.0]:
            est = MonocularDepthEstimator(area_weight=w)
            cfg = est.get_config()
            assert cfg["area_weight"] + cfg["y_weight"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestDepthFromArea
# ---------------------------------------------------------------------------

class TestDepthFromArea:
    def setup_method(self):
        self.est = MonocularDepthEstimator(area_weight=1.0)  # pure area cue

    def _depth_area(self, w: float, h: float) -> float:
        # invoke via estimate with fixed y so y-cue is neutral (area_weight=1.0)
        det = make_det(x=0.0, y=0.0, w=w, h=h)
        results = self.est.estimate([det])
        return results[0].depth_relative

    def test_large_bbox_gives_low_depth(self):
        # Large object occupies much of the frame → closer → low depth
        depth = self._depth_area(0.5, 0.6)
        assert depth < 0.4

    def test_small_bbox_gives_high_depth(self):
        # Tiny object → far → high depth
        depth = self._depth_area(0.02, 0.02)
        assert depth > 0.7

    def test_unit_area_clamps_to_zero(self):
        # w=1.0, h=1.0 → area=1.0 → depth_from_area = 1 - sqrt(1)*2.5 = -1.5 → clamped to 0
        depth = self._depth_area(1.0, 1.0)
        assert depth == pytest.approx(0.0)

    def test_tiny_area_approaches_one(self):
        # Very small bbox → depth close to 1
        depth = self._depth_area(0.001, 0.001)
        assert depth > 0.9

    def test_medium_bbox_midrange_depth(self):
        # Medium area → intermediate depth
        depth = self._depth_area(0.15, 0.2)
        assert 0.0 <= depth <= 1.0

    def test_monotone_decreasing_with_area(self):
        # As w grows (h fixed), depth should be non-increasing
        depths = [self._depth_area(w, 0.2) for w in [0.05, 0.1, 0.2, 0.4]]
        for i in range(len(depths) - 1):
            assert depths[i] >= depths[i + 1]


# ---------------------------------------------------------------------------
# TestDepthFromY
# ---------------------------------------------------------------------------

class TestDepthFromY:
    def setup_method(self):
        self.est = MonocularDepthEstimator(area_weight=0.0)  # pure y-cue

    def _depth_y(self, y: float, h: float = 0.1) -> float:
        det = make_det(x=0.0, y=y, w=0.1, h=h)
        results = self.est.estimate([det])
        return results[0].depth_relative

    def test_bottom_of_frame_low_depth(self):
        # y near 1 → center near 1 → depth low
        depth = self._depth_y(y=0.85, h=0.1)
        assert depth < 0.2

    def test_top_of_frame_high_depth(self):
        # y near 0 → center near 0 → depth high
        depth = self._depth_y(y=0.0, h=0.05)
        assert depth > 0.9

    def test_midframe_midrange_depth(self):
        depth = self._depth_y(y=0.45, h=0.1)
        assert 0.0 <= depth <= 1.0

    def test_monotone_decreasing_with_y(self):
        # Higher y value (lower in frame) → lower depth
        depths = [self._depth_y(y=y, h=0.05) for y in [0.0, 0.2, 0.5, 0.8, 0.9]]
        for i in range(len(depths) - 1):
            assert depths[i] >= depths[i + 1]

    def test_clamped_at_zero(self):
        # y=1.0, h=0.0 → center=1.0 → 1-1.0=0 → 0
        depth = self._depth_y(y=1.0, h=0.0)
        assert depth == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestDepthRelativeOrdering
# ---------------------------------------------------------------------------

class TestDepthRelativeOrdering:
    def test_near_person_smaller_depth_than_far_person(self):
        est = MonocularDepthEstimator()
        # Near: large bbox at bottom of frame
        near = make_det(x=0.0, y=0.7, w=0.4, h=0.5, label="person")
        # Far: small bbox at top of frame
        far = make_det(entity_id=2, x=0.4, y=0.05, w=0.05, h=0.1, label="person")
        results = est.estimate([near, far])
        assert results[0].depth_relative < results[1].depth_relative

    def test_same_position_larger_is_closer(self):
        est = MonocularDepthEstimator(area_weight=1.0)
        big = make_det(x=0.0, y=0.5, w=0.4, h=0.4)
        small = make_det(entity_id=2, x=0.0, y=0.5, w=0.1, h=0.1)
        results = est.estimate([big, small])
        assert results[0].depth_relative < results[1].depth_relative

    def test_same_size_lower_is_closer(self):
        est = MonocularDepthEstimator(area_weight=0.0)
        lower = make_det(x=0.3, y=0.8, w=0.1, h=0.1)
        upper = make_det(entity_id=2, x=0.3, y=0.1, w=0.1, h=0.1)
        results = est.estimate([lower, upper])
        assert results[0].depth_relative < results[1].depth_relative


# ---------------------------------------------------------------------------
# TestMetricDepth
# ---------------------------------------------------------------------------

class TestMetricDepth:
    def test_no_focal_returns_none(self):
        est = MonocularDepthEstimator(focal_length_px=None)
        det = make_det(label="person", h=0.4)
        results = est.estimate([det])
        assert results[0].depth_metric_m is None

    def test_known_label_person_metric_value(self):
        # person height = 1.7m, focal = 700px, frame_h = 720px
        # bbox_h_px = 0.4 * 720 = 288px → depth = 1.7*700/288 ≈ 4.13m
        est = MonocularDepthEstimator(focal_length_px=700.0, frame_height_px=720)
        det = make_det(label="person", h=0.4)
        results = est.estimate([det])
        expected = round((1.7 * 700.0) / (0.4 * 720), 2)
        assert results[0].depth_metric_m == pytest.approx(expected, abs=0.05)

    def test_known_label_car_metric_value(self):
        est = MonocularDepthEstimator(focal_length_px=500.0, frame_height_px=720)
        det = make_det(label="car", h=0.3)
        results = est.estimate([det])
        expected = round((1.5 * 500.0) / (0.3 * 720), 2)
        assert results[0].depth_metric_m == pytest.approx(expected, abs=0.05)

    def test_unknown_label_returns_none(self):
        est = MonocularDepthEstimator(focal_length_px=700.0)
        det = make_det(label="forklift", h=0.3)
        results = est.estimate([det])
        assert results[0].depth_metric_m is None

    def test_bbox_h_zero_returns_none(self):
        est = MonocularDepthEstimator(focal_length_px=700.0, frame_height_px=720)
        det = make_det(label="person", h=0.0)
        results = est.estimate([det])
        assert results[0].depth_metric_m is None

    def test_worker_label_recognized(self):
        est = MonocularDepthEstimator(focal_length_px=600.0, frame_height_px=720)
        det = make_det(label="worker", h=0.35)
        results = est.estimate([det])
        assert results[0].depth_metric_m is not None
        assert results[0].depth_metric_m > 0.0

    def test_label_case_insensitive(self):
        est = MonocularDepthEstimator(focal_length_px=700.0, frame_height_px=720)
        det_lower = make_det(label="person", h=0.4)
        det_upper = make_det(entity_id=2, label="PERSON", h=0.4, bbox_as_list=True)
        r1 = est.estimate([det_lower])
        r2 = est.estimate([det_upper])
        # Both should return the same metric depth (case-insensitive lookup)
        assert r1[0].depth_metric_m == r2[0].depth_metric_m

    def test_metric_depth_minimum_clamped(self):
        # Very large h → tiny computed depth → should be clamped to 0.1
        est = MonocularDepthEstimator(focal_length_px=10.0, frame_height_px=720)
        det = make_det(label="person", h=0.99)
        results = est.estimate([det])
        assert results[0].depth_metric_m >= 0.1


# ---------------------------------------------------------------------------
# TestEstimateEmpty
# ---------------------------------------------------------------------------

class TestEstimateEmpty:
    def test_empty_input_returns_empty_list(self):
        est = MonocularDepthEstimator()
        results = est.estimate([])
        assert results == []
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TestEstimateSingle
# ---------------------------------------------------------------------------

class TestEstimateSingle:
    def test_all_fields_populated(self):
        est = MonocularDepthEstimator(focal_length_px=700.0, frame_height_px=720)
        det = make_det(entity_id=42, label="person", x=0.1, y=0.4, w=0.2, h=0.4)
        results = est.estimate([det])
        assert len(results) == 1
        r = results[0]
        assert r.entity_id == 42
        assert r.label == "person"
        assert len(r.bbox) == 4
        assert 0.0 <= r.depth_relative <= 1.0
        assert r.depth_metric_m is not None
        assert r.depth_metric_m > 0
        assert 0.0 <= r.confidence <= 1.0

    def test_returns_depth_estimate_type(self):
        est = MonocularDepthEstimator()
        det = make_det()
        results = est.estimate([det])
        assert isinstance(results[0], DepthEstimate)

    def test_bbox_preserved_correctly(self):
        est = MonocularDepthEstimator()
        det = make_det(x=0.15, y=0.25, w=0.30, h=0.45)
        results = est.estimate([det])
        assert results[0].bbox == pytest.approx([0.15, 0.25, 0.30, 0.45])


# ---------------------------------------------------------------------------
# TestEstimateMultiple
# ---------------------------------------------------------------------------

class TestEstimateMultiple:
    def test_five_detections_all_get_estimates(self):
        est = MonocularDepthEstimator()
        dets = [
            make_det(entity_id=i, label="person", x=0.1 * i, y=0.2 * i, w=0.1 + 0.05 * i, h=0.15 + 0.05 * i)
            for i in range(1, 6)
        ]
        results = est.estimate(dets)
        assert len(results) == 5

    def test_entity_ids_preserved(self):
        est = MonocularDepthEstimator()
        dets = [make_det(entity_id=i) for i in [10, 20, 30]]
        results = est.estimate(dets)
        ids = [r.entity_id for r in results]
        assert ids == [10, 20, 30]

    def test_all_depths_in_range(self):
        est = MonocularDepthEstimator()
        dets = [
            make_det(entity_id=i, x=0.1, y=0.1 * i, w=0.1 + 0.05 * i, h=0.1 + 0.05 * i)
            for i in range(1, 6)
        ]
        results = est.estimate(dets)
        for r in results:
            assert 0.0 <= r.depth_relative <= 1.0
            assert 0.0 <= r.confidence <= 1.0

    def test_order_preserved(self):
        est = MonocularDepthEstimator()
        labels = ["person", "car", "truck", "bicycle", "motorcycle"]
        dets = [make_det(entity_id=i, label=labels[i]) for i in range(5)]
        results = est.estimate(dets)
        result_labels = [r.label for r in results]
        assert result_labels == labels


# ---------------------------------------------------------------------------
# TestBBoxFormats
# ---------------------------------------------------------------------------

class TestBBoxFormats:
    def test_bbox_object_with_xy_attrs(self):
        est = MonocularDepthEstimator()
        det = make_det(x=0.1, y=0.2, w=0.3, h=0.4, bbox_as_list=False)
        results = est.estimate([det])
        assert results[0].bbox == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_bbox_as_list(self):
        est = MonocularDepthEstimator()
        det = make_det(x=0.1, y=0.2, w=0.3, h=0.4, bbox_as_list=True)
        results = est.estimate([det])
        assert results[0].bbox == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_bbox_object_and_list_same_depth(self):
        est = MonocularDepthEstimator()
        det_obj = make_det(x=0.2, y=0.3, w=0.15, h=0.2, bbox_as_list=False)
        det_lst = make_det(entity_id=2, x=0.2, y=0.3, w=0.15, h=0.2, bbox_as_list=True)
        r_obj = est.estimate([det_obj])[0]
        r_lst = est.estimate([det_lst])[0]
        assert r_obj.depth_relative == pytest.approx(r_lst.depth_relative)

    def test_missing_bbox_uses_fallback(self):
        est = MonocularDepthEstimator()

        class DetNoBbox:
            entity_id = 99
            label = "unknown"
            confidence = 0.5
            # no bbox attribute

        results = est.estimate([DetNoBbox()])
        assert len(results) == 1
        assert results[0].entity_id == 99
        # fallback bbox [0.5, 0.5, 0.1, 0.1]
        assert results[0].bbox == pytest.approx([0.5, 0.5, 0.1, 0.1])

    def test_bbox_as_tuple(self):
        est = MonocularDepthEstimator()

        class DetTuple:
            entity_id = 7
            label = "car"
            bbox = (0.1, 0.2, 0.3, 0.4)
            confidence = 0.8

        results = est.estimate([DetTuple()])
        assert results[0].bbox == pytest.approx([0.1, 0.2, 0.3, 0.4])


# ---------------------------------------------------------------------------
# TestGetConfig
# ---------------------------------------------------------------------------

class TestGetConfig:
    def test_config_has_expected_keys(self):
        est = MonocularDepthEstimator()
        cfg = est.get_config()
        expected_keys = {
            "camera_height_m",
            "tilt_angle_deg",
            "focal_length_px",
            "frame_width_px",
            "frame_height_px",
            "area_weight",
            "y_weight",
        }
        assert set(cfg.keys()) == expected_keys

    def test_config_tilt_degrees_round_trip(self):
        est = MonocularDepthEstimator(tilt_angle_deg=45.0)
        cfg = est.get_config()
        assert cfg["tilt_angle_deg"] == pytest.approx(45.0)

    def test_config_focal_none_by_default(self):
        est = MonocularDepthEstimator()
        assert est.get_config()["focal_length_px"] is None

    def test_config_focal_set(self):
        est = MonocularDepthEstimator(focal_length_px=1200.0)
        assert est.get_config()["focal_length_px"] == 1200.0


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_estimates_no_errors(self):
        est = MonocularDepthEstimator(focal_length_px=700.0)
        errors = []

        def worker():
            try:
                dets = [
                    make_det(entity_id=i, label="person", x=0.05 * i, y=0.1 * i, w=0.1, h=0.2)
                    for i in range(1, 6)
                ]
                results = est.estimate(dets)
                assert len(results) == 5
                for r in results:
                    assert 0.0 <= r.depth_relative <= 1.0
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_no_cross_contamination(self):
        est = MonocularDepthEstimator()
        results_by_thread: dict[int, list] = {}
        lock = threading.Lock()

        def worker(tid: int):
            dets = [make_det(entity_id=tid * 100 + i) for i in range(3)]
            res = est.estimate(dets)
            with lock:
                results_by_thread[tid] = [r.entity_id for r in res]

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for tid, ids in results_by_thread.items():
            assert ids == [tid * 100, tid * 100 + 1, tid * 100 + 2]


# ---------------------------------------------------------------------------
# TestBoundedOutputs
# ---------------------------------------------------------------------------

class TestBoundedOutputs:
    def _run_extremes(self, est: MonocularDepthEstimator) -> list[DepthEstimate]:
        extremes = [
            make_det(x=0.0, y=0.0, w=0.001, h=0.001, label="person"),   # tiny + top
            make_det(entity_id=2, x=0.0, y=0.9, w=0.99, h=0.99, label="person"),  # huge + bottom
            make_det(entity_id=3, x=0.5, y=0.5, w=0.5, h=0.5, label="car"),       # large centre
            make_det(entity_id=4, x=0.0, y=1.0, w=1.0, h=1.0, label="truck"),     # overflow
        ]
        return est.estimate(extremes)

    def test_depth_relative_always_in_01(self):
        est = MonocularDepthEstimator(focal_length_px=700.0)
        results = self._run_extremes(est)
        for r in results:
            assert 0.0 <= r.depth_relative <= 1.0, f"depth_relative={r.depth_relative} out of [0,1]"

    def test_confidence_always_in_01(self):
        est = MonocularDepthEstimator(focal_length_px=700.0)
        results = self._run_extremes(est)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0, f"confidence={r.confidence} out of [0,1]"

    def test_bounds_with_no_focal(self):
        est = MonocularDepthEstimator(focal_length_px=None)
        results = self._run_extremes(est)
        for r in results:
            assert 0.0 <= r.depth_relative <= 1.0
            assert 0.0 <= r.confidence <= 1.0


# ---------------------------------------------------------------------------
# TestToDict (extra coverage)
# ---------------------------------------------------------------------------

class TestToDict:
    def test_all_required_keys_present(self):
        est = MonocularDepthEstimator()
        det = make_det(label="person")
        results = est.estimate([det])
        d = results[0].to_dict()
        for key in ["entity_id", "label", "bbox", "depth_relative", "depth_metric_m", "confidence"]:
            assert key in d

    def test_depth_metric_none_when_no_focal(self):
        est = MonocularDepthEstimator(focal_length_px=None)
        det = make_det(label="person")
        results = est.estimate([det])
        d = results[0].to_dict()
        assert d["depth_metric_m"] is None

    def test_depth_metric_present_with_focal(self):
        est = MonocularDepthEstimator(focal_length_px=700.0, frame_height_px=720)
        det = make_det(label="person", h=0.4)
        results = est.estimate([det])
        d = results[0].to_dict()
        assert d["depth_metric_m"] is not None
        assert isinstance(d["depth_metric_m"], float)

    def test_bbox_in_dict_is_list(self):
        est = MonocularDepthEstimator()
        det = make_det()
        results = est.estimate([det])
        d = results[0].to_dict()
        assert isinstance(d["bbox"], list)
        assert len(d["bbox"]) == 4
