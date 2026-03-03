"""Tests for sopilot/perception/scene_understanding.py

All sopilot imports are avoided; everything is mocked via SimpleNamespace.
"""
from __future__ import annotations

import math
import sys
import os
import threading
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — make the package importable without installation
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sopilot.perception.scene_understanding import SceneAnalysis, SceneUnderstanding


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def make_entity(entity_id, label="person", cx=0.5, cy=0.5):
    bbox = SimpleNamespace(x=cx - 0.05, y=cy - 0.05, w=0.1, h=0.1)
    return SimpleNamespace(entity_id=entity_id, label=label, bbox=bbox, confidence=0.9)


def make_track(entity_id, vx=0.0, vy=0.0):
    return SimpleNamespace(entity_id=entity_id, velocity=(vx, vy))


def make_event(event_type_name):
    et = SimpleNamespace(name=event_type_name)
    return SimpleNamespace(event_type=et, details={})


def make_world_state(
    entities=None,
    tracks=None,
    events=None,
    zones=None,
    entity_count=None,
    timestamp=1000.0,
    frame_number=0,
):
    sg = SimpleNamespace(entities=entities or [])
    active_tracks = {t.entity_id: t for t in (tracks or [])}
    return SimpleNamespace(
        scene_graph=sg,
        active_tracks=active_tracks,
        events=events or [],
        zone_occupancy=zones or {},
        entity_count=entity_count if entity_count is not None else len(entities or []),
        timestamp=timestamp,
        frame_number=frame_number,
    )


def make_spatial_map(crowd=0.0, flow=(0.0, 0.0), hotspots=None):
    return SimpleNamespace(
        get_crowd_density=lambda: crowd,
        get_flow_vector=lambda: flow,
        get_hotspots=lambda top_n=3: hotspots or [],
    )


# ---------------------------------------------------------------------------
# SceneAnalysis — to_dict
# ---------------------------------------------------------------------------

class TestSceneAnalysisToDict:
    def _make(self):
        return SceneAnalysis(
            timestamp=1234.5,
            frame_number=7,
            entity_count=3,
            crowd_density=0.15,
            flow_dx=0.01,
            flow_dy=-0.02,
            flow_speed=0.022,
            anomaly_count=1,
            violation_count=0,
            anomaly_density=0.333,
            risk_index=0.25,
            active_zones=["zone_a"],
            dominant_activity="person",
            summary_ja="テスト",
            summary_en="test.",
        )

    def test_all_14_keys_present(self):
        d = self._make().to_dict()
        expected = {
            "timestamp", "frame_number", "entity_count", "crowd_density",
            "flow_dx", "flow_dy", "flow_speed", "anomaly_count",
            "violation_count", "anomaly_density", "risk_index",
            "active_zones", "dominant_activity", "summary_ja", "summary_en",
        }
        assert set(d.keys()) == expected

    def test_timestamp_value(self):
        assert self._make().to_dict()["timestamp"] == 1234.5

    def test_frame_number_value(self):
        assert self._make().to_dict()["frame_number"] == 7

    def test_entity_count_value(self):
        assert self._make().to_dict()["entity_count"] == 3

    def test_active_zones_is_list(self):
        assert isinstance(self._make().to_dict()["active_zones"], list)

    def test_risk_index_rounded(self):
        a = SceneAnalysis(
            timestamp=0.0, frame_number=0, entity_count=0,
            crowd_density=0.0, flow_dx=0.0, flow_dy=0.0, flow_speed=0.0,
            anomaly_count=0, violation_count=0, anomaly_density=0.0,
            risk_index=0.123456789, active_zones=[], dominant_activity="",
            summary_ja="", summary_en="",
        )
        assert a.to_dict()["risk_index"] == round(0.123456789, 4)


class TestSceneAnalysisRiskBounded:
    def test_risk_in_0_1(self):
        su = SceneUnderstanding()
        ws = make_world_state(entity_count=0)
        a = su.analyze(ws)
        assert 0.0 <= a.risk_index <= 1.0

    def test_risk_bounded_high(self):
        su = SceneUnderstanding()
        entities = [make_entity(i) for i in range(20)]
        events = [make_event("ANOMALY")] * 20 + [make_event("RULE_VIOLATION")] * 20
        ws = make_world_state(entities=entities, events=events)
        a = su.analyze(ws)
        assert 0.0 <= a.risk_index <= 1.0


class TestSceneAnalysisFlowSpeed:
    def test_flow_speed_equals_magnitude(self):
        su = SceneUnderstanding()
        sm = make_spatial_map(flow=(0.03, 0.04))
        ws = make_world_state()
        a = su.analyze(ws, spatial_map=sm)
        expected = round(math.sqrt(0.03**2 + 0.04**2), 4)
        assert abs(a.flow_speed - expected) < 1e-6


# ---------------------------------------------------------------------------
# SceneUnderstanding — init
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_history_size(self):
        su = SceneUnderstanding()
        assert su._history_size == 60

    def test_custom_history_size(self):
        su = SceneUnderstanding(history_size=30)
        assert su._history_size == 30

    def test_history_size_minimum_1(self):
        su = SceneUnderstanding(history_size=0)
        assert su._history_size == 1

    def test_custom_risk_weights_applied(self):
        weights = {"crowd_density": 0.5, "anomaly_density": 0.5, "violation_rate": 0.0, "flow_speed": 0.0}
        su = SceneUnderstanding(risk_weights=weights)
        assert su._risk_w["crowd_density"] == 0.5

    def test_default_risk_weights_fallback(self):
        su = SceneUnderstanding(risk_weights={})
        assert su._risk_w["crowd_density"] == 0.25

    def test_custom_thresholds(self):
        su = SceneUnderstanding(high_crowd_threshold=0.8, high_flow_threshold=0.05)
        assert su._high_crowd == 0.8
        assert su._high_flow == 0.05

    def test_history_initially_empty(self):
        su = SceneUnderstanding()
        assert su.get_history() == []


# ---------------------------------------------------------------------------
# Analyze — basic
# ---------------------------------------------------------------------------

class TestAnalyzeEmpty:
    def test_zero_entity_count(self):
        su = SceneUnderstanding()
        ws = make_world_state()
        a = su.analyze(ws)
        assert a.entity_count == 0

    def test_zero_anomaly_count(self):
        su = SceneUnderstanding()
        a = su.analyze(make_world_state())
        assert a.anomaly_count == 0

    def test_zero_violation_count(self):
        su = SceneUnderstanding()
        a = su.analyze(make_world_state())
        assert a.violation_count == 0

    def test_empty_active_zones(self):
        su = SceneUnderstanding()
        a = su.analyze(make_world_state())
        assert a.active_zones == []

    def test_near_zero_risk(self):
        su = SceneUnderstanding()
        a = su.analyze(make_world_state())
        assert a.risk_index >= 0.0


class TestAnalyzeReturnsSceneAnalysis:
    def test_returns_scene_analysis_instance(self):
        su = SceneUnderstanding()
        result = su.analyze(make_world_state())
        assert isinstance(result, SceneAnalysis)


class TestAnalyzeTimestamp:
    def test_timestamp_propagated(self):
        su = SceneUnderstanding()
        ws = make_world_state(timestamp=9999.5)
        a = su.analyze(ws)
        assert a.timestamp == 9999.5


class TestAnalyzeFrameNumber:
    def test_frame_number_from_world_state(self):
        su = SceneUnderstanding()
        ws = make_world_state(frame_number=42)
        a = su.analyze(ws)
        assert a.frame_number == 42

    def test_frame_number_override(self):
        su = SceneUnderstanding()
        ws = make_world_state(frame_number=0)
        a = su.analyze(ws, frame_number=99)
        assert a.frame_number == 99


# ---------------------------------------------------------------------------
# Analyze — crowd density
# ---------------------------------------------------------------------------

class TestCrowdDensityFromSpatialMap:
    def test_uses_spatial_map_value(self):
        su = SceneUnderstanding()
        sm = make_spatial_map(crowd=0.75)
        ws = make_world_state()
        a = su.analyze(ws, spatial_map=sm)
        assert abs(a.crowd_density - 0.75) < 1e-4

    def test_spatial_map_overrides_entity_count(self):
        su = SceneUnderstanding()
        sm = make_spatial_map(crowd=0.1)
        entities = [make_entity(i) for i in range(15)]
        ws = make_world_state(entities=entities)
        a = su.analyze(ws, spatial_map=sm)
        assert abs(a.crowd_density - 0.1) < 1e-4


class TestCrowdDensityFromEntityCount:
    def test_entity_count_divided_by_20(self):
        su = SceneUnderstanding()
        entities = [make_entity(i) for i in range(10)]
        ws = make_world_state(entities=entities)
        a = su.analyze(ws)
        assert abs(a.crowd_density - 0.5) < 1e-4

    def test_crowd_density_capped_at_1(self):
        su = SceneUnderstanding()
        ws = make_world_state(entity_count=100)
        a = su.analyze(ws)
        assert a.crowd_density <= 1.0


# ---------------------------------------------------------------------------
# Analyze — flow
# ---------------------------------------------------------------------------

class TestFlowFromSpatialMap:
    def test_flow_dx_from_spatial_map(self):
        su = SceneUnderstanding()
        sm = make_spatial_map(flow=(0.05, -0.03))
        ws = make_world_state()
        a = su.analyze(ws, spatial_map=sm)
        assert abs(a.flow_dx - 0.05) < 1e-4

    def test_flow_dy_from_spatial_map(self):
        su = SceneUnderstanding()
        sm = make_spatial_map(flow=(0.05, -0.03))
        ws = make_world_state()
        a = su.analyze(ws, spatial_map=sm)
        assert abs(a.flow_dy - (-0.03)) < 1e-4


class TestFlowFromTracks:
    def test_flow_computed_from_tracks(self):
        su = SceneUnderstanding()
        tracks = [make_track(0, vx=0.04, vy=0.0), make_track(1, vx=0.02, vy=0.0)]
        ws = make_world_state(tracks=tracks)
        a = su.analyze(ws)
        assert abs(a.flow_dx - 0.03) < 1e-4

    def test_flow_dy_averaged(self):
        su = SceneUnderstanding()
        tracks = [make_track(0, vx=0.0, vy=0.06), make_track(1, vx=0.0, vy=0.02)]
        ws = make_world_state(tracks=tracks)
        a = su.analyze(ws)
        assert abs(a.flow_dy - 0.04) < 1e-4


class TestFlowZeroNoTracks:
    def test_zero_flow_no_tracks(self):
        su = SceneUnderstanding()
        ws = make_world_state()
        a = su.analyze(ws)
        assert a.flow_dx == 0.0
        assert a.flow_dy == 0.0


class TestFlowSpeedIsVectorMagnitude:
    def test_flow_speed_sqrt_dx2_dy2(self):
        su = SceneUnderstanding()
        tracks = [make_track(0, vx=0.03, vy=0.04)]
        ws = make_world_state(tracks=tracks)
        a = su.analyze(ws)
        expected = round(math.sqrt(0.03**2 + 0.04**2), 4)
        assert abs(a.flow_speed - expected) < 1e-6


# ---------------------------------------------------------------------------
# Analyze — events
# ---------------------------------------------------------------------------

class TestAnomalyCountCounted:
    def test_anomaly_events_counted(self):
        su = SceneUnderstanding()
        events = [make_event("ANOMALY"), make_event("ANOMALY")]
        ws = make_world_state(events=events, entity_count=5)
        a = su.analyze(ws)
        assert a.anomaly_count == 2

    def test_single_anomaly(self):
        su = SceneUnderstanding()
        ws = make_world_state(events=[make_event("ANOMALY")], entity_count=3)
        a = su.analyze(ws)
        assert a.anomaly_count == 1


class TestViolationCountCounted:
    def test_rule_violation_counted(self):
        su = SceneUnderstanding()
        events = [make_event("RULE_VIOLATION")]
        ws = make_world_state(events=events, entity_count=2)
        a = su.analyze(ws)
        assert a.violation_count == 1

    def test_prolonged_presence_counted(self):
        su = SceneUnderstanding()
        events = [make_event("PROLONGED_PRESENCE")]
        ws = make_world_state(events=events, entity_count=2)
        a = su.analyze(ws)
        assert a.violation_count == 1

    def test_zone_violation_counted(self):
        su = SceneUnderstanding()
        events = [make_event("ZONE_VIOLATION")]
        ws = make_world_state(events=events, entity_count=2)
        a = su.analyze(ws)
        assert a.violation_count == 1

    def test_multiple_violation_types(self):
        su = SceneUnderstanding()
        events = [make_event("RULE_VIOLATION"), make_event("ZONE_VIOLATION")]
        ws = make_world_state(events=events, entity_count=4)
        a = su.analyze(ws)
        assert a.violation_count == 2


class TestOtherEventsIgnored:
    def test_goal_detected_not_counted(self):
        su = SceneUnderstanding()
        events = [make_event("GOAL_DETECTED")]
        ws = make_world_state(events=events, entity_count=2)
        a = su.analyze(ws)
        assert a.anomaly_count == 0
        assert a.violation_count == 0

    def test_track_started_not_counted(self):
        su = SceneUnderstanding()
        events = [make_event("TRACK_STARTED")]
        ws = make_world_state(events=events, entity_count=2)
        a = su.analyze(ws)
        assert a.anomaly_count == 0
        assert a.violation_count == 0


class TestAnomalyDensityCalculated:
    def test_density_is_count_over_entities(self):
        su = SceneUnderstanding()
        events = [make_event("ANOMALY")] * 2
        ws = make_world_state(events=events, entity_count=4)
        a = su.analyze(ws)
        assert abs(a.anomaly_density - 0.5) < 1e-4

    def test_density_zero_no_anomalies(self):
        su = SceneUnderstanding()
        ws = make_world_state(entity_count=5)
        a = su.analyze(ws)
        assert a.anomaly_density == 0.0


# ---------------------------------------------------------------------------
# Analyze — risk
# ---------------------------------------------------------------------------

class TestZeroRiskNoThreats:
    def test_near_zero_risk_empty_scene(self):
        su = SceneUnderstanding()
        ws = make_world_state(entity_count=0)
        a = su.analyze(ws)
        assert a.risk_index >= 0.0

    def test_risk_bounded_above_zero(self):
        su = SceneUnderstanding()
        a = su.analyze(make_world_state())
        assert a.risk_index <= 1.0


class TestHighRiskWithAnomalies:
    def test_anomalies_increase_risk(self):
        su = SceneUnderstanding()
        ws_clean = make_world_state(entity_count=5)
        ws_anomaly = make_world_state(
            events=[make_event("ANOMALY")] * 5, entity_count=5
        )
        r_clean = su.analyze(ws_clean).risk_index
        su2 = SceneUnderstanding()
        r_anomaly = su2.analyze(ws_anomaly).risk_index
        assert r_anomaly > r_clean


class TestRiskAlwaysBounded:
    def test_extreme_inputs_bounded(self):
        su = SceneUnderstanding()
        events = [make_event("ANOMALY")] * 100 + [make_event("RULE_VIOLATION")] * 100
        ws = make_world_state(entity_count=1, events=events)
        sm = make_spatial_map(crowd=5.0, flow=(10.0, 10.0))
        a = su.analyze(ws, spatial_map=sm)
        assert 0.0 <= a.risk_index <= 1.0


class TestRiskWeightsApplied:
    def test_zero_anomaly_weight_reduces_anomaly_influence(self):
        # With anomaly_density weight = 0, many anomalies should have less impact
        weights_normal = {"crowd_density": 0.0, "anomaly_density": 0.45, "violation_rate": 0.0, "flow_speed": 0.0}
        weights_zero = {"crowd_density": 0.0, "anomaly_density": 0.0, "violation_rate": 0.0, "flow_speed": 0.0}
        su_normal = SceneUnderstanding(risk_weights=weights_normal)
        su_zero = SceneUnderstanding(risk_weights=weights_zero)
        events = [make_event("ANOMALY")] * 5
        ws = make_world_state(events=events, entity_count=5)
        r_normal = su_normal.analyze(ws).risk_index
        r_zero = su_zero.analyze(ws).risk_index
        assert r_normal > r_zero


# ---------------------------------------------------------------------------
# Analyze — zones
# ---------------------------------------------------------------------------

class TestActiveZonesFromOccupancy:
    def test_zone_with_entities_included(self):
        su = SceneUnderstanding()
        ws = make_world_state(zones={"zone_a": [1, 2], "zone_b": [3]})
        a = su.analyze(ws)
        assert "zone_a" in a.active_zones
        assert "zone_b" in a.active_zones

    def test_zone_order_preserved(self):
        su = SceneUnderstanding()
        ws = make_world_state(zones={"z1": [1], "z2": [2]})
        a = su.analyze(ws)
        assert len(a.active_zones) == 2


class TestEmptyZoneNotActive:
    def test_empty_zone_excluded(self):
        su = SceneUnderstanding()
        ws = make_world_state(zones={"zone_empty": [], "zone_full": [1]})
        a = su.analyze(ws)
        assert "zone_empty" not in a.active_zones
        assert "zone_full" in a.active_zones


class TestNoZones:
    def test_no_zones_empty_list(self):
        su = SceneUnderstanding()
        ws = make_world_state(zones={})
        a = su.analyze(ws)
        assert a.active_zones == []


# ---------------------------------------------------------------------------
# Analyze — dominant activity
# ---------------------------------------------------------------------------

class TestDominantLabelMostFrequent:
    def test_person_most_frequent(self):
        su = SceneUnderstanding()
        entities = [
            make_entity(0, "person"),
            make_entity(1, "person"),
            make_entity(2, "person"),
            make_entity(3, "car"),
        ]
        ws = make_world_state(entities=entities)
        a = su.analyze(ws)
        assert a.dominant_activity == "person"

    def test_single_entity_dominant(self):
        su = SceneUnderstanding()
        entities = [make_entity(0, "forklift")]
        ws = make_world_state(entities=entities)
        a = su.analyze(ws)
        assert a.dominant_activity == "forklift"


class TestDominantLabelEmpty:
    def test_no_entities_returns_empty_string(self):
        su = SceneUnderstanding()
        ws = make_world_state(entities=[])
        a = su.analyze(ws)
        assert a.dominant_activity == ""


# ---------------------------------------------------------------------------
# Analyze — summaries
# ---------------------------------------------------------------------------

class TestSummaryJaContainsCount:
    def test_entity_count_in_ja_summary(self):
        su = SceneUnderstanding()
        ws = make_world_state(entity_count=7)
        a = su.analyze(ws)
        assert "7" in a.summary_ja

    def test_risk_label_ja_in_summary(self):
        su = SceneUnderstanding()
        ws = make_world_state(entity_count=0)
        a = su.analyze(ws)
        assert "リスク" in a.summary_ja


class TestSummaryEnContainsRisk:
    def test_risk_colon_in_en_summary(self):
        su = SceneUnderstanding()
        ws = make_world_state(entity_count=0)
        a = su.analyze(ws)
        assert "risk:" in a.summary_en

    def test_en_summary_ends_with_period(self):
        su = SceneUnderstanding()
        ws = make_world_state(entity_count=0)
        a = su.analyze(ws)
        assert a.summary_en.endswith(".")


class TestSummaryWithAnomalies:
    def test_anomaly_count_in_ja_summary(self):
        su = SceneUnderstanding()
        events = [make_event("ANOMALY")] * 3
        ws = make_world_state(events=events, entity_count=5)
        a = su.analyze(ws)
        assert "3" in a.summary_ja or "異常" in a.summary_ja

    def test_anomaly_in_en_summary(self):
        su = SceneUnderstanding()
        events = [make_event("ANOMALY")] * 2
        ws = make_world_state(events=events, entity_count=4)
        a = su.analyze(ws)
        assert "anomal" in a.summary_en


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistoryGrows:
    def test_history_grows_with_calls(self):
        su = SceneUnderstanding()
        for i in range(5):
            su.analyze(make_world_state(frame_number=i))
        assert len(su.get_history()) == 5

    def test_each_analysis_in_history(self):
        su = SceneUnderstanding()
        su.analyze(make_world_state(frame_number=10))
        h = su.get_history()
        assert h[0].frame_number == 10


class TestHistoryLimit:
    def test_history_capped_at_max(self):
        su = SceneUnderstanding(history_size=3)
        for i in range(10):
            su.analyze(make_world_state(frame_number=i))
        assert len(su.get_history(n=100)) == 3

    def test_history_retains_latest(self):
        su = SceneUnderstanding(history_size=3)
        for i in range(10):
            su.analyze(make_world_state(frame_number=i))
        h = su.get_history(n=100)
        assert h[-1].frame_number == 9


class TestGetHistoryN:
    def test_get_last_n(self):
        su = SceneUnderstanding()
        for i in range(20):
            su.analyze(make_world_state(frame_number=i))
        h = su.get_history(n=5)
        assert len(h) == 5
        assert h[-1].frame_number == 19

    def test_get_all_if_n_large(self):
        su = SceneUnderstanding()
        for i in range(4):
            su.analyze(make_world_state(frame_number=i))
        h = su.get_history(n=100)
        assert len(h) == 4


class TestGetHistoryEmpty:
    def test_empty_history_returns_empty_list(self):
        su = SceneUnderstanding()
        assert su.get_history() == []

    def test_empty_history_returns_list_type(self):
        su = SceneUnderstanding()
        assert isinstance(su.get_history(), list)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

class TestTrendEmpty:
    def test_empty_returns_zeros(self):
        su = SceneUnderstanding()
        t = su.get_trend()
        assert t["mean_risk"] == 0.0
        assert t["max_risk"] == 0.0
        assert t["delta_risk"] == 0.0
        assert t["n"] == 0


class TestTrendMeanMax:
    def test_mean_and_max_computed(self):
        su = SceneUnderstanding()
        # Force known risk values by using anomaly events
        su.analyze(make_world_state(entity_count=0))
        su.analyze(make_world_state(events=[make_event("ANOMALY")], entity_count=1))
        t = su.get_trend()
        assert t["mean_risk"] >= 0.0
        assert t["max_risk"] >= t["mean_risk"]

    def test_max_risk_is_max_of_all(self):
        su = SceneUnderstanding()
        for _ in range(5):
            su.analyze(make_world_state(entity_count=0))
        big_ws = make_world_state(events=[make_event("ANOMALY")] * 10, entity_count=1)
        a_big = su.analyze(big_ws)
        t = su.get_trend()
        assert t["max_risk"] == a_big.risk_index


class TestTrendDelta:
    def test_delta_is_last_minus_first(self):
        su = SceneUnderstanding()
        first = su.analyze(make_world_state(entity_count=0))
        last = su.analyze(make_world_state(events=[make_event("ANOMALY")] * 5, entity_count=5))
        t = su.get_trend()
        expected = round(last.risk_index - first.risk_index, 4)
        assert abs(t["delta_risk"] - expected) < 1e-6

    def test_delta_zero_single_entry(self):
        su = SceneUnderstanding()
        su.analyze(make_world_state())
        t = su.get_trend()
        assert t["delta_risk"] == 0.0


class TestTrendN:
    def test_n_equals_history_size(self):
        su = SceneUnderstanding()
        for i in range(7):
            su.analyze(make_world_state(frame_number=i))
        assert su.get_trend()["n"] == 7

    def test_n_capped_by_history_limit(self):
        su = SceneUnderstanding(history_size=4)
        for i in range(10):
            su.analyze(make_world_state(frame_number=i))
        assert su.get_trend()["n"] == 4


# ---------------------------------------------------------------------------
# State dict
# ---------------------------------------------------------------------------

class TestGetStateDictKeys:
    def test_all_required_keys(self):
        su = SceneUnderstanding()
        su.analyze(make_world_state())
        sd = su.get_state_dict()
        assert "history_size" in sd
        assert "latest" in sd
        assert "trend" in sd

    def test_latest_is_none_when_empty(self):
        su = SceneUnderstanding()
        sd = su.get_state_dict()
        assert sd["latest"] is None


class TestGetStateDictLatest:
    def test_latest_matches_last_analysis(self):
        su = SceneUnderstanding()
        a = su.analyze(make_world_state(frame_number=55, timestamp=777.0))
        sd = su.get_state_dict()
        assert sd["latest"]["frame_number"] == 55
        assert sd["latest"]["timestamp"] == 777.0

    def test_history_size_in_state_dict(self):
        su = SceneUnderstanding()
        for i in range(3):
            su.analyze(make_world_state())
        sd = su.get_state_dict()
        assert sd["history_size"] == 3


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_analyze_no_errors(self):
        su = SceneUnderstanding(history_size=200)
        errors = []

        def worker(thread_id):
            try:
                for i in range(10):
                    entities = [make_entity(j, "person") for j in range(3)]
                    events = [make_event("ANOMALY")] if i % 3 == 0 else []
                    ws = make_world_state(
                        entities=entities,
                        events=events,
                        frame_number=thread_id * 100 + i,
                    )
                    su.analyze(ws)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_get_history_safe(self):
        su = SceneUnderstanding(history_size=100)
        errors = []

        def reader():
            try:
                for _ in range(20):
                    su.get_history()
                    su.get_trend()
                    su.get_state_dict()
            except Exception as exc:
                errors.append(exc)

        def writer():
            try:
                for i in range(20):
                    su.analyze(make_world_state(frame_number=i))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer)] + [
            threading.Thread(target=reader) for _ in range(4)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == [], f"Thread errors: {errors}"
