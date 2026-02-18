"""Extended tests for Fault Assessment Engine.

Covers edge cases not in test_insurance_domain.py:
- Right turn scenarios
- Parking lot scenarios
- Intersection extended cases
- Combined adjustments (speed + weather + road)
- detect_scenario_type parametrized coverage
"""

import pytest

from insurance_mvp.insurance.fault_assessment import (
    FaultAssessmentEngine,
    FaultAssessmentConfig,
    ScenarioContext,
    ScenarioType,
    TrafficSignal,
    detect_scenario_type,
)


# ============================================================================
# TestRightTurnScenario
# ============================================================================

class TestRightTurnScenario:
    """Right turn collision edge cases."""

    def test_right_turn_no_yield(self):
        """ego_right_of_way=False → 100% fault."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.RIGHT_TURN,
            ego_right_of_way=False,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 100.0
        assert "yield" in result.reasoning.lower()

    def test_right_turn_unclear(self):
        """ego_right_of_way=None → 50% default."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.RIGHT_TURN,
            ego_right_of_way=None,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 50.0

    def test_right_turn_ego_had_right_of_way(self):
        """ego_right_of_way=True → 50% (unclear circumstances)."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.RIGHT_TURN,
            ego_right_of_way=True,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 50.0


# ============================================================================
# TestParkingLotScenario
# ============================================================================

class TestParkingLotScenario:
    """Parking lot collision edge cases."""

    def test_parking_ego_maneuvering(self):
        """ego_lane_change=True → 80% fault."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.PARKING_LOT,
            ego_lane_change=True,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 80.0

    def test_parking_other_maneuvering(self):
        """other_lane_change=True → 20% fault."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.PARKING_LOT,
            other_lane_change=True,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 20.0

    def test_parking_no_maneuvering(self):
        """Both False → 50% split."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.PARKING_LOT,
            ego_lane_change=False,
            other_lane_change=False,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 50.0

    def test_parking_both_maneuvering(self):
        """ego_lane_change=True + other=True → 80% (ego checked first)."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.PARKING_LOT,
            ego_lane_change=True,
            other_lane_change=True,
        )
        result = engine.assess_fault(ctx)
        # The if checks `ego_lane_change or other_lane_change`, then inner if checks ego first
        assert result.fault_ratio == 80.0


# ============================================================================
# TestIntersectionExtended
# ============================================================================

class TestIntersectionExtended:
    """Intersection collision extended cases."""

    def test_intersection_green_ego_right_of_way(self):
        """Green light + ego has right of way → 0% fault."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.INTERSECTION,
            traffic_signal=TrafficSignal.GREEN,
            ego_right_of_way=True,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 0.0

    def test_intersection_green_other_right_of_way(self):
        """Green light + ego does NOT have right of way → 100% fault."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.INTERSECTION,
            traffic_signal=TrafficSignal.GREEN,
            ego_right_of_way=False,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 100.0

    def test_intersection_yellow(self):
        """Yellow light → 50% comparative negligence."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.INTERSECTION,
            traffic_signal=TrafficSignal.YELLOW,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 50.0

    def test_intersection_no_signal(self):
        """No signal → config default (50%)."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.INTERSECTION,
            traffic_signal=TrafficSignal.NONE,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 50.0

    def test_intersection_red_ego_right_of_way(self):
        """Red light + ego has right of way → 0% (other ran red)."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.INTERSECTION,
            traffic_signal=TrafficSignal.RED,
            ego_right_of_way=True,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 0.0

    def test_intersection_unknown_signal(self):
        """Unknown signal → treated as no signal."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.INTERSECTION,
            traffic_signal=TrafficSignal.UNKNOWN,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 50.0


# ============================================================================
# TestCombinedAdjustments
# ============================================================================

class TestCombinedAdjustments:
    """Test combinations of speed, weather, and road adjustments."""

    def test_speed_plus_weather(self):
        """Head-on(50%) + speed(>80) + rain(+5%)."""
        engine = FaultAssessmentEngine(FaultAssessmentConfig(
            weather_adjustment=5.0,
            excessive_speed_adjustment=10.0,
        ))
        ctx = ScenarioContext(
            scenario_type=ScenarioType.HEAD_ON,
            speed_ego_kmh=100.0,
            weather_conditions="rain",
        )
        result = engine.assess_fault(ctx)
        # Base 50% + speed adjustment + 5% weather
        assert result.fault_ratio > 55.0
        assert "speed" in result.reasoning.lower()
        assert "weather" in result.reasoning.lower()

    def test_speed_plus_road(self):
        """Both speed and road adjustments accumulate."""
        engine = FaultAssessmentEngine(FaultAssessmentConfig(
            weather_adjustment=5.0,
            excessive_speed_adjustment=10.0,
        ))
        ctx = ScenarioContext(
            scenario_type=ScenarioType.HEAD_ON,
            speed_ego_kmh=100.0,
            road_conditions="icy",
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio > 55.0
        assert "speed" in result.reasoning.lower()
        assert "road conditions" in result.reasoning.lower()

    def test_weather_plus_road(self):
        """Both weather and road adjustments apply."""
        engine = FaultAssessmentEngine(FaultAssessmentConfig(
            weather_adjustment=5.0,
        ))
        ctx = ScenarioContext(
            scenario_type=ScenarioType.HEAD_ON,
            weather_conditions="snow",
            road_conditions="wet",
        )
        result = engine.assess_fault(ctx)
        # Base 50% + 5% weather + 5% road = 60%
        assert result.fault_ratio == 60.0

    def test_speed_80_exact_threshold(self):
        """speed=80 does NOT trigger >80 check."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.HEAD_ON,
            speed_ego_kmh=80.0,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 50.0  # No adjustment
        assert "speed" not in result.reasoning.lower()

    def test_speed_81_triggers(self):
        """speed=81 triggers adjustment."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.HEAD_ON,
            speed_ego_kmh=81.0,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio > 50.0
        assert "speed" in result.reasoning.lower()

    def test_speed_adjustment_cap_15pct(self):
        """Very high speed → capped at 15% additional fault."""
        engine = FaultAssessmentEngine(FaultAssessmentConfig(
            excessive_speed_adjustment=10.0,
        ))
        ctx = ScenarioContext(
            scenario_type=ScenarioType.HEAD_ON,
            speed_ego_kmh=200.0,  # Extreme speed
        )
        result = engine.assess_fault(ctx)
        # Base 50% + 15% cap = 65%
        assert result.fault_ratio == 65.0

    def test_fault_clamp_to_100(self):
        """Base 100% + adjustments → still clamped to 100%."""
        engine = FaultAssessmentEngine(FaultAssessmentConfig(
            weather_adjustment=5.0,
        ))
        ctx = ScenarioContext(
            scenario_type=ScenarioType.REAR_END,
            speed_ego_kmh=100.0,
            weather_conditions="fog",
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 100.0

    def test_fault_clamp_to_0(self):
        """Base 0% → stays 0% even if adjustments aren't relevant."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.HEAD_ON,
            ego_lane_change=False,
            other_lane_change=True,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 0.0


# ============================================================================
# TestDetectScenarioType
# ============================================================================

class TestDetectScenarioType:
    """Test keyword-based scenario detection."""

    @pytest.mark.parametrize("desc,expected", [
        ("rear-end collision", ScenarioType.REAR_END),
        ("rear end crash", ScenarioType.REAR_END),
        ("hit from behind", ScenarioType.REAR_END),
        ("head-on collision", ScenarioType.HEAD_ON),
        ("head on crash", ScenarioType.HEAD_ON),
        ("frontal impact", ScenarioType.HEAD_ON),
        ("side-swipe on highway", ScenarioType.SIDE_SWIPE),
        ("side swipe during merge", ScenarioType.SIDE_SWIPE),
        ("sideswiped by truck", ScenarioType.SIDE_SWIPE),
        ("left turn accident", ScenarioType.LEFT_TURN),
        ("turning left at light", ScenarioType.LEFT_TURN),
        ("right turn crash", ScenarioType.RIGHT_TURN),
        ("turning right into traffic", ScenarioType.RIGHT_TURN),
        ("intersection collision", ScenarioType.INTERSECTION),
        ("crossroads accident", ScenarioType.INTERSECTION),
        ("parking lot fender bender", ScenarioType.PARKING_LOT),
        ("parking accident", ScenarioType.PARKING_LOT),
        ("pedestrian struck", ScenarioType.PEDESTRIAN),
        ("walker hit by car", ScenarioType.PEDESTRIAN),
    ])
    def test_all_keywords(self, desc, expected):
        """Parametrized test for all 9 scenario keywords."""
        assert detect_scenario_type(desc) == expected

    def test_unknown_description(self):
        """Random text → UNKNOWN."""
        assert detect_scenario_type("random text no keywords") == ScenarioType.UNKNOWN

    def test_case_insensitive(self):
        """'REAR-END' → REAR_END."""
        assert detect_scenario_type("REAR-END collision") == ScenarioType.REAR_END
        assert detect_scenario_type("Head-On crash") == ScenarioType.HEAD_ON

    def test_lane_change_flag_priority(self):
        """Lane change flags override keywords when no specific keyword matches."""
        # ego_lane_change flag detected before other keywords
        result = detect_scenario_type("collision during merge", ego_lane_change=True)
        assert result == ScenarioType.LANE_CHANGE

    def test_lane_change_both_flags(self):
        """Both lane change flags → LANE_CHANGE."""
        result = detect_scenario_type(
            "some collision",
            ego_lane_change=True,
            other_lane_change=True,
        )
        assert result == ScenarioType.LANE_CHANGE

    def test_keyword_beats_lane_change_flag(self):
        """Specific keywords (rear-end) take priority over lane change flags."""
        result = detect_scenario_type("rear-end collision", ego_lane_change=True)
        assert result == ScenarioType.REAR_END


# ============================================================================
# TestLaneChangeScenario (delegates to side_swipe)
# ============================================================================

class TestLaneChangeScenario:
    """Lane change uses same logic as side-swipe."""

    def test_lane_change_ego(self):
        """LANE_CHANGE delegates to _assess_side_swipe."""
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.LANE_CHANGE,
            ego_lane_change=True,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 80.0

    def test_lane_change_other(self):
        engine = FaultAssessmentEngine()
        ctx = ScenarioContext(
            scenario_type=ScenarioType.LANE_CHANGE,
            other_lane_change=True,
        )
        result = engine.assess_fault(ctx)
        assert result.fault_ratio == 20.0
