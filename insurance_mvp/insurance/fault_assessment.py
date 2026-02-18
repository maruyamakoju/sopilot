"""Fault Assessment Logic for Insurance Claims.

Production-grade fault ratio calculation based on traffic scenarios and rules.
Implements industry-standard fault determination logic for collision scenarios.

References:
- National Association of Insurance Commissioners (NAIC) guidelines
- State comparative negligence statutes
- Insurance industry best practices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .schema import FaultAssessment

logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    """Standard collision scenario types."""

    REAR_END = "rear_end"
    HEAD_ON = "head_on"
    SIDE_SWIPE = "side_swipe"
    LEFT_TURN = "left_turn"
    RIGHT_TURN = "right_turn"
    LANE_CHANGE = "lane_change"
    INTERSECTION = "intersection"
    PARKING_LOT = "parking_lot"
    PEDESTRIAN = "pedestrian"
    UNKNOWN = "unknown"


class TrafficSignal(str, Enum):
    """Traffic signal states."""

    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    NONE = "none"
    UNKNOWN = "unknown"


@dataclass
class ScenarioContext:
    """Context information for fault assessment."""

    scenario_type: ScenarioType
    traffic_signal: TrafficSignal = TrafficSignal.UNKNOWN
    speed_ego_kmh: float | None = None
    speed_other_kmh: float | None = None
    ego_lane_change: bool = False
    other_lane_change: bool = False
    ego_braking: bool = False
    other_braking: bool = False
    ego_right_of_way: bool | None = None
    witness_statements: list[str] = None
    weather_conditions: str = "clear"
    road_conditions: str = "dry"

    def __post_init__(self):
        if self.witness_statements is None:
            self.witness_statements = []


class FaultAssessmentConfig:
    """Configuration for fault assessment thresholds and rules."""

    def __init__(
        self,
        # Scenario-specific fault ratios
        rear_end_default: float = 100.0,
        rear_end_sudden_stop: float = 70.0,
        head_on_default: float = 50.0,
        side_swipe_lane_change: float = 80.0,
        side_swipe_unknown: float = 50.0,
        left_turn_default: float = 75.0,
        intersection_no_signal: float = 50.0,
        # Adjustments
        red_light_violation_fault: float = 100.0,
        excessive_speed_adjustment: float = 10.0,
        weather_adjustment: float = 5.0,
        # Thresholds
        excessive_speed_threshold_kmh: float = 20.0,
        min_fault_ratio: float = 0.0,
        max_fault_ratio: float = 100.0,
    ):
        """Initialize fault assessment configuration.

        Args:
            rear_end_default: Default fault for rear vehicle in rear-end collision (%)
            rear_end_sudden_stop: Fault if front vehicle stopped suddenly (%)
            head_on_default: Default fault in head-on collision (50-50 split)
            side_swipe_lane_change: Fault for lane-changing vehicle (%)
            side_swipe_unknown: Fault when lane change is unclear (%)
            left_turn_default: Fault for left-turning vehicle (%)
            intersection_no_signal: Fault at unmarked intersection (%)
            red_light_violation_fault: Fault for red light violation (%)
            excessive_speed_adjustment: Additional fault % per excessive speed
            weather_adjustment: Additional fault % for adverse weather
            excessive_speed_threshold_kmh: Speed over limit considered excessive
            min_fault_ratio: Minimum fault ratio (clamping)
            max_fault_ratio: Maximum fault ratio (clamping)
        """
        self.rear_end_default = rear_end_default
        self.rear_end_sudden_stop = rear_end_sudden_stop
        self.head_on_default = head_on_default
        self.side_swipe_lane_change = side_swipe_lane_change
        self.side_swipe_unknown = side_swipe_unknown
        self.left_turn_default = left_turn_default
        self.intersection_no_signal = intersection_no_signal
        self.red_light_violation_fault = red_light_violation_fault
        self.excessive_speed_adjustment = excessive_speed_adjustment
        self.weather_adjustment = weather_adjustment
        self.excessive_speed_threshold_kmh = excessive_speed_threshold_kmh
        self.min_fault_ratio = min_fault_ratio
        self.max_fault_ratio = max_fault_ratio


class FaultAssessmentEngine:
    """Production-grade fault assessment engine.

    Determines fault ratio (0-100%) based on collision scenario, traffic rules,
    and contextual factors. Implements industry-standard fault determination logic.

    Typical usage:
        >>> engine = FaultAssessmentEngine()
        >>> context = ScenarioContext(
        ...     scenario_type=ScenarioType.REAR_END,
        ...     traffic_signal=TrafficSignal.GREEN,
        ...     speed_ego_kmh=45.0,
        ...     ego_braking=True,
        ... )
        >>> assessment = engine.assess_fault(context)
        >>> print(f"Fault: {assessment.fault_ratio}%")
    """

    def __init__(self, config: FaultAssessmentConfig | None = None):
        """Initialize fault assessment engine.

        Args:
            config: Configuration for fault thresholds and rules.
                    If None, uses default configuration.
        """
        self.config = config or FaultAssessmentConfig()
        logger.info(
            "fault_assessment_engine_initialized",
            rear_end_default=self.config.rear_end_default,
            excessive_speed_threshold=self.config.excessive_speed_threshold_kmh,
        )

    def assess_fault(self, context: ScenarioContext) -> FaultAssessment:
        """Assess fault ratio based on scenario context.

        Args:
            context: Scenario context with collision details.

        Returns:
            FaultAssessment with fault ratio, reasoning, and applicable rules.
        """
        logger.debug(
            "assessing_fault",
            scenario_type=context.scenario_type.value,
            traffic_signal=context.traffic_signal.value,
            speed_ego=context.speed_ego_kmh,
            speed_other=context.speed_other_kmh,
        )

        # Dispatch to scenario-specific handler
        handler_map = {
            ScenarioType.REAR_END: self._assess_rear_end,
            ScenarioType.HEAD_ON: self._assess_head_on,
            ScenarioType.SIDE_SWIPE: self._assess_side_swipe,
            ScenarioType.LEFT_TURN: self._assess_left_turn,
            ScenarioType.RIGHT_TURN: self._assess_right_turn,
            ScenarioType.LANE_CHANGE: self._assess_lane_change,
            ScenarioType.INTERSECTION: self._assess_intersection,
            ScenarioType.PARKING_LOT: self._assess_parking_lot,
            ScenarioType.PEDESTRIAN: self._assess_pedestrian,
        }

        handler = handler_map.get(context.scenario_type, self._assess_unknown)
        base_fault, reasoning_parts, rules = handler(context)

        # Apply contextual adjustments
        adjusted_fault, adjustments = self._apply_adjustments(base_fault, context)
        reasoning_parts.extend(adjustments)

        # Clamp to valid range
        final_fault = np.clip(
            adjusted_fault,
            self.config.min_fault_ratio,
            self.config.max_fault_ratio,
        )

        reasoning = " ".join(reasoning_parts)

        logger.info(
            "fault_assessment_complete",
            scenario_type=context.scenario_type.value,
            base_fault=round(base_fault, 1),
            adjusted_fault=round(adjusted_fault, 1),
            final_fault=round(final_fault, 1),
            num_rules=len(rules),
        )

        return FaultAssessment(
            fault_ratio=round(final_fault, 1),
            reasoning=reasoning,
            applicable_rules=rules,
            scenario_type=context.scenario_type.value,
            traffic_signal=context.traffic_signal.value if context.traffic_signal != TrafficSignal.UNKNOWN else None,
            right_of_way="ego"
            if context.ego_right_of_way is True
            else "other"
            if context.ego_right_of_way is False
            else None,
        )

    def _assess_rear_end(self, context: ScenarioContext) -> tuple[float, list[str], list[str]]:
        """Assess rear-end collision.

        Standard rule: Rear vehicle is typically 100% at fault.
        Exception: Front vehicle made sudden unsafe stop.
        """
        rules = ["Rear vehicle must maintain safe following distance"]

        # Check for sudden stop by front vehicle
        if context.other_braking and context.speed_other_kmh is not None and context.speed_other_kmh < 10.0:
            fault = self.config.rear_end_sudden_stop
            reasoning = [
                "Rear-end collision with sudden stop by front vehicle.",
                f"Fault reduced to {fault}% due to unsafe stop.",
            ]
            rules.append("Front vehicle has duty not to stop suddenly without cause")
        else:
            fault = self.config.rear_end_default
            reasoning = [
                "Rear-end collision.",
                f"Rear vehicle is {fault}% at fault for failing to maintain safe distance.",
            ]

        return fault, reasoning, rules

    def _assess_head_on(self, context: ScenarioContext) -> tuple[float, list[str], list[str]]:
        """Assess head-on collision.

        Standard rule: 50-50 split unless one vehicle crossed center line.
        """
        rules = ["Vehicles must stay in their designated lane"]

        if context.ego_lane_change and not context.other_lane_change:
            # Ego crossed center line
            fault = 100.0
            reasoning = [
                "Head-on collision with ego vehicle crossing center line.",
                "Ego vehicle is 100% at fault for lane violation.",
            ]
            rules.append("Crossing center line into oncoming traffic is prohibited")
        elif context.other_lane_change and not context.ego_lane_change:
            # Other crossed center line
            fault = 0.0
            reasoning = [
                "Head-on collision with other vehicle crossing center line.",
                "Other vehicle is 100% at fault for lane violation.",
            ]
            rules.append("Crossing center line into oncoming traffic is prohibited")
        else:
            # Unclear or both at fault
            fault = self.config.head_on_default
            reasoning = [
                "Head-on collision with unclear lane violation.",
                f"Defaulting to {fault}% fault split (comparative negligence).",
            ]

        return fault, reasoning, rules

    def _assess_side_swipe(self, context: ScenarioContext) -> tuple[float, list[str], list[str]]:
        """Assess side-swipe collision.

        Standard rule: Lane-changing vehicle is typically at fault.
        """
        rules = ["Vehicle changing lanes must ensure safe clearance"]

        if context.ego_lane_change and not context.other_lane_change:
            fault = self.config.side_swipe_lane_change
            reasoning = [
                "Side-swipe collision while ego vehicle changing lanes.",
                f"Ego vehicle is {fault}% at fault for unsafe lane change.",
            ]
        elif context.other_lane_change and not context.ego_lane_change:
            fault = 100.0 - self.config.side_swipe_lane_change
            reasoning = [
                "Side-swipe collision while other vehicle changing lanes.",
                f"Other vehicle is {100 - fault}% at fault for unsafe lane change.",
            ]
        else:
            # Both changing lanes or unclear
            fault = self.config.side_swipe_unknown
            reasoning = [
                "Side-swipe collision with unclear lane change responsibility.",
                f"Defaulting to {fault}% fault split.",
            ]

        return fault, reasoning, rules

    def _assess_left_turn(self, context: ScenarioContext) -> tuple[float, list[str], list[str]]:
        """Assess left turn collision.

        Standard rule: Left-turning vehicle has lower priority.
        """
        rules = ["Left-turning vehicle must yield to oncoming traffic"]

        if context.ego_right_of_way is True:
            # Ego had right of way, other turned in front
            fault = 0.0
            reasoning = [
                "Collision during left turn by other vehicle.",
                "Other vehicle is 100% at fault for failure to yield.",
            ]
        elif context.ego_right_of_way is False:
            # Ego turned in front of other vehicle
            fault = self.config.left_turn_default
            reasoning = [
                "Collision during left turn by ego vehicle.",
                f"Ego vehicle is {fault}% at fault for failure to yield to oncoming traffic.",
            ]
        else:
            # Unclear right of way
            fault = self.config.left_turn_default
            reasoning = [
                "Collision during left turn with unclear right of way.",
                f"Defaulting to {fault}% fault for turning vehicle.",
            ]

        return fault, reasoning, rules

    def _assess_right_turn(self, context: ScenarioContext) -> tuple[float, list[str], list[str]]:
        """Assess right turn collision.

        Standard rule: Right-turning vehicle must yield to pedestrians and cross traffic.
        """
        rules = ["Right-turning vehicle must yield to pedestrians in crosswalk"]

        if context.ego_right_of_way is False:
            fault = 100.0
            reasoning = [
                "Collision during right turn without yielding.",
                "Ego vehicle is 100% at fault for failure to yield.",
            ]
        else:
            fault = 50.0
            reasoning = [
                "Collision during right turn.",
                "Defaulting to 50% fault due to unclear circumstances.",
            ]
            rules.append("Turning vehicle has heightened duty of care")

        return fault, reasoning, rules

    def _assess_lane_change(self, context: ScenarioContext) -> tuple[float, list[str], list[str]]:
        """Assess lane change collision (similar to side-swipe)."""
        return self._assess_side_swipe(context)

    def _assess_intersection(self, context: ScenarioContext) -> tuple[float, list[str], list[str]]:
        """Assess intersection collision.

        Priority: Traffic signal > Stop signs > Right-of-way rules.
        """
        rules = ["Vehicles must obey traffic signals and signs"]

        # Traffic signal present
        if context.traffic_signal in [TrafficSignal.RED, TrafficSignal.YELLOW, TrafficSignal.GREEN]:
            if context.traffic_signal == TrafficSignal.RED:
                if context.ego_right_of_way is False:
                    # Ego ran red light
                    fault = self.config.red_light_violation_fault
                    reasoning = [
                        "Intersection collision after ego ran red light.",
                        f"Ego vehicle is {fault}% at fault for red light violation.",
                    ]
                    rules.append("Running red light is strict liability violation")
                else:
                    # Other ran red light
                    fault = 0.0
                    reasoning = [
                        "Intersection collision after other ran red light.",
                        "Other vehicle is 100% at fault for red light violation.",
                    ]
            elif context.traffic_signal == TrafficSignal.GREEN:
                if context.ego_right_of_way is True:
                    fault = 0.0
                    reasoning = ["Ego had green light.", "Other vehicle violated traffic signal."]
                else:
                    fault = 100.0
                    reasoning = ["Other had green light.", "Ego vehicle violated traffic signal."]
            else:  # Yellow
                fault = 50.0
                reasoning = ["Collision during yellow light.", "Comparative negligence applies."]
        else:
            # No signal (4-way stop or unmarked)
            fault = self.config.intersection_no_signal
            reasoning = [
                "Intersection collision without traffic signal.",
                f"Defaulting to {fault}% fault split.",
            ]
            rules.append("First to arrive at intersection has right of way")

        return fault, reasoning, rules

    def _assess_parking_lot(self, context: ScenarioContext) -> tuple[float, list[str], list[str]]:
        """Assess parking lot collision.

        Special rules apply in parking lots (lower speed, heightened duty).
        """
        rules = ["All vehicles in parking lots must proceed with extra caution"]

        if context.ego_lane_change or context.other_lane_change:
            # Backing/maneuvering vehicle typically at fault
            if context.ego_lane_change:
                fault = 80.0
                reasoning = ["Parking lot collision while ego maneuvering.", "Higher duty on maneuvering vehicle."]
            else:
                fault = 20.0
                reasoning = ["Parking lot collision while other maneuvering.", "Higher duty on maneuvering vehicle."]
        else:
            fault = 50.0
            reasoning = ["Parking lot collision.", "Defaulting to 50% split due to shared duty of care."]

        return fault, reasoning, rules

    def _assess_pedestrian(self, context: ScenarioContext) -> tuple[float, list[str], list[str]]:
        """Assess vehicle-pedestrian collision.

        Vehicles have extremely high duty of care to pedestrians.
        """
        rules = [
            "Vehicles must yield to pedestrians in crosswalks",
            "Drivers have heightened duty to avoid pedestrians",
        ]

        # Vehicle is nearly always at fault
        fault = 100.0
        reasoning = [
            "Vehicle-pedestrian collision.",
            "Vehicle has duty to avoid pedestrians regardless of circumstances.",
        ]

        # Rare exception: pedestrian darted into traffic
        if any("jaywalking" in stmt.lower() or "darted" in stmt.lower() for stmt in context.witness_statements):
            fault = 70.0
            reasoning.append("Fault reduced to 70% due to pedestrian jaywalking.")
            rules.append("Pedestrians must use crosswalks and obey signals")

        return fault, reasoning, rules

    def _assess_unknown(self, context: ScenarioContext) -> tuple[float, list[str], list[str]]:
        """Assess unknown/unclear scenario.

        Conservative default: 50-50 split pending investigation.
        """
        rules = ["Assessment requires manual review due to unclear scenario"]
        fault = 50.0
        reasoning = [
            "Scenario type unclear or insufficient information.",
            "Defaulting to 50% fault split pending manual review.",
        ]
        return fault, reasoning, rules

    def _apply_adjustments(
        self,
        base_fault: float,
        context: ScenarioContext,
    ) -> tuple[float, list[str]]:
        """Apply contextual adjustments to base fault.

        Args:
            base_fault: Base fault percentage before adjustments.
            context: Scenario context.

        Returns:
            Tuple of (adjusted_fault, reasoning_parts).
        """
        adjusted_fault = base_fault
        adjustments = []

        # Excessive speed adjustment
        if context.speed_ego_kmh is not None:
            # Check if speed is significantly high (simple heuristic)
            # In production, this would compare against speed limit
            if context.speed_ego_kmh > 80.0:  # Assuming 60 km/h typical limit
                excess = context.speed_ego_kmh - 60.0
                speed_adjustment = min(
                    (excess / self.config.excessive_speed_threshold_kmh) * self.config.excessive_speed_adjustment,
                    15.0,  # Cap at 15% additional fault
                )
                adjusted_fault += speed_adjustment
                adjustments.append(
                    f"Additional {speed_adjustment:.1f}% fault for excessive speed ({context.speed_ego_kmh:.0f} km/h)."
                )

        # Weather adjustment
        if context.weather_conditions in ["rain", "snow", "fog"]:
            adjusted_fault += self.config.weather_adjustment
            adjustments.append(
                f"Additional {self.config.weather_adjustment:.1f}% fault for adverse weather ({context.weather_conditions})."
            )

        # Road conditions
        if context.road_conditions in ["wet", "icy", "slippery"]:
            adjusted_fault += self.config.weather_adjustment
            adjustments.append(
                f"Additional {self.config.weather_adjustment:.1f}% fault for poor road conditions ({context.road_conditions})."
            )

        return adjusted_fault, adjustments


def detect_scenario_type(
    description: str,
    ego_lane_change: bool = False,
    other_lane_change: bool = False,
) -> ScenarioType:
    """Heuristic scenario type detection from text description.

    Args:
        description: Free-text description of collision.
        ego_lane_change: Whether ego vehicle was changing lanes.
        other_lane_change: Whether other vehicle was changing lanes.

    Returns:
        Detected scenario type (may be UNKNOWN).
    """
    desc_lower = description.lower()

    # Keyword-based detection
    if "rear-end" in desc_lower or "rear end" in desc_lower or "hit from behind" in desc_lower:
        return ScenarioType.REAR_END
    elif "head-on" in desc_lower or "head on" in desc_lower or "frontal" in desc_lower:
        return ScenarioType.HEAD_ON
    elif "side-swipe" in desc_lower or "side swipe" in desc_lower or "sideswiped" in desc_lower:
        return ScenarioType.SIDE_SWIPE
    elif ego_lane_change or other_lane_change:
        return ScenarioType.LANE_CHANGE
    elif "left turn" in desc_lower or "turning left" in desc_lower:
        return ScenarioType.LEFT_TURN
    elif "right turn" in desc_lower or "turning right" in desc_lower:
        return ScenarioType.RIGHT_TURN
    elif "intersection" in desc_lower or "crossroads" in desc_lower:
        return ScenarioType.INTERSECTION
    elif "parking lot" in desc_lower or "parking" in desc_lower:
        return ScenarioType.PARKING_LOT
    elif "pedestrian" in desc_lower or "walker" in desc_lower:
        return ScenarioType.PEDESTRIAN
    else:
        return ScenarioType.UNKNOWN
