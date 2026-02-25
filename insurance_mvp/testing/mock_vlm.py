"""Mock VLM inference for demo/testing mode.

Generates realistic VLM-style JSON output without requiring a GPU or model.
Used when ``backend="mock"`` in cosmos client configuration.
"""

from __future__ import annotations

import hashlib
import json
import random


def mock_inference(prompt: str) -> str:
    """Generate deterministic mock VLM output based on prompt keywords."""
    # NOTE: For pipeline-level mocking, prefer mock_vlm_result() from
    # pipeline/stages/vlm_inference.py which is filename-aware and gives
    # scenario-specific ground truth. This function is retained for
    # cosmos/client.py backward compatibility only.
    prompt_lower = prompt.lower()
    is_collision = any(w in prompt_lower for w in ["collision", "crash", "impact", "hit"])
    is_near_miss = any(w in prompt_lower for w in ["near-miss", "near miss", "avoid", "brake"])
    is_pedestrian = any(w in prompt_lower for w in ["pedestrian", "person", "walker"])
    has_danger = any(w in prompt_lower for w in ["danger", "hazard", "risk", "emergency"])

    # Deterministic randomness keyed on prompt content
    seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    if is_collision and not is_near_miss:
        scenario = _collision_scenario(rng)
    elif is_near_miss or (has_danger and is_pedestrian):
        scenario = _near_miss_scenario(rng)
    else:
        scenario = _normal_scenario(rng)

    hazards = _hazards(rng, is_collision, is_near_miss)

    return json.dumps(
        {
            **scenario,
            "fault_assessment": {
                "fault_ratio": 50.0,
                "reasoning": "VLM analysis complete - see causal reasoning",
                "applicable_rules": [],
                "scenario_type": "auto_detected",
                "traffic_signal": None,
                "right_of_way": None,
            },
            "fraud_risk": {
                "risk_score": 0.0,
                "indicators": [],
                "reasoning": "Video evidence consistent with described scenario",
            },
            "hazards": hazards,
            "evidence": [],
        }
    )


def _collision_scenario(rng: random.Random) -> dict:
    severity = rng.choice(["HIGH", "HIGH", "MEDIUM"])
    return {
        "severity": severity,
        "confidence": round(rng.uniform(0.82, 0.94), 2),
        "prediction_set": ["HIGH", "MEDIUM"] if severity == "HIGH" else ["MEDIUM", "HIGH", "LOW"],
        "review_priority": "URGENT",
        "causal_reasoning": rng.choice([
            "Video analysis reveals rear-end collision scenario. The dashcam footage shows the ego vehicle approaching a slowing vehicle ahead. At approximately 18-20 seconds, brake lights are visible on the lead vehicle, followed by emergency braking. Impact occurs at the 20-second mark with visible forward jolt. The collision appears to be caused by insufficient following distance combined with delayed reaction time.",
            "Analysis of the video indicates a high-severity rear-end collision. The footage clearly shows the lead vehicle's brake lights activating around the 15-18 second mark. Despite this warning, the ego vehicle continues at speed until the last moment, resulting in significant impact force. Weather and road conditions appear clear, suggesting the collision was preventable with proper attention and following distance.",
            "The video evidence demonstrates a classic rear-end collision scenario. Frame-by-frame analysis shows: (1) Lead vehicle begins decelerating at 15s mark, (2) Brake lights clearly visible from 16-20s, (3) Ego vehicle maintains speed until 19s, (4) Emergency braking initiated too late at 19.5s, (5) Impact at 20s with substantial force. The primary causal factor appears to be inadequate following distance and possible driver distraction.",
        ]),
        "recommended_action": "REVIEW",
    }


def _near_miss_scenario(rng: random.Random) -> dict:
    severity = rng.choice(["MEDIUM", "MEDIUM", "LOW"])
    return {
        "severity": severity,
        "confidence": round(rng.uniform(0.75, 0.88), 2),
        "prediction_set": ["MEDIUM", "LOW"] if severity == "MEDIUM" else ["LOW", "MEDIUM", "NONE"],
        "review_priority": "STANDARD",
        "causal_reasoning": rng.choice([
            "The dashcam footage captures a near-miss incident involving a pedestrian. At the 14-15 second mark, a pedestrian enters the vehicle's path from the right side. The driver demonstrates appropriate defensive driving by applying emergency brakes, bringing the vehicle to a stop approximately 2-3 meters before the pedestrian's position. While no collision occurred, the incident represents a moderate hazard that warrants documentation.",
            "Video analysis shows a pedestrian avoidance scenario. The footage indicates good situational awareness by the driver, who identified the pedestrian hazard early and responded with controlled braking. Deceleration begins at 14s, with the vehicle coming to a complete stop well before any collision risk. This represents effective defensive driving in response to an unexpected pedestrian crossing.",
            "The video demonstrates successful hazard avoidance involving a pedestrian crossing. Analysis reveals: (1) Pedestrian enters frame at 13s, (2) Driver initiates braking response at 14s, (3) Vehicle decelerates smoothly from approximately 45 km/h to full stop, (4) Final distance to pedestrian approximately 3 meters. The incident classification is near-miss with no fault attributed to the driver.",
        ]),
        "recommended_action": "DOCUMENT",
    }


def _normal_scenario(rng: random.Random) -> dict:
    severity = rng.choice(["NONE", "NONE", "LOW"])
    return {
        "severity": severity,
        "confidence": round(rng.uniform(0.88, 0.96), 2),
        "prediction_set": ["NONE"] if severity == "NONE" else ["LOW", "NONE"],
        "review_priority": "LOW",
        "causal_reasoning": rng.choice([
            "The dashcam footage shows standard highway driving with no incidents or violations observed. The vehicle maintains consistent speed, appropriate lane position, and safe following distances throughout the recorded period. Traffic conditions are normal with moderate density. No hazardous situations, sudden maneuvers, or safety concerns are evident in the video. This represents routine, safe driving behavior.",
            "Analysis of the video reveals normal driving conditions with no noteworthy events. The driver maintains steady speed appropriate for highway conditions, executes smooth lane changes when necessary, and demonstrates proper following distance. No aggressive driving, traffic violations, or hazardous situations are detected. The footage is consistent with standard safe driving practices.",
            "The video documentation shows routine highway travel. Frame analysis indicates: (1) Consistent speed of 80-90 km/h appropriate for highway, (2) Proper lane discipline maintained, (3) Safe following distances observed, (4) No sudden braking or evasive maneuvers, (5) Clear weather and good visibility throughout. No incidents or concerns identified.",
        ]),
        "recommended_action": "APPROVE",
    }


def _hazards(rng: random.Random, is_collision: bool, is_near_miss: bool) -> list[dict]:
    if is_collision:
        return [{"type": "collision", "actors": ["insured_vehicle", "lead_vehicle"],
                 "spatial_relation": "front", "timestamp_sec": rng.uniform(19.0, 21.0)}]
    elif is_near_miss:
        return [{"type": "near_miss", "actors": ["insured_vehicle", "pedestrian"],
                 "spatial_relation": "front", "timestamp_sec": rng.uniform(14.0, 16.0)}]
    return []
