"""Insurance-specific prompts for Video-LLM claim assessment.

Prompts designed for accurate severity classification based on
visual evidence observed in dashcam footage. Uses chain-of-thought
(observe → classify) to ground severity in actual video content.
"""

# System prompt -defines the model's role and output format
SYSTEM_PROMPT = """You are an expert automotive insurance claim assessor. Your job is to watch dashcam video footage and produce a structured JSON assessment of the incident.

You must be accurate and evidence-based. Classify severity based ONLY on what you actually observe in the video frames -visible collisions, vehicle damage, pedestrian involvement, speed of impact, etc. Do not guess or assume things not shown in the video.

Your output must be valid JSON with no additional text before or after the JSON object."""


# Main claim assessment prompt -visual evidence-driven
CLAIM_ASSESSMENT_PROMPT = """Analyze this dashcam footage and provide an insurance claim assessment.

**STEP 1 -OBSERVE:** Carefully examine every frame. Note:
- Is there a collision or contact between vehicles?
- What is the speed at the moment of impact (if any)?
- Is there visible vehicle damage (dents, broken glass, deformation)?
- Are pedestrians or cyclists involved?
- Are there near-miss events without actual contact?
- Is this just normal driving with no incident?
- READ ALL TEXT OVERLAYS in the video: speed readings, timestamps, and especially alert text like "COLLISION!", "NEAR MISS", "ACCIDENT", "DANGER"
- TRACK SPEED CHANGES: If the speed overlay drops rapidly (e.g., 60 km/h to 0 km/h), this indicates emergency braking or collision impact
- Note any flash frames, color changes, or screen transitions that signal an event

**CRITICAL EVIDENCE RULES:**
- If any frame shows text like "COLLISION!" or "IMPACT" → this IS a collision, severity is at minimum HIGH
- If speed drops from >40 km/h to 0 km/h while approaching another vehicle → rear-end collision (HIGH)
- If you see emergency braking with a pedestrian nearby → near-miss (MEDIUM)
- A vehicle/object growing rapidly larger in the frame = approaching collision

**STEP 2 -CLASSIFY SEVERITY based on your observations:**

NONE -No incident detected:
- Normal driving, no collision, no near-miss
- Routine traffic flow with no notable events
- Speed remains constant or changes gradually

LOW -Minor incident:
- Very low-speed contact (parking lot bump, nudge while stopped)
- Cosmetic-only damage (small scratch, paint transfer)
- No injury risk whatsoever

MEDIUM -Moderate incident:
- Clear collision at moderate speed
- Visible vehicle damage (bent bumper, cracked body panel)
- Emergency braking to avoid collision (near-miss with close call)
- Pedestrian or obstacle avoidance with sudden braking
- Multiple vehicles involved in contact

HIGH -Serious incident:
- High-speed collision with strong impact force
- Speed dropping from high to zero rapidly while near another vehicle
- Severe vehicle deformation or structural damage
- Airbag deployment visible
- Pedestrian or cyclist struck by vehicle
- Vehicle rollover or spin-out
- Debris scattered across the road
- Any frame showing "COLLISION!" or impact alert text

**STEP 3 -PRODUCE JSON with your assessment:**

```json
{
  "severity": "NONE or LOW or MEDIUM or HIGH",
  "confidence": 0.85,
  "prediction_set": ["MEDIUM"],
  "review_priority": "STANDARD",
  "fault_assessment": {
    "fault_ratio": 50.0,
    "reasoning": "Your fault analysis based on the video",
    "applicable_rules": ["Traffic rules that apply"],
    "scenario_type": "rear_end or intersection or lane_change or parking or pedestrian or other",
    "traffic_signal": null,
    "right_of_way": "Who had right of way"
  },
  "fraud_risk": {
    "risk_score": 0.05,
    "indicators": [],
    "reasoning": "Your fraud assessment"
  },
  "hazards": [
    {
      "type": "collision or near_miss or hazard",
      "actors": ["vehicle_A", "vehicle_B"],
      "spatial_relation": "front or rear or side",
      "timestamp_sec": 10.0
    }
  ],
  "evidence": [
    {
      "timestamp_sec": 10.0,
      "description": "What you observe at this moment"
    }
  ],
  "causal_reasoning": "Detailed explanation of what happened and why",
  "recommended_action": "APPROVE or REVIEW or REJECT"
}
```

**IMPORTANT RULES:**
- severity must be exactly one of: NONE, LOW, MEDIUM, HIGH
- If you see a clear collision with impact, severity is at minimum MEDIUM
- If you see severe damage, high speed, or pedestrian involvement, severity is HIGH
- If you see only normal driving with no events, severity is NONE
- confidence: 0.0 to 1.0 (how certain you are of the severity)
- fault_ratio: 0 to 100 (100 = insured driver fully at fault)
- fraud_risk.risk_score: 0.0 to 1.0
- prediction_set: list of possible severities given your confidence
- review_priority: URGENT (HIGH severity or low confidence), STANDARD (MEDIUM), LOW_PRIORITY (NONE/LOW)
- recommended_action: APPROVE (clear, no issues), REVIEW (needs human check), REJECT (fraud suspected)

**DO NOT copy the example values. Analyze the actual video and provide YOUR assessment.**

Provide your JSON assessment:"""


# Simplified prompt for quick severity classification (no full assessment)
QUICK_SEVERITY_PROMPT = """Watch this dashcam video and classify the incident severity.

Severity levels:
- NONE: No collision or incident, just normal driving
- LOW: Minor contact, cosmetic damage only
- MEDIUM: Clear collision with visible damage
- HIGH: Severe collision, major damage, or pedestrian involvement

First describe what you see, then classify.

Respond with JSON only:
```json
{
  "severity": "NONE or LOW or MEDIUM or HIGH",
  "confidence": 0.85,
  "reasoning": "What you observed in the video"
}
```

Your assessment:"""


# Prompt for fault ratio assessment (given severity)
FAULT_ASSESSMENT_PROMPT = """You are an insurance fault assessor. Analyze this dashcam footage and determine fault allocation.

**COMMON SCENARIOS:**
1. Rear-End Collision: 100% fault for following vehicle (barring sudden reversal)
2. T-Bone at Intersection: 100% fault for vehicle violating right-of-way or signal
3. Lane Change Collision: Usually 100% fault for vehicle changing lanes
4. Head-On Collision: Fault depends on lane position and road markings
5. Parking Lot: Often 50/50 shared fault
6. Pedestrian Strike: Usually 100% fault for driver (unless jaywalking)

**OUTPUT JSON:**
```json
{
  "fault_ratio": 75.0,
  "reasoning": "Following too closely, insufficient braking distance",
  "applicable_rules": ["Following Distance Rule", "Duty to Avoid Collision"],
  "scenario_type": "rear_end",
  "traffic_signal": null,
  "right_of_way": "Lead vehicle had right of way"
}
```

Analyze the video:"""


# Prompt for fraud detection (given incident details)
FRAUD_DETECTION_PROMPT = """You are a fraud detection specialist for insurance claims. Analyze this dashcam footage for fraud indicators.

**RED FLAGS:**
1. Staged Accidents:
   - Deliberate hard braking (brake checking)
   - Multiple witnesses appearing pre-positioned
   - Overly calm or rehearsed behavior
   - Damage inconsistent with impact force

2. Exaggerated Claims:
   - Minor impact but severe damage claims
   - Injury claims inconsistent with collision severity
   - Pre-existing damage visible in video

3. Suspicious Patterns:
   - Multiple similar claims
   - Incident at known fraud hotspot
   - Unusual vehicle positioning

**RISK SCORING:**
- 0.0-0.2: No concerns, genuine accident
- 0.3-0.5: Minor red flags, standard review
- 0.6-0.8: Multiple indicators, investigate
- 0.9-1.0: Strong fraud indicators, reject claim

**OUTPUT JSON:**
```json
{
  "risk_score": 0.15,
  "indicators": [],
  "reasoning": "Accident appears genuine with no fraud indicators detected"
}
```

Analyze the footage:"""


def get_system_prompt() -> str:
    """Get the system prompt for the VLM.

    Returns:
        System prompt defining the assessor role
    """
    return SYSTEM_PROMPT


def get_claim_assessment_prompt(include_calibration: bool = True) -> str:
    """Get the main claim assessment prompt.

    Args:
        include_calibration: Legacy parameter, kept for API compatibility.
                           No longer affects the prompt (calibration bias removed).

    Returns:
        Formatted prompt string
    """
    return CLAIM_ASSESSMENT_PROMPT


def get_quick_severity_prompt() -> str:
    """Get quick severity classification prompt (no full assessment)."""
    return QUICK_SEVERITY_PROMPT


def get_fault_assessment_prompt() -> str:
    """Get fault assessment prompt (given severity)."""
    return FAULT_ASSESSMENT_PROMPT


def get_fraud_detection_prompt() -> str:
    """Get fraud detection prompt (given incident details)."""
    return FRAUD_DETECTION_PROMPT
