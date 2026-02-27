"""Insurance-specific prompts for Video-LLM claim assessment.

Prompts designed for accurate severity classification based on
visual evidence observed in dashcam footage. Uses chain-of-thought
(observe → classify) to ground severity in actual video content.
"""

# System prompt -defines the model's role and output format
SYSTEM_PROMPT = """You are an expert automotive insurance claim assessor. Your job is to watch dashcam video footage and produce a structured JSON assessment of the incident.

You must be accurate and evidence-based. Classify severity based ONLY on what you actually observe in the video frames — visible collisions, vehicle damage, pedestrian involvement, speed, etc.

KEY PRINCIPLE: Vehicle-to-vehicle CONTACT = HIGH severity, regardless of speed. Near-miss (close call, no contact) = MEDIUM. No incident = NONE/LOW.

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
- **CONTACT RULE**: If two vehicles visibly make contact — even at low speed — severity is HIGH. Vehicle contact is never LOW or MEDIUM.
- If any frame shows text like "COLLISION!", "IMPACT", or "ACCIDENT" → HIGH severity
- If speed drops rapidly while approaching another vehicle and impact occurs → HIGH (rear-end collision)
- Sudden violent camera shake or jolt combined with nearby vehicles → HIGH (indicates collision force)
- A vehicle/object growing rapidly in frame until filling it = imminent or occurring collision → HIGH
- If you see emergency braking with a pedestrian nearby but NO contact → MEDIUM (near-miss only)
- IMPORTANT: MEDIUM = close call with NO physical contact. HIGH = any physical contact.
- WRONG reasoning: "The impact was minor so severity is MEDIUM" — any vehicle-to-vehicle contact → HIGH
- WRONG reasoning: "No collision occurred so severity is LOW" — near-miss with emergency braking → MEDIUM

**STEP 2 -CLASSIFY SEVERITY based on your observations:**

NONE — No incident:
- Normal driving, no collision, no near-miss, no emergency events
- Routine traffic flow, gradual speed changes

LOW — Minor incident (rare):
- Very low-speed parking lot nudge at walking speed (< 5 km/h) where it is IMPOSSIBLE for injury to occur
- Pre-existing damage visible, no new incident in video
- Do NOT use LOW for any actual moving-vehicle collision — use HIGH

MEDIUM — Near-miss (close call, NO contact):
- Emergency braking near a pedestrian, cyclist, or vehicle with NO physical contact
- Speed drops rapidly while another road user is nearby, vehicles pass dangerously close
- Swerving or evasive maneuver to avoid collision — no contact occurs
**THE MEDIUM/HIGH LINE IS CONTACT: No contact = MEDIUM. Any contact = HIGH.**
**CRITICAL: Do NOT use MEDIUM if vehicles actually touch. Do NOT use MEDIUM for a moving-speed collision.**

HIGH — Collision (any vehicle contact) OR serious incident:
- **Any vehicle-to-vehicle contact, regardless of speed** — a tap, bump, or full crash are all HIGH
- Vehicle contacts a pedestrian, cyclist, or fixed object
- Sudden camera jolt/shake indicating collision impact force
- Speed drops suddenly while vehicles are in proximity (rear-end or intersection collision)
- Airbag deployment, severe deformation, debris from impact
- Vehicle rollover, spin-out, or leaving the road
- Any alert text: "COLLISION!", "IMPACT!", "ACCIDENT"
**WHEN IN DOUBT between MEDIUM and HIGH: if you see or strongly suspect physical contact occurred, classify HIGH.**

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
- **Vehicle contact (any speed) → HIGH** — this is the most important rule
- Near-miss (close call, no contact) → MEDIUM
- Normal driving → NONE
- LOW is rare: only for a parking-lot nudge at walking speed with no injury risk
- confidence: 0.0 to 1.0 (how certain you are of the severity)
- fault_ratio: 0 to 100 (100 = insured driver fully at fault)
- fraud_risk.risk_score: 0.0 to 1.0
- prediction_set: list of possible severities given your confidence
- review_priority: URGENT (HIGH severity or low confidence), STANDARD (MEDIUM), LOW_PRIORITY (NONE/LOW)
- recommended_action: APPROVE (clear, no issues), REVIEW (needs human check), REJECT (fraud suspected)

**DO NOT copy the example values. Analyze the actual video and provide YOUR assessment.**

**OUTPUT ORDER: Determine severity FIRST, then fill in the remaining fields. Start your JSON with the severity field.**

Provide your JSON assessment:"""


# Simplified prompt for quick severity classification (no full assessment)
QUICK_SEVERITY_PROMPT = """Watch this dashcam video and classify the incident severity.

Severity levels:
- NONE: No collision or incident, just normal driving
- LOW: Minor contact, cosmetic damage only
- MEDIUM: Collision with visible damage, OR near-miss with emergency braking / pedestrian avoidance
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


def get_mining_context_addendum(
    peak_sec: float,
    danger_score: float,
    motion_score: float,
    proximity_score: float,
    audio_score: float = 0.0,
) -> str:
    """Build a signal-context string to prepend to the VLM prompt.

    Tells the model WHERE the danger peak is and what the sensor signals
    indicate, so it can focus on the right frames instead of averaging
    over the entire clip.

    Args:
        peak_sec: Time (seconds into the clip) of the danger peak
        danger_score: Fused danger score 0-1
        motion_score: Motion component 0-1 (sudden acceleration/camera jolt)
        proximity_score: Proximity component 0-1 (nearby vehicles/pedestrians)
        audio_score: Audio component 0-1 (impact sounds, horn, tires)

    Returns:
        Context string to prepend to the main prompt
    """
    if danger_score > 0.85:
        intensity = "VERY HIGH — strong collision indicator"
    elif danger_score > 0.6:
        intensity = "HIGH — significant hazard detected"
    else:
        intensity = "MODERATE"

    return (
        f"[SIGNAL ANALYSIS CONTEXT]\n"
        f"Automated hazard sensors detected a DANGER PEAK at t≈{peak_sec:.1f}s in this clip.\n"
        f"- Danger score: {danger_score:.2f}/1.00 ({intensity})\n"
        f"- Motion score: {motion_score:.2f} (sudden acceleration/deceleration, camera jolt)\n"
        f"- Proximity score: {proximity_score:.2f} (nearby vehicles or pedestrians)\n"
        f"- Audio score: {audio_score:.2f} (impact sounds, horn, tire screech)\n"
        f"INSTRUCTION: Focus your analysis on the frames at approximately t={peak_sec:.1f}s.\n"
        f"If you see vehicle contact at or near this timestamp, classify as HIGH.\n\n"
    )
