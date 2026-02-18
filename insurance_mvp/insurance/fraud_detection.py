"""Fraud Detection Logic for Insurance Claims.

Production-grade fraud risk analysis with statistical anomaly detection
and pattern matching against known fraud indicators.

Features:
- Multi-signal fraud detection (visual, audio, behavioral)
- Statistical outlier detection
- Pattern matching against fraud database
- Explainable risk scoring with detailed indicators
- Configurable sensitivity thresholds

References:
- National Insurance Crime Bureau (NICB) fraud patterns
- Coalition Against Insurance Fraud (CAIF) guidelines
- Insurance industry fraud detection best practices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .schema import FraudRisk

logger = logging.getLogger(__name__)


@dataclass
class FraudIndicator:
    """Individual fraud indicator with severity."""

    type: str  # e.g., "no_collision_sound", "damage_inconsistent"
    description: str
    severity: float = 0.0  # 0.0 to 1.0
    confidence: float = 1.0  # How confident we are in this indicator


@dataclass
class VideoEvidence:
    """Video analysis results for fraud detection."""

    has_collision_sound: bool = False
    has_pre_collision_braking: bool = False
    damage_visible: bool = False
    damage_severity: str = "none"  # none, minor, moderate, severe
    vehicle_positioned_suspiciously: bool = False
    speed_at_impact_kmh: float | None = None
    impact_force_estimated: float | None = None  # Relative scale 0-10
    video_quality: str = "good"  # poor, fair, good, excellent
    video_duration_sec: float = 0.0
    suspicious_edits: bool = False


@dataclass
class ClaimHistory:
    """Historical claim data for pattern analysis."""

    vehicle_id: str
    num_previous_claims: int = 0
    claims_last_year: int = 0
    claims_last_month: int = 0
    previous_claim_dates: list[datetime] = field(default_factory=list)
    previous_fraud_flags: int = 0
    total_claimed_amount: float = 0.0
    average_claim_amount: float = 0.0


@dataclass
class ClaimDetails:
    """Current claim details."""

    claimed_amount: float
    injury_claimed: bool = False
    property_damage_claimed: float = 0.0
    medical_claimed: float = 0.0
    claimant_name: str = ""
    claimant_phone: str = ""
    time_to_report_hours: float = 0.0  # Hours from incident to report


class FraudDetectionConfig:
    """Configuration for fraud detection thresholds."""

    def __init__(
        self,
        # Thresholds
        high_risk_threshold: float = 0.65,
        medium_risk_threshold: float = 0.4,
        # Weights for different indicators (sum to ~1.0)
        weight_audio_visual_mismatch: float = 0.25,
        weight_damage_inconsistency: float = 0.20,
        weight_suspicious_positioning: float = 0.15,
        weight_claim_history: float = 0.20,
        weight_claim_amount_anomaly: float = 0.10,
        weight_timing_anomaly: float = 0.10,
        # Claim history thresholds
        suspicious_claims_per_year: int = 3,
        suspicious_claims_per_month: int = 2,
        claim_cluster_days: int = 30,
        # Amount thresholds (multiples of standard deviation)
        claim_amount_outlier_threshold: float = 3.0,
        # Speed/damage consistency
        min_speed_for_damage_kmh: float = 15.0,
        max_speed_no_damage_kmh: float = 10.0,
        # Reporting timing
        suspicious_delay_hours: float = 72.0,
        suspicious_quick_report_hours: float = 0.5,
    ):
        """Initialize fraud detection configuration.

        Args:
            high_risk_threshold: Score above this is high fraud risk (0-1)
            medium_risk_threshold: Score above this is medium fraud risk (0-1)
            weight_audio_visual_mismatch: Weight for audio/visual inconsistency
            weight_damage_inconsistency: Weight for damage pattern mismatch
            weight_suspicious_positioning: Weight for pre-positioning indicator
            weight_claim_history: Weight for historical pattern analysis
            weight_claim_amount_anomaly: Weight for unusually high claim
            weight_timing_anomaly: Weight for suspicious reporting timing
            suspicious_claims_per_year: Claims/year considered suspicious
            suspicious_claims_per_month: Claims/month considered suspicious
            claim_cluster_days: Days window for claim clustering
            claim_amount_outlier_threshold: Std devs for amount outlier
            min_speed_for_damage_kmh: Min speed expected for visible damage
            max_speed_no_damage_kmh: Max speed with no damage is suspicious
            suspicious_delay_hours: Delayed reporting threshold
            suspicious_quick_report_hours: Too-quick reporting threshold
        """
        self.high_risk_threshold = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold
        self.weight_audio_visual_mismatch = weight_audio_visual_mismatch
        self.weight_damage_inconsistency = weight_damage_inconsistency
        self.weight_suspicious_positioning = weight_suspicious_positioning
        self.weight_claim_history = weight_claim_history
        self.weight_claim_amount_anomaly = weight_claim_amount_anomaly
        self.weight_timing_anomaly = weight_timing_anomaly
        self.suspicious_claims_per_year = suspicious_claims_per_year
        self.suspicious_claims_per_month = suspicious_claims_per_month
        self.claim_cluster_days = claim_cluster_days
        self.claim_amount_outlier_threshold = claim_amount_outlier_threshold
        self.min_speed_for_damage_kmh = min_speed_for_damage_kmh
        self.max_speed_no_damage_kmh = max_speed_no_damage_kmh
        self.suspicious_delay_hours = suspicious_delay_hours
        self.suspicious_quick_report_hours = suspicious_quick_report_hours


class FraudDetectionEngine:
    """Production-grade fraud detection engine.

    Analyzes insurance claims for fraud risk using multi-signal analysis:
    - Audio/visual consistency
    - Damage pattern analysis
    - Historical claim patterns
    - Statistical outlier detection
    - Behavioral indicators

    Typical usage:
        >>> engine = FraudDetectionEngine()
        >>> video_evidence = VideoEvidence(
        ...     has_collision_sound=False,
        ...     damage_visible=True,
        ...     speed_at_impact_kmh=45.0,
        ... )
        >>> claim_details = ClaimDetails(claimed_amount=15000.0)
        >>> fraud_risk = engine.detect_fraud(
        ...     video_evidence=video_evidence,
        ...     claim_details=claim_details,
        ... )
        >>> if fraud_risk.risk_score > 0.7:
        ...     print("HIGH FRAUD RISK - Recommend investigation")
    """

    def __init__(
        self,
        config: FraudDetectionConfig | None = None,
        claim_amount_stats: dict[str, float] | None = None,
    ):
        """Initialize fraud detection engine.

        Args:
            config: Configuration for detection thresholds.
            claim_amount_stats: Statistics for claim amounts (mean, std).
                               If None, uses industry defaults.
        """
        self.config = config or FraudDetectionConfig()
        self.claim_amount_stats = claim_amount_stats or {
            "mean": 8000.0,  # Industry average ~$8k
            "std": 5000.0,  # Industry std dev ~$5k
        }

        logger.info(
            "fraud_detection_engine_initialized",
            high_risk_threshold=self.config.high_risk_threshold,
            medium_risk_threshold=self.config.medium_risk_threshold,
            claim_amount_mean=self.claim_amount_stats["mean"],
        )

    def detect_fraud(
        self,
        video_evidence: VideoEvidence,
        claim_details: ClaimDetails,
        claim_history: ClaimHistory | None = None,
    ) -> FraudRisk:
        """Detect fraud risk for an insurance claim.

        Args:
            video_evidence: Video analysis results.
            claim_details: Current claim details.
            claim_history: Historical claim data (if available).

        Returns:
            FraudRisk with overall score, indicators, and reasoning.
        """
        logger.debug(
            "detecting_fraud",
            claimed_amount=claim_details.claimed_amount,
            damage_visible=video_evidence.damage_visible,
            collision_sound=video_evidence.has_collision_sound,
        )

        # Collect all fraud indicators
        indicators: list[FraudIndicator] = []

        # 1. Audio/Visual Consistency Check
        av_indicators = self._check_audio_visual_consistency(video_evidence)
        indicators.extend(av_indicators)

        # 2. Damage Pattern Consistency
        damage_indicators = self._check_damage_consistency(video_evidence)
        indicators.extend(damage_indicators)

        # 3. Suspicious Pre-positioning
        positioning_indicators = self._check_suspicious_positioning(video_evidence)
        indicators.extend(positioning_indicators)

        # 4. Claim History Analysis
        if claim_history is not None:
            history_indicators = self._check_claim_history(claim_history)
            indicators.extend(history_indicators)

        # 5. Claim Amount Anomaly
        amount_indicators = self._check_claim_amount_anomaly(claim_details)
        indicators.extend(amount_indicators)

        # 6. Reporting Timing Anomaly
        timing_indicators = self._check_reporting_timing(claim_details)
        indicators.extend(timing_indicators)

        # Calculate weighted fraud score
        fraud_score = self._calculate_fraud_score(indicators)

        # Generate reasoning
        reasoning = self._generate_reasoning(fraud_score, indicators)

        # Extract indicator descriptions
        indicator_descriptions = [
            f"{ind.type}: {ind.description} (severity={ind.severity:.2f})" for ind in indicators if ind.severity > 0.0
        ]

        logger.info(
            "fraud_detection_complete",
            fraud_score=round(fraud_score, 3),
            num_indicators=len(indicator_descriptions),
            risk_level="HIGH"
            if fraud_score >= self.config.high_risk_threshold
            else "MEDIUM"
            if fraud_score >= self.config.medium_risk_threshold
            else "LOW",
        )

        return FraudRisk(
            risk_score=round(fraud_score, 3),
            indicators=indicator_descriptions,
            reasoning=reasoning,
        )

    def _check_audio_visual_consistency(self, video: VideoEvidence) -> list[FraudIndicator]:
        """Check for audio/visual mismatches.

        Red flags:
        - Claimed collision but no collision sound
        - Damage visible but no impact sound
        - Sound present but no visible damage
        """
        indicators = []

        # No collision sound but damage claimed
        if video.damage_visible and not video.has_collision_sound:
            severity = 0.8 if video.damage_severity in ["moderate", "severe"] else 0.5
            indicators.append(
                FraudIndicator(
                    type="audio_visual_mismatch",
                    description="Visible damage but no collision sound detected in video",
                    severity=severity,
                    confidence=0.9,
                )
            )

        # Collision sound but no visible damage (less suspicious)
        if video.has_collision_sound and not video.damage_visible:
            if video.speed_at_impact_kmh and video.speed_at_impact_kmh > self.config.min_speed_for_damage_kmh:
                indicators.append(
                    FraudIndicator(
                        type="audio_visual_mismatch",
                        description=f"Collision sound at {video.speed_at_impact_kmh:.0f} km/h but no visible damage",
                        severity=0.3,
                        confidence=0.7,
                    )
                )

        return indicators

    def _check_damage_consistency(self, video: VideoEvidence) -> list[FraudIndicator]:
        """Check if damage is consistent with video evidence.

        Red flags:
        - High speed but no damage
        - Low speed but severe damage
        - Damage pattern doesn't match collision angle
        """
        indicators = []

        if video.speed_at_impact_kmh is not None:
            speed = video.speed_at_impact_kmh

            # High speed but no damage
            if speed > self.config.min_speed_for_damage_kmh and not video.damage_visible:
                indicators.append(
                    FraudIndicator(
                        type="damage_inconsistency",
                        description=f"Impact at {speed:.0f} km/h but no visible damage",
                        severity=min(0.7, speed / 50.0),  # Higher speed = more suspicious
                        confidence=0.8,
                    )
                )

            # Low speed but severe damage
            if speed < self.config.max_speed_no_damage_kmh and video.damage_severity == "severe":
                indicators.append(
                    FraudIndicator(
                        type="damage_inconsistency",
                        description=f"Severe damage claimed at low speed ({speed:.0f} km/h)",
                        severity=0.9,
                        confidence=0.85,
                    )
                )

        # Damage claimed but video quality too poor to verify
        if video.damage_visible and video.video_quality == "poor":
            indicators.append(
                FraudIndicator(
                    type="damage_inconsistency",
                    description="Damage claimed but video quality insufficient for verification",
                    severity=0.4,
                    confidence=0.6,
                )
            )

        # Video shows signs of editing
        if video.suspicious_edits:
            indicators.append(
                FraudIndicator(
                    type="video_tampering",
                    description="Video shows signs of editing or tampering",
                    severity=0.95,
                    confidence=0.9,
                )
            )

        return indicators

    def _check_suspicious_positioning(self, video: VideoEvidence) -> list[FraudIndicator]:
        """Check for suspicious vehicle positioning before collision.

        Red flags:
        - Vehicle waiting at intersection unusually long
        - Sudden acceleration into collision
        - Positioning that suggests staged accident
        """
        indicators = []

        if video.vehicle_positioned_suspiciously:
            indicators.append(
                FraudIndicator(
                    type="suspicious_positioning",
                    description="Vehicle appeared to be pre-positioned before collision",
                    severity=0.75,
                    confidence=0.7,
                )
            )

        # No braking before collision (could indicate intentional)
        if not video.has_pre_collision_braking and video.speed_at_impact_kmh and video.speed_at_impact_kmh > 20.0:
            indicators.append(
                FraudIndicator(
                    type="suspicious_positioning",
                    description=f"No braking detected before impact at {video.speed_at_impact_kmh:.0f} km/h",
                    severity=0.5,
                    confidence=0.6,
                )
            )

        return indicators

    def _check_claim_history(self, history: ClaimHistory) -> list[FraudIndicator]:
        """Check claim history for fraud patterns.

        Red flags:
        - Multiple claims in short time period
        - Suspiciously high claim frequency
        - Previous fraud flags
        - Claim amount pattern anomalies
        """
        indicators = []

        # Too many claims per year
        if history.claims_last_year >= self.config.suspicious_claims_per_year:
            severity = min(1.0, history.claims_last_year / (self.config.suspicious_claims_per_year * 2))
            indicators.append(
                FraudIndicator(
                    type="claim_frequency",
                    description=f"Unusually high claim frequency: {history.claims_last_year} claims in past year",
                    severity=severity,
                    confidence=0.9,
                )
            )

        # Too many claims per month
        if history.claims_last_month >= self.config.suspicious_claims_per_month:
            indicators.append(
                FraudIndicator(
                    type="claim_frequency",
                    description=f"Suspicious claim clustering: {history.claims_last_month} claims in past month",
                    severity=0.85,
                    confidence=0.95,
                )
            )

        # Previous fraud flags
        if history.previous_fraud_flags > 0:
            severity = min(1.0, 0.5 + (history.previous_fraud_flags * 0.2))
            indicators.append(
                FraudIndicator(
                    type="fraud_history",
                    description=f"Vehicle has {history.previous_fraud_flags} previous fraud flag(s)",
                    severity=severity,
                    confidence=1.0,
                )
            )

        # Check for claim clustering (multiple claims within short window)
        if len(history.previous_claim_dates) >= 2:
            claim_dates = sorted(history.previous_claim_dates)
            min_gap_days = min((claim_dates[i + 1] - claim_dates[i]).days for i in range(len(claim_dates) - 1))
            if min_gap_days < self.config.claim_cluster_days:
                indicators.append(
                    FraudIndicator(
                        type="claim_clustering",
                        description=f"Multiple claims within {min_gap_days} days",
                        severity=0.7,
                        confidence=0.8,
                    )
                )

        return indicators

    def _check_claim_amount_anomaly(self, claim: ClaimDetails) -> list[FraudIndicator]:
        """Check if claim amount is statistically anomalous.

        Uses z-score to detect outliers (claims far above average).
        """
        indicators = []

        mean = self.claim_amount_stats["mean"]
        std = self.claim_amount_stats["std"]

        # Calculate z-score
        z_score = (claim.claimed_amount - mean) / std if std > 0 else 0.0

        # Unusually high claim
        if z_score > self.config.claim_amount_outlier_threshold:
            severity = min(1.0, (z_score - self.config.claim_amount_outlier_threshold) / 3.0)
            indicators.append(
                FraudIndicator(
                    type="claim_amount_anomaly",
                    description=(
                        f"Claimed amount ${claim.claimed_amount:,.0f} is {z_score:.1f} "
                        f"standard deviations above average (${mean:,.0f})"
                    ),
                    severity=severity,
                    confidence=0.7,
                )
            )

        # Disproportionate medical claim
        if claim.injury_claimed and claim.medical_claimed > 0:
            medical_ratio = claim.medical_claimed / claim.claimed_amount
            if medical_ratio > 0.8:  # Medical > 80% of total claim
                indicators.append(
                    FraudIndicator(
                        type="claim_amount_anomaly",
                        description=f"Medical claims are {medical_ratio * 100:.0f}% of total (unusually high)",
                        severity=0.6,
                        confidence=0.65,
                    )
                )

        return indicators

    def _check_reporting_timing(self, claim: ClaimDetails) -> list[FraudIndicator]:
        """Check for suspicious reporting timing.

        Red flags:
        - Very delayed reporting (suggests planning)
        - Extremely quick reporting (suggests pre-planning)
        """
        indicators = []

        hours = claim.time_to_report_hours

        # Suspicious delay
        if hours > self.config.suspicious_delay_hours:
            severity = min(0.7, hours / (self.config.suspicious_delay_hours * 2))
            indicators.append(
                FraudIndicator(
                    type="timing_anomaly",
                    description=f"Claim reported {hours:.1f} hours after incident (unusually delayed)",
                    severity=severity,
                    confidence=0.6,
                )
            )

        # Suspiciously quick (could indicate pre-planning)
        if hours < self.config.suspicious_quick_report_hours:
            indicators.append(
                FraudIndicator(
                    type="timing_anomaly",
                    description=f"Claim reported within {hours * 60:.0f} minutes (unusually quick)",
                    severity=0.4,
                    confidence=0.5,
                )
            )

        return indicators

    def _calculate_fraud_score(self, indicators: list[FraudIndicator]) -> float:
        """Calculate overall fraud score from indicators.

        Uses weighted sum of indicator severities, adjusted by confidence.
        """
        # Group indicators by type
        indicator_map: dict[str, list[FraudIndicator]] = {}
        for ind in indicators:
            if ind.type not in indicator_map:
                indicator_map[ind.type] = []
            indicator_map[ind.type].append(ind)

        # Calculate weighted score
        weighted_score = 0.0

        # Audio/visual mismatch
        if "audio_visual_mismatch" in indicator_map:
            avg_severity = np.mean([ind.severity * ind.confidence for ind in indicator_map["audio_visual_mismatch"]])
            weighted_score += avg_severity * self.config.weight_audio_visual_mismatch

        # Damage inconsistency
        damage_types = ["damage_inconsistency", "video_tampering"]
        damage_inds = [ind for t in damage_types if t in indicator_map for ind in indicator_map[t]]
        if damage_inds:
            avg_severity = np.mean([ind.severity * ind.confidence for ind in damage_inds])
            weighted_score += avg_severity * self.config.weight_damage_inconsistency

        # Suspicious positioning
        if "suspicious_positioning" in indicator_map:
            avg_severity = np.mean([ind.severity * ind.confidence for ind in indicator_map["suspicious_positioning"]])
            weighted_score += avg_severity * self.config.weight_suspicious_positioning

        # Claim history
        history_types = ["claim_frequency", "fraud_history", "claim_clustering"]
        history_inds = [ind for t in history_types if t in indicator_map for ind in indicator_map[t]]
        if history_inds:
            avg_severity = np.mean([ind.severity * ind.confidence for ind in history_inds])
            weighted_score += avg_severity * self.config.weight_claim_history

        # Claim amount anomaly
        if "claim_amount_anomaly" in indicator_map:
            avg_severity = np.mean([ind.severity * ind.confidence for ind in indicator_map["claim_amount_anomaly"]])
            weighted_score += avg_severity * self.config.weight_claim_amount_anomaly

        # Timing anomaly
        if "timing_anomaly" in indicator_map:
            avg_severity = np.mean([ind.severity * ind.confidence for ind in indicator_map["timing_anomaly"]])
            weighted_score += avg_severity * self.config.weight_timing_anomaly

        # Clamp to [0, 1]
        return np.clip(weighted_score, 0.0, 1.0)

    def _generate_reasoning(self, score: float, indicators: list[FraudIndicator]) -> str:
        """Generate human-readable reasoning for fraud assessment."""
        if score >= self.config.high_risk_threshold:
            risk_level = "HIGH FRAUD RISK"
            recommendation = "Recommend immediate investigation by fraud unit."
        elif score >= self.config.medium_risk_threshold:
            risk_level = "MEDIUM FRAUD RISK"
            recommendation = "Recommend manual review and verification."
        else:
            risk_level = "LOW FRAUD RISK"
            recommendation = "No immediate fraud concerns detected."

        # Summarize top indicators
        top_indicators = sorted(
            [ind for ind in indicators if ind.severity > 0.0],
            key=lambda x: x.severity * x.confidence,
            reverse=True,
        )[:3]

        if top_indicators:
            indicator_summary = " ".join([f"({i + 1}) {ind.description}." for i, ind in enumerate(top_indicators)])
            reasoning = f"{risk_level} (score={score:.2f}). Key indicators: {indicator_summary} {recommendation}"
        else:
            reasoning = f"{risk_level} (score={score:.2f}). No significant fraud indicators detected. {recommendation}"

        return reasoning
