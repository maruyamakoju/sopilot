"""Normality model persistence — save/load learned anomaly baselines.

The AnomalyDetectorEnsemble learns "normal" patterns over time via EMA.
This module serialises that learned state to JSON files so that:

1. A trained baseline can survive process restarts
2. Different baselines can be managed as named profiles (e.g. "dayshift", "nightshift")
3. Profiles can be shared across cameras with similar environments

File format: ``data/anomaly_profiles/{name}.json`` — pure JSON, no binary deps.

Usage::

    from sopilot.perception.anomaly_profile import (
        save_profile, load_profile, apply_profile, list_profiles,
    )

    # Save current learned state
    path = save_profile(ensemble, "dayshift", Path("data/anomaly_profiles"))

    # Load and apply later
    profile = load_profile(path)
    apply_profile(ensemble, profile)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NormalityProfile:
    """Serialisable snapshot of the AnomalyDetectorEnsemble state."""

    name: str
    created_at: str  # ISO 8601 timestamp
    observations: int
    config: dict[str, Any]  # anomaly_* config snapshot
    behavioral: dict[str, Any] = field(default_factory=dict)
    spatial: dict[str, Any] = field(default_factory=dict)
    temporal: dict[str, Any] = field(default_factory=dict)
    interaction: dict[str, Any] = field(default_factory=dict)


def save_profile(
    ensemble: Any,
    name: str,
    directory: Path,
) -> Path:
    """Serialise ensemble state → JSON file.

    Args:
        ensemble: AnomalyDetectorEnsemble instance.
        name: Human-readable profile name (used as filename stem).
        directory: Target directory for the JSON file.

    Returns:
        Path to the written JSON file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    profile = NormalityProfile(
        name=name,
        created_at=datetime.now(timezone.utc).isoformat(),
        observations=ensemble._observations,
        config={
            "warmup_frames": ensemble._warmup_frames,
            "sigma_threshold": ensemble._sigma_threshold,
            "cooldown_seconds": ensemble._cooldown_seconds,
        },
        behavioral=_extract_behavioral(ensemble._behavioral),
        spatial=_extract_spatial(ensemble._spatial),
        temporal=_extract_temporal(ensemble._temporal),
        interaction=_extract_interaction(ensemble._interaction),
    )

    path = directory / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(profile), f, ensure_ascii=False, indent=2)

    logger.info(
        "Anomaly profile saved: %s (%d observations)", path, profile.observations
    )
    return path


def load_profile(path: Path) -> NormalityProfile:
    """Load a NormalityProfile from a JSON file.

    Args:
        path: Path to the JSON profile file.

    Returns:
        NormalityProfile instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profile = NormalityProfile(
        name=data.get("name", path.stem),
        created_at=data.get("created_at", ""),
        observations=data.get("observations", 0),
        config=data.get("config", {}),
        behavioral=data.get("behavioral", {}),
        spatial=data.get("spatial", {}),
        temporal=data.get("temporal", {}),
        interaction=data.get("interaction", {}),
    )
    logger.info("Anomaly profile loaded: %s (%d observations)", path, profile.observations)
    return profile


def apply_profile(ensemble: Any, profile: NormalityProfile) -> None:
    """Restore ensemble internal state from a profile.

    Args:
        ensemble: AnomalyDetectorEnsemble instance to restore.
        profile: NormalityProfile with saved state.
    """
    ensemble._observations = profile.observations

    # Restore each detector
    ensemble._behavioral.load_state(profile.behavioral)
    ensemble._spatial.load_state(profile.spatial)
    ensemble._temporal.load_state(profile.temporal)
    ensemble._interaction.load_state(profile.interaction)

    logger.info(
        "Anomaly profile applied: %s (%d observations)",
        profile.name, profile.observations,
    )


def list_profiles(directory: Path) -> list[dict[str, Any]]:
    """List available profiles with metadata.

    Args:
        directory: Directory containing profile JSON files.

    Returns:
        List of dicts with keys: name, created_at, observations, path.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    profiles: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            profiles.append({
                "name": data.get("name", path.stem),
                "created_at": data.get("created_at", ""),
                "observations": data.get("observations", 0),
                "path": str(path),
            })
        except Exception:
            logger.warning("Failed to read profile: %s", path)

    return profiles


# ── Internal helpers ──────────────────────────────────────────────────────


def _extract_behavioral(det: Any) -> dict[str, Any]:
    """Extract serialisable state from BehavioralAnomalyDetector."""
    return {
        "speed_mean": det._speed_mean,
        "speed_var": det._speed_var,
        "observations": det._observations,
        "activity_freq": dict(det._activity_freq),
        "activity_observations": det._activity_observations,
    }


def _extract_spatial(det: Any) -> dict[str, Any]:
    """Extract serialisable state from SpatialAnomalyDetector."""
    return {
        "grid": [row[:] for row in det._grid],
        "observations": det._observations,
        "grid_size": det._grid_size,
    }


def _extract_temporal(det: Any) -> dict[str, Any]:
    """Extract serialisable state from TemporalPatternDetector."""
    return {
        "hourly_mean": list(det._hourly_mean),
        "hourly_var": list(det._hourly_var),
        "hourly_obs": list(det._hourly_obs),
    }


def _extract_interaction(det: Any) -> dict[str, Any]:
    """Extract serialisable state from InteractionAnomalyDetector."""
    return {
        "pair_freq": dict(det._pair_freq),
        "observations": det._observations,
    }
