"""Multi-modal fusion engine.

Correlates visual perception events with non-visual signals:
- AUDIO: microphone anomaly detections, sound events
- IOT: sensor triggers (temperature, motion PIR, vibration)
- ACCESS: access control log events (badge swipe, door open/close)
- CUSTOM: any other time-stamped signal

FusionEvents are generated when:
1. AUDIO anomaly co-occurs with VISUAL anomaly within window_seconds → "multi-modal anomaly"
2. ACCESS entry co-occurs with VISUAL person detection → "confirmed access"
3. IOT alert co-occurs with VISUAL zone entry → "sensor-confirmed intrusion"
4. AUDIO loud event co-occurs with VISUAL running/running → "urgency confirmed"

Signals are buffered in-memory (max MAX_SIGNALS). Old signals are auto-pruned.
Thread-safe. No external dependencies beyond stdlib + numpy.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SignalSource(str, Enum):
    AUDIO = "audio"
    IOT = "iot"
    ACCESS = "access"
    CUSTOM = "custom"


class FusionType(str, Enum):
    MULTIMODAL_ANOMALY = "multimodal_anomaly"
    CONFIRMED_ACCESS = "confirmed_access"
    SENSOR_CONFIRMED_INTRUSION = "sensor_confirmed_intrusion"
    URGENCY_CONFIRMED = "urgency_confirmed"
    CORRELATED_EVENT = "correlated_event"


@dataclass
class MultimodalSignal:
    """A non-visual signal ingested from an external source."""

    signal_id: str
    source: SignalSource
    signal_type: str          # e.g. "gunshot", "motion_pir", "door_open", "temperature_high"
    timestamp: float          # Unix time
    value: float = 0.0        # Normalized signal magnitude [0.0, 1.0]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "source": self.source.value,
            "signal_type": self.signal_type,
            "timestamp": self.timestamp,
            "value": self.value,
            "metadata": self.metadata,
        }


@dataclass
class FusionEvent:
    """A cross-modal fusion result."""

    fusion_id: str
    timestamp: float
    fusion_type: FusionType
    signal_ids: list[str]       # Contributing signal IDs
    visual_event_type: str      # Contributing visual event type name (or "")
    confidence: float           # [0.0, 1.0]
    description_ja: str
    description_en: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fusion_id": self.fusion_id,
            "timestamp": self.timestamp,
            "fusion_type": self.fusion_type.value,
            "signal_ids": self.signal_ids,
            "visual_event_type": self.visual_event_type,
            "confidence": self.confidence,
            "description_ja": self.description_ja,
            "description_en": self.description_en,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# FusionRule helpers
# ---------------------------------------------------------------------------

_DESCRIPTIONS: dict[FusionType, tuple[str, str]] = {
    FusionType.MULTIMODAL_ANOMALY: (
        "マルチモーダル異常: 視覚的異常と音響異常が同時に検出されました",
        "Multi-modal anomaly: visual and audio anomalies detected simultaneously",
    ),
    FusionType.CONFIRMED_ACCESS: (
        "確認済みアクセス: 視覚的な人物検出とアクセスログが一致しました",
        "Confirmed access: visual person detection corroborated by access control log",
    ),
    FusionType.SENSOR_CONFIRMED_INTRUSION: (
        "センサー確認侵入: ゾーン侵入がIoTセンサーにより確認されました",
        "Sensor-confirmed intrusion: zone entry corroborated by IoT sensor trigger",
    ),
    FusionType.URGENCY_CONFIRMED: (
        "緊急事態確認: 走行動作と音響警報が同時検出されました",
        "Urgency confirmed: running motion and audio alarm detected simultaneously",
    ),
    FusionType.CORRELATED_EVENT: (
        "相関イベント: 視覚イベントと高強度信号が時間的に相関しています",
        "Correlated event: visual event temporally correlated with high-magnitude signal",
    ),
}


def _make_fusion_event(
    fusion_type: FusionType,
    timestamp: float,
    signal_ids: list[str],
    visual_event_type: str,
    confidence: float,
    details: dict[str, Any] | None = None,
) -> FusionEvent:
    desc_ja, desc_en = _DESCRIPTIONS[fusion_type]
    return FusionEvent(
        fusion_id=str(uuid.uuid4()),
        timestamp=timestamp,
        fusion_type=fusion_type,
        signal_ids=signal_ids,
        visual_event_type=visual_event_type,
        confidence=round(min(1.0, max(0.0, confidence)), 4),
        description_ja=desc_ja,
        description_en=desc_en,
        details=details or {},
    )


# ---------------------------------------------------------------------------
# MultimodalFusionEngine
# ---------------------------------------------------------------------------


class MultimodalFusionEngine:
    """Correlates visual WorldState events with non-visual signals."""

    MAX_SIGNALS: int = 1000
    DEFAULT_WINDOW: float = 5.0   # seconds for co-occurrence check

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._signals: list[MultimodalSignal] = []
        self._fusion_log: list[FusionEvent] = []
        self._total_ingested: int = 0
        self._total_fused: int = 0
        # Track which (signal_id, visual_event_key) pairs have already fired
        # to prevent duplicate fusion events within a single call.
        self._fired_pairs: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------
    # Signal ingestion
    # ------------------------------------------------------------------

    def ingest_signal(self, signal: MultimodalSignal) -> str:
        """Add a signal to the buffer. Returns signal_id.

        If the buffer exceeds MAX_SIGNALS, the oldest 20 % are pruned first.
        """
        with self._lock:
            # Prune if needed before adding
            if len(self._signals) >= self.MAX_SIGNALS:
                prune_count = max(1, int(self.MAX_SIGNALS * 0.20))
                # Sort by timestamp ascending so oldest are first
                self._signals.sort(key=lambda s: s.timestamp)
                self._signals = self._signals[prune_count:]
                logger.debug(
                    "multimodal: pruned %d oldest signals (buffer was full)", prune_count
                )

            self._signals.append(signal)
            self._total_ingested += 1
            return signal.signal_id

    def create_signal(
        self,
        source: SignalSource | str,
        signal_type: str,
        timestamp: float | None = None,
        value: float = 0.0,
        metadata: dict | None = None,
    ) -> MultimodalSignal:
        """Convenience constructor. Also ingests the signal. Returns it."""
        if isinstance(source, str):
            source = SignalSource(source)
        signal = MultimodalSignal(
            signal_id=str(uuid.uuid4()),
            source=source,
            signal_type=signal_type,
            timestamp=timestamp if timestamp is not None else time.time(),
            value=float(value),
            metadata=metadata or {},
        )
        self.ingest_signal(signal)
        return signal

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def fuse_with_visual(
        self,
        world_state: Any,   # WorldState
        window_seconds: float = DEFAULT_WINDOW,
    ) -> list[FusionEvent]:
        """Correlate current WorldState events with recent signals.

        For each visual EntityEvent in world_state.events:
          1. Find signals within [event.timestamp - window, event.timestamp + window]
          2. Apply fusion rules (see module docstring)
          3. Build FusionEvent if any rule matches

        Returns list of new FusionEvents (also stored in _fusion_log).
        Deduplication: a (signal_id, visual_event_key) pair fires at most once
        per call to this method (pairs are reset between calls).
        """
        from sopilot.perception.types import EntityEventType  # lazy import

        new_fusions: list[FusionEvent] = []

        with self._lock:
            # Reset pair tracking for this fusion pass
            self._fired_pairs = set()

            visual_events = getattr(world_state, "events", [])
            if not visual_events or not self._signals:
                return []

            for visual_event in visual_events:
                evt_ts: float = getattr(visual_event, "timestamp", 0.0)
                evt_type = getattr(visual_event, "event_type", None)
                evt_details: dict = getattr(visual_event, "details", {})
                evt_key = f"{evt_type}_{getattr(visual_event, 'entity_id', 0)}_{evt_ts}"

                # Find all signals within the co-occurrence window
                window_signals = [
                    s for s in self._signals
                    if abs(s.timestamp - evt_ts) <= window_seconds
                ]
                if not window_signals:
                    continue

                audio_signals = [s for s in window_signals if s.source == SignalSource.AUDIO]
                iot_signals   = [s for s in window_signals if s.source == SignalSource.IOT]
                access_signals = [s for s in window_signals if s.source == SignalSource.ACCESS]

                fused: FusionEvent | None = None

                # ── Rule 1: ANOMALY visual + AUDIO anomaly (value >= 0.6) ──
                if evt_type == EntityEventType.ANOMALY:
                    strong_audio = [
                        s for s in audio_signals if s.value >= 0.6
                    ]
                    if strong_audio:
                        best = max(strong_audio, key=lambda s: s.value)
                        pair_key = (best.signal_id, evt_key)
                        if pair_key not in self._fired_pairs:
                            self._fired_pairs.add(pair_key)
                            confidence = min(1.0, 0.7 + best.value * 0.3)
                            fused = _make_fusion_event(
                                FusionType.MULTIMODAL_ANOMALY,
                                timestamp=evt_ts,
                                signal_ids=[best.signal_id],
                                visual_event_type=evt_type.value if hasattr(evt_type, "value") else str(evt_type),
                                confidence=confidence,
                                details={
                                    "audio_signal_type": best.signal_type,
                                    "audio_value": best.value,
                                },
                            )

                # ── Rule 2: ZONE_ENTERED + ACCESS door_open/badge ──
                if fused is None and evt_type == EntityEventType.ZONE_ENTERED:
                    access_door_badge = [
                        s for s in access_signals
                        if "door_open" in s.signal_type or "badge" in s.signal_type
                    ]
                    if access_door_badge:
                        best = access_door_badge[0]
                        pair_key = (best.signal_id, evt_key)
                        if pair_key not in self._fired_pairs:
                            self._fired_pairs.add(pair_key)
                            fused = _make_fusion_event(
                                FusionType.CONFIRMED_ACCESS,
                                timestamp=evt_ts,
                                signal_ids=[best.signal_id],
                                visual_event_type=evt_type.value if hasattr(evt_type, "value") else str(evt_type),
                                confidence=0.85,
                                details={
                                    "access_signal_type": best.signal_type,
                                },
                            )

                # ── Rule 3: ZONE_ENTERED + IOT motion/pir ──
                if fused is None and evt_type == EntityEventType.ZONE_ENTERED:
                    iot_motion = [
                        s for s in iot_signals
                        if "motion" in s.signal_type or "pir" in s.signal_type
                    ]
                    if iot_motion:
                        best = iot_motion[0]
                        pair_key = (best.signal_id, evt_key)
                        if pair_key not in self._fired_pairs:
                            self._fired_pairs.add(pair_key)
                            fused = _make_fusion_event(
                                FusionType.SENSOR_CONFIRMED_INTRUSION,
                                timestamp=evt_ts,
                                signal_ids=[best.signal_id],
                                visual_event_type=evt_type.value if hasattr(evt_type, "value") else str(evt_type),
                                confidence=0.90,
                                details={
                                    "iot_signal_type": best.signal_type,
                                },
                            )

                # ── Rule 4: "running" in description_ja + AUDIO alarm/shout ──
                if fused is None:
                    desc_ja: str = evt_details.get("description_ja", "")
                    if "running" in desc_ja or "走" in desc_ja:
                        urgency_audio = [
                            s for s in audio_signals
                            if "alarm" in s.signal_type or "shout" in s.signal_type
                        ]
                        if urgency_audio:
                            best = urgency_audio[0]
                            pair_key = (best.signal_id, evt_key)
                            if pair_key not in self._fired_pairs:
                                self._fired_pairs.add(pair_key)
                                fused = _make_fusion_event(
                                    FusionType.URGENCY_CONFIRMED,
                                    timestamp=evt_ts,
                                    signal_ids=[best.signal_id],
                                    visual_event_type=evt_type.value if hasattr(evt_type, "value") else str(evt_type),
                                    confidence=0.80,
                                    details={
                                        "audio_signal_type": best.signal_type,
                                        "description_ja": desc_ja,
                                    },
                                )

                # ── Catch-all: any signal with value >= 0.8 co-occurring ──
                if fused is None:
                    high_value = [s for s in window_signals if s.value >= 0.8]
                    if high_value:
                        best = high_value[0]
                        pair_key = (best.signal_id, evt_key)
                        if pair_key not in self._fired_pairs:
                            self._fired_pairs.add(pair_key)
                            fused = _make_fusion_event(
                                FusionType.CORRELATED_EVENT,
                                timestamp=evt_ts,
                                signal_ids=[best.signal_id],
                                visual_event_type=evt_type.value if hasattr(evt_type, "value") else str(evt_type),
                                confidence=0.5,
                                details={
                                    "signal_source": best.source.value,
                                    "signal_type": best.signal_type,
                                    "signal_value": best.value,
                                },
                            )

                if fused is not None:
                    self._fusion_log.append(fused)
                    self._total_fused += 1
                    new_fusions.append(fused)

        return new_fusions

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_signals(
        self,
        source: SignalSource | str | None = None,
        lookback_seconds: float = 60.0,
    ) -> list[MultimodalSignal]:
        """Return signals from the last lookback_seconds, optionally filtered by source."""
        cutoff = time.time() - lookback_seconds
        with self._lock:
            result = [s for s in self._signals if s.timestamp >= cutoff]
            if source is not None:
                if isinstance(source, str):
                    source = SignalSource(source)
                result = [s for s in result if s.source == source]
            return list(result)

    def get_fusion_log(self, n: int = 20) -> list[FusionEvent]:
        """Return the most recent n fusion events."""
        with self._lock:
            return list(self._fusion_log[-n:])

    def clear_signals(self, older_than_seconds: float = 300.0) -> int:
        """Remove signals older than older_than_seconds. Returns count removed."""
        cutoff = time.time() - older_than_seconds
        with self._lock:
            before = len(self._signals)
            self._signals = [s for s in self._signals if s.timestamp >= cutoff]
            removed = before - len(self._signals)
            return removed

    def get_state_dict(self) -> dict:
        """Return summary statistics for API."""
        with self._lock:
            signals_by_source: dict[str, int] = {}
            for s in self._signals:
                key = s.source.value
                signals_by_source[key] = signals_by_source.get(key, 0) + 1

            return {
                "buffered_signals": len(self._signals),
                "total_ingested": self._total_ingested,
                "total_fused": self._total_fused,
                "signals_by_source": signals_by_source,
                "recent_fusion_count": len(self._fusion_log),
            }
