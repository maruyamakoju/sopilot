"""Causal Chain Analyzer — temporal backtracking for incident root-cause.

Phase 44: Given an incident event, searches a rolling event buffer to
reconstruct the causal chain leading to the incident.

Example chain::

    [15s ago] PPE未着用検出         (confidence 0.91)
    [8s ago]  制限区域侵入           (confidence 0.87)
    [3s ago]  走行検出 (behavior)    (confidence 0.82)
    [0s]      転倒検出 → INCIDENT    (confidence 0.95)

Architecture::

    CausalAnalyzer.observe_event(event_type, ...) → buffered
    IncidentReporter → CausalAnalyzer.analyze(incident_type, ts, cam)
    → CausalChain with JP/EN explanation + LLaVA enhancement

Causal rules (priority-based matching):
    - 制限区域侵入 → causes → 転倒 / その他 (spatial violation)
    - PPE未着用    → causes → 安全違反全般
    - 走行         → causes → 転倒 / 衝突
    - 徘徊         → causes → 不審行動全般

Usage::

    analyzer = CausalAnalyzer()
    analyzer.observe_event("PPE未着用", ts=t-15, camera_id="cam01")
    analyzer.observe_event("走行", ts=t-3, camera_id="cam01")
    chain = analyzer.analyze("転倒", incident_time=t, camera_id="cam01")
    print(chain.explanation_ja)
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BUFFER_SIZE = 500       # max events in rolling buffer
_DEFAULT_LOOKBACK = 30.0  # seconds to look back for causes

# ---------------------------------------------------------------------------
# Causal rules: predecessor_keyword → set of incident types it may cause
# ---------------------------------------------------------------------------

_CAUSAL_RULES: dict[str, list[str]] = {
    "制限区域": ["転倒", "衝突", "立入禁止", "危険行為"],
    "ppe": ["安全違反", "ヘルメット", "ベスト", "helmet"],
    "ヘルメット": ["安全違反", "ppe"],
    "ベスト": ["安全違反", "ppe"],
    "走行": ["転倒", "衝突"],
    "徘徊": ["不審", "滞留"],
    "転倒": ["負傷", "倒れ"],
    "密集": ["混雑", "パニック"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CausalEvent:
    """A single observed event stored in the causal buffer."""

    event_id: str
    event_type: str
    timestamp: float
    camera_id: str
    confidence: float = 1.0
    description_ja: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "confidence": round(self.confidence, 3),
            "description_ja": self.description_ja,
            "seconds_ago": round(time.time() - self.timestamp, 1),
        }


@dataclass
class CausalChain:
    """Reconstructed causal chain for an incident."""

    chain_id: str
    incident_type: str
    incident_time: float
    camera_id: str
    chain: list[CausalEvent] = field(default_factory=list)
    explanation_ja: str = ""
    explanation_en: str = ""
    time_span_seconds: float = 0.0
    causal_confidence: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "incident_type": self.incident_type,
            "incident_time": self.incident_time,
            "camera_id": self.camera_id,
            "chain": [e.to_dict() for e in self.chain],
            "explanation_ja": self.explanation_ja,
            "explanation_en": self.explanation_en,
            "time_span_seconds": round(self.time_span_seconds, 1),
            "causal_confidence": round(self.causal_confidence, 3),
            "chain_length": len(self.chain),
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# CausalAnalyzer
# ---------------------------------------------------------------------------


class CausalAnalyzer:
    """Rolling event buffer + temporal backtracking for incident root-cause.

    Parameters
    ----------
    buffer_size:
        Maximum events to retain in the rolling buffer.
    lookback_seconds:
        Default time window to search for causal events.
    llava_client:
        Optional LLaVA client for natural language explanation enhancement.
    persist_path:
        Optional JSON path for chain persistence.
    max_chains:
        Maximum chains to retain in memory.
    """

    def __init__(
        self,
        buffer_size: int = _BUFFER_SIZE,
        lookback_seconds: float = _DEFAULT_LOOKBACK,
        llava_client: Any = None,
        persist_path: Path | None = None,
        max_chains: int = 100,
    ) -> None:
        self._buffer_size = buffer_size
        self._lookback = lookback_seconds
        self._llava = llava_client
        self._persist_path = persist_path
        self._max_chains = max_chains

        self._lock = threading.Lock()
        self._buffer: deque[CausalEvent] = deque(maxlen=buffer_size)
        self._chains: deque[CausalChain] = deque(maxlen=max_chains)
        self._observe_count = 0
        self._analyze_count = 0
        self._created_at = time.time()

        if persist_path:
            self._load()

        logger.info(
            "CausalAnalyzer created (buffer=%d, lookback=%.0fs, llava=%s)",
            buffer_size,
            lookback_seconds,
            "yes" if llava_client else "no",
        )

    # ── Observation ────────────────────────────────────────────────────

    def observe_event(
        self,
        event_type: str,
        timestamp: float | None = None,
        camera_id: str = "cam",
        confidence: float = 1.0,
        description_ja: str = "",
    ) -> str:
        """Buffer an observed event for future causal analysis.

        Parameters
        ----------
        event_type:
            Short event label (e.g., "PPE未着用", "走行", "制限区域侵入").
        timestamp:
            Unix timestamp (default: now).
        camera_id:
            Source camera ID.
        confidence:
            Detection confidence [0, 1].
        description_ja:
            Optional Japanese description.

        Returns
        -------
        str
            Event ID (UUID4).
        """
        eid = str(uuid.uuid4())
        ts = timestamp or time.time()
        event = CausalEvent(
            event_id=eid,
            event_type=event_type,
            timestamp=ts,
            camera_id=camera_id,
            confidence=float(confidence),
            description_ja=description_ja or f"{event_type}を検出",
        )
        with self._lock:
            self._buffer.append(event)
            self._observe_count += 1
        return eid

    # ── Analysis ───────────────────────────────────────────────────────

    def analyze(
        self,
        incident_type: str,
        incident_time: float | None = None,
        camera_id: str | None = None,
        lookback_seconds: float | None = None,
    ) -> CausalChain:
        """Reconstruct the causal chain leading to an incident.

        Parameters
        ----------
        incident_type:
            The incident event type (e.g., "転倒", "PPE未着用").
        incident_time:
            Unix timestamp of the incident (default: now).
        camera_id:
            Restrict search to this camera (None = all cameras).
        lookback_seconds:
            Time window to search (default: self._lookback).

        Returns
        -------
        CausalChain
            Reconstructed chain with JP/EN explanations.
        """
        ts = incident_time or time.time()
        window = lookback_seconds or self._lookback
        window_start = ts - window

        with self._lock:
            buffer_snapshot = list(self._buffer)
            self._analyze_count += 1

        # Filter: same camera, within lookback window, before incident
        candidates = [
            e for e in buffer_snapshot
            if e.timestamp >= window_start
            and e.timestamp <= ts
            and (camera_id is None or e.camera_id == camera_id)
            and e.event_type != incident_type  # exclude the incident itself
        ]

        # Sort chronologically
        candidates.sort(key=lambda e: e.timestamp)

        # Score each candidate by causal relevance
        scored: list[tuple[CausalEvent, float]] = []
        for event in candidates:
            rel = _causal_relevance(event.event_type, incident_type)
            if rel > 0:
                scored.append((event, rel))
            # Also include all events from the same camera (context)
            elif camera_id and event.camera_id == camera_id:
                scored.append((event, 0.1))

        # Keep top-5 by relevance, preserve chronological order
        scored.sort(key=lambda x: x[1], reverse=True)
        top_events = [e for e, _ in scored[:5]]
        top_events.sort(key=lambda e: e.timestamp)

        # Calculate span
        time_span = ts - top_events[0].timestamp if top_events else 0.0

        # Causal confidence: mean relevance of chain events
        if scored:
            chain_events_set = {e.event_id for e in top_events}
            chain_scored = [(e, r) for e, r in scored if e.event_id in chain_events_set]
            causal_conf = sum(r for _, r in chain_scored) / max(1, len(chain_scored))
        else:
            causal_conf = 0.0

        # Build explanations
        explanation_ja = _build_explanation_ja(incident_type, top_events, ts)
        explanation_en = _build_explanation_en(incident_type, top_events, ts)

        # LLaVA enhancement (optional)
        if self._llava and top_events:
            try:
                context = "\n".join(
                    f"  [{_rel_time(e.timestamp, ts)}秒前] {e.event_type}: {e.description_ja}"
                    for e in top_events
                )
                prompt = (
                    f"以下のイベント系列が観測された後、「{incident_type}」が発生しました。\n"
                    f"{context}\n"
                    f"因果関係を日本語で簡潔に説明してください（2文以内）。"
                )
                llava_answer = self._llava.answer_question(None, prompt)
                if llava_answer and len(llava_answer) > 10:
                    explanation_ja = llava_answer
            except Exception:
                logger.debug("CausalAnalyzer: LLaVA failed", exc_info=True)

        chain = CausalChain(
            chain_id=str(uuid.uuid4()),
            incident_type=incident_type,
            incident_time=ts,
            camera_id=camera_id or "all",
            chain=top_events,
            explanation_ja=explanation_ja,
            explanation_en=explanation_en,
            time_span_seconds=time_span,
            causal_confidence=round(causal_conf, 3),
        )

        with self._lock:
            self._chains.append(chain)

        if self._persist_path:
            self._save()

        logger.info(
            "CausalAnalyzer: chain for '%s' — %d events, span=%.1fs, conf=%.2f",
            incident_type,
            len(top_events),
            time_span,
            causal_conf,
        )
        return chain

    # ── Query ──────────────────────────────────────────────────────────

    def get_recent_chains(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            chains = list(self._chains)
        chains.sort(key=lambda c: c.incident_time, reverse=True)
        return [c.to_dict() for c in chains[:limit]]

    def get_buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "buffer_events": len(self._buffer),
                "chains_stored": len(self._chains),
                "total_observed": self._observe_count,
                "total_analyzed": self._analyze_count,
                "lookback_seconds": self._lookback,
                "llava_available": self._llava is not None,
                "uptime_seconds": round(time.time() - self._created_at, 1),
            }

    # ── Persistence ────────────────────────────────────────────────────

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {"chains": [c.to_dict() for c in self._chains]}
        try:
            with open(self._persist_path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False)
        except Exception:
            logger.debug("CausalAnalyzer: save failed", exc_info=True)

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            logger.info(
                "CausalAnalyzer: loaded %d chains from %s",
                len(data.get("chains", [])),
                self._persist_path,
            )
        except Exception:
            logger.debug("CausalAnalyzer: load failed", exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _causal_relevance(event_type: str, incident_type: str) -> float:
    """Score how likely event_type caused incident_type (0=unrelated, 1=direct)."""
    et_lower = event_type.lower()
    it_lower = incident_type.lower()

    # Direct match (event is same type as incident, earlier occurrence)
    if et_lower == it_lower:
        return 0.5

    # Rule-based causal matching
    for keyword, caused_by in _CAUSAL_RULES.items():
        if keyword in et_lower:
            for c in caused_by:
                if c.lower() in it_lower:
                    return 0.9
    for keyword, caused_by in _CAUSAL_RULES.items():
        if keyword in it_lower:
            for c in caused_by:
                if c.lower() in et_lower:
                    return 0.8

    return 0.0


def _rel_time(ts: float, ref: float) -> int:
    return max(0, int(ref - ts))


def _build_explanation_ja(
    incident_type: str, events: list[CausalEvent], ts: float
) -> str:
    if not events:
        return f"「{incident_type}」が発生しました。直前の先行イベントは検出されませんでした。"

    parts = []
    for e in events:
        ago = _rel_time(e.timestamp, ts)
        parts.append(f"{ago}秒前に{e.event_type}")

    chain_str = "、".join(parts)
    return (
        f"{chain_str}が観測された後、「{incident_type}」が発生しました。"
        f"これらのイベントが連鎖的に引き起こした可能性があります。"
    )


def _build_explanation_en(
    incident_type: str, events: list[CausalEvent], ts: float
) -> str:
    if not events:
        return f"Incident '{incident_type}' occurred with no prior causal events detected."

    parts = []
    for e in events:
        ago = _rel_time(e.timestamp, ts)
        parts.append(f"{e.event_type} ({ago}s before)")

    return (
        f"Incident '{incident_type}' preceded by: {', '.join(parts)}. "
        f"These events may have formed a causal chain."
    )
