"""Agent Episodic Memory — Phase 62.

Stores OODA decision episodes (situation → action → outcome) and enables
similarity-based recall during the ORIENT phase of the AgentLoop.

Each episode captures:
  - situation fingerprint (6-dim feature vector)
  - decision action + reasoning (JP/EN)
  - verification outcome (filled in asynchronously by ClosedLoopVerifier)

Recall uses cosine similarity between the current situation fingerprint and
all stored episode fingerprints.  No external libraries required.

Integration:
  AgentLoop.think()
      └─ _orient():  episode_store.recall(situation) → recalled_episodes
      └─ LEARN:      episode_store.store_decision()  → episode_id

  ClosedLoopVerifier.check_pending()
      └─ episode_store.update_outcome(episode_id, outcome)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_MAX_EPISODES = 1000
_PERSIST_PATH = os.path.join(os.path.dirname(__file__), "agent_episodes.json")

# ── Feature extraction ────────────────────────────────────────────────────

_RISK_SCORES: dict[str, float] = {
    "low": 0.0,
    "medium": 0.33,
    "high": 0.67,
    "critical": 1.0,
}


def _violation_type_hash(vtype: str) -> float:
    """Hash a violation-type string to [0, 1] (reproducible, no crypto)."""
    if not vtype:
        return 0.0
    digest = int(
        hashlib.md5(vtype.encode(), usedforsecurity=False).hexdigest()[:4], 16
    )
    return digest / 65535.0


def _extract_fingerprint(
    risk_level: str,
    severity_counts: dict[str, int],
    camera_violations: dict[str, int],
    dominant_violation_type: str,
) -> list[float]:
    """Return a 6-dim feature vector from situation data.

    Dimensions:
      [0] risk_score          — 0.0(low)→1.0(critical)
      [1] critical_ratio      — critical / total_severity
      [2] warning_ratio       — warning / total_severity
      [3] camera_count_norm   — camera count / 10 (capped at 1)
      [4] violations_norm     — total violations / 20 (capped at 1)
      [5] violation_type_hash — hash of dominant type → [0,1]
    """
    total_violations = max(sum(camera_violations.values()), 0)
    total_severity = max(sum(severity_counts.values()), 1)

    risk_score = _RISK_SCORES.get(risk_level, 0.0)
    crit_ratio = severity_counts.get("critical", 0) / total_severity
    warn_ratio = severity_counts.get("warning", 0) / total_severity
    cam_count = min(len(camera_violations) / 10.0, 1.0)
    viol_norm = min(total_violations / 20.0, 1.0)
    vtype_hash = _violation_type_hash(dominant_violation_type)

    return [risk_score, crit_ratio, warn_ratio, cam_count, viol_norm, vtype_hash]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors (no numpy)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Data structures ───────────────────────────────────────────────────────


@dataclass
class AgentEpisode:
    """A stored OODA decision episode."""
    episode_id: str
    fingerprint: list[float]
    decision_action: str
    risk_level: str
    dominant_violation_type: str
    reasoning_ja: str
    reasoning_en: str
    camera_id: str = ""
    outcome: str | None = None          # None until verified
    reduction_ratio: float | None = None
    created_at: float = field(default_factory=time.time)
    outcome_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── AgentEpisodeStore ─────────────────────────────────────────────────────

_PERSIST_INTERVAL = 50  # save every N store/update operations


class AgentEpisodeStore:
    """Rolling episodic memory for OODA agent decisions.

    Thread-safety: single-threaded (called from pipeline Stage 6v).
    All sub-system references are optional; no circular imports.

    Usage::

        store = AgentEpisodeStore()

        # After AgentLoop.think():
        ep_id = store.store_decision(
            risk_level="high",
            severity_counts={"critical": 1, "warning": 2, "info": 0},
            camera_violations={"cam1": 3},
            dominant_violation_type="PPE違反",
            decision_action="escalate",
            reasoning_ja="重大違反検知",
            reasoning_en="Critical violation detected",
            camera_id="cam1",
        )

        # During _orient():
        similar = store.recall(
            risk_level="high",
            severity_counts={"critical": 1, "warning": 1, "info": 0},
            camera_violations={"cam1": 2},
            dominant_violation_type="PPE違反",
            top_k=3,
        )

        # After ClosedLoopVerifier.check_pending():
        store.update_outcome(ep_id, "success", reduction_ratio=0.75)
    """

    def __init__(
        self,
        max_episodes: int = _MAX_EPISODES,
        persist_path: str | None = None,
        recall_min_similarity: float = 0.7,
    ) -> None:
        self._max_episodes = max_episodes
        self._persist_path = persist_path if persist_path is not None else _PERSIST_PATH
        self._recall_min_similarity = recall_min_similarity

        self._episodes: deque[AgentEpisode] = deque(maxlen=max_episodes)
        self._index: dict[str, AgentEpisode] = {}  # episode_id → episode
        self._store_count: int = 0
        self._recall_count: int = 0
        self._outcome_update_count: int = 0
        self._start_time: float = time.time()

        self._load_persisted()

    # ── Store ─────────────────────────────────────────────────────────────

    def store_decision(
        self,
        *,
        risk_level: str,
        severity_counts: dict[str, int],
        camera_violations: dict[str, int],
        dominant_violation_type: str,
        decision_action: str,
        reasoning_ja: str = "",
        reasoning_en: str = "",
        camera_id: str = "",
    ) -> str:
        """Store a new decision episode.  Returns episode_id."""
        ep_id = str(uuid.uuid4())[:8]
        fp = _extract_fingerprint(
            risk_level, severity_counts, camera_violations, dominant_violation_type
        )

        # If deque is full, the oldest item will be evicted — remove from index
        if len(self._episodes) == self._max_episodes and self._episodes:
            oldest = self._episodes[0]
            self._index.pop(oldest.episode_id, None)

        ep = AgentEpisode(
            episode_id=ep_id,
            fingerprint=fp,
            decision_action=decision_action,
            risk_level=risk_level,
            dominant_violation_type=dominant_violation_type,
            reasoning_ja=reasoning_ja,
            reasoning_en=reasoning_en,
            camera_id=camera_id,
            created_at=time.time(),
        )
        self._episodes.append(ep)
        self._index[ep_id] = ep
        self._store_count += 1

        self._persist_if_needed()
        logger.debug(
            "Episode stored: id=%s action=%s risk=%s", ep_id, decision_action, risk_level
        )
        return ep_id

    # ── Update outcome ────────────────────────────────────────────────────

    def update_outcome(
        self,
        episode_id: str,
        outcome: str,
        reduction_ratio: float = 0.0,
    ) -> bool:
        """Update the verification outcome for an episode.  Returns True if found."""
        ep = self._index.get(episode_id)
        if ep is None:
            return False
        ep.outcome = outcome
        ep.reduction_ratio = round(reduction_ratio, 3)
        ep.outcome_at = time.time()
        self._outcome_update_count += 1
        self._persist_if_needed()
        return True

    # ── Recall ────────────────────────────────────────────────────────────

    def recall(
        self,
        *,
        risk_level: str,
        severity_counts: dict[str, int],
        camera_violations: dict[str, int],
        dominant_violation_type: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return top-k most similar past episodes (descending similarity).

        Only episodes with cosine similarity ≥ recall_min_similarity are returned.
        Each result dict includes a ``similarity`` key.
        """
        self._recall_count += 1
        query_fp = _extract_fingerprint(
            risk_level, severity_counts, camera_violations, dominant_violation_type
        )

        scored: list[tuple[float, AgentEpisode]] = []
        for ep in self._episodes:
            sim = _cosine_sim(query_fp, ep.fingerprint)
            if sim >= self._recall_min_similarity:
                scored.append((sim, ep))

        scored.sort(key=lambda x: -x[0])

        results = []
        for sim, ep in scored[:top_k]:
            d = ep.to_dict()
            d["similarity"] = round(sim, 3)
            results.append(d)
        return results

    # ── Search helpers ────────────────────────────────────────────────────

    def search_by_action(self, action: str, limit: int = 10) -> list[dict[str, Any]]:
        """Return episodes matching the given action (most recent first)."""
        matched = [ep for ep in self._episodes if ep.decision_action == action]
        recent = matched[-limit:] if limit < len(matched) else matched
        return [ep.to_dict() for ep in reversed(recent)]

    # ── Persistence ───────────────────────────────────────────────────────

    def _persist_if_needed(self) -> None:
        """Best-effort JSON save every _PERSIST_INTERVAL operations."""
        total = self._store_count + self._outcome_update_count
        if total % _PERSIST_INTERVAL != 0:
            return
        try:
            data = [ep.to_dict() for ep in self._episodes]
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.debug("Episode persist failed", exc_info=True)

    def _load_persisted(self) -> None:
        """Best-effort load of persisted episodes on startup."""
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, encoding="utf-8") as f:
                data = json.load(f)
            fields = AgentEpisode.__dataclass_fields__.keys()
            for d in data:
                ep = AgentEpisode(**{k: v for k, v in d.items() if k in fields})
                self._episodes.append(ep)
                self._index[ep.episode_id] = ep
            logger.info("Loaded %d persisted episodes", len(self._episodes))
        except Exception:
            logger.debug("Episode load failed", exc_info=True)

    # ── Public API ────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return store statistics."""
        total = len(self._episodes)
        with_outcome = sum(1 for ep in self._episodes if ep.outcome is not None)

        action_dist: dict[str, int] = {}
        outcome_dist: dict[str, int] = {}
        for ep in self._episodes:
            action_dist[ep.decision_action] = action_dist.get(ep.decision_action, 0) + 1
            if ep.outcome:
                outcome_dist[ep.outcome] = outcome_dist.get(ep.outcome, 0) + 1

        return {
            "enabled": True,
            "total_episodes": total,
            "with_outcome": with_outcome,
            "store_count": self._store_count,
            "recall_count": self._recall_count,
            "outcome_update_count": self._outcome_update_count,
            "action_distribution": action_dist,
            "outcome_distribution": outcome_dist,
            "recall_min_similarity": self._recall_min_similarity,
            "max_episodes": self._max_episodes,
            "uptime_s": round(time.time() - self._start_time, 1),
        }

    def get_recent_episodes(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return most recent episodes (newest first)."""
        episodes = list(self._episodes)
        recent = episodes[-limit:] if limit < len(episodes) else episodes
        return [ep.to_dict() for ep in reversed(recent)]

    def reset(self) -> None:
        """Clear all episodes and counters."""
        self._episodes.clear()
        self._index.clear()
        self._store_count = 0
        self._recall_count = 0
        self._outcome_update_count = 0
        self._start_time = time.time()
