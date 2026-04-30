"""Counterfactual Reasoning Engine — Phase 63.

Closes Pearl's causal hierarchy on the perception side:

    1. Associational  →  Phase 44  CausalAnalyzer
    2. Interventional →  Phase 53  CausalInferenceEngine
    3. Counterfactual →  Phase 63  THIS MODULE

Question this module answers:

    "The agent chose ESCALATE. What would have happened on average if it
     had chosen PLAN, ADAPT, or MONITOR instead, in past situations like
     the current one?"

Approach (observational counterfactual estimate)::

    1. Recall past episodes whose 6-dim situation fingerprint is close to
       the current one (cosine ≥ ``recall_min_similarity``).
    2. Group those episodes by ``decision_action``.
    3. For each action, compute the empirical expected reduction over the
       *verified* subset (episodes whose closed-loop verifier has fired).
    4. Rank actions; report the best alternative and the regret —
       E[best] − E[chosen] — if any.

This is *observational* counterfactual reasoning. It assumes the
recalled episodes are exchangeable with the current situation, which the
fingerprint similarity bound enforces in practice. It does not perform a
do-calculus intervention; for that, see Phase 53. The two layers are
complementary.

Integration::

    AgentLoop.think()
        └─ decide(situation)            →  action, reasoning
        └─ store_decision(situation)    →  episode_id (Phase 62)
        └─ verifier.schedule(...)       →  outcome later (Phase 60)

    PerceptionEngine Stage 6w (post-decision)::
        cf = counterfactual_reasoner.evaluate(decision)
        decision.params["counterfactual"] = cf.to_dict()
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CounterfactualEstimate:
    """Per-action expected outcome from observational data."""

    action: str
    sample_size: int            # past episodes with this action in similar situations
    verified_size: int          # subset with outcome / reduction_ratio data
    expected_reduction: float | None  # mean over verified, or None
    stddev_reduction: float | None    # population stddev over verified, or None
    confidence: float           # 0..1 — function of verified_size + spread
    rank: int                   # 1 = highest expected reduction (None if no data)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Round floats for clean JSON
        for k in ("expected_reduction", "stddev_reduction", "confidence"):
            if d[k] is not None:
                d[k] = round(d[k], 4)
        return d


@dataclass
class CounterfactualResult:
    """Result of evaluating a single agent decision against past data."""

    chosen_action: str
    chosen_estimate: CounterfactualEstimate | None
    alternatives: list[CounterfactualEstimate]   # ordered by rank
    best_alternative: CounterfactualEstimate | None
    is_chosen_best: bool
    regret: float | None        # E[best] − E[chosen]; None if estimate unavailable
    sample_size_total: int      # total similar episodes recalled
    has_data: bool              # any action had ≥ min_episodes_per_action verified samples
    reasoning_ja: str = ""
    reasoning_en: str = ""
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen_action": self.chosen_action,
            "chosen_estimate": (
                self.chosen_estimate.to_dict() if self.chosen_estimate else None
            ),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "best_alternative": (
                self.best_alternative.to_dict() if self.best_alternative else None
            ),
            "is_chosen_best": self.is_chosen_best,
            "regret": round(self.regret, 4) if self.regret is not None else None,
            "sample_size_total": self.sample_size_total,
            "has_data": self.has_data,
            "reasoning_ja": self.reasoning_ja,
            "reasoning_en": self.reasoning_en,
            "evaluated_at": self.evaluated_at,
        }


# ── CounterfactualReasoner ───────────────────────────────────────────────────


class CounterfactualReasoner:
    """Observational counterfactual estimator over past agent episodes.

    Thread-safety: single-threaded (called from pipeline Stage 6w, after
    agent.think() returns). The episode store is also single-threaded.
    """

    def __init__(
        self,
        episode_store: Any,
        *,
        min_episodes_per_action: int = 2,
        recall_top_k: int = 50,
        recall_min_similarity: float = 0.6,
    ) -> None:
        """
        Parameters
        ----------
        episode_store:
            An ``AgentEpisodeStore`` (Phase 62) — duck-typed for
            ``recall(...)``.
        min_episodes_per_action:
            Minimum verified episodes required before an action gets a
            point estimate. Below this, ``expected_reduction`` is ``None``.
        recall_top_k:
            How many similar episodes to fetch per evaluation. Larger →
            steadier estimates but more lag if the store is large.
        recall_min_similarity:
            Cosine threshold for "similar enough." Lower than the default
            episode-store threshold (0.7) on purpose — counterfactual
            estimation needs a wider net than per-decision recall.
        """
        if episode_store is None:
            raise ValueError("CounterfactualReasoner requires an episode_store")
        self._episode_store = episode_store
        self._min_episodes = max(1, int(min_episodes_per_action))
        self._recall_top_k = max(1, int(recall_top_k))
        self._recall_min_similarity = max(0.0, min(1.0, recall_min_similarity))

        self._evaluation_count: int = 0
        self._has_data_count: int = 0
        self._regret_sum: float = 0.0
        self._regret_observations: int = 0
        self._chosen_was_best_count: int = 0
        self._start_time: float = time.time()

    # ── Public API ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        *,
        chosen_action: str,
        risk_level: str,
        severity_counts: dict[str, int],
        camera_violations: dict[str, int],
        dominant_violation_type: str,
    ) -> CounterfactualResult:
        """Estimate per-action expected outcome for the given situation.

        Returns
        -------
        CounterfactualResult
            Always returned, even when no data is available — callers can
            inspect ``has_data`` to decide whether to surface it.
        """
        self._evaluation_count += 1

        # 1. Recall similar past episodes
        similar = self._safe_recall(
            risk_level=risk_level,
            severity_counts=severity_counts,
            camera_violations=camera_violations,
            dominant_violation_type=dominant_violation_type,
        )
        sample_size_total = len(similar)

        # 2. Group by action
        per_action: dict[str, list[dict[str, Any]]] = {}
        for ep in similar:
            action = ep.get("decision_action", "")
            if not action:
                continue
            per_action.setdefault(action, []).append(ep)

        # 3. Build estimates per action
        estimates: list[CounterfactualEstimate] = []
        for action, episodes in per_action.items():
            verified = [
                ep for ep in episodes
                if ep.get("reduction_ratio") is not None
            ]
            verified_size = len(verified)
            sample_size = len(episodes)

            if verified_size >= self._min_episodes:
                reductions = [float(ep["reduction_ratio"]) for ep in verified]
                mean = sum(reductions) / verified_size
                if verified_size > 1:
                    var = sum((r - mean) ** 2 for r in reductions) / verified_size
                    stddev = math.sqrt(var)
                else:
                    stddev = 0.0
                confidence = self._confidence(verified_size, stddev)
                estimates.append(
                    CounterfactualEstimate(
                        action=action,
                        sample_size=sample_size,
                        verified_size=verified_size,
                        expected_reduction=mean,
                        stddev_reduction=stddev,
                        confidence=confidence,
                        rank=0,  # filled below
                    )
                )
            else:
                # Below threshold — record the action exists but no estimate
                estimates.append(
                    CounterfactualEstimate(
                        action=action,
                        sample_size=sample_size,
                        verified_size=verified_size,
                        expected_reduction=None,
                        stddev_reduction=None,
                        confidence=0.0,
                        rank=0,
                    )
                )

        # 4. Rank: estimates with data, sorted by expected_reduction desc
        with_data = [e for e in estimates if e.expected_reduction is not None]
        with_data.sort(key=lambda e: -e.expected_reduction)  # type: ignore[arg-type, operator]
        for i, est in enumerate(with_data):
            # Replace with ranked copy (frozen dataclass)
            new = CounterfactualEstimate(
                action=est.action,
                sample_size=est.sample_size,
                verified_size=est.verified_size,
                expected_reduction=est.expected_reduction,
                stddev_reduction=est.stddev_reduction,
                confidence=est.confidence,
                rank=i + 1,
            )
            with_data[i] = new
        without_data = [e for e in estimates if e.expected_reduction is None]
        all_estimates = with_data + without_data

        # 5. Locate chosen + best
        chosen_estimate = next(
            (e for e in all_estimates if e.action == chosen_action), None
        )
        best_alternative = with_data[0] if with_data else None
        has_data = len(with_data) > 0

        if (
            chosen_estimate is not None
            and chosen_estimate.expected_reduction is not None
            and best_alternative is not None
        ):
            is_chosen_best = chosen_estimate.action == best_alternative.action
            regret = (
                best_alternative.expected_reduction  # type: ignore[operator]
                - chosen_estimate.expected_reduction
            )
            # Aggregate stats
            self._regret_sum += regret
            self._regret_observations += 1
            if is_chosen_best:
                self._chosen_was_best_count += 1
        else:
            is_chosen_best = False
            regret = None

        if has_data:
            self._has_data_count += 1

        # 6. Build narrative
        reasoning_ja, reasoning_en = self._narrate(
            chosen_action=chosen_action,
            chosen=chosen_estimate,
            best=best_alternative,
            is_best=is_chosen_best,
            regret=regret,
            sample_size=sample_size_total,
        )

        return CounterfactualResult(
            chosen_action=chosen_action,
            chosen_estimate=chosen_estimate,
            alternatives=all_estimates,
            best_alternative=best_alternative,
            is_chosen_best=is_chosen_best,
            regret=regret,
            sample_size_total=sample_size_total,
            has_data=has_data,
            reasoning_ja=reasoning_ja,
            reasoning_en=reasoning_en,
        )

    def get_stats(self) -> dict[str, Any]:
        """Aggregate stats across all evaluations performed so far."""
        avg_regret = (
            self._regret_sum / self._regret_observations
            if self._regret_observations > 0
            else None
        )
        chosen_best_rate = (
            self._chosen_was_best_count / self._regret_observations
            if self._regret_observations > 0
            else None
        )
        return {
            "enabled": True,
            "evaluation_count": self._evaluation_count,
            "evaluations_with_data": self._has_data_count,
            "regret_observations": self._regret_observations,
            "average_regret": (
                round(avg_regret, 4) if avg_regret is not None else None
            ),
            "chosen_best_rate": (
                round(chosen_best_rate, 4) if chosen_best_rate is not None else None
            ),
            "min_episodes_per_action": self._min_episodes,
            "recall_top_k": self._recall_top_k,
            "recall_min_similarity": self._recall_min_similarity,
            "uptime_s": round(time.time() - self._start_time, 1),
        }

    def reset(self) -> None:
        self._evaluation_count = 0
        self._has_data_count = 0
        self._regret_sum = 0.0
        self._regret_observations = 0
        self._chosen_was_best_count = 0
        self._start_time = time.time()

    # ── Internals ────────────────────────────────────────────────────────────

    def _safe_recall(
        self,
        *,
        risk_level: str,
        severity_counts: dict[str, int],
        camera_violations: dict[str, int],
        dominant_violation_type: str,
    ) -> list[dict[str, Any]]:
        """Recall similar episodes with a temporarily-relaxed threshold.

        ``AgentEpisodeStore.recall`` enforces its own ``recall_min_similarity``
        on the way out. To widen the net for counterfactual estimation, we
        temporarily lower that threshold and restore it afterwards.
        """
        original_threshold = getattr(
            self._episode_store, "_recall_min_similarity", None
        )
        try:
            if original_threshold is not None:
                self._episode_store._recall_min_similarity = (
                    self._recall_min_similarity
                )
            return self._episode_store.recall(
                risk_level=risk_level,
                severity_counts=severity_counts,
                camera_violations=camera_violations,
                dominant_violation_type=dominant_violation_type,
                top_k=self._recall_top_k,
            )
        except Exception:
            logger.debug("CounterfactualReasoner recall failed", exc_info=True)
            return []
        finally:
            if original_threshold is not None:
                self._episode_store._recall_min_similarity = original_threshold

    @staticmethod
    def _confidence(verified_size: int, stddev: float) -> float:
        """Heuristic 0..1 confidence given sample size and outcome spread.

        Uses ``1 − 1/√n`` as a sample-size factor and a spread factor that
        decays linearly with stddev. Both bounded to [0, 1] and combined
        multiplicatively.
        """
        if verified_size <= 0:
            return 0.0
        size_factor = max(0.0, 1.0 - 1.0 / math.sqrt(verified_size))
        # Reduction ratios cluster in [-1, 1]; treat stddev > 0.5 as noisy
        spread_factor = max(0.0, 1.0 - min(stddev, 1.0))
        return round(size_factor * spread_factor, 4)

    @staticmethod
    def _narrate(
        *,
        chosen_action: str,
        chosen: CounterfactualEstimate | None,
        best: CounterfactualEstimate | None,
        is_best: bool,
        regret: float | None,
        sample_size: int,
    ) -> tuple[str, str]:
        """Return (jp, en) narratives describing the counterfactual finding."""
        if best is None or chosen is None or chosen.expected_reduction is None:
            ja = (
                f"類似の過去エピソードが不足 (サンプル数={sample_size}件)。"
                f"反実仮想評価を保留。"
            )
            en = (
                f"Insufficient similar past episodes (n={sample_size}); "
                f"counterfactual estimate unavailable."
            )
            return ja, en

        chosen_pct = chosen.expected_reduction * 100
        best_pct = best.expected_reduction * 100  # type: ignore[operator]

        if is_best:
            ja = (
                f"選択 {chosen_action.upper()} は過去類似 {chosen.verified_size}件で"
                f"最良 (期待削減 {chosen_pct:+.0f}%)。妥当な選択。"
            )
            en = (
                f"Chosen {chosen_action.upper()} matches the best historical "
                f"action (E[reduction]={chosen_pct:+.0f}% over n={chosen.verified_size}). "
                f"Optimal choice."
            )
        else:
            regret_pct = (regret or 0.0) * 100
            ja = (
                f"選択 {chosen_action.upper()} (期待削減 {chosen_pct:+.0f}%) "
                f"より {best.action.upper()} が優位 ({best_pct:+.0f}%)。"
                f"後悔値 {regret_pct:+.0f}%。"
            )
            en = (
                f"{best.action.upper()} historically performs better "
                f"(E[reduction]={best_pct:+.0f}% vs chosen "
                f"{chosen_action.upper()}={chosen_pct:+.0f}%); "
                f"regret={regret_pct:+.0f}%."
            )
        return ja, en
