"""Phase 18: Perception Engine Health Score.

PerceptionHealthScorer は Phase 12–16 のすべてのシグナルを集約し、
カメラ知覚システムの総合的な健康スコア [0, 100] とグレード (A–F) を計算する。

スコア計算式:
    score = max(0, 100 - total_penalty)

ペナルティ (最大合計 100 点):
    early_warning  : 0–30 pts  (EarlyWarningEngine overall_risk × 30)
    fp_rate        : 0–25 pts  (最大FP率 / MAX_FP_RATE × 25)
    sigma_drift    : 0–20 pts  (最大σドリフト / MAX_DRIFT_RATE × 20)
    anomaly_burst  : 0–15 pts  (最大バースト / MAX_BURST_RATE × 15)
    review_backlog : 0–10 pts  (pending件数 / BACKLOG_CAP × 10)

グレード:
    A : 90–100
    B : 75–89
    C : 60–74
    D : 40–59
    F :  0–39
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Weight constants ──────────────────────────────────────────────────────────

W_EARLY_WARNING: float = 30.0
W_FP_RATE: float = 25.0
W_SIGMA_DRIFT: float = 20.0
W_ANOMALY_BURST: float = 15.0
W_REVIEW_BACKLOG: float = 10.0

# ── Normalization caps ────────────────────────────────────────────────────────

MAX_FP_RATE: float = 0.70        # fraction → norm=1.0
MAX_DRIFT_RATE: float = 0.50     # σ/min  → norm=1.0
MAX_BURST_RATE: float = 5.0      # anomalies/min → norm=1.0
BACKLOG_CAP: int = 50            # pending reviews → norm=1.0


# ── Grade mapping ─────────────────────────────────────────────────────────────

def _grade(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


# ── PerceptionHealthScorer ────────────────────────────────────────────────────


class PerceptionHealthScorer:
    """Aggregate Phase 12–16 signals into a single health score (Phase 18).

    Usage:
        scorer = PerceptionHealthScorer()
        result = scorer.compute(engine)
        # {"score": 87, "grade": "B", "factors": {...}, "computed_at": ...}
    """

    def compute(self, engine: Any) -> dict[str, Any]:
        """Compute health score from a PerceptionEngine instance.

        Args:
            engine: PerceptionEngine (or any object with optional perception attrs).

        Returns:
            dict with keys: score, grade, factors, computed_at.
        """
        factors: dict[str, Any] = {}
        penalty_total: float = 0.0

        # ── Factor 1: EarlyWarning overall risk ──────────────────────────────
        ew_risk = 0.0
        try:
            if getattr(engine, "_early_warning", None) is not None:
                ew_state = engine._early_warning.get_state()
                ew_risk = float(ew_state.get("overall_risk", 0.0))
        except Exception:
            logger.debug("Health: early_warning query failed", exc_info=True)
        ew_penalty = W_EARLY_WARNING * min(ew_risk, 1.0)
        penalty_total += ew_penalty
        factors["early_warning"] = {
            "penalty": round(ew_penalty, 2),
            "overall_risk": round(ew_risk, 4),
            "description": "予兆検知リスクスコア (EarlyWarningEngine)",
        }

        # ── Factor 2: FP rate ─────────────────────────────────────────────────
        max_fp = 0.0
        try:
            tuner = getattr(engine, "_anomaly_tuner", None)
            if tuner is not None:
                stats = tuner.get_stats()
                for ps in stats.get("pair_stats", []):
                    total = int(ps.get("total", 0))
                    denied = int(ps.get("denied", 0))
                    if total > 0:
                        max_fp = max(max_fp, denied / total)
        except Exception:
            logger.debug("Health: fp_rate query failed", exc_info=True)
        fp_penalty = W_FP_RATE * min(max_fp / MAX_FP_RATE, 1.0)
        penalty_total += fp_penalty
        factors["fp_rate"] = {
            "penalty": round(fp_penalty, 2),
            "max_fp_rate": round(max_fp, 4),
            "description": "最大誤検知率 (AnomalyTuner pair_stats)",
        }

        # ── Factor 3: Sigma drift velocity ────────────────────────────────────
        max_drift = 0.0
        try:
            if getattr(engine, "_early_warning", None) is not None:
                ew = engine._early_warning
                with ew._lock:
                    velocities = dict(ew._drift_velocity)
                if velocities:
                    max_drift = max(velocities.values())
        except Exception:
            logger.debug("Health: sigma_drift query failed", exc_info=True)
        drift_penalty = W_SIGMA_DRIFT * min(max_drift / MAX_DRIFT_RATE, 1.0)
        penalty_total += drift_penalty
        factors["sigma_drift"] = {
            "penalty": round(drift_penalty, 2),
            "max_drift_velocity": round(max_drift, 4),
            "description": "最大σドリフト速度 (EarlyWarningEngine)",
        }

        # ── Factor 4: Anomaly burst rate ──────────────────────────────────────
        max_burst = 0.0
        try:
            if getattr(engine, "_early_warning", None) is not None:
                ew = engine._early_warning
                now = time.time()
                cutoff = now - ew._burst_window_seconds
                with ew._lock:
                    ts_map = dict(ew._anomaly_timestamps)
                for buf in ts_map.values():
                    recent = [t for t in buf if t >= cutoff]
                    rate = len(recent) / (ew._burst_window_seconds / 60.0)
                    max_burst = max(max_burst, rate)
        except Exception:
            logger.debug("Health: anomaly_burst query failed", exc_info=True)
        burst_penalty = W_ANOMALY_BURST * min(max_burst / MAX_BURST_RATE, 1.0)
        penalty_total += burst_penalty
        factors["anomaly_burst"] = {
            "penalty": round(burst_penalty, 2),
            "max_burst_rate": round(max_burst, 4),
            "description": "最大異常バースト頻度 (anomalies/min)",
        }

        # ── Factor 5: Review queue backlog ────────────────────────────────────
        pending_count = 0
        try:
            rq = getattr(engine, "_review_queue", None)
            if rq is not None:
                rq_stats = rq.get_stats()
                pending_count = int(rq_stats.get("pending_count", 0))
        except Exception:
            logger.debug("Health: review_queue query failed", exc_info=True)
        backlog_penalty = W_REVIEW_BACKLOG * min(pending_count / BACKLOG_CAP, 1.0)
        penalty_total += backlog_penalty
        factors["review_backlog"] = {
            "penalty": round(backlog_penalty, 2),
            "pending_count": pending_count,
            "description": "レビュー待ち件数 (ReviewQueue)",
        }

        # ── Final score ───────────────────────────────────────────────────────
        score = max(0, min(100, round(100.0 - penalty_total)))

        return {
            "score": score,
            "grade": _grade(score),
            "factors": factors,
            "total_penalty": round(penalty_total, 2),
            "computed_at": time.time(),
        }
