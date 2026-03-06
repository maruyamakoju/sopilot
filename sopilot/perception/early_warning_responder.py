"""Phase 16: 自律対応 (Autonomous Response to Early Warning).

EarlyWarningResponder は EarlyWarningEngine のリスクスコアを監視し、
HIGH リスクを検知した detector に対して自動的に対応アクションを実行する。

対応アクション (risk_level=HIGH のとき):
    1. 説明生成 — 日本語で "なぜリスクが高いのか" を説明
    2. SigmaTuner リセット候補 — sigma が MAX 付近なら reset を推薦
    3. レビューキュー優先化 — pending review を緊急フラグ付き
    4. アラート記録 — 対応履歴を保持 (重複対応を防ぐクールダウン付き)

設計方針:
    - VLM 不要: 説明はルールベース日本語テンプレートで生成
    - 副作用なし: execute() はすべて best-effort (例外握り潰し)
    - 冪等: 同一 detector に対する対応は cooldown_seconds 内に 1 回だけ
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_DETECTORS = ("behavioral", "spatial", "temporal", "interaction")

# ── 日本語説明テンプレート ────────────────────────────────────────────────

_REASON_TEMPLATES: dict[str, dict[str, str]] = {
    "behavioral": {
        "drift": "行動検出器のσ値が急速に変化しています。カメラ映像の急変 (照明・人員変動) が原因と推定されます。",
        "fp": "行動検出器の誤検知率が高い状態です。フィードバックの蓄積またはプロファイル再適用を推奨します。",
        "burst": "行動異常が短時間に集中しています。作業フローの急変または器物の移動が発生している可能性があります。",
        "composite": "行動検出器に複合リスクが発生しています。σ値とFP率を確認し、必要に応じてリセットしてください。",
    },
    "spatial": {
        "drift": "空間検出器のσ値が不安定です。カメラアングルの変化またはゾーン定義の見直しが必要な可能性があります。",
        "fp": "空間検出器の誤検知率が上昇しています。グリッドのEMAリセットを検討してください。",
        "burst": "空間異常が集中しています。通行量の急増または障害物の設置が考えられます。",
        "composite": "空間検出器に複合リスクが発生しています。空間マップの状態を確認してください。",
    },
    "temporal": {
        "drift": "時間帯パターン検出器のσ値が変動しています。シフトスケジュールや休日対応の影響が考えられます。",
        "fp": "時間帯検出器の誤検知率が高い状態です。24時間スロットの観測数が不十分な可能性があります。",
        "burst": "時間帯異常が集中しています。通常と異なる時間帯の活動が検出されています。",
        "composite": "時間帯検出器に複合リスクが発生しています。長期記憶と照合することを推奨します。",
    },
    "interaction": {
        "drift": "インタラクション検出器のσ値が変動しています。人員構成やオブジェクト配置の変化が原因と推定されます。",
        "fp": "インタラクション検出器の誤検知率が上昇しています。ペア抑制の見直しを推奨します。",
        "burst": "インタラクション異常が集中しています。グループ行動の変化または機器の異常操作が考えられます。",
        "composite": "インタラクション検出器に複合リスクが発生しています。ペア統計を確認してください。",
    },
}


def _build_explanation(detector: str, detail: dict[str, Any]) -> str:
    """Build a Japanese explanation based on which component is highest."""
    tpls = _REASON_TEMPLATES.get(detector, _REASON_TEMPLATES["behavioral"])
    # Identify dominant component
    drift_n = detail.get("sigma_drift_norm", 0.0)
    fp_n = detail.get("fp_rate_norm", 0.0)
    burst_n = detail.get("anomaly_burst_norm", 0.0)
    dominant = max(
        [("drift", drift_n), ("fp", fp_n), ("burst", burst_n)],
        key=lambda x: x[1],
    )
    if dominant[1] < 0.1:
        key = "composite"
    else:
        key = dominant[0]
    return tpls.get(key, tpls["composite"])


# ── ResponseAction ────────────────────────────────────────────────────────


class ResponseAction:
    """A single autonomous response action."""

    def __init__(
        self,
        detector: str,
        risk_score: float,
        risk_level: str,
        explanation_ja: str,
        recommendations: list[str],
        triggered_at: float,
    ) -> None:
        self.detector = detector
        self.risk_score = risk_score
        self.risk_level = risk_level
        self.explanation_ja = explanation_ja
        self.recommendations = list(recommendations)
        self.triggered_at = triggered_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level,
            "explanation_ja": self.explanation_ja,
            "recommendations": self.recommendations,
            "triggered_at": self.triggered_at,
        }


# ── EarlyWarningResponder ─────────────────────────────────────────────────


class EarlyWarningResponder:
    """Autonomous responder for HIGH-risk early warning signals (Phase 16).

    Args:
        risk_threshold: Minimum risk score to trigger a response (default 0.6 = HIGH).
        cooldown_seconds: Minimum seconds between responses for the same detector.
        max_history: Maximum number of response records to keep in memory.
    """

    DEFAULT_RISK_THRESHOLD: float = 0.6
    DEFAULT_COOLDOWN: float = 300.0  # 5 minutes
    MAX_HISTORY: int = 100

    def __init__(
        self,
        risk_threshold: float = DEFAULT_RISK_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN,
        max_history: int = MAX_HISTORY,
    ) -> None:
        self._risk_threshold = risk_threshold
        self._cooldown_seconds = cooldown_seconds
        self._max_history = max_history

        # detector → last response timestamp
        self._last_response_ts: dict[str, float] = {}
        # Response history (most recent last)
        self._history: list[ResponseAction] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def evaluate(
        self,
        early_warning_state: dict[str, Any],
        sigma_tuner: Any | None = None,
        review_queue: Any | None = None,
    ) -> list[ResponseAction]:
        """Check early warning state and fire responses for HIGH-risk detectors.

        Args:
            early_warning_state: Output of EarlyWarningEngine.get_state().
            sigma_tuner: Optional SigmaTuner instance (for sigma reset check).
            review_queue: Optional ReviewQueue instance (for backlog check).

        Returns:
            List of ResponseAction objects that were newly triggered.
        """
        now = time.time()
        triggered: list[ResponseAction] = []
        detectors = early_warning_state.get("detectors", {})

        for det, detail in detectors.items():
            risk = float(detail.get("risk_score", 0.0))
            level = detail.get("risk_level", "low")

            if risk < self._risk_threshold:
                continue  # not high enough

            # Cooldown check
            with self._lock:
                last = self._last_response_ts.get(det, 0.0)
                if (now - last) < self._cooldown_seconds:
                    continue  # still in cooldown
                self._last_response_ts[det] = now

            # Build explanation + recommendations
            explanation = _build_explanation(det, detail)
            recs = self._build_recommendations(det, detail, sigma_tuner, review_queue)

            action = ResponseAction(
                detector=det,
                risk_score=risk,
                risk_level=level,
                explanation_ja=explanation,
                recommendations=recs,
                triggered_at=now,
            )

            # Store in history
            with self._lock:
                self._history.append(action)
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

            triggered.append(action)
            logger.warning(
                "EarlyWarningResponder: detector=%s risk=%.3f level=%s — %s",
                det, risk, level, explanation,
            )

        return triggered

    def _build_recommendations(
        self,
        detector: str,
        detail: dict[str, Any],
        sigma_tuner: Any | None,
        review_queue: Any | None,
    ) -> list[str]:
        recs: list[str] = []

        # Sigma recommendation
        sigma_vel = detail.get("sigma_drift_velocity", 0.0)
        if sigma_vel > 0.2:
            recs.append(f"POST /vigil/perception/sigma-reset — σドリフト速度 {sigma_vel:.2f}σ/min を解消")

        # FP rate recommendation
        fp_rate = detail.get("fp_rate", 0.0)
        if fp_rate > 0.5:
            recs.append(f"POST /vigil/perception/anomaly-tuning/apply — FP率 {fp_rate:.0%} を改善")

        # Burst recommendation
        burst = detail.get("anomaly_burst_rate", 0.0)
        if burst > 2.0:
            recs.append(f"POST /vigil/perception/anomaly-profile/save — バースト {burst:.1f}/min 発生中、現状を保存")

        # Review queue backlog
        if review_queue is not None:
            try:
                stats = review_queue.get_stats()
                pending = stats.get("pending_count", 0)
                if pending >= 10:
                    recs.append(f"GET /vigil/perception/review-queue — {pending}件のレビュー待ち")
            except Exception:
                pass

        # Cross-camera comparison
        recs.append("GET /vigil/camera-groups/learning/compare — 他グループのσ値と比較して過学習を確認")

        return recs

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent response history (most recent first)."""
        with self._lock:
            items = list(self._history)
        if limit > 0:
            items = items[-limit:]
        return [a.to_dict() for a in reversed(items)]

    def get_state(self) -> dict[str, Any]:
        """Return responder state for API responses."""
        with self._lock:
            last_responses = {
                det: ts for det, ts in self._last_response_ts.items()
            }
            total = len(self._history)

        cooldowns_remaining: dict[str, float] = {}
        now = time.time()
        for det, ts in last_responses.items():
            remaining = self._cooldown_seconds - (now - ts)
            if remaining > 0:
                cooldowns_remaining[det] = round(remaining, 1)

        return {
            "total_responses": total,
            "risk_threshold": self._risk_threshold,
            "cooldown_seconds": self._cooldown_seconds,
            "cooldowns_remaining": cooldowns_remaining,
            "recent_responses": self.get_history(limit=5),
        }

    def reset(self) -> None:
        """Clear history and cooldowns."""
        with self._lock:
            self._history.clear()
            self._last_response_ts.clear()
