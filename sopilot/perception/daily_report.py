"""Phase 20: 日次インテリジェントレポート (Daily Intelligence Report).

DailyReportGenerator は過去 24 時間の知覚パイプラインの活動を集約し、
オペレータ向けの構造化レポートを生成する。

レポート構成:
    1. summary      — 健康スコア推移、総合評価
    2. anomalies    — 検知数、検出器別内訳、FP率
    3. early_warning — 予兆検知イベント数、リスクピーク
    4. responses    — 自律対応回数、検出器別詳細
    5. recommendations — 上位 3 件の推奨アクション (Phase 16 由来)
    6. metadata     — 生成日時、対象期間

設計:
    - VLM 不要: ルールベース日本語テンプレートで説明生成
    - 全フィールドはオプション (コンポーネントが存在しなければ null)
    - best-effort: コンポーネント例外は握り潰し、フィールドを null で埋める
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_DAY_SECONDS: float = 86400.0


def _overall_assessment(health_score: int | None, ew_risk: float | None,
                         response_count: int) -> str:
    """日本語で総合評価コメントを生成する。"""
    if health_score is None:
        return "評価データが不足しています。"
    if health_score >= 90 and (ew_risk or 0.0) < 0.3 and response_count == 0:
        return "システムは安定して稼働しており、過去24時間に異常な兆候は検出されませんでした。"
    if health_score >= 75:
        comment = "システムは概ね正常に稼働しています。"
        if response_count > 0:
            comment += f" {response_count} 件の自律対応が実行されました。"
        return comment
    if health_score >= 60:
        return (f"健康スコア {health_score} (グレードC) — 一部の検出器でリスクが上昇しています。"
                f" 推奨アクションを確認してください。")
    return (f"健康スコア {health_score} — システムの状態が悪化しています。"
            f" 早急な対応が必要です。")


class DailyReportGenerator:
    """Generate a 24-hour intelligence report from a PerceptionEngine (Phase 20).

    Usage:
        gen = DailyReportGenerator()
        report = gen.generate(engine)
    """

    def generate(self, engine: Any) -> dict[str, Any]:
        """Generate the daily report for the last 24 hours.

        Args:
            engine: PerceptionEngine instance.

        Returns:
            Structured report dict with sections: summary, anomalies,
            early_warning, responses, recommendations, metadata.
        """
        now = time.time()
        window_start = now - _DAY_SECONDS

        report: dict[str, Any] = {
            "summary": self._build_summary(engine, window_start, now),
            "anomalies": self._build_anomalies(engine, window_start, now),
            "early_warning": self._build_early_warning(engine),
            "responses": self._build_responses(engine),
            "recommendations": self._build_recommendations(engine),
            "metadata": {
                "generated_at": now,
                "window_start": window_start,
                "window_end": now,
                "window_hours": 24,
                "generator": "DailyReportGenerator (Phase 20)",
            },
        }
        return report

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_summary(
        self, engine: Any, window_start: float, now: float
    ) -> dict[str, Any]:
        section: dict[str, Any] = {
            "health_score": None,
            "health_grade": None,
            "health_trend": None,
            "overall_assessment": None,
        }
        try:
            hs = engine.get_health_score(record=False)
            section["health_score"] = hs.get("score")
            section["health_grade"] = hs.get("grade")
        except Exception:
            logger.debug("summary: health_score failed", exc_info=True)

        try:
            hist = engine.get_health_history(days=1.0)
            section["health_trend"] = hist.get("trend") or {}
        except Exception:
            logger.debug("summary: health_history failed", exc_info=True)

        # Overall EW risk
        ew_risk: float | None = None
        try:
            if getattr(engine, "_early_warning", None) is not None:
                ew_state = engine._early_warning.get_state()
                ew_risk = float(ew_state.get("overall_risk", 0.0))
        except Exception:
            pass

        # Response count
        response_count = 0
        try:
            rs = engine.get_early_warning_responder_state()
            if rs:
                response_count = int(rs.get("total_responses", 0))
        except Exception:
            pass

        section["overall_assessment"] = _overall_assessment(
            section["health_score"], ew_risk, response_count
        )
        return section

    def _build_anomalies(
        self, engine: Any, window_start: float, now: float
    ) -> dict[str, Any]:
        section: dict[str, Any] = {
            "total_detected": None,
            "fp_rate_by_detector": None,
            "high_z_count": None,
        }
        try:
            tuner = getattr(engine, "_anomaly_tuner", None)
            if tuner is not None:
                stats = tuner.get_stats()
                fp_map: dict[str, float] = {}
                for ps in stats.get("pair_stats", []):
                    det = ps.get("detector", "")
                    total = int(ps.get("total", 0))
                    denied = int(ps.get("denied", 0))
                    if total > 0:
                        fp_map[det] = round(denied / total, 4)
                section["fp_rate_by_detector"] = fp_map
        except Exception:
            logger.debug("anomalies: tuner failed", exc_info=True)

        try:
            rq = getattr(engine, "_review_queue", None)
            if rq is not None:
                rq_stats = rq.get_stats()
                section["total_detected"] = int(rq_stats.get("total_added", 0))
                section["high_z_count"] = int(rq_stats.get("total_added", 0))
        except Exception:
            logger.debug("anomalies: review_queue failed", exc_info=True)

        return section

    def _build_early_warning(self, engine: Any) -> dict[str, Any]:
        section: dict[str, Any] = {
            "overall_risk": None,
            "risk_level": None,
            "peak_detector": None,
            "peak_risk": None,
            "detector_risks": None,
        }
        try:
            ew = getattr(engine, "_early_warning", None)
            if ew is None:
                return section
            tuner_stats = None
            try:
                t = getattr(engine, "_anomaly_tuner", None)
                if t:
                    tuner_stats = t.get_stats()
            except Exception:
                pass
            ew_state = ew.get_state(tuner_stats)
            section["overall_risk"] = round(ew_state.get("overall_risk", 0.0), 4)
            section["risk_level"] = ew_state.get("overall_level", "low")
            dets = ew_state.get("detectors", {})
            section["detector_risks"] = {
                d: round(info.get("risk_score", 0.0), 4)
                for d, info in dets.items()
            }
            if dets:
                peak = max(dets.items(), key=lambda x: x[1].get("risk_score", 0.0))
                section["peak_detector"] = peak[0]
                section["peak_risk"] = round(peak[1].get("risk_score", 0.0), 4)
        except Exception:
            logger.debug("early_warning section failed", exc_info=True)
        return section

    def _build_responses(self, engine: Any) -> dict[str, Any]:
        section: dict[str, Any] = {
            "total_responses": None,
            "recent_responses": None,
            "cooldowns_active": None,
        }
        try:
            rs = engine.get_early_warning_responder_state()
            if rs is None:
                return section
            section["total_responses"] = rs.get("total_responses", 0)
            section["cooldowns_active"] = len(rs.get("cooldowns_remaining", {}))
            # Keep last 5 for report
            section["recent_responses"] = (rs.get("recent_responses") or [])[:5]
        except Exception:
            logger.debug("responses section failed", exc_info=True)
        return section

    def _build_recommendations(self, engine: Any) -> list[dict[str, str]]:
        """Collect top-3 distinct recommendations from recent Phase 16 responses."""
        recs: list[dict[str, str]] = []
        seen: set[str] = set()
        try:
            rs = engine.get_early_warning_responder_state()
            if rs is None:
                return recs
            for resp in (rs.get("recent_responses") or []):
                det = resp.get("detector", "")
                for r in resp.get("recommendations", []):
                    if r not in seen:
                        seen.add(r)
                        recs.append({"detector": det, "action": r})
                    if len(recs) >= 3:
                        return recs
        except Exception:
            logger.debug("recommendations section failed", exc_info=True)

        # Always include health-score check if list is short
        action = "GET /vigil/perception/health-score — システム健康スコアを確認"
        if len(recs) < 3 and action not in seen:
            recs.append({"detector": "system", "action": action})

        return recs[:3]
