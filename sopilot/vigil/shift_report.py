"""Phase 12B: 自動シフトレポート (Automated Shift Report).

ShiftReportGenerator は vigil セッション + 学習状態から
構造化レポートを生成する。

レポート内容:
    - セッション概要 (期間、総イベント数、severity 別件数)
    - 検知パターン (detector 別、rule 別 top-10)
    - 学習進捗 (FP率、sigma 調整数、ドリフト検出、再調整)
    - 推奨アクション (ルールベース、自動生成)
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DetectorSummary:
    detector: str
    event_count: int
    fp_count: int = 0
    tp_count: int = 0

    @property
    def fp_rate(self) -> float:
        total = self.fp_count + self.tp_count
        return (self.fp_count / total) if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "event_count": self.event_count,
            "fp_count": self.fp_count,
            "tp_count": self.tp_count,
            "fp_rate": round(self.fp_rate, 3),
        }


@dataclass
class ShiftReport:
    """シフトレポート — セッション単位の学習付き異常検知サマリー。"""

    session_id: int
    session_name: str
    generated_at: float

    # Session overview
    duration_seconds: float | None
    status: str
    total_events: int
    events_by_severity: dict[str, int] = field(default_factory=dict)

    # Top detection patterns
    top_rules: list[dict[str, Any]] = field(default_factory=list)
    top_detectors: list[DetectorSummary] = field(default_factory=list)

    # Learning progress
    tuner_feedback_total: int = 0
    tuner_confirm_rate: float = 0.0
    sigma_adjustments_total: int = 0
    detector_sigmas: dict[str, float] = field(default_factory=dict)
    drift_events: int = 0
    recalibrations: int = 0

    # Active query stats
    review_pending: int = 0
    review_confirmed: int = 0
    review_denied: int = 0

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "generated_at": self.generated_at,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "total_events": self.total_events,
            "events_by_severity": self.events_by_severity,
            "top_rules": self.top_rules,
            "top_detectors": [d.to_dict() for d in self.top_detectors],
            "tuner_feedback_total": self.tuner_feedback_total,
            "tuner_confirm_rate": round(self.tuner_confirm_rate, 3),
            "sigma_adjustments_total": self.sigma_adjustments_total,
            "detector_sigmas": {k: round(v, 4) for k, v in self.detector_sigmas.items()},
            "drift_events": self.drift_events,
            "recalibrations": self.recalibrations,
            "review_pending": self.review_pending,
            "review_confirmed": self.review_confirmed,
            "review_denied": self.review_denied,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class ShiftReportGenerator:
    """ShiftReport を生成するユーティリティ。

    vigil_repo から session/events を取得し、
    オプションで engine からリアルタイム学習状態を取得する。
    """

    def generate(
        self,
        session_id: int,
        vigil_repo: Any,
        engine: Any | None = None,
        sigma_tuner: Any | None = None,
        anomaly_tuner: Any | None = None,
        review_queue: Any | None = None,
    ) -> ShiftReport:
        """ShiftReport を生成して返す。

        Args:
            session_id:   vigil セッション ID。
            vigil_repo:   VigilRepository インスタンス。
            engine:       PerceptionEngine (オプション、学習状態取得用)。
            sigma_tuner:  SigmaTuner (オプション)。
            anomaly_tuner: AnomalyTuner (オプション)。
            review_queue: ReviewQueue (オプション)。
        """
        # ── Session metadata ─────────────────────────────────────────
        session = vigil_repo.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        session_name = session.get("name", f"Session {session_id}")
        status = session.get("status", "unknown")

        # Duration
        created_at = session.get("created_at")
        updated_at = session.get("updated_at")
        duration: float | None = None
        if created_at and updated_at:
            try:
                from datetime import datetime
                fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
                try:
                    t0 = datetime.fromisoformat(created_at)
                    t1 = datetime.fromisoformat(updated_at)
                    duration = (t1 - t0).total_seconds()
                except Exception:
                    pass
            except Exception:
                pass

        # ── Events ───────────────────────────────────────────────────
        events = vigil_repo.list_events(session_id)
        total_events = len(events)

        # severity counts
        sev_counter: Counter = Counter()
        rule_counter: Counter = Counter()
        detector_counter: Counter = Counter()

        for ev in events:
            sev = ev.get("severity", "info")
            sev_counter[sev] += 1
            for viol in ev.get("violations", []):
                rule = viol.get("rule") or viol.get("description_ja", "")
                if rule:
                    rule_counter[rule] += 1
                det = (viol.get("evidence") or {}).get("detector", "")
                if not det:
                    # check source
                    src = viol.get("source", "")
                    if src in ("anomaly", "pose"):
                        det = src
                if det:
                    detector_counter[det] += 1

        events_by_severity = dict(sev_counter)
        top_rules = [
            {"rule": rule, "count": count}
            for rule, count in rule_counter.most_common(10)
        ]

        # Detector summaries (without FP/TP here; filled from tuner below)
        detector_summaries: dict[str, DetectorSummary] = {
            det: DetectorSummary(detector=det, event_count=count)
            for det, count in detector_counter.most_common(8)
        }

        # ── Learning state ───────────────────────────────────────────
        # Try engine first
        tuner_stats: dict[str, Any] = {}
        sigma_state: dict[str, Any] = {}
        al_state: dict[str, Any] = {}

        if engine is not None:
            try:
                combined = engine.get_adaptive_learner_state()
                al_state = combined.get("adaptive_learner", {})
                tuner_stats = combined.get("tuner", {})
            except Exception:
                pass
            if sigma_tuner is None:
                sigma_tuner = getattr(engine, "_sigma_tuner", None)
            if anomaly_tuner is None:
                anomaly_tuner = getattr(engine, "_anomaly_tuner", None)
            if review_queue is None:
                review_queue = getattr(engine, "_review_queue", None)

        # Fallback to direct object calls
        if not tuner_stats and anomaly_tuner is not None:
            try:
                tuner_stats = anomaly_tuner.get_stats()
            except Exception:
                pass

        if sigma_tuner is not None:
            try:
                sigma_state = sigma_tuner.get_state()
            except Exception:
                pass

        tuner_feedback_total = tuner_stats.get("total_feedback", 0)
        tuner_confirm_rate = tuner_stats.get("overall_confirm_rate", 0.0)
        sigma_adjustments = sigma_state.get("total_adjustments", 0)
        drift_events = al_state.get("drift_count", 0)
        recalibrations = al_state.get("recalibration_count", 0)

        # Fill FP/TP per detector from pair_stats
        for ps in tuner_stats.get("pair_stats", []):
            det = ps.get("detector", "")
            if det in detector_summaries:
                detector_summaries[det].fp_count += ps.get("denied", 0)
                detector_summaries[det].tp_count += ps.get("confirmed", 0)

        # Sigma snapshot
        det_sigmas: dict[str, float] = {}
        for det, info in sigma_state.get("detector_sigmas", {}).items():
            det_sigmas[det] = info.get("current_sigma", sigma_state.get("base_sigma", 2.0))

        # Review queue stats
        rq_pending = rq_confirmed = rq_denied = 0
        if review_queue is not None:
            try:
                rq_stats = review_queue.get_stats()
                rq_pending = rq_stats.get("pending_count", 0)
                rq_confirmed = rq_stats.get("confirmed_count", 0)
                rq_denied = rq_stats.get("denied_count", 0)
            except Exception:
                pass

        # ── Recommendations ──────────────────────────────────────────
        recommendations = _build_recommendations(
            events_by_severity=events_by_severity,
            tuner_stats=tuner_stats,
            sigma_state=sigma_state,
            detector_summaries=list(detector_summaries.values()),
            drift_events=drift_events,
            review_pending=rq_pending,
        )

        return ShiftReport(
            session_id=session_id,
            session_name=session_name,
            generated_at=time.time(),
            duration_seconds=duration,
            status=status,
            total_events=total_events,
            events_by_severity=events_by_severity,
            top_rules=top_rules,
            top_detectors=list(detector_summaries.values()),
            tuner_feedback_total=tuner_feedback_total,
            tuner_confirm_rate=float(tuner_confirm_rate),
            sigma_adjustments_total=sigma_adjustments,
            detector_sigmas=det_sigmas,
            drift_events=drift_events,
            recalibrations=recalibrations,
            review_pending=rq_pending,
            review_confirmed=rq_confirmed,
            review_denied=rq_denied,
            recommendations=recommendations,
        )


# ---------------------------------------------------------------------------
# Recommendation engine (rule-based)
# ---------------------------------------------------------------------------


def _build_recommendations(
    events_by_severity: dict[str, int],
    tuner_stats: dict[str, Any],
    sigma_state: dict[str, Any],
    detector_summaries: list[DetectorSummary],
    drift_events: int,
    review_pending: int,
) -> list[str]:
    recs: list[str] = []

    # Critical events
    crit = events_by_severity.get("critical", 0)
    if crit >= 5:
        recs.append(f"重大な違反が {crit} 件検出されました。現場確認を推奨します。")

    # High FP rate per detector
    for ds in detector_summaries:
        if ds.fp_rate >= 0.7 and (ds.fp_count + ds.tp_count) >= 5:
            recs.append(
                f"「{ds.detector}」detector の FP 率が {ds.fp_rate:.0%} です。"
                f" sigma 閾値が自動引き上げ中です。"
            )

    # Sigma adjustments
    adj = sigma_state.get("total_adjustments", 0)
    if adj >= 3:
        recs.append(f"sigma が {adj} 回自動調整されました。安定稼働しています。")

    # Concept drift
    if drift_events >= 2:
        recs.append(f"概念ドリフトが {drift_events} 回検出されました。プロファイル再保存を検討してください。")

    # Pending reviews
    if review_pending >= 5:
        recs.append(f"確認待ちの異常が {review_pending} 件あります。レビューキューの確認を推奨します。")

    # Low total feedback
    total_fb = tuner_stats.get("total_feedback", 0)
    if total_fb == 0:
        recs.append("フィードバックが 0 件です。オペレーターのレビューで学習精度が向上します。")
    elif total_fb < 10:
        recs.append(f"フィードバックが {total_fb} 件です。引き続き確認作業を続けることで精度が向上します。")

    if not recs:
        recs.append("異常なし。システムは正常稼働中です。")

    return recs
