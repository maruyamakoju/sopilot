"""Vigil Shift Report + Daily Dossier Generator.

Phase 12B: Session-based ShiftReport (ShiftReportGenerator)
Phase 51: Date-based daily dossier (generate_daily_dossier)

Aggregates events, VLM deliberations, and spatial map history into unified
management reports.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data classes (Phase 12B)
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
# Generator (Phase 12B)
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
        """ShiftReport を生成して返す。"""
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
                t0 = datetime.fromisoformat(created_at)
                t1 = datetime.fromisoformat(updated_at)
                duration = (t1 - t0).total_seconds()
            except Exception:
                pass

        # ── Events ───────────────────────────────────────────────────
        events = vigil_repo.list_events(session_id)
        total_events = len(events)

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

        detector_summaries: dict[str, DetectorSummary] = {
            det: DetectorSummary(detector=det, event_count=count)
            for det, count in detector_counter.most_common(8)
        }

        # ── Learning state ───────────────────────────────────────────
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

        for ps in tuner_stats.get("pair_stats", []):
            det = ps.get("detector", "")
            if det in detector_summaries:
                detector_summaries[det].fp_count += ps.get("denied", 0)
                detector_summaries[det].tp_count += ps.get("confirmed", 0)

        det_sigmas: dict[str, float] = {}
        for det, info in sigma_state.get("detector_sigmas", {}).items():
            det_sigmas[det] = info.get("current_sigma", sigma_state.get("base_sigma", 2.0))

        rq_pending = rq_confirmed = rq_denied = 0
        if review_queue is not None:
            try:
                rq_stats = review_queue.get_stats()
                rq_pending = rq_stats.get("pending_count", 0)
                rq_confirmed = rq_stats.get("confirmed_count", 0)
                rq_denied = rq_stats.get("denied_count", 0)
            except Exception:
                pass

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

    crit = events_by_severity.get("critical", 0)
    if crit >= 5:
        recs.append(f"重大な違反が {crit} 件検出されました。現場確認を推奨します。")

    for ds in detector_summaries:
        if ds.fp_rate >= 0.7 and (ds.fp_count + ds.tp_count) >= 5:
            recs.append(
                f"「{ds.detector}」detector の FP 率が {ds.fp_rate:.0%} です。"
                f" sigma 閾値が自動引き上げ中です。"
            )

    adj = sigma_state.get("total_adjustments", 0)
    if adj >= 3:
        recs.append(f"sigma が {adj} 回自動調整されました。安定稼働しています。")

    if drift_events >= 2:
        recs.append(f"概念ドリフトが {drift_events} 回検出されました。プロファイル再保存を検討してください。")

    if review_pending >= 5:
        recs.append(f"確認待ちの異常が {review_pending} 件あります。レビューキューの確認を推奨します。")

    total_fb = tuner_stats.get("total_feedback", 0)
    if total_fb == 0:
        recs.append("フィードバックが 0 件です。オペレーターのレビューで学習精度が向上します。")
    elif total_fb < 10:
        recs.append(f"フィードバックが {total_fb} 件です。引き続き確認作業を続けることで精度が向上します。")

    if not recs:
        recs.append("異常なし。システムは正常稼働中です。")

    return recs


# ---------------------------------------------------------------------------
# Daily Dossier (Phase 51)
# ---------------------------------------------------------------------------


def generate_daily_dossier(
    repo: Any,
    date_str: str | None = None,
    output_dir: Path | None = None,
) -> str:
    """Generate a markdown report for a specific day.

    Aggregates events, VLM deliberations, and spatial map history into a
    unified management report saved as markdown.

    Args:
        repo: VigilRepository instance.
        date_str: Date in 'YYYY-MM-DD' format (defaults to today UTC).
        output_dir: Directory to save the report (defaults to results/reports/).

    Returns:
        Path to the generated markdown file.
    """
    if not output_dir:
        output_dir = Path("results/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not date_str:
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")

    start_time = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    end_time = start_time + timedelta(days=1)

    report_md = f"# SOPilot 現場安全レポート: {date_str}\n\n"
    report_md += "## 本日のサマリー\n\n"

    with repo._connect() as conn:
        total_events = conn.execute(
            "SELECT COUNT(*) FROM vigil_events WHERE created_at >= ? AND created_at < ?",
            (start_time.isoformat(), end_time.isoformat()),
        ).fetchone()[0]

        report_md += f"- **総違反・警告件数:** {total_events}件\n"

        if total_events == 0:
            report_md += "\n本日は記録された違反や警告はありませんでした。素晴らしい安全記録です！\n"
        else:
            report_md += "\n## 重大なインシデント (Urgency: CRITICAL / HIGH)\n\n"

            rows = conn.execute(
                "SELECT * FROM vigil_events WHERE created_at >= ? AND created_at < ?"
                " ORDER BY timestamp_sec DESC",
                (start_time.isoformat(), end_time.isoformat()),
            ).fetchall()

            for row in rows:
                d = dict(row)
                d["violations"] = json.loads(d.pop("violations_json", "[]"))
                delib_str = d.pop("deliberation_json", None)
                d["deliberation"] = json.loads(delib_str) if delib_str else None

                delib = d.get("deliberation")
                if delib and delib.get("urgency") in ("CRITICAL", "HIGH"):
                    hyp = delib.get("best_hypothesis", {}).get("claim_ja", "不明な仮説")
                    act = delib.get("action_ja", "")
                    rule = (
                        d["violations"][0].get("rule", "不明なルール")
                        if d["violations"]
                        else "不明"
                    )
                    time_fmt = datetime.fromisoformat(d["created_at"]).strftime("%H:%M:%S")

                    report_md += f"### 🕒 {time_fmt} - {rule}\n"
                    report_md += f"- **AIの推論 (System 2):** {hyp}\n"
                    report_md += f"- **推奨アクション:** {act}\n"
                    report_md += f"- **緊急度:** `{delib.get('urgency')}`\n"
                    report_md += (
                        f"- **ステータス:** "
                        f"{'✅ 確認済み' if d.get('acknowledged_at') else '⚠️ 未確認'}\n\n"
                    )

    report_md += "\n---\n*このレポートは SOPilot Perception Engine によって自動生成されました。*\n"

    out_path = output_dir / f"dossier_{date_str}.md"
    out_path.write_text(report_md, encoding="utf-8")
    return str(out_path)
