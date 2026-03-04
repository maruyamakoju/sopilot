"""Phase 11A: 能動的クエリ戦略 (Active Query Strategy).

ReviewQueue はエンジンが検出した高不確実性の異常イベントを蓄積し、
オペレーターへラベル付けを要請する。

フロー:
    ANOMALY event (z_score ≥ threshold)
        → ReviewQueue.maybe_add()
        → UI に「確認待ち N 件」バッジ
        → POST /vigil/perception/review/{id} {confirmed, note}
        → AnomalyTuner.record_feedback() へ自動ルーティング
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sopilot.perception.types import EntityEvent


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ReviewItem:
    """オペレーターへのラベル付けリクエスト。"""

    review_id: str
    detector: str
    metric: str
    entity_id: int
    z_score: float
    timestamp: float
    frame_number: int
    description_ja: str
    priority: float          # 高いほど先に表示 (= z_score)
    created_at: float
    reviewed: bool = False
    confirmed: bool | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "detector": self.detector,
            "metric": self.metric,
            "entity_id": self.entity_id,
            "z_score": round(self.z_score, 3),
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "description_ja": self.description_ja,
            "priority": round(self.priority, 3),
            "created_at": self.created_at,
            "reviewed": self.reviewed,
            "confirmed": self.confirmed,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# ReviewQueue
# ---------------------------------------------------------------------------


class ReviewQueue:
    """高 z_score の異常イベントを保留リストに蓄積し、
    オペレーターレビューを要求するキュー。

    Args:
        z_threshold: この値以上の z_score を持つイベントを追加対象とする。
        max_pending:  保留リストの最大件数 (超えると古いものから削除)。
        dedup_seconds: 同一 (detector, metric) を重複追加しない時間窓 (秒)。
        min_feedback_for_skip: AnomalyTuner にこの件数以上のフィードバックが
                               あるペアはスキップ（すでに十分学習済み）。
    """

    def __init__(
        self,
        z_threshold: float = 2.5,
        max_pending: int = 50,
        dedup_seconds: float = 60.0,
        min_feedback_for_skip: int = 10,
    ) -> None:
        self._z_threshold = z_threshold
        self._max_pending = max_pending
        self._dedup_seconds = dedup_seconds
        self._min_feedback = min_feedback_for_skip

        self._pending: list[ReviewItem] = []
        self._reviewed: list[ReviewItem] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_add(
        self,
        event: "EntityEvent",
        z_score: float,
        tuner: Any | None = None,
    ) -> bool:
        """イベントを review queue に追加するか判断する。

        Returns:
            True  — 追加した
            False — スキップ (threshold 未満、重複、十分学習済み)
        """
        if z_score < self._z_threshold:
            return False

        detector = event.details.get("detector", "") if event.details else ""
        metric = event.details.get("metric", "") if event.details else ""

        # 十分フィードバックがあるペアはスキップ
        if tuner is not None:
            try:
                stats = tuner.get_pair_stats(detector, metric)
                if stats is not None and stats.total >= self._min_feedback:
                    return False
            except Exception:
                pass

        now = time.time()

        with self._lock:
            # 重複チェック: 直近 dedup_seconds 以内に同一ペアがあればスキップ
            for item in self._pending:
                if (
                    item.detector == detector
                    and item.metric == metric
                    and (now - item.created_at) < self._dedup_seconds
                ):
                    return False

            desc_ja = ""
            if event.details:
                desc_ja = event.details.get("description_ja", "")

            new_item = ReviewItem(
                review_id=uuid.uuid4().hex[:8],
                detector=detector,
                metric=metric,
                entity_id=event.entity_id,
                z_score=z_score,
                timestamp=event.timestamp,
                frame_number=event.frame_number,
                description_ja=desc_ja,
                priority=z_score,
                created_at=now,
            )
            self._pending.append(new_item)

            # max_pending を超えたら priority 最低のものを削除
            if len(self._pending) > self._max_pending:
                self._pending.sort(key=lambda x: -x.priority)
                self._pending = self._pending[: self._max_pending]

            return True

    def get_pending(self, n: int = 10) -> list[ReviewItem]:
        """priority 降順で最大 n 件の保留アイテムを返す。"""
        with self._lock:
            items = sorted(self._pending, key=lambda x: -x.priority)
            return items[:n]

    def record_review(
        self,
        review_id: str,
        confirmed: bool,
        note: str = "",
    ) -> ReviewItem | None:
        """レビュー結果を記録し、pending から reviewed へ移動する。

        Returns:
            ReviewItem — 見つかった場合
            None       — 該当 review_id が pending にない場合
        """
        with self._lock:
            for i, item in enumerate(self._pending):
                if item.review_id == review_id:
                    item.reviewed = True
                    item.confirmed = confirmed
                    item.note = note
                    self._pending.pop(i)
                    self._reviewed.append(item)
                    return item
            return None

    def get_stats(self) -> dict[str, Any]:
        """キューの統計情報を返す。"""
        with self._lock:
            pending = list(self._pending)
            reviewed = list(self._reviewed)

        confirmed_count = sum(1 for r in reviewed if r.confirmed is True)
        denied_count = sum(1 for r in reviewed if r.confirmed is False)

        # detector ごとの pending 件数
        detector_counts: dict[str, int] = {}
        for item in pending:
            detector_counts[item.detector] = detector_counts.get(item.detector, 0) + 1

        return {
            "pending_count": len(pending),
            "reviewed_count": len(reviewed),
            "confirmed_count": confirmed_count,
            "denied_count": denied_count,
            "confirm_rate": (
                confirmed_count / len(reviewed) if reviewed else 0.0
            ),
            "detector_counts": detector_counts,
            "z_threshold": self._z_threshold,
            "max_pending": self._max_pending,
        }

    def clear(self) -> None:
        """保留リストと履歴をすべてクリアする。"""
        with self._lock:
            self._pending.clear()
            self._reviewed.clear()

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)
