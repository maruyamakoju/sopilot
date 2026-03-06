"""Phase 19: Perception Engine Health History (ヘルスヒストリ).

HealthHistoryStore は Phase 18 のヘルススコアを時系列で蓄積し、
最大 30 日分のトレンドを提供する。

設計:
    - JSON ファイルに 1 時間ごとのスナップショットを保存 (最大 720 件)
    - record() は前回記録から RECORD_INTERVAL_SECONDS 未満なら no-op (レート制限)
    - get_history() / get_trend() はロック不要 (読み取り専用コピーを返す)
    - best-effort: 保存失敗は logger.exception で握り潰す
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RECORD_INTERVAL_SECONDS: float = 3600.0   # 1 hour between auto-snapshots
MAX_RECORDS: int = 720                     # 30 days × 24 h
_DAY_SECONDS: float = 86400.0


class HealthHistoryStore:
    """Persistent time-series store for Perception Engine health scores (Phase 19).

    Args:
        state_path: Path to the JSON file for persistence. If None, in-memory only.
        record_interval_seconds: Minimum seconds between auto-recorded snapshots.
        max_records: Maximum number of snapshots to keep.
    """

    def __init__(
        self,
        state_path: Path | str | None = None,
        record_interval_seconds: float = RECORD_INTERVAL_SECONDS,
        max_records: int = MAX_RECORDS,
    ) -> None:
        self._state_path = Path(state_path) if state_path else None
        self._interval = record_interval_seconds
        self._max_records = max_records

        # list of dicts: {score, grade, factors, total_penalty, recorded_at}
        self._records: list[dict[str, Any]] = []
        self._last_ts: float = 0.0
        self._lock = threading.Lock()

        self._load()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        score: int,
        grade: str,
        factors: dict[str, Any],
        total_penalty: float = 0.0,
        *,
        force: bool = False,
    ) -> bool:
        """Append a health snapshot if the rate-limit interval has elapsed.

        Args:
            score: Health score [0, 100].
            grade: Grade string (A/B/C/D/F).
            factors: Factor breakdown dict from PerceptionHealthScorer.
            total_penalty: Sum of all penalties.
            force: If True, bypass the rate-limit check.

        Returns:
            True if the snapshot was actually recorded, False if rate-limited.
        """
        now = time.time()
        with self._lock:
            if not force and (now - self._last_ts) < self._interval:
                return False

            snapshot: dict[str, Any] = {
                "score": score,
                "grade": grade,
                "total_penalty": round(total_penalty, 2),
                "recorded_at": now,
                # Store a lightweight factors summary (penalty values only)
                "factor_penalties": {
                    k: round(v.get("penalty", 0.0), 2)
                    for k, v in factors.items()
                } if factors else {},
            }
            self._records.append(snapshot)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]
            self._last_ts = now

        self._save()
        return True

    def clear(self) -> None:
        """Remove all records and reset the rate-limit timer."""
        with self._lock:
            self._records.clear()
            self._last_ts = 0.0
        self._save()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_history(self, days: float = 7.0, limit: int = 0) -> list[dict[str, Any]]:
        """Return snapshots from the last N days, most recent first.

        Args:
            days: Look-back window in days.
            limit: Maximum number of records to return (0 = all).

        Returns:
            List of snapshot dicts sorted newest first.
        """
        cutoff = time.time() - days * _DAY_SECONDS
        with self._lock:
            items = [r for r in self._records if r["recorded_at"] >= cutoff]
        items = list(reversed(items))  # newest first
        if limit > 0:
            items = items[:limit]
        return items

    def get_trend(self, days: float = 7.0) -> dict[str, Any]:
        """Return trend statistics for the last N days.

        Returns:
            dict with keys: avg_score, min_score, max_score,
            first_score, last_score, improvement, record_count, days.
        """
        items = self.get_history(days)  # newest first
        if not items:
            return {
                "avg_score": None,
                "min_score": None,
                "max_score": None,
                "first_score": None,
                "last_score": None,
                "improvement": None,
                "record_count": 0,
                "days": days,
            }

        scores = [r["score"] for r in items]
        first_score = items[-1]["score"]   # oldest
        last_score = items[0]["score"]     # newest

        return {
            "avg_score": round(sum(scores) / len(scores), 1),
            "min_score": min(scores),
            "max_score": max(scores),
            "first_score": first_score,
            "last_score": last_score,
            "improvement": last_score - first_score,
            "record_count": len(items),
            "days": days,
        }

    def get_sparkline_data(self, days: float = 7.0, points: int = 30) -> list[dict[str, Any]]:
        """Return downsampled history suitable for a sparkline chart.

        Args:
            days: Look-back window in days.
            points: Maximum data points to return.

        Returns:
            List of {recorded_at, score} dicts, oldest first.
        """
        items = list(reversed(self.get_history(days)))  # oldest first
        if len(items) <= points:
            return [{"recorded_at": r["recorded_at"], "score": r["score"]} for r in items]

        # Uniform downsampling
        step = len(items) / points
        sampled = [items[int(i * step)] for i in range(points)]
        return [{"recorded_at": r["recorded_at"], "score": r["score"]} for r in sampled]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        if self._state_path is None:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {"records": list(self._records), "last_ts": self._last_ts}
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            tmp.replace(self._state_path)
        except Exception:
            logger.exception("HealthHistoryStore._save() failed")

    def _load(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            with self._lock:
                self._records = data.get("records", [])
                self._last_ts = float(data.get("last_ts", 0.0))
            # Enforce max_records after load
            with self._lock:
                if len(self._records) > self._max_records:
                    self._records = self._records[-self._max_records:]
        except Exception:
            logger.exception("HealthHistoryStore._load() failed")
