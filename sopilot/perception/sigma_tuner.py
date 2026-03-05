"""Phase 12A: 自動σ調整 (Adaptive Sigma Tuning).

SigmaTuner はオペレーターのフィードバック (AnomalyTuner の pair_stats) から
detector ごとの FP 率を集計し、sigma_threshold を自動調整する。

デッドゾーン (±15% of TARGET_FP_RATE) 内では変更しない。
- FP率 > TARGET + DEAD_ZONE → sigma を上げる (感度を下げる)
- FP率 < TARGET - DEAD_ZONE → sigma を下げる (感度を上げる、慎重に)

調整後の sigma は engine._events_to_violations() でフィルタとして使用される。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DETECTORS = ("behavioral", "spatial", "temporal", "interaction")


# ---------------------------------------------------------------------------
# SigmaTuner
# ---------------------------------------------------------------------------


class SigmaTuner:
    """per-detector 可変 sigma_threshold マネージャー。

    Args:
        base_sigma: グローバルデフォルト sigma (PerceptionConfig.anomaly_sigma_threshold)。
        target_fp_rate: 目標 FP 率 (0.0〜1.0)。デフォルト 0.30。
        dead_zone: この範囲内では調整しない (FP率の絶対偏差)。
        lr_up: FP率が高いときの引き上げ学習率。
        lr_down: FP率が低いときの引き下げ学習率 (上より保守的)。
        min_samples: 調整に必要な最小フィードバック件数。
        sigma_min / sigma_max: クランプ範囲。
    """

    TARGET_FP_RATE: float = 0.30
    DEAD_ZONE: float = 0.15
    LR_UP: float = 0.20
    LR_DOWN: float = 0.10
    MIN_SAMPLES: int = 5
    SIGMA_MIN: float = 1.0
    SIGMA_MAX: float = 6.0

    def __init__(
        self,
        base_sigma: float = 2.0,
        target_fp_rate: float = TARGET_FP_RATE,
        dead_zone: float = DEAD_ZONE,
        lr_up: float = LR_UP,
        lr_down: float = LR_DOWN,
        min_samples: int = MIN_SAMPLES,
        sigma_min: float = SIGMA_MIN,
        sigma_max: float = SIGMA_MAX,
        state_path: "Path | str | None" = None,
    ) -> None:
        self._base_sigma = base_sigma
        self._target_fp_rate = target_fp_rate
        self._dead_zone = dead_zone
        self._lr_up = lr_up
        self._lr_down = lr_down
        self._min_samples = min_samples
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._state_path = Path(state_path) if state_path else None

        # detector → adjusted sigma (absent = use base_sigma)
        self._detector_sigmas: dict[str, float] = {}
        self._adjustment_history: list[dict[str, Any]] = []
        self._total_adjustments: int = 0

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """状態を JSON ファイルに保存する (best-effort)。"""
        if self._state_path is None:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "detector_sigmas": self._detector_sigmas,
                "total_adjustments": self._total_adjustments,
                "adjustment_history": self._adjustment_history,
            }
            self._state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("SigmaTuner: failed to save state to %s", self._state_path)

    def _load(self) -> None:
        """JSON ファイルから状態を復元する。ファイルがなければ何もしない。"""
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._detector_sigmas = {
                str(k): float(v) for k, v in data.get("detector_sigmas", {}).items()
            }
            self._total_adjustments = int(data.get("total_adjustments", 0))
            self._adjustment_history = list(data.get("adjustment_history", []))
        except Exception:
            logger.exception("SigmaTuner: failed to load state from %s", self._state_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sigma(self, detector: str) -> float:
        """detector の現在 sigma を返す。未調整なら base_sigma。"""
        return self._detector_sigmas.get(detector, self._base_sigma)

    def compute_and_apply(self, tuner_stats: dict[str, Any]) -> list[dict[str, Any]]:
        """AnomalyTuner.get_stats() の結果から per-detector sigma を更新する。

        Returns:
            変更リスト (変更がなければ空リスト)。
        """
        pair_stats = tuner_stats.get("pair_stats", [])
        if not pair_stats:
            return []

        # detector ごとに集計
        agg: dict[str, dict[str, int]] = {}
        for ps in pair_stats:
            det = ps.get("detector", "")
            if not det:
                continue
            if det not in agg:
                agg[det] = {"total": 0, "denied": 0, "confirmed": 0}
            agg[det]["total"] += int(ps.get("total", 0))
            agg[det]["denied"] += int(ps.get("denied", 0))
            agg[det]["confirmed"] += int(ps.get("confirmed", 0))

        changes: list[dict[str, Any]] = []
        now = time.time()

        for det, stats in agg.items():
            total = stats["total"]
            if total < self._min_samples:
                continue

            fp_rate = stats["denied"] / total
            old_sigma = self.get_sigma(det)

            if fp_rate > self._target_fp_rate + self._dead_zone:
                # FP が多すぎる → sigma を上げる
                excess = fp_rate - self._target_fp_rate
                new_sigma = min(
                    old_sigma * (1.0 + self._lr_up * excess),
                    self._sigma_max,
                )
            elif fp_rate < self._target_fp_rate - self._dead_zone:
                # FP が少ない (TP が多い) → sigma を慎重に下げる
                gap = self._target_fp_rate - fp_rate
                new_sigma = max(
                    old_sigma * (1.0 - self._lr_down * gap),
                    self._sigma_min,
                )
            else:
                continue  # dead zone 内 → 変更なし

            if abs(new_sigma - old_sigma) < 0.005:
                continue  # 実質変化なし

            self._detector_sigmas[det] = round(new_sigma, 4)
            change: dict[str, Any] = {
                "detector": det,
                "old_sigma": round(old_sigma, 4),
                "new_sigma": round(new_sigma, 4),
                "fp_rate": round(fp_rate, 4),
                "samples": total,
                "timestamp": now,
                "direction": "up" if new_sigma > old_sigma else "down",
            }
            self._adjustment_history.append(change)
            self._total_adjustments += 1
            changes.append(change)

        if changes:
            self._save()

        return changes

    def get_state(self) -> dict[str, Any]:
        """現在の sigma 状態を返す (API レスポンス用)。"""
        detector_sigmas = {}
        for det in _DETECTORS:
            cur = self._detector_sigmas.get(det, self._base_sigma)
            detector_sigmas[det] = {
                "current_sigma": round(cur, 4),
                "base_sigma": self._base_sigma,
                "delta": round(cur - self._base_sigma, 4),
                "adjusted": det in self._detector_sigmas,
            }

        return {
            "base_sigma": self._base_sigma,
            "target_fp_rate": self._target_fp_rate,
            "total_adjustments": self._total_adjustments,
            "detector_sigmas": detector_sigmas,
            "recent_adjustments": self._adjustment_history[-5:],
        }

    def apply_detector_sigmas(self, sigmas: dict[str, float]) -> list[str]:
        """外部スナップショット (グループ学習等) からσ値を一括適用する。

        Args:
            sigmas: {detector: sigma_value} dict。不明なdetectorはスキップ。

        Returns:
            適用されたdetectorのリスト。
        """
        applied: list[str] = []
        for det, val in sigmas.items():
            clamped = max(self._sigma_min, min(self._sigma_max, float(val)))
            self._detector_sigmas[det] = round(clamped, 4)
            applied.append(det)
        if applied:
            self._save()
        return applied

    def get_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """σ調整履歴を返す (最新 limit 件)。limit=0 で全件。"""
        if limit <= 0:
            return list(self._adjustment_history)
        return self._adjustment_history[-limit:]

    def reset(self) -> None:
        """全 sigma をベースラインに戻し、履歴もクリアする。"""
        self._detector_sigmas.clear()
        self._adjustment_history.clear()
        self._total_adjustments = 0
        self._save()
