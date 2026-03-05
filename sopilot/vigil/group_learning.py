"""Phase 14B: グループ学習アーカイブ (Cross-Camera Federated Learning).

GroupLearningStore はカメラグループごとの学習済みσ値とフィードバック統計を
JSON ファイルに永続化し、グループ間での知識移植 (export → import) を可能にする。

フロー:
    エンジンで学習 → export → group_{id}.json に保存
    別グループで import → SigmaTuner にσ値を適用 → 即座に感度が最適化

compare() でグループ間のσ差異を分析し、「入口カメラは behavioral FP が多い」
等のインサイトを自動生成 (フェデレーテッドラーニング的推論)。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DETECTORS = ("behavioral", "spatial", "temporal", "interaction")

# σ差がこの値以上なら推奨 (recommendation) を生成
_RECOMMEND_DELTA_THRESHOLD: float = 0.30


class GroupLearningStore:
    """グループごとの学習σスナップショットを JSON ファイルで管理する。

    保存ディレクトリ: {store_dir}/group_{group_id}.json

    Args:
        store_dir: スナップショット保存ディレクトリ。起動時に自動作成。
    """

    def __init__(self, store_dir: Path | str) -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── internal ──────────────────────────────────────────────────────

    def _path(self, group_id: int) -> Path:
        return self._dir / f"group_{group_id}.json"

    # ── Public API ────────────────────────────────────────────────────

    def save(
        self,
        group_id: int,
        group_name: str,
        sigma_state: dict[str, Any],
        tuner_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """現在の学習状態をグループに紐づけて保存する。

        Args:
            group_id:    カメラグループ ID
            group_name:  カメラグループ名 (表示用)
            sigma_state: SigmaTuner.get_state() の返却値
            tuner_stats: AnomalyTuner.get_stats() の返却値

        Returns:
            保存されたスナップショット dict
        """
        # detector ごとの現在σを抽出
        base_sigma = sigma_state.get("base_sigma", 2.0)
        detector_sigmas: dict[str, float] = {}
        for det in _DETECTORS:
            info = sigma_state.get("detector_sigmas", {}).get(det, {})
            detector_sigmas[det] = round(
                info.get("current_sigma", base_sigma), 4
            )

        # suppressed / trusted ペアの識別子リスト
        suppressed = [
            f"{p['detector']}/{p['metric']}"
            for p in tuner_stats.get("suppressed_pairs", [])
        ]
        trusted = [
            f"{p['detector']}/{p['metric']}"
            for p in tuner_stats.get("trusted_pairs", [])
        ]

        snapshot: dict[str, Any] = {
            "group_id": group_id,
            "group_name": group_name,
            "saved_at": time.time(),
            "base_sigma": base_sigma,
            "total_adjustments": sigma_state.get("total_adjustments", 0),
            "detector_sigmas": detector_sigmas,
            "total_feedback": tuner_stats.get("total_feedback", 0),
            "overall_confirm_rate": round(
                tuner_stats.get("overall_confirm_rate", 0.0), 4
            ),
            "suppressed_pairs": suppressed,
            "trusted_pairs": trusted,
        }

        try:
            self._path(group_id).write_text(
                json.dumps(snapshot, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            logger.exception(
                "GroupLearningStore: failed to save group %d", group_id
            )
        return snapshot

    def load(self, group_id: int) -> dict[str, Any] | None:
        """グループの学習スナップショットを読み込む。

        Returns:
            スナップショット dict、存在しなければ None
        """
        p = self._path(group_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            logger.exception(
                "GroupLearningStore: failed to load group %d", group_id
            )
            return None

    def delete(self, group_id: int) -> bool:
        """グループの学習スナップショットを削除する。

        Returns:
            True — 削除した / False — ファイルが存在しなかった
        """
        p = self._path(group_id)
        if not p.exists():
            return False
        try:
            p.unlink()
            return True
        except Exception:
            logger.exception(
                "GroupLearningStore: failed to delete group %d", group_id
            )
            return False

    def list_all(self) -> list[dict[str, Any]]:
        """全グループのスナップショット一覧を返す (saved_at 降順)。"""
        result: list[dict[str, Any]] = []
        for p in self._dir.glob("group_*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                result.append(data)
            except Exception:
                logger.debug("GroupLearningStore: skip malformed %s", p.name)
        result.sort(key=lambda x: x.get("saved_at", 0.0), reverse=True)
        return result

    def compare(self) -> dict[str, Any]:
        """全グループ間のσ値を比較し、差異と推奨インサイトを返す。

        Returns:
            {
                "groups": [...],        # list_all() と同じ
                "recommendations": [    # σ差が閾値以上のdetector
                    {
                        "detector": "behavioral",
                        "highest_sigma": {"group_id": 1, "group_name": "入口", "sigma": 3.2},
                        "lowest_sigma":  {"group_id": 2, "group_name": "製造", "sigma": 1.9},
                        "delta": 1.3,
                        "note_ja": "...",
                    }
                ],
            }
        """
        snapshots = self.list_all()
        if not snapshots:
            return {"groups": [], "recommendations": []}

        recommendations: list[dict[str, Any]] = []
        for det in _DETECTORS:
            sigmas = [
                (
                    s["group_id"],
                    s.get("group_name", f"group_{s['group_id']}"),
                    s.get("detector_sigmas", {}).get(det, s.get("base_sigma", 2.0)),
                )
                for s in snapshots
                if "detector_sigmas" in s
            ]
            if len(sigmas) < 2:
                continue
            sigmas.sort(key=lambda x: x[2])
            lo_id, lo_name, lo_val = sigmas[0]
            hi_id, hi_name, hi_val = sigmas[-1]
            delta = round(hi_val - lo_val, 4)
            if delta >= _RECOMMEND_DELTA_THRESHOLD:
                recommendations.append(
                    {
                        "detector": det,
                        "highest_sigma": {
                            "group_id": hi_id,
                            "group_name": hi_name,
                            "sigma": round(hi_val, 4),
                        },
                        "lowest_sigma": {
                            "group_id": lo_id,
                            "group_name": lo_name,
                            "sigma": round(lo_val, 4),
                        },
                        "delta": delta,
                        "note_ja": (
                            f"{hi_name} は {det} の FP が多い傾向 "
                            f"(σ={hi_val:.2f})。"
                            f"{lo_name} の学習をインポートすると"
                            f"感度が改善する可能性があります。"
                        ),
                    }
                )

        return {"groups": snapshots, "recommendations": recommendations}
