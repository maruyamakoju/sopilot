"""Service layer for SOP step definition management."""

from __future__ import annotations

import logging

from sopilot.database import Database

logger = logging.getLogger(__name__)


class StepDefinitionService:
    """Manages named SOP step definitions and time thresholds."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def get_steps(self, task_id: str) -> dict:
        """Return step definitions for a task."""
        steps = self._db.get_sop_steps(task_id)
        return {
            "task_id": task_id,
            "step_count": len(steps),
            "steps": steps,
        }

    def upsert_steps(self, task_id: str, steps: list[dict]) -> dict:
        """Upsert step definitions. Returns updated step list."""
        # Validate
        for s in steps:
            if not isinstance(s.get("step_index"), int) or s["step_index"] < 0:
                raise ValueError(f"Invalid step_index: {s.get('step_index')!r}")
            exp = s.get("expected_duration_sec")
            mn = s.get("min_duration_sec")
            mx = s.get("max_duration_sec")
            if exp is not None and exp <= 0:
                raise ValueError(f"expected_duration_sec must be positive, got {exp}")
            if mn is not None and mx is not None and mn > mx:
                raise ValueError(f"min_duration_sec ({mn}) must be <= max_duration_sec ({mx})")
        count = self._db.upsert_sop_steps(task_id, steps)
        logger.info("upserted %d step definitions for task %s", count, task_id)
        return self.get_steps(task_id)

    def delete_steps(self, task_id: str) -> dict:
        """Delete all step definitions for a task."""
        count = self._db.delete_sop_steps(task_id)
        logger.info("deleted %d step definitions for task %s", count, task_id)
        return {"task_id": task_id, "deleted_count": count}

    def compute_time_compliance(
        self,
        task_id: str,
        boundaries: list[int],
        *,
        sample_fps: int = 4,
        clip_seconds: int = 4,
    ) -> list[dict]:
        """
        Given gold video boundary clip indices and frame/clip settings,
        compute per-step actual duration and compliance status.

        boundaries: list of clip indices where steps start (len = n_steps + 1,
                    last element = total clip count)

        Returns list of dicts with keys:
            step_index, name_ja, name_en, actual_duration_sec,
            expected_duration_sec, min_duration_sec, max_duration_sec,
            is_critical, compliance ('ok'|'too_fast'|'too_slow'|'undefined')
        """
        step_defs = {s["step_index"]: s for s in self._db.get_sop_steps(task_id)}
        seconds_per_clip = clip_seconds  # each clip = clip_seconds of video

        results = []
        n_steps = max(len(boundaries) - 1, 0)
        for idx in range(n_steps):
            clip_count = boundaries[idx + 1] - boundaries[idx]
            actual_sec = clip_count * seconds_per_clip
            defn = step_defs.get(idx, {})
            mn = defn.get("min_duration_sec")
            mx = defn.get("max_duration_sec")
            if mn is None and mx is None:
                compliance = "undefined"
            elif mn is not None and actual_sec < mn:
                compliance = "too_fast"
            elif mx is not None and actual_sec > mx:
                compliance = "too_slow"
            else:
                compliance = "ok"
            results.append(
                {
                    "step_index": idx,
                    "name_ja": defn.get("name_ja", f"手順{idx + 1}"),
                    "name_en": defn.get("name_en", f"Step {idx + 1}"),
                    "actual_duration_sec": round(actual_sec, 1),
                    "expected_duration_sec": defn.get("expected_duration_sec"),
                    "min_duration_sec": mn,
                    "max_duration_sec": mx,
                    "is_critical": bool(defn.get("is_critical")),
                    "compliance": compliance,
                }
            )
        return results
