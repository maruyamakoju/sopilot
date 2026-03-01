"""Task profile management."""

from typing import Any

from sopilot.config import Settings
from sopilot.constants import DEFAULT_DEVIATION_POLICY, DEFAULT_WEIGHTS
from sopilot.database import Database
from sopilot.exceptions import InvalidStateError, NotFoundError
from sopilot.schemas import ScoreWeights


class TaskProfileService:
    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.database = database
        self._ensure_primary_task_profile()

    def _ensure_primary_task_profile(self) -> None:
        existing = self.database.get_task_profile(self.settings.primary_task_id)
        if existing:
            return
        self.database.upsert_task_profile(
            task_id=self.settings.primary_task_id,
            task_name=self.settings.primary_task_name,
            pass_score=self.settings.default_pass_score,
            retrain_score=self.settings.default_retrain_score,
            default_weights=DEFAULT_WEIGHTS,
            deviation_policy=DEFAULT_DEVIATION_POLICY,
        )

    def get_task_profile(self) -> dict[str, Any]:
        profile = self.database.get_task_profile(self.settings.primary_task_id)
        if profile is None:
            raise NotFoundError("Primary task profile not found")
        return dict(profile)

    def update_task_profile(
        self,
        *,
        task_name: str | None,
        pass_score: float | None,
        retrain_score: float | None,
        default_weights: dict[str, float] | None,
        deviation_policy: dict[str, str] | None,
    ) -> dict[str, Any]:
        profile = self.get_task_profile()
        new_name = task_name if task_name is not None else profile["task_name"]
        new_pass = pass_score if pass_score is not None else float(profile["pass_score"])
        new_retrain = retrain_score if retrain_score is not None else float(profile["retrain_score"])
        if new_retrain > new_pass:
            raise InvalidStateError("retrain_score must be <= pass_score")
        new_weights = default_weights if default_weights is not None else dict(profile["default_weights"])
        parsed_weights = ScoreWeights(**new_weights)
        new_policy = deviation_policy if deviation_policy is not None else dict(profile["deviation_policy"])
        self.database.upsert_task_profile(
            task_id=self.settings.primary_task_id,
            task_name=new_name,
            pass_score=float(new_pass),
            retrain_score=float(new_retrain),
            default_weights=parsed_weights.model_dump(),
            deviation_policy=new_policy,
        )
        return self.get_task_profile()

    def get_task_profile_for(self, task_id: str) -> dict[str, Any]:
        profile = self.database.get_task_profile(task_id)
        if profile is None:
            raise NotFoundError(f"Task profile for '{task_id}' not found")
        return dict(profile)
