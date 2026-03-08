"""Phase 54: Long-Horizon Safety Planner.

Translates high-level safety goals into concrete, time-bounded subtask plans,
monitors execution progress, and adapts plans when new violations are observed.

Architecture::

    SafetyPlanner.submit_goal(description, priority, horizon_s, camera_ids)
        → Plan (with auto-decomposed SubTasks)

    SafetyPlanner.tick(observations)
        → list[SubTask]   # newly activated tasks for current tick

    SafetyPlanner.report_task_complete(task_id, success, obs)
        → updates plan progress, may spawn follow-up tasks

    SafetyPlanner.get_active_plans()
        → list[Plan]

Goal types and decomposition rules
------------------------------------
The planner uses a rule-based decomposition table.  Each goal pattern maps to
a list of subtask templates.  Task types:

    monitor     — continuously watch a zone/camera
    check_ppe   — verify PPE compliance for detected persons
    alert       — trigger operator notification
    investigate — dispatch VLM/RAG query on historical footage
    cordon      — mark a zone as restricted (advisory)

Usage::

    planner = SafetyPlanner()
    plan_id = planner.submit_goal(
        description="倉庫エリアの安全スコアを最大化する",
        priority=2,
        camera_ids=["cam01", "cam02"],
        horizon_seconds=3600,
    )
    tasks = planner.tick({"violations": ["ヘルメット未着用"], "camera_id": "cam01"})
    planner.report_task_complete(tasks[0].task_id, success=True)
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    MONITOR = "monitor"
    CHECK_PPE = "check_ppe"
    ALERT = "alert"
    INVESTIGATE = "investigate"
    CORDON = "cordon"


class TaskStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class GoalPriority(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SubTask:
    """A single atomic action within a plan.

    Attributes
    ----------
    task_id:
        Unique identifier.
    task_type:
        Action category (monitor, check_ppe, alert, …).
    description_ja:
        Human-readable Japanese description.
    camera_id:
        Target camera, or 'all'.
    deadline_absolute:
        Unix timestamp by which this task should complete.
    status:
        Current execution status.
    priority:
        Inherited from parent plan.
    result:
        Optional result dict set when completed.
    """

    task_id: str
    plan_id: str
    task_type: TaskType
    description_ja: str
    camera_id: str
    deadline_absolute: float
    status: TaskStatus = TaskStatus.PENDING
    priority: int = GoalPriority.MEDIUM
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    result: dict[str, Any] | None = None

    def is_expired(self, now: float | None = None) -> bool:
        t = now or time.time()
        return t > self.deadline_absolute and self.status in (
            TaskStatus.PENDING, TaskStatus.ACTIVE
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "plan_id": self.plan_id,
            "task_type": self.task_type.value,
            "description_ja": self.description_ja,
            "camera_id": self.camera_id,
            "deadline_absolute": self.deadline_absolute,
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
        }


@dataclass
class Plan:
    """A long-horizon plan decomposed into subtasks.

    Attributes
    ----------
    plan_id:
        Unique identifier.
    goal_description:
        Original goal text.
    priority:
        Goal priority (1=low … 4=critical).
    camera_ids:
        Cameras in scope.
    horizon_seconds:
        Planning horizon from creation.
    tasks:
        Ordered list of SubTasks.
    status:
        Overall plan status.
    progress:
        Fraction of tasks completed [0, 1].
    """

    plan_id: str
    goal_description: str
    priority: int
    camera_ids: list[str]
    horizon_seconds: float
    tasks: list[SubTask] = field(default_factory=list)
    status: PlanStatus = PlanStatus.RUNNING
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    completion_notes: str = ""

    @property
    def deadline(self) -> float:
        return self.created_at + self.horizon_seconds

    @property
    def progress(self) -> float:
        if not self.tasks:
            return 0.0
        done = sum(
            1 for t in self.tasks
            if t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
        )
        return done / len(self.tasks)

    @property
    def active_tasks(self) -> list[SubTask]:
        return [t for t in self.tasks if t.status == TaskStatus.ACTIVE]

    @property
    def pending_tasks(self) -> list[SubTask]:
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def is_expired(self, now: float | None = None) -> bool:
        return (now or time.time()) > self.deadline

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal_description": self.goal_description,
            "priority": self.priority,
            "camera_ids": self.camera_ids,
            "horizon_seconds": self.horizon_seconds,
            "status": self.status.value,
            "progress": round(self.progress, 3),
            "task_count": len(self.tasks),
            "completed_tasks": sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED),
            "active_tasks": len(self.active_tasks),
            "pending_tasks": len(self.pending_tasks),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "deadline": self.deadline,
            "completion_notes": self.completion_notes,
            "tasks": [t.to_dict() for t in self.tasks],
        }


# ---------------------------------------------------------------------------
# Goal decomposition rules
# ---------------------------------------------------------------------------

# Pattern (substring match in goal) → list of task templates
# Each template: (task_type, description_ja, deadline_fraction, priority_boost)
# deadline_fraction: fraction of horizon allocated to this task

_DECOMPOSITION_RULES: list[tuple[list[str], list[tuple[TaskType, str, float, int]]]] = [
    # PPE compliance goals
    (
        ["ppe", "ヘルメット", "helmet", "安全装備", "保護具"],
        [
            (TaskType.MONITOR, "カメラ映像の継続的モニタリングを開始", 0.1, 0),
            (TaskType.CHECK_PPE, "全検出人物のPPEコンプライアンスを確認", 0.3, 1),
            (TaskType.ALERT, "PPE未着用者を発見した場合は即時アラートを送信", 0.5, 1),
            (TaskType.INVESTIGATE, "過去24時間のPPE違反履歴を照会", 0.8, 0),
        ],
    ),
    # Zone / intrusion goals
    (
        ["侵入", "立入", "zone", "エリア", "restricted"],
        [
            (TaskType.CORDON, "制限エリアの境界マーカーを設定", 0.05, 1),
            (TaskType.MONITOR, "制限エリアの継続監視を有効化", 0.15, 0),
            (TaskType.ALERT, "侵入検知時に即時アラートを発報", 0.4, 2),
            (TaskType.INVESTIGATE, "侵入パターンの因果連鎖を分析", 0.7, 0),
        ],
    ),
    # Safety score maximization goals
    (
        ["安全スコア", "safety score", "最大化", "improve", "向上"],
        [
            (TaskType.MONITOR, "全カメラの安全スコア継続モニタリング", 0.1, 0),
            (TaskType.CHECK_PPE, "PPEコンプライアンス全数チェック", 0.25, 1),
            (TaskType.INVESTIGATE, "過去の違反パターンから根本原因を特定", 0.5, 0),
            (TaskType.ALERT, "スコア低下時にオペレーターへ通知", 0.6, 1),
            (TaskType.INVESTIGATE, "改善介入の効果を因果推論で評価", 0.9, 0),
        ],
    ),
    # Generic / catch-all
    (
        [],  # matches any goal
        [
            (TaskType.MONITOR, "対象カメラの監視を開始", 0.1, 0),
            (TaskType.ALERT, "異常検知時にアラートを発報", 0.5, 0),
            (TaskType.INVESTIGATE, "インシデントの根本原因を調査", 0.8, 0),
        ],
    ),
]


def _decompose_goal(
    plan_id: str,
    goal: str,
    priority: int,
    camera_ids: list[str],
    horizon_seconds: float,
    created_at: float,
) -> list[SubTask]:
    """Match the goal text against decomposition rules and generate subtasks."""
    goal_l = goal.lower()
    tasks: list[SubTask] = []

    # Find first matching rule
    matched_templates: list[tuple[TaskType, str, float, int]] | None = None
    for keywords, templates in _DECOMPOSITION_RULES:
        if not keywords or any(kw in goal_l for kw in keywords):
            matched_templates = templates
            break

    if matched_templates is None:
        return tasks

    # Use "all" if multiple cameras; otherwise expand per camera for MONITOR/CHECK tasks
    primary_cam = camera_ids[0] if len(camera_ids) == 1 else "all"

    for task_type, desc_ja, dl_frac, prio_boost in matched_templates:
        # For per-camera tasks, one task per camera (up to 4)
        cameras_for_task = camera_ids[:4] if task_type in (TaskType.MONITOR, TaskType.CHECK_PPE) else [primary_cam]
        for cam in cameras_for_task:
            tid = str(uuid.uuid4())
            deadline = created_at + horizon_seconds * dl_frac
            cam_label = f"[{cam}]" if cam != "all" else "[全カメラ]"
            tasks.append(SubTask(
                task_id=tid,
                plan_id=plan_id,
                task_type=task_type,
                description_ja=f"{cam_label} {desc_ja}",
                camera_id=cam,
                deadline_absolute=deadline,
                status=TaskStatus.PENDING,
                priority=min(4, priority + prio_boost),
                created_at=created_at,
            ))

    return tasks


# ---------------------------------------------------------------------------
# SafetyPlanner
# ---------------------------------------------------------------------------


class SafetyPlanner:
    """Long-horizon goal planner for factory safety monitoring.

    Parameters
    ----------
    default_horizon_seconds:
        Default planning horizon when not specified per-goal.
    max_plans:
        Maximum concurrent plans to retain.
    causal_engine:
        Optional CausalInferenceEngine for intervention-aware planning.
    """

    def __init__(
        self,
        default_horizon_seconds: float = 3600.0,
        max_plans: int = 20,
        causal_engine: Any = None,
    ) -> None:
        self._default_horizon = default_horizon_seconds
        self._max_plans = max_plans
        self._causal_engine = causal_engine

        self._lock = threading.Lock()
        self._plans: deque[Plan] = deque(maxlen=max_plans)
        self._task_index: dict[str, SubTask] = {}   # task_id → SubTask
        self._plan_index: dict[str, Plan] = {}       # plan_id → Plan
        self._tick_count = 0
        self._total_goals_submitted = 0
        self._total_tasks_completed = 0
        self._created_at = time.time()

        logger.info(
            "SafetyPlanner created (horizon=%.0fs, max_plans=%d, causal=%s)",
            default_horizon_seconds, max_plans, "yes" if causal_engine else "no",
        )

    # ── Goal submission ──────────────────────────────────────────────────────

    def submit_goal(
        self,
        description: str,
        priority: int = GoalPriority.MEDIUM,
        camera_ids: list[str] | None = None,
        horizon_seconds: float | None = None,
    ) -> str:
        """Decompose a natural-language safety goal into a plan.

        Parameters
        ----------
        description:
            Goal description (e.g., "倉庫エリアのPPEコンプライアンスを確保").
        priority:
            GoalPriority (1=low, 2=medium, 3=high, 4=critical).
        camera_ids:
            Cameras in scope.  Defaults to ["all"].
        horizon_seconds:
            Planning horizon.  Defaults to `default_horizon_seconds`.

        Returns
        -------
        str
            Plan ID.
        """
        plan_id = str(uuid.uuid4())
        now = time.time()
        cams = camera_ids or ["all"]
        horizon = horizon_seconds or self._default_horizon
        prio = int(priority)

        tasks = _decompose_goal(plan_id, description, prio, cams, horizon, now)

        plan = Plan(
            plan_id=plan_id,
            goal_description=description,
            priority=prio,
            camera_ids=cams,
            horizon_seconds=horizon,
            tasks=tasks,
            status=PlanStatus.RUNNING,
            created_at=now,
        )

        with self._lock:
            self._plans.append(plan)
            self._plan_index[plan_id] = plan
            for t in tasks:
                self._task_index[t.task_id] = t
            self._total_goals_submitted += 1

        logger.info(
            "SafetyPlanner: goal submitted plan=%s prio=%d tasks=%d horizon=%.0fs '%s'",
            plan_id[:8], prio, len(tasks), horizon, description[:60],
        )
        return plan_id

    # ── Tick (called each frame by engine Stage 6r) ──────────────────────────

    def tick(self, observations: dict[str, Any]) -> list[SubTask]:
        """Advance the planner by one step.

        Activates pending tasks whose deadlines are approaching, expires
        overdue tasks, and marks plans as complete or expired.

        Parameters
        ----------
        observations:
            Dict with keys "violations" (list[str]), "camera_id" (str),
            "timestamp" (float, optional).

        Returns
        -------
        list[SubTask]
            Newly activated tasks for this tick.
        """
        now = observations.get("timestamp") or time.time()
        camera_id = observations.get("camera_id", "all")
        violations: list[str] = observations.get("violations", [])

        newly_active: list[SubTask] = []

        with self._lock:
            self._tick_count += 1
            for plan in list(self._plans):
                if plan.status != PlanStatus.RUNNING:
                    continue
                if plan.is_expired(now):
                    _expire_plan(plan, now)
                    continue

                # Activate pending tasks whose deadline is within next 10% of horizon
                activation_threshold = plan.created_at + plan.horizon_seconds * 0.1
                for task in plan.pending_tasks:
                    if now >= activation_threshold or now >= task.deadline_absolute - plan.horizon_seconds * 0.05:
                        task.status = TaskStatus.ACTIVE
                        newly_active.append(task)

                # Auto-complete MONITOR tasks when no recent violations for this camera
                for task in plan.active_tasks:
                    if task.task_type == TaskType.MONITOR and task.is_expired(now):
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = now
                        task.result = {"auto_completed": True, "reason": "deadline_reached"}

                    # Auto-complete ALERT tasks if violation matches
                    elif task.task_type == TaskType.ALERT and violations:
                        if task.camera_id in ("all", camera_id):
                            task.status = TaskStatus.COMPLETED
                            task.completed_at = now
                            task.result = {"triggered": True, "violations": violations[:3]}
                            self._total_tasks_completed += 1

                # Check if plan is fully resolved
                if plan.progress >= 1.0:
                    plan.status = PlanStatus.COMPLETED
                    plan.completed_at = now
                    plan.completion_notes = "All tasks resolved."
                    logger.info("SafetyPlanner: plan %s completed (100%%)", plan.plan_id[:8])

        # Use causal engine to optionally prioritise follow-up tasks
        if violations and self._causal_engine is not None:
            try:
                for viol in violations[:2]:
                    self._causal_engine.observe(viol, timestamp=now, camera_id=camera_id)
            except Exception:
                pass

        return newly_active

    # ── Task completion reporting ─────────────────────────────────────────────

    def report_task_complete(
        self,
        task_id: str,
        success: bool = True,
        result: dict[str, Any] | None = None,
    ) -> bool:
        """Mark a subtask as completed or failed.

        Parameters
        ----------
        task_id:
            Subtask identifier.
        success:
            True = completed; False = failed.
        result:
            Optional result data.

        Returns
        -------
        bool
            True if the task was found and updated.
        """
        with self._lock:
            task = self._task_index.get(task_id)
            if task is None:
                return False
            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            task.completed_at = time.time()
            task.result = result or {}
            if success:
                self._total_tasks_completed += 1

            # Check plan completion
            plan = self._plan_index.get(task.plan_id)
            if plan and plan.status == PlanStatus.RUNNING and plan.progress >= 1.0:
                plan.status = PlanStatus.COMPLETED
                plan.completed_at = task.completed_at
                plan.completion_notes = "All tasks resolved."

        logger.info(
            "SafetyPlanner: task %s → %s",
            task_id[:8],
            "completed" if success else "failed",
        )
        return True

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_plan(self, plan_id: str) -> Plan | None:
        """Return a plan by ID."""
        with self._lock:
            return self._plan_index.get(plan_id)

    def get_active_plans(self) -> list[Plan]:
        """Return all currently running plans, newest first."""
        with self._lock:
            return [p for p in reversed(self._plans) if p.status == PlanStatus.RUNNING]

    def get_all_plans(self, limit: int = 20) -> list[Plan]:
        """Return all plans (newest first) up to limit."""
        with self._lock:
            plans = list(reversed(self._plans))
        return plans[:limit]

    def get_pending_tasks(self, camera_id: str | None = None) -> list[SubTask]:
        """Return pending/active tasks, optionally filtered by camera."""
        with self._lock:
            tasks = [t for t in self._task_index.values()
                     if t.status in (TaskStatus.PENDING, TaskStatus.ACTIVE)]
        if camera_id:
            tasks = [t for t in tasks if t.camera_id in ("all", camera_id)]
        tasks.sort(key=lambda t: (-t.priority, t.deadline_absolute))
        return tasks

    def cancel_plan(self, plan_id: str) -> bool:
        """Cancel a running plan and mark all its pending tasks as skipped."""
        with self._lock:
            plan = self._plan_index.get(plan_id)
            if plan is None:
                return False
            plan.status = PlanStatus.FAILED
            plan.completed_at = time.time()
            plan.completion_notes = "Cancelled by operator."
            for t in plan.tasks:
                if t.status in (TaskStatus.PENDING, TaskStatus.ACTIVE):
                    t.status = TaskStatus.SKIPPED
        return True

    def get_stats(self) -> dict[str, Any]:
        """Return operational statistics."""
        with self._lock:
            n_running = sum(1 for p in self._plans if p.status == PlanStatus.RUNNING)
            n_completed = sum(1 for p in self._plans if p.status == PlanStatus.COMPLETED)
            n_expired = sum(1 for p in self._plans if p.status == PlanStatus.EXPIRED)
            n_pending_tasks = sum(
                1 for t in self._task_index.values()
                if t.status == TaskStatus.PENDING
            )
            n_active_tasks = sum(
                1 for t in self._task_index.values()
                if t.status == TaskStatus.ACTIVE
            )
        return {
            "total_goals_submitted": self._total_goals_submitted,
            "total_tasks_completed": self._total_tasks_completed,
            "plans_running": n_running,
            "plans_completed": n_completed,
            "plans_expired": n_expired,
            "tasks_pending": n_pending_tasks,
            "tasks_active": n_active_tasks,
            "tick_count": self._tick_count,
            "default_horizon_seconds": self._default_horizon,
            "causal_engine_linked": self._causal_engine is not None,
            "uptime_seconds": round(time.time() - self._created_at, 1),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expire_plan(plan: Plan, now: float) -> None:
    plan.status = PlanStatus.EXPIRED
    plan.completed_at = now
    plan.completion_notes = "Planning horizon exceeded."
    for t in plan.tasks:
        if t.status in (TaskStatus.PENDING, TaskStatus.ACTIVE):
            t.status = TaskStatus.SKIPPED
    logger.info("SafetyPlanner: plan %s expired", plan.plan_id[:8])
