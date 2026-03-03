"""NL Task Specification — parse natural-language monitoring rules into structured tasks.

Supported task types:
  loiter          — entity stays in area longer than N seconds
  zone_entry      — entity enters a named zone
  approach        — entity gets within distance of a target
  count_threshold — N or more entities simultaneously
  time_filter     — only monitor during specific hours

Parser: pure regex, no VLM/external deps required.
"""

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

TASK_TYPES = ("loiter", "zone_entry", "approach", "count_threshold", "time_filter")

SEVERITY_MAP = {
    "緊急": "critical", "critical": "critical", "危険": "critical",
    "警告": "warning", "warn": "warning", "warning": "warning", "アラート": "warning",
    "通知": "info", "info": "info", "お知らせ": "info",
}

ENTITY_KEYWORDS = {
    "人": "person", "person": "person", "worker": "person", "作業者": "person",
    "車": "vehicle", "vehicle": "vehicle", "car": "vehicle",
    "機械": "equipment", "equipment": "equipment",
}


@dataclass
class NLTask:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    raw_text: str = ""
    task_type: str = "loiter"
    parameters: dict[str, Any] = field(default_factory=dict)
    active: bool = True
    created_at: float = field(default_factory=time.time)
    description_ja: str = ""
    description_en: str = ""
    severity: str = "warning"

    def to_dict(self) -> dict:
        return {
            "id": self.id, "raw_text": self.raw_text, "task_type": self.task_type,
            "parameters": self.parameters, "active": self.active,
            "created_at": self.created_at, "description_ja": self.description_ja,
            "description_en": self.description_en, "severity": self.severity,
        }


@dataclass
class TaskTrigger:
    task_id: str
    task_type: str
    entity_id: int
    entity_label: str
    description_ja: str
    severity: str


class NLTaskParser:
    """Regex-based NL task parser. No external dependencies."""
    _DURATION_MIN_RE = re.compile(r'(\d+)\s*分', re.IGNORECASE)
    _DURATION_SEC_RE = re.compile(r'(\d+)\s*秒', re.IGNORECASE)
    _ZONE_RE = re.compile(
        r'[Zz]one[\s　]*([A-Za-z0-9]+)|エリア[\s　]*([A-Za-z0-9]+)|ゾーン[\s　]*([A-Za-z0-9]+)'
        r'|制限区域|立入禁止|立ち入り禁止', re.IGNORECASE)
    _ENTRY_RE = re.compile(r'入る|入っ|侵入|進入|enter|entered', re.IGNORECASE)
    _APPROACH_RE = re.compile(r'近づく|近づい|接近|approach', re.IGNORECASE)
    _COUNT_RE = re.compile(r'(\d+)\s*(人|台|体|名)(以上|より多)', re.IGNORECASE)
    _TIME_RE = re.compile(r'(\d{1,2})[時:：](?:\d{2})?\s*(?:〜|~|から|\-)\s*(\d{1,2})[時:：]', re.IGNORECASE)
    _SEV_RE = re.compile('|'.join(re.escape(k) for k in SEVERITY_MAP), re.IGNORECASE)
    _ENT_RE = re.compile('|'.join(re.escape(k) for k in ENTITY_KEYWORDS), re.IGNORECASE)

    def parse_rule(self, text: str) -> NLTask:
        t = text.strip()
        task = NLTask(raw_text=t)

        # Severity
        sev_m = self._SEV_RE.search(t)
        if sev_m:
            task.severity = SEVERITY_MAP.get(sev_m.group(0), "warning")

        # Entity type
        ent_m = self._ENT_RE.search(t)
        entity_type = ENTITY_KEYWORDS.get(ent_m.group(0), "") if ent_m else ""
        if entity_type:
            task.parameters["entity_type"] = entity_type

        # Zone
        zone_m = self._ZONE_RE.search(t)
        zone_name = ""
        if zone_m:
            for g in zone_m.groups():
                if g:
                    zone_name = f"Zone {g}"
                    break
            if not zone_name:
                zone_name = zone_m.group(0)
        if zone_name:
            task.parameters["zone"] = zone_name

        # Count threshold
        count_m = self._COUNT_RE.search(t)
        if count_m:
            task.task_type = "count_threshold"
            task.parameters["threshold"] = int(count_m.group(1))
            task.description_ja = f"{count_m.group(1)}{count_m.group(2)}以上同時検出で{task.severity}"
            task.description_en = f"Alert when {count_m.group(1)}+ {entity_type or 'entities'} detected"
            return task

        # Time filter (only if no entry/approach keywords)
        time_m = self._TIME_RE.search(t)
        if time_m and not self._ENTRY_RE.search(t) and not self._APPROACH_RE.search(t):
            task.task_type = "time_filter"
            task.parameters["start_hour"] = int(time_m.group(1))
            task.parameters["end_hour"] = int(time_m.group(2))
            task.description_ja = f"{time_m.group(1)}時〜{time_m.group(2)}時の間のみ監視"
            task.description_en = f"Monitor only {time_m.group(1)}:00–{time_m.group(2)}:00"
            return task

        # Zone entry
        if self._ENTRY_RE.search(t):
            task.task_type = "zone_entry"
            task.description_ja = f"{zone_name or '指定エリア'}への侵入を検出"
            task.description_en = f"Detect entry into {zone_name or 'specified zone'}"
            return task

        # Approach
        if self._APPROACH_RE.search(t):
            task.task_type = "approach"
            dist_m = re.search(r'(\d+(?:\.\d+)?)\s*m(?:eter)?', t, re.IGNORECASE)
            task.parameters["distance_threshold"] = float(dist_m.group(1)) if dist_m else 0.15
            task.description_ja = f"{zone_name or '対象'}への接近を検出"
            task.description_en = f"Detect approach to {zone_name or 'target'}"
            return task

        # Default: loiter
        task.task_type = "loiter"
        min_m = self._DURATION_MIN_RE.search(t)
        sec_m = self._DURATION_SEC_RE.search(t)
        duration_s = 120.0
        if min_m:
            duration_s = float(min_m.group(1)) * 60
        elif sec_m:
            duration_s = float(sec_m.group(1))
        task.parameters["duration_seconds"] = duration_s
        task.parameters["area_radius"] = 0.10

        dur_text = (f"{int(min_m.group(1))}分" if min_m
                    else (f"{int(sec_m.group(1))}秒" if sec_m else "2分"))
        task.description_ja = f"{zone_name or 'エリア'}で{dur_text}以上滞留したら{task.severity}"
        task.description_en = f"Alert if entity loiters >{duration_s:.0f}s in {zone_name or 'any area'}"
        return task


class NLTaskManager:
    """Manages active NL tasks and checks entities against them."""

    def __init__(self):
        self._tasks: dict[str, NLTask] = {}
        self._parser = NLTaskParser()
        # Loiter tracking: entity_id → (start_time, start_pos)
        self._loiter_start: dict[int, tuple[float, tuple[float, float]]] = {}

    def add_task(self, task: NLTask) -> NLTask:
        self._tasks[task.id] = task
        return task

    def parse_and_add(self, text: str) -> NLTask:
        return self.add_task(self._parser.parse_rule(text))

    def remove_task(self, task_id: str) -> bool:
        return self._tasks.pop(task_id, None) is not None

    def get_tasks(self) -> list[NLTask]:
        return list(self._tasks.values())

    def get_task(self, task_id: str) -> NLTask | None:
        return self._tasks.get(task_id)

    def set_active(self, task_id: str, active: bool) -> bool:
        t = self._tasks.get(task_id)
        if t:
            t.active = active
            return True
        return False

    def check_entity(self, entity_id: int, entity_label: str,
                     position: tuple[float, float],
                     current_time: float | None = None) -> list[TaskTrigger]:
        """Check entity against all active loiter tasks. Returns triggered TaskTriggers."""
        if current_time is None:
            current_time = time.time()
        triggers = []
        entity_type = self._normalize_label(entity_label)
        for task in self._tasks.values():
            if not task.active:
                continue
            if task.parameters.get("entity_type") and entity_type != task.parameters["entity_type"]:
                continue
            if task.task_type == "loiter":
                t = self._check_loiter(task, entity_id, entity_label, position, current_time)
                if t:
                    triggers.append(t)
        return triggers

    def update_entity_counts(self, counts: dict[str, int]) -> list[TaskTrigger]:
        """Check count_threshold tasks against current entity counts."""
        triggers = []
        for task in self._tasks.values():
            if not task.active or task.task_type != "count_threshold":
                continue
            threshold = task.parameters.get("threshold", 1)
            etype = task.parameters.get("entity_type", "")
            total = counts.get(etype, 0) if etype else sum(counts.values())
            if total >= threshold:
                triggers.append(TaskTrigger(
                    task_id=task.id, task_type="count_threshold",
                    entity_id=-1, entity_label=etype or "all",
                    description_ja=f"{total}{etype or 'エンティティ'}検出 (閾値: {threshold})",
                    severity=task.severity,
                ))
        return triggers

    def is_within_monitor_hours(self) -> bool:
        """Returns False if a time_filter task is active and current hour is outside range."""
        hour = time.localtime().tm_hour
        for task in self._tasks.values():
            if task.active and task.task_type == "time_filter":
                start = task.parameters.get("start_hour", 0)
                end = task.parameters.get("end_hour", 24)
                if not (start <= hour < end):
                    return False
        return True

    def reset(self) -> None:
        self._loiter_start.clear()

    def get_state_dict(self) -> dict:
        return {
            "task_count": len(self._tasks),
            "active_count": sum(1 for t in self._tasks.values() if t.active),
            "tasks": [t.to_dict() for t in self._tasks.values()],
        }

    def _normalize_label(self, label: str) -> str:
        label = label.lower()
        for kw, normalized in ENTITY_KEYWORDS.items():
            if kw in label:
                return normalized
        return label

    def _check_loiter(self, task: NLTask, entity_id: int, label: str,
                      pos: tuple[float, float], now: float) -> "TaskTrigger | None":
        duration_s = task.parameters.get("duration_seconds", 120.0)
        radius = task.parameters.get("area_radius", 0.10)
        if entity_id not in self._loiter_start:
            self._loiter_start[entity_id] = (now, pos)
            return None
        start_time, start_pos = self._loiter_start[entity_id]
        dist = ((pos[0] - start_pos[0]) ** 2 + (pos[1] - start_pos[1]) ** 2) ** 0.5
        if dist > radius:
            self._loiter_start[entity_id] = (now, pos)
            return None
        elapsed = now - start_time
        if elapsed >= duration_s:
            return TaskTrigger(
                task_id=task.id, task_type="loiter",
                entity_id=entity_id, entity_label=label,
                description_ja=f"エンティティ{entity_id}({label})が{int(elapsed)}秒間滞留",
                severity=task.severity,
            )
        return None
