"""Generic serialization helpers for Insurance MVP.

Provides ``to_serializable()`` — a recursive converter that turns
dataclasses, Pydantic models, enums, sets, datetimes and other
common types into plain JSON-compatible dicts/lists.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


def to_serializable(obj: Any) -> Any:
    """Recursively convert *obj* to a JSON-serializable structure.

    Handles: None, primitives, datetime, Enum, set, list/tuple,
    dict, dataclass, Pydantic BaseModel, and arbitrary objects with
    ``__dict__``.  Falls back to ``str(obj)`` for unknown types.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, BaseModel):
        return to_serializable(obj.model_dump())
    if hasattr(obj, "__dataclass_fields__"):
        return {
            field_name: to_serializable(getattr(obj, field_name))
            for field_name in obj.__dataclass_fields__
        }
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dict__"):
        return to_serializable(obj.__dict__)
    return str(obj)


def clip_to_dict(clip: Any) -> dict[str, Any]:
    """Convert a danger clip (dict, dataclass, or object) to a plain dict."""
    if isinstance(clip, dict):
        return clip
    return to_serializable(clip)
