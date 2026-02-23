"""Tests for insurance_mvp.serialization module."""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from insurance_mvp.serialization import clip_to_dict, to_serializable
from pydantic import BaseModel


class TestToSerializable:
    def test_none(self):
        assert to_serializable(None) is None

    def test_primitives(self):
        assert to_serializable(42) == 42
        assert to_serializable(3.14) == 3.14
        assert to_serializable("hello") == "hello"
        assert to_serializable(True) is True

    def test_datetime(self):
        dt = datetime(2026, 1, 15, 10, 30, 0)
        result = to_serializable(dt)
        assert result == "2026-01-15T10:30:00"

    def test_set_sorted(self):
        result = to_serializable({"HIGH", "LOW", "MEDIUM"})
        assert result == ["HIGH", "LOW", "MEDIUM"]

    def test_list(self):
        result = to_serializable([1, "two", None])
        assert result == [1, "two", None]

    def test_tuple(self):
        result = to_serializable((1, 2, 3))
        assert result == [1, 2, 3]

    def test_dict(self):
        result = to_serializable({"key": "value", "num": 42})
        assert result == {"key": "value", "num": 42}

    def test_nested_dict_with_datetime(self):
        data = {"created": datetime(2026, 1, 1), "items": [1, 2]}
        result = to_serializable(data)
        assert result["created"] == "2026-01-01T00:00:00"
        assert result["items"] == [1, 2]

    def test_enum(self):
        class Color(str, Enum):
            RED = "red"
            BLUE = "blue"

        assert to_serializable(Color.RED) == "red"

    def test_dataclass(self):
        @dataclass
        class Point:
            x: float
            y: float

        result = to_serializable(Point(1.0, 2.0))
        assert result == {"x": 1.0, "y": 2.0}

    def test_pydantic_model(self):
        class Item(BaseModel):
            name: str
            value: int

        result = to_serializable(Item(name="test", value=42))
        assert result == {"name": "test", "value": 42}

    def test_nested_dataclass(self):
        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            inner: Inner
            items: list

        result = to_serializable(Outer(inner=Inner(value=5), items=[1, 2]))
        assert result == {"inner": {"value": 5}, "items": [1, 2]}

    def test_object_with_dict(self):
        class Custom:
            def __init__(self):
                self.name = "test"
                self.score = 0.5

        result = to_serializable(Custom())
        assert result["name"] == "test"
        assert result["score"] == 0.5

    def test_unknown_type_falls_back_to_str(self):
        result = to_serializable(complex(1, 2))
        assert result == "(1+2j)"

    def test_json_roundtrip(self):
        """Full roundtrip: complex object → serializable → JSON string → dict."""
        @dataclass
        class Assessment:
            severity: str
            confidence: float
            tags: set

        obj = Assessment(severity="HIGH", confidence=0.9, tags={"urgent", "auto"})
        serialized = to_serializable(obj)
        json_str = json.dumps(serialized)
        restored = json.loads(json_str)
        assert restored["severity"] == "HIGH"
        assert restored["confidence"] == 0.9
        assert set(restored["tags"]) == {"urgent", "auto"}


class TestClipToDict:
    def test_dict_passthrough(self):
        clip = {"clip_id": "c1", "score": 0.9}
        assert clip_to_dict(clip) is clip

    def test_dataclass_conversion(self):
        @dataclass
        class Clip:
            clip_id: str
            score: float

        result = clip_to_dict(Clip(clip_id="c1", score=0.8))
        assert result == {"clip_id": "c1", "score": 0.8}
