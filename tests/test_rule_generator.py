"""Tests for RuleGenerator (services/rule_generator.py) and RuleRepository."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sopilot.repositories.rule_repository import RuleRepository
from sopilot.services.rule_generator import (
    RuleGenerator,
    _deduplicate,
    _make_default_rules,
    _parse_vlm_response,
)


# ── RuleRepository tests ──────────────────────────────────────────────────────


def _make_connect():
    """In-memory SQLite connect factory for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    @contextmanager
    def connect():
        yield conn

    return connect


def test_rule_repo_save_and_get():
    repo = RuleRepository(_make_connect())
    count = repo.save_rules("task1", ["ヘルメット着用必須", "ベスト着用必須"])
    assert count == 2
    rules = repo.get_rules("task1")
    assert len(rules) == 2
    assert rules[0]["rule"] == "ヘルメット着用必須"


def test_rule_repo_get_rule_strings():
    repo = RuleRepository(_make_connect())
    repo.save_rules("task1", ["ルールA", "ルールB"])
    strings = repo.get_rule_strings("task1")
    assert strings == ["ルールA", "ルールB"]


def test_rule_repo_save_replaces_existing():
    repo = RuleRepository(_make_connect())
    repo.save_rules("task1", ["old rule"])
    repo.save_rules("task1", ["new rule"])
    rules = repo.get_rule_strings("task1")
    assert rules == ["new rule"]


def test_rule_repo_save_empty():
    repo = RuleRepository(_make_connect())
    repo.save_rules("task1", ["rule"])
    repo.save_rules("task1", [])
    assert repo.get_rule_strings("task1") == []


def test_rule_repo_multiple_tasks_isolated():
    connect = _make_connect()
    repo = RuleRepository(connect)
    repo.save_rules("taskA", ["A1", "A2"])
    repo.save_rules("taskB", ["B1"])
    assert repo.get_rule_strings("taskA") == ["A1", "A2"]
    assert repo.get_rule_strings("taskB") == ["B1"]


def test_rule_repo_delete():
    repo = RuleRepository(_make_connect())
    repo.save_rules("task1", ["R1", "R2"])
    deleted = repo.delete_rules("task1")
    assert deleted == 2
    assert repo.get_rule_strings("task1") == []


def test_rule_repo_delete_nonexistent_task():
    repo = RuleRepository(_make_connect())
    deleted = repo.delete_rules("no-such-task")
    assert deleted == 0


def test_rule_repo_add_rule():
    repo = RuleRepository(_make_connect())
    repo.save_rules("task1", ["existing"])
    result = repo.add_rule("task1", "new manual rule", source="manual", confidence=0.9)
    assert result["rule"] == "new manual rule"
    assert result["source"] == "manual"
    rules = repo.get_rule_strings("task1")
    assert "existing" in rules
    assert "new manual rule" in rules


def test_rule_repo_confidence_stored():
    repo = RuleRepository(_make_connect())
    repo.save_rules("task1", ["R1"], confidences=[0.85])
    rows = repo.get_rules("task1")
    assert abs(rows[0]["confidence"] - 0.85) < 0.001


def test_rule_repo_source_stored():
    repo = RuleRepository(_make_connect())
    repo.save_rules("task1", ["R1"], source="manual")
    rows = repo.get_rules("task1")
    assert rows[0]["source"] == "manual"


# ── _parse_vlm_response tests ─────────────────────────────────────────────────


def test_parse_valid_json():
    text = json.dumps({
        "rules": [
            {"rule": "ヘルメット着用必須", "confidence": 0.9, "category": "ppe"},
            {"rule": "立入禁止エリア侵入禁止", "confidence": 0.8, "category": "zone"},
        ]
    })
    rules = _parse_vlm_response(text)
    assert len(rules) == 2
    assert rules[0]["rule"] == "ヘルメット着用必須"
    assert rules[0]["confidence"] == 0.9
    assert rules[0]["category"] == "ppe"


def test_parse_json_with_preamble():
    text = "以下がルールです：\n" + json.dumps({"rules": [{"rule": "安全確認必須", "confidence": 0.7, "category": "behavior"}]})
    rules = _parse_vlm_response(text)
    assert len(rules) == 1
    assert rules[0]["rule"] == "安全確認必須"


def test_parse_empty_rules_list():
    text = json.dumps({"rules": []})
    rules = _parse_vlm_response(text)
    assert rules == []


def test_parse_malformed_json_fallback():
    text = "- ヘルメット着用必須\n- 手袋着用必須\n"
    rules = _parse_vlm_response(text)
    # Fallback may or may not extract — just ensure no crash
    assert isinstance(rules, list)


def test_parse_missing_rule_key_skipped():
    text = json.dumps({"rules": [{"confidence": 0.5, "category": "ppe"}, {"rule": "有効ルール", "confidence": 0.8, "category": "ppe"}]})
    rules = _parse_vlm_response(text)
    assert len(rules) == 1
    assert rules[0]["rule"] == "有効ルール"


# ── _deduplicate tests ────────────────────────────────────────────────────────


def test_deduplicate_removes_exact_duplicates():
    rules = [
        {"rule": "ヘルメット着用必須", "confidence": 0.9, "category": "ppe"},
        {"rule": "ヘルメット着用必須", "confidence": 0.7, "category": "ppe"},
    ]
    result = _deduplicate(rules)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.9  # keeps highest


def test_deduplicate_case_insensitive():
    rules = [
        {"rule": "helmet required", "confidence": 0.9, "category": "ppe"},
        {"rule": "HELMET REQUIRED", "confidence": 0.6, "category": "ppe"},
    ]
    result = _deduplicate(rules)
    assert len(result) == 1


def test_deduplicate_distinct_rules_preserved():
    rules = [
        {"rule": "ルールA", "confidence": 0.8, "category": "ppe"},
        {"rule": "ルールB", "confidence": 0.7, "category": "zone"},
    ]
    result = _deduplicate(rules)
    assert len(result) == 2


# ── _make_default_rules tests ─────────────────────────────────────────────────


def test_make_default_rules_returns_list():
    rules = _make_default_rules()
    assert isinstance(rules, list)
    assert len(rules) > 0


def test_make_default_rules_have_required_keys():
    rules = _make_default_rules()
    for r in rules:
        assert "rule" in r
        assert "confidence" in r
        assert "category" in r


# ── RuleGenerator tests ───────────────────────────────────────────────────────


def test_rule_generator_no_vlm_returns_defaults():
    gen = RuleGenerator(vlm_callable=None)
    # Simulate no video by using a non-existent path
    rules = gen.generate("/no/such/video.mp4")
    assert isinstance(rules, list)
    assert len(rules) > 0


def test_rule_generator_no_frames_returns_defaults():
    gen = RuleGenerator(vlm_callable=None, n_frames=5)
    with patch.object(gen, "_sample_frames", return_value=[]):
        rules = gen.generate("/fake/video.mp4")
    assert len(rules) > 0  # defaults


def test_rule_generator_with_mock_vlm():
    vlm_response = json.dumps({
        "rules": [
            {"rule": "安全ベスト着用必須", "confidence": 0.85, "category": "ppe"},
            {"rule": "立入禁止エリア侵入禁止", "confidence": 0.9, "category": "zone"},
        ]
    })
    mock_vlm = MagicMock(return_value=vlm_response)
    gen = RuleGenerator(vlm_callable=mock_vlm, n_frames=2)

    fake_jpeg = b"\xff\xd8\xff" + b"\x00" * 100  # minimal JPEG header
    with patch.object(gen, "_sample_frames", return_value=[fake_jpeg, fake_jpeg]):
        rules = gen.generate("/fake/video.mp4")

    assert len(rules) > 0
    rule_texts = [r["rule"] for r in rules]
    assert "安全ベスト着用必須" in rule_texts


def test_rule_generator_vlm_exception_returns_defaults():
    mock_vlm = MagicMock(side_effect=RuntimeError("VLM failed"))
    gen = RuleGenerator(vlm_callable=mock_vlm, n_frames=1)

    fake_jpeg = b"\xff\xd8\xff" + b"\x00" * 100
    with patch.object(gen, "_sample_frames", return_value=[fake_jpeg]):
        rules = gen.generate("/fake/video.mp4")

    assert isinstance(rules, list)  # defaults returned
    assert len(rules) > 0


def test_rule_generator_deduplicates():
    duplicated_response = json.dumps({
        "rules": [
            {"rule": "重複ルール", "confidence": 0.9, "category": "ppe"},
            {"rule": "重複ルール", "confidence": 0.6, "category": "ppe"},
        ]
    })
    mock_vlm = MagicMock(return_value=duplicated_response)
    gen = RuleGenerator(vlm_callable=mock_vlm, n_frames=1)
    fake_jpeg = b"\xff\xd8\xff" + b"\x00" * 100
    with patch.object(gen, "_sample_frames", return_value=[fake_jpeg]):
        rules = gen.generate("/fake/video.mp4")
    # Should be deduplicated
    rule_texts = [r["rule"] for r in rules]
    assert rule_texts.count("重複ルール") == 1
