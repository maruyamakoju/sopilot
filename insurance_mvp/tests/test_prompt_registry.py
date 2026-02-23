"""Tests for prompt versioning registry.

Validates immutable versioning, content-addressed hashing, A/B evaluation,
and prompt diff functionality.
"""

import pytest

from insurance_mvp.cosmos.prompt_registry import (
    ABEvaluationReport,
    PromptDiff,
    PromptRegistry,
    PromptSection,
    PromptVersion,
    create_v1_baseline,
    create_v2_near_miss_priority,
    diff_prompts,
    get_registry,
)


class TestPromptVersion:
    def test_version_creation(self):
        v = PromptVersion(
            version_id="test-v1",
            description="Test",
            sections=(
                PromptSection(name="intro", content="Hello", position=0),
            ),
        )
        assert v.version_id == "test-v1"
        assert len(v.sections) == 1

    def test_render_orders_by_position(self):
        v = PromptVersion(
            version_id="test",
            description="",
            sections=(
                PromptSection(name="second", content="World", position=20),
                PromptSection(name="first", content="Hello", position=10),
            ),
        )
        rendered = v.render()
        assert rendered.index("Hello") < rendered.index("World")

    def test_content_hash_deterministic(self):
        v1 = create_v1_baseline()
        v2 = create_v1_baseline()
        assert v1.content_hash == v2.content_hash

    def test_content_hash_changes_with_content(self):
        v1 = create_v1_baseline()
        v2 = create_v2_near_miss_priority()
        assert v1.content_hash != v2.content_hash

    def test_get_section(self):
        v = create_v1_baseline()
        section = v.get_section("observe")
        assert section is not None
        assert "OBSERVE" in section.content

    def test_get_section_not_found(self):
        v = create_v1_baseline()
        assert v.get_section("nonexistent") is None

    def test_has_tag(self):
        v = create_v2_near_miss_priority()
        assert v.has_tag("near_miss")
        assert v.has_tag("negative_example")
        assert not v.has_tag("nonexistent")

    def test_frozen_immutability(self):
        v = create_v1_baseline()
        with pytest.raises(AttributeError):
            v.version_id = "modified"


class TestPromptDiff:
    def test_no_changes_same_version(self):
        v1 = create_v1_baseline()
        diff = diff_prompts(v1, v1)
        assert not diff.has_changes

    def test_detects_modifications(self):
        v1 = create_v1_baseline()
        v2 = create_v2_near_miss_priority()
        diff = diff_prompts(v1, v2)
        assert diff.has_changes
        # v2 modifies existing sections
        assert len(diff.modified_sections) > 0

    def test_detects_added_sections(self):
        v1 = create_v1_baseline()
        v2 = create_v2_near_miss_priority()
        diff = diff_prompts(v1, v2)
        # v2 adds "output_order" section
        assert "output_order" in diff.added_sections


class TestPromptRegistry:
    def test_register_and_get(self):
        registry = PromptRegistry()
        v1 = create_v1_baseline()
        registry.register(v1)
        retrieved = registry.get(v1.version_id)
        assert retrieved.version_id == v1.version_id

    def test_register_idempotent(self):
        registry = PromptRegistry()
        v1 = create_v1_baseline()
        registry.register(v1)
        registry.register(v1)  # Should not raise
        assert len(registry.list_versions()) == 1

    def test_register_conflict_raises(self):
        registry = PromptRegistry()
        v1 = create_v1_baseline()
        registry.register(v1)
        # Same ID but different content
        v_conflict = PromptVersion(
            version_id=v1.version_id,
            description="Different",
            sections=(PromptSection(name="x", content="different", position=0),),
        )
        with pytest.raises(ValueError, match="already exists"):
            registry.register(v_conflict)

    def test_get_unknown_raises(self):
        registry = PromptRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_versions(self):
        registry = PromptRegistry()
        registry.register(create_v1_baseline())
        registry.register(create_v2_near_miss_priority())
        versions = registry.list_versions()
        assert len(versions) == 2

    def test_diff_via_registry(self):
        registry = PromptRegistry()
        registry.register(create_v1_baseline())
        registry.register(create_v2_near_miss_priority())
        diff = registry.diff("v1-baseline", "v2-near-miss-priority")
        assert diff.has_changes

    def test_ab_evaluation(self):
        registry = PromptRegistry()
        registry.register(create_v1_baseline())
        registry.register(create_v2_near_miss_priority())

        y_true = ["HIGH", "MEDIUM", "NONE", "LOW", "MEDIUM"]
        y_pred_a = ["HIGH", "LOW", "NONE", "LOW", "LOW"]     # v1: 3/5 correct
        y_pred_b = ["HIGH", "MEDIUM", "NONE", "LOW", "MEDIUM"]  # v2: 5/5 correct

        report = registry.evaluate_ab(
            "v1-baseline", "v2-near-miss-priority",
            y_true, y_pred_a, y_pred_b,
        )
        assert isinstance(report, ABEvaluationReport)
        assert report.overall_winner == "B"


class TestDefaultRegistry:
    def test_get_registry_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_default_has_v1_and_v2(self):
        registry = get_registry()
        versions = registry.list_versions()
        assert "v1-baseline" in versions
        assert "v2-near-miss-priority" in versions

    def test_v2_has_near_miss_tag(self):
        registry = get_registry()
        v2 = registry.get("v2-near-miss-priority")
        assert v2.has_tag("near_miss")

    def test_v1_lacks_negative_example_tag(self):
        registry = get_registry()
        v1 = registry.get("v1-baseline")
        assert not v1.has_tag("negative_example")
