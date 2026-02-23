"""Tests for Phase 1: Prompt engineering improvements.

Validates that the restructured prompts correctly prioritize near-miss
criteria and include negative examples for the 7B model.
"""

from insurance_mvp.cosmos.prompt import (
    get_claim_assessment_prompt,
    get_quick_severity_prompt,
)


class TestMediumDefinition:
    """MEDIUM severity definition must lead with near-miss criteria."""

    def test_medium_includes_near_miss_first(self):
        """Near-miss criteria should appear before collision in MEDIUM block."""
        prompt = get_claim_assessment_prompt()
        medium_start = prompt.index("MEDIUM")
        # Find the next severity block (HIGH)
        high_start = prompt.index("HIGH -Serious", medium_start)
        medium_block = prompt[medium_start:high_start]

        near_miss_pos = medium_block.index("Near-miss")
        collision_pos = medium_block.index("collision at moderate speed")
        assert near_miss_pos < collision_pos, (
            "Near-miss criteria must appear before collision criteria in MEDIUM definition"
        )

    def test_medium_header_mentions_near_miss(self):
        """MEDIUM header should explicitly mention near-miss."""
        prompt = get_claim_assessment_prompt()
        assert "MEDIUM -Moderate incident OR Near-miss:" in prompt

    def test_critical_near_miss_rule_present(self):
        """Bold CRITICAL rule about near-miss = MEDIUM must be present."""
        prompt = get_claim_assessment_prompt()
        assert "CRITICAL: A near-miss with emergency braking IS MEDIUM severity" in prompt


class TestNegativeExamples:
    """Negative examples prevent common 7B model errors."""

    def test_negative_example_wrong_reasoning(self):
        """WRONG reasoning example must be present."""
        prompt = get_claim_assessment_prompt()
        assert "WRONG reasoning" in prompt

    def test_negative_example_not_low(self):
        """'do NOT' instruction for near-miss must be present."""
        prompt = get_claim_assessment_prompt()
        assert "Do NOT classify a near-miss as LOW" in prompt

    def test_important_near_miss_rule(self):
        """IMPORTANT rule about emergency braking near pedestrian must exist."""
        prompt = get_claim_assessment_prompt()
        assert "IMPORTANT: If emergency braking occurs near a pedestrian" in prompt


class TestQuickSeverityPrompt:
    """QUICK_SEVERITY_PROMPT must cover near-miss scenarios."""

    def test_quick_prompt_medium_includes_near_miss(self):
        """MEDIUM in quick prompt must mention near-miss."""
        prompt = get_quick_severity_prompt()
        # Find the MEDIUM line
        lines = prompt.split("\n")
        medium_lines = [l for l in lines if "MEDIUM" in l]
        assert any("near-miss" in l for l in medium_lines), (
            f"MEDIUM in quick prompt must mention near-miss. Got: {medium_lines}"
        )

    def test_quick_prompt_medium_includes_pedestrian(self):
        """MEDIUM in quick prompt must mention pedestrian avoidance."""
        prompt = get_quick_severity_prompt()
        assert "pedestrian avoidance" in prompt.lower()


class TestSeverityFirstInstruction:
    """Output order instruction must be present."""

    def test_severity_first_instruction(self):
        """Prompt must instruct model to determine severity first."""
        prompt = get_claim_assessment_prompt()
        assert "OUTPUT ORDER: Determine severity FIRST" in prompt

    def test_severity_first_before_json_output(self):
        """Severity-first instruction must appear before 'Provide your JSON'."""
        prompt = get_claim_assessment_prompt()
        severity_first_pos = prompt.index("OUTPUT ORDER")
        json_output_pos = prompt.index("Provide your JSON assessment:")
        assert severity_first_pos < json_output_pos
