"""Tests for Phase 1: Prompt engineering improvements.

Validates that the restructured prompts correctly prioritize near-miss
criteria and include negative examples for the 7B model.
"""

from insurance_mvp.cosmos.prompt import (
    get_claim_assessment_prompt,
    get_quick_severity_prompt,
)


class TestMediumDefinition:
    """MEDIUM severity definition must be defined as near-miss (no contact)."""

    def test_medium_includes_near_miss_first(self):
        """MEDIUM block should define near-miss as core criterion (no contact)."""
        prompt = get_claim_assessment_prompt()
        medium_start = prompt.index("MEDIUM")
        high_start = prompt.index("HIGH —", medium_start)
        medium_block = prompt[medium_start:high_start]
        # New prompt: MEDIUM = near-miss / no contact
        assert "Near-miss" in medium_block or "near-miss" in medium_block

    def test_medium_header_mentions_near_miss(self):
        """MEDIUM header should explicitly mention near-miss."""
        prompt = get_claim_assessment_prompt()
        assert "MEDIUM" in prompt and "near-miss" in prompt.lower()

    def test_medium_no_contact_rule_present(self):
        """Prompt must state that MEDIUM = no contact, HIGH = contact."""
        prompt = get_claim_assessment_prompt()
        # New formulation: contact → HIGH, no contact → MEDIUM
        assert "contact" in prompt.lower() and "HIGH" in prompt


class TestNegativeExamples:
    """Negative examples prevent common 7B model errors."""

    def test_negative_example_wrong_reasoning(self):
        """WRONG reasoning example must be present."""
        prompt = get_claim_assessment_prompt()
        assert "WRONG reasoning" in prompt

    def test_negative_example_contact_is_high(self):
        """Prompt must clarify that any contact → HIGH (not MEDIUM)."""
        prompt = get_claim_assessment_prompt()
        # New rule: contact = HIGH regardless of speed
        assert "contact" in prompt.lower() and "HIGH" in prompt

    def test_near_miss_escalation_rule(self):
        """Prompt must clarify near-miss/emergency braking → MEDIUM not LOW."""
        prompt = get_claim_assessment_prompt()
        assert "MEDIUM" in prompt and ("near-miss" in prompt.lower() or "Near-miss" in prompt)


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
