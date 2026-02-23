"""Versioned prompt registry with A/B evaluation support.

Provides:
- Immutable, versioned prompt definitions with semantic metadata
- A/B evaluation harness for comparing prompt variants
- Prompt diff for regression detection
- Structured prompt composition (system + evidence rules + classification + output)

Design rationale:
  7B VLMs are highly sensitive to prompt ordering, emphasis placement, and
  negative examples. This registry enables controlled experimentation with
  prompt variants while maintaining full traceability for research papers.

References:
  Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in LLMs"
  Reynolds & McDonell (2021) "Prompt Programming for Large Language Models"
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt version definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptSection:
    """An immutable section of a structured prompt."""
    name: str
    content: str
    position: int  # Ordering within the prompt (lower = earlier)
    tags: tuple[str, ...] = ()  # e.g., ("near_miss", "negative_example")


@dataclass(frozen=True)
class PromptVersion:
    """Immutable, versioned prompt definition.

    Each version is content-addressed: the version_hash is computed from
    the concatenated sections, ensuring identical content produces identical
    hashes regardless of metadata changes.
    """
    version_id: str  # Semantic version e.g. "v2.1-near-miss-first"
    description: str
    sections: tuple[PromptSection, ...]
    created_at: str = ""
    author: str = ""

    def __post_init__(self):
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.utcnow().isoformat())

    @property
    def content_hash(self) -> str:
        """Content-addressed hash of prompt text."""
        full_text = self.render()
        return hashlib.sha256(full_text.encode()).hexdigest()[:12]

    def render(self) -> str:
        """Render the full prompt by concatenating sections in order."""
        sorted_sections = sorted(self.sections, key=lambda s: s.position)
        return "\n\n".join(s.content for s in sorted_sections)

    def get_section(self, name: str) -> PromptSection | None:
        """Get a section by name."""
        for s in self.sections:
            if s.name == name:
                return s
        return None

    def has_tag(self, tag: str) -> bool:
        """Check if any section has the given tag."""
        return any(tag in s.tags for s in self.sections)


# ---------------------------------------------------------------------------
# Prompt diff
# ---------------------------------------------------------------------------

@dataclass
class PromptDiff:
    """Differences between two prompt versions."""
    version_a: str
    version_b: str
    added_sections: list[str]
    removed_sections: list[str]
    modified_sections: list[str]
    tag_changes: dict[str, tuple[set[str], set[str]]]  # section -> (added_tags, removed_tags)

    @property
    def has_changes(self) -> bool:
        return bool(self.added_sections or self.removed_sections or self.modified_sections)


def diff_prompts(a: PromptVersion, b: PromptVersion) -> PromptDiff:
    """Compute structural diff between two prompt versions."""
    sections_a = {s.name: s for s in a.sections}
    sections_b = {s.name: s for s in b.sections}

    added = [name for name in sections_b if name not in sections_a]
    removed = [name for name in sections_a if name not in sections_b]
    modified = [
        name for name in sections_a
        if name in sections_b and sections_a[name].content != sections_b[name].content
    ]

    tag_changes = {}
    for name in sections_a:
        if name in sections_b:
            tags_a = set(sections_a[name].tags)
            tags_b = set(sections_b[name].tags)
            if tags_a != tags_b:
                tag_changes[name] = (tags_b - tags_a, tags_a - tags_b)

    return PromptDiff(
        version_a=a.version_id,
        version_b=b.version_id,
        added_sections=added,
        removed_sections=removed,
        modified_sections=modified,
        tag_changes=tag_changes,
    )


# ---------------------------------------------------------------------------
# A/B evaluation result
# ---------------------------------------------------------------------------

@dataclass
class ABResult:
    """Result of A/B comparison between two prompt versions."""
    prompt_a: str
    prompt_b: str
    metric_name: str
    value_a: float
    value_b: float
    delta: float  # b - a
    relative_delta: float  # (b - a) / a if a > 0
    winner: str  # "A", "B", or "TIE"
    n_samples: int


@dataclass
class ABEvaluationReport:
    """Complete A/B evaluation report."""
    prompt_a_id: str
    prompt_b_id: str
    results: list[ABResult]
    overall_winner: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class PromptRegistry:
    """Registry of versioned prompts with evaluation support.

    Usage:
        registry = PromptRegistry()
        registry.register(prompt_v1)
        registry.register(prompt_v2)

        # Compare
        diff = registry.diff("v1", "v2")

        # Get latest
        prompt = registry.get("v2")
        rendered = prompt.render()
    """

    def __init__(self):
        self._versions: dict[str, PromptVersion] = {}
        self._eval_history: list[ABEvaluationReport] = []

    def register(self, version: PromptVersion) -> None:
        """Register a prompt version."""
        if version.version_id in self._versions:
            existing = self._versions[version.version_id]
            if existing.content_hash != version.content_hash:
                raise ValueError(
                    f"Version '{version.version_id}' already exists with different content. "
                    "Use a new version_id for modified prompts."
                )
            return  # Idempotent for identical content
        self._versions[version.version_id] = version
        logger.info(
            "Registered prompt version: %s (hash=%s, sections=%d)",
            version.version_id, version.content_hash, len(version.sections),
        )

    def get(self, version_id: str) -> PromptVersion:
        """Get a prompt version by ID."""
        if version_id not in self._versions:
            raise KeyError(f"Prompt version '{version_id}' not found. Available: {list(self._versions.keys())}")
        return self._versions[version_id]

    def list_versions(self) -> list[str]:
        """List all registered version IDs."""
        return list(self._versions.keys())

    def diff(self, version_a_id: str, version_b_id: str) -> PromptDiff:
        """Compute diff between two versions."""
        return diff_prompts(self.get(version_a_id), self.get(version_b_id))

    def evaluate_ab(
        self,
        version_a_id: str,
        version_b_id: str,
        y_true: list[str],
        y_pred_a: list[str],
        y_pred_b: list[str],
        metrics: dict[str, Any] | None = None,
    ) -> ABEvaluationReport:
        """Run A/B evaluation between two prompt versions.

        Args:
            version_a_id: Prompt version A identifier.
            version_b_id: Prompt version B identifier.
            y_true: Ground truth severity labels.
            y_pred_a: Predictions using prompt A.
            y_pred_b: Predictions using prompt B.
            metrics: Optional pre-computed metrics dict {name: (val_a, val_b)}.

        Returns:
            ABEvaluationReport with per-metric comparison.
        """
        n = len(y_true)
        results = []

        if metrics is None:
            # Compute basic metrics
            acc_a = sum(t == p for t, p in zip(y_true, y_pred_a)) / n if n > 0 else 0
            acc_b = sum(t == p for t, p in zip(y_true, y_pred_b)) / n if n > 0 else 0
            metrics = {"accuracy": (acc_a, acc_b)}

        wins_a, wins_b = 0, 0
        for name, (val_a, val_b) in metrics.items():
            delta = val_b - val_a
            relative = delta / val_a if val_a > 0 else float("inf") if delta > 0 else 0.0
            winner = "B" if delta > 0.001 else "A" if delta < -0.001 else "TIE"
            if winner == "A":
                wins_a += 1
            elif winner == "B":
                wins_b += 1

            results.append(ABResult(
                prompt_a=version_a_id,
                prompt_b=version_b_id,
                metric_name=name,
                value_a=val_a,
                value_b=val_b,
                delta=delta,
                relative_delta=relative,
                winner=winner,
                n_samples=n,
            ))

        overall = "B" if wins_b > wins_a else "A" if wins_a > wins_b else "TIE"

        report = ABEvaluationReport(
            prompt_a_id=version_a_id,
            prompt_b_id=version_b_id,
            results=results,
            overall_winner=overall,
        )
        self._eval_history.append(report)
        return report

    def save_history(self, path: str) -> None:
        """Save evaluation history to JSON."""
        data = [asdict(r) for r in self._eval_history]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_history(self, path: str) -> None:
        """Load evaluation history from JSON."""
        with open(path) as f:
            data = json.load(f)
        # Store raw dicts (for inspection; not reconstructed into dataclasses)
        self._eval_history_raw = data


# ---------------------------------------------------------------------------
# Pre-built prompt versions for insurance severity classification
# ---------------------------------------------------------------------------

def create_v1_baseline() -> PromptVersion:
    """V1: Original prompt (near-miss buried in MEDIUM, no negative examples)."""
    return PromptVersion(
        version_id="v1-baseline",
        description="Original prompt with near-miss as 3rd bullet in MEDIUM",
        author="baseline",
        sections=(
            PromptSection(
                name="observe",
                content=(
                    "**STEP 1 -OBSERVE:** Carefully examine every frame. Note:\n"
                    "- Is there a collision or contact between vehicles?\n"
                    "- What is the speed at the moment of impact (if any)?\n"
                    "- Is there visible vehicle damage?\n"
                    "- Are pedestrians or cyclists involved?\n"
                    "- Are there near-miss events without actual contact?\n"
                    "- Is this just normal driving with no incident?"
                ),
                position=10,
                tags=("observation",),
            ),
            PromptSection(
                name="evidence_rules",
                content=(
                    "**CRITICAL EVIDENCE RULES:**\n"
                    "- If any frame shows text like 'COLLISION!' → severity is at minimum HIGH\n"
                    "- If speed drops from >40 km/h to 0 near another vehicle → rear-end collision (HIGH)\n"
                    "- If you see emergency braking with a pedestrian nearby → near-miss (MEDIUM)"
                ),
                position=20,
                tags=("rules",),
            ),
            PromptSection(
                name="classification",
                content=(
                    "**STEP 2 -CLASSIFY SEVERITY:**\n\n"
                    "NONE: Normal driving, no collision, no near-miss\n"
                    "LOW: Very low-speed contact, cosmetic damage only\n"
                    "MEDIUM: Clear collision at moderate speed; Visible vehicle damage; "
                    "Emergency braking to avoid collision (near-miss)\n"
                    "HIGH: High-speed collision; Severe damage; Pedestrian struck"
                ),
                position=30,
                tags=("classification",),
            ),
        ),
    )


def create_v2_near_miss_priority() -> PromptVersion:
    """V2: Near-miss criteria first in MEDIUM + negative examples + severity-first."""
    return PromptVersion(
        version_id="v2-near-miss-priority",
        description="Near-miss criteria first in MEDIUM, negative examples, severity-first instruction",
        author="prompt-engineering-phase1",
        sections=(
            PromptSection(
                name="observe",
                content=(
                    "**STEP 1 -OBSERVE:** Carefully examine every frame. Note:\n"
                    "- Is there a collision or contact between vehicles?\n"
                    "- What is the speed at the moment of impact (if any)?\n"
                    "- Is there visible vehicle damage?\n"
                    "- Are pedestrians or cyclists involved?\n"
                    "- Are there near-miss events without actual contact?\n"
                    "- Is this just normal driving with no incident?\n"
                    "- READ ALL TEXT OVERLAYS: speed, timestamps, alert text\n"
                    "- TRACK SPEED CHANGES: rapid drops indicate emergency braking"
                ),
                position=10,
                tags=("observation",),
            ),
            PromptSection(
                name="evidence_rules",
                content=(
                    "**CRITICAL EVIDENCE RULES:**\n"
                    "- If any frame shows text like 'COLLISION!' → severity is at minimum HIGH\n"
                    "- If speed drops from >40 km/h to 0 near another vehicle → rear-end collision (HIGH)\n"
                    "- If you see emergency braking with a pedestrian nearby → near-miss (MEDIUM)\n"
                    "- IMPORTANT: Emergency braking near a pedestrian/vehicle = NEAR-MISS → MEDIUM (NOT LOW)\n"
                    "- WRONG reasoning: 'No collision occurred so severity is LOW' — "
                    "near-miss with pedestrian avoidance is MEDIUM"
                ),
                position=20,
                tags=("rules", "negative_example", "near_miss"),
            ),
            PromptSection(
                name="classification",
                content=(
                    "**STEP 2 -CLASSIFY SEVERITY:**\n\n"
                    "NONE: Normal driving, no collision, no near-miss\n\n"
                    "LOW: Very low-speed contact (parking bump), cosmetic damage only, no injury risk\n\n"
                    "MEDIUM -Moderate incident OR Near-miss:\n"
                    "- Near-miss: Emergency braking near a pedestrian, cyclist, or vehicle — NO contact but CLOSE CALL\n"
                    "- Near-miss: Speed drops rapidly while another road user is nearby\n"
                    "- Swerving to avoid collision, vehicle stops within meters of hazard\n"
                    "- Clear collision at moderate speed with visible vehicle damage\n"
                    "**CRITICAL: A near-miss with emergency braking IS MEDIUM, even without collision. "
                    "Do NOT classify as LOW.**\n\n"
                    "HIGH: High-speed collision; Severe damage; Pedestrian struck; 'COLLISION!' text visible"
                ),
                position=30,
                tags=("classification", "near_miss"),
            ),
            PromptSection(
                name="output_order",
                content=(
                    "**OUTPUT ORDER: Determine severity FIRST, then fill in remaining fields. "
                    "Start your JSON with the severity field.**"
                ),
                position=40,
                tags=("output",),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Module-level registry instance
# ---------------------------------------------------------------------------

_default_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    """Get or create the default prompt registry with pre-built versions."""
    global _default_registry
    if _default_registry is None:
        _default_registry = PromptRegistry()
        _default_registry.register(create_v1_baseline())
        _default_registry.register(create_v2_near_miss_priority())
    return _default_registry
