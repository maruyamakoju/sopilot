"""Shared utilities for parsing Video-LLM structured output.

Centralizes JSON extraction from LLM responses, handling common issues:
- Markdown code fences (```json ... ```)
- JSON embedded in surrounding text
- Graceful fallback on parse failure
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def parse_llm_json(text: str, fallback: dict | None = None) -> dict:
    """Parse LLM output as JSON, handling common formatting issues.

    Tries in order:
    1. Direct JSON parse (after stripping markdown fences)
    2. Find JSON object boundaries ({...}) in the text
    3. Return fallback dict if all parsing fails

    Args:
        text: Raw LLM output string.
        fallback: Default dict to return on parse failure.
            If None, returns {"_parse_error": True, "_raw": text[:200]}.

    Returns:
        Parsed dict, or the fallback.
    """
    if fallback is None:
        fallback = {"_parse_error": True, "_raw": text[:200]}

    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass

    logger.warning("Failed to parse LLM JSON: %s", text[:100])
    return fallback
