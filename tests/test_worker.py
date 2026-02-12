from __future__ import annotations

from sopilot.worker import _parse_queues


class TestParseQueues:
    def test_all_keyword(self) -> None:
        result = _parse_queues("all", "sopilot")
        assert result == ["sopilot_ingest", "sopilot_score", "sopilot_training"]

    def test_star_keyword(self) -> None:
        result = _parse_queues("*", "sopilot")
        assert result == ["sopilot_ingest", "sopilot_score", "sopilot_training"]

    def test_empty_string(self) -> None:
        result = _parse_queues("", "sopilot")
        assert result == ["sopilot_ingest", "sopilot_score", "sopilot_training"]

    def test_single_canonical_queue(self) -> None:
        result = _parse_queues("ingest", "sopilot")
        assert result == ["sopilot_ingest"]

    def test_multiple_canonical_queues(self) -> None:
        result = _parse_queues("ingest,score", "myapp")
        assert result == ["myapp_ingest", "myapp_score"]

    def test_custom_prefix(self) -> None:
        result = _parse_queues("training", "custom_prefix")
        assert result == ["custom_prefix_training"]

    def test_explicit_queue_name_passthrough(self) -> None:
        result = _parse_queues("my_custom_queue", "sopilot")
        assert result == ["my_custom_queue"]

    def test_mixed_canonical_and_explicit(self) -> None:
        result = _parse_queues("ingest,my_custom_queue", "sopilot")
        assert result == ["sopilot_ingest", "my_custom_queue"]

    def test_case_insensitive(self) -> None:
        result = _parse_queues("INGEST,SCORE", "sopilot")
        assert result == ["sopilot_ingest", "sopilot_score"]

    def test_whitespace_handling(self) -> None:
        result = _parse_queues("  ingest , score  ", "sopilot")
        assert result == ["sopilot_ingest", "sopilot_score"]

    def test_empty_prefix_uses_empty(self) -> None:
        result = _parse_queues("ingest", "")
        assert result == ["_ingest"]
