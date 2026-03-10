"""Unit tests for processing.parser module.

Tests cover:
- ResponseParser.parse_llm_response() — JSON marker splitting, empty response, parse failures
- ResponseParser.parse_entities_json() — valid JSON, invalid JSON, structure validation,
  entity creation (valid/invalid/mixed), logging
- ResponseParser.parse_batch_response() — multi-record splitting, fallback generation
- ResponseParser.format_csv_row() — CSV output formatting
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from ai_ner_system.processing.exceptions import LLMResponseError, ParseError
from ai_ner_system.processing.parser import ResponseParser

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / sample data
# ---------------------------------------------------------------------------

BREVID = "601"

# Mirrors ResponseParser._MAX_SNIPPET without importing the private constant
MAX_SNIPPET = 200

# Olauer;Person Name;N/A;1;601;Abbot;Male;non
# Ollum monnum þæim sæm þetta bref sea æder høyra sændir < Olauer;Person Name;N/A;1;601 > med gudz

VALID_ENTITY_DATA: dict[str, Any] = {
    "name": "Olauer",
    "type": "Person Name",
    "preposition": "N/A",
    "order": 1,
    "description": "Abbot",
    "gender": "Male",
    "language": "non",
}

VALID_ENTITIES_JSON = json.dumps(
    {
        "entities": [
            VALID_ENTITY_DATA,
        ]
    }
)

ANNOTATED_TEXT = "Ollum monnum þæim sæm þetta bref sea æder høyra sændir < Olauer;Person Name;N/A;1;601 > med gudz"

SAMPLE_LLM_RESPONSE = (
    f"{ANNOTATED_TEXT}\n{ResponseParser.JSON_MARKER}\n{VALID_ENTITIES_JSON}"
)

VALID_RECORD: dict[str, Any] = {
    "Bindnr": "1",
    "Brevid": BREVID,
    "Tekst": "Ollum monnum þæim sæm þetta bref sea æder høyra sændir Olauer med gudz",
}


# ===================================================================
# parse_llm_response
# ===================================================================
class TestParseLlmResponse:
    """Tests for ResponseParser.parse_llm_response()."""

    def test_with_json_marker(self) -> None:
        """Test response correctly split on ===JSON=== marker."""
        annotated_text, entities = ResponseParser.parse_llm_response(
            BREVID, SAMPLE_LLM_RESPONSE
        )

        log.debug("Annotated text: %s", annotated_text)
        log.debug("Entities: %s", entities)
        assert annotated_text == ANNOTATED_TEXT
        assert len(entities) == 1
        assert entities[0].name == VALID_ENTITY_DATA["name"]

    def test_without_json_marker(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test response without JSON marker falls back to empty entities."""
        raw_response = "Just some annotated text without JSON marker."

        with caplog.at_level(logging.WARNING):
            annotated_text, entities = ResponseParser.parse_llm_response(
                BREVID, raw_response
            )

        assert annotated_text == raw_response.strip()
        assert len(entities) == 0
        assert entities == []
        assert "No JSON marker found in response" in caplog.text

    def test_empty_response_raises(self) -> None:
        """Test empty response raises LLMResponseError."""
        with pytest.raises(
            LLMResponseError, match="Empty response from LLM"
        ) as exc_info:
            ResponseParser.parse_llm_response(BREVID, "")

        log.debug("Caught expected exception: %s", exc_info.value)

        assert exc_info.value.brevid == BREVID
        assert exc_info.value.operation == "parse_llm_response"

    def test_invalid_json_raises_parse_error(self) -> None:
        """Test invalid JSON after marker raises ParseError."""
        raw_response = f"text\n{ResponseParser.JSON_MARKER}\n{{not valid json}}"
        with pytest.raises(ParseError, match="Invalid JSON format") as exc_info:
            ResponseParser.parse_llm_response(BREVID, raw_response)

        log.debug("Caught expected exception: %s", exc_info.value)
        assert exc_info.value.brevid == BREVID
        assert exc_info.value.operation == "parse_entities_json"
        assert exc_info.value.parse_type == "json"
        assert exc_info.value.content == "{not valid json}"

    def test_reraises_parse_error(self) -> None:
        """Test ParseError from parse_entities_json is re-raised as-is."""
        raw_response = f"text\n{ResponseParser.JSON_MARKER}\n[1, 2, 3]"

        with pytest.raises(ParseError, match="Expected JSON object") as exc_info:
            ResponseParser.parse_llm_response(BREVID, raw_response)

        log.debug("Caught expected exception: %s", exc_info.value)
        assert exc_info.value.brevid == BREVID
        assert exc_info.value.operation == "parse_entities_json"
        assert exc_info.value.parse_type == "json_structure"
        assert exc_info.value.content == "[1, 2, 3]"

    def test_reraises_llm_response_error(self) -> None:
        """Test LLMResponseError is re-raised as-is (not double-wrapped)."""
        with pytest.raises(LLMResponseError, match="Empty response") as exc_info:
            ResponseParser.parse_llm_response(BREVID, "")

        log.debug("Caught expected exception: %s", exc_info.value)
        assert exc_info.value.brevid == BREVID
        assert exc_info.value.operation == "parse_llm_response"
