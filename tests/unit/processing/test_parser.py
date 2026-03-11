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

from ai_ner_system.processing.entities import EntityRecord
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
class TestParseLLMResponse:
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

    def test_unexpected_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test unexpected exception from parsing is wrapped in LLMResponseError."""

        # Monkeypatch parse_entities_json to raise a RuntimeError
        def raise_runtime_exception(_json_text: str, _brevid: str) -> None:
            raise RuntimeError("Unexpected runtime error during parsing")

        monkeypatch.setattr(
            ResponseParser,
            "parse_entities_json",
            raise_runtime_exception,
        )

        raw_response = f"text\n{ResponseParser.JSON_MARKER}\nsomething"
        with pytest.raises(
            LLMResponseError, match="Failed to parse LLM response"
        ) as exc_info:
            ResponseParser.parse_llm_response(BREVID, raw_response)

        log.debug("Caught expected exception: %s", exc_info.value)
        assert exc_info.value.brevid == BREVID
        assert exc_info.value.operation == "parse_llm_response"
        assert "Unexpected runtime error during parsing" in str(
            exc_info.value.__cause__
        )


# ===================================================================
# parse_entities_json
# ===================================================================
class TestParseEntitiesJson:
    """Tests for ResponseParser.parse_entities_json().

    Also tests internal helper methods:
    - _parse_json_structure (JSON parsing, non-dict detection, truncation)
    - _validate_entities_structure (list validation, missing key)
    - _create_entity_records (valid/invalid/mixed entities)
    - _log_entity_creation_results (success/failure logging)
    """

    def test_valid_json(self) -> None:
        """Test valid JSON with one entity."""
        entities = ResponseParser.parse_entities_json(VALID_ENTITIES_JSON, BREVID)

        log.debug("Parsed entities: %s", entities)
        assert len(entities) == 1
        assert isinstance(entities[0], EntityRecord)
        assert entities[0].name == VALID_ENTITY_DATA["name"]

    def test_multiple_entities(self) -> None:
        """Test JSON with multiple entities."""
        # Olafsklaustre;Place Name;j;2;601;Monastery;N/A;non
        data = {
            "entities": [
                VALID_ENTITY_DATA,
                {
                    "name": "Olafsklaustre",
                    "type": "Place Name",
                    "preposition": "j",
                    "order": 2,
                    "description": "Monastery",
                    "gender": "N/A",
                    "language": "non",
                },
            ]
        }

        entities = ResponseParser.parse_entities_json(json.dumps(data), BREVID)

        log.debug("Parsed entities: %s", entities)

        assert len(entities) == 2
        assert entities[0].name == VALID_ENTITY_DATA["name"]
        assert entities[1].name == "Olafsklaustre"

    @pytest.mark.parametrize(
        "json_text",
        [
            "",
            "   ",
        ],
    )
    def test_empty_or_blank_returns_empty(
        self,
        json_text: str,
    ) -> None:
        """Test empty or blank JSON text returns empty entity list."""
        entities = ResponseParser.parse_entities_json(json_text, BREVID)
        assert entities == []

    def test_with_invalid_entity_skips(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test JSON with one valid and one invalid entity skips invalid one."""
        data = {
            "entities": [
                VALID_ENTITY_DATA,
                {
                    "name": "",
                    "type": "Place Name",
                },
            ]
        }

        with caplog.at_level(logging.WARNING):
            entities = ResponseParser.parse_entities_json(json.dumps(data), BREVID)

        log.debug("Parsed entities: %s", entities)
        log.debug("Captured logs: %s", caplog.text)
        assert len(entities) == 1
        assert entities[0].name == VALID_ENTITY_DATA["name"]
        assert "Invalid entity data" in caplog.text

    def test_empty_entities_list(self) -> None:
        """Test JSON with empty entities list."""
        entities = ResponseParser.parse_entities_json('{"entities": []}', BREVID)

        log.debug("Parsed entities: %s", entities)
        assert entities == []

    def test_missing_entities_key_returns_empty(self) -> None:
        """Test JSON missing 'entities' key returns empty list."""
        entities = ResponseParser.parse_entities_json("{}", BREVID)

        log.debug("Parsed entities: %s", entities)
        assert entities == []

    # --- _parse_json_structure coverage (via parse_entities_json) ---
    def test_invalid_json_format_raises(self) -> None:
        """Test invalid JSON format raises ParseError."""
        invalid_json = "{not valid json}"
        with pytest.raises(ParseError, match="Invalid JSON format") as exc_info:
            ResponseParser.parse_entities_json(invalid_json, BREVID)

        log.debug("Caught expected exception: %s", exc_info.value)
        assert exc_info.value.brevid == BREVID
        assert exc_info.value.operation == "parse_entities_json"
        assert exc_info.value.parse_type == "json"
        assert exc_info.value.content == invalid_json

    @pytest.mark.parametrize(
        ("json_text", "type_name"),
        [
            ("[1, 2, 3]", "list"),
            ('"a string"', "str"),
            ("42", "int"),
            ("true", "bool"),
        ],
    )
    def test_non_dict_json_raises(
        self,
        json_text: str,
        type_name: str,
    ) -> None:
        """Test non-dict JSON raises ParseError with type info."""
        with pytest.raises(ParseError, match="Expected JSON object") as exc_info:
            ResponseParser.parse_entities_json(json_text, BREVID)

        log.debug("Caught expected exception: %s", exc_info.value)
        assert exc_info.value.brevid == BREVID
        assert exc_info.value.operation == "parse_entities_json"
        assert exc_info.value.parse_type == "json_structure"
        assert exc_info.value.content == json_text
        assert type_name in str(exc_info.value)

    # --- _validate_entities_structure coverage (via parse_entities_json) ---
    @pytest.mark.parametrize(
        ("entities_value", "type_name"),
        [
            ("a string", "str"),
            ({"name": "Olauer"}, "dict"),
            (123, "int"),
        ],
    )
    def test_non_list_entities_raises(
        self,
        entities_value: Any,
        type_name: str,
    ) -> None:
        """Test non-list entities value raises ParseError."""
        json_text = json.dumps({"entities": entities_value})
        log.debug("Testing with JSON: %s", json_text)

        with pytest.raises(ParseError, match="Entities must be a list") as exc_info:
            ResponseParser.parse_entities_json(json_text, BREVID)

        log.debug("Caught expected exception: %s", exc_info.value)
        assert exc_info.value.brevid == BREVID
        assert exc_info.value.operation == "parse_entities_json"
        assert exc_info.value.parse_type == "entities_structure"
        assert type_name in str(exc_info.value)

    # --- _create_entity_records coverage (via parse_entities_json) ---
    def test_all_invalid_entities_returns_empty(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test all invalid entities returns empty list."""
        data = {
            "entities": [
                {
                    "name": "",
                    "type": "Person Name",
                },
                {
                    "name": "X",
                    "type": "",
                },
            ]
        }

        with caplog.at_level(logging.WARNING):
            entities = ResponseParser.parse_entities_json(json.dumps(data), BREVID)

        log.debug("Parsed entities: %s", entities)
        log.debug("Captured logs: %s", caplog.text)
        assert entities == []

    # --- _log_entity_creation_results coverage (via parse_entities_json) ---
    def test_all_success_logs_info(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test successful entity creation logs info message."""
        with caplog.at_level(logging.INFO):
            ResponseParser.parse_entities_json(VALID_ENTITIES_JSON, BREVID)

        log.debug("Captured logs: %s", caplog.text)
        assert "Parsed all 1 entities successfully" in caplog.text

    def test_partial_failure_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test partial entity failure logs warning with counts."""
        data = {
            "entities": [
                VALID_ENTITY_DATA,
                {
                    "name": "",
                    "type": "Place Name",
                },
            ]
        }

        with caplog.at_level(logging.WARNING):
            ResponseParser.parse_entities_json(json.dumps(data), BREVID)

        log.debug("Captured logs: %s", caplog.text)
        assert "1/2 valid entities" in caplog.text
        assert "1 failed" in caplog.text
