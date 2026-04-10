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

from .conftest import (
    ANNOTATED_TEXT,
    BREVID,
    SAMPLE_LLM_RESPONSE,
    VALID_ENTITIES_JSON,
    VALID_ENTITY_DATA,
    VALID_RECORD,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / sample data
# ---------------------------------------------------------------------------

# Mirrors ResponseParser._MAX_SNIPPET without importing the private constant
MAX_SNIPPET = 200


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


# ===================================================================
# format_csv_row
# ===================================================================
class TestFormatCSVRow:
    """Tests for ResponseParser.format_csv_row()."""

    @pytest.mark.parametrize(
        ("bindnr", "brevid", "text", "expected"),
        [
            ("1", "601", "Some text", "1;601;Some text"),
            ("2", "602", "Text with ; semicolon", '2;602;"Text with ; semicolon"'),
            ("3", "603", 'Text with "quotes"', '3;603;"Text with ""quotes"""'),
            ("4", "604", "Text with\nnew line", '4;604;"Text with\nnew line"'),
            ("5", "605", "", "5;605;"),
        ],
    )
    def test_format_csv_row(
        self,
        bindnr: str,
        brevid: str,
        text: str,
        expected: str,
    ) -> None:
        """Test format_csv_row with various inputs."""
        result = ResponseParser.format_csv_row(bindnr, brevid, text)

        log.debug("Formatted CSV row: %s", result)
        assert result == expected
        assert not result.endswith("\n")
        assert not result.endswith("\r\n")


# ===================================================================
# parse_batch_response
# ===================================================================
class TestParseBatchResponse:
    """Tests for ResponseParser.parse_batch_response().

    Also tests internal helper methods:
    - _split_batch_response (section splitting, mismatch warnings)
    - _process_record_sections / _process_single_record_section
    - _extract_result_content (RESULT marker detection)
    - _format_entity_metadata
    - _create_fallback_records (fallback generation)
    """

    def test_empty_response_returns_fallback(self) -> None:
        """Test empty batch response returns fallback records."""
        records = [VALID_RECORD]
        expected_fallback_values = list(VALID_RECORD.values())

        annotated, metadata = ResponseParser.parse_batch_response(records, "")
        log.debug("Annotated: %s", annotated)
        log.debug("Metadata: %s", metadata)

        assert len(annotated) == 1
        for i, v in enumerate(annotated[0].split(";")):
            assert v == expected_fallback_values[i]
        assert metadata == []

    def test_single_record_with_result_marker(self) -> None:
        """Test batch response with RESULT marker for a single record."""
        expected_annotated_text = (
            f'{VALID_RECORD["Bindnr"]};{VALID_RECORD["Brevid"]};"{ANNOTATED_TEXT}"'
        )
        section_content = f"1 {ResponseParser.RESULT_MARKER}\n{SAMPLE_LLM_RESPONSE}"
        raw_response = f"{ResponseParser.RECORD_MARKER}{section_content}"

        annotated, metadata = ResponseParser.parse_batch_response(
            [VALID_RECORD], raw_response
        )
        log.debug("Annotated: %s", annotated)
        log.debug("Metadata: %s", metadata)
        assert len(annotated) == 1
        assert annotated[0] == expected_annotated_text
        assert len(metadata) == 1

    def test_single_record_without_result_marker(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test batch response without RESULT marker uses full content."""
        raw_content = "Raw annotated text without RESULT marker."

        raw_response = f"{ResponseParser.RECORD_MARKER}1\n{raw_content}"

        log.debug("Testing with raw response: %s", raw_response)

        with caplog.at_level(logging.WARNING):
            annotated, metadata = ResponseParser.parse_batch_response(
                [VALID_RECORD], raw_response
            )
        log.debug("Annotated: %s", annotated)
        log.debug("Metadata: %s", metadata)
        assert len(annotated) == 1
        assert (
            annotated[0]
            == f"{VALID_RECORD['Bindnr']};{VALID_RECORD['Brevid']};{raw_content}"
        )
        assert metadata == []
        assert "No RESULT marker found" in caplog.text

    def test_single_record_without_result_marker_single_line(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test section without RESULT marker and only one line (record number only)."""
        raw_response = f"{ResponseParser.RECORD_MARKER}1"

        with caplog.at_level(logging.WARNING):
            annotated, metadata = ResponseParser.parse_batch_response(
                [VALID_RECORD], raw_response
            )
        log.debug("Annotated: %s", annotated)
        log.debug("Metadata: %s", metadata)
        # Single-line section "1" has no content after the header line,
        # so it falls back to section.strip() which is "1" → triggers
        # "No JSON marker" warning → annotated_text="1", entities=[]
        assert len(annotated) == 1
        assert annotated[0] == f"{VALID_RECORD['Bindnr']};{VALID_RECORD['Brevid']};1"
        assert metadata == []
        assert "No RESULT marker found" in caplog.text

    def test_mismatched_sections_warns(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warning when section count doesn't match record count."""
        raw_response = (
            f"{ResponseParser.RECORD_MARKER}1 {ResponseParser.RESULT_MARKER}\n"
            f"{SAMPLE_LLM_RESPONSE}"
        )

        # two records but only one section in response
        records = [
            VALID_RECORD,
            {
                **VALID_RECORD,
                "Brevid": "602",
            },
        ]

        log.debug("Testing with raw response:\n%s\n", raw_response)
        log.debug("Testing with records:\n%s\n", records)

        with caplog.at_level(logging.WARNING):
            annotated, metadata = ResponseParser.parse_batch_response(
                records, raw_response
            )

        log.debug("Captured logs: %s", caplog.text)
        log.debug("Annotated: %s", annotated)
        log.debug("Metadata: %s", metadata)

        assert "Expected 2 record sections, found 1" in caplog.text

    def test_more_sections_than_records_stops(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test processing stops when there are more sections than records."""
        section = f"1 {ResponseParser.RESULT_MARKER}\n{SAMPLE_LLM_RESPONSE}"
        raw_response = (
            f"{ResponseParser.RECORD_MARKER}{section}"
            f"{ResponseParser.RECORD_MARKER}{section}"
        )

        records = [VALID_RECORD]  # Only 1 record but 2 sections
        with caplog.at_level(logging.WARNING):
            annotated, metadata = ResponseParser.parse_batch_response(
                records, raw_response
            )

        log.debug("Captured logs: %s", caplog.text)
        log.debug("Annotated: %s", annotated)
        log.debug("Metadata: %s", metadata)

        assert "More sections" in caplog.text
        assert len(annotated) == 1
        assert len(metadata) == 1

    def test_critical_error_returns_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test critical exception during parsing returns fallback records."""

        def _raise_exception(
            _raw_response: str, _records: list[dict[str, str]]
        ) -> None:
            raise RuntimeError("Simulated critical error during parsing")

        monkeypatch.setattr(ResponseParser, "_split_batch_response", _raise_exception)
        records = [VALID_RECORD]

        with caplog.at_level(logging.ERROR):
            annotated, metadata = ResponseParser.parse_batch_response(
                records, "some response"
            )

        log.debug("Annotated: %s", annotated)
        log.debug("Metadata: %s", metadata)
        log.debug("Captured logs: %s", caplog.text)
        assert "Simulated critical error during parsing" in caplog.text
        assert "Critical error parsing batch response" in caplog.text
        assert len(annotated) == 1
        assert (
            annotated[0]
            == f"{VALID_RECORD['Bindnr']};{VALID_RECORD['Brevid']};{VALID_RECORD['Tekst']}"
        )
        assert metadata == []

    def test_multi_record_batch(self) -> None:
        """Test batch response with multiple records."""
        section = f"1 {ResponseParser.RESULT_MARKER}\n{SAMPLE_LLM_RESPONSE}"

        raw_response = (
            f"{ResponseParser.RECORD_MARKER}{section}"
            f"{ResponseParser.RECORD_MARKER}{section}"
        )

        records = [
            VALID_RECORD,
            {
                "Bindnr": "2",
                "Brevid": "602",
                "Tekst": "Another record text",
            },
        ]

        annotated, metadata = ResponseParser.parse_batch_response(records, raw_response)

        log.debug("Annotated: %s", annotated)
        log.debug("Metadata: %s", metadata)
        assert len(annotated) == 2
        assert len(metadata) == 2
        for i, record in enumerate(records):
            expected_annotated_text = (
                f'{record["Bindnr"]};{record["Brevid"]};"{ANNOTATED_TEXT}"'
            )
            assert annotated[i] == expected_annotated_text

    def test_record_parse_failure_produces_fallback_row(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that a single record parse failure produces a fallback row."""
        section = f"1 {ResponseParser.RESULT_MARKER}\n"
        raw_response = f"{ResponseParser.RECORD_MARKER}{section}"  # Empty content → will raise LLMResponseError

        with caplog.at_level(logging.ERROR):
            annotated, metadata = ResponseParser.parse_batch_response(
                [VALID_RECORD], raw_response
            )

        log.debug("Annotated: %s", annotated)
        log.debug("Metadata: %s", metadata)
        log.debug("Captured logs: %s", caplog.text)
        assert "Error parsing record 1 in batch" in caplog.text
        assert "Empty response from LLM" in caplog.text
        assert len(annotated) == 1
        assert (
            annotated[0]
            == f"{VALID_RECORD['Bindnr']};{VALID_RECORD['Brevid']};{VALID_RECORD['Tekst']}"
        )
        assert metadata == []

    # --- _create_fallback_records coverage (via parse_batch_response) ---
    def test_fallback_missing_keys_default_to_unknown(self) -> None:
        """Test fallback uses 'unknown' for missing record keys."""
        records: list[dict[str, str]] = [{}]
        annotated, metadata = ResponseParser.parse_batch_response(records, "")

        log.debug("Annotated: %s", annotated)
        log.debug("Metadata: %s", metadata)
        assert len(annotated) == 1
        assert annotated[0] == "unknown;unknown;unknown"
        assert metadata == []

    def test_fallback_multiple_records(self) -> None:
        """Test fallback with multiple records."""
        records = [
            VALID_RECORD,
            {"Bindnr": "002", "Brevid": "602", "Tekst": "Second text."},
        ]
        annotated, metadata = ResponseParser.parse_batch_response(records, "")

        log.debug("Annotated: %s", annotated)
        log.debug("Metadata: %s", metadata)
        assert len(annotated) == 2
        assert metadata == []
