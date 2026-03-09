"""Unit tests for processing.validator module.

Tests cover:
- RecordValidator._is_missing_or_blank() static helper
- RecordValidator.validate_record() — missing fields, empty/non-string fields, valid record
- RecordValidator.validate_records() — empty list, single/multiple records, error wrapping
- Edge cases (whitespace-only values, brevid extraction for error context)
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from ai_ner_system.processing.exceptions import ValidationError
from ai_ner_system.processing.validator import RecordValidator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

VALID_RECORD: dict[str, str] = {
    "Bindnr": "001",
    "Brevid": "601",
    "Tekst": "Ollum monnum þæim sæm þetta bref sea æder høyra sændir Olauer",
}


class TestValidateRecord:
    """Tests for RecordValidator.validate_record()."""

    def test_validate_record_success(self) -> None:
        """Test validate_record() passes for a valid record."""
        # Should not raise an exception
        RecordValidator.validate_record(VALID_RECORD)

        log.debug("Valid record passed validation: %s", VALID_RECORD)

    @pytest.mark.parametrize(
        ("record", "error_match_pattern", "expected_missing"),
        [
            (
                {"Brevid": "601", "Tekst": "Some text"},
                "Record missing required fields",
                ["Bindnr"],
            ),
            (
                {"Bindnr": "001"},
                "Record missing required fields",
                ["Brevid", "Tekst"],
            ),
            (
                {},
                "Record missing required fields",
                ["Bindnr", "Brevid", "Tekst"],
            ),
        ],
    )
    def test_validate_record_missing_fields_raises(
        self,
        record: dict[str, str],
        error_match_pattern: str,
        expected_missing: list[str],
    ) -> None:
        """Test validate_record() raises ValidationError when required fields are absent."""
        with pytest.raises(ValidationError, match=error_match_pattern) as exc_info:
            RecordValidator.validate_record(record)

        log.debug("ValidationError raised as expected: %s", exc_info.value)
        assert exc_info.value.missing_fields == expected_missing, (
            f"Expected missing fields {expected_missing} but got {exc_info.value.missing_fields}"
        )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("Bindnr", ""),
            ("Bindnr", "   "),
            ("Brevid", ""),
            ("Brevid", "\t"),
            ("Tekst", ""),
            ("Tekst", "  \n  "),
        ],
    )
    def test_validate_record_empty_or_blank_fields_raises(
        self,
        field: str,
        bad_value: str,
    ) -> None:
        """Test validate_record() raises ValidationError for empty/blank required fields."""
        record: dict[str, str] = {**VALID_RECORD, field: bad_value}

        with pytest.raises(
            ValidationError, match="Record has empty or invalid required fields"
        ) as exc_info:
            RecordValidator.validate_record(record)

        log.debug(
            "ValidationError raised for invalid field '%s': %s", field, exc_info.value
        )
        assert exc_info.value.missing_fields == [field], (
            f"Expected invalid field '{field}' but got {exc_info.value.missing_fields}"
        )

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("Bindnr", 123),
            ("Brevid", None),
            ("Tekst", ["some", "list"]),
        ],
    )
    def test_validate_record_non_string_fields_raises(
        self, field: str, bad_value: Any
    ) -> None:
        """Test validate_record() raises ValidationError for non-string required fields."""
        record: dict[str, Any] = {**VALID_RECORD, field: bad_value}

        with pytest.raises(
            ValidationError, match="Record has empty or invalid required fields"
        ) as exc_info:
            RecordValidator.validate_record(record)

        log.debug(
            "ValidationError raised for non-string field '%s': %s",
            field,
            exc_info.value,
        )
        assert exc_info.value.missing_fields == [field], (
            f"Expected invalid field '{field}' but got {exc_info.value.missing_fields}"
        )

    def test_validate_record_brevid_extracted_for_error(self) -> None:
        """Test that brevid is extracted from the record and included in the error."""
        record: dict[str, str] = {
            "Bindnr": "001",
            "Brevid": "601",
            # Missing "Tekst"
        }

        with pytest.raises(ValidationError) as exc_info:
            RecordValidator.validate_record(record)

        log.debug("ValidationError raised with brevid context: %s", exc_info.value)
        assert exc_info.value.brevid == "601", (
            f"Expected brevid '601' in error context but got '{exc_info.value.brevid}'"
        )

    def test_validate_record_brevid_defaults_to_unknown(self) -> None:
        """Test that brevid defaults to 'unknown', when Brevid is missing or non-string."""
        record: dict[str, Any] = {
            "Bindnr": "001",
            "Brevid": 999,
        }

        with pytest.raises(ValidationError) as exc_info:
            RecordValidator.validate_record(record)

        log.debug(
            "ValidationError raised with default brevid context: %s",
            exc_info.value.brevid,
        )

        assert exc_info.value.brevid == "unknown", (
            f"Expected brevid 'unknown' in error context but got '{exc_info.value.brevid}'"
        )

    def test_validate_record_brevid_stripped(self) -> None:
        """Test that brevid is stripped of whitespace for the error context."""
        record: dict[str, str] = {
            "Bindnr": "1",
            "Brevid": "  601  ",
        }

        with pytest.raises(ValidationError) as exc_info:
            RecordValidator.validate_record(record)

        log.debug(
            "ValidationError raised with stripped brevid context: %s",
            exc_info.value.brevid,
        )

        assert exc_info.value.brevid == "601", (
            f"Expected brevid '601' in error context but got '{exc_info.value.brevid}'"
        )

    def test_validate_record_operation_set(self) -> None:
        """Test that operation is set to 'validate_record' in the error."""
        record: dict[str, str] = {
            "Brevid": "601",
        }

        with pytest.raises(ValidationError) as exc_info:
            RecordValidator.validate_record(record)

        log.debug(
            "ValidationError raised with operation context: %s",
            exc_info.value.operation,
        )

        assert exc_info.value.operation == "validate_record", (
            f"Expected operation 'validate_record' in error context but got '{exc_info.value.operation}'"
        )

    def test_validate_record_extra_fields_ignored(self) -> None:
        """Test that extra fields do not cause validation to fail."""
        record: dict[str, Any] = {**VALID_RECORD, "ExtraField": "extra value"}

        # This should not raise a ValidationError
        RecordValidator.validate_record(record)


class TestValidateRecords:
    """Tests for RecordValidator.validate_records()."""

    def test_validate_records_success(self) -> None:
        """Test validate_records() passes for a list of valid records."""
        records = [
            VALID_RECORD,
            {
                "Bindnr": "002",
                "Brevid": "602",
                "Tekst": "Another valid record text.",
            },
        ]

        # Should not raise an exception
        RecordValidator.validate_records(records)

    def test_validate_records_empty_list_raises(self) -> None:
        """Test validate_records() raises ValidationError for empty list."""
        with pytest.raises(
            ValidationError, match="Records list cannot be empty"
        ) as exc_info:
            RecordValidator.validate_records([])

        log.debug("ValidationError raised for empty records list: %s", exc_info.value)
        assert exc_info.value.operation == "validate_records", (
            f"Expected operation 'validate_records' in error context but got '{exc_info.value.operation}'"
        )

    def test_validate_records_wraps_inner_error_with_index(self) -> None:
        """Test validate_records() wraps inner ValidationError with record index."""
        records = [
            VALID_RECORD,
            {
                "Bindnr": "002",
                "Brevid": "602",
                # Missing "Tekst"
            },
        ]

        with pytest.raises(
            ValidationError, match="Validation failed at index 1"
        ) as exc_info:
            RecordValidator.validate_records(records)

        log.debug("ValidationError raised with index context: %s", exc_info.value)
        assert exc_info.value.operation == "validate_records", (
            f"Expected operation 'validate_records' in error context but got '{exc_info.value.operation}'"
        )
        assert exc_info.value.brevid == "602", (
            f"Expected brevid '602' in error context but got '{exc_info.value.brevid}'"
        )
        assert exc_info.value.missing_fields == ["Tekst"], (
            f"Expected missing fields ['Tekst'] in error context but got {exc_info.value.missing_fields}"
        )

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValidationError)

    def test_validate_records_stops_at_first_invalid(self) -> None:
        """Test validate_records() raises on the first invalid record."""
        records = [
            {"Brevid": "601"},  # index 0: Missing Bindnr and Tekst
            {"Brevid": "602"},  # index 1: also invalid, but not reached
        ]

        with pytest.raises(
            ValidationError, match="Validation failed at index 0"
        ) as exc_info:
            RecordValidator.validate_records(records)

        log.debug("Stopped at first invalid record: %s", exc_info.value)

        assert exc_info.value.brevid == "601", (
            f"Expected brevid '601' in error context but got '{exc_info.value.brevid}'"
        )
        assert exc_info.value.missing_fields == ["Bindnr", "Tekst"], (
            f"Expected missing fields ['Bindnr', 'Tekst'] in error context but got {exc_info.value.missing_fields}"
        )
        assert exc_info.value.operation == "validate_records", (
            f"Expected operation 'validate_records' in error context but got '{exc_info.value.operation}'"
        )
