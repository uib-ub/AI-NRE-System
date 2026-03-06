"""Unit tests for processing.entities module.

Tests cover:
- EntityRecord creation, __post_init__ validation, to_csv_row(), create_entity_record()
- ProcessingResult creation, __post_init__ validation, defaults
- BatchProcessingResult creation, __post_init__ validation, defaults
- Edge cases (empty strings, boundary values, type conversion)
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from ai_ner_system.processing.entities import (
    BatchProcessingResult,
    EntityRecord,
    ProcessingResult,
)
from ai_ner_system.processing.exceptions import ValidationError

log = logging.getLogger(__name__)


class TestEntityRecord:
    """Tests for EntityRecord dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating EntityRecord with required fields only."""
        # A sample record: Æirike;Person Name;N/A;13;601;Priest;Male;non
        record = EntityRecord(
            name="Æirike",
            entity_type="Person Name",
        )

        log.debug("Created EntityRecord: %s", record)

        assert record.name == "Æirike"
        assert record.entity_type == "Person Name"
        assert record.preposition == "N/A"
        assert record.order == 0
        assert record.brevid == ""
        assert record.description == ""
        assert record.gender == "N/A"
        assert record.language == ""

    def test_creation_with_all_fields(self) -> None:
        """Test creating EntityRecord with all fields provided."""
        record = EntityRecord(
            name="Æirike",
            entity_type="Person Name",
            preposition="N/A",
            order=13,
            brevid="601",
            description="Priest",
            gender="Male",
            language="non",
        )

        log.debug("Created EntityRecord with all fields: %s", record)

        assert record.name == "Æirike"
        assert record.entity_type == "Person Name"
        assert record.preposition == "N/A"
        assert record.order == 13
        assert record.brevid == "601"
        assert record.description == "Priest"
        assert record.gender == "Male"
        assert record.language == "non"

    @pytest.mark.parametrize(
        "gender",
        ["Male", "Female", "N/A"],
    )
    def test_valid_genders(self, gender: str) -> None:
        """Test all allowed gender values are accepted.

        Args:
            gender: A valid gender value from ALLOWED_GENDERS.
        """
        record = EntityRecord(name="Æirike", entity_type="Person Name", gender=gender)

        log.debug("Created EntityRecord with gender '%s': %s", gender, record)
        assert record.gender == gender

    @pytest.mark.parametrize(
        ("kwargs", "error_match_pattern"),
        [
            (
                {"name": "", "entity_type": "Person Name"},
                '"name" cannot be empty',
            ),
            (
                {"name": "   ", "entity_type": "Person Name"},
                '"name" cannot be empty',
            ),
            (
                {"name": "Æirike", "entity_type": ""},
                '"entity_type" cannot be empty',
            ),
            (
                {"name": "Æirike", "entity_type": "   "},
                '"entity_type" cannot be empty',
            ),
            (
                {"name": "Æirike", "entity_type": "Person Name", "preposition": ""},
                '"preposition" must be a non-empty string',
            ),
            (
                {"name": "Æirike", "entity_type": "Person Name", "preposition": "   "},
                '"preposition" must be a non-empty string',
            ),
            (
                {"name": "Æirike", "entity_type": "Person Name", "order": -1},
                '"order" must be non-negative',
            ),
            (
                {"name": "Æirike", "entity_type": "Person Name", "gender": "Unknown"},
                "Invalid gender Unknown",
            ),
            (
                {"name": "Æirike", "entity_type": "Person Name", "gender": "male"},
                "Invalid gender male",
            ),
        ],
    )
    def test_post_init_validation_raises(
        self,
        kwargs: dict[str, Any],
        error_match_pattern: str,
    ) -> None:
        """Test __post_init__ raises validationError for invalid data."""
        with pytest.raises(ValidationError, match=error_match_pattern) as exc_info:
            EntityRecord(**kwargs)

        log.debug("ValidationError raised as expected: %s", exc_info.value)

    def test_to_csv_row(self) -> None:
        """Test to_csv_row() produces semicolon-delimited output."""
        record = EntityRecord(
            name="Æirike",
            entity_type="Person Name",
            preposition="N/A",
            order=13,
            brevid="601",
            description="Priest",
            gender="Male",
            language="non",
        )

        csv_row_result = record.to_csv_row()

        log.debug("CSV row output: %s", csv_row_result)

        expected = "Æirike;Person Name;N/A;13;601;Priest;Male;non"
        assert csv_row_result == expected

    def test_to_csv_row_with_quotes(self) -> None:
        """Test to_csv_row() properly escapes fields containing quotes."""
        record = EntityRecord(
            name='Æjrike "test ""',
            entity_type="Person Name",
        )

        csv_row_result = record.to_csv_row()

        log.debug("CSV row output: %s", csv_row_result)
        # csv.writer escapes quotes by doubling them
        assert '"Æjrike ""test """""' in csv_row_result

    def test_to_csv_row_with_semicolons(self) -> None:
        """Test to_csv_row() properly quotes fields containing semicolons."""
        record = EntityRecord(
            name="Æjrike;test",
            entity_type="Person Name",
        )

        csv_row_result = record.to_csv_row()

        log.debug("CSV row output: %s", csv_row_result)
        # csv.writer should quote the entire field if it contains a semicolon
        assert '"Æjrike;test"' in csv_row_result

    def test_create_entity_record_success(self) -> None:
        """Test create_entity_record() with valid entity data."""
        data: dict[str, Any] = {
            "name": "Æirike",
            "type": "Person Name",
            "preposition": "N/A",
            "order": 13,
            "brevid": "601",
            "description": "Priest",
            "gender": "Male",
            "language": "non",
        }

        record = EntityRecord.create_entity_record(data, brevid="601")
        log.debug("Created EntityRecord from data: %s", record)

        assert record.name == "Æirike"
        assert record.entity_type == "Person Name"
        assert record.preposition == "N/A"
        assert record.order == 13
        assert record.brevid == "601"
        assert record.description == "Priest"
        assert record.gender == "Male"
        assert record.language == "non"

    def test_create_entity_record_defaults(self) -> None:
        """Test create_entity_record() uses defaults for missing keys."""
        data: dict[str, Any] = {
            "name": "Æirike",
            "type": "Person Name",
            "gender": "N/A",
        }

        record = EntityRecord.create_entity_record(data, brevid="601")
        log.debug("Created EntityRecord with defaults: %s", record)

        assert record.preposition == "N/A"
        assert record.order == 0
        assert record.description == ""
        assert record.gender == "N/A"
        assert record.language == ""

    @pytest.mark.parametrize(
        ("raw_order", "expected_order"),
        [
            (3, 3),
            ("3", 3),
            ("", 0),
            ("   ", 0),
        ],
    )
    def test_create_entity_record_normalizes_order(
        self,
        raw_order: Any,
        expected_order: int,
    ) -> None:
        """Test create EntityRecord normalizes 'order' field to non-negative integer."""
        data: dict[str, Any] = {
            "name": "Æirike",
            "type": "Person Name",
            "order": raw_order,
            "gender": "N/A",
        }

        record = EntityRecord.create_entity_record(
            data,
            brevid="601",
        )

        log.debug("Created EntityRecord with raw order '%s': %s", raw_order, record)

        assert record.order == expected_order

    def test_create_entity_record_brevid_override(self) -> None:
        """Test create_entity_record() uses brevid from data if provided by entity_data."""
        data: dict[str, Any] = {
            "name": "Æirike",
            "type": "Person Name",
            "brevid": "001",
            "gender": "N/A",
        }

        record = EntityRecord.create_entity_record(data, brevid="601")
        log.debug("Created EntityRecord with brevid override: %s", record)
        assert record.brevid == "001"

    @pytest.mark.parametrize(
        ("field", "raw_value", "expected_value"),
        [
            ("name", "  Æirike  ", "Æirike"),
            ("type", "  Person Name  ", "Person Name"),
            ("preposition", "  N/A  ", "N/A"),
            ("description", "  Priest  ", "Priest"),
            ("gender", "  Male  ", "Male"),
            ("language", "  non  ", "non"),
        ],
    )
    def test_create_entity_record_strips_whitespace(
        self,
        field: str,
        raw_value: str,
        expected_value: str,
    ) -> None:
        """Test create_entity_record() strips whitespace from string fields."""
        data: dict[str, Any] = {
            "name": "Æirike",
            "type": "Person Name",
            "gender": "Male",
        }
        data[field] = raw_value
        record = EntityRecord.create_entity_record(data, brevid="601")
        log.debug("Created EntityRecord with raw %s '%s': %s", field, raw_value, record)

        if field == "type":
            assert record.entity_type == expected_value
        else:
            assert getattr(record, field) == expected_value

    @pytest.mark.parametrize(
        ("entity_data", "error_match_pattern"),
        [
            (
                {"name": "", "type": "Person Name"},
                '"name" cannot be empty',
            ),
            (
                {"name": "Æirike", "type": ""},
                '"entity_type" cannot be empty',
            ),
            (
                {"name": "Æirike", "type": "Person Name", "order": "abc"},
                "Invalid entity data",
            ),
            (
                {"name": "Æirike", "type": "Person Name", "order": None},
                "Invalid entity data",
            ),
        ],
    )
    def test_create_entity_record_invalid_raises(
        self,
        entity_data: dict[str, Any],
        error_match_pattern: str,
    ) -> None:
        """Test create_entity_record() raises ValidationError for invalid data."""
        with pytest.raises(ValidationError, match=error_match_pattern) as exc_info:
            EntityRecord.create_entity_record(entity_data, brevid="601")

        log.debug("ValidationError raised as expected: %s", exc_info.value)


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating ProcessingResult with required fields."""
        result = ProcessingResult(
            record_id="rec_001",
            brevid="601",
        )

        log.debug("Created ProcessingResult: %s", result)

        assert result.record_id == "rec_001"
        assert result.brevid == "601"
        assert result.annotated_text == ""
        assert result.entities == []
        assert result.processing_time == 0.0
        assert result.success is True
        assert result.error_message is None

    def test_creation_with_all_fields(self) -> None:
        """Test creating ProcessingResult with all fields specified."""
        entities = [
            EntityRecord(
                name="Æirike",
                entity_type="Person Name",
                preposition="N/A",
                order=13,
                brevid="601",
                description="Priest",
                gender="Male",
                language="non",
            )
        ]

        result = ProcessingResult(
            record_id="rec_001",
            brevid="601",
            annotated_text="Æirike annotated text",
            entities=entities,
            processing_time=1.5,
            success=True,
            error_message=None,
        )

        assert result.annotated_text == "Æirike annotated text"
        assert len(result.entities) == 1
        assert result.processing_time == 1.5
        assert result.success is True

    def test_failure_result(self) -> None:
        """Test creating a ProcessingResult representing a failure."""
        result = ProcessingResult(
            record_id="rec_002",
            brevid="602",
            success=False,
            error_message="Processing failed due to timeout.",
        )

        log.debug("Created failure ProcessingResult: %s", result)

        assert result.record_id == "rec_002"
        assert result.brevid == "602"
        assert result.annotated_text == ""
        assert result.entities == []
        assert result.processing_time == 0.0
        assert result.success is False
        assert result.error_message == "Processing failed due to timeout."

    def test_default_factory_creates_independent_lists(self) -> None:
        """Test each instance gets its own list to avoid shared-state bugs."""
        result1 = ProcessingResult(record_id="rec_001", brevid="601")
        result2 = ProcessingResult(record_id="rec_002", brevid="602")

        result1.entities.append(
            EntityRecord(
                name="Æirike",
                entity_type="Person Name",
                preposition="N/A",
                order=13,
                brevid="601",
                description="Priest",
                gender="Male",
                language="non",
            )
        )

        assert len(result1.entities) == 1
        assert len(result2.entities) == 0
        assert result2.entities == []

    @pytest.mark.parametrize(
        ("kwargs", "error_match_pattern"),
        [
            (
                {"record_id": "", "brevid": "601"},
                "ProcessingResult record_id cannot be empty",
            ),
            (
                {"record_id": "rec_001", "brevid": ""},
                "ProcessingResult brevid cannot be empty",
            ),
            (
                {"record_id": "rec_001", "brevid": "DN1_001", "processing_time": -1.0},
                "Processing time must be non-negative",
            ),
        ],
    )
    def test_post_init_validation_raises(
        self,
        kwargs: dict[str, Any],
        error_match_pattern: str,
    ) -> None:
        """Test __post_init__ raises ValidationError for invalid data."""
        with pytest.raises(ValidationError, match=error_match_pattern) as exc_info:
            ProcessingResult(**kwargs)

        log.debug("ValidationError raised as expected: %s", exc_info.value)


class TestBatchProcessingResult:
    """Tests for BatchProcessingResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating BatchProcessingResult with required fields."""
        batch_result = BatchProcessingResult(batch_id="batch_001")

        log.debug("Created BatchProcessingResult: %s", batch_result)

        assert batch_result.batch_id == "batch_001"
        assert batch_result.results == []
        assert batch_result.total_processing_time == 0.0
        assert batch_result.successful_count == 0
        assert batch_result.failed_count == 0
        assert batch_result.batch_info is None

    def test_creation_with_all_fields(self) -> None:
        """Test creating BatchProcessingResult with all fields specified."""
        processing_results = [
            ProcessingResult(record_id="rec_001", brevid="601"),
            ProcessingResult(record_id="rec_002", brevid="602"),
        ]

        batch_processing_result = BatchProcessingResult(
            batch_id="batch_001",
            results=processing_results,
            total_processing_time=3.5,
            successful_count=2,
            failed_count=0,
            batch_info={
                "id": "test_123",
                "type": "batch",
            },
        )

        log.debug(
            "Created BatchProcessingResult with all fields: %s", batch_processing_result
        )
        assert len(batch_processing_result.results) == 2
        assert batch_processing_result.total_processing_time == 3.5
        assert batch_processing_result.successful_count == 2
        assert batch_processing_result.failed_count == 0
        assert batch_processing_result.batch_info == {"id": "test_123", "type": "batch"}

    def test_default_factory_creates_independent_lists(self) -> None:
        """Test each instance gets its own list to avoid shared-state bugs."""
        batch1 = BatchProcessingResult(batch_id="batch_001")
        batch2 = BatchProcessingResult(batch_id="batch_002")

        batch1.results.append(ProcessingResult(record_id="rec_001", brevid="601"))

        assert len(batch1.results) == 1
        assert len(batch2.results) == 0
        assert batch2.results == []

    @pytest.mark.parametrize(
        ("kwargs", "error_match_pattern"),
        [
            (
                {"batch_id": ""},
                "BatchProcessingResult batch_id cannot be empty",
            ),
            (
                {"batch_id": "batch_001", "total_processing_time": -1.0},
                "Total processing time must be non-negative",
            ),
            (
                {"batch_id": "batch_001", "successful_count": -1},
                "Counts must be non-negative",
            ),
            (
                {"batch_id": "batch_001", "failed_count": -1},
                "Counts must be non-negative",
            ),
            (
                {"batch_id": "batch_001", "successful_count": -1, "failed_count": -1},
                "Counts must be non-negative",
            ),
        ],
    )
    def test_post_init_validation_raises(
        self,
        kwargs: dict[str, Any],
        error_match_pattern: str,
    ) -> None:
        """Test __post_init__ raises ValidationError for invalid data."""
        with pytest.raises(ValidationError, match=error_match_pattern) as exc_info:
            BatchProcessingResult(**kwargs)

        log.debug("ValidationError raised as expected: %s", exc_info.value)
