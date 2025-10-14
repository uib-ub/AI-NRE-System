"""Input validation for AI NER System processing.

This module provides validation utilities for ensuring record data integrity
before processing with LLM services.
"""

from __future__ import annotations

from typing import ClassVar

from .exceptions import ValidationError


class RecordValidator:
    """Validates record data for processing.

    This validator ensures that records contain all required fields and
    have valid data before being processed by LLM services.

    Attributes:
        REQUIRED_FIELDS: Frozenset of field names that must be present
            and non-empty in each record.
    """

    REQUIRED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {'Bindnr', 'Brevid', 'Tekst'}
    )

    @staticmethod
    def _is_missing_or_blank(value: object) -> bool:
        """Check if a value is missing, non-string, or blank.

        Args:
            value: The value to check.

        Returns:
            True if the value is None, not a string, or a blank string.
        """
        if not isinstance(value, str):
            return True
        return not value.strip()

    @classmethod
    def validate_record(cls, record: dict[str, str]) -> None:
        """Validate a single record for processing.

        Args:
            record: Dictionary containing record data with string keys and values.

        Raises:
            ValidationError: If record validation fails. The exception includes
                the brevid and list of missing/invalid fields.
        """
        brevid_raw = record.get('Brevid')
        brevid = brevid_raw.strip() if isinstance(brevid_raw, str) else 'unknown'

        # Check for missing fields
        missing_fields = cls.REQUIRED_FIELDS.difference(record)

        if missing_fields:
            missing = sorted(missing_fields)
            raise ValidationError(
                f'Record missing required fields: {missing}',
                brevid=brevid,
                operation='validate_record',
                missing_fields=missing,
            )

        # Check for empty or non-string values for required fields
        invalid_fields: list[str] = []
        for field in cls.REQUIRED_FIELDS:
            value = record.get(field)
            if cls._is_missing_or_blank(value):
                invalid_fields.append(field)

        if invalid_fields:
            invalid_sorted = sorted(invalid_fields)
            raise ValidationError(
                f'Record has empty or invalid required fields: {invalid_sorted}',
                brevid=brevid,
                operation='validate_record',
                missing_fields=invalid_sorted
            )

    @classmethod
    def validate_records(cls, records: list[dict[str, str]]) -> None:
        """Validate a list of records

        Args:
            records: List of dictionaries containing record data.

        Raises:
            ValidationError: If any record validation fails.
        """
        if not records:
            raise ValidationError(
                f'Records list cannot be empty',
                operation='validate_records',
            )

        # Validate each record
        for i, record in enumerate(records):
            try:
                cls.validate_record(record)
            except ValidationError as e:
                # Extract brevid for error reporting
                brevid_raw = record.get('Brevid')
                brevid = brevid_raw.strip() if isinstance(brevid_raw, str) else 'unknown'

                raise ValidationError(
                    f'Validation failed at index {i}: {e}',
                    brevid=brevid,
                    operation='validate_records',
                    missing_fields=getattr(e, 'missing_fields', []),
                ) from e
