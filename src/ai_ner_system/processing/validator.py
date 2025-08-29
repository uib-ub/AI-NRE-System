"""Input validation for AI NER System processing.

This module provides validation functions for records and data structures
used in medieval text processing with LLM services.
"""

from typing import Dict, List
from .exceptions import ValidationError

class RecordValidator:
    """Validates record data for processing."""

    @staticmethod
    def validate_record(record: Dict[str, str]) -> None:
        """Validate a single record for processing

        Args:
            record: Dictionary containing record data.

        Raises:
            ValidationError: If record validation fails.
        """

        if not isinstance(record, dict):
            raise ValidationError("Record must be a dictionary")

        # Check required fields
        required_fields = ["Bindnr", "Brevid", "Tekst"]
        missing_fields = []

        for field in required_fields:
            if field not in record:
                missing_fields.append(field)
            elif not str(record[field]).strip():
                missing_fields.append(f'{field} (empty)')

        if missing_fields:
            raise ValidationError(
                f'Missing required fields: {missing_fields}',
                brevid=record.get("Brevid", "Unknown"),
                missing_fields=missing_fields
            )


    @staticmethod
    def validate_records(records: List[Dict[str, str]]) -> None:
        """Validate a list of records

        Args:
            records: List of dictionaries containing record data.

        Raises:
            ValidationError: If any record validation fails.
        """
        if not isinstance(records, list):
            raise ValidationError("Records must be a list")

        if not records:
            raise ValidationError("Records list cannot be empty")

        # Validate each record
        for i, record in enumerate(records):
            try:
                RecordValidator.validate_record(record)
            except ValidationError as e:
                raise ValidationError(
                    f'Error in record at index {i}: {e}',
                    brevid=record.get("Brevid", "Unknown"),
                    missing_fields=e.missing_fields
                ) from e





