"""Data models and entities for AI NER System processing.

This module defines the core data models used by the processing package.
It provides data classes representing entities, processing results,
and batch processing outcomes for medieval text annotation tasks.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .exceptions import ValidationError


@dataclass
class EntityRecord:
    """Data class representing an entity record extracted from medieval text by LLM response.

    Attributes:
        name: The proper noun itself.
        entity_type: Type of proper noun (Person Name, Place Name, etc.).
        preposition: Preposition used with the proper noun (if applicable, otherwise use “N/A”),
        order: Order of occurrence in the text.
        brevid: The Brevid identifier from the source record.
        description: Brief description/status for people or type for places.
        gender: Gender information, "Male", "Female", or "N/A" for non-persons.
        language: Language code (ISO 639-3) (e.g., "lat", "non").
    """

    ALLOWED_GENDERS: ClassVar[frozenset[str]] = frozenset(
        {"Male", "Female", "N/A"},
    )

    name: str
    entity_type: str
    preposition: str = "N/A"
    order: int = 0
    brevid: str = ""
    description: str = ""
    gender: str = "N/A"
    language: str = ""

    def __post_init__(self) -> None:
        """Validate field values after initialization.

        Raises:
            ValidationError: If any field contains invalid data.
        """
        # Required textual fields must be non-blank strings.
        if not self.name.strip():
            raise ValidationError(
                'Entity "name" cannot be empty',
                brevid=self.brevid,
                operation="entity_validation",
            )

        if not self.entity_type.strip():
            raise ValidationError(
                'Entity "entity_type" cannot be empty',
                brevid=self.brevid,
                operation="entity_validation",
            )

        # Preposition must be non-empty after stripping (default is "N/A")
        if not self.preposition.strip():
            raise ValidationError(
                'Entity "preposition" must be a non-empty string or "N/A"',
                brevid=self.brevid,
                operation="entity_validation",
            )

        # order must be non-negative integer.
        if self.order < 0:
            raise ValidationError(
                f'Entity "order" must be non-negative, got {self.order}',
                brevid=self.brevid,
                operation="entity_validation",
            )

        # gender must be in allowed set (case-sensitive to keep a single canonical form).
        if self.gender not in self.ALLOWED_GENDERS:
            raise ValidationError(
                f"Invalid gender {self.gender}. Must be one of: {self.ALLOWED_GENDERS}",
                brevid=self.brevid,
                operation="entity_validation",
            )

    def to_csv_row(self) -> str:
        """Convert entity to semicolon-separated CSV row with proper quoting.

        Returns:
            Semicolon-separated string representation suitable for CSV output.
        """
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                self.name,
                self.entity_type,
                self.preposition,
                self.order,
                self.brevid,
                self.description,
                self.gender,
                self.language,
            ],
        )
        return buf.getvalue().rstrip("\r\n")

    @classmethod
    def create_entity_record(
        cls,
        entity_data: dict[str, Any],
        brevid: str,
    ) -> EntityRecord:
        """Create an EntityRecord from dictionary data with validation.

        its jobs is to convert raw dictionary data into a validated EntityRecord.

        Args:
            entity_data: Dictionary containing entity information with keys:
                - name: Entity name
                - type: Entity type. Will be mapped to entity_type.
                - preposition: Preposition, defaults to "N/A"
                - order: Order number, defaults to 0
                - brevid: Overrides the brevid parameter
                - description: Entity description
                - gender: Gender, defaults to "N/A"
                - language: Language code
            brevid: Brevid identifier for the record.

        Returns:
            An instance of EntityRecord.

        Raises:
            ValidationError: If required fields are missing, invalid,
            or type conversion fails.
        """
        try:
            # Extract and normalize string fields
            name = str(entity_data.get("name", "")).strip()
            entity_type = str(entity_data.get("type", "")).strip()
            preposition = str(entity_data.get("preposition", "N/A")).strip()
            description = str(entity_data.get("description", "")).strip()
            gender = str(entity_data.get("gender", "")).strip()
            language = str(entity_data.get("language", "")).strip()

            # Extract numeric field with validation
            order_raw = entity_data.get("order", 0)
            if isinstance(order_raw, str):
                order_raw = order_raw.strip()
                order = int(order_raw) if order_raw else 0
            else:
                order = int(order_raw)

            # Use provided brevid or override from entity_data
            record_brevid = str(entity_data.get("brevid", brevid)).strip()

            return cls(
                name=name,
                entity_type=entity_type,
                preposition=preposition,
                order=order,
                brevid=record_brevid,
                description=description,
                gender=gender,
                language=language,
            )
        except (ValueError, TypeError) as e:
            raise ValidationError(
                f"Invalid entity data: {e}",
                brevid=brevid,
                operation="create_entity_record",
            ) from e


@dataclass
class ProcessingResult:
    """Represents the result of processing a single record (for async code paths).

    Attributes:
        record_id: Unique identifier for the record.
        brevid: Brevid ID from the source record.
        annotated_text: Text with proper nouns marked up.
        entities: List of extracted entities.
        processing_time: Time taken to process in seconds.
        success: Whether processing was successful.
        error_message: Error message if processing failed.
    """

    record_id: str
    brevid: str
    annotated_text: str = ""
    # Creates NEW EntityRecord list for each instance
    entities: list[EntityRecord] = field(default_factory=list[EntityRecord])
    processing_time: float = 0.0
    success: bool = True
    error_message: str | None = None

    def __post_init__(self) -> None:
        """Validate processing result after initialization.

        Raises:
            ValidationError: If validation fails.
        """
        if not self.record_id:
            raise ValidationError(
                "ProcessingResult record_id cannot be empty",
                brevid=self.brevid,
                operation="processing_result_validation",
            )

        if not self.brevid:
            raise ValidationError(
                "ProcessingResult brevid cannot be empty",
                brevid=self.brevid,
                operation="processing_result_validation",
            )

        if self.processing_time < 0:
            raise ValidationError(
                f"Processing time must be non-negative, got {self.processing_time}",
                brevid=self.brevid,
                operation="processing_result_validation",
            )


@dataclass
class BatchProcessingResult:
    """Represents the result of processing a batch of records (in async mode).

    The async batch path needs one object that summarizes a whole batch.
    So, this class encapsulates the output of batch processing operations,
    aggregating results from multiple records and providing batch-level
    statistics and metadata.

    Attributes:
        batch_id: Unique identifier for the batch.
        results: List of individual processing results.
        total_processing_time: Total time for batch processing.
        successful_count: Number of successfully processed records.
        failed_count: Number of failed records.
        batch_info: Additional batch information from the API.
    """

    batch_id: str
    results: list[ProcessingResult] = field(
        default_factory=lambda: [],  # Creates NEW ProcessingResult list for each instance
    )
    total_processing_time: float = 0.0
    successful_count: int = 0
    failed_count: int = 0
    batch_info: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate batch processing result after initialization.

        Raises:
            ValidationError: If validation fails.
        """
        if not self.batch_id:
            raise ValidationError(
                "BatchProcessingResult batch_id cannot be empty",
                operation="batch_result_validation",
            )

        if self.total_processing_time < 0:
            raise ValidationError(
                f"Total processing time must be non-negative, got {self.total_processing_time}",
                operation="batch_result_validation",
            )

        if self.successful_count < 0 or self.failed_count < 0:
            raise ValidationError(
                f"Counts must be non-negative: successful={self.successful_count}, failed={self.failed_count}",
                operation="batch_result_validation",
            )
