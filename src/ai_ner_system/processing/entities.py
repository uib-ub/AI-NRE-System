"""Data models and entities for AI NER System processing.

This module provides data classes representing entities, processing results,
and batch processing outcomes for medieval text annotation tasks.
"""

import io
import csv
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from .exceptions import ValidationError

@dataclass
class EntityRecord:
    """Data class representing an entity record extracted from medieval text by LLM response.

    Attributes:
        name: The proper noun itself.
        type: Type of proper noun (Person Name, Place Name, etc.).
        preposition: Preposition used with the proper noun (if applicable, otherwise use “N/A”),
        order: Order of occurrence in the text.
        brevid: The Brevid identifier from the source record.
        description: Brief description/status for people or type for places.
        gender: Gender information, "Male", "Female", or "N/A" for non-persons.
        language: Language code (ISO 639-3) (e.g., "lat", "non").
    """
    name: str
    type: str
    preposition: str = "N/A"
    order: int = 0
    brevid: str = ""
    description: str = ""
    gender: str = "N/A"
    language: str = ""

    # def to_csv_row(self) -> str:
    #     """Convert entity record to CSV row format.

    #     # Returns:
    #         Semicolon-separated string representation.
    #     """
    #     return ";".join([
    #         self.name,
    #         self.type,
    #         self.preposition,
    #         str(self.order),
    #         self.brevid,
    #         self.description,
    #         self.gender,
    #         self.language
    #     ])

    def to_csv_row(self) -> str:
        """Semicolon-separated row with proper quoting.

        Returns:
            Semicolon-separated string representation.
        """
        buf = io.StringIO()
        csv.writer(buf, delimiter=';', quoting=csv.QUOTE_MINIMAL).writerow([
            self.name, self.type, self.preposition, self.order,
            self.brevid, self.description, self.gender, self.language
        ])
        return buf.getvalue().rstrip("\r\n")

    @classmethod
    def create_entity_record(cls, entity_data: Dict[str, str], brevid: str) -> 'EntityRecord':
        """Create an EntityRecord from dictionary data.

        Args:
            entity_data: Dictionary containing entity information.
            brevid: Brevid identifier for the record.

        Returns:
            An instance of EntityRecord.

        Raises:
            ValidationError: If required fields are missing or invalid.
        """
        try:
            return cls(
                name=str(entity_data.get("name", "")).strip(),
                type=str(entity_data.get("type", "")).strip(),
                preposition=str(entity_data.get("preposition", "N/A")).strip(),
                order=int(entity_data.get("order", 0)),
                brevid=str(entity_data.get("brevid", brevid)).strip(),
                description=str(entity_data.get("description", "")).strip(),
                gender=str(entity_data.get("gender", "")).strip(),
                language=str(entity_data.get("language", "")).strip()
            )
        except (ValueError, TypeError) as e:
            raise ValidationError(f'Invalid entity data: {e}') from e

@dataclass
class ProcessingResult:
    """Represents the result of processing a single record (for async methods).

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
    entities: List[EntityRecord] = field(default_factory=list)
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class BatchProcessingResult:
    """Represents the result of processing a batch of records (for async methods).

    Attributes:
        batch_id: Unique identifier for the batch.
        results: List of individual processing results.
        total_processing_time: Total time for batch processing.
        successful_count: Number of successfully processed records.
        failed_count: Number of failed records.
        batch_info: Additional batch information from the API.
    """
    batch_id: str
    results: List[ProcessingResult] = field(default_factory=list)
    total_processing_time: float = 0.0
    successful_count: int = 0
    failed_count: int = 0
    batch_info: Optional[Dict[str, Any]] = None