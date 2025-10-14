"""Statistics and utility classes for medieval text processing pipeline.

This module provides data classes and utilities for tracking processing statistics,
managing progress, and handling application-level errors in the medieval text
processing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..processing import ProcessingResult


class ApplicationError(Exception):
    """Custom exception for application-level errors.

    This exception is raised for high-level application errors that don't
    fit into more specific exception categories.
    """


@dataclass
class AsyncProcessingStats:
    """Statistics for async processing operations

    This class tracks comprehensive statistics during asynchronous processing,
    including timing, success rates, and detailed batch information.

    Attributes:
        total_records: Total number of records to process.
        processed_records: Number of successfully processed records.
        failed_records: Number of failed records.
        start_time: Processing start time (Unix timestamp).
        end_time: Processing end time (Unix timestamp), None if still running.
        processing_time: Total processing time in seconds.
        batch_info: Information about batch processing (if used).
        results: List of ProcessingResult objects for detailed tracking.
    """
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    start_time: float = 0.0
    end_time: float | None = None
    processing_time: float = 0.0
    batch_info: dict[str, Any] | None = None
    results: list[ProcessingResult] = field(default_factory=lambda: [])

    def __post_init__(self):
        """Validate statistics after initialization.

        Raises:
            ValueError: If statistics are invalid (e.g., negative counts).
        """
        # Validate non-negative values
        if self.total_records < 0:
            raise ValueError(
                f'Total records cannot be negative, got {self.total_records}'
            )
        if self.processed_records < 0:
            raise ValueError(
                f'Processed records cannot be negative, got {self.processed_records}'
            )
        if self.failed_records < 0:
            raise ValueError(
                f'Failed records cannot be negative, got {self.failed_records}'
            )
        if self.processing_time < 0:
            raise ValueError(
                f'Processing time cannot be negative, got {self.processing_time}'
            )

    @property
    def success_rate(self) -> float:
        """Calculate success rate of processing as percentage.

        Returns:
            Success rate as a percentage (0-100) of processed records over total records.
            Returns 0.0 if total_records is 0.
        """
        if self.total_records == 0:
            return 0.0
        return (self.processed_records / self.total_records) * 100.0

    @property
    def is_complete(self) -> bool:
        """Check if processing is complete.

        Returns:
            True if processing has ended (end_time is set), False otherwise.
        """
        return self.end_time is not None

    @property
    def throughput(self) -> float:
        """Calculate records processed per second.

        Returns:
            Throughput as records per second. Returns 0.0 if processing_time is zero.
        """
        if self.processing_time == 0.0:
            return 0.0
        return self.processed_records / self.processing_time

    def summary(self) -> dict[str, Any]:
        """Generate a summary dictionary of statistics.

        Returns:
            Dictionary containing all key statistics.
        """
        return {
            'total_records': self.total_records,
            'processed_records': self.processed_records,
            'failed_records': self.failed_records,
            'success_rate': self.success_rate,
            'processing_time': self.processing_time,
            'throughput': self.throughput,
            'is_complete': self.is_complete,
            'start_time': self.start_time,
            'end_time': self.end_time,
        }

    def __str__(self) -> str:
        """Return human-readable string representation.

        Returns:
            String summarizing key statistics.
        """
        return (
            f'AsyncProcessingStats('
            f'{self.processed_records}/{self.total_records} processed, '
            f'{self.success_rate:.1f}% success, '
            f'{self.processing_time:.1f}s)'
        )
