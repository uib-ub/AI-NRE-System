"""Statistics and utility classes for medieval text processing pipeline.

This module provides data classes and utilities for tracking processing statistics,
managing progress, and handling application-level errors in the medieval text
processing pipeline.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..processing import ProcessingResult


class ApplicationError(Exception):
    """Custom exception for application-level errors."""


@dataclass
class AsyncProcessingStats:
    """Statistics for async processing operations

    This class tracks comprehensive statistics during asynchronous processing,
    including timing, success rates, and detailed batch information.

    Attributes:
        total_records: Total number of records to process.
        processed_records: Number of successfully processed records.
        failed_records: Number of failed records.
        start_time: Processing start time.
        end_time: Processing end time (None if still running).
        processing_time: Total processing time in seconds.
        batch_info: Information about batch processing (if used).
        results: List of ProcessingResult objects for detailed tracking.
    """
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    processing_time: float = 0.0
    batch_info: Optional[Dict[str, Any]] = None
    results: List[ProcessingResult] = None

    def __post_init__(self):
        """Initialize results list if not provided."""
        if self.results is None:
            self.results = []

    @property
    def success_rate(self) -> float:
        """Calculate success rate of processing as percentage.

        Returns:
            Success rate as a percentage of processed records over total records.
        """
        if self.total_records == 0:
            return 0.0
        return (self.processed_records / self.total_records) * 100.0

    @property
    def is_complete(self) -> bool:
        """Check if processing is complete.

        Returns:
            True if processing has ended (end_time is not None), False otherwise.
        """
        return self.end_time is not None

    @property
    def throughput(self) -> float:
        """Calculate records processed per second.

        Returns:
            Throughput as records per second. Returns 0 if processing time is zero.
        """
        if self.processing_time == 0:
            return 0.0
        return self.processed_records / self.processing_time