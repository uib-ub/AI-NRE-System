"""Medieval text processing pipeline components.

This package provides comprehensive pipeline management for medieval text processing
with Large Language Models. It includes main processor orchestration, synchronous
and asynchronous processing workflows, statistics tracking, and error handling.

The pipeline supports both individual record processing and batch processing modes,
with automatic fallback mechanisms and comprehensive progress monitoring.
"""

from .main_processor import MedievalTextProcessor
from .sync_processor import SyncProcessor
from .async_processor import AsyncProcessor
from .stats import (
    ApplicationError,
    AsyncProcessingStats,
)

__all__ = [
    # Main processor class
    "MedievalTextProcessor",

    # Processing workflow classes
    "SyncProcessor",
    "AsyncProcessor",

    # Statistics and error handling
    "ApplicationError",
    "AsyncProcessingStats",
]
