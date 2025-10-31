"""Medieval text processing pipeline components.

This package provides comprehensive pipeline management for medieval text processing
with Large Language Models. It includes main processor orchestration, synchronous
and asynchronous processing workflows, statistics tracking, and error handling.

The pipeline supports both individual record processing and batch processing modes,
with automatic fallback mechanisms and comprehensive progress monitoring.
"""

from __future__ import annotations

from .async_processor import AsyncProcessor
from .main_processor import MedievalTextProcessor
from .processor_protocol import ProcessorContext
from .stats import ApplicationError, AsyncProcessingStats
from .sync_processor import SyncProcessor

__all__ = [
    "ApplicationError",
    "AsyncProcessingStats",
    "AsyncProcessor",
    "MedievalTextProcessor",
    "ProcessorContext",
    "SyncProcessor",
]
