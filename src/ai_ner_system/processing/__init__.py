"""Processing module for medieval text annotation with LLM services.

This module provides classes and functions for processing medieval text records
using Large Language Models, with support for both individual and batch processing.
It includes validation, parsing, and comprehensive error handling for robust
text annotation workflows.
"""

from __future__ import annotations

# Data models and entities
from .entities import BatchProcessingResult, EntityRecord, ProcessingResult

# Exceptions
from .exceptions import (
    BatchProcessingError,
    LLMResponseError,
    ParseError,
    ProcessingError,
    ValidationError,
)

# Validation and parsing
from .parser import ResponseParser

# Core processor
from .processor import RecordProcessor, create_progress_logger
from .validator import RecordValidator

__all__ = [
    "BatchProcessingError",
    "BatchProcessingResult",
    "EntityRecord",
    "LLMResponseError",
    "ParseError",
    "ProcessingError",
    "ProcessingResult",
    "RecordProcessor",
    "RecordValidator",
    "ResponseParser",
    "ValidationError",
    "create_progress_logger",
]
