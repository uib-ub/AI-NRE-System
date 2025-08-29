"""Processing module for medieval text annotation with LLM services.

This module provides classes and functions for processing medieval text records
using Large Language Models, with support for both individual and batch processing.
It includes validation, parsing, and comprehensive error handling for robust
text annotation workflows.
"""

# Core processor
from .processor import RecordProcessor

# Data models and entities
from .entities import EntityRecord, ProcessingResult, BatchProcessingResult

# Validation and parsing
from .validator import RecordValidator
from .parser import ResponseParser

# Exceptions
from .exceptions import (
    ProcessingError,
    ValidationError,
    LLMResponseError,
    ParseError,
    BatchProcessingError
)

# Utility functions for monitoring
from .processor import create_progress_logger

__all__ = [
    # Core processor
    "RecordProcessor",

    # Data models
    "EntityRecord",
    "ProcessingResult",
    "BatchProcessingResult",

    # Processing components
    "RecordValidator",
    "ResponseParser",

    # Exceptions
    "ProcessingError",
    "ValidationError",
    "LLMResponseError",
    "ParseError",
    "BatchProcessingError",

    # Utilities
    "create_progress_logger",
]
