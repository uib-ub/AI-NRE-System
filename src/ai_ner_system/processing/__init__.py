"""Processing module for medieval text annotation with LLM services.

This module provides classes and functions for processing medieval text records
using Large Language Models, with support for both individual and batch processing.
It includes validation, parsing, and comprehensive error handling for robust
text annotation workflows.
"""

from __future__ import annotations

# Core processor
from .processor import RecordProcessor, create_progress_logger

# Data models and entities
from .entities import BatchProcessingResult, EntityRecord, ProcessingResult

# Validation and parsing
from .parser import ResponseParser
from .validator import RecordValidator

# Exceptions
from .exceptions import (
    BatchProcessingError,
    LLMResponseError,
    ParseError,
    ProcessingError,
    ValidationError,
)

__all__ = [
    # Core processor
    'RecordProcessor',
    # Data models
    'EntityRecord',
    'ProcessingResult',
    'BatchProcessingResult',
    # Processing components
    'RecordValidator',
    'ResponseParser',
    # Exceptions
    'ProcessingError',
    'ValidationError',
    'LLMResponseError',
    'ParseError',
    'BatchProcessingError',
    # Utilities
    'create_progress_logger',
]
