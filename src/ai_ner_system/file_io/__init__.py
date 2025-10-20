"""Input/Output operations for AI NER System.

This package provides robust CSV reading and various output writing capabilities
for processing medieval texts with structured error handling and validation.
"""

from __future__ import annotations

from .csv_reader import CSVReader
from .exceptions import (
    CSVError,
    EncodingError,
    FileIOError,
    FileValidationError,
    OutputError,
)
from .output_writers import OutputWriter

__all__ = [
    "CSVError",
    "CSVReader",
    "EncodingError",
    "FileIOError",
    "FileValidationError",
    "OutputError",
    "OutputWriter",
]
