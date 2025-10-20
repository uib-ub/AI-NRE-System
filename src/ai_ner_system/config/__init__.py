"""Configuration management for AI NER System.

This package provides comprehensive configuration management with environment
variables loading, validation, and error handling for medieval text processing.
"""

from __future__ import annotations

from .exceptions import (
    ConfigError,
    ConfigValidationError,
    DirectoryValidationError,
    FileValidationError,
)
from .settings import Settings
from .validation import ConfigValidator

__all__ = [
    "ConfigError",
    "ConfigValidationError",
    "ConfigValidator",
    "DirectoryValidationError",
    "FileValidationError",
    "Settings",
]
