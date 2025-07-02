"""AI NER System for processing historical texts with unstandardized orthography.

This package provides tools for processing medieval texts using Large Language Models
for Named Entity Recognition and text annotation tasks.
"""

__version__ = "0.1.0"

# Import main classes and functions for easy access
from .config import Config, ConfigError
from .prompts import PromptBuilder, GenericPromptBuilder, PromptError
from .processing import RecordProcessor, EntityRecord, ProcessingError, ValidationError, LLMResponseError

__all__ = [
    "Config",
    "ConfigError",
    "PromptBuilder",
    "GenericPromptBuilder",
    "PromptError",
    "RecordProcessor",
    "EntityRecord",
    "ProcessingError",
    "ValidationError",
    "LLMResponseError",
]