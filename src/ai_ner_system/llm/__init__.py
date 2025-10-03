"""LLM client implementations for AI NER System.

This package provides LLM client implementations for various providers
including Claude and Ollama, with support for both synchronous and
asynchronous batch processing.
"""

from .base_client import Client
from .claude_client import ClaudeClient
from .ollama_client import OllamaClient
from .factory import create_llm_client
from .exceptions import (
    APIError,
    AuthenticationError,
    BatchProcessingError,
    BatchTimeoutError,
    LLMClientError,
    LLMConnectionError,
    LLMValidationError,
    RateLimitError,
)
from .batch_models import BatchStatus, BatchRequest, BatchProgress, BatchResponse

__all__ = [
    "Client",
    "ClaudeClient",
    "OllamaClient",
    "create_llm_client",
    "LLMClientError",
    "APIError",
    "BatchTimeoutError",
    "AuthenticationError",
    "LLMConnectionError",
    "LLMValidationError",
    "RateLimitError",
    "BatchProcessingError",
    "BatchStatus",
    "BatchRequest",
    "BatchProgress",
    "BatchResponse",
]