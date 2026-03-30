"""LLM client implementations for AI NER System.

This package provides LLM client implementations for various providers
including Claude and Ollama, with support for both synchronous and
asynchronous batch processing.
"""

from __future__ import annotations

from .base_client import Client
from .batch_models import BatchProgress, BatchRequest, BatchResponse, BatchStatus
from .claude_client import ClaudeClient
from .exceptions import (
    APIError,
    AuthenticationError,
    BatchProcessingError,
    BatchTimeoutError,
    LLMClientError,
    LLMConnectionError,
    RateLimitError,
)
from .factory import create_llm_client
from .ollama_client import OllamaClient

__all__ = [
    "APIError",
    "AuthenticationError",
    "BatchProcessingError",
    "BatchProgress",
    "BatchRequest",
    "BatchResponse",
    "BatchStatus",
    "BatchTimeoutError",
    "ClaudeClient",
    "Client",
    "LLMClientError",
    "LLMConnectionError",
    "OllamaClient",
    "RateLimitError",
    "create_llm_client",
]
