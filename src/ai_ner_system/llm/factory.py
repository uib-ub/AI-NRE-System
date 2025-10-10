"""LLM client factory for AI NER System."""

from __future__ import annotations

import logging
from typing import Any

from ..config.settings import Settings
from .base_client import Client
from .ollama_client import OllamaClient
from .claude_client import ClaudeClient
from .exceptions import LLMClientError


def create_llm_client(client_type: str, **kwargs: Any) -> Client:
    """Factory function to create LLM clients.

    Note: This factory assumes configuration has already been validated
    by ConfigValidator.validate_all() in the main application flow.

    Args:
        client_type: Type of client ('claude' or 'ollama').
        **kwargs: Additional arguments for client initialization.

    Returns:
        Initialized LLM client.

    Raises:
        ValueError: If client_type is empty or invalid.
        LLMClientError: If client type is unsupported or initialization fails.
    """
    if not client_type:
        raise ValueError('client_type must be provided')

    client_type = client_type.lower().strip()
    try:
        if client_type == 'claude':
            # Type assertions: ConfigValidator ensures these are not None before factory is called
            assert Settings.ANTHROPIC_API_KEY is not None, 'ANTHROPIC_API_KEY must be set'
            assert Settings.CLAUDE_MODEL is not None, 'CLAUDE_MODEL must be set'
            return ClaudeClient(
                api_key=Settings.ANTHROPIC_API_KEY,
                model=Settings.CLAUDE_MODEL
            )
        elif client_type == 'ollama':
            # Type assertions: ConfigValidator ensures these are not None before factory is called
            assert Settings.OPENWEBUI_ENDPOINT is not None, 'OPENWEBUI_ENDPOINT must be set'
            assert Settings.OPENWEBUI_TOKEN is not None, 'OPENWEBUI_TOKEN must be set'
            assert Settings.OLLAMA_MODEL is not None, 'OLLAMA_MODEL must be set'
            return OllamaClient(
                endpoint=Settings.OPENWEBUI_ENDPOINT,
                token=Settings.OPENWEBUI_TOKEN,
                model=Settings.OLLAMA_MODEL
            )
        else:
            supported_types = ['claude', 'ollama']
            raise LLMClientError(
                f'Unsupported client type: {client_type}.'
                f'Supported types: {", ".join(supported_types)}',
                client_type=client_type,
                operation='factory_creation',
            )
    except Exception as e:
        # Wrap unexpected exceptions in LLMClientError
        logging.error(
            'Unexpected error creating %s client: %s',
            client_type,
            e,
            exc_info=True,
        )
        # Catch any unexpected exceptions
        raise LLMClientError(
            f'Failed to create {client_type} client: {e}',
            client_type=client_type,
            operation='factory_creation',
        ) from e