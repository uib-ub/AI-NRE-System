"""LLM client factory for AI NER System."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_client import Client

from ..config.settings import Settings
from .claude_client import ClaudeClient
from .exceptions import LLMClientError
from .ollama_client import OllamaClient


def create_llm_client(client_type: str) -> Client:
    """Factory function to create LLM clients.

    Note: This factory assumes configuration has already been validated
    by ConfigValidator.validate_all() in the main application flow.

    Args:
        client_type: Type of client ('claude' or 'ollama').

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
            # Runtime checks: ConfigValidator ensures these are not None before factory is called
            if Settings.ANTHROPIC_API_KEY is None:
                msg = 'ANTHROPIC_API_KEY must be set'
                raise LLMClientError(
                    msg,
                    client_type=client_type,
                    operation='factory_creation',
                )
            if Settings.CLAUDE_MODEL is None:
                msg = 'CLAUDE_MODEL must be set'
                raise LLMClientError(
                    msg,
                    client_type=client_type,
                    operation='factory_creation',
                )
            return ClaudeClient(
                api_key=Settings.ANTHROPIC_API_KEY,
                model=Settings.CLAUDE_MODEL,
            )
        elif client_type == 'ollama':
            # Runtime checks: ConfigValidator ensures these are not None before factory is called
            if Settings.OPENWEBUI_ENDPOINT is None:
                msg = 'OPENWEBUI_ENDPOINT must be set'
                raise LLMClientError(
                    msg,
                    client_type=client_type,
                    operation='factory_creation',
                )
            if Settings.OPENWEBUI_TOKEN is None:
                msg = 'OPENWEBUI_TOKEN must be set'
                raise LLMClientError(
                    msg,
                    client_type=client_type,
                    operation='factory_creation',
                )
            if Settings.OLLAMA_MODEL is None:
                msg = 'OLLAMA_MODEL must be set'
                raise LLMClientError(
                    msg,
                    client_type=client_type,
                    operation='factory_creation',
                )
            return OllamaClient(
                endpoint=Settings.OPENWEBUI_ENDPOINT,
                token=Settings.OPENWEBUI_TOKEN,
                model=Settings.OLLAMA_MODEL,
            )
        else:
            supported_types = ['claude', 'ollama']
            msg = (
                f'Unsupported client type: {client_type}. '
                f'Supported types: {", ".join(supported_types)}'
            )
            raise LLMClientError(
                msg,
                client_type=client_type,
                operation='factory_creation',
            )
    except Exception as e:
        # Wrap unexpected exceptions in LLMClientError
        logging.exception(
            'Unexpected error creating %s client: %s',
            client_type,
            e,
        )
        # Catch any unexpected exceptions
        msg = f'Failed to create {client_type} client: {e}'
        raise LLMClientError(
            msg,
            client_type=client_type,
            operation='factory_creation',
        ) from e
