"""LLM client factory for AI NER System."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from .base_client import Client

from ai_ner_system.config.settings import Settings

from .claude_client import ClaudeClient
from .exceptions import LLMClientError
from .ollama_client import OllamaClient

_SUPPORTED_TYPES: Final[tuple[str, str]] = ("claude", "ollama")


def _require_config_value(name: str, value: str | None, client_type: str) -> str:
    """Ensure a configuration value is a non-empty string.

    Args:
        name: Name of the configuration parameter.
        value: The configuration value to check.
        client_type: Type of client for error context.

    Returns:
        The validated string value.

    Raises:
        LLMClientError: If the value is None or empty.
    """
    if value is None or value.strip() == "":
        msg = f"{name} must be set and non-empty"
        raise LLMClientError(
            msg,
            client_type=client_type,
            operation="factory_creation",
        )
    return value


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
        msg = "client_type must be provided"
        raise ValueError(msg)

    client_type = client_type.lower().strip()

    if client_type not in _SUPPORTED_TYPES:
        supported_types = ", ".join(_SUPPORTED_TYPES)
        msg = f"Unsupported client type: {client_type}. Supported types: {supported_types}"
        raise LLMClientError(
            msg,
            client_type=client_type,
            operation="factory_creation",
        )

    try:
        if client_type == "claude":
            api_key = _require_config_value(
                "ANTHROPIC_API_KEY",
                Settings.ANTHROPIC_API_KEY,
                client_type,
            )
            model = _require_config_value("CLAUDE_MODEL", Settings.CLAUDE_MODEL, client_type)
            return ClaudeClient(api_key=api_key, model=model)

        if client_type == "ollama":
            endpoint = _require_config_value(
                "OPENWEBUI_ENDPOINT",
                Settings.OPENWEBUI_ENDPOINT,
                client_type,
            )
            token = _require_config_value("OPENWEBUI_TOKEN", Settings.OPENWEBUI_TOKEN, client_type)
            model = _require_config_value("OLLAMA_MODEL", Settings.OLLAMA_MODEL, client_type)
            return OllamaClient(endpoint=endpoint, token=token, model=model)

    except LLMClientError:
        # Preserve LLMClientError from _require_config_value or client initialization
        raise
    except Exception as e:
        # Wrap unexpected exceptions in LLMClientError
        msg = f"Failed to create {client_type} client: {e}"
        raise LLMClientError(
            msg,
            client_type=client_type,
            operation="factory_creation",
        ) from e
