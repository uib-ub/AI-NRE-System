"""LLM client factory for AI NER System."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from .base_client import Client

from ai_ner_system.config.settings import Settings

from .claude_client import ClaudeClient
from .exceptions import LLMClientError
from .ollama_client import OllamaClient

# Client class registry: maps client type to client class
_CLIENT_CLASSES: dict[str, type[Client]] = {
    "claude": ClaudeClient,
    "ollama": OllamaClient,
}


def _raise_unsupported_type_error(client_type: str) -> NoReturn:
    """Raise error for unsupported client type.

    Args:
        client_type: The unsupported client type.

    Raises:
        LLMClientError: Always raised for unsupported types.
    """
    msg = f"Unsupported client type: {client_type}"
    raise LLMClientError(
        msg,
        client_type=client_type,
        operation="factory_creation",
    )


def _raise_missing_config_error(
    missing_params: list[str],
    client_type: str,
) -> NoReturn:
    """Raise error for missing or empty configuration parameters.

    Args:
        missing_params: List of missing or empty parameter names.
        client_type: The client type for error context.

    Raises:
        LLMClientError: Always raised for missing/empty configuration.
    """
    params_str = ", ".join(missing_params)
    msg = f"Missing or empty required configuration: {params_str}"
    raise LLMClientError(
        msg,
        client_type=client_type,
        operation="factory_creation",
    )


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

    client_type = client_type.strip().lower()

    if client_type not in Settings.SUPPORTED_CLIENTS:
        supported_types = ", ".join(sorted(Settings.SUPPORTED_CLIENTS))
        msg = f"Unsupported client type: {client_type}. Supported types: {supported_types}"
        raise LLMClientError(
            msg,
            client_type=client_type,
            operation="factory_creation",
        )

    try:
        # Get initialization parameters from Settings registry
        init_params = Settings.get_client_init_params(client_type)

        # Validate all required parameters are present and non-empty
        missing_or_empty = [
            param_name
            for param_name, param_value in init_params.items()
            if not param_value or not param_value.strip()
        ]

        if missing_or_empty:
            _raise_missing_config_error(missing_or_empty, client_type)

        # Get the client class and instantiate
        client_class = _CLIENT_CLASSES.get(client_type)
        if not client_class:
            _raise_unsupported_type_error(client_type)

        # Type checker doesn't know we've validated values above
        return client_class(**init_params)  # type: ignore[arg-type]

    except LLMClientError:
        # Preserve LLMClientError from validation or client initialization
        raise
    except Exception as e:
        # Wrap unexpected exceptions in LLMClientError
        msg = f"Failed to create {client_type} client: {e}"
        raise LLMClientError(
            msg,
            client_type=client_type,
            operation="factory_creation",
        ) from e
