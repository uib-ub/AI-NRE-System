"""LLM client factory for AI NER System."""

from ..config.settings import Settings
from .base_client import Client
from .ollama_client import OllamaClient
from .claude_client import ClaudeClient
from .exceptions import LLMClientError


def create_llm_client(client_type: str, **kwargs) -> Client:
    """Factory function to create LLM clients.

    Args:
        client_type: Type of client ('claude' or 'ollama').
        **kwargs: Additional arguments for client initialization.

    Returns:
        Initialized LLM client.

    Raises:
        LLMClientError: If client type is unsupported or initialization fails.
    """
    client_type = client_type.lower()
    try:
        if client_type == 'claude':
            return ClaudeClient(
                api_key=Settings.ANTHROPIC_API_KEY,
                model=Settings.CLAUDE_MODEL
            )
        elif client_type == 'ollama':
            return OllamaClient(
                endpoint=Settings.OPENWEBUI_ENDPOINT,
                token=Settings.OPENWEBUI_TOKEN,
                model=Settings.OLLAMA_MODEL
            )
        else:
            raise LLMClientError(
                f'Unsupported client type: {client_type}. Supported: claude, ollama')
    except Exception as e:
        # Catch any unexpected exceptions and wrap them
        raise LLMClientError(f'Failed to create {client_type} client: {e}') from e