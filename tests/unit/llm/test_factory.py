"""Unit tests for ai_ner_system.llm.factory module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from ai_ner_system.llm.exceptions import LLMClientError

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from ai_ner_system.llm.base_client import Client

from ai_ner_system.config.settings import ConfigError, Settings
from ai_ner_system.llm import create_llm_client, factory
from ai_ner_system.llm.claude_client import ClaudeClient
from ai_ner_system.llm.ollama_client import OllamaClient

log = logging.getLogger(__name__)


@pytest.mark.usefixtures("no_dotenv")
class TestLLMClientFactory:
    """Tests for LLM client factory function."""

    def test_factory_registry_matches_supported_clients(self) -> None:
        """Test factory registry keys stay aligned with supported client types."""
        assert set(factory._CLIENT_CLASSES) == set(  # pyright: ignore[reportPrivateUsage]
            Settings.SUPPORTED_CLIENTS
        )

    @pytest.mark.parametrize(
        ("client_type", "expected_class"),
        [
            ("claude", ClaudeClient),
            ("ollama", OllamaClient),
            ("CLAUDE", ClaudeClient),  # Test case-insensitivity
            ("  ollama  ", OllamaClient),  # Test whitespace stripping
        ],
    )
    @pytest.mark.usefixtures("mock_env_claude", "mock_env_ollama")
    def test_create_llm_client_success(
        self,
        client_type: str,
        expected_class: type[Client],
    ) -> None:
        """Test successful creation of LLM clients.

        Args:
            client_type: The type of client to create.
            expected_class: The expected class of the created client.
        """
        Settings.initialize(reload_env=False, create_dirs=False)
        client = create_llm_client(client_type)
        assert isinstance(client, expected_class)

    @pytest.mark.parametrize(
        ("client_type", "exception_type", "match_pattern"),
        [
            ("", ValueError, r"(?i)client_type must be provided"),
            ("   ", LLMClientError, r"(?i)unsupported client type"),
            ("invalid_type", LLMClientError, r"(?i)unsupported client type"),
        ],
    )
    def test_create_llm_client_invalid_type(
        self,
        client_type: str,
        exception_type: type[Exception],
        match_pattern: str,
    ) -> None:
        """Test that creating a client with an invalid type raises an error."""
        Settings.initialize(reload_env=False, create_dirs=False)
        with pytest.raises(exception_type, match=match_pattern) as exc_info:
            create_llm_client(client_type)

        log.debug("Caught expected %s: %s", exception_type.__name__, exc_info.value)
        assert match_pattern.strip(r"(?i)") in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        ("side_effect", "exception_type", "match_pattern", "expected_info"),
        [
            (
                ConfigError(message="Configuration error"),
                LLMClientError,
                r"(?i)configuration error",
                "configuration error",
            ),
            (
                RuntimeError("Runtime error"),
                LLMClientError,
                r"(?i)failed to create claude client",
                "failed to create claude client",
            ),
        ],
    )
    @pytest.mark.usefixtures("mock_env_claude")
    def test_create_llm_client_error(
        self,
        mocker: MockerFixture,
        side_effect: Exception,
        exception_type: type[Exception],
        match_pattern: str,
        expected_info: str,
    ) -> None:
        """Test that errors during client creation are handled properly."""
        Settings.initialize(reload_env=False, create_dirs=False)

        mocker.patch.object(
            Settings,
            "get_client_init_params",
            side_effect=side_effect,
        )

        with pytest.raises(exception_type, match=match_pattern) as exc_info:
            create_llm_client("claude")

        err = exc_info.value
        log.debug("Caught expected %s: %s", exception_type.__name__, err)
        assert expected_info in str(err).lower()
        assert isinstance(err, LLMClientError)
        assert err.client_type == "claude"
        assert err.operation == "factory_creation"

    def test_factory_registry_mismatch_raise(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test that a mismatch between factory registry and client classes raises an error."""
        Settings.initialize(reload_env=False, create_dirs=False)

        # Patch SUPPORTED_CLIENTS to include only 'claude' to create a mismatch with _CLIENT_CLASSES
        mocker.patch.object(Settings, "SUPPORTED_CLIENTS", new=frozenset({"claude"}))

        # Remove mapping "claude" from _CLIENT_CLASSES
        patched = dict(factory._CLIENT_CLASSES)  # pyright: ignore[reportPrivateUsage]
        patched.pop("claude", None)
        mocker.patch.object(factory, "_CLIENT_CLASSES", new=patched)

        # Settings.get_client_init_params must still return something valid
        mocker.patch.object(
            Settings,
            "get_client_init_params",
            return_value={"api_key": "fake_key", "model": "fake_model"},
        )

        with pytest.raises(
            LLMClientError, match=r"(?i)unsupported client type"
        ) as exc_info:
            create_llm_client("claude")

        log.debug("Caught expected LLMClientError: %s", exc_info.value)
        assert "unsupported client type" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "factory_creation"
