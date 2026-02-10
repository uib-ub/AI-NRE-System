"""Unit tests for ai_ner_system.llm.factory module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from pytest_mock import MockerFixture

from ai_ner_system.llm.exceptions import LLMClientError

if TYPE_CHECKING:
    from ai_ner_system.llm.base_client import Client

from ai_ner_system.config.settings import ConfigError, Settings
from ai_ner_system.llm import create_llm_client, factory
from ai_ner_system.llm.claude_client import ClaudeClient
from ai_ner_system.llm.ollama_client import OllamaClient

log = logging.getLogger(__name__)


@pytest.mark.usefixtures("no_dotenv")
class TestLLMClientFactory:
    """Tests for LLM client factory function."""

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

    def test_create_llm_client_invalid_type(self) -> None:
        """Test that creating a client with an invalid type raises an error."""
        Settings.initialize(reload_env=False, create_dirs=False)
        with pytest.raises(
            ValueError, match=r"(?i)client_type must be provided"
        ) as exc_info:
            create_llm_client("")

        log.debug("Caught expected ValueError: %s", exc_info.value)
        assert "client_type must be provided" in str(exc_info.value).lower()

        with pytest.raises(
            LLMClientError, match=r"(?i)unsupported client type"
        ) as exc_info_ws:
            create_llm_client("   ")

        log.debug("Caught expected LLMClientError: %s", exc_info_ws.value)
        assert "unsupported client type" in str(exc_info_ws.value).lower()

        with pytest.raises(
            LLMClientError,
            match=r"(?i)unsupported client type",
        ) as exc_info_invalid:
            create_llm_client("invalid_type")

        log.debug("Caught expected LLMClientError: %s", exc_info_invalid.value)
        assert "unsupported client type" in str(exc_info_invalid.value).lower()

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
    def test_create_llm_client_configuration_error(
        self,
        mocker: MockerFixture,
        side_effect: Exception,
        exception_type: type[Exception],
        match_pattern: str,
        expected_info: str,
    ) -> None:
        """Test that configuration errors during client creation are handled properly."""
        Settings.initialize(reload_env=False, create_dirs=False)

        mocker.patch.object(
            Settings,
            "get_client_init_params",
            side_effect=side_effect,
        )

        with pytest.raises(exception_type, match=match_pattern) as exc_info:
            create_llm_client("claude")

        err = exc_info.value
        log.debug("Caught expected exception: %s", err)
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
