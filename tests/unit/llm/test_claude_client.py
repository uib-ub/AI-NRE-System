"""Unit tests for llm.claude_client module.

Tests cover:
- ClaudeClient initialization and validation
- Single call methods (sync and async)
- Batch processing methods (create, status, info, results, cancel, monitor)
- Error handling (authentication, rate limit, API errors)
- Helper methods for message parsing
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

from ai_ner_system.llm.claude_client import ClaudeClient
from ai_ner_system.llm.exceptions import LLMClientError

log = logging.getLogger(__name__)

# Test constants to avoid S106 warnings
TEST_API_KEY = "sk-ant-test-key-123456789"
TEST_MODEL = "claude-sonnet-4-20240307"


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def mock_anthropic_clients() -> Generator[dict[str, MagicMock]]:
    """Create mock for both Anthropic and AsyncAnthropic clients."""
    with (
        patch("ai_ner_system.llm.claude_client.Anthropic") as mock_anthropic,
        patch("ai_ner_system.llm.claude_client.AsyncAnthropic") as mock_async_anthropic,
        patch("ai_ner_system.llm.claude_client.tiktoken.get_encoding") as mock_tiktoken,
    ):
        # Configure tiktoken mock
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_tiktoken.return_value = mock_encoder

        yield {
            "anthropic_client": mock_anthropic,
            "async_anthropic_client": mock_async_anthropic,
            "tiktoken": mock_tiktoken,
            "encoder": mock_encoder,
        }


@pytest.fixture
def claude_client(mock_anthropic_clients: dict[str, MagicMock]) -> ClaudeClient:  # noqa: ARG001 silences "unused argument" warning
    """Create a ClaudeClient instance with mocked dependencies."""
    return ClaudeClient(
        api_key=TEST_API_KEY,
        model=TEST_MODEL,
    )


# =============================================================================
# TestClaudeClientInit
# =============================================================================
class TestClaudeClientInit:
    """Tests for ClaudeClient initialization."""

    def test_init_with_valid_params(
        self, mock_anthropic_clients: dict[str, MagicMock]
    ) -> None:
        """Test successful initialization with valid parameters."""
        client = ClaudeClient(
            api_key=TEST_API_KEY,
            model=TEST_MODEL,
        )

        assert client.api_key == TEST_API_KEY
        assert client.model == TEST_MODEL
        assert client.max_tokens == ClaudeClient.MAX_ALLOWED_TOKENS
        assert client.temperature == ClaudeClient.DEFAULT_TEMPERATURE

        # Verify clients were initialized
        mock_anthropic_clients["anthropic_client"].assert_called_once_with(
            api_key=TEST_API_KEY
        )
        mock_anthropic_clients["async_anthropic_client"].assert_called_once_with(
            api_key=TEST_API_KEY
        )
        mock_anthropic_clients["tiktoken"].assert_called_once_with("cl100k_base")

    @pytest.mark.usefixtures("mock_anthropic_clients")
    def test_init_with_custom_max_tokens_and_temperature(
        self,
    ) -> None:
        """Test initialization with custom max_tokens and temperature."""
        custom_max_tokens = 8000
        custom_temperature = 0.7

        client = ClaudeClient(
            api_key=TEST_API_KEY,
            model=TEST_MODEL,
            max_tokens=custom_max_tokens,
            temperature=custom_temperature,
        )

        assert client.max_tokens == custom_max_tokens
        assert client.temperature == custom_temperature

    @pytest.mark.parametrize(
        ("api_key", "model", "match_pattern"),
        [
            (None, TEST_MODEL, r"(?i)api key must be provided"),
            ("", TEST_MODEL, r"(?i)api key must be provided"),
            (TEST_API_KEY, None, r"(?i)model must be provided"),
            (TEST_API_KEY, "", r"(?i)model must be provided"),
        ],
    )
    @pytest.mark.usefixtures("mock_anthropic_clients")
    def test_init_raises_for_missing_required_params(
        self,
        api_key: str | None,
        model: str | None,
        match_pattern: str,
    ) -> None:
        """Test initialization raises ValueError for missing required parameters."""
        with pytest.raises(ValueError, match=match_pattern) as exc_info:
            ClaudeClient(
                api_key=api_key,  # type: ignore[arg-type]
                model=model,  # type: ignore[arg-type]
            )

        log.debug("ValueError raised as expected: %s", exc_info.value)

    @pytest.mark.parametrize(
        ("max_tokens", "temperature", "match_pattern"),
        [
            (0, 0.5, r"(?i)max_tokens must be between"),
            (-1, 0.5, r"(?i)max_tokens must be between"),
            (300000, 0.5, r"(?i)max_tokens must be between"),
            (1000, -0.1, r"(?i)temperature must be between"),
            (1000, 1.5, r"(?i)temperature must be between"),
        ],
    )
    def test_init_raises_for_invalid_max_tokens_or_temperature(
        self,
        max_tokens: int,
        temperature: float,
        match_pattern: str,
    ) -> None:
        """Test initialization raises ValueError for invalid max_tokens or temperature."""
        with pytest.raises(ValueError, match=match_pattern) as exc_info:
            ClaudeClient(
                api_key=TEST_API_KEY,
                model=TEST_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        log.debug("ValueError raised as expected: %s", exc_info.value)

    @pytest.mark.parametrize(
        ("client_sync_async"),
        [
            ("ai_ner_system.llm.claude_client.Anthropic"),
            ("ai_ner_system.llm.claude_client.AsyncAnthropic"),
        ],
    )
    def test_init_raises_llm_client_error_on_client_initialization_failure(
        self, client_sync_async: str
    ) -> None:
        """Test initialization raises LLMClientError when client init fails."""
        with (
            patch(
                client_sync_async,
                side_effect=Exception("Connection failed"),
            ),
            pytest.raises(
                LLMClientError, match=r"(?i)failed to initialize"
            ) as exc_info,
        ):
            ClaudeClient(
                api_key=TEST_API_KEY,
                model=TEST_MODEL,
            )
        log.debug("LLMClientError raised as expected: %s", exc_info.value)
