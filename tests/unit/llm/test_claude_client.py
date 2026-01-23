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
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from ai_ner_system.llm.claude_client import ClaudeClient
from ai_ner_system.llm.exceptions import AuthenticationError, LLMClientError

log = logging.getLogger(__name__)

# Test constants to avoid S106 warnings
TEST_API_KEY = "sk-ant-test-key-123456789"
TEST_MODEL = "claude-sonnet-4-20240307"


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def mock_anthropic_clients(mocker: MockerFixture) -> dict[str, Any]:
    """Create mock for both Anthropic and AsyncAnthropic clients."""
    mock_sync = mocker.patch("ai_ner_system.llm.claude_client.Anthropic")
    mock_async = mocker.patch("ai_ner_system.llm.claude_client.AsyncAnthropic")
    mock_tiktoken = mocker.patch(
        "ai_ner_system.llm.claude_client.tiktoken.get_encoding"
    )

    # Configure tiktoken mock
    mock_encoder = mocker.MagicMock()
    mock_encoder.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
    mock_tiktoken.return_value = mock_encoder

    return {
        "sync_client": mock_sync,
        "async_client": mock_async,
        "tiktoken": mock_tiktoken,
        "encoder": mock_encoder,
    }


@pytest.fixture
def claude_client(mock_anthropic_clients: dict[str, Any]) -> ClaudeClient:  # noqa: ARG001 silences "unused argument" warning
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
        self, mock_anthropic_clients: dict[str, Any]
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
        mock_anthropic_clients["sync_client"].assert_called_once_with(
            api_key=TEST_API_KEY
        )
        mock_anthropic_clients["async_client"].assert_called_once_with(
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
    @pytest.mark.usefixtures("mock_anthropic_clients")
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
        self, client_sync_async: str, mocker: MockerFixture
    ) -> None:
        """Test initialization raises LLMClientError when client init fails."""
        mocker.patch(
            client_sync_async,
            side_effect=RuntimeError("Connection failed"),
        )
        with pytest.raises(
            LLMClientError, match=r"(?i)failed to initialize"
        ) as exc_info:
            ClaudeClient(
                api_key=TEST_API_KEY,
                model=TEST_MODEL,
            )

        log.debug("LLMClientError raised as expected: %s", exc_info.value)


# =============================================================================
# TestClaudeClientProperties
# =============================================================================
class TestClaudeClientProperties:
    """Tests for ClaudeClient properties."""

    def test_client_type(self, claude_client: ClaudeClient) -> None:
        """Test client_type return 'claude'."""
        assert claude_client.client_type == "claude"

    def test_supports_async_batch_true(self, claude_client: ClaudeClient) -> None:
        """Test supports_async_batch returns True."""
        assert claude_client.supports_async_batch() is True


# =============================================================================
# TestClaudeClientHelpers
# =============================================================================
class TestClaudeClientHelpers:
    """Tests for ClaudeClient helper methods."""

    def test_count_tokens(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test _count_tokens returns token count."""
        prompt = "This is a test prompt."
        expected_token_count = 5  # Based on mock tiktoken encoder in fixture

        token_count = claude_client._count_tokens(prompt)  # pyright: ignore[reportPrivateUsage]
        log.debug("Token count for prompt '%s': %d", prompt, token_count)
        assert token_count == expected_token_count

    def test_count_tokens_returns_zero_on_error(
        self,
        mock_anthropic_clients: dict[str, Any],
    ) -> None:
        """Test _count_tokens returns 0 when tokenizer fails."""
        mock_anthropic_clients["encoder"].encode.side_effect = RuntimeError("Failed")
        client = ClaudeClient(
            api_key=TEST_API_KEY,
            model=TEST_MODEL,
        )

        prompt = "This is a test prompt."
        token_count = client._count_tokens(prompt)  # pyright: ignore[reportPrivateUsage]
        assert token_count == 0

    def test_system_message(self, claude_client: ClaudeClient) -> None:
        """Test _system_message returns expected system prompt."""
        message = claude_client._system_message()  # pyright: ignore[reportPrivateUsage]
        log.debug("System message: %s", message)
        assert "medieval" in message.lower()
        assert "proper nouns" in message.lower()

    @pytest.mark.parametrize(
        ("prompt", "match_pattern"),
        [
            ("", r"(?i)prompt must not be empty"),
            ("   ", r"(?i)prompt must not be empty"),
            ("\n\t", r"(?i)prompt must not be empty"),
        ],
    )
    def test_validate_prompt_raises_for_empty(
        self,
        claude_client: ClaudeClient,
        prompt: str,
        match_pattern: str,
    ) -> None:
        """Test _validate_prompt raises ValueError for empty prompts."""
        with pytest.raises(ValueError, match=match_pattern) as exc_info:
            claude_client._validate_prompt(prompt)  # pyright: ignore[reportPrivateUsage]

        log.debug("ValueError raised as expected: %s", exc_info.value)

    def test_message_payload_structure(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test _message_payload returns correct payload structure."""
        payload = claude_client._message_payload("Test prompt")  # pyright: ignore[reportPrivateUsage]

        log.debug("Message payload: %s", payload)

        assert payload["model"] == TEST_MODEL
        assert payload["messages"] == [{"role": "user", "content": "Test prompt"}]
        assert payload["max_tokens"] == ClaudeClient.MAX_ALLOWED_TOKENS
        assert payload["temperature"] == ClaudeClient.DEFAULT_TEMPERATURE
        assert payload["top_p"] == 1.0
        assert payload["top_k"] == 1
        assert payload["stream"] is False
        assert "system" in payload

    def test_message_payload_with_overrides(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test _message_payload accepts max_tokens and temperature overrides."""
        custom_max_tokens = 5000
        custom_temperature = 0.3

        payload = claude_client._message_payload(  # pyright: ignore[reportPrivateUsage]
            "Test prompt",
            max_tokens=custom_max_tokens,
            temperature=custom_temperature,
        )

        log.debug("Message payload with overrides: %s", payload)

        assert payload["max_tokens"] == custom_max_tokens
        assert payload["temperature"] == custom_temperature

    def test_handle_auth_error(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test _handle_auth_error returns AuthenticationError."""
        exc = Exception("Invalid API key")
        result = claude_client._handle_auth_error(exc, operation="test_op")  # pyright: ignore[reportPrivateUsage]

        assert isinstance(result, AuthenticationError)
        assert "authentication failed" in str(result).lower()
        assert result.client_type == "claude"
        assert result.operation == "test_op"

    def test_handle_rate_limit_error(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test _handle_rate_limit_error returns RateLimitError."""
        exc = Exception("Rate limit exceeded")
        result = claude_client._handle_rate_limit_error(exc, operation="test_op")  # pyright: ignore[reportPrivateUsage]

        assert isinstance(result, LLMClientError)
        assert "rate limit exceeded" in str(result).lower()
        assert result.client_type == "claude"
        assert result.operation == "test_op"
