"""Unit tests for llm.claude_client module.

Tests cover:
- ClaudeClient initialization and validation
- Single call methods (sync and async)
- Batch processing methods (create, status, info, results, cancel, monitor)
- Error handling (authentication, rate limit, API errors)
- Helper methods for message parsing
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import anthropic
import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pytest_mock import MockerFixture

from ai_ner_system.llm.batch_models import BatchProgress, BatchRequest, BatchStatus
from ai_ner_system.llm.claude_client import ClaudeClient
from ai_ner_system.llm.exceptions import (
    APIError,
    AuthenticationError,
    LLMClientError,
    RateLimitError,
)

log = logging.getLogger(__name__)

# Test constants to avoid S106 warnings
TEST_API_KEY = "sk-ant-test-key-123456789"
TEST_MODEL = "claude-sonnet-4-20240307"
TEST_BATCH_ID = "msgbatch_013Zva2CMHLNnXjNJJKqJ2EF"


def httpx_request() -> httpx.Request:
    """Helper to create a dummy httpx.Request for testing."""
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def httpx_response(status_code: int) -> httpx.Response:
    """Helper to create a dummy httpx.Response for testing."""
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(status_code=status_code, request=req)


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


@pytest.fixture
def batch_requests() -> list[BatchRequest]:
    """Create a standard list of BatchRequest objects for testing."""
    return [
        BatchRequest(custom_id="req1", prompt="Test Prompt 1"),
        BatchRequest(custom_id="req2", prompt="Test Prompt 2"),
    ]


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


# =============================================================================
# TestClaudeClientCall
# =============================================================================
class TestClaudeClientCall:
    """Tests for ClaudeClient.call() synchronous method."""

    def test_call_success(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test successful synchronous call."""
        text_block = SimpleNamespace(type="text", text="Generated response.")
        mock_message = SimpleNamespace(content=[text_block])

        claude_client.client.messages.create.return_value = mock_message  # type: ignore[attr-defined]

        result = claude_client.call("Test prompt")

        log.debug("Call result: %s", result)

        assert result == "Generated response."
        claude_client.client.messages.create.assert_called_once()  # type: ignore[attr-defined]

    def test_call_raises_for_empty_prompt(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test call raises ValueError for empty prompt."""
        with pytest.raises(
            ValueError, match=r"(?i)prompt must not be empty"
        ) as exc_info:
            claude_client.call("")

        log.debug("ValueError raised as expected: %s", exc_info.value)

    def test_call_raises_api_error_for_empty_response(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test call raises LLMClientError when response is empty."""
        mock_message = SimpleNamespace(content=[])
        claude_client.client.messages.create.return_value = mock_message  # type: ignore[attr-defined]

        with pytest.raises(LLMClientError, match=r"(?i)empty response") as exc_info:
            claude_client.call("Test prompt")

        log.debug("LLMClientError raised as expected: %s", exc_info.value)
        assert "empty response received from claude api" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "single_call"

    def test_call_authentication_error(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test call raises AuthenticationError on auth failure."""
        claude_client.client.messages.create.side_effect = (  # type: ignore[attr-defined]
            anthropic.AuthenticationError(
                message="claude authentication failed",
                response=httpx_response(status_code=401),
                body=None,
            )
        )

        with pytest.raises(
            AuthenticationError, match=r"(?i)authentication failed"
        ) as exc_info:
            claude_client.call("Test prompt")

        log.debug("AuthenticationError raised as expected: %s", exc_info.value)
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "single_call"
        assert "claude authentication failed" in str(exc_info.value).lower()

    def test_call_rate_limit_error(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test call raises RateLimitError on rate limit."""
        claude_client.client.messages.create.side_effect = anthropic.RateLimitError(  # type: ignore[attr-defined]
            message="Rate limit exceeded",
            response=httpx_response(status_code=429),
            body=None,
        )

        with pytest.raises(
            RateLimitError, match=r"(?i)rate limit exceeded"
        ) as exc_info:
            claude_client.call("Test prompt")

        log.debug("RateLimitError raised as expected: %s", exc_info.value)
        assert "rate limit exceeded" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "single_call"

    def test_call_api_error(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test call raises APIError on API failure."""
        claude_client.client.messages.create.side_effect = anthropic.APIError(  # type: ignore[attr-defined]
            message="API error occurred",
            request=mocker.MagicMock(),
            body=None,
        )

        with pytest.raises(APIError, match=r"(?i)claude api error") as exc_info:
            claude_client.call("Test prompt")

        log.debug("APIError raised as expected: %s", exc_info.value)

        assert "api error" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "single_call"

    def test_call_unexpected_exception(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test call raises LLMClientError on unexpected exception."""
        claude_client.client.messages.create.side_effect = RuntimeError("Unexpected")  # type: ignore[attr-defined]

        with pytest.raises(
            LLMClientError, match=r"(?i)claude api call failed"
        ) as exc_info:
            claude_client.call("Test prompt")

        log.debug("Exception raised as expected: %s", exc_info.value)

        assert "claude api call failed" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "single_call"


# =============================================================================
# TestClaudeClientCallAsync
# =============================================================================
class TestClaudeClientCallAsync:
    """Tests for ClaudeClient.call_async() asynchronous method."""

    @pytest.mark.asyncio
    async def test_call_async_success(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test successful asynchronous call."""
        # Build a realistic Message object using SimpleNamespace
        text_block_1 = SimpleNamespace(type="text", text="Hi there! ")
        tool_block = SimpleNamespace(
            type="tool_use",
            id="toolu_123",
            name="web_search",
            input={"q": "x"},
        )
        thinking_block = SimpleNamespace(
            type="thinking",
            thinking="(internal reasoning here)",
            signature="sig_12345",  # any placeholder string is fine for unit tests
        )
        text_block_2 = SimpleNamespace(type="text", text="My name is Claude.")

        mock_message = SimpleNamespace(
            id="msg_123456",
            model="claude-sonnet-4-5-20250929",
            role="assistant",
            type="message",
            stop_reason="end_turn",
            stop_sequence=None,
            content=[text_block_1, tool_block, thinking_block, text_block_2],
            usage=SimpleNamespace(input_tokens=2095, output_tokens=503),
        )
        # Patch the async create method
        create_mock = mocker.AsyncMock(return_value=mock_message)
        claude_client.async_client.messages.create = create_mock  # type: ignore[method-assign]

        result = await claude_client.call_async("Test prompt")

        log.debug("Async call result: %s", result)

        assert result == "Hi there! My name is Claude."
        create_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_call_async_raises_for_empty_prompt(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test call_async raises ValueError for empty prompt."""
        with pytest.raises(
            ValueError, match=r"(?i)prompt must not be empty"
        ) as exc_info:
            await claude_client.call_async("")

        log.debug("ValueError raised as expected: %s", exc_info.value)

    @pytest.mark.asyncio
    async def test_call_async_raises_api_error_for_empty_response(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test call_async raises LLMClientError when response is empty."""
        # mock an empty message response
        mock_message_resp = SimpleNamespace(content=[])
        # Patch the async create method
        create_mock = mocker.AsyncMock(return_value=mock_message_resp)
        claude_client.async_client.messages.create = create_mock  # type: ignore[method-assign]

        with pytest.raises(
            LLMClientError, match=r"(?i)empty response received"
        ) as exc_info:
            await claude_client.call_async("Test prompt")

        log.debug("LLMClientError raised as expected: %s", exc_info.value)
        assert "empty response received from claude api" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "async_single_call"

    @pytest.mark.asyncio
    async def test_call_async_cancelled_error_propagates(
        self, claude_client: ClaudeClient, mocker: MockerFixture
    ) -> None:
        """Test asyncio.CancelledError propagates."""
        # Patch the async create method to raise CancelledError
        create_mock = mocker.AsyncMock(side_effect=asyncio.CancelledError())
        claude_client.async_client.messages.create = create_mock  # type: ignore[method-assign]

        with pytest.raises(asyncio.CancelledError):
            await claude_client.call_async("Test prompt")

    @pytest.mark.asyncio
    async def test_call_async_authentication_error(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test call_async raises AuthenticationError on auth failure."""
        # Patch the async create method to raise AuthenticationError
        create_mock = mocker.AsyncMock(
            side_effect=anthropic.AuthenticationError(
                message="claude authentication failed",
                response=mocker.MagicMock(status_code=401),
                body=None,
            )
        )
        claude_client.async_client.messages.create = create_mock  # type: ignore[method-assign]

        with pytest.raises(
            AuthenticationError, match=r"(?i)authentication failed"
        ) as exc_info:
            await claude_client.call_async("Test prompt")

        log.debug("AuthenticationError raised expected: %s", exc_info.value)
        assert "claude authentication failed" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "async_single_call"

    @pytest.mark.asyncio
    async def test_call_async_rate_limit_error(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test call_async raises RateLimitError on rate limit."""
        # Patch the async create method to raise RateLimitError
        create_mock = mocker.AsyncMock(
            side_effect=anthropic.RateLimitError(
                message="Rate limit exceeded",
                response=mocker.MagicMock(status_code=429),
                body=None,
            )
        )
        claude_client.async_client.messages.create = create_mock  # type: ignore[method-assign]

        with pytest.raises(
            RateLimitError, match=r"(?i)rate limit exceeded"
        ) as exc_info:
            await claude_client.call_async("Test prompt")

        log.debug("RateLimitError raised as expected: %s", exc_info.value)
        assert "rate limit exceeded" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "async_single_call"

    @pytest.mark.asyncio
    async def test_call_async_api_error(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test call_async raises APIError on API failure."""
        # Patch the async create method to raise APIError
        create_mock = mocker.AsyncMock(
            side_effect=anthropic.APIError(
                message="API error occurred",
                request=mocker.MagicMock(),
                body=None,
            )
        )
        claude_client.async_client.messages.create = create_mock  # type: ignore[method-assign]

        with pytest.raises(APIError, match=r"(?i)claude api error") as exc_info:
            await claude_client.call_async("Test prompt")

        log.debug("APIError raised as expected: %s", exc_info.value)

        assert "api error occurred" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "async_single_call"

    @pytest.mark.asyncio
    async def test_call_async_unexpected_exception(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test call_async raises LLMClientError on unexpected exception."""
        # Patch the async create method to raise RuntimeError
        create_mock = mocker.AsyncMock(side_effect=RuntimeError("Unexpected"))
        claude_client.async_client.messages.create = create_mock  # type: ignore[method-assign]

        with pytest.raises(
            LLMClientError, match=r"(?i)claude api call failed"
        ) as exc_info:
            await claude_client.call_async("Test prompt")

        log.debug("LLMClientError raised as expected: %s", exc_info.value)

        assert "claude api call failed" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "async_single_call"


# =============================================================================
# TestClaudeClientCreateBatchAsync
# =============================================================================
class TestClaudeClientCreateBatchAsync:
    """Tests for ClaudeClient.create_batch_async() method."""

    @pytest.mark.asyncio
    async def test_create_batch_async_success(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
        batch_requests: list[BatchRequest],
    ) -> None:
        """Test successful batch creation."""
        mock_batch_resp = SimpleNamespace(
            id=TEST_BATCH_ID,
            type="message_batch",
            processing_status="in_progress",
            request_counts={
                "canceled": 0,
                "errored": 0,
                "expired": 0,
                "processing": 1,
                "succeeded": 1,
            },
            results_url=f"https://api.anthropic.com/v1/messages/batches/{TEST_BATCH_ID}/results",
        )
        # Patch the batches create method
        batches_create_mock = mocker.AsyncMock(return_value=mock_batch_resp)
        claude_client.async_client.messages.batches.create = batches_create_mock  # type: ignore[method-assign]

        # Call create_batch_async
        batch_id = await claude_client.create_batch_async(batch_requests)
        log.debug("Created batch ID: %s", batch_id)

        assert batch_id == TEST_BATCH_ID
        batches_create_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_batch_async_empty_requests_raises(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test create_batch_async raises ValueError for empty requests."""
        with pytest.raises(
            ValueError, match=r"(?i)batch requests list cannot be empty"
        ) as exc_info:
            await claude_client.create_batch_async([])

        log.debug("ValueError raised as expected: %s", exc_info.value)
        assert "batch requests list cannot be empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_create_batch_async_cancelled_error_propagates(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
        batch_requests: list[BatchRequest],
    ) -> None:
        """Test asyncio.CancelledError propagates in create_batch_async."""
        # Patch the batches create method to raise CancelledError
        batches_create_mock = mocker.AsyncMock(side_effect=asyncio.CancelledError())
        claude_client.async_client.messages.batches.create = batches_create_mock  # type: ignore[method-assign]

        with pytest.raises(asyncio.CancelledError):
            await claude_client.create_batch_async(batch_requests)

    @pytest.mark.asyncio
    async def test_create_batch_async_authentication_error(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
        batch_requests: list[BatchRequest],
    ) -> None:
        """Test create_batch_async raises AuthenticationError."""
        # Patch the batches create method to raise AuthenticationError
        batches_create_mock = mocker.AsyncMock(
            side_effect=anthropic.AuthenticationError(
                message="claude authentication failed",
                response=mocker.MagicMock(status_code=401),
                body=None,
            )
        )
        claude_client.async_client.messages.batches.create = batches_create_mock  # type: ignore[method-assign]

        with pytest.raises(
            AuthenticationError, match=r"(?i)claude authentication failed"
        ) as exc_info:
            await claude_client.create_batch_async(batch_requests)

        log.debug("AuthenticationError raised as expected: %s", exc_info.value)
        assert "claude authentication failed" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "async_create_batch"

    @pytest.mark.asyncio
    async def test_create_batch_async_rate_limit_error(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
        batch_requests: list[BatchRequest],
    ) -> None:
        """Test create_batch_async raises RateLimitError."""
        # Patch the batches create method to raise RateLimitError
        batches_create_mock = mocker.AsyncMock(
            side_effect=anthropic.RateLimitError(
                message="Rate limit exceeded",
                response=mocker.MagicMock(status_code=429),
                body=None,
            )
        )
        claude_client.async_client.messages.batches.create = batches_create_mock  # type: ignore[method-assign]

        with pytest.raises(
            RateLimitError, match=r"(?i)rate limit exceeded"
        ) as exc_info:
            await claude_client.create_batch_async(batch_requests)

        log.debug("RateLimitError raised as expected: %s", exc_info.value)
        assert "rate limit exceeded" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "async_create_batch"

    @pytest.mark.asyncio
    async def test_create_batch_async_api_error(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
        batch_requests: list[BatchRequest],
    ) -> None:
        """Test create_batch_async raises APIError."""
        # Patch the batches create method to raise APIError
        batches_create_mock = mocker.AsyncMock(
            side_effect=anthropic.APIError(
                message="API error occurred",
                request=mocker.MagicMock(),
                body=None,
            )
        )
        claude_client.async_client.messages.batches.create = batches_create_mock  # type: ignore[method-assign]

        with pytest.raises(APIError, match=r"(?i)claude api error") as exc_info:
            await claude_client.create_batch_async(batch_requests)

        log.debug("APIError raised as expected: %s", exc_info.value)

        assert "api error occurred" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "async_create_batch"

    @pytest.mark.asyncio
    async def test_create_batch_async_unexpected_exception(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
        batch_requests: list[BatchRequest],
    ) -> None:
        """Test create_batch_async raises unexpected exception."""
        # Patch the batches create method to raise RuntimeError
        batches_create_mock = mocker.AsyncMock(side_effect=RuntimeError("Unexpected"))
        claude_client.async_client.messages.batches.create = batches_create_mock  # type: ignore[method-assign]

        with pytest.raises(
            LLMClientError, match=r"(?i)failed to create batch job"
        ) as exc_info:
            await claude_client.create_batch_async(batch_requests)

        log.debug("LLMClientError raised as expected: %s", exc_info.value)

        assert "failed to create batch job: unexpected" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "async_create_batch"


# =============================================================================
# TestClaudeClientGetBatchStatusAsync
# =============================================================================
class TestClaudeClientGetBatchStatusAsync:
    """Tests for ClaudeClient.get_batch_status_async() method."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("processing_status", "expected_status"),
        [
            ("in_progress", BatchStatus.IN_PROGRESS),
            ("canceling", BatchStatus.ENDED),
            ("ended", BatchStatus.ENDED),
        ],
    )
    async def test_get_batch_status(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
        processing_status: str,
        expected_status: BatchStatus,
    ) -> None:
        """Test get_batch_status_async returns correct status based on processing_status."""
        batch_id = TEST_BATCH_ID
        mock_batch_status = SimpleNamespace(
            id=batch_id,
            type="message_batch",
            processing_status=processing_status,
            results_url=None,
        )

        # Patch the batches retrieve method
        batches_retrieve_mock = mocker.AsyncMock(return_value=mock_batch_status)
        claude_client.async_client.messages.batches.retrieve = batches_retrieve_mock  # type: ignore[method-assign]
        # Call get_batch_status_async
        status = await claude_client.get_batch_status_async(batch_id)

        log.debug("Batch status for %s: %s", batch_id, status)
        assert status == expected_status
        batches_retrieve_mock.assert_awaited_once_with(batch_id)

    @pytest.mark.asyncio
    async def test_get_batch_no_batch_id_raise(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test get batch_status_async raises ValueError for empty batch_id."""
        with pytest.raises(
            ValueError, match=r"(?i)batch_id cannot be empty"
        ) as exc_info:
            await claude_client.get_batch_status_async("")

        log.debug("ValueError raised as expected: %s", exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("side_effect", "exception_type", "match_pattern", "expected_info"),
        [
            (
                asyncio.CancelledError(),
                asyncio.CancelledError,
                None,  # No error pattern since we expect the error to propagate
                None,  # No expected info since we expect the error to propagate
            ),
            (
                anthropic.AuthenticationError(
                    message="claude authentication failed",
                    response=httpx_response(status_code=401),
                    body=None,
                ),
                AuthenticationError,
                r"(?i)authentication failed",
                "claude authentication failed",
            ),
            (
                anthropic.RateLimitError(
                    message="Rate limit exceeded",
                    response=httpx_response(status_code=429),
                    body=None,
                ),
                RateLimitError,
                r"(?i)rate limit exceeded",
                "rate limit exceeded",
            ),
            (
                anthropic.APIError(
                    message="API error occurred",
                    request=httpx_request(),
                    body=None,
                ),
                APIError,
                r"(?i)api error occurred",
                "api error occurred",
            ),
            (
                RuntimeError("Unexpected error"),
                LLMClientError,
                r"(?i)failed to get batch status",
                "unexpected error",
            ),
        ],
    )
    async def test_get_batch_status_errors(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
        side_effect: Exception,
        exception_type: type[Exception],
        match_pattern: str | None,
        expected_info: str | None,
    ) -> None:
        """Test get_batch_status_async error handling."""
        # Patch the batches retrieve method to raise the specified side effect
        batches_retrieve_mock = mocker.AsyncMock(side_effect=side_effect)
        claude_client.async_client.messages.batches.retrieve = batches_retrieve_mock  # type: ignore[method-assign]

        if isinstance(side_effect, asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await claude_client.get_batch_status_async(TEST_BATCH_ID)
            log.debug("asyncio.CancelledError propagated as expected")
        else:
            with pytest.raises(exception_type, match=match_pattern) as exc_info:
                await claude_client.get_batch_status_async(TEST_BATCH_ID)

            log.debug("Exception raised as expected: %s", exc_info.value)
            if expected_info:
                assert expected_info in str(exc_info.value).lower()
                assert "client: claude" in str(exc_info.value).lower()
                assert (
                    "operation: async_get_batch_status" in str(exc_info.value).lower()
                )


# =============================================================================
# TestClaudeClientGetBatchInfoAsync
# =============================================================================
class TestClaudeClientGetBatchInfoAsync:
    """Tests for ClaudeClient.get_batch_info_async() method."""

    @pytest.mark.asyncio
    async def test_get_batch_info_success(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test get_batch_info_async returns correct info."""
        batch_id = TEST_BATCH_ID
        results_url = (
            f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"
        )

        mock_request_counts = SimpleNamespace(
            processing=5,
            succeeded=10,
            errored=2,
            canceled=0,
            expired=1,
        )

        mock_batch_info = SimpleNamespace(
            id=batch_id,
            type="message_batch",
            processing_status="ended",
            request_counts=mock_request_counts,
            created_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-02T00:00:00Z",
            ended_at="2026-01-01T12:00:00Z",
            cancel_initiated_at=None,
            results_url=results_url,
        )

        # Patch the batches retrieve method
        batches_retrieve_mock = mocker.AsyncMock(return_value=mock_batch_info)
        claude_client.async_client.messages.batches.retrieve = batches_retrieve_mock  # type: ignore[method-assign]

        # Call get_batch_info_async
        info = await claude_client.get_batch_info_async(batch_id)

        log.debug("Batch info for %s: %s", batch_id, info)

        assert info["id"] == batch_id
        assert info["type"] == "message_batch"
        assert info["processing_status"] == "ended"
        assert info["request_counts"]["processing"] == 5
        assert info["request_counts"]["succeeded"] == 10
        assert info["results_url"] == results_url
        batches_retrieve_mock.assert_awaited_once_with(batch_id)

    @pytest.mark.asyncio
    async def test_get_batch_info_no_batch_id_raise(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test get_batch_info_async raises ValueError for empty batch_id."""
        with pytest.raises(
            ValueError, match=r"(?i)batch_id cannot be empty"
        ) as exc_info:
            await claude_client.get_batch_info_async("")

        log.debug("ValueError raised as expected: %s", exc_info.value)
        assert "batch_id cannot be empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("side_effect", "exception_type", "match_pattern", "expected_info"),
        [
            (
                asyncio.CancelledError(),
                asyncio.CancelledError,
                None,  # No error pattern since we expect the error to propagate
                None,  # No expected info since we expect the error to propagate
            ),
            (
                anthropic.AuthenticationError(
                    message="Authentication failed",
                    response=httpx_response(status_code=401),
                    body=None,
                ),
                AuthenticationError,
                r"(?i)authentication failed",
                "authentication failed",
            ),
            (
                anthropic.RateLimitError(
                    message="Rate limit exceeded",
                    response=httpx_response(status_code=429),
                    body=None,
                ),
                RateLimitError,
                r"(?i)rate limit exceeded",
                "rate limit exceeded",
            ),
            (
                anthropic.APIError(
                    message="API error occurred",
                    request=httpx_request(),
                    body=None,
                ),
                APIError,
                r"(?i)api error occurred",
                "api error occurred",
            ),
            (
                RuntimeError("Unexpected error"),
                LLMClientError,
                r"(?i)failed to get batch info",
                "unexpected error",
            ),
        ],
    )
    async def test_get_batch_info_errors(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
        side_effect: Exception,
        exception_type: type[Exception],
        match_pattern: str | None,
        expected_info: str | None,
    ) -> None:
        """Test get_batch_info_async error handling."""
        # Patch the batches retrieve method to raise the specified side effect
        batches_retrieve_mock = mocker.AsyncMock(side_effect=side_effect)
        claude_client.async_client.messages.batches.retrieve = batches_retrieve_mock  # type: ignore[method-assign]

        if isinstance(side_effect, asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await claude_client.get_batch_info_async(TEST_BATCH_ID)
            log.debug("asyncio.CancelledError propagated as expected")
        else:
            with pytest.raises(exception_type, match=match_pattern) as exc_info:
                await claude_client.get_batch_info_async(TEST_BATCH_ID)

            log.debug("Exception raised as expected: %s", exc_info.value)
            if expected_info:
                assert expected_info in str(exc_info.value).lower()
                assert "client: claude" in str(exc_info.value).lower()
                assert "operation: async_get_batch_info" in str(exc_info.value).lower()


# =============================================================================
# TestClaudeClientGetBatchResultsAsync
# =============================================================================
class TestClaudeClientGetBatchResultsAsync:
    """Tests for ClaudeClient.get_batch_results_async() method."""

    @pytest.mark.asyncio
    async def test_get_batch_results_success(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test get_batch_results_async returns correct results."""
        batch_id = TEST_BATCH_ID

        # Create success batch result item
        content_text_block = SimpleNamespace(type="text", text="Response for request 1")
        message_block = SimpleNamespace(
            id=batch_id,
            content=[content_text_block],
            model="claude-sonnet-4-5-20250929",
            type="message",
            role="assistant",
            stop_reason="end_turn",
            stop_sequence=None,
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )

        result_block = SimpleNamespace(type="succeeded", message=message_block)

        result_ok = SimpleNamespace(custom_id="custom_req_1", result=result_block)

        result_error = SimpleNamespace(
            custom_id="custom_req_2",
            result=SimpleNamespace(
                type="errored",
                error=SimpleNamespace(
                    type="error",
                    error=SimpleNamespace(
                        type="rate_limit_error",
                        message="Rate limit exceeded",
                    ),
                ),
            ),
        )

        result_canceled = SimpleNamespace(
            custom_id="custom_req_3",
            result=SimpleNamespace(
                type="canceled",
            ),
        )

        result_expired = SimpleNamespace(
            custom_id="custom_req_4",
            result=SimpleNamespace(
                type="expired",
            ),
        )

        result_empty = SimpleNamespace()

        result_error_no_message = SimpleNamespace(
            custom_id="custom_req_5",
            result=SimpleNamespace(
                type="errored",
                error=SimpleNamespace(type="error"),
            ),
        )

        result_unknown_type = SimpleNamespace(
            custom_id="custom_req_6",
            result=SimpleNamespace(
                type="unknown_type",
            ),
        )

        mock_results = SimpleNamespace(
            results=[
                result_ok,
                result_error,
                result_canceled,
                result_expired,
                result_empty,
                result_error_no_message,
                result_unknown_type,
            ]
        )

        async def mock_results_iter() -> AsyncIterator[Any]:
            for res in mock_results.results:
                yield res

        claude_client._validate_batch_ready = mocker.AsyncMock(return_value=None)  # type: ignore[method-assign]

        # Patch the batches results method
        batches_results_mock = mocker.AsyncMock(return_value=mock_results_iter())
        claude_client.async_client.messages.batches.results = batches_results_mock  # type: ignore[method-assign]

        # Call get_batch_results_async
        results = await claude_client.get_batch_results_async(batch_id)

        for i, res in enumerate(results):
            log.debug("Batch result %d: %s", i, res)

        assert len(results) == 7
        assert results[0].custom_id == "custom_req_1"
        assert results[0].success is True
        assert results[0].response_text == "Response for request 1"
        assert results[0].error_message == ""

        assert results[1].custom_id == "custom_req_2"
        assert results[1].success is False
        assert results[1].response_text == ""
        assert results[1].error_message == "Rate limit exceeded"

        assert results[2].custom_id == "custom_req_3"
        assert results[2].success is False
        assert results[2].response_text == ""
        assert results[2].error_message == "Request was canceled before execution"

        assert results[3].custom_id == "custom_req_4"
        assert results[3].success is False
        assert results[3].response_text == ""
        assert (
            results[3].error_message
            == "Request expired (not processed within the batch time window)"
        )

        assert results[4].custom_id == "unknown_custom_id"
        assert results[4].success is False
        assert results[4].response_text == ""
        assert results[4].error_message == "Missing result object"

        assert results[5].custom_id == "custom_req_5"
        assert results[5].success is False
        assert results[5].response_text == ""
        assert (
            results[5].error_message == "Batch request failed: namespace(type='error')"
        )

        assert results[6].custom_id == "custom_req_6"
        assert results[6].success is False
        assert results[6].response_text == ""
        assert (
            results[6].error_message
            == "Failed to parse result: Unhandled result type: unknown_type"
        )

    @pytest.mark.asyncio
    async def test_get_batch_results_not_completed_raises(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test get_batch_results_async raises LLMClientError if batch not completed."""
        # Mock batch status to be in_progress
        mock_batch_status = SimpleNamespace(
            id=TEST_BATCH_ID,
            type="message_batch",
            processing_status="in_progress",
            results_url=None,
        )
        batches_retrieve_mock = mocker.AsyncMock(return_value=mock_batch_status)
        claude_client.async_client.messages.batches.retrieve = batches_retrieve_mock  # type: ignore[method-assign]

        with pytest.raises(LLMClientError, match=r"(?i)not completed") as exc_info:
            await claude_client.get_batch_results_async(TEST_BATCH_ID)

        log.debug("LLMClientError raised as expected: %s", exc_info.value)
        assert "not completed" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "async_get_batch_results"

    @pytest.mark.asyncio
    async def test_get_batch_results_no_results_url_raises(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test get_batch_results_async raises LLMClientError if results_url is missing."""
        # Mock batch status to be ended but with no results_url
        mock_batch_status = SimpleNamespace(
            id=TEST_BATCH_ID,
            type="message_batch",
            processing_status="ended",
            request_counts=SimpleNamespace(
                processing=0, succeeded=1, errored=0, canceled=0, expired=0
            ),
            created_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-02T00:00:00Z",
            ended_at="2026-01-01T12:00:00Z",
            cancel_initiated_at=None,
            results_url=None,
        )
        batches_retrieve_mock = mocker.AsyncMock(return_value=mock_batch_status)
        claude_client.async_client.messages.batches.retrieve = batches_retrieve_mock  # type: ignore[method-assign]

        with pytest.raises(
            LLMClientError, match=r"(?i)no results URL available"
        ) as exc_info:
            await claude_client.get_batch_results_async(TEST_BATCH_ID)

        log.debug("LLMClientError raised as expected: %s", exc_info.value)
        assert "no results url available" in str(exc_info.value).lower()
        assert exc_info.value.client_type == "claude"
        assert exc_info.value.operation == "async_get_batch_results"

    @pytest.mark.asyncio
    async def test_get_batch_results_no_batch_id_raise(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test get_batch_results_async raises ValueError for empty batch_id."""
        with pytest.raises(
            ValueError, match=r"(?i)batch_id cannot be empty"
        ) as exc_info:
            await claude_client.get_batch_results_async("")

        log.debug("ValueError raised as expected: %s", exc_info.value)
        assert "batch_id cannot be empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("side_effect", "exception_type", "match_pattern", "expected_info"),
        [
            (
                asyncio.CancelledError(),
                asyncio.CancelledError,
                None,  # No error pattern since we expect the error to propagate
                None,  # No expected info since we expect the error to propagate
            ),
            (
                anthropic.AuthenticationError(
                    message="Authentication failed",
                    response=httpx_response(status_code=401),
                    body=None,
                ),
                AuthenticationError,
                r"(?i)authentication failed",
                "authentication failed",
            ),
            (
                anthropic.RateLimitError(
                    message="Rate limit exceeded",
                    response=httpx_response(status_code=429),
                    body=None,
                ),
                RateLimitError,
                r"(?i)rate limit exceeded",
                "rate limit exceeded",
            ),
            (
                anthropic.APIError(
                    message="API error occurred",
                    request=httpx_request(),
                    body=None,
                ),
                APIError,
                r"(?i)api error occurred",
                "api error occurred",
            ),
            (
                RuntimeError("Unexpected error"),
                LLMClientError,
                r"(?i)failed to get batch results",
                "unexpected error",
            ),
        ],
    )
    async def test_get_batch_results(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
        side_effect: Exception,
        exception_type: type[Exception],
        match_pattern: str | None,
        expected_info: str | None,
    ) -> None:
        """Test get_batch_results_async error handling."""
        claude_client._validate_batch_ready = mocker.AsyncMock(return_value=None)  # type: ignore[method-assign]

        # Patch the batches results method to raise the specified side effect
        batches_results_mock = mocker.AsyncMock(side_effect=side_effect)
        claude_client.async_client.messages.batches.results = batches_results_mock  # type: ignore[method-assign]

        if isinstance(side_effect, asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await claude_client.get_batch_results_async(TEST_BATCH_ID)
            log.debug("asyncio.CancelledError propagated as expected")
        else:
            with pytest.raises(exception_type, match=match_pattern) as exc_info:
                await claude_client.get_batch_results_async(TEST_BATCH_ID)

            log.debug("Exception raised as expected: %s", exc_info.value)
            if expected_info:
                assert expected_info in str(exc_info.value).lower()
                assert "client: claude" in str(exc_info.value).lower()
                assert (
                    "operation: async_get_batch_results" in str(exc_info.value).lower()
                )


# =============================================================================
# TestClaudeClientCancelBatchAsync
# =============================================================================
class TestClaudeClientCancelBatchAsync:
    """Tests for ClaudeClient.cancel_batch_async() method."""

    @pytest.mark.asyncio
    async def test_cancel_batch_async_success(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test cancel_batch_async successfully cancels a batch."""
        batch_id = TEST_BATCH_ID
        results_url = (
            f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"
        )

        cancelled_batch = SimpleNamespace(
            id=batch_id,
            type="message_batch",
            processing_status="canceling",
            created_at="2026-01-20T18:37:24.100435Z",
            expires_at="2026-01-20T18:37:24.100435Z",
            cancel_initiated_at="2026-01-20T18:40:00.000000Z",
            ended_at=None,
            archived_at=None,
            request_counts={
                "processing": 0,
                "succeeded": 0,
                "errored": 0,
                "canceled": 2,
                "expired": 0,
            },
            results_url=results_url,
        )

        # Patch the batches cancel method to return None (indicating success)
        batches_cancel_mock = mocker.AsyncMock(return_value=cancelled_batch)
        claude_client.async_client.messages.batches.cancel = batches_cancel_mock  # type: ignore[method-assign]

        # Call cancel_batch_async
        result = await claude_client.cancel_batch_async(batch_id)

        log.debug("Batch %s canceled successfully", batch_id)
        assert result is True
        batches_cancel_mock.assert_awaited_once_with(batch_id)

    @pytest.mark.asyncio
    async def test_cancel_batch_async_no_batch_id_raise(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test cancel_batch_async raises ValueError for empty batch_id."""
        with pytest.raises(
            ValueError, match=r"(?i)batch_id cannot be empty"
        ) as exc_info:
            await claude_client.cancel_batch_async("")

        log.debug("ValueError raised as expected: %s", exc_info.value)
        assert "batch_id cannot be empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("side_effect", "exception_type", "match_pattern", "expected_info"),
        [
            (
                asyncio.CancelledError(),
                asyncio.CancelledError,
                None,  # No error pattern since we expect the error to propagate
                None,  # No expected info since we expect the error to propagate
            ),
            (
                anthropic.AuthenticationError(
                    message="Authentication failed",
                    response=httpx_response(status_code=401),
                    body=None,
                ),
                AuthenticationError,
                r"(?i)authentication failed",
                "authentication failed",
            ),
            (
                anthropic.RateLimitError(
                    message="Rate limit exceeded",
                    response=httpx_response(status_code=429),
                    body=None,
                ),
                RateLimitError,
                r"(?i)rate limit exceeded",
                "rate limit exceeded",
            ),
            (
                anthropic.APIError(
                    message="API error occurred",
                    request=httpx_request(),
                    body=None,
                ),
                APIError,
                r"(?i)api error occurred",
                "api error occurred",
            ),
            (
                RuntimeError("Unexpected error"),
                LLMClientError,
                r"(?i)failed to cancel batch",
                "unexpected error",
            ),
        ],
    )
    async def test_cancel_batch_async_errors(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
        side_effect: Exception,
        exception_type: type[Exception],
        match_pattern: str | None,
        expected_info: str | None,
    ) -> None:
        """Test cancel_batch_async error handling."""
        # Patch the batches cancel method to raise the specified side effect
        batches_cancel_mock = mocker.AsyncMock(side_effect=side_effect)
        claude_client.async_client.messages.batches.cancel = batches_cancel_mock  # type: ignore[method-assign]

        if isinstance(side_effect, asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await claude_client.cancel_batch_async(TEST_BATCH_ID)
            log.debug("asyncio.CancelledError propagated as expected")
        else:
            with pytest.raises(exception_type, match=match_pattern) as exc_info:
                await claude_client.cancel_batch_async(TEST_BATCH_ID)

            log.debug("Exception raised as expected: %s", exc_info.value)
            if expected_info:
                assert expected_info in str(exc_info.value).lower()
                assert "client: claude" in str(exc_info.value).lower()
                assert "operation: async_cancel_batch" in str(exc_info.value).lower()


# =============================================================================
# TestClaudeClientMonitorBatchProgressAsync
# =============================================================================
class TestClaudeClientMonitorBatchProgressAsync:
    """Tests for ClaudeClient.monitor_batch_progress_async() method."""

    @pytest.mark.asyncio
    async def test_monitor_batch_progress_async_yields_progress(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test monitor yields BatchProgress until ENDED."""
        batch_id = TEST_BATCH_ID
        results_url = (
            f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"
        )
        batch_num = 1
        poll_interval = 3.0

        # Mock batch status responses for in_progress -> ended
        mock_batch_in_progress = SimpleNamespace(
            id=batch_id,
            type="message_batch",
            processing_status="in_progress",
            created_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-02T00:00:00Z",
            ended_at=None,
            cancel_initiated_at=None,
            results_url=None,
            request_counts=SimpleNamespace(
                processing=5,
                succeeded=0,
                errored=0,
                canceled=0,
                expired=0,
            ),
        )

        mock_batch_ended = SimpleNamespace(
            id=batch_id,
            type="message_batch",
            processing_status="ended",
            created_at="2024-01-01T00:00:00Z",
            expires_at="2024-01-02T00:00:00Z",
            ended_at="2024-01-01T01:00:00Z",
            cancel_initiated_at=None,
            results_url=results_url,
            request_counts=SimpleNamespace(
                processing=0,
                succeeded=5,
                errored=0,
                canceled=0,
                expired=0,
            ),
        )

        # Patch the batches retrieve method
        # Note: Each iteration of monitor_batch_progress_async calls BOTH:
        #   1. get_batch_status_async -> batches.retrieve
        #   2. get_batch_info_async -> batches.retrieve
        # So we need 2 mock values per iteration:
        #   - Iteration 1: in_progress (status), in_progress (info)
        #   - Iteration 2: ended (status), ended (info)
        batches_retrieve_mock = mocker.AsyncMock(
            side_effect=[
                mock_batch_in_progress,  # get_batch_status_async (iter 1)
                mock_batch_in_progress,  # get_batch_info_async (iter 1)
                mock_batch_ended,  # get_batch_status_async (iter 2)
                mock_batch_ended,  # get_batch_info_async (iter 2)
            ]
        )
        claude_client.async_client.messages.batches.retrieve = batches_retrieve_mock  # type: ignore[method-assign]

        # Call monitor_batch_progress_async
        progress_list: list[BatchProgress] = []
        async for progress in claude_client.monitor_batch_progress_async(
            batch_num=batch_num,
            batch_id=batch_id,
            poll_interval=poll_interval,
        ):
            log.debug("Batch progress update: %s", progress)
            progress_list.append(progress)

        log.debug("Final batch progress: %s", progress_list)
        for idx, progress in enumerate(progress_list):
            log.debug("Progress %d: %s", idx, progress)

        assert len(progress_list) == 2
        assert progress_list[0].status == BatchStatus.IN_PROGRESS
        assert progress_list[0].request_counts["processing"] == 5
        assert progress_list[0].request_counts["succeeded"] == 0
        assert progress_list[1].status == BatchStatus.ENDED
        assert progress_list[1].request_counts["processing"] == 0
        assert progress_list[1].request_counts["succeeded"] == 5

    @pytest.mark.asyncio
    async def test_monitor_batch_progress_async_no_batch_id_raise(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test monitor_batch_progress_async raises ValueError for empty batch_id."""
        with pytest.raises(
            ValueError, match=r"(?i)batch_id cannot be empty"
        ) as exc_info:
            async for _ in claude_client.monitor_batch_progress_async(
                batch_num=1,
                batch_id="",
                poll_interval=3.0,
            ):
                pass

        log.debug("ValueError raised as expected: %s", exc_info.value)
        assert "batch_id cannot be empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_monitor_batch_progress_async_none_poll_interval(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test monitor_batch_progress_async handles None poll_interval."""
        batch_id = TEST_BATCH_ID

        # Patch the batches retrieve method to return a valid batch status
        mock_batch_status = SimpleNamespace(
            id=batch_id,
            type="message_batch",
            processing_status="in_progress",
            created_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-02T00:00:00Z",
            ended_at=None,
            cancel_initiated_at=None,
            results_url=None,
            request_counts=SimpleNamespace(
                processing=5,
                succeeded=0,
                errored=0,
                canceled=0,
                expired=0,
            ),
        )
        batches_retrieve_mock = mocker.AsyncMock(return_value=mock_batch_status)
        claude_client.async_client.messages.batches.retrieve = batches_retrieve_mock  # type: ignore[method-assign]

        # Call monitor_batch_progress_async with None poll_interval
        progress_list: list[BatchProgress] = []
        async for progress in claude_client.monitor_batch_progress_async(
            batch_num=1,
            batch_id=batch_id,
            poll_interval=None,  # Should default to 5.0 seconds
        ):
            log.debug("Batch progress update with None poll_interval: %s", progress)
            progress_list.append(progress)
            if len(progress_list) >= 1:
                break  # Only need one iteration to confirm it works

        log.debug("Batch progress with None poll_interval: %s", progress_list)
        assert len(progress_list) == 1
        assert progress_list[0].status == BatchStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_monitor_batch_progress_async_negative_poll_interval_raise(
        self,
        claude_client: ClaudeClient,
    ) -> None:
        """Test monitor_batch_progress_async raises ValueError for negative poll_interval."""
        with pytest.raises(
            ValueError, match=r"(?i)poll_interval must be positive"
        ) as exc_info:
            async for _ in claude_client.monitor_batch_progress_async(
                batch_num=1,
                batch_id=TEST_BATCH_ID,
                poll_interval=-1.0,
            ):
                pass

        log.debug("ValueError raised as expected: %s", exc_info.value)
        assert "poll_interval must be positive" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_monitor_batch_progress_async_error_yields_ended(
        self,
        claude_client: ClaudeClient,
        mocker: MockerFixture,
    ) -> None:
        """Test monitor_batch_progress_async_exception handling yields ENDED status."""
        poll_interval = 3.0

        # Patch the batches retrieve method to raise an API error
        batches_retrieve_mock = mocker.AsyncMock(
            side_effect=RuntimeError("Network failure")
        )
        claude_client.async_client.messages.batches.retrieve = batches_retrieve_mock  # type: ignore[method-assign]

        # Call monitor_batch_progress_async and capture the yielded progress
        progress_list: list[BatchProgress] = []
        async for progress in claude_client.monitor_batch_progress_async(
            batch_num=1,
            batch_id=TEST_BATCH_ID,
            poll_interval=poll_interval,
        ):
            log.debug("Batch progress update during error: %s", progress)
            progress_list.append(progress)

        log.debug("Batch progress after error: %s", progress_list)
        assert len(progress_list) == 1
        assert progress_list[0].status == BatchStatus.ENDED
