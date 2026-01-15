"""Unit tests for llm.ollama_client module.

Tests cover:
- OllamaClient initialization and validation
- Header and payload building
- Prompt validation
- Synchronous call method
- Asynchronous call_async method
- Error handling (timeout, connection, API errors)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import requests

from ai_ner_system.llm.exceptions import APIError, LLMClientError, LLMConnectionError
from ai_ner_system.llm.ollama_client import OllamaClient

log = logging.getLogger(__name__)

# Test token constant to avoid S106 warnings
TEST_TOKEN = "test-token"


class TestOllamaClientInit:
    """Tests for OllamaClient initialization."""

    def test_init_with_valid_params(self) -> None:
        """Test successful initialization with valid parameters."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        log.debug("Created OllamaClient: %s", client)

        assert client.endpoint == "http://localhost:11434/api/generate"
        assert client.token == TEST_TOKEN
        assert client.model == "llama3.2"
        assert client.timeout == 10800.0  # default 3 hours
        assert client.temperature == 0.0  # default

    def test_init_with_custom_timeout_and_temperature(self) -> None:
        """Test initialization with custom timeout and temperature."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
            timeout=3600.0,
            temperature=0.7,
        )

        assert client.timeout == 3600.0
        assert client.temperature == 0.7

    def test_init_strips_trailing_slash_from_endpoint(self) -> None:
        """Test endpoint trailing slash is stripped."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate/",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        assert client.endpoint == "http://localhost:11434/api/generate"

    @pytest.mark.parametrize(
        ("endpoint", "token", "model", "match_pattern"),
        [
            ("", "token", "model", r"(?i)endpoint must be provided"),
            (None, "token", "model", r"(?i)endpoint must be provided"),
            ("http://localhost", "", "model", r"(?i)token must be provided"),
            ("http://localhost", None, "model", r"(?i)token must be provided"),
            ("http://localhost", "token", "", r"(?i)model.*provided"),
            ("http://localhost", "token", None, r"(?i)model.*provided"),
        ],
    )
    def test_init_raises_for_missing_required_params(
        self,
        endpoint: str | None,
        token: str | None,
        model: str | None,
        match_pattern: str,
    ) -> None:
        """Test initialization raises ValueError for missing required params.

        Args:
            endpoint: The endpoint URL.
            token: The authentication token.
            model: The model name.
            match_pattern: Regex pattern to match error message.
        """
        with pytest.raises(ValueError, match=match_pattern):
            OllamaClient(
                endpoint=endpoint,  # type: ignore[arg-type]
                token=token,  # type: ignore[arg-type]
                model=model,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        ("timeout", "temperature", "match_pattern"),
        [
            (0.0, 0.5, r"(?i)timeout must be > 0"),
            (-1.0, 0.5, r"(?i)timeout must be > 0"),
            (100.0, -0.1, r"(?i)temperature must be in"),
            (100.0, 1.1, r"(?i)temperature must be in"),
        ],
    )
    def test_init_raises_for_invalid_timeout_or_temperature(
        self,
        timeout: float,
        temperature: float,
        match_pattern: str,
    ) -> None:
        """Test initialization raises ValueError for invalid timeout/temperature.

        Args:
            timeout: The timeout value.
            temperature: The temperature value.
            match_pattern: Regex pattern to match error message.
        """
        with pytest.raises(ValueError, match=match_pattern):
            OllamaClient(
                endpoint="http://localhost:11434/api/generate",
                token=TEST_TOKEN,
                model="llama3.2",
                timeout=timeout,
                temperature=temperature,
            )


class TestOllamaClientProperties:
    """Tests for OllamaClient properties."""

    def test_client_type(self) -> None:
        """Test client_type returns 'ollama'."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )
        assert client.client_type == "ollama"

    def test_supports_async_batch_false(self) -> None:
        """Test supports_async_batch returns False."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )
        assert client.supports_async_batch() is False


class TestOllamaClientHelpers:
    """Tests for OllamaClient helper methods."""

    def test_build_headers(self) -> None:
        """Test _build_headers returns correct headers."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        headers = client._build_headers()  # pyright: ignore[reportPrivateUsage]

        assert headers["Authorization"] == f"Bearer {TEST_TOKEN}"
        assert headers["Content-Type"] == "application/json"

    def test_build_payload(self) -> None:
        """Test _build_payload returns correct payload structure."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
            temperature=0.5,
        )

        payload = client._build_payload("Test prompt")  # pyright: ignore[reportPrivateUsage]

        assert payload["model"] == "llama3.2"
        assert payload["prompt"] == "Test prompt"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["top_p"] == 1.0
        assert payload["options"]["top_k"] == 1
        assert payload["options"]["seed"] == 42

    @pytest.mark.parametrize(
        ("prompt", "match_pattern"),
        [
            ("", r"(?i)prompt must not be empty"),
            ("   ", r"(?i)prompt must not be empty"),
            ("\n\t", r"(?i)prompt must not be empty"),
        ],
    )
    def test_validate_prompt_raises_for_empty(
        self, prompt: str, match_pattern: str
    ) -> None:
        """Test _validate_prompt raises ValueError for empty prompts.

        Args:
            prompt: The prompt to validate.
            match_pattern: Regex pattern to match error message.
        """
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        with pytest.raises(ValueError, match=match_pattern) as exc_info:
            client._validate_prompt(prompt)  # pyright: ignore[reportPrivateUsage]

        assert "Prompt must not be empty" in str(exc_info.value)

    def test_extract_text_from_json_success(self) -> None:
        """Test _extract_text_from_json extracts response text."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        result = client._extract_text_from_json({"response": "Hello, world!"})  # pyright: ignore[reportPrivateUsage]

        assert result == "Hello, world!"

    @pytest.mark.parametrize(
        "response_data",
        [
            {},
            {"response": ""},
            {"response": None},
            {"other_field": "value"},
        ],
    )
    def test_extract_text_from_json_raises_for_invalid(
        self, response_data: dict[str, Any]
    ) -> None:
        """Test _extract_text_from_json raises APIError for invalid response.

        Args:
            response_data: The response data to test.
        """
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        with pytest.raises(
            APIError, match=r"(?i)invalid or empty response"
        ) as exc_info:
            client._extract_text_from_json(response_data)  # pyright: ignore[reportPrivateUsage]

        log.debug("APIError raised: %s", exc_info.value)
        assert "Invalid or empty response payload" in str(exc_info.value)
        assert exc_info.value.client_type == "ollama"
        assert exc_info.value.operation == "extract_response"


class TestOllamaClientCall:
    """Tests for OllamaClient.call() synchronous method."""

    def test_call_success(self) -> None:
        """Test successful synchronous call."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Generated text"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = client.call("Test prompt")

        # Verify result
        assert result == "Generated text"

        # Verify requests.post was called with correct arguments
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": "Test prompt",
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "seed": 42,
                },
            },
            headers={
                "Authorization": f"Bearer {TEST_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=10800.0,
        )

        # Verify response handling
        mock_response.raise_for_status.assert_called_once()
        mock_response.json.assert_called_once()

    def test_call_raises_for_empty_prompt(self) -> None:
        """Test call raises ValueError for empty prompt."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        with pytest.raises(
            ValueError, match=r"(?i)prompt must not be empty"
        ) as exc_info:
            client.call("")

        assert "Prompt must not be empty" in str(exc_info.value)

    def test_call_timeout_error(self) -> None:
        """Test call raises APIError on timeout."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
            timeout=30,
        )

        with (
            patch(
                "requests.post",
                side_effect=requests.exceptions.Timeout("Connection timed out"),
            ),
            pytest.raises(APIError, match=r"(?i)timed out") as exc_info,
        ):
            client.call("Test prompt")

        log.debug("APIError raised: %s", exc_info.value)
        assert "timed out after 30 seconds" in str(exc_info.value)
        assert exc_info.value.client_type == "ollama"
        assert exc_info.value.operation == "single_call"

    def test_call_connection_error(self) -> None:
        """Test call raises LLMConnectionError on connection failure."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        with (
            patch(
                "requests.post",
                side_effect=requests.exceptions.ConnectionError("Connection refused"),
            ),
            pytest.raises(
                LLMConnectionError, match=r"(?i)failed to connect"
            ) as exc_info,
        ):
            client.call("Test prompt")

        log.debug("LLMConnectionError raised: %s", exc_info.value)

        assert exc_info.value.endpoint == "http://localhost:11434/api/generate"
        assert exc_info.value.client_type == "ollama"
        assert exc_info.value.operation == "single_call"

    def test_call_request_exception(self) -> None:
        """Test call raises APIError on generic request exception."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        with (
            patch(
                "requests.post",
                side_effect=requests.exceptions.RequestException("Request failed"),
            ),
            pytest.raises(APIError, match=r"(?i)request failed") as exc_info,
        ):
            client.call("Test prompt")

        assert "Request failed" in str(exc_info.value)
        assert exc_info.value.client_type == "ollama"
        assert exc_info.value.operation == "single_call"

    def test_call_json_decode_error(self) -> None:
        """Test call raises APIError on invalid JSON response."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        with (
            patch("requests.post", return_value=mock_response) as mock_post,
            pytest.raises(APIError, match=r"(?i)invalid json") as exc_info,
        ):
            client.call("Test prompt")

        assert "Invalid JSON response from Ollama API" in str(exc_info.value)
        assert "Invalid JSON" in str(exc_info.value)
        assert exc_info.value.client_type == "ollama"
        assert exc_info.value.operation == "single_call"

        # Verify request was made and status was checked before JSON decode failed
        mock_post.assert_called_once()
        mock_response.raise_for_status.assert_called_once()
        mock_response.json.assert_called_once()

    def test_call_unexpected_exception(self) -> None:
        """Test call raises LLMClientError on unexpected exception."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        with (
            patch("requests.post", side_effect=RuntimeError("Unexpected")),
            pytest.raises(
                LLMClientError, match=r"(?i)ollama api call failed"
            ) as exc_info,
        ):
            client.call("Test prompt")

        log.debug("LLMClientError raised: %s", exc_info.value)

        assert "Ollama API call failed" in str(exc_info.value)
        assert "Unexpected" in str(exc_info.value)
        assert exc_info.value.client_type == "ollama"
        assert exc_info.value.operation == "single_call"


class TestOllamaClientCallAsync:
    """Tests for OllamaClient.call_async() asynchronous method."""

    @pytest.mark.asyncio
    async def test_call_async_success(self) -> None:
        """Test successful asynchronous call."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
            timeout=30.0,
        )

        # Create mock response with async json method
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={"response": "Async generated text"}
        )

        # Create mock for session.post() async context manager
        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__.return_value = mock_response
        mock_post_cm.__aexit__.return_value = None

        # Create mock for ClientSession async context manager
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post.return_value = mock_post_cm

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cm.__aexit__.return_value = None

        with patch("aiohttp.ClientSession", return_value=mock_session_cm):
            result = await client.call_async("Test prompt")

        log.debug("Async call result: %s", result)

        # Verify result
        assert result == "Async generated text"

        # Verify session.post was called with correct arguments
        mock_session.post.assert_called_once_with(
            url="http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": "Test prompt",
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "seed": 42,
                },
            },
            headers={
                "Authorization": f"Bearer {TEST_TOKEN}",
                "Content-Type": "application/json",
            },
        )

        # Verify response handling
        mock_response.raise_for_status.assert_called_once()
        mock_response.json.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_call_async_raises_for_empty_prompt(self) -> None:
        """Test call_async raises ValueError for empty prompt."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        with pytest.raises(
            ValueError, match=r"(?i)prompt must not be empty"
        ) as exc_info:
            await client.call_async("")

        log.debug("ValueError raised: %s", exc_info.value)
        assert "Prompt must not be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_async_timeout_error(self) -> None:
        """Test call_async raises APIError on timeout."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        # Create mock session that raises TimeoutError on post
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post.side_effect = TimeoutError("Request timed out")

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cm.__aexit__.return_value = None

        with (
            patch("aiohttp.ClientSession", return_value=mock_session_cm),
            pytest.raises(APIError, match=r"(?i)timed out") as exc_info,
        ):
            await client.call_async("Test prompt")

        log.debug("APIError raised: %s", exc_info.value)
        assert "API request timed out" in str(exc_info.value)
        assert exc_info.value.client_type == "ollama"
        assert exc_info.value.operation == "async_single_call"

    @pytest.mark.asyncio
    async def test_call_async_connection_error(self) -> None:
        """Test call_async raises LLMConnectionError on connection failure."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        # Create mock session that raises ClientConnectorError on post
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post.side_effect = aiohttp.ClientConnectorError(
            MagicMock(), OSError("Connection refused")
        )

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cm.__aexit__.return_value = None

        with (
            patch("aiohttp.ClientSession", return_value=mock_session_cm),
            pytest.raises(
                LLMConnectionError, match=r"(?i)failed to connect"
            ) as exc_info,
        ):
            await client.call_async("Test prompt")

        log.debug("LLMConnectionError raised: %s", exc_info.value)
        assert exc_info.value.endpoint == "http://localhost:11434/api/generate"
        assert exc_info.value.client_type == "ollama"
        assert exc_info.value.operation == "async_single_call"

    @pytest.mark.asyncio
    async def test_call_async_client_error(self) -> None:
        """Test call_async raises APIError on client error."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        # Create mock session that raises ClientError on post
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post.side_effect = aiohttp.ClientError("Client error")

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cm.__aexit__.return_value = None

        with (
            patch("aiohttp.ClientSession", return_value=mock_session_cm),
            pytest.raises(APIError, match=r"(?i)client error") as exc_info,
        ):
            await client.call_async("Test prompt")

        log.debug("APIError raised: %s", exc_info.value)
        assert "Client error" in str(exc_info.value)
        assert exc_info.value.client_type == "ollama"
        assert exc_info.value.operation == "async_single_call"

    @pytest.mark.asyncio
    async def test_call_async_unexpected_exception(self) -> None:
        """Test call_async raises LLMClientError on unexpected exception."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        # Create mock session that raises RuntimeError on post
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post.side_effect = RuntimeError("Unexpected error")

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cm.__aexit__.return_value = None

        with (
            patch("aiohttp.ClientSession", return_value=mock_session_cm),
            pytest.raises(
                LLMClientError, match=r"(?i)async api call failed"
            ) as exc_info,
        ):
            await client.call_async("Test prompt")

        log.debug("LLMClientError raised: %s", exc_info.value)
        assert "Unexpected error" in str(exc_info.value)
        assert exc_info.value.client_type == "ollama"
        assert exc_info.value.operation == "async_single_call"

    @pytest.mark.asyncio
    async def test_call_async_cancelled_error_propagates(self) -> None:
        """Test asyncio.CancelledError propagates without wrapping."""
        client = OllamaClient(
            endpoint="http://localhost:11434/api/generate",
            token=TEST_TOKEN,
            model="llama3.2",
        )

        # Create mock session that raises CancelledError on post
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.post.side_effect = asyncio.CancelledError()

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cm.__aexit__.return_value = None

        with (
            patch("aiohttp.ClientSession", return_value=mock_session_cm),
            pytest.raises(asyncio.CancelledError),
        ):
            await client.call_async("Test prompt")
