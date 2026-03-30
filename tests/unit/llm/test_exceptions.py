"""Unit tests for llm.exceptions module.

Tests cover:
- Exception creation with various parameters
- String formatting for each exception type
- is_retryable() logic for APIError
- Exception inheritance hierarchy
- Edge cases (None values, empty strings)
"""

from __future__ import annotations

import logging

import pytest

from ai_ner_system.llm.exceptions import (
    APIError,
    AuthenticationError,
    BatchProcessingError,
    BatchTimeoutError,
    LLMClientError,
    LLMConnectionError,
    RateLimitError,
)

log = logging.getLogger(__name__)


class TestLLMClientError:
    """Tests for base LLMClientError exception."""

    def test_basic_creation(self) -> None:
        """Test creating LLMClientError with message only."""
        error = LLMClientError("Test error message")

        log.debug("Created LLMClientError: %s", error)

        assert str(error) == "Test error message"
        assert error.client_type is None
        assert error.operation is None

    @pytest.mark.parametrize(
        ("err_msg", "client_type", "operation"),
        [
            ("Test error message", "claude", "single_call"),
            ("Test error message", "ollama", "batch_processing"),
            ("Test error message", "claude", ""),
            ("Test error message", "", "single_call"),
            ("Test error message", "", ""),
            ("Test error message", None, None),
        ],
    )
    def test_basic_creation_with_params(
        self, err_msg: str, client_type: str | None, operation: str | None
    ) -> None:
        """Test creating LLMClientError with message and parameters.

        Args:
            err_msg: The error message.
            client_type: The type of LLM client.
            operation: The operation being performed.
        """
        error = LLMClientError(
            err_msg,
            client_type=client_type,
            operation=operation,
        )

        log.debug("Created LLMClientError with params: %s", error)

        assert err_msg in str(error)
        assert error.client_type == client_type
        assert error.operation == operation
        assert (
            f"Client: {client_type}" not in str(error)
            if not client_type
            else f"Client: {client_type}" in str(error)
        )
        assert (
            f"Operation: {operation}" not in str(error)
            if not operation
            else f"Operation: {operation}" in str(error)
        )

    def test_inheritance(self) -> None:
        """Test LLMClientError inherits from Exception."""
        error = LLMClientError("Test")
        assert isinstance(error, Exception)


class TestAPIError:
    """Tests for APIError exception."""

    def test_basic_creation(self) -> None:
        """Test creating APIError with message only."""
        error = APIError("API error occurred")

        assert "API error occurred" in str(error)
        assert error.status_code is None
        assert error.response_text is None
        assert error.request_id is None

    @pytest.mark.parametrize(
        ("status_code", "response_text", "request_id"),
        [
            (500, "Internal Server Error", "req-001"),
            (503, "Service Unavailable", "req-002"),
        ],
    )
    def test_basic_creation_with_params(
        self, status_code: int, response_text: str, request_id: str
    ) -> None:
        """Test creating APIError with all parameters.

        Args:
            status_code: The HTTP status code.
            response_text: The response text from the API.
            request_id: The request ID from the API.
        """
        error = APIError(
            "Request failed",
            client_type="claude",
            operation="single_call",
            status_code=status_code,
            response_text=response_text,
            request_id=request_id,
        )

        log.debug("Created APIError with params: %s", error)

        assert "Request failed" in str(error)
        assert f"Status: {status_code}" in str(error)
        assert f"Request ID: {request_id}" in str(error)
        assert f"Response: {response_text}" in str(error)
        assert error.status_code == status_code
        assert error.response_text == response_text
        assert error.request_id == request_id

    @pytest.mark.parametrize(
        ("err_msg", "status_code", "expected_retryable"),
        [
            ("Rate limited", 429, True),
            ("Timeout", 408, True),
            ("Internal server error", 500, True),
            ("Bad gateway", 502, True),
            ("Service unavailable", 503, True),
            ("Network connect timeout error", 599, True),
            ("Bad request", 400, False),
            ("Unauthorized", 401, False),
            ("Forbidden", 403, False),
            ("Not found", 404, False),
            ("Unknown error", None, False),
        ],
    )
    def test_is_retryable(
        self, err_msg: str, status_code: int | None, expected_retryable: bool
    ) -> None:
        """Test is_retryable method for various status codes.

        Args:
            err_msg: The error message.
            status_code: The HTTP status code.
            expected_retryable: Expected result of is_retryable().
        """
        error = APIError(err_msg, status_code=status_code)
        assert error.is_retryable() is expected_retryable

    def test_inheritance(self) -> None:
        """Test APIError inherits from LLMClientError."""
        error = APIError("Test")
        assert isinstance(error, LLMClientError)
        assert isinstance(error, Exception)


class TestLLMConnectionError:
    """Tests for LLMConnectionError exception."""

    def test_basic_creation(self) -> None:
        """Test creating LLMConnectionError with message only."""
        error = LLMConnectionError("Connection failed")

        assert "Connection failed" in str(error)
        assert error.endpoint is None

    def test_with_endpoint(self) -> None:
        """Test creating LLMConnectionError with endpoint."""
        error = LLMConnectionError(
            "Failed to connect",
            client_type="ollama",
            operation="single_call",
            endpoint="http://localhost:11434",
        )

        log.debug("Created LLMConnectionError with endpoint: %s", error)

        assert "Failed to connect" in str(error)
        assert "Endpoint: http://localhost:11434" in str(error)
        assert error.endpoint == "http://localhost:11434"

    def test_inheritance(self) -> None:
        """Test LLMConnectionError inherits from LLMClientError."""
        error = LLMConnectionError("Test")
        assert isinstance(error, LLMClientError)
        assert isinstance(error, Exception)


class TestAuthenticationError:
    """Tests for AuthenticationError exception."""

    def test_basic_creation(self) -> None:
        """Test creating AuthenticationError with message only."""
        error = AuthenticationError("Invalid API key")

        log.debug("Created AuthenticationError: %s", error)

        assert "Invalid API key" in str(error)

    def test_with_params(self) -> None:
        """Test creating AuthenticationError with params."""
        error = AuthenticationError(
            "API key expired",
            client_type="claude",
            operation="single_call",
        )

        log.debug("Created AuthenticationError: %s", error)

        assert "API key expired" in str(error)
        assert error.client_type == "claude"
        assert error.operation == "single_call"

    def test_inheritance(self) -> None:
        """Test AuthenticationError inherits from LLMClientError."""
        error = AuthenticationError("Test")
        assert isinstance(error, LLMClientError)
        assert isinstance(error, Exception)


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_basic_creation(self) -> None:
        """Test creating RateLimitError with message only."""
        error = RateLimitError("Rate limit exceeded")

        assert "Rate limit exceeded" in str(error)
        assert error.retry_after is None
        assert error.limit_type is None
        # RateLimitError should have 429 status code
        assert error.status_code == 429

    def test_with_retry_after(self) -> None:
        """Test creating RateLimitError with retry_after."""
        error = RateLimitError(
            "Too many requests",
            retry_after=60,
            limit_type="requests",
        )

        log.debug("Created RateLimitError: %s", error)

        assert "Too many requests" in str(error)
        assert "Retry After: 60s" in str(error)
        assert "Limit Type: requests" in str(error)
        assert error.retry_after == 60
        assert error.limit_type == "requests"
        assert error.status_code == 429

    def test_is_retryable(self) -> None:
        """Test RateLimitError is always retryable (429)."""
        error = RateLimitError("Rate limited")
        assert error.is_retryable() is True

    def test_inheritance(self) -> None:
        """Test RateLimitError inherits from APIError."""
        error = RateLimitError("Test")
        assert isinstance(error, APIError)
        assert isinstance(error, LLMClientError)


class TestBatchTimeoutError:
    """Tests for BatchTimeoutError exception."""

    def test_basic_creation(self) -> None:
        """Test creating BatchTimeoutError with message only."""
        error = BatchTimeoutError("Batch timed out")

        assert "Batch timed out" in str(error)
        assert error.batch_id is None
        assert error.timeout_seconds is None

    def test_with_params(self) -> None:
        """Test creating BatchTimeoutError with batch context."""
        error = BatchTimeoutError(
            "Batch processing timeout",
            client_type="claude",
            operation="batch_waiting",
            batch_id="batch_123",
            timeout_seconds=3600,
        )

        log.debug("Created BatchTimeoutError: %s", error)

        assert error.batch_id == "batch_123"
        assert error.timeout_seconds == 3600
        assert "Batch processing timeout" in str(error)
        assert "Batch ID: batch_123" in str(error)
        assert "Timeout: 3600s" in str(error)

    def test_inheritance(self) -> None:
        """Test BatchTimeoutError inherits from LLMClientError."""
        error = BatchTimeoutError("Test")
        assert isinstance(error, LLMClientError)


class TestBatchProcessingError:
    """Tests for BatchProcessingError exception."""

    def test_basic_creation(self) -> None:
        """Test creating BatchProcessingError with message only."""
        error = BatchProcessingError("Batch failed")

        assert "Batch failed" in str(error)
        assert error.batch_id is None
        assert error.failed_requests == []
        assert "Failed Requests:" not in str(error)

    def test_with_failed_requests(self) -> None:
        """Test creating BatchProcessingError with failed requests."""
        error = BatchProcessingError(
            "Some requests failed",
            client_type="ollama",
            operation="single_call",
            batch_id="batch_123",
            failed_requests=["req_1", "req_2", "req_3"],
        )

        log.debug("Created BatchProcessingError: %s", error)

        assert error.batch_id == "batch_123"
        assert error.client_type == "ollama"
        assert error.operation == "single_call"
        assert error.failed_requests == ["req_1", "req_2", "req_3"]
        assert "Some requests failed" in str(error)
        assert "Operation: single_call" in str(error)
        assert "Batch ID: batch_123" in str(error)
        assert f"Failed Requests: {len(error.failed_requests)}" in str(error)

    def test_inheritance(self) -> None:
        """Test BatchProcessingError inherits from LLMClientError."""
        error = BatchProcessingError("Test")
        assert isinstance(error, LLMClientError)
