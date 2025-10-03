"""Exception classes for LLM client operations in AI NER System.

This module provides a comprehensive hierarchy of exception classes for handling
various error conditions that can occur during LLM client operations, including
API communication errors, batch processing failures, and timeout conditions.

The exception hierarchy follows a structured approach:
- LLMClientError: Base class for all LLM-related errors
- APIError: HTTP API communication failures
- BatchTimeoutError: Batch processing timeout conditions
- ConnectionError: Network connectivity issues
- AuthenticationError: API key/authentication failures
- RateLimitError: API rate limiting issues
"""

from __future__ import annotations

from typing import Any


class LLMClientError(Exception):
    """Base exception class for all LLM client operations.

    This serves as the root exception for all LLM-related errors, providing
    common attributes and functionality for error context tracking.
    """

    def __init__(
        self,
        message: str,
        *,
        client_type: str | None = None,
        operation: str | None = None,
    ) -> None:
        """Initialize LLMClientError with context information.

        Args:
            message: Descriptive error message.
            client_type: Type of LLM client ('claude', 'ollama', etc.).
            operation: Operation being performed when error occurred.
        """
        super().__init__(message)
        self.client_type = client_type
        self.operation = operation

    def __str__(self) -> str:
        """Return formatted error message with context."""
        parts = [super().__str__()]
        if self.client_type:
            parts.append(f'Client: {self.client_type}')
        if self.operation:
            parts.append(f'Operation: {self.operation}')
        return ' | '.join(parts)


class APIError(LLMClientError):
    """Exception for HTTP API communication errors.

    Raised when LLM API calls fail due to HTTP errors, invalid responses,
    or other API-specific issues.
    """

    def __init__(
        self,
        message: str,
        *,
        client_type: str | None = None,
        operation: str | None = None,
        status_code: int | None = None,
        response_text: str | None = None,
        request_id: str | None = None
    ) -> None:
        """Initialize APIError with detailed API context.

        Args:
            message: Descriptive error message.
            client_type: Type of LLM client ('claude', 'ollama', etc.).
            operation: Operation being performed when error occurred.
            status_code: HTTP status code from the API response.
            response_text: Raw response text from the API.
            request_id: Unique request identifier for debugging.
        """
        super().__init__(message, client_type=client_type, operation=operation)
        self.status_code = status_code
        self.response_text = response_text
        self.request_id = request_id

    def is_retryable(self) -> bool:
        """Check if the API error is potentially retryable.

        Returns:
            True if the error might succeed on retry (429/408/5xx errors, timeouts).
        """
        sc = self.status_code
        if sc is None:
            return False
        # 5xx server errors and 429 rate limits are generally retryable
        return sc == 429 or sc == 408 or (500 <= sc <= 599)


class LLMConnectionError(LLMClientError):
    """Exception for network connectivity issues.

    Raised when the client cannot establish or maintain a connection
    to the LLM service.
    """

    def __init__(
        self,
        message: str,
        *,
        client_type: str | None = None,
        operation: str | None = None,
        endpoint: str | None = None
    ) -> None:
        """Initialize LLMConnectionError with network context.

        Args:
            message: Descriptive error message.
            client_type: Type of LLM client ('claude', 'ollama', etc.).
            operation: Operation being performed when error occurred.
            endpoint: API endpoint that failed to connect.
        """
        super().__init__(message, client_type=client_type, operation=operation)
        self.endpoint = endpoint


class AuthenticationError(LLMClientError):
    """Exception for API authentication and authorization failures.

    Raised when API key is invalid, missing, or lacks required permissions.
    """

    def __init__(
        self,
        message: str,
        *,
        client_type: str | None = None,
        operation: str | None = None
    ) -> None:
        """Initialize AuthenticationError.

        Args:
            message: Descriptive error message.
            client_type: Type of LLM client ('claude', 'ollama', etc.).
            operation: Operation being performed when error occurred.
        """
        super().__init__(message, client_type=client_type, operation=operation)


class RateLimitError(APIError):
    """Exception for API rate limiting errors.

    Raised when the client exceeds the API's rate limits.
    """

    def __init__(
        self,
        message: str,
        *,
        client_type: str | None = None,
        operation: str | None = None,
        retry_after: int | None = None,
        limit_type: str | None = None
    ) -> None:
        """Initialize RateLimitError with rate limit context.

        Args:
            message: Descriptive error message.
            client_type: Type of LLM client ('claude', 'ollama', etc.).
            operation: Operation being performed when error occurred.
            retry_after: Seconds to wait before retrying (if provided by API).
            limit_type: Type of rate limit hit ('requests', 'tokens', etc.).
        """
        super().__init__(
            message,
            client_type=client_type,
            operation=operation,
            status_code=429
        )
        self.retry_after = retry_after
        self.limit_type = limit_type


class BatchTimeoutError(LLMClientError):
    """Exception for batch processing timeout conditions.

    Raised when batch operations exceed their maximum allowed processing time.
    """

    def __init__(
        self,
        message: str,
        *,
        client_type: str | None = None,
        operation: str | None = None,
        batch_id: str | None = None,
        timeout_seconds: int | None = None
    ) -> None:
        """Initialize BatchTimeoutError with timeout context.

        Args:
            message: Descriptive error message.
            client_type: Type of LLM client ('claude', 'ollama', etc.).
            operation: Operation being performed when error occurred.
            batch_id: Identifier of the batch that timed out.
            timeout_seconds: Timeout duration that was exceeded.
        """
        super().__init__(message, client_type=client_type, operation=operation)
        self.batch_id = batch_id
        self.timeout_seconds = timeout_seconds


class BatchProcessingError(LLMClientError):
    """Exception for batch processing failures.

    Raised when batch operations fail due to invalid requests, processing
    errors, or other batch-specific issues.
    """

    def __init__(
        self,
        message: str,
        *,
        client_type: str | None = None,
        operation: str | None = None,
        batch_id: str | None = None,
        failed_requests: list[str] | None = None
    ) -> None:
        """Initialize BatchProcessingError with batch context.

        Args:
            message: Descriptive error message.
            client_type: Type of LLM client ('claude', 'ollama', etc.).
            operation: Operation being performed when error occurred.
            batch_id: Identifier of the failed batch.
            failed_requests: List of request IDs that failed within the batch.
        """
        super().__init__(message, client_type=client_type, operation=operation)
        self.batch_id = batch_id
        self.failed_requests = failed_requests or []


class LLMValidationError(LLMClientError):
    """Exception for request validation failures.

    Raised when client requests fail validation before being sent to the API.
    """

    def __init__(
        self,
        message: str,
        *,
        client_type: str | None = None,
        operation: str | None = None,
        field: str | None = None,
        value: Any = None
    ) -> None:
        """Initialize LLMValidationError with validation context.

        Args:
            message: Descriptive error message.
            client_type: Type of LLM client ('claude', 'ollama', etc.).
            operation: Operation being performed when error occurred.
            field: Name of the field that failed validation.
            value: Value that failed validation.
        """
        super().__init__(message, client_type=client_type, operation=operation)
        self.field = field
        self.value = value