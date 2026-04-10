"""Exception classes for LLM client operations at the provider/API layer in AI NER System.

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

# HTTP status codes for retryable errors
_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_REQUEST_TIMEOUT = 408
_HTTP_SERVER_ERROR_MIN = 500
_HTTP_SERVER_ERROR_MAX = 599


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
        """Return formatted error message with context.

        Appends client type and operation information to log which client failed
        and during which operation, if available.

        """
        parts = [super().__str__()]
        if self.client_type:
            parts.append(f"Client: {self.client_type}")
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        return " | ".join(parts)


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
        request_id: str | None = None,
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

    # TODO: this method can be used by client code to determine
    # if an API error is retryable (e.g. 429 Too Many Requests,
    # 408 Request Timeout, 5xx Server Errors), but currently,
    # it's not used in the client code. We can consider using it in the future
    # to implement retry logic for retryable errors.
    def is_retryable(self) -> bool:
        """Check if the API error is potentially retryable.

        Returns:
            True if the error might succeed on retry (429/408/5xx errors, timeouts).
        """
        sc = self.status_code
        if sc is None:
            return False
        # 5xx server errors and 429/408 are generally retryable
        return sc in {_HTTP_TOO_MANY_REQUESTS, _HTTP_REQUEST_TIMEOUT} or (
            _HTTP_SERVER_ERROR_MIN <= sc <= _HTTP_SERVER_ERROR_MAX
        )

    def __str__(self) -> str:
        """Return formatted error message with API context."""
        parts = [super().__str__()]
        if self.status_code:
            parts.append(f"Status: {self.status_code}")
        if self.request_id:
            parts.append(f"Request ID: {self.request_id}")
        if self.response_text:
            parts.append(f"Response: {self.response_text}")
        return " | ".join(parts)


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
        endpoint: str | None = None,
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

    def __str__(self) -> str:
        """Return formatted error message with connection context."""
        parts = [super().__str__()]
        if self.endpoint:
            parts.append(f"Endpoint: {self.endpoint}")
        return " | ".join(parts)


class AuthenticationError(LLMClientError):
    """Exception for API authentication and authorization failures.

    Raised when API key is invalid, missing, or lacks required permissions.
    """

    def __init__(
        self,
        message: str,
        *,
        client_type: str | None = None,
        operation: str | None = None,
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
        limit_type: str | None = None,
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
            status_code=_HTTP_TOO_MANY_REQUESTS,
        )
        # TODO: consider using retry_after and limit_type in client code to
        # implement smarter retry logic (e.g. wait for retry_after seconds before
        # retrying if provided by API)
        self.retry_after = retry_after
        self.limit_type = limit_type

    def __str__(self) -> str:
        """Return formatted error message with rate limit context."""
        parts = [super().__str__()]
        if self.limit_type:
            parts.append(f"Limit Type: {self.limit_type}")
        if self.retry_after:
            parts.append(f"Retry After: {self.retry_after}s")
        return " | ".join(parts)


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
        timeout_seconds: float | None = None,
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

    def __str__(self) -> str:
        """Return formatted error message with timeout context."""
        parts = [super().__str__()]
        if self.batch_id:
            parts.append(f"Batch ID: {self.batch_id}")
        if self.timeout_seconds:
            parts.append(f"Timeout: {self.timeout_seconds}s")
        return " | ".join(parts)


# TODO: This exception is currently not used in the client code,
# but we can consider using it in the future to raise specific exceptions
# for batch processing failures (e.g. invalid batch request, batch processing errors
# returned by API, etc.) to provide more granular error handling for batch operations.
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
        failed_requests: list[str] | None = None,
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

    def __str__(self) -> str:
        """Return formatted error message with batch processing context."""
        parts = [super().__str__()]
        if self.batch_id:
            parts.append(f"Batch ID: {self.batch_id}")
        if self.failed_requests:
            parts.append(f"Failed Requests: {len(self.failed_requests)}")
        return " | ".join(parts)
