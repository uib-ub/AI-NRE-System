"""LLM client exceptions for AI NER System."""

class LLMClientError(Exception):
    """Base exception for LLM client errors."""

    def __init__(self, message: str, client_type: str = None) -> None:
        """Initialize LLMClientError.

        Args:
            message: Error message.
            client_type: Optional client type (e.g., 'claude', 'ollama').
        """
        super().__init__(message)
        self.client_type = client_type


class APIError(LLMClientError):
    """Exception raised for LLM API errors."""

    def __init__(
        self,
        message: str,
        client_type: str = None,
        status_code: int = None,
        response_text: str = None,
    ) -> None:
        """Initialize APIError.

        Args:
            message: Error message.
            client_type: Optional client type (e.g., 'claude', 'ollama').
            status_code: HTTP status code if applicable.
            response_text: Response text from the API if available.
        """
        super().__init__(message, client_type)
        self.status_code = status_code
        self.response_text = response_text


class BatchTimeoutError(LLMClientError):
    """Exception raised when batch processing times out."""

    def __init__(
        self,
        message: str,
        client_type: str = None,
        batch_id: str = None,
    ) -> None:
        """Initialize BatchTimeoutError.

        Args:
            message: Error message.
            client_type: Optional client type (e.g., 'claude', 'ollama').
            batch_id: Optional batch identifier.
        """
        super().__init__(message, client_type)
        self.batch_id = batch_id