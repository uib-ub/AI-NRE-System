"""Processing-related exceptions for AI NER System."""


class ProcessingError(Exception):
    """Base exception for processing-related errors."""

    def __init__(self, message: str, brevid: str = None) -> None:
        """Initialize ProcessingError.

        Args:
            message: Error message.
            brevid: Optional brevid identifier related to the error.
        """
        super().__init__(message)
        self.brevid = brevid


class ValidationError(ProcessingError):
    """Exception raised when data validation fails."""
    def __init__(
            self,
            message: str,
            brevid: str = None,
            missing_fields: list[str] = None,
    ) -> None:
        """Initialize ValidationError.

        Args:
            message: Error message.
            brevid: Optional record identifier.
            missing_fields: Optional list of missing required fields.
        """
        super().__init__(message, brevid)
        self.missing_fields = missing_fields or []


class LLMResponseError(ProcessingError):
    """Exception raised when LLM response parsing fails."""

    def __init__(
            self,
            message: str,
            brevid: str = None,
            response_text: str = None,
    ) -> None:
        """Initialize LLMResponseError.

        Args:
            message: Error message.
            brevid: Optional record identifier.
            response_text: Optional LLM response text that caused the error.
        """
        super().__init__(message, brevid)
        self.response_text = response_text


class ParseError(ProcessingError):
    """Exception raised when parsing LLM response fails."""

    def __init__(
            self,
            message: str,
            brevid: str = None,
            parse_type: str = None,
            content: str = None,
    ) -> None:
        """Initialize ParseError.

        Args:
            message: Error message.
            brevid: Optional record identifier.
            parse_type: Optional type of parsing that failed (e.g., 'json').
            content: Optional content that failed to parse.
        """
        super().__init__(message, brevid)
        self.parse_type = parse_type
        self.content = content

class BatchProcessingError(ProcessingError):
    """Exception for batch processing failures."""
