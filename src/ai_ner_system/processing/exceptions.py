"""Processing-related exceptions for AI NER System.

Defines a small hierarchy of exceptions used during data validation,
LLM response handling, parsing, and batch orchestration in the
processing pipeline.
"""

from __future__ import annotations

# Maximum length for truncated content in error messages
_MAX_CONTENT_LENGTH = 100


class ProcessingError(Exception):
    """Base exception for processing-related errors.

    This is the base class for all processing exceptions in the AI NER system.
    It provides common functionality for associating errors with specific records
    via the brevid identifier.
    """

    def __init__(
        self,
        message: str,
        *,
        brevid: str | None = None,
        operation: str | None = None,
    ) -> None:
        """Initialize ProcessingError.

        Args:
            message: Error message.
            brevid: Optional brevid identifier related to the error.
            operation: Operation being performed when error occurred.
        """
        super().__init__(message)
        self.brevid = brevid
        self.operation = operation

    def __str__(self) -> str:
        """Return detailed error description."""
        base_msg = super().__str__()
        parts: list[str] = []
        if self.brevid:
            parts.append(f"brevid: {self.brevid}")
        if self.operation:
            parts.append(f"operation: {self.operation}")

        if parts:
            return f"{base_msg} ({', '.join(parts)})"
        return base_msg


class ValidationError(ProcessingError):
    """Exception raised when data validation fails.

    This exception is raised when record data does not meet validation
    requirements, such as missing required fields or invalid field values.
    """

    def __init__(
        self,
        message: str,
        *,
        brevid: str | None = None,
        operation: str | None = None,
        missing_fields: list[str] | None = None,
    ) -> None:
        """Initialize ValidationError.

        Args:
            message: Error message.
            brevid: Optional record identifier.
            operation: Operation being performed when error occurred.
            missing_fields: Optional list of missing required fields.
        """
        super().__init__(message, brevid=brevid, operation=operation)
        self.missing_fields = missing_fields or []

    def __str__(self) -> str:
        """Return detailed error description."""
        base_msg = super().__str__()
        if self.missing_fields and base_msg.endswith(")"):
            fields_str = ", ".join(self.missing_fields)
            return f"{base_msg[:-1]}, missing_fields: [{fields_str}])"
        return base_msg


class LLMResponseError(ProcessingError):
    """Exception raised when LLM response parsing fails.

    This exception is raised when there are issues with the LLM response,
    such as unexpected format, empty responses, or API errors.
    """

    def __init__(
        self,
        message: str,
        *,
        brevid: str | None = None,
        operation: str | None = None,
        response_text: str | None = None,
    ) -> None:
        """Initialize LLMResponseError.

        Args:
            message: Error message.
            brevid: Optional record identifier.
            operation: Operation being performed when error occurred.
            response_text: Optional LLM response text that caused the error.
        """
        super().__init__(message, brevid=brevid, operation=operation)
        self.response_text = response_text

    def __str__(self) -> str:
        """Return detailed error description."""
        base_msg = super().__str__()
        if self.response_text and base_msg.endswith(")"):
            truncated = (
                self.response_text[:_MAX_CONTENT_LENGTH] + "..."
                if len(self.response_text) > _MAX_CONTENT_LENGTH
                else self.response_text
            )
            return f"{base_msg[:-1]}, response_text: '{truncated}')"
        return base_msg


class ParseError(ProcessingError):
    """Exception raised when parsing LLM response fails.

    This exception is raised specifically when parsing structured data
    (e.g., JSON) from LLM responses fails.
    """

    def __init__(
        self,
        message: str,
        *,
        brevid: str | None = None,
        operation: str | None = None,
        parse_type: str | None = None,
        content: str | None = None,
    ) -> None:
        """Initialize ParseError.

        Args:
            message: Error message.
            brevid: Optional record identifier.
            operation: Operation being performed when error occurred.
            parse_type: Optional type of parsing that failed (e.g., 'json').
            content: Optional content that failed to parse.
        """
        super().__init__(message, brevid=brevid, operation=operation)
        self.parse_type = parse_type
        self.content = content

    def __str__(self) -> str:
        """Return detailed error description."""
        base_msg = super().__str__()
        additional_parts: list[str] = []
        if self.parse_type:
            additional_parts.append(f"parse_type: {self.parse_type}")
        if self.content:
            truncated = (
                self.content[:_MAX_CONTENT_LENGTH] + "..."
                if len(self.content) > _MAX_CONTENT_LENGTH
                else self.content
            )
            additional_parts.append(f"content: '{truncated}'")

        if additional_parts and base_msg.endswith(")"):
            return f"{base_msg[:-1]}, {', '.join(additional_parts)})"
        return base_msg


class BatchProcessingError(ProcessingError):
    """Exception for batch processing failures.

    This exception is raised when batch processing operations fail,
    potentially affecting multiple records.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        batch_id: str | None = None,
    ) -> None:
        """Initialize BatchProcessingError.

        Args:
            message: Error message.
            operation: Operation being performed when error occurred.
            batch_id: Optional identifier for the batch that caused the error.
        """
        super().__init__(message, operation=operation)
        self.batch_id = batch_id

    def __str__(self) -> str:
        """Return detailed error description."""
        base_msg = super().__str__()
        if self.batch_id and base_msg.endswith(")"):
            return f"{base_msg[:-1]}, batch_id: {self.batch_id})"
        return base_msg
