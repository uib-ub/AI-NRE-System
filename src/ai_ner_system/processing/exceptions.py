"""Processing-related exceptions for AI NER System.

Defines a small hierarchy of exceptions used during data validation,
LLM response handling, parsing, and batch orchestration in the
processing pipeline.
"""

from __future__ import annotations


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
            batch_ID: Optional identifier for the batch that caused the error.
        """
        super().__init__(message, operation=operation)
        self.batch_id = batch_id
