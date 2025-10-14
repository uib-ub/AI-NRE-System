"""Prompt-related exceptions for AI NER System."""

from pathlib import Path

Pathish = str | Path


class PromptError(Exception):
    """Base exception for prompt-related errors.

    This exception provides structured error information for prompt template
    loading, validation, and building operations.
    """

    def __init__(
        self,
        message: str,
        *,
        template_file: Pathish | None = None,
        operation: str | None = None,
    ) -> None:
        """Initialize PromptError.

        Args:
            message: Descriptive error message.
            template_file: Optional template file path related to the error.
            operation: Type of operation that failed (e.g., 'load', 'build', 'validate').
        """
        super().__init__(message)
        self.template_file = Path(template_file) if template_file else None
        self.operation = operation

    def __str__(self) -> str:
        """Return detailed error description."""
        base_msg = super().__str__()
        if self.template_file:
            return f'{base_msg} (template: {self.template_file}, operation: {self.operation})'
        return f'{base_msg} (operation: {self.operation})'


class TemplateNotFoundError(PromptError):
    """Raised when a template file cannot be found."""

    def __init__(self, template_file: Pathish) -> None:
        super().__init__(
            f'Template file not found: {template_file}',
            template_file=template_file,
            operation='load'
        )


class PromptBuildError(PromptError):
    """Raised when building a prompt from a template fails."""

    def __init__(
        self,
        message: str,
        template_file: Pathish | None = None,
        data_type: str | None = None
    ) -> None:
        super().__init__(
            message,
            template_file=template_file,
            operation='build'
        )
        self.data_type = data_type