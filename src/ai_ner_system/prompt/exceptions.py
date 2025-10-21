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
        parts: list[str] = []
        if self.template_file:
            parts.append(f"template: {self.template_file}")
        if self.operation:
            parts.append(f"operation: {self.operation}")

        if parts:
            return f"{base_msg} ({', '.join(parts)})"
        return base_msg


class TemplateNotFoundError(PromptError):
    """Raised when a template file cannot be found."""

    def __init__(self, template_file: Pathish) -> None:
        super().__init__(
            f"Template file not found: {template_file}",
            template_file=template_file,
            operation="load",
        )


class PromptBuildError(PromptError):
    """Raised when building a prompt from a template fails.

    This exception is raised during prompt construction when template
    formatting fails, required fields are missing, or data validation
    fails.
    """

    def __init__(
        self,
        message: str,
        template_file: Pathish | None = None,
        data_type: str | None = None,
    ) -> None:
        """Initialize PromptBuildError.

        Args:
            message: Descriptive error message.
            template_file: Optional template file path related to the error.
            data_type: Optional data type being processed (e.g., 'single', 'batch').
        """
        super().__init__(
            message,
            template_file=template_file,
            operation="build",
        )
        self.data_type = data_type

    def __str__(self) -> str:
        """Return detailed error description."""
        base_msg = super().__str__()
        if self.data_type and base_msg.endswith(")"):
            # Insert data_type before the closing parenthesis
            return f"{base_msg[:-1]}, data_type: {self.data_type})"
        return base_msg
