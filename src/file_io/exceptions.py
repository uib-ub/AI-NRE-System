"""Input/Output exceptions for AI NER System."""

from __future__ import annotations

class IOError(Exception):
    """Base class for all I/O related exceptions in the AI NER System."""

    def __init__(self, message: str, file_path: str | None = None) -> None:
        """Initialize IOError

        Args:
            message: Error message.
            file_path: Optional file path related to the error.
        """
        super().__init__(message)
        self.file_path = file_path


class CSVError(IOError):
    """Exception raised for CSV file operations errors."""

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        line_number: int | None = None
    ) -> None:
        """Initialize CSVError.

        Args:
            message: Error message.
            file_path: Optional CSV file path.
            line_number: Optional line number where error occurred.
        """
        super().__init__(message, file_path)
        self.line_number = line_number


class OutputError(IOError):
    """Exception raised for output operations errors."""

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        output_type: str | None = None
    ) -> None:
        """Initialize OutputError.

        Args:
            message: Error message.
            file_path: Optional output file path.
            output_type: Optional type of output operation.
        """
        super().__init__(message, file_path)
        self.output_type = output_type


class FileValidationError(IOError):
    """Exception raised when file validation fails."""

    def __init__(
        self,
        message: str,
        file_path: str,
        validation_type: str | None = None
    ) -> None:
        """Initialize FileValidationError.

        Args:
            message: Error message.
            file_path: Path to the file that failed validation.
            validation_type: Type of validation that failed (e.g., 'existence', 'file_type', 'file_size').
        """
        super().__init__(message, file_path)
        self.validation_type = validation_type


class EncodingError(CSVError):
    """Exception raised for file encoding issues."""

    def __init__(
        self,
        message: str,
        file_path: str,
        encoding: str | None = None
    ) -> None:
        """Initialize EncodingError.

        Args:
            message: Error message.
            file_path: Path to the file with encoding issues.
            encoding: The encoding that was attempted.
        """
        super().__init__(message, file_path)
        self.encoding = encoding
