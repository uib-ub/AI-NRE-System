"""Input/Output exceptions for AI NER System."""


class IOError(Exception):
    """Base class for all I/O related exceptions in the AI NER System."""

    def __init__(self, message: str, file_path: str = None) -> None:
        """Initialize IOError

        Args:
            message: Error message.
            file_path: Optional file path related to the error.
        """
        super().__init__(message)
        self.file_path = file_path

class CSVError(IOError):
    """Exception raised for CSV file operations errors."""

    def __init__(self, message: str, file_path: str = None, line_number: int = None) -> None:
        """Initialize CSVError.

        Args:
            message: Error message.
            file_path: Optional CSV file path.
            line_number: Optional line number where error occurred.
        """
        super().__init__(message, file_path)
        self.line_number = line_number

class CSVValidationError(CSVError):
    """Exception raised when CSV validation fails."""

    def __init__(
        self,
        message: str,
        file_path: str = None,
        line_number: int = None,
        missing_columns: list[str] = None,
    ) -> None:
        """Initialize CSVValidationError.

        Args:
            message: Error message.
            file_path: Optional CSV file path.
            line_number: Optional line number where error occurred.
            missing_columns: Optional list of missing required columns.
        """
        super().__init__(message, file_path, line_number)
        self.missing_columns = missing_columns or []

class OutputError(IOError):
    """Exception raised for output operations errors."""

    def __init__(self, message: str, file_path: str = None, output_type: str = None) -> None:
        """Initialize OutputError.

        Args:
            message: Error message.
            file_path: Optional output file path.
            output_type: Optional type of output operation.
        """
        super().__init__(message, file_path)
        self.output_type = output_type