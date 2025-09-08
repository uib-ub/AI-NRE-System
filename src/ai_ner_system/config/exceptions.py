"""Configuration-related exceptions for AI NER System."""


class ConfigError(Exception):
    """Base exception for configuration-related errors."""

    def __init__(self, message: str, config_key: str = None) -> None:
        """Initialize ConfigError.

        Args:
            message: Error message.
            config_key: Optional key related to the configuration error.
        """
        super().__init__(message)
        self.config_key = config_key

class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails."""

    def __init__(self, message: str, missing_keys: list[str] = None) -> None:
        """Initialize ConfigValidationError.

        Args:
            message: Error message.
            missing_keys: List of missing configuration keys.

        """
        super().__init__(message)
        self.missing_keys = missing_keys or []


class FileValidationError(ConfigError):
    """Exception raised when file validation fails."""

    def __init__(self, message: str, file_path: str, config_key: str = None) -> None:
        """Initialize FileValidationError.

        Args:
            message: Error message.
            file_path: Path to the file that failed validation.
            config_key: Configuration key related to the file.
        """
        super().__init__(message, config_key)
        self.file_path = file_path


class DirectoryValidationError(ConfigError):
    """Exception raised when directory validation fails."""

    def __init__(self, message: str, directory_path: str, config_key: str = None) -> None:
        """Initialize DirectoryValidationError.

        Args:
            message: Error message.
            directory_path: Path to the directory that failed validation.
            config_key: Configuration key related to the directory.
        """
        super().__init__(message, config_key)
        self.directory_path = directory_path
