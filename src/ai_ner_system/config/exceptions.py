"""Configuration-related exceptions for AI NER System."""

from __future__ import annotations

from pathlib import Path

Pathish = str | Path  # Type alias for path-like objects


class ConfigError(Exception):
    """Base exception for configuration-related errors."""

    def __init__(self, message: str, *, config_key: str | None = None) -> None:
        """Initialize a ConfigError.

        Args:
            message: Error message.
            config_key: Optional configuration key that the error concerns.
        """
        super().__init__(message)
        self.config_key = config_key

    def __str__(self) -> str:
        base_message = super().__str__()
        if self.config_key:
            return f"{base_message} | config_key={self.config_key}"
        return base_message


class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails."""

    def __init__(self, message: str, *, missing_keys: list[str] | None = None) -> None:
        """Initialize ConfigValidationError.

        Args:
            message: Error message.
            missing_keys: List of missing configuration keys.
        """
        super().__init__(message)
        # Normalize: unique + sorted for stable messages/tests.
        self.missing_keys = sorted(set(missing_keys)) if missing_keys else []

    def __str__(self) -> str:
        base_message = super().__str__()
        if self.missing_keys:
            missing = ", ".join(self.missing_keys)
            return f"{base_message} | missing_keys=[{missing}]"
        return base_message


class FileValidationError(ConfigError):
    """Exception raised when file validation fails."""

    def __init__(
        self,
        message: str,
        *,
        config_key: str | None = None,
        file_path: Pathish,
    ) -> None:
        """Initialize FileValidationError.

        Args:
            message: Error message.
            file_path: Path to the file that failed validation.
            config_key: Configuration key related to the file.
        """
        super().__init__(message, config_key=config_key)
        self.file_path = Path(file_path)

    def __str__(self) -> str:
        base_message = super().__str__()
        return f"{base_message} | file_path={self.file_path}"


class DirectoryValidationError(ConfigError):
    """Exception raised when directory validation fails."""

    def __init__(
        self,
        message: str,
        *,
        config_key: str | None = None,
        directory_path: Pathish,
    ) -> None:
        """Initialize DirectoryValidationError.

        Args:
            message: Error message.
            directory_path: Path to the directory that failed validation.
            config_key: Configuration key related to the directory.
        """
        super().__init__(message, config_key=config_key)
        self.directory_path = Path(directory_path)

    def __str__(self) -> str:
        base_message = super().__str__()
        return f"{base_message} | directory_path={self.directory_path}"
