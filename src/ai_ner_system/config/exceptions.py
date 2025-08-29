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