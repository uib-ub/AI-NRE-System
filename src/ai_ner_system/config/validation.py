"""Configuration validation for AI NER System."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import ClassVar

from .exceptions import (
    ConfigError,
    ConfigValidationError,
    DirectoryValidationError,
    FileValidationError,
)
from .settings import Settings


class ConfigValidator:
    """Validates configuration settings for the AI NER System."""

    # Class constants for better maintainability and type safety
    TEMPLATE_FILES: ClassVar[dict[str, str]] = {
        "PROMPT_TEMPLATE_FILE": "prompt template",
        "BATCH_TEMPLATE_FILE": "batch template",
    }

    OUTPUT_FILES: ClassVar[dict[str, str]] = {
        "OUTPUT_TEXT_FILE": "output text file",
        "OUTPUT_TABLE_FILE": "output table file",
        "OUTPUT_STATS_FILE": "output stats file",
    }

    TEMPLATE_FILE_ATTRS: ClassVar[tuple[str, ...]] = (
        "PROMPT_TEMPLATE_FILE",
        "BATCH_TEMPLATE_FILE",
    )

    OUTPUT_FILE_ATTRS: ClassVar[tuple[str, ...]] = (
        "OUTPUT_TEXT_FILE",
        "OUTPUT_TABLE_FILE",
        "OUTPUT_STATS_FILE",
    )

    @staticmethod
    def validate_for_client(client_type: str) -> None:
        """Validate configuration for a given LLM client type.

        Args:
            client_type: client choice ('claude' or 'ollama').

        Raises:
            ConfigValidationError: If the client type is unsupported or required
            configuration keys are missing/empty.
        """
        if not client_type or not client_type.strip():
            msg = "Client type must be provided."
            raise ConfigValidationError(msg)

        client_type = client_type.strip().lower()

        try:
            # Validate client-specific configuration
            # This will raise ConfigError if client type is unsupported
            Settings.validate_client_config(client_type)

            # Validate common configuration
            Settings.validate_common_config()

            logging.info(
                "Configuration validation passed for %s client",
                client_type,
            )

        except ConfigError as e:
            raise ConfigValidationError(str(e)) from e

    @staticmethod
    def validate_file_paths() -> None:
        """Validate that all file paths are accessible and directories exist.

        Raises:
            ConfigValidationError: If any file path is invalid or inaccessible.
        """
        try:
            ConfigValidator._validate_input_file()
            ConfigValidator._validate_template_files()
            ConfigValidator._validate_output_paths_writable()
            logging.info("File path validation completed successfully")
        except (
            OSError,
            ConfigError,
            FileValidationError,
            DirectoryValidationError,
        ) as e:
            msg = f"File path validation failed: {e}"
            raise ConfigValidationError(msg) from e

    @staticmethod
    def _validate_input_file() -> None:
        """Validate that the optional input file exists, is a file, and is readable.

        Raises:
            FileValidationError: If the file is missing, not a file, or unreadable.
        """
        if not Settings.INPUT_FILE or not Settings.INPUT_FILE.strip():
            logging.warning("INPUT_FILE not configured, skipping validation")
            return  # Optional validation, will be caught by required config check

        # Validate input file exists
        input_path = Path(Settings.INPUT_FILE)
        ConfigValidator._validate_file_exists_and_readable(
            input_path,
            "INPUT_FILE",
            "Input file",
        )

        try:
            # Size check (0-byte input usually indicates misconfiguration).
            file_size = input_path.stat().st_size
            if file_size == 0:
                msg = "Input file is empty"
                raise FileValidationError(
                    msg,
                    file_path=str(input_path),
                    config_key="INPUT_FILE",
                )
        except OSError as e:
            msg = f"Cannot access input file: {e}"
            raise FileValidationError(
                msg,
                file_path=str(input_path),
                config_key="INPUT_FILE",
            ) from e

    @staticmethod
    def _validate_template_files() -> None:
        """Validate template files exist and are readable.

        Raises:
            FileValidationError: If template files are invalid.
        """
        for config_key in ConfigValidator.TEMPLATE_FILE_ATTRS:
            file_path = getattr(Settings, config_key)
            if not file_path or not file_path.strip():
                logging.debug(
                    "Template file %s not configured, skipping",
                    config_key,
                )
                continue  # Optional files

            template_path = Path(file_path)
            file_description = ConfigValidator.TEMPLATE_FILES.get(
                config_key,
                "Template file",
            )
            ConfigValidator._validate_file_exists_and_readable(
                template_path,
                config_key,
                file_description,
            )

    @staticmethod
    def _validate_file_exists_and_readable(
        file_path: Path,
        config_key: str,
        file_description: str,
    ) -> None:
        """Validate that a file exists, is a file, and can be opened for reading.

        Args:
            file_path: Path to the file to validate.
            config_key: Configuration key for error reporting.
            file_description: Human-readable description of the file.

        Raises:
            FileValidationError: If file validation fails.
        """
        if not file_path.exists():
            msg = f"{file_description} does not exist"
            raise FileValidationError(
                msg,
                file_path=str(file_path),
                config_key=config_key,
            )

        if not file_path.is_file():
            msg = f"{file_description} path is not a file"
            raise FileValidationError(
                msg,
                file_path=str(file_path),
                config_key=config_key,
            )
        # Most reliable check: try opening the file.
        try:
            with file_path.open("rb"):
                pass
        except OSError as e:
            msg = f"{file_description} is not readable: {e}"
            raise FileValidationError(
                msg,
                file_path=str(file_path),
                config_key=config_key,
            ) from e

    @staticmethod
    def _validate_output_paths_writable() -> None:
        """Validate that output file parent directories are writable.

        Raises:
            DirectoryValidationError: If output paths are not writable.
        """
        for config_key in ConfigValidator.OUTPUT_FILE_ATTRS:
            file_path = getattr(Settings, config_key)
            if not file_path:
                logging.debug(
                    "Output file %s not configured, skipping",
                    config_key,
                )
                continue
            output_path = Path(file_path)
            file_description = ConfigValidator.OUTPUT_FILES.get(
                config_key,
                "output file",
            )
            ConfigValidator._validate_output_directory_writable(
                output_path,
                config_key,
                file_description,
            )

    @staticmethod
    def _validate_output_directory_writable(
        output_path: Path,
        config_key: str,
        file_description: str,
    ) -> None:
        """Validate that the output directory exists, is a directory, and is writable.

        Args:
            output_path: Path to the output file.
            config_key: Configuration key for error reporting.
            file_description: Human-readable description of the file.

        Raises:
            DirectoryValidationError: If the directory is missing/invalid or
                cannot accept new files from this process.
        """
        output_dir = output_path.parent

        # Check if directory exists (should be created by Settings.initialize())
        if not output_dir.exists():
            msg = (
                f"Output directory for {file_description} does not exist: {output_dir}. "
                "Make sure Settings.initialize() was called."
            )
            raise DirectoryValidationError(
                msg,
                config_key=config_key,
                directory_path=str(output_dir),
            )

        # Check if directory is actually a directory
        if not output_dir.is_dir():
            msg = f"Output path for {file_description} is not a directory: {output_dir}"
            raise DirectoryValidationError(
                msg,
                config_key=config_key,
                directory_path=str(output_dir),
            )

        # Test writability
        if not os.access(output_dir, os.W_OK):
            msg = (
                f"Output directory for {file_description} is not writable: {output_dir}"
            )
            raise DirectoryValidationError(
                msg,
                config_key=config_key,
                directory_path=str(output_dir),
            )

    @staticmethod
    def validate_all(client_type: str | None = None) -> None:
        """Perform comprehensive validation of all configuration.

        Note: This method assumes Settings.initialize() has already been called
        by the caller. It only validates configuration values and file paths.

        Args:
            client_type: Optional client type to validate client-specific config.

        Raises:
            ConfigValidationError: If any validation fails.
        """
        try:
            # Validate file paths (checks accessibility)
            ConfigValidator.validate_file_paths()

            # Validate client-specific configuration if client type provided
            if client_type:
                ConfigValidator.validate_for_client(client_type)

            logging.info(
                "Comprehensive configuration validation completed successfully",
            )
        except (
            ConfigError,
            ConfigValidationError,
            FileValidationError,
            DirectoryValidationError,
        ):
            logging.exception("Configuration validation failed")
            raise

    @staticmethod
    def is_valid(client_type: str | None = None, *, silent: bool = True) -> bool:
        """Check if configuration is valid without raising exceptions.

        Args:
            client_type: Optional client type for client-specific validation.
            silent: If True, suppress logging of validation errors (default: True).

        Returns:
            True if configuration is passes, False otherwise.
        """
        try:
            ConfigValidator.validate_all(client_type)
        except (
            ConfigError,
            ConfigValidationError,
            FileValidationError,
            DirectoryValidationError,
        ) as e:
            if not silent:
                logging.warning("Configuration validation failed: %s", e)
            return False
        except Exception:
            # Unexpected errors should be logged
            logging.exception("Unexpected error during validation")
            return False
        else:
            return True
