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
    FileValidationError
)
from .settings import Settings


class ConfigValidator:
    """Validates configuration settings for the AI NER System."""

    # Class constants for better maintainability and type safety
    SUPPORTED_CLIENT_TYPES: ClassVar[frozenset[str]] = frozenset({'claude', 'ollama'})
    TEMPLATE_FILES: ClassVar[dict[str, str]] = {
        'PROMPT_TEMPLATE_FILE': 'prompt template',
        'BATCH_TEMPLATE_FILE': 'batch template',
    }
    OUTPUT_FILES: ClassVar[dict[str, str]] = {
        'OUTPUT_TEXT_FILE': 'output text file',
        'OUTPUT_TABLE_FILE': 'output table file',
        'OUTPUT_STATS_FILE': 'output stats file',
    }

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
            raise ConfigValidationError('Client type must be provided.')

        client_type = client_type.strip().lower()
        if client_type not in ConfigValidator.SUPPORTED_CLIENT_TYPES:
            supported_types = ', '.join(sorted(ConfigValidator.SUPPORTED_CLIENT_TYPES))
            raise ConfigValidationError(
                f'Unsupported client type: {client_type}. Supported types: {supported_types}.'
            )

        try:
            # Validate client-specific configuration
            client_configs = Settings.get_client_required_configs(client_type)
            missing_client_configs = [
                key for key, value in client_configs.items() if not value
            ]

            # Validate common configuration
            common_configs = Settings.get_common_required_configs()
            missing_common_configs = [
                key for key, value in common_configs.items() if not value
            ]

            # Combine all missing configurations
            missing_configs = missing_client_configs + missing_common_configs

            if missing_configs:
                raise ConfigValidationError(
                    f'Missing required configuration for {client_type} client: '
                    f'{", ".join(missing_configs)}. '
                    'Set them in environment variables or your .env file.',
                    missing_keys=missing_configs
                )

            logging.info('Configuration validation passed for %s client', client_type)

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
            logging.info('File path validation completed successfully')
        except (OSError, ConfigError, FileValidationError, DirectoryValidationError) as e:
            raise ConfigValidationError(f'File path validation failed: {e}') from e


    @staticmethod
    def _validate_input_file() -> None:
        """Validate that the optional input file exists, is a file, and is readable.

        Raises:
            FileValidationError: If the file is missing, not a file, or unreadable.
        """
        if not Settings.INPUT_FILE or not Settings.INPUT_FILE.strip():
            logging.warning('INPUT_FILE not configured, skipping validation')
            return # Optional validation, will be caught by required config check

        # Validate input file exists
        input_path = Path(Settings.INPUT_FILE)
        ConfigValidator._validate_file_exists_and_readable(
            input_path, 'INPUT_FILE', 'Input file'
        )

        try:
            # Size check (0-byte input usually indicates misconfiguration).
            file_size = input_path.stat().st_size
            if file_size == 0:
                raise FileValidationError(
                    f'Input file is empty',
                    file_path=str(input_path),
                    config_key='INPUT_FILE',
                )
        except OSError as e:
            raise FileValidationError(
                f'Cannot access input file: {e}',
                file_path=str(input_path),
                config_key='INPUT_FILE'
            ) from e

    @staticmethod
    def _validate_template_files() -> None:
        """Validate template files exist and are readable.

        Raises:
            FileValidationError: If template files are invalid.
        """
        template_configs = {
            'PROMPT_TEMPLATE_FILE': Settings.PROMPT_TEMPLATE_FILE,
            'BATCH_TEMPLATE_FILE': Settings.BATCH_TEMPLATE_FILE,
        }

        for config_key, file_path in template_configs.items():
            if not file_path or not file_path.strip():
                logging.debug('Template file %s not configured, skipping', config_key)
                continue # Optional files

            template_path = Path(file_path)
            file_description = ConfigValidator.TEMPLATE_FILES.get(config_key, 'Template file')
            ConfigValidator._validate_file_exists_and_readable(
                template_path, config_key, file_description
            )

    @staticmethod
    def _validate_file_exists_and_readable(
        file_path: Path, 
        config_key: str, 
        file_description: str
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
            raise FileValidationError(
                f'{file_description} does not exist',
                file_path=str(file_path),
                config_key=config_key
            )

        if not file_path.is_file():
            raise FileValidationError(
                f'{file_description} path is not a file',
                file_path=str(file_path),
                config_key=config_key
            )
        # Most reliable check: try opening the file.
        try:
            with file_path.open('rb'):
                pass
        except OSError as e:
            raise FileValidationError(
                f'{file_description} is not readable: {e}',
                file_path=str(file_path),
                config_key=config_key,
            ) from e

    @staticmethod
    def _validate_output_paths_writable() -> None:
        """Validate that output file parent directories are writable.

        Raises:
            DirectoryValidationError: If output paths are not writable.
        """
        output_configs = {
            'OUTPUT_TEXT_FILE': Settings.OUTPUT_TEXT_FILE,
            'OUTPUT_TABLE_FILE': Settings.OUTPUT_TABLE_FILE,
            'OUTPUT_STATS_FILE': Settings.OUTPUT_STATS_FILE,
        }

        for config_key, file_path in output_configs.items():
            if not file_path:
                logging.debug('Output file %s not configured, skipping', config_key)
                continue
            output_path = Path(file_path)
            file_description = ConfigValidator.OUTPUT_FILES.get(config_key, 'output file')
            ConfigValidator._validate_output_directory_writable(
                output_path, config_key, file_description
            )

    @staticmethod
    def _validate_output_directory_writable(
        output_path: Path, 
        config_key: str, 
        file_description: str
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
            raise DirectoryValidationError(
                f'Output directory for {file_description} does not exist: {output_dir}. '
                'Make sure Settings.initialize() was called.',
                config_key=config_key,
                directory_path=str(output_dir),
            )

        # Check if directory is actually a directory
        if not output_dir.is_dir():
            raise DirectoryValidationError(
                f'Output path for {file_description} is not a directory: {output_dir}',
                config_key=config_key,
                directory_path=str(output_dir),
            )

        # Test writability
        if not os.access(output_dir, os.W_OK):
            raise DirectoryValidationError(
                f'Output directory for {file_description} is not writable: {output_dir}',
                config_key=config_key,
                directory_path=str(output_dir),
            )

    @staticmethod
    def validate_all(client_type: str | None = None) -> None:
        """Perform comprehensive validation of all configuration.

        Args:
            client_type: Optional client type to validate client-specific config.

        Raises:
            ConfigValidationError: If any validation fails.
        """
        try:
            # Initialize settings first (creates directories)
            Settings.initialize()

            # Validate file paths (checks accessibility)
            ConfigValidator.validate_file_paths()

            # Validate client-specific configuration if client type provided
            if client_type:
                ConfigValidator.validate_for_client(client_type)

            logging.info('Comprehensive configuration validation completed successfully')
        except (
            ConfigError, 
            ConfigValidationError, 
            FileValidationError, 
            DirectoryValidationError
        ) as e:
            logging.error('Configuration validation failed: %s', e)
            raise

    @staticmethod
    def is_valid(client_type: str | None = None, *, silent: bool = True) -> bool:
        """Check if configuration is valid without raising exceptions.

        Args:
            client_type: Optional client type for client-specific validation.

        Returns:
            True if configuration is passes, False otherwise.
        """
        try:
            ConfigValidator.validate_all(client_type)
            return True
        except (
            ConfigError, 
            ConfigValidationError,
            FileValidationError,
            DirectoryValidationError,
        ) as e:
            if not silent:
                logging.warning('Configuration validation failed: %s', e)
            return False
        except Exception as e:
            # Unexpected errors should be logged
            logging.error('Unexpected error during validation: %s', e, exc_info=True)
            return False