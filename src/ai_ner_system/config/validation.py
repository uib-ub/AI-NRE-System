"""Configuration validation for AI NER System."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .exceptions import ConfigError, ConfigValidationError, DirectoryValidationError, FileValidationError
from .settings import Settings


class ConfigValidator:
    """Validates configuration settings for the AI NER System."""

    # Class constants for better maintainability and type safety
    _SUPPORTED_CLIENT_TYPES = {'claude', 'ollama'}
    _TEMPLATE_FILES = {
        'PROMPT_TEMPLATE_FILE': 'prompt template',
        'BATCH_TEMPLATE_FILE': 'batch template',
    }
    _OUTPUT_FILES = {
        'OUTPUT_TEXT_FILE': 'output text file',
        'OUTPUT_TABLE_FILE': 'output table file',
        'OUTPUT_STATS_FILE': 'output stats file',
    }

    @staticmethod
    def validate_for_client(client_type: str) -> None:
        """Validate configuration for specified client type.

        Args:
            client_type: Type of client ('claude' or 'ollama').

        Raises:
            ConfigValidationError: If required configuration is missing or invalid.
        """
        if client_type.lower() not in ConfigValidator._SUPPORTED_CLIENT_TYPES:
            raise ConfigValidationError(
                f'Unsupported client type: {client_type}. '
                f'Supported types: {", ".join(ConfigValidator._SUPPORTED_CLIENT_TYPES)}'
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
                    f'{", ".join(missing_configs)}. Please set these in your '
                    'environment variables or .env file.',
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
            raise ConfigValidationError(f'File path validation failed: {e}')


    @staticmethod
    def _validate_input_file() -> None:
        """Validate input file exists and is readable.

        Raises:
            ConfigError: If input file is invalid.
        """
        if not Settings.INPUT_FILE:
            return # Optional validation, will be caught by required config check

        # Validate input file exists
        input_path = Path(Settings.INPUT_FILE)

        try:
            ConfigValidator._validate_file_exists_and_readable(
                input_path, 'INPUT_FILE', 'Input file'
            )

            if input_path.stat().st_size == 0:
                raise ConfigError(
                    f'Input file is empty: {Settings.INPUT_FILE}',
                    config_key='INPUT_FILE',
                )
        except OSError as e:
            raise FileValidationError(
                f'Cannot access input file {Settings.INPUT_FILE}: {e}',
                file_path=str(input_path),
                config_key='INPUT_FILE'
            ) from e


    @staticmethod
    def _validate_template_files() -> None:
        """Validate template files exist and are readable.

        Raises:
            ConfigError: If template files are invalid.
        """
        template_configs = {
            'PROMPT_TEMPLATE_FILE': Settings.PROMPT_TEMPLATE_FILE,
            'BATCH_TEMPLATE_FILE': Settings.BATCH_TEMPLATE_FILE,
        }

        for config_key, file_path in template_configs.items():
            if not file_path:
                continue # Optional files

            template_path = Path(file_path)
            file_description = ConfigValidator._TEMPLATE_FILES.get(config_key, 'Template file')

            try:
                ConfigValidator._validate_file_exists_and_readable(
                    template_path, config_key, file_description
                )
            except OSError as e:
                raise FileValidationError(
                    f'Cannot access {file_description} {file_path}: {e}',
                    file_path=str(template_path),
                    config_key=config_key
                ) from e


    @staticmethod
    def _validate_file_exists_and_readable(
            file_path: Path, config_key: str, file_description: str
    ) -> None:
        """Validate that a file exists and is readable.

        Args:
            file_path: Path to the file to validate.
            config_key: Configuration key for error reporting.
            file_description: Human-readable description of the file.

        Raises:
            ConfigError: If file validation fails.
        """
        if not file_path.exists():
            raise FileValidationError(
                f'{file_description} does not exist: {file_path}',
                file_path=str(file_path),
                config_key=config_key
            )

        if not file_path.is_file():
            raise FileValidationError(
                f'{file_description} path is not a file: {file_path}',
                file_path=str(file_path),
                config_key=config_key
            )


    @staticmethod
    def _validate_output_paths_writable() -> None:
        """Validate that output file parent directories are writable.

        Raises:
            ConfigError: If output paths are not writable.
        """
        output_configs = {
            'OUTPUT_TEXT_FILE': Settings.OUTPUT_TEXT_FILE,
            'OUTPUT_TABLE_FILE': Settings.OUTPUT_TABLE_FILE,
            'OUTPUT_STATS_FILE': Settings.OUTPUT_STATS_FILE,
        }

        for config_key, file_path in output_configs.items():
            if not file_path:
                continue

            output_path = Path(file_path)
            file_description = ConfigValidator._OUTPUT_FILES.get(config_key, 'output file')

            try:
                ConfigValidator._validate_output_directory_writable(
                    output_path, config_key, file_description
                )
            except OSError as e:
                raise DirectoryValidationError(
                    f'Cannot access directory for {file_description} {file_path}: {e}',
                    directory_path=str(output_path.parent),
                    config_key=config_key
                ) from e



    @staticmethod
    def _validate_output_directory_writable(
        output_path: Path, config_key: str, file_description: str
    ) -> None:
        """Validate that an output directory exists and is writable.

       Args:
           output_path: Path to the output file.
           config_key: Configuration key for error reporting.
           file_description: Human-readable description of the file.

       Raises:
           ConfigError: If directory validation fails.
       """
        output_dir = output_path.parent

        # Check if directory exists (should be created by Settings.initialize())
        if not output_dir.exists():
            raise DirectoryValidationError(
                f'Output directory for {file_description} does not exist: {output_dir}. '
                'Make sure Settings.initialize() was called.',
                directory_path=str(output_dir),
                config_key=config_key
            )

        # Check if directory is actually a directory
        if not output_dir.is_dir():
            raise DirectoryValidationError(
                f'Output path for {file_description} is not a directory: {output_dir}',
                directory_path=str(output_dir),
                config_key=config_key
            )

        # Test writability
        if not os.access(output_dir, os.W_OK):
            raise DirectoryValidationError(
                f'Output directory for {file_description} is not writable: {output_dir}',
                directory_path=str(output_dir),
                config_key=config_key
            )


    @staticmethod
    def validate_all(client_type: str | None = None) -> None:
        """Perform comprehensive validation of all configuration.

        Args:
            client_type: Optional client type for client-specific validation

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

            logging.info("Comprehensive configuration validation completed successfully")

        except (ConfigError, ConfigValidationError, FileValidationError, DirectoryValidationError) as e:
            logging.error("Configuration validation failed: %s", e)
            raise

    @staticmethod
    def is_valid(client_type: str | None = None) -> bool:
        """Check if configuration is valid without raising exceptions.

        Args:
            client_type: Optional client type for client-specific validation.

        Returns:
            True if configuration is valid, False otherwise.
        """
        try:
            ConfigValidator.validate_all(client_type)
            return True
        except (ConfigError, ConfigValidationError):
            return False