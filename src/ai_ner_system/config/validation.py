"""Configuration validation for AI NER System."""

import logging
import os
from pathlib import Path
from typing import Optional

from .exceptions import ConfigError, ConfigValidationError
from .settings import Settings


class ConfigValidator:
    """Validates configuration settings for the AI NER System."""

    @staticmethod
    def validate_for_client(client_type: str) -> None:
        """Validate configuration for specified client type.

        Args:
            client_type: Type of client ('claude' or 'ollama').

        Raises:
            ConfigValidationError: If required configuration is missing or invalid.
        """
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

        except (OSError, ConfigError) as e:
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
        if not input_path.exists():
            raise ConfigError(
                f'Input file does not exist: {Settings.INPUT_FILE}',
                config_key='INPUT_FILE',
            )
        if not input_path.is_file():
            raise ConfigError(
                f'Input path is not a file: {Settings.INPUT_FILE}',
                config_key='INPUT_FILE',
            )
        if not input_path.stat().st_size > 0:
            raise ConfigError(
                f'Input file is empty: {Settings.INPUT_FILE}',
                config_key='INPUT_FILE',
            )

    @staticmethod
    def _validate_template_files() -> None:
        """Validate template files exist and are readable.

        Raises:
            ConfigError: If template files are invalid.
        """
        template_files = [
            ('PROMPT_TEMPLATE_FILE', Settings.PROMPT_TEMPLATE_FILE),
            ('BATCH_TEMPLATE_FILE', Settings.BATCH_TEMPLATE_FILE),
        ]

        for config_key, file_path in template_files:
            if not file_path:
                continue # Optional files

            template_path = Path(file_path)
            if not template_path.exists():
                raise ConfigError(
                    f'Template file does not exist: {file_path}',
                    config_key=config_key,
                )
            if not template_path.is_file():
                raise ConfigError(
                    f'Template path is not a file: {file_path}',
                    config_key=config_key,
                )

    @staticmethod
    def _validate_output_paths_writable() -> None:
        """Validate that output file parent directories are writable.

        Raises:
            ConfigError: If output paths are not writable.
        """
        output_files = [
            ("OUTPUT_TEXT_FILE", Settings.OUTPUT_TEXT_FILE),
            ("OUTPUT_TABLE_FILE", Settings.OUTPUT_TABLE_FILE),
            ("OUTPUT_STATS_FILE", Settings.OUTPUT_STATS_FILE),
        ]

        for config_key, file_path in output_files:
            if not file_path:
                continue

            output_path = Path(file_path)
            output_dir = output_path.parent

            # Check if directory exists (should be created by Settings.initialize())
            if not output_dir.exists():
                raise ConfigError(
                    f'Output directory does not exist: {output_dir}. '
                    'Make sure Settings.initialize() was called.',
                    config_key=config_key,
                )

            # Check if directory is writable
            if not output_dir.is_dir():
                raise ConfigError(
                    f'Output path is not a directory: {output_dir}',
                    config_key=config_key,
                )

            # Test writability by checking permissions
            if not os.access(output_dir, os.W_OK):
                raise ConfigError(
                    f'Output directory is not writable: {output_dir}',
                    config_key=config_key,
                )

    @staticmethod
    def validate_all(client_type: Optional[str] = None) -> None:
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

        except (ConfigError, ConfigValidationError) as e:
            logging.error("Configuration validation failed: %s", e)
            raise

    @staticmethod
    def is_valid(client_type: Optional[str] = None) -> bool:
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