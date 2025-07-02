"""Configuration module for Medieval texts with LLM processing application.

This module provides configuration management with environment variables loading,
validation, and error handling for the Medieval texts with LLM processing application.

"""

import os
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration-related errors."""

class Config:
    """Configuration class for LLM processing application.

    Loads configuration from environment variables from a .env file.
    Provides validation and type safety for configuration values.

    Attributes:
        ANTHROPIC_API_KEY: API key for Anthropic Claude service.
        OPENWEBUI_TOKEN: Authentication token for OpenWebUI.
        OPENWEBUI_ENDPOINT: Endpoint URL for OpenWebUI service.
        OLLAMA_MODEL: Model name for Ollama service.
        CLAUDE_MODEL: Model name for Claude service.
        INPUT_FILE: Path to the input CSV file containing records.
        OUTPUT_TEXT_FILE: Path to the output text file for annotated records.
        OUTPUT_TABLE_FILE: Path to the output table file for metadata.
        PROMPT_TEMPLATE_FILE: Path to the prompt template file.
        BATCH_TEMPLATE_FILE: Path to the batch processing template file.
        CACHE_DIR: Directory for caching LLM responses, defaults to ".cache_llm".
    """

    # These must be set in your environment or .env file
    # API Configuration
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    OPENWEBUI_TOKEN: Optional[str] = os.getenv("OPENWEBUI_TOKEN")
    OPENWEBUI_ENDPOINT: Optional[str] = os.getenv("OPENWEBUI_ENDPOINT")

    # Model Configuration
    OLLAMA_MODEL: Optional[str] = os.getenv("OLLAMA_MODEL")
    CLAUDE_MODEL: Optional[str] = os.getenv("CLAUDE_MODEL")

    # File Input/Output Configuration
    INPUT_FILE: Optional[str] = os.getenv("INPUT_FILE")
    OUTPUT_TEXT_FILE: Optional[str] = os.getenv("OUTPUT_TEXT_FILE")
    OUTPUT_TABLE_FILE: Optional[str] = os.getenv("OUTPUT_TABLE_FILE")
    PROMPT_TEMPLATE_FILE: Optional[str] = os.getenv("PROMPT_TEMPLATE_FILE")
    BATCH_TEMPLATE_FILE: Optional[str] = os.getenv("BATCH_TEMPLATE_FILE")

    # CACHE_DIR has a default value
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", ".cache_llm"))

    @classmethod
    def initialize(cls) -> None:
        """ Initialize configuration and create necessary directories.

        Should be called once at the application startup.

        Raises:
            ConfigError: If cache directory creation fails.
        """
        try:
            cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logging.info('Cache directory initialized: %s', cls.CACHE_DIR)
        except OSError as e:
            raise ConfigError(f'Error initializing cache directory {cls.CACHE_DIR}: {e}') from e

    @classmethod
    def validate_required_config(cls) -> None:
        """Validate that all required configuration values are present.

        Raises:
            ConfigError: If any required configuration is missing.
        """
        # Define required configurations with their names
        required_configs = [
            ('ANTHROPIC_API_KEY', cls.ANTHROPIC_API_KEY),
            ('OPENWEBUI_ENDPOINT', cls.OPENWEBUI_ENDPOINT),
            ('OPENWEBUI_TOKEN', cls.OPENWEBUI_TOKEN),
            ('CLAUDE_MODEL', cls.CLAUDE_MODEL),
            ('OLLAMA_MODEL', cls.OLLAMA_MODEL),
            ('INPUT_FILE', cls.INPUT_FILE),
            ('OUTPUT_TEXT_FILE', cls.OUTPUT_TEXT_FILE),
            ('OUTPUT_TABLE_FILE', cls.OUTPUT_TABLE_FILE),
        ]

        missing_config = [name for name, value in required_configs if not value]

        if missing_config:
            raise ConfigError(
                f'Missing required configuration: {", ".join(missing_config)}. '
                'Please set these in your environment variables or .env file.'
            )

    @classmethod
    def validate_file_paths(cls) -> None:
        """Validate that all file paths are accessible and directories exist.

        Raises:
            ConfigError: If any file path is invalid or inaccessible.
        """
        # Validate input file exists
        if cls.INPUT_FILE:
            input_path = Path(cls.INPUT_FILE)
            if not input_path.exists():
                raise ConfigError(f'Input file does not exist: {cls.INPUT_FILE}')
            if not input_path.is_file():
                raise ConfigError(f'Input path is not a file: {cls.INPUT_FILE}')

        # Validate and create output directories
        output_files = [cls.OUTPUT_TEXT_FILE, cls.OUTPUT_TABLE_FILE]
        for file_config in output_files:
            if file_config:
                cls._ensure_output_directory(file_config)

        # Validate prompt template file if specified
        if cls.PROMPT_TEMPLATE_FILE:
            prompt_path = Path(cls.PROMPT_TEMPLATE_FILE)
            if not prompt_path.exists():
                raise ConfigError(
                    f'Prompt template file does not exist: {cls.PROMPT_TEMPLATE_FILE}'
                )

        # Validate batch template file if specified
        if cls.BATCH_TEMPLATE_FILE:
            batch_path = Path(cls.BATCH_TEMPLATE_FILE)
            if not batch_path.exists():
                raise ConfigError(
                    f'Batch template file does not exist: {cls.BATCH_TEMPLATE_FILE}'
                )

    @classmethod
    def _ensure_output_directory(cls, file_path: str) -> None:
        """Ensure the directory for the given file path exists.

        Args:
            file_path: Path to the output file.

        Raises:
            ConfigError: If directory creation failed.
        """
        output_path = Path(file_path)
        output_dir = output_path.parent

        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logging.info('Created output directory: %s', output_dir)
            except OSError as e:
                raise ConfigError(
                    f'Error creating output directory {output_dir}: {e}'
                ) from e

    @classmethod
    def is_valid(cls) -> bool:
        """Check if configuration is valid without raising exceptions.

        Returns:
            True if configuration is valid, False otherwise.
        """
        try:
            cls.validate_required_config()
            cls.validate_file_paths()
            logging.info('Config validated successfully.')
            return True
        except ConfigError:
            return False

# Initialize configuration on module import
try:
    Config.initialize()
except ConfigError as e:
    logging.error('Configuration initialization failed: %s', e)
    # NOTE: Don't raise here to allow for graceful handling by the application