"""Configuration settings for Medieval texts LLM processing application.

This module provides configuration management with environment variables loading,
validation, and error handling with type safety and client-specific validation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

from .exceptions import ConfigError


class Settings:
    """Configuration settings for the Medieval texts LLM processing application.

    Loads configuration from environment variables with improved validation
    and type safety. Supports client-specific configuration validation.

    All configuration values are class-level attributes that can be accessed
    without instantiating the class.

    Attributes:
        ANTHROPIC_API_KEY: API key for Anthropic Claude service.
        OPENWEBUI_TOKEN: Authentication token for OpenWebUI.
        OPENWEBUI_ENDPOINT: Endpoint URL for OpenWebUI service.
        OLLAMA_MODEL: Model name for Ollama service.
        CLAUDE_MODEL: Model name for Claude service.
        INPUT_FILE: Path to the input CSV file containing records.
        OUTPUT_TEXT_FILE: Path to the output text file for annotated records.
        OUTPUT_TABLE_FILE: Path to the output table file for metadata.
        OUTPUT_STATS_FILE: Path to the output statistics file.
        PROMPT_TEMPLATE_FILE: Path to the prompt template file.
        BATCH_TEMPLATE_FILE: Path to the batch processing template file.
        CACHE_DIR: Directory for caching LLM responses.
    """

    # Default values as class constants
    DEFAULT_INPUT_FILE: ClassVar[str] = 'input/Brevid-DN-AI.txt'
    DEFAULT_OUTPUT_TEXT_FILE: ClassVar[str] = 'output/annotated_texts.txt'
    DEFAULT_OUTPUT_TABLE_FILE: ClassVar[str] = 'output/metadata_table.txt'
    DEFAULT_OUTPUT_STATS_FILE: ClassVar[str] = 'output/processing_stats.json'
    DEFAULT_PROMPT_TEMPLATE_FILE: ClassVar[str] = 'prompt/prompt.txt'
    DEFAULT_BATCH_TEMPLATE_FILE: ClassVar[str] = 'prompt/batch_template.txt'
    DEFAULT_CACHE_DIR: ClassVar[str] = '.cache_llm'

    # Supported client types
    SUPPORTED_CLIENTS: ClassVar[frozenset[str]] = frozenset(
        {'claude', 'ollama'}
    )

    # Flag to track initialization
    _initialized: ClassVar[bool] = False

    # API Configuration - These are loaded dynamically, so they are NOT ClassVar
    ANTHROPIC_API_KEY: str | None = None
    OPENWEBUI_TOKEN: str | None = None
    OPENWEBUI_ENDPOINT: str | None = None

    # Model Configuration - These are loaded dynamically, so they are NOT ClassVar
    OLLAMA_MODEL: str | None = None
    CLAUDE_MODEL: str | None = None

    # File I/O Configuration - These are loaded dynamically, so they are NOT ClassVar
    INPUT_FILE: str = DEFAULT_INPUT_FILE
    OUTPUT_TEXT_FILE: str = DEFAULT_OUTPUT_TEXT_FILE
    OUTPUT_TABLE_FILE: str = DEFAULT_OUTPUT_TABLE_FILE
    OUTPUT_STATS_FILE: str = DEFAULT_OUTPUT_STATS_FILE

    # Template Configuration - These are loaded dynamically, so they are NOT ClassVar
    PROMPT_TEMPLATE_FILE: str = DEFAULT_PROMPT_TEMPLATE_FILE
    BATCH_TEMPLATE_FILE: str = DEFAULT_BATCH_TEMPLATE_FILE

    # Cache Configuration - This is loaded dynamically, so it is NOT ClassVar
    CACHE_DIR: Path = Path(DEFAULT_CACHE_DIR)

    @classmethod
    def initialize(cls, reload_env: bool = False) -> None:
        """Initialize configuration and create necessary directories.

        Should be called once at the application startup. Safe to call multiple
        times (subsequent calls are no-ops unless reload_env=True).

        Args:
            reload_env: If True, reload environment variables from .env file.

        Raises:
            ConfigError: If initialization fails.
        """

        if cls._initialized and not reload_env:
            logging.debug('Settings already initialized, skipping')
            return

        try:
            # Load environment variables from .env file
            load_dotenv(override=reload_env)

            # Load configuration from environment
            cls._load_from_environment()

            # Create necessary directories
            cls._create_cache_directory()
            cls._ensure_output_directories()

            cls._initialized = True
            logging.info('Configuration initialized successfully')

        except OSError as e:
            raise ConfigError(
                f'Failed to initialize configuration: {e}'
            ) from e

    @classmethod
    def _load_from_environment(cls) -> None:
        """Load configuration values from environment variables."""
        # API Configuration
        cls.ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
        cls.OPENWEBUI_TOKEN = os.getenv('OPENWEBUI_TOKEN')
        cls.OPENWEBUI_ENDPOINT = os.getenv('OPENWEBUI_ENDPOINT')

        # Model Configuration
        cls.OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')
        cls.CLAUDE_MODEL = os.getenv('CLAUDE_MODEL')

        # File I/O Configuration
        cls.INPUT_FILE = os.getenv('INPUT_FILE', cls.DEFAULT_INPUT_FILE)
        cls.OUTPUT_TEXT_FILE = os.getenv(
            'OUTPUT_TEXT_FILE', cls.DEFAULT_OUTPUT_TEXT_FILE
        )
        cls.OUTPUT_TABLE_FILE = os.getenv(
            'OUTPUT_TABLE_FILE', cls.DEFAULT_OUTPUT_TABLE_FILE
        )
        cls.OUTPUT_STATS_FILE = os.getenv(
            'OUTPUT_STATS_FILE', cls.DEFAULT_OUTPUT_STATS_FILE
        )

        # Template Configuration
        cls.PROMPT_TEMPLATE_FILE = os.getenv(
            'PROMPT_TEMPLATE_FILE', cls.DEFAULT_PROMPT_TEMPLATE_FILE
        )
        cls.BATCH_TEMPLATE_FILE = os.getenv(
            'BATCH_TEMPLATE_FILE', cls.DEFAULT_BATCH_TEMPLATE_FILE
        )

        # Cache Configuration
        cache_dir_str = os.getenv('CACHE_DIR', cls.DEFAULT_CACHE_DIR)
        cls.CACHE_DIR = Path(cache_dir_str).expanduser()

    @classmethod
    def _create_cache_directory(cls) -> None:
        """Create cache directory if it doesn't exist.

        Raises:
           OSError: If directory creation fails.
        """
        try:
            cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logging.info('Cache directory created: %s', cls.CACHE_DIR)
        except OSError as e:
            logging.error(
                'Failed to create cache directory %s: %s', cls.CACHE_DIR, e
            )
            raise

    @classmethod
    def _ensure_output_directories(cls) -> None:
        """Ensure all output directories exist.

        Raises:
            OSError: If directory creation fails.
        """
        output_files = [
            cls.OUTPUT_TEXT_FILE,
            cls.OUTPUT_TABLE_FILE,
            cls.OUTPUT_STATS_FILE,
        ]

        # Validate and create output directories
        for file_path in output_files:
            if file_path:
                cls._ensure_directory_exists(file_path)

    @classmethod
    def _ensure_directory_exists(cls, file_path: str) -> None:
        """Ensure directory for given file path exists.

        Args:
            file_path: Path to the file.

        Raises:
            OSError: If directory creation fails.
        """
        output_path = Path(file_path)
        output_dir = output_path.parent

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info('Created output directory: %s', output_dir)

    @classmethod
    def get_client_required_configs(cls, client_type: str) -> dict[str, str | None]:
        """Get required configurations for specified client type.

        Args:
            client_type: Type of client ('claude' or 'ollama'), case-insensitive.

        Returns:
            Dictionary of required configuration keys and their values.

        Raises:
            ConfigError: If client type is unsupported.
        """
        client_type = client_type.lower()

        if client_type not in cls.SUPPORTED_CLIENTS:
            supported = ', '.join(sorted(cls.SUPPORTED_CLIENTS))
            raise ConfigError(
                f'Unsupported client type: {client_type}. Supported types: {supported}'
            )

        if client_type == 'claude':
            return {
                'ANTHROPIC_API_KEY': cls.ANTHROPIC_API_KEY,
                'CLAUDE_MODEL': cls.CLAUDE_MODEL,
            }
        elif client_type == 'ollama':
            return {
                'OLLAMA_MODEL': cls.OLLAMA_MODEL,
                'OPENWEBUI_TOKEN': cls.OPENWEBUI_TOKEN,
                'OPENWEBUI_ENDPOINT': cls.OPENWEBUI_ENDPOINT,
            }

        # This should never be reached due to check above, but for type safety
        raise ConfigError(f'Unsupported client type: {client_type}')

    @classmethod
    def get_common_required_configs(cls) -> dict[str, str | None]:
        """Get common required configurations for all clients.

        Returns:
            Dictionary of common required configuration keys and their values.
        """
        return {
            'INPUT_FILE': cls.INPUT_FILE,
            'OUTPUT_TEXT_FILE': cls.OUTPUT_TEXT_FILE,
            'OUTPUT_TABLE_FILE': cls.OUTPUT_TABLE_FILE,
            'OUTPUT_STATS_FILE': cls.OUTPUT_STATS_FILE,
            'PROMPT_TEMPLATE_FILE': cls.PROMPT_TEMPLATE_FILE,
        }

    @classmethod
    def reset(cls) -> None:
        """Reset settings to defaults.

        Useful for testing. Clears all configuration and resets initialization flag.

        Example:
            >>> Settings.initialize()
            >>> Settings.reset()
            >>> assert not Settings._initialized
        """
        cls._initialized = False
        cls.ANTHROPIC_API_KEY = None
        cls.OPENWEBUI_TOKEN = None
        cls.OPENWEBUI_ENDPOINT = None
        cls.OLLAMA_MODEL = None
        cls.CLAUDE_MODEL = None
        cls.INPUT_FILE = cls.DEFAULT_INPUT_FILE
        cls.OUTPUT_TEXT_FILE = cls.DEFAULT_OUTPUT_TEXT_FILE
        cls.OUTPUT_TABLE_FILE = cls.DEFAULT_OUTPUT_TABLE_FILE
        cls.OUTPUT_STATS_FILE = cls.DEFAULT_OUTPUT_STATS_FILE
        cls.PROMPT_TEMPLATE_FILE = cls.DEFAULT_PROMPT_TEMPLATE_FILE
        cls.BATCH_TEMPLATE_FILE = cls.DEFAULT_BATCH_TEMPLATE_FILE
        cls.CACHE_DIR = Path(cls.DEFAULT_CACHE_DIR)
