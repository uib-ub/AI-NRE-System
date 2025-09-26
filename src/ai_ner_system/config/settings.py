"""Configuration settings for Medieval texts LLM processing application.

This module provides configuration management with environment variables loading,
validation, and error handling with type safety and client-specific validation.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .exceptions import ConfigError

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Configuration settings for the Medieval texts LLM processing application.

    Loads configuration from environment variables with improved validation
    and type safety. Supports client-specific configuration validation.

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
        CACHE_DIR: Directory for caching LLM responses, defaults to ".cache_llm".
    """

    # API Configuration
    ANTHROPIC_API_KEY: str | None = os.getenv('ANTHROPIC_API_KEY')
    OPENWEBUI_TOKEN: str | None = os.getenv('OPENWEBUI_TOKEN')
    OPENWEBUI_ENDPOINT: str | None = os.getenv('OPENWEBUI_ENDPOINT')

    # Model Configuration
    OLLAMA_MODEL: str | None = os.getenv('OLLAMA_MODEL')
    CLAUDE_MODEL: str | None = os.getenv('CLAUDE_MODEL')

    # File I/O Configuration
    INPUT_FILE: str = os.getenv('INPUT_FILE', 'input/Brevid-DN-AI.txt')
    OUTPUT_TEXT_FILE: str = os.getenv('OUTPUT_TEXT_FILE', 'output/annotated_texts.txt')
    OUTPUT_TABLE_FILE: str = os.getenv('OUTPUT_TABLE_FILE', 'output/metadata_table.txt')
    OUTPUT_STATS_FILE: str = os.getenv('OUTPUT_STATS_FILE', 'output/processing_stats.json')

    # Template Configuration
    PROMPT_TEMPLATE_FILE: str = os.getenv('PROMPT_TEMPLATE_FILE', 'prompt/prompt.txt')
    BATCH_TEMPLATE_FILE: str = os.getenv('BATCH_TEMPLATE_FILE', 'prompt/batch_template.txt')

    # Cache Configuration
    CACHE_DIR: Path = Path(os.getenv('CACHE_DIR', '.cache_llm'))

    @classmethod
    def initialize(cls) -> None:
        """Initialize configuration and create necessary directories.

        Should be called once at the application startup.

        Raises:
            ConfigError: If initialization fails.
        """
        try:
            cls._create_cache_directory()
            cls._ensure_output_directories()
            logging.info('Configuration initialized successfully')

        except OSError as e:
            raise ConfigError(f'Failed to initialize configuration: {e}') from e


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
            logging.error('Failed to create cache directory %s: %s', cls.CACHE_DIR, e)
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
            client_type: Type of client ('claude' or 'ollama').

        Returns:
            Dictionary of required configuration keys and their values.

        Raises:
            ConfigError: If client type is unsupported.
        """
        client_type = client_type.lower()

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
        else:
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