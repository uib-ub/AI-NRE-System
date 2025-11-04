"""Configuration settings for Medieval texts LLM processing application.

This module provides configuration management with environment variables loading,
validation, and error handling with type safety and client-specific validation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import ClassVar, TypeAlias

from dotenv import load_dotenv

from .exceptions import ConfigError

# Type alias for registry entries
ClientConfigEntry: TypeAlias = tuple[str, str]  # (settings_attr, init_param)
ClientConfigRegistry: TypeAlias = dict[str, list[ClientConfigEntry]]


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
    DEFAULT_INPUT_FILE: ClassVar[str] = "input/Brevid-DN-AI.txt"
    DEFAULT_OUTPUT_TEXT_FILE: ClassVar[str] = "output/annotated_texts.txt"
    DEFAULT_OUTPUT_TABLE_FILE: ClassVar[str] = "output/metadata_table.txt"
    DEFAULT_OUTPUT_STATS_FILE: ClassVar[str] = "output/processing_stats.json"
    DEFAULT_PROMPT_TEMPLATE_FILE: ClassVar[str] = "prompt/prompt.txt"
    DEFAULT_BATCH_TEMPLATE_FILE: ClassVar[str] = "prompt/batch_template.txt"
    DEFAULT_CACHE_DIR: ClassVar[str] = ".cache_llm"

    # Client configuration registry: maps client type to required config keys.
    # Each entry maps a client type to a list of (Settings attribute, init parameter) pairs.
    _CLIENT_CONFIG_REGISTRY: ClassVar[ClientConfigRegistry] = {
        "claude": [
            ("ANTHROPIC_API_KEY", "api_key"),
            ("CLAUDE_MODEL", "model"),
        ],
        "ollama": [
            ("OPENWEBUI_ENDPOINT", "endpoint"),
            ("OPENWEBUI_TOKEN", "token"),
            ("OLLAMA_MODEL", "model"),
        ],
    }

    # Common configuration attributes required for all clients
    # ellipsis literal to indicate any number of strings
    _COMMON_CONFIG_ATTRS: ClassVar[tuple[str, ...]] = (
        "INPUT_FILE",
        "OUTPUT_TEXT_FILE",
        "OUTPUT_TABLE_FILE",
        "OUTPUT_STATS_FILE",
        "PROMPT_TEMPLATE_FILE",
    )

    _OUTPUT_FILE_ATTRS: ClassVar[tuple[str, ...]] = (
        "OUTPUT_TEXT_FILE",
        "OUTPUT_TABLE_FILE",
        "OUTPUT_STATS_FILE",
    )

    # Supported client types - derived from registry keys
    SUPPORTED_CLIENTS: ClassVar[frozenset[str]] = frozenset(
        _CLIENT_CONFIG_REGISTRY.keys()
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
    def initialize(
        cls,
        *,
        reload_env: bool = False,
        validate: bool = False,
        create_dirs: bool = True,
    ) -> None:
        """Initialize configuration and optionally validate/create directories.

        Should be called once at the application startup. Safe to call multiple
        times (subsequent calls are no-ops unless reload_env=True).

        Args:
            reload_env: If True, reload environment variables from .env file.
            validate: If True, validate common configuration after loading.
            create_dirs: If True, create necessary directories (cache and output).

        Raises:
            ConfigError: If initialization or validation fails.
        """
        if cls._initialized and not reload_env:
            logging.debug("Settings already initialized, skipping")
            return

        # If reloading, reset initialization flag first
        if reload_env:
            cls._initialized = False
            logging.info("Reloading configuration from environment")

        try:
            # Load environment variables from .env file
            load_dotenv(override=reload_env)

            # Load configuration from environment
            cls._load_from_environment()

            # Create necessary directories (optional)
            if create_dirs:
                cls._create_cache_directory()
                cls._ensure_output_directories()

            # Validate configuration (optional)
            if validate:
                cls.validate_common_config()

            cls._initialized = True
            logging.info("Configuration initialized successfully")

        except OSError as e:
            msg = f"Failed to initialize configuration: {e}"
            raise ConfigError(msg) from e

    @classmethod
    def _load_from_environment(cls) -> None:
        """Load configuration values from environment variables."""
        # API Configuration
        cls.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        cls.OPENWEBUI_TOKEN = os.getenv("OPENWEBUI_TOKEN")
        cls.OPENWEBUI_ENDPOINT = os.getenv("OPENWEBUI_ENDPOINT")

        # Model Configuration
        cls.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
        cls.CLAUDE_MODEL = os.getenv("CLAUDE_MODEL")

        # File I/O Configuration
        cls.INPUT_FILE = os.getenv("INPUT_FILE", cls.DEFAULT_INPUT_FILE)
        cls.OUTPUT_TEXT_FILE = os.getenv(
            "OUTPUT_TEXT_FILE",
            cls.DEFAULT_OUTPUT_TEXT_FILE,
        )
        cls.OUTPUT_TABLE_FILE = os.getenv(
            "OUTPUT_TABLE_FILE",
            cls.DEFAULT_OUTPUT_TABLE_FILE,
        )
        cls.OUTPUT_STATS_FILE = os.getenv(
            "OUTPUT_STATS_FILE",
            cls.DEFAULT_OUTPUT_STATS_FILE,
        )

        # Template Configuration
        cls.PROMPT_TEMPLATE_FILE = os.getenv(
            "PROMPT_TEMPLATE_FILE",
            cls.DEFAULT_PROMPT_TEMPLATE_FILE,
        )
        cls.BATCH_TEMPLATE_FILE = os.getenv(
            "BATCH_TEMPLATE_FILE",
            cls.DEFAULT_BATCH_TEMPLATE_FILE,
        )

        # Cache Configuration
        cache_dir_str = os.getenv("CACHE_DIR", cls.DEFAULT_CACHE_DIR)
        cls.CACHE_DIR = Path(cache_dir_str).expanduser()

    @classmethod
    def _create_cache_directory(cls) -> None:
        """Create cache directory if it doesn't exist.

        Raises:
           OSError: If directory creation fails.
        """
        try:
            cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logging.info("Cache directory created: %s", cls.CACHE_DIR)
        except OSError:
            logging.exception(
                "Failed to create cache directory %s",
                cls.CACHE_DIR,
            )
            raise

    @classmethod
    def _ensure_output_directories(cls) -> None:
        """Ensure all output directories exist.

        Raises:
            OSError: If directory creation fails.
        """
        # Validate and create output directories
        for attr_name in cls._OUTPUT_FILE_ATTRS:
            file_path = getattr(cls, attr_name)
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
            logging.info("Created output directory: %s", output_dir)

    @classmethod
    def apply_cli_overrides(
        cls,
        *,
        input_file: str | None = None,
        output_text_file: str | None = None,
        output_table_file: str | None = None,
        output_stats_file: str | None = None,
        prompt_template_file: str | None = None,
        batch_template_file: str | None = None,
    ) -> None:
        """Apply command-line overrides to current configuration.

        Updates Settings attributes with provided overrides and ensures
        necessary directories exist for output files.

        Args:
            input_file: Override for input file path.
            output_text_file: Override for output text file path.
            output_table_file: Override for output table file path.
            output_stats_file: Override for output stats file path.
            prompt_template_file: Override for prompt template file path.
            batch_template_file: Override for batch template file path.
        """
        if input_file:
            cls.INPUT_FILE = input_file

        # Apply output file overrides and ensure directories exist
        output_overrides = {
            "OUTPUT_TEXT_FILE": output_text_file,
            "OUTPUT_TABLE_FILE": output_table_file,
            "OUTPUT_STATS_FILE": output_stats_file,
        }

        for attr_name, override_value in output_overrides.items():
            if override_value:
                setattr(cls, attr_name, override_value)
                cls._ensure_directory_exists(override_value)

        if prompt_template_file:
            cls.PROMPT_TEMPLATE_FILE = prompt_template_file

        if batch_template_file:
            cls.BATCH_TEMPLATE_FILE = batch_template_file

    @classmethod
    def validate_client_config(cls, client_type: str) -> None:
        """Validate that all required configuration for a client type is present.

        This method checks that all required configuration values are non-empty.
        It's intended for validation purposes (e.g., in ConfigValidator) where
        the actual parameter values aren't needed, only confirmation they exist.

        Args:
            client_type: Type of client ('claude' or 'ollama'), case-insensitive.

        Raises:
            ConfigError: If client type is unsupported or required parameters are missing/empty.
        """
        # Delegate to get_client_init_params which does the validation
        # We don't need the return value, just the validation side effect
        _ = cls.get_client_init_params(client_type)

    @classmethod
    def get_client_init_params(cls, client_type: str) -> dict[str, str]:
        """Get client initialization parameters for specified client type.

        Returns a dictionary mapping parameter names to their non-empty string values,
        ready to be unpacked into a client constructor. Only validated non-empty
        values are included.

        Args:
            client_type: Type of client ('claude' or 'ollama'), case-insensitive.

        Returns:
            Dictionary mapping init parameter names to their non-empty string values.
            Example: {"api_key": "...", "model": "..."}

        Raises:
            ConfigError: If client type is unsupported or required parameters are missing/empty.
        """
        client_type = client_type.lower()

        if client_type not in cls.SUPPORTED_CLIENTS:
            supported = ", ".join(sorted(cls.SUPPORTED_CLIENTS))
            msg = (
                f"Unsupported client type: {client_type}. Supported types: {supported}"
            )
            raise ConfigError(msg)

        # Build init params dict from registry, validating as we go
        config_keys = cls._CLIENT_CONFIG_REGISTRY.get(client_type, [])
        init_params: dict[str, str] = {}
        missing_params: list[str] = []

        for config_attr, init_param in config_keys:
            value = getattr(cls, config_attr)
            # Validate value is present and non-empty
            if value is None or not value.strip():
                missing_params.append(init_param)
            else:
                init_params[init_param] = value.strip()

        if missing_params:
            params_str = ", ".join(missing_params)
            msg = (
                f"Missing or empty required configuration for {client_type}: {params_str}. "
                "Set them in environment variables or your .env file."
            )
            raise ConfigError(msg, config_key=f"{client_type}_client_params")

        return init_params

    @classmethod
    def validate_common_config(cls) -> None:
        """Validate that all common required configuration is present.

        Common configuration includes file paths that are required for all clients
        (input file, output files, template files, etc.).

        Raises:
            ConfigError: If any required common configuration is missing or empty.
        """
        missing_configs = [
            attr for attr in cls._COMMON_CONFIG_ATTRS if not getattr(cls, attr)
        ]

        if missing_configs:
            params_str = ", ".join(missing_configs)
            msg = (
                f"Missing or empty required common configuration: {params_str}. "
                "Set them in environment variables or your .env file."
            )
            raise ConfigError(msg, config_key="common_configs")

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
