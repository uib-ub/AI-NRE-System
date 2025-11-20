"""Unit tests for Settings configuration management.

Tests cover:
- Initialization with environment variables
- CLI overrides
- Directory creation
- Client-specific configuration
- Error handling and validation
- Python 3.11+ type safety
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ai_ner_system.config.exceptions import ConfigError
from ai_ner_system.config.settings import Settings

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

log = logging.getLogger(__name__)


@pytest.mark.usefixtures("no_dotenv")
class TestSettingsInitialization:
    """Test Settings class initialization and loading."""

    @pytest.mark.parametrize(
        ("setting_attr", "setting_default"),
        [
            ("INPUT_FILE", "DEFAULT_INPUT_FILE"),
            ("OUTPUT_TEXT_FILE", "DEFAULT_OUTPUT_TEXT_FILE"),
            ("OUTPUT_TABLE_FILE", "DEFAULT_OUTPUT_TABLE_FILE"),
            ("OUTPUT_STATS_FILE", "DEFAULT_OUTPUT_STATS_FILE"),
            ("PROMPT_TEMPLATE_FILE", "DEFAULT_PROMPT_TEMPLATE_FILE"),
            ("BATCH_TEMPLATE_FILE", "DEFAULT_BATCH_TEMPLATE_FILE"),
        ],
    )
    def test_initialize_with_defaults(
        self,
        setting_attr: str,
        setting_default: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test initialization with default environment settings.

        Args:
            setting_attr: The Settings attribute to check.
            setting_default: The corresponding default constant attribute.
            tmp_path: Pytest fixture providing a temporary directory.
            monkeypatch: Pytest fixture for environment modification.
        """
        monkeypatch.chdir(tmp_path)
        Settings.initialize(reload_env=False, create_dirs=False)

        actual = getattr(Settings, setting_attr)
        expected = str((tmp_path / getattr(Settings, setting_default)).resolve())

        log.debug("%s: actual=%r, expected=%r", setting_attr, actual, expected)

        # Check that configuration loaded from environment
        assert actual == expected, (
            f"{setting_attr} should default to {expected}, but got {actual}"
        )

    @pytest.mark.usefixtures("mock_env_claude")
    def test_initialize_from_environment_claude(self) -> None:
        """Test loading Claude configuration from environment variables."""
        # Don't use load_dotenv to avoid .env file interference
        Settings.initialize(reload_env=False, create_dirs=False)

        assert Settings.ANTHROPIC_API_KEY == "sk-ant-test-key-123456789"
        assert Settings.CLAUDE_MODEL == "claude-sonnet-4"

    @pytest.mark.usefixtures("mock_env_ollama")
    def test_initialize_from_environment_ollama(self) -> None:
        """Test loading Ollama configuration from environment variables."""
        # Don't use load_dotenv to avoid .env file interference
        Settings.initialize(reload_env=False, create_dirs=False)

        assert Settings.OPENWEBUI_ENDPOINT == "http://localhost:11434"
        assert Settings.OPENWEBUI_TOKEN == "test-token-123"
        assert Settings.OLLAMA_MODEL == "gemma3:12b"

    def test_initialize_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that multiple initialize() calls are safe (no-op after first)."""
        # Set initial environment value
        monkeypatch.setenv("CLAUDE_MODEL", "claude-first")
        Settings.initialize(reload_env=False, create_dirs=False)
        first_value = Settings.CLAUDE_MODEL
        # Change environment and call initialize again (should be no-op)
        monkeypatch.setenv("CLAUDE_MODEL", "claude-second")
        Settings.initialize(reload_env=False, create_dirs=False)  # Should not reload
        second_value = Settings.CLAUDE_MODEL
        # Value should remain unchanged because initialize is idempotent
        assert first_value == "claude-first"
        assert second_value == "claude-first"

    def test_initialize_with_reload_env_reloads_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """reload_env=True resets and reloads environment-derived values."""
        monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-4")
        Settings.initialize(reload_env=False, create_dirs=False)
        assert Settings.CLAUDE_MODEL == "claude-sonnet-4"

        # Change env and call with reload_env=True → should update in-memory value.
        monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-3")
        Settings.initialize(reload_env=True, create_dirs=False)
        assert Settings.CLAUDE_MODEL == "claude-sonnet-3"

    @pytest.mark.parametrize(
        ("env_var", "setting_attr", "is_path"),
        [
            ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY", False),
            ("CLAUDE_MODEL", "CLAUDE_MODEL", False),
            ("OPENWEBUI_TOKEN", "OPENWEBUI_TOKEN", False),
            ("OLLAMA_MODEL", "OLLAMA_MODEL", False),
            ("INPUT_FILE", "INPUT_FILE", True),
        ],
    )
    def test_initialize_parametrized_env_vars(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        env_var: str,
        setting_attr: str,
        is_path: bool,
    ) -> None:
        """Test loading various environment variables parametrically.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            monkeypatch: Pytest fixture for environment modification.
            env_var: Environment variable name to set.
            setting_attr: Settings attribute to check.
            is_path: Whether the setting is a file path (requires normalization).
        """
        if is_path:
            test_value = str((tmp_path / "custom" / "test_input.txt").resolve())
        else:
            test_value = "some-value"

        monkeypatch.setenv(env_var, test_value)
        Settings.initialize(reload_env=False, create_dirs=False)

        actual = getattr(Settings, setting_attr)
        assert actual == (test_value if not is_path else str(Path(test_value)))

    def test_initialize_does_not_create_dirs_when_disabled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When create_dirs=False, no output/cache directories are created.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            monkeypatch: Pytest fixture for environment modification.
        """
        monkeypatch.chdir(tmp_path)
        # Point an output file into the sandbox; directory should not be created.
        out_text = tmp_path / "out" / "x.txt"
        monkeypatch.setenv("OUTPUT_TEXT_FILE", str(out_text))

        Settings.initialize(reload_env=False, create_dirs=False)
        assert not out_text.parent.exists()

    def test_initialize_creates_cache_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that cache directory is created during initialization.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            monkeypatch: Pytest fixture for environment modification.
        """
        monkeypatch.chdir(tmp_path)
        cache_dir = tmp_path / ".test_cache"
        monkeypatch.setenv("CACHE_DIR", str(cache_dir))

        # Neutralize other outputs so we don't create defaults outside sandbox.
        monkeypatch.setenv("OUTPUT_TEXT_FILE", str(tmp_path / "o" / "t.txt"))
        monkeypatch.setenv("OUTPUT_TABLE_FILE", str(tmp_path / "o" / "tab.txt"))
        monkeypatch.setenv("OUTPUT_STATS_FILE", str(tmp_path / "o" / "stats.json"))

        Settings.initialize(reload_env=False, create_dirs=True)

        assert cache_dir.exists()
        assert cache_dir.is_dir()
        assert cache_dir.expanduser() == Settings.CACHE_DIR

    def test_initialize_creates_output_directories(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that output directories are created during initialization.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            monkeypatch: Pytest fixture for environment modification.
        """
        monkeypatch.chdir(tmp_path)
        out_text = tmp_path / "output" / "text.txt"
        out_table = tmp_path / "output" / "table.txt"
        out_stats = tmp_path / "output" / "stats.json"

        monkeypatch.setenv("OUTPUT_TEXT_FILE", str(out_text))
        monkeypatch.setenv("OUTPUT_TABLE_FILE", str(out_table))
        monkeypatch.setenv("OUTPUT_STATS_FILE", str(out_stats))

        Settings.initialize(reload_env=False, create_dirs=True)

        assert out_text.parent.exists()
        assert out_table.parent.exists()
        assert out_stats.parent.exists()

    def test_initialize_raises_config_error_on_os_error(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test OSError handling using mock.

        This test uses mocking to inject an OSError without filesystem manipulation.

        GIVEN `_create_cache_directory` fails with `OSError("Permission denied")`
        WHEN `Settings.initialize(create_dirs=True)` is called
        THEN a `ConfigError` is raised whose message mentions initialization failure,
        and the original `OSError` is preserved via exception chaining (`__cause__`).
        """
        # Mock _create_cache_directory to raise OSError,
        # forcing the directory creation to fail
        mock_create = mocker.patch.object(
            Settings,
            "_create_cache_directory",
            side_effect=OSError("Permission denied"),
        )

        # Act + Assert: ConfigError is raised with the expected message
        with pytest.raises(
            ConfigError, match="Failed to initialize configuration"
        ) as exc_info:
            Settings.initialize(reload_env=False, create_dirs=True)

        # Assert: the original OSError is chained as the cause with the right message
        cause = exc_info.value.__cause__
        assert isinstance(cause, OSError)
        assert "Permission denied" in str(cause)

        # Assert: actually hit the code path we intended
        mock_create.assert_called_once()


@pytest.mark.usefixtures("no_dotenv")
class TestSettingsClientConfiguration:
    """Test client-specific configuration retrieval and validation."""

    @pytest.mark.usefixtures("mock_env_claude")
    def test_get_client_init_params_claude_success(self) -> None:
        """Test successful retrieval of Claude client initialization parameters."""
        Settings.initialize(reload_env=False, create_dirs=False)

        params = Settings.get_client_init_params("claude")

        expected = {
            "api_key": "sk-ant-test-key-123456789",
            "model": "claude-sonnet-4",
        }

        assert params == expected, f"Expected {expected}, but got {params}"

    @pytest.mark.usefixtures("mock_env_ollama")
    def test_get_client_init_params_ollama_success(
        self,
    ) -> None:
        """Test successful retrieval of Ollama client initialization parameters.

        Args:
            mock_env_ollama: Fixture setting Ollama-related environment variables defined in conftest
        """
        Settings.initialize(reload_env=False, create_dirs=False)

        params = Settings.get_client_init_params("ollama")

        expected = {
            "endpoint": "http://localhost:11434",
            "token": "test-token-123",
            "model": "gemma3:12b",
        }

        assert params == expected, f"Expected {expected}, but got {params}"

    @pytest.mark.usefixtures("mock_env_claude")
    def test_get_client_init_params_case_insensitive(self) -> None:
        """Test that client type is case-insensitive."""
        Settings.initialize(reload_env=False, create_dirs=False)

        params_lower = Settings.get_client_init_params("claude")
        params_upper = Settings.get_client_init_params("CLAUDE")
        params_mixed = Settings.get_client_init_params("Claude")

        assert params_lower == params_upper == params_mixed, (
            "Client initialization parameters should be case-insensitive"
        )

    @pytest.mark.parametrize(
        ("client_type", "required_params"),
        [
            ("claude", {"model": "claude-4-opus", "api_key": ""}),  # Missing API key
            (
                "claude",
                {"model": "", "api_key": "sk-ant-test-key-123456789"},
            ),  # missing model
            ("claude", {"model": "", "api_key": ""}),  # both missing
            (
                "ollama",
                {"model": "gemma3:12b", "token": "test-token-123", "endpoint": ""},
            ),  # Missing endpoint
            (
                "ollama",
                {
                    "model": "gemma3:12b",
                    "token": "",
                    "endpoint": "http://localhost:11434",
                },
            ),  # Missing token
        ],
    )
    def test_get_client_init_params_missing_required(
        self,
        client_type: str,
        required_params: dict[str, str],
    ) -> None:
        """Test error when required Claude parameters are missing.

        Args:
            client_type: The client type to test ("claude" or "ollama").
            required_params: Dictionary of parameters to set in Settings,
                             with some intentionally missing/empty.
        """
        Settings.initialize(reload_env=False, create_dirs=False)

        if client_type == "claude":
            Settings.CLAUDE_MODEL = required_params.get("model")
            Settings.ANTHROPIC_API_KEY = required_params.get("api_key")
        elif client_type == "ollama":
            Settings.OLLAMA_MODEL = required_params.get("model")
            Settings.OPENWEBUI_TOKEN = required_params.get("token")
            Settings.OPENWEBUI_ENDPOINT = required_params.get("endpoint")

        with pytest.raises(ConfigError) as exc_info:
            Settings.get_client_init_params(client_type)

        error_msg = str(exc_info.value)
        assert "Missing or empty required configuration" in error_msg
        assert any(s in error_msg for s in ("api_key", "model", "endpoint", "token"))

    @pytest.mark.parametrize(
        ("raw", "ok"),
        [
            ("  model-x  ", True),
            ("   ", False),
            ("", False),
        ],
    )
    def test_get_client_init_params_trims_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
        raw: str,
        ok: bool,
    ) -> None:
        """Test that leading/trailing whitespace in config values is trimmed.

        Args:
            monkeypatch: Pytest fixture for environment modification.
            raw: Raw string value to set for the model.
            ok: Whether the trimmed value should be considered valid.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-123456789")
        monkeypatch.setenv("CLAUDE_MODEL", raw)
        Settings.initialize(reload_env=False, create_dirs=False)
        if ok:
            assert Settings.get_client_init_params("claude")["model"] == "model-x"
        else:
            with pytest.raises(ConfigError):
                Settings.get_client_init_params("claude")

    def test_get_client_init_params_unsupported_client(
        self,
    ) -> None:
        """Test error for unsupported client type."""
        Settings.initialize(reload_env=False, create_dirs=False)

        with pytest.raises(ConfigError) as exc_info:
            Settings.get_client_init_params("unsupported_client")

        assert "Unsupported client type" in str(exc_info.value)
        assert "unsupported_client" in str(exc_info.value)

    def test_get_client_init_params_empty_string_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that empty string values are rejected.

        Args:
            monkeypatch: Pytest fixture for environment modification.
        """
        # Set values to empty strings
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")  # Whitespace only
        monkeypatch.setenv("CLAUDE_MODEL", "")
        Settings.initialize(reload_env=False, create_dirs=False)

        with pytest.raises(ConfigError) as exc_info:
            Settings.get_client_init_params("claude")

        assert "Missing or empty" in str(exc_info.value)

    @pytest.mark.usefixtures("mock_env_claude")
    def test_validate_client_config_success(self) -> None:
        """Test successful client config validation."""
        Settings.initialize(reload_env=False, create_dirs=False)

        # Should not raise
        Settings.validate_client_config("claude")

    def test_validate_client_config_failure(self) -> None:
        """Test client config validation failure."""
        Settings.initialize(reload_env=False, create_dirs=False)

        # Clear API key to cause failure
        Settings.ANTHROPIC_API_KEY = ""

        with pytest.raises(ConfigError):
            Settings.validate_client_config("claude")


@pytest.mark.usefixtures("no_dotenv")
class TestSettingsCLIOverrides:
    """Test command-line override functionality."""

    def test_apply_cli_overrides_input_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Test overriding input file path.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        Settings.initialize(reload_env=False, create_dirs=False)
        custom_input = tmp_path / "custom_input.txt"
        custom_input.touch()

        Settings.apply_cli_overrides(input_file=str(custom_input))

        assert str(custom_input.resolve()) == Settings.INPUT_FILE

    def test_apply_cli_overrides_output_files(
        self,
        tmp_path: Path,
    ) -> None:
        """Test overriding all output file paths.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        Settings.initialize(reload_env=False, create_dirs=False)

        custom_text = tmp_path / "output" / "custom_text.txt"
        custom_table = tmp_path / "output" / "custom_table.txt"
        custom_stats = tmp_path / "output" / "custom_stats.json"

        Settings.apply_cli_overrides(
            output_text_file=str(custom_text),
            output_table_file=str(custom_table),
            output_stats_file=str(custom_stats),
        )

        assert str(custom_text.resolve()) == Settings.OUTPUT_TEXT_FILE
        assert str(custom_table.resolve()) == Settings.OUTPUT_TABLE_FILE
        assert str(custom_stats.resolve()) == Settings.OUTPUT_STATS_FILE

        # Verify directories were created
        assert custom_text.parent.exists()
        assert custom_table.parent.exists()
        assert custom_stats.parent.exists()

    def test_apply_cli_overrides_template_files(
        self,
        tmp_path: Path,
    ) -> None:
        """Test overriding template file paths.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        Settings.initialize(reload_env=False, create_dirs=False)

        custom_prompt = tmp_path / "custom_prompt.txt"
        custom_batch = tmp_path / "custom_batch.txt"
        custom_prompt.touch()
        custom_batch.touch()

        Settings.apply_cli_overrides(
            prompt_template_file=str(custom_prompt),
            batch_template_file=str(custom_batch),
        )

        assert str(custom_prompt.resolve()) == Settings.PROMPT_TEMPLATE_FILE
        assert str(custom_batch.resolve()) == Settings.BATCH_TEMPLATE_FILE

    def test_apply_cli_overrides_creates_nested_directories(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that deeply nested output directories are created.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        Settings.initialize(reload_env=False, create_dirs=False)

        deep_output = tmp_path / "level1" / "level2" / "level3" / "output.txt"

        Settings.apply_cli_overrides(output_text_file=str(deep_output))

        assert deep_output.parent.exists()
        assert str(deep_output.resolve()) == Settings.OUTPUT_TEXT_FILE

    def test_apply_cli_overrides_normalizes_paths(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that CLI overrides normalize paths (expand ~ and resolve).

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            monkeypatch: Pytest fixture for modifying environment variables.
        """
        # Create a test file in temp
        test_file = tmp_path / "test.txt"
        test_file.touch()

        # Set HOME to tmp_path for testing ~ expansion
        monkeypatch.setenv("HOME", str(tmp_path))

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.apply_cli_overrides(input_file="~/test.txt")

        # Should expand ~ to tmp_path
        assert str(test_file) == Settings.INPUT_FILE


@pytest.mark.usefixtures("no_dotenv")
class TestSettingsPathNormalization:
    """Test path normalization and directory operations."""

    def test_normalize_path_expands_user_home(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test that ~ is expanded to user home directory.

        Args:
            monkeypatch: Pytest fixture for modifying environment variables.
            tmp_path: Pytest fixture providing a temporary directory.
        """
        monkeypatch.setenv("HOME", str(tmp_path))

        normalized = Settings._normalize_path("~/test.txt")  # pyright: ignore[reportPrivateUsage]

        assert normalized == str(tmp_path / "test.txt")
        assert "~" not in normalized

    def test_normalize_path_resolves_relative(
        self,
    ) -> None:
        """Test that relative paths are resolved to absolute."""
        normalized = Settings._normalize_path("./relative/path.txt")  # pyright: ignore[reportPrivateUsage]

        assert Path(normalized).is_absolute()

    def test_ensure_directory_exists_creates_missing(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that missing directories are created.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        new_dir = tmp_path / "new" / "nested" / "dir" / "file.txt"

        Settings._ensure_directory_exists(str(new_dir))  # pyright: ignore[reportPrivateUsage]

        assert new_dir.parent.exists()

    def test_ensure_directory_exists_handles_existing(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that existing directories don't cause errors.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        existing_file = tmp_path / "existing" / "file.txt"
        existing_file.parent.mkdir(parents=True, exist_ok=True)

        # Should not raise
        Settings._ensure_directory_exists(str(existing_file))  # pyright: ignore[reportPrivateUsage]

        assert existing_file.parent.exists()


@pytest.mark.usefixtures("no_dotenv")
class TestSettingsReset:
    """Test Settings reset functionality."""

    def test_reset_clears_initialization_flag(self) -> None:
        """Test that reset clears the initialization flag."""
        Settings.initialize(reload_env=False, create_dirs=False)
        assert Settings._initialized is True  # pyright: ignore[reportPrivateUsage]

        Settings.reset()

        assert Settings._initialized is False  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.usefixtures("mock_env_claude")
    def test_reset_clears_api_configuration(self) -> None:
        """Test that reset clears API-related configuration."""
        Settings.initialize(reload_env=False, create_dirs=False)
        assert Settings.ANTHROPIC_API_KEY is not None
        assert Settings.CLAUDE_MODEL is not None

        Settings.reset()
        assert Settings.ANTHROPIC_API_KEY is None
        assert Settings.CLAUDE_MODEL is None  # type: ignore[unreachable]

    def test_reset_restores_defaults(self) -> None:
        """Test that reset restores default values."""
        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.INPUT_FILE = "custom/path.txt"

        Settings.reset()
        assert Settings.INPUT_FILE == Settings.DEFAULT_INPUT_FILE

    def test_reset_clears_model_configuration(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that reset clears model configuration.

        Args:
            monkeypatch: Pytest fixture for modifying environment variables.
        """
        monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-4")
        monkeypatch.setenv("OLLAMA_MODEL", "gemma3:12b")

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.reset()

        assert Settings.CLAUDE_MODEL is None
        assert Settings.OLLAMA_MODEL is None


@pytest.mark.usefixtures("no_dotenv")
class TestSettingsCommonValidation:
    """Test common configuration validation."""

    def test_validate_common_config_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test successful common configuration validation.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            monkeypatch: Pytest fixture for modifying environment variables.
        """
        # Create necessary files
        input_file = tmp_path / "input.txt"
        input_file.touch()

        # Set all required common config
        monkeypatch.setenv("INPUT_FILE", str(input_file))
        monkeypatch.setenv("OUTPUT_TEXT_FILE", str(tmp_path / "output.txt"))
        monkeypatch.setenv("OUTPUT_TABLE_FILE", str(tmp_path / "table.txt"))
        monkeypatch.setenv("OUTPUT_STATS_FILE", str(tmp_path / "stats.json"))
        monkeypatch.setenv("PROMPT_TEMPLATE_FILE", str(tmp_path / "prompt.txt"))

        # No need to create directories for this validation.
        Settings.initialize(reload_env=False, create_dirs=False)
        # Should not raise
        Settings.validate_common_config()

    def test_validate_common_config_missing_files(self) -> None:
        """Test validation fails when required files are missing."""
        Settings.initialize(reload_env=False, create_dirs=False)

        # Force a missing value
        Settings.INPUT_FILE = ""

        with pytest.raises(ConfigError) as exc_info:
            Settings.validate_common_config()

        assert "Missing or empty required common configuration" in str(exc_info.value)
        assert "INPUT_FILE" in str(exc_info.value)


class TestSettingsConstants:
    """Test that Settings constants are properly defined."""

    def test_supported_clients_includes_claude_and_ollama(self) -> None:
        """Test that SUPPORTED_CLIENTS includes expected client types."""
        assert "claude" in Settings.SUPPORTED_CLIENTS
        assert "ollama" in Settings.SUPPORTED_CLIENTS

    def test_supported_clients_is_frozen(self) -> None:
        """Test that SUPPORTED_CLIENTS is immutable (frozenset)."""
        assert isinstance(Settings.SUPPORTED_CLIENTS, frozenset)

        with pytest.raises((TypeError, AttributeError)):
            Settings.SUPPORTED_CLIENTS.add("new_client")  # type: ignore[attr-defined]

    def test_supported_clients_matches_registry_keys(self) -> None:
        """SUPPORTED_CLIENTS matches the keys of the client registry."""
        assert (
            frozenset(
                Settings._CLIENT_CONFIG_REGISTRY.keys()  # pyright: ignore[reportPrivateUsage]
            )
            == Settings.SUPPORTED_CLIENTS
        )

    @pytest.mark.parametrize(
        ("default_constant_attr", "expected_type"),
        [
            ("DEFAULT_INPUT_FILE", str),
            ("DEFAULT_OUTPUT_TEXT_FILE", str),
            ("DEFAULT_OUTPUT_TABLE_FILE", str),
            ("DEFAULT_OUTPUT_STATS_FILE", str),
            ("DEFAULT_PROMPT_TEMPLATE_FILE", str),
            ("DEFAULT_BATCH_TEMPLATE_FILE", str),
        ],
    )
    def test_default_constants_are_strings(
        self,
        default_constant_attr: str,
        expected_type: type,
    ) -> None:
        """Test that default file path constants are strings."""
        constant_value = getattr(Settings, default_constant_attr)
        assert isinstance(constant_value, expected_type), (
            f"{default_constant_attr} should be of type {expected_type.__name__}, "
            f"but got {type(constant_value).__name__}"
        )

    def test_default_concurrency_constants_are_positive(self) -> None:
        """Test that concurrency constants are positive integers."""
        assert Settings.DEFAULT_MAX_CONCURRENT_BATCHES > 0
        assert Settings.DEFAULT_MAX_CONCURRENT_INDIVIDUAL > 0
        assert Settings.DEFAULT_FALLBACK_CONCURRENCY > 0

    def test_default_batch_size_is_positive(self) -> None:
        """Test that default batch size is positive."""
        assert Settings.DEFAULT_BATCH_SIZE > 0

    def test_client_config_registry_structure(self) -> None:
        """Test that client config registry has expected structure."""
        registry = Settings._CLIENT_CONFIG_REGISTRY  # pyright: ignore[reportPrivateUsage]

        assert "claude" in registry
        assert "ollama" in registry

        # Each entry should be a list of tuples
        for config_list in registry.values():
            assert isinstance(config_list, list)
            for entry in config_list:
                assert isinstance(entry, tuple)
                assert len(entry) == 2
                assert isinstance(entry[0], str)  # Settings attribute
                assert isinstance(entry[1], str)  # Init parameter
