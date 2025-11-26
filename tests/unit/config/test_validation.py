"""Unit tests for ConfigValidator configuration validation.

Tests cover:
- Client-specific validation
- File path validation (input, templates)
- Output directory validation
- Permission error simulation
- Comprehensive validation orchestration
- Python 3.11+ type safety and error handling
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest

from ai_ner_system.config.exceptions import (
    ConfigValidationError,
    DirectoryValidationError,
    FileValidationError,
)
from ai_ner_system.config.settings import Settings
from ai_ner_system.config.validation import ConfigValidator

if TYPE_CHECKING:
    from pathlib import Path


log = logging.getLogger(__name__)


@pytest.mark.usefixtures("no_dotenv")
class TestConfigValidatorClientValidation:
    """Test client-specific configuration validation."""

    @pytest.mark.usefixtures("mock_env_claude", "tmp_input_file")
    def test_validate_for_client_claude_success(self) -> None:
        """Test successful Claude client validation."""
        Settings.initialize(reload_env=False, create_dirs=False)

        # Should not raise
        ConfigValidator.validate_for_client("claude")

    @pytest.mark.usefixtures("mock_env_ollama", "tmp_input_file")
    def test_validate_for_client_ollama_success(self) -> None:
        """Test successful Ollama client validation."""
        Settings.initialize(reload_env=False, create_dirs=False)

        # Should not raise
        ConfigValidator.validate_for_client("ollama")

    def test_validate_for_client_empty_type(self) -> None:
        """Test validation fails with empty client type."""
        Settings.initialize(reload_env=False, create_dirs=False)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigValidator.validate_for_client("")

        assert "Client type must be provided" in str(exc_info.value)

    def test_validate_for_client_whitespace_only(self) -> None:
        """Test validation fails with whitespace-only client type."""
        Settings.initialize(reload_env=False, create_dirs=False)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigValidator.validate_for_client("   ")

        assert "Client type must be provided" in str(exc_info.value)

    def test_validate_for_client_missing_api_key(self, tmp_path: Path) -> None:
        """Test validation fails when API key is missing.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        # Create minimal valid file structure
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test\n")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Template")

        # Initialize settings
        Settings.initialize(reload_env=False, create_dirs=False)

        # Set all required fields EXCEPT API key
        Settings.CLAUDE_MODEL = "claude-sonnet-4"
        Settings.ANTHROPIC_API_KEY = ""  # Empty API key should fail validation
        Settings.INPUT_FILE = str(input_file)
        Settings.OUTPUT_TEXT_FILE = str(output_dir / "text.txt")
        Settings.OUTPUT_TABLE_FILE = str(output_dir / "table.txt")
        Settings.OUTPUT_STATS_FILE = str(output_dir / "stats.json")
        Settings.PROMPT_TEMPLATE_FILE = str(prompt_file)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigValidator.validate_for_client("claude")

        # Verify it's about the API key
        assert "api_key" in str(exc_info.value).lower()

    @pytest.mark.usefixtures("mock_env_claude")
    def test_validate_for_client_case_insensitive(self) -> None:
        """Test that client type validation is case-insensitive."""
        Settings.initialize(reload_env=False, create_dirs=False)

        # All should succeed
        ConfigValidator.validate_for_client("claude")
        ConfigValidator.validate_for_client("CLAUDE")
        ConfigValidator.validate_for_client("Claude")


@pytest.mark.usefixtures("no_dotenv")
class TestConfigValidatorInputFileValidation:
    """Test input file validation."""

    def test_validate_input_file_exists_and_readable(self, tmp_path: Path) -> None:
        """Test validation passes for existing, readable input file.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        input_file = tmp_path / "test_input.txt"
        input_file.write_text("Bindnr;Brevid;Tekst\n001;Test\n")

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.INPUT_FILE = str(input_file)

        # Should not raise
        ConfigValidator._validate_input_file()  # pyright: ignore[reportPrivateUsage]

    def test_validate_input_file_not_exists(self) -> None:
        """Test validation fails for non-existent input file."""
        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.INPUT_FILE = "/nonexistent/path/to/file.txt"

        with pytest.raises(FileValidationError) as exc_info:
            ConfigValidator._validate_input_file()  # pyright: ignore[reportPrivateUsage]

        assert "does not exist" in str(exc_info.value)
        assert exc_info.value.config_key == "INPUT_FILE"

    def test_validate_input_file_empty(self, tmp_path: Path) -> None:
        """Test validation fails for empty input file.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        empty_file = tmp_path / "empty.txt"
        empty_file.touch()

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.INPUT_FILE = str(empty_file)

        with pytest.raises(FileValidationError) as exc_info:
            ConfigValidator._validate_input_file()  # pyright: ignore[reportPrivateUsage]

        assert "empty" in str(exc_info.value).lower()
        assert exc_info.value.config_key == "INPUT_FILE"

    def test_validate_input_file_is_directory(self, tmp_path: Path) -> None:
        """Test validation fails when input path is a directory.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        dir_path = tmp_path / "not_a_file"
        dir_path.mkdir()

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.INPUT_FILE = str(dir_path)

        with pytest.raises(FileValidationError) as exc_info:
            ConfigValidator._validate_input_file()  # pyright: ignore[reportPrivateUsage]

        assert "not a file" in str(exc_info.value)
        assert exc_info.value.config_key == "INPUT_FILE"

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Permission tests unreliable on Windows",
    )
    def test_validate_input_file_not_readable(self, tmp_path: Path) -> None:
        """Test validation fails for unreadable input file.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        unreadable_file = tmp_path / "unreadable.txt"
        unreadable_file.write_text("Test content")
        unreadable_file.chmod(0o000)  # No permissions

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.INPUT_FILE = str(unreadable_file)

        try:
            with pytest.raises(FileValidationError) as exc_info:
                ConfigValidator._validate_input_file()  # pyright: ignore[reportPrivateUsage]

            assert "not readable" in str(exc_info.value)
        finally:
            # Restore permissions for cleanup
            unreadable_file.chmod(0o644)

    def test_validate_input_file_not_configured(self) -> None:
        """Test validation skips when input file is not configured."""
        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.INPUT_FILE = ""

        # Should not raise (optional validation)
        ConfigValidator._validate_input_file()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.usefixtures("no_dotenv")
class TestConfigValidatorTemplateFileValidation:
    """Test template file validation."""

    def test_validate_template_files_exist(self, tmp_path: Path) -> None:
        """Test validation passes for existing template files.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        prompt_template = tmp_path / "prompt.txt"
        prompt_template.write_text("Template: {text}")

        batch_template = tmp_path / "batch.txt"
        batch_template.write_text("Batch: {content}")

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.PROMPT_TEMPLATE_FILE = str(prompt_template)
        Settings.BATCH_TEMPLATE_FILE = str(batch_template)

        # Should not raise
        ConfigValidator._validate_template_files()  # pyright: ignore[reportPrivateUsage]

    def test_validate_template_file_not_exists(self) -> None:
        """Test validation fails for non-existent template file."""
        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.PROMPT_TEMPLATE_FILE = "/nonexistent/template.txt"

        with pytest.raises(FileValidationError) as exc_info:
            ConfigValidator._validate_template_files()  # pyright: ignore[reportPrivateUsage]

        assert "does not exist" in str(exc_info.value)
        assert exc_info.value.config_key == "PROMPT_TEMPLATE_FILE"

    def test_validate_template_files_optional(self) -> None:
        """Test validation skips when template files are not configured."""
        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.PROMPT_TEMPLATE_FILE = ""
        Settings.BATCH_TEMPLATE_FILE = ""

        # Should not raise (optional files)
        ConfigValidator._validate_template_files()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.usefixtures("no_dotenv")
class TestConfigValidatorOutputValidation:
    """Test output directory and path validation."""

    def test_validate_output_paths_writable(self, tmp_path: Path) -> None:
        """Test validation passes for writable output directories.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        output_text = output_dir / "text.txt"
        output_table = output_dir / "table.txt"
        output_stats = output_dir / "stats.json"

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.OUTPUT_TEXT_FILE = str(output_text)
        Settings.OUTPUT_TABLE_FILE = str(output_table)
        Settings.OUTPUT_STATS_FILE = str(output_stats)

        # Should not raise
        ConfigValidator._validate_output_paths_writable()  # pyright: ignore[reportPrivateUsage]

    def test_validate_output_directory_not_exists(self) -> None:
        """Test validation fails when output directory doesn't exist."""
        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.OUTPUT_TEXT_FILE = "/nonexistent/output.txt"

        with pytest.raises(DirectoryValidationError) as exc_info:
            ConfigValidator._validate_output_paths_writable()  # pyright: ignore[reportPrivateUsage]

        assert "does not exist" in str(exc_info.value)
        assert exc_info.value.config_key == "OUTPUT_TEXT_FILE"

    def test_validate_output_path_parent_is_file(self, tmp_path: Path) -> None:
        """Test validation fails when output path parent is a file.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        # Create a file where directory should be
        not_a_dir = tmp_path / "not_a_directory"
        not_a_dir.write_text("I'm a file, not a directory")

        output_file = not_a_dir / "output.txt"

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.OUTPUT_TEXT_FILE = str(output_file)

        with pytest.raises(DirectoryValidationError) as exc_info:
            ConfigValidator._validate_output_paths_writable()  # pyright: ignore[reportPrivateUsage]

        assert "not a directory" in str(exc_info.value)
        assert exc_info.value.config_key == "OUTPUT_TEXT_FILE"

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Permission tests unreliable on Windows",
    )
    def test_validate_output_directory_not_writable(self, tmp_path: Path) -> None:
        """Test validation fails for non-writable output directory.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only

        output_file = readonly_dir / "output.txt"

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.OUTPUT_TEXT_FILE = str(output_file)

        try:
            with pytest.raises(DirectoryValidationError) as exc_info:
                ConfigValidator._validate_output_paths_writable()  # pyright: ignore[reportPrivateUsage]

            assert "not writable" in str(exc_info.value)
            assert exc_info.value.config_key == "OUTPUT_TEXT_FILE"
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)


@pytest.mark.usefixtures("no_dotenv")
class TestConfigValidatorFilePathValidation:
    """Test comprehensive file path validation."""

    @pytest.mark.usefixtures("mock_env_claude")
    def test_validate_file_paths_success(self, tmp_path: Path) -> None:
        """Test successful file path validation.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        # Create input file
        input_file = tmp_path / "input.txt"
        input_file.write_text("Bindnr;Brevid;Tekst\n001;Test\n")

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create template files
        prompt_template = tmp_path / "prompt.txt"
        prompt_template.write_text("Template")

        batch_prompt_template = tmp_path / "batch_prompt.txt"
        batch_prompt_template.write_text("Batch Template")

        Settings.initialize(reload_env=False, create_dirs=False)

        Settings.INPUT_FILE = str(input_file)
        Settings.OUTPUT_TEXT_FILE = str(output_dir / "text.txt")
        Settings.OUTPUT_TABLE_FILE = str(output_dir / "table.txt")
        Settings.OUTPUT_STATS_FILE = str(output_dir / "stats.json")
        Settings.PROMPT_TEMPLATE_FILE = str(prompt_template)
        Settings.BATCH_TEMPLATE_FILE = str(batch_prompt_template)

        # Should not raise
        ConfigValidator.validate_file_paths()

    def test_validate_file_paths_input_file_error(self) -> None:
        """Test file path validation fails on input file error."""
        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.INPUT_FILE = "/nonexistent/input.txt"

        with pytest.raises(ConfigValidationError):
            ConfigValidator.validate_file_paths()


@pytest.mark.usefixtures("no_dotenv")
class TestConfigValidatorComprehensiveValidation:
    """Test comprehensive validation functionality."""

    @pytest.mark.usefixtures("mock_env_claude")
    def test_validate_all_success(self, tmp_path: Path) -> None:
        """Test successful comprehensive validation.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        # Set up complete valid configuration
        input_file = tmp_path / "input.txt"
        input_file.write_text("Bindnr;Brevid;Tekst\n001;Test\n")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        prompt_template = tmp_path / "prompt.txt"
        prompt_template.write_text("Template")
        batch_prompt_template = tmp_path / "batch_prompt.txt"
        batch_prompt_template.write_text("Batch Template")

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.INPUT_FILE = str(input_file)
        Settings.OUTPUT_TEXT_FILE = str(output_dir / "text.txt")
        Settings.OUTPUT_TABLE_FILE = str(output_dir / "table.txt")
        Settings.OUTPUT_STATS_FILE = str(output_dir / "stats.json")
        Settings.PROMPT_TEMPLATE_FILE = str(prompt_template)
        Settings.BATCH_TEMPLATE_FILE = str(batch_prompt_template)

        # Should not raise
        ConfigValidator.validate_all(client_type="claude")

    def test_validate_all_without_client_type(self, tmp_path: Path) -> None:
        """Test validation without client type only validates paths.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.INPUT_FILE = str(input_file)
        Settings.OUTPUT_TEXT_FILE = str(output_dir / "text.txt")
        Settings.PROMPT_TEMPLATE_FILE = ""
        Settings.BATCH_TEMPLATE_FILE = ""

        # Should not raise (no client validation)
        ConfigValidator.validate_all(client_type=None)

    @pytest.mark.usefixtures("mock_env_claude")
    def test_is_valid_returns_true(self, tmp_path: Path) -> None:
        """Test is_valid returns True for valid configuration.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        # Set up valid configuration
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test\n")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create prompt template file (required for validation)
        prompt_template = tmp_path / "prompt.txt"
        prompt_template.write_text("Template content")
        batch_prompt_template = tmp_path / "batch_prompt.txt"
        batch_prompt_template.write_text("Batch Template")

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.INPUT_FILE = str(input_file)
        Settings.OUTPUT_TEXT_FILE = str(output_dir / "text.txt")
        Settings.OUTPUT_TABLE_FILE = str(output_dir / "table.txt")
        Settings.OUTPUT_STATS_FILE = str(output_dir / "stats.json")
        Settings.PROMPT_TEMPLATE_FILE = str(prompt_template)
        Settings.BATCH_TEMPLATE_FILE = str(batch_prompt_template)

        assert ConfigValidator.is_valid(client_type="claude") is True

    def test_is_valid_returns_false(self) -> None:
        """Test is_valid returns False for invalid configuration."""
        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.INPUT_FILE = "/nonexistent/file.txt"

        assert ConfigValidator.is_valid(client_type="claude") is False

    def test_is_valid_silent_mode(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test is_valid in silent mode suppresses warnings.

        Args:
            caplog: Pytest fixture for capturing log output.
        """
        # Capture only WARNING level (silent mode shouldn't produce warnings)
        caplog.set_level(logging.WARNING)

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.INPUT_FILE = "/nonexistent/file.txt"

        result = ConfigValidator.is_valid(client_type="claude", silent=True)

        assert result is False
        # Should not log at WARNING level in silent mode
        # (ERROR logs from validate_all are internal and expected)
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) == 0

    def test_is_valid_non_silent_mode(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test is_valid in non-silent mode logs warnings.

        Args:
            caplog: Pytest fixture for capturing log output.
        """
        caplog.set_level(logging.WARNING)

        Settings.initialize(reload_env=False, create_dirs=False)
        Settings.INPUT_FILE = "/nonexistent/file.txt"

        result = ConfigValidator.is_valid(client_type="claude", silent=False)

        assert result is False
        # Should log warnings in non-silent mode
        assert "Configuration validation failed" in caplog.text


class TestConfigValidatorHelperMethods:
    """Test ConfigValidator helper methods and edge cases."""

    def test_validate_file_exists_and_readable_success(self, tmp_path: Path) -> None:
        """Test file existence and readability check passes.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")

        # Should not raise
        ConfigValidator._validate_file_exists_and_readable(  # pyright: ignore[reportPrivateUsage]
            test_file,
            "TEST_KEY",
            "Test file",
        )

    def test_validate_file_exists_and_readable_not_exists(self, tmp_path: Path) -> None:
        """Test file validation fails for non-existent file.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        nonexistent = tmp_path / "nonexistent.txt"

        with pytest.raises(FileValidationError) as exc_info:
            ConfigValidator._validate_file_exists_and_readable(  # pyright: ignore[reportPrivateUsage]
                nonexistent,
                "TEST_KEY",
                "Test file",
            )

        assert "does not exist" in str(exc_info.value)
        assert exc_info.value.config_key == "TEST_KEY"

    def test_validate_output_directory_writable_success(self, tmp_path: Path) -> None:
        """Test output directory writability check passes.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        output_file = output_dir / "test.txt"

        # Should not raise
        ConfigValidator._validate_output_directory_writable(  # pyright: ignore[reportPrivateUsage]
            output_file,
            "TEST_KEY",
            "test file",
        )

    def test_constants_are_defined(self) -> None:
        """Test that ConfigValidator constants are properly defined."""
        assert "PROMPT_TEMPLATE_FILE" in ConfigValidator.TEMPLATE_FILES
        assert "BATCH_TEMPLATE_FILE" in ConfigValidator.TEMPLATE_FILES
        assert "OUTPUT_TEXT_FILE" in ConfigValidator.OUTPUT_FILES
        assert "OUTPUT_TABLE_FILE" in ConfigValidator.OUTPUT_FILES
        assert "OUTPUT_STATS_FILE" in ConfigValidator.OUTPUT_FILES

    def test_constant_tuples_are_correct(self) -> None:
        """Test that constant tuples match expected attributes."""
        assert "PROMPT_TEMPLATE_FILE" in ConfigValidator.TEMPLATE_FILE_ATTRS
        assert "BATCH_TEMPLATE_FILE" in ConfigValidator.TEMPLATE_FILE_ATTRS
        assert "OUTPUT_TEXT_FILE" in ConfigValidator.OUTPUT_FILE_ATTRS
        assert "OUTPUT_TABLE_FILE" in ConfigValidator.OUTPUT_FILE_ATTRS
        assert "OUTPUT_STATS_FILE" in ConfigValidator.OUTPUT_FILE_ATTRS
