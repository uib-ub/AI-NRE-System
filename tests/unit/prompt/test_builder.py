"""Unit tests for prompt.builder module.

Tests cover:
- Template loading and validation
- Placeholder extraction
- Single-record prompt building
- Batch prompt building
- Unicode text handling (medieval Norwegian: æ, ø, å, ð, þ)
- Error handling for missing fields, empty text, invalid templates
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from ai_ner_system.prompt.builder import GenericPromptBuilder
from ai_ner_system.prompt.exceptions import (
    PromptBuildError,
    PromptError,
    TemplateNotFoundError,
)

log = logging.getLogger(__name__)


class TestGenericPromptBuilder:
    """Unit tests for GenericPromptBuilder."""

    # Constants from GenericPromptBuilder
    SRC_KEY_BREVID = GenericPromptBuilder.SRC_KEY_BREVID
    SRC_KEY_TEXT = GenericPromptBuilder.SRC_KEY_TEXT

    # ====================
    # Fixtures
    # ====================
    @pytest.fixture
    def empty_template_file(self, tmp_path: Path) -> Path:
        """Create an empty template file."""
        template = tmp_path / "empty_template.txt"
        template.touch()
        return template

    @pytest.fixture
    def sample_record(self) -> dict[str, Any]:
        """Create a sample record with standard keys."""
        return {
            self.SRC_KEY_BREVID: "001",
            self.SRC_KEY_TEXT: "Ollum monnum þæim sæm þetta bref sea æder høyra sændir "
            "Olauer med gudz nadh abote j Olafsklaustre j Tunsbergi q. g. ok sina ...",
        }

    @pytest.fixture
    def sample_unicode_record(self) -> dict[str, Any]:
        """Create a sample record with medieval Norwegian unicode characters."""
        return {
            self.SRC_KEY_BREVID: "002",
            self.SRC_KEY_TEXT: "Hákon konungr réð fyrir Nóregi með æ, ø, å, ð, þ",
        }

    @pytest.fixture
    def sample_batch_records(self) -> list[dict[str, Any]]:
        """Create a list of sample records for batch processing."""
        return [
            {
                self.SRC_KEY_BREVID: "001",
                self.SRC_KEY_TEXT: "Ollum monnum þæim sæm þetta bref sea æder høyra sændir "
                "Olauer med gudz nadh abote j Olafsklaustre j Tunsbergi q. g. ok sina ...",
            },
            {
                self.SRC_KEY_BREVID: "002",
                self.SRC_KEY_TEXT: "Second record with special: æ, ø, å, ð, þ",
            },
            {
                self.SRC_KEY_BREVID: "003",
                self.SRC_KEY_TEXT: "Ollom monnom þeim sæm þetta bref sea ædher høyra sænda "
                "Hakon Amundason ok Arne Drængsson quædiu gudz ok sina kunnikt gerande ...",
            },
        ]

    # ====================
    # Template Loading Tests
    # ====================
    def test_load_template_success(self, single_template_file: Path) -> None:
        """Test successful template loading.

        Args:
            single_template_file: a simple single-record template file defined in fixture.
        """
        builder = GenericPromptBuilder(single_template_file)

        log.debug("Template loaded:\n%s", builder.template)
        assert builder.template == "Brevid: {brevid}\nText: {text}"
        assert builder.template_file == single_template_file

    def test_load_template_file_not_found(self, tmp_path: Path) -> None:
        """Test template loading raises error when file does not exist.

        Args:
            tmp_path: temporary directory provided by pytest.
        """
        nonexistent = tmp_path / "nonexistent.txt"
        with pytest.raises(TemplateNotFoundError) as exc_info:
            GenericPromptBuilder(nonexistent)

        log.debug("Template load error:\n%s", exc_info.value)

        assert "Template file not found" in str(exc_info.value)
        assert exc_info.value.operation == "load"
        assert str(nonexistent) in str(exc_info.value)
        assert exc_info.value.template_file == nonexistent

    def test_load_template_not_a_file(self, tmp_path: Path) -> None:
        """Test template loading raises error when path is a directory."""
        directory = tmp_path / "directory"
        directory.mkdir()

        with pytest.raises(PromptError) as exc_info:
            GenericPromptBuilder(directory)

        log.debug("Template load error:\n%s", exc_info.value)

        assert "Template path is not a file" in str(exc_info.value)
        assert exc_info.value.operation == "load"
        assert exc_info.value.template_file == directory

    def test_load_template_empty_file(self, empty_template_file: Path) -> None:
        """Test template loading raises error when file is empty.

        Args:
            empty_template_file: fixture creating an empty template file.
        """
        with pytest.raises(PromptError) as exc_info:
            GenericPromptBuilder(empty_template_file)

        log.debug("Template load error:\n%s", exc_info.value)

        assert "template file is empty" in str(exc_info.value).lower()
        assert exc_info.value.operation == "load"
        assert exc_info.value.template_file == empty_template_file

    def test_load_template_read_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test template loading handles read errors gracefully.

        Args:
            tmp_path: temporary directory provided by pytest.
            monkeypatch: pytest fixture for monkeypatching.
        """
        template = tmp_path / "template.txt"
        template.write_text("Test template: {brevid}")

        # Mock Path.read_text to raise OSError
        original_read_text = Path.read_text

        def mock_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
            if self == template:
                raise OSError("Permission denied")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", mock_read_text)

        with pytest.raises(PromptError) as exc_info:
            GenericPromptBuilder(template)

        log.debug("Template load error:\n%s", exc_info.value)

        assert "Error reading template file" in str(exc_info.value)
        assert "Permission denied" in str(exc_info.value)
        assert exc_info.value.operation == "load"
        assert exc_info.value.template_file == template

    # ====================
    # Placeholder Extraction Tests
    # ====================
    def test_extract_placeholders_single_template(
        self, single_template_file: Path
    ) -> None:
        """Test placeholder extraction from single-record template.

        Args:
            single_template_file: fixture providing a simple single-record template file.
        """
        builder = GenericPromptBuilder(single_template_file)
        assert builder.template is not None
        placeholders = builder._extract_placeholders(builder.template)  # pyright: ignore[reportPrivateUsage]

        log.debug("Extracted placeholders: %s", placeholders)
        assert placeholders == {"brevid", "text"}

    def test_extract_placeholders_batch_template(
        self, batch_template_file: Path
    ) -> None:
        """Test placeholder extraction from batch template.

        Args:
            batch_template_file: fixture providing a batch template file.
        """
        builder = GenericPromptBuilder(batch_template_file)
        assert builder.template is not None
        placeholders = builder._extract_placeholders(builder.template)  # pyright: ignore[reportPrivateUsage]

        log.debug("Extracted placeholders: %s", placeholders)
        assert placeholders == {"num_records", "batch_content"}

    def test_extract_placeholders_no_placeholders(self, tmp_path: Path) -> None:
        """Test placeholder extraction from template without placeholders.

        Args:
            tmp_path: temporary directory provided by pytest.
        """
        template = tmp_path / "no_placeholders.txt"
        template.write_text("This is a static template without any placeholders.")

        builder = GenericPromptBuilder(template)
        assert builder.template is not None
        placeholders = builder._extract_placeholders(builder.template)  # pyright: ignore[reportPrivateUsage]

        log.debug("Extracted placeholders: %s", placeholders)
        assert placeholders == set()

    # ====================
    # Field Validation Tests
    # ====================
    def test_require_template_fields_success(self, single_template_file: Path) -> None:
        """Test field validation passes when all required fields are present.

        Args:
            single_template_file: fixture providing a simple single-record template file.
        """
        builder = GenericPromptBuilder(single_template_file)
        assert builder.template is not None
        present = builder._extract_placeholders(builder.template)  # pyright: ignore[reportPrivateUsage]
        required = {"brevid", "text"}

        # Should not raise
        builder._require_template_fields(present, required, builder.template_file)  # pyright: ignore[reportPrivateUsage]

    def test_require_template_fields_missing(self, single_template_file: Path) -> None:
        """Test field validation raises error when required fields are missing.

        Args:
            single_template_file: fixture providing a simple single-record template file.
        """
        builder = GenericPromptBuilder(single_template_file)
        assert builder.template is not None
        present = builder._extract_placeholders(builder.template)  # pyright: ignore[reportPrivateUsage]
        required = {"brevid", "text", "missing_field"}

        with pytest.raises(PromptBuildError) as exc_info:
            builder._require_template_fields(present, required, builder.template_file)  # pyright: ignore[reportPrivateUsage]

        log.debug("Field validation error:\n%s", exc_info.value)
        assert "Template is missing required fields" in str(exc_info.value)
        assert "missing_field" in str(exc_info.value)
        assert exc_info.value.operation == "build"

    # ====================
    # Single-Record Prompt Building Tests
    # ====================
    def test_build_single_record_success(
        self,
        single_template_file: Path,
        sample_record: dict[str, Any],
    ) -> None:
        """Test building a single-record prompt with valid data.

        Args:
            single_template_file: fixture providing a simple single-record template file.
            sample_record: fixture providing a sample record with standard keys.
        """
        builder = GenericPromptBuilder(single_template_file)
        prompt = builder.build(sample_record)

        assert "001" in prompt
        assert "Ollum monnum þæim sæm þetta bref sea æder høyra sændir" in prompt

    def test_build_single_record_with_unicode(
        self,
        single_template_file: Path,
        sample_unicode_record: dict[str, Any],
    ) -> None:
        """Test building a single-record prompt with medieval Norwegian unicode characters.

        Args:
            single_template_file: fixture providing a simple single-record template file.
            sample_unicode_record: fixture providing a sample record with unicode characters.
        """
        builder = GenericPromptBuilder(single_template_file)
        prompt = builder.build(sample_unicode_record)

        assert "002" in prompt
        assert "Hákon konungr réð fyrir Nóregi með æ, ø, å, ð, þ" in prompt
        # Verify specific Norwegian characters are preserved
        assert "æ" in prompt
        assert "ø" in prompt
        assert "å" in prompt
        assert "ð" in prompt
        assert "þ" in prompt

    def test_build_single_record_missing_brevid(
        self,
        single_template_file: Path,
    ) -> None:
        """Test building single-record prompt raises error when brevid is missing.

        Args:
            single_template_file: fixture providing a simple single-record template file.
        """
        builder = GenericPromptBuilder(single_template_file)
        record = {self.SRC_KEY_TEXT: "Text without brevid"}

        with pytest.raises(
            ValueError, match=r"(?i)brevid must be a non-empty string"
        ) as exc_info:
            builder.build(record)

        log.debug("Build prompt error: %s", exc_info.value)

    def test_build_single_record_missing_text(
        self,
        single_template_file: Path,
    ) -> None:
        """Test building single-record prompt raises error when text is missing.

        Args:
            single_template_file: fixture providing a simple single-record template file.
        """
        builder = GenericPromptBuilder(single_template_file)
        record = {self.SRC_KEY_BREVID: "001"}

        with pytest.raises(
            ValueError, match=r"(?i)text must be a non-empty string"
        ) as exc_info:
            builder.build(record)

        log.debug("Build prompt error: %s", exc_info.value)

    def test_build_single_record_empty_brevid(
        self,
        single_template_file: Path,
    ) -> None:
        """Test building single-record prompt raises error when brevid is empty.

        Args:
            single_template_file: fixture providing a simple single-record template file.
        """
        builder = GenericPromptBuilder(single_template_file)
        record = {self.SRC_KEY_BREVID: "", self.SRC_KEY_TEXT: "Some text"}

        with pytest.raises(
            ValueError, match=r"(?i)brevid must be a non-empty string"
        ) as exc_info:
            builder.build(record)

        log.debug("Build prompt error: %s", exc_info.value)

    def test_build_single_record_whitespace_only_brevid(
        self,
        single_template_file: Path,
    ) -> None:
        """Test building single-record prompt raises error when brevid is whitespace only.

        Args:
            single_template_file: fixture providing a simple single-record template file.
        """
        builder = GenericPromptBuilder(single_template_file)
        record = {self.SRC_KEY_BREVID: "   \n\t  ", self.SRC_KEY_TEXT: "Valid text"}

        with pytest.raises(
            ValueError, match=r"(?i)brevid must be a non-empty string"
        ) as exc_info:
            builder.build(record)

        log.debug("Build prompt error: %s", exc_info.value)

    def test_build_single_record_empty_text(
        self,
        single_template_file: Path,
    ) -> None:
        """Test building single-record prompt raises error when text is empty.

        Args:
            single_template_file: fixture providing a simple single-record template file.
        """
        builder = GenericPromptBuilder(single_template_file)
        record = {self.SRC_KEY_BREVID: "001", self.SRC_KEY_TEXT: ""}

        with pytest.raises(
            ValueError, match=r"(?i)text must be a non-empty string"
        ) as exc_info:
            builder.build(record)

        log.debug("Build prompt error: %s", exc_info.value)

    def test_build_single_record_whitespace_only_text(
        self,
        single_template_file: Path,
    ) -> None:
        """Test building single-record prompt raises error when text is whitespace only.

        Args:
            single_template_file: fixture providing a simple single-record template file.
        """
        builder = GenericPromptBuilder(single_template_file)
        record = {self.SRC_KEY_BREVID: "001", self.SRC_KEY_TEXT: "   \n\t  "}
        with pytest.raises(
            ValueError, match=r"(?i)text must be a non-empty string"
        ) as exc_info:
            builder.build(record)

        log.debug("Build prompt error: %s", exc_info.value)

    # ====================
    # Batch Prompt Building Tests
    # ====================
    def test_build_batch_prompt_success(
        self,
        batch_template_file: Path,
        sample_batch_records: list[dict[str, Any]],
    ) -> None:
        """Test building a batch prompt with multiple valid records.

        Args:
            batch_template_file: fixture providing a batch template file.
            sample_batch_records: fixture providing a list of sample records for batch processing.
        """
        builder = GenericPromptBuilder(batch_template_file)
        prompt = builder.build(sample_batch_records)

        assert "Processing 3 records:" in prompt
        assert "RECORD 1:" in prompt
        assert "RECORD 2:" in prompt
        assert "RECORD 3:" in prompt
        assert "001" in prompt
        assert "Ollum monnum þæim sæm þetta bref sea æder høyra sændir" in prompt
        assert "002" in prompt
        assert "æ, ø, å" in prompt

    def test_build_batch_prompt_single_record(
        self,
        batch_template_file: Path,
        sample_record: dict[str, Any],
    ) -> None:
        """Test building a batch prompt with a single record.

        Args:
            batch_template_file: fixture providing a batch template file.
            sample_record: fixture providing a sample record with standard keys.
        """
        builder = GenericPromptBuilder(batch_template_file)
        prompt = builder.build([sample_record])

        log.debug("Batch prompt with single record: %s", prompt)

        assert "Processing 1 records:" in prompt
        assert "RECORD 1:" in prompt
        assert "001" in prompt
        assert "Ollum monnum þæim sæm þetta bref sea æder høyra sændir" in prompt

    def test_build_batch_prompt_empty_list(self, batch_template_file: Path) -> None:
        """Test building a batch prompt with an empty list raises error.

        Args:
            batch_template_file: fixture providing a batch template file.
        """
        builder = GenericPromptBuilder(batch_template_file)

        with pytest.raises(PromptBuildError) as exc_info:
            builder.build([])

        log.debug("Batch prompt build error: %s", exc_info.value)

        assert (
            "records list cannot be empty for batch processing"
            in str(exc_info.value).lower()
        )

    def test_build_batch_prompt_invalid_record(
        self,
        batch_template_file: Path,
    ) -> None:
        """Test building a batch prompt with an invalid record raises error.

        Args:
            batch_template_file: fixture providing a batch template file.
        """
        builder = GenericPromptBuilder(batch_template_file)
        records = [
            {self.SRC_KEY_BREVID: "001", self.SRC_KEY_TEXT: "Valid text"},
            {self.SRC_KEY_BREVID: "002"},  # Missing text
        ]

        with pytest.raises(PromptBuildError) as exc_info:
            builder.build(records)

        log.debug("Batch prompt build error: %s", exc_info.value)

        assert "text must be a non-empty string" in str(exc_info.value).lower()
        assert "validation failed" in str(exc_info.value).lower()

    def test_build_batch_prompt_invalid_record_first_position(
        self,
        batch_template_file: Path,
    ) -> None:
        """Test building batch with invalid record at first position."""
        builder = GenericPromptBuilder(batch_template_file)
        records = [
            {self.SRC_KEY_BREVID: ""},  # Invalid first
            {self.SRC_KEY_BREVID: "002", self.SRC_KEY_TEXT: "Valid"},
        ]

        with pytest.raises(
            PromptBuildError, match=r"(?i)validation failed"
        ) as exc_info:
            builder.build(records)

        log.debug("Batch prompt build error: %s", exc_info.value)

        assert "brevid must be a non-empty string" in str(exc_info.value).lower()
        assert "validation failed" in str(exc_info.value).lower()

    def test_build_batch_prompt_with_unicode(
        self,
        batch_template_file: Path,
        sample_unicode_record: dict[str, Any],
    ) -> None:
        """Test building a batch prompt with unicode characters.

        Args:
            batch_template_file: fixture providing a batch template file.
            sample_unicode_record: fixture providing a sample record with unicode characters.
        """
        builder = GenericPromptBuilder(batch_template_file)
        prompt = builder.build([sample_unicode_record])

        assert "Hákon konungr réð fyrir Nóregi með æ, ø, å, ð, þ" in prompt
        assert "æ" in prompt
        assert "ø" in prompt
        assert "å" in prompt
        assert "ð" in prompt
        assert "þ" in prompt

    # ====================
    # Record Validation Tests
    # ====================
    @pytest.mark.parametrize(
        ("brevid_test_val", "text_test_val"),
        [
            ("001", "Test text"),  # success case
            ("  001  ", "  Test text  "),  # strips leading/trailing whitespace
        ],
    )
    def test_validate_and_clean_record(
        self,
        single_template_file: Path,
        brevid_test_val: Any,
        text_test_val: Any,
    ) -> None:
        """Test record validation raises error when required fields are missing.

        Args:
            single_template_file: fixture providing a simple single-record template file.
            brevid_test_val: test value for brevid field.
            text_test_val: test value for text field.
        """
        builder = GenericPromptBuilder(single_template_file)
        record = {
            self.SRC_KEY_BREVID: brevid_test_val,
            self.SRC_KEY_TEXT: text_test_val,
        }
        cleaned = builder._validate_and_clean_record(record)  # pyright: ignore[reportPrivateUsage]

        log.debug("Cleaned record: %s", cleaned)

        assert cleaned == {
            "brevid": brevid_test_val.strip(),
            "text": text_test_val.strip(),
        }

    # ====================
    # Batch Content Formatting Tests
    # ====================
    def test_format_batch_content_multiple_records(
        self,
        batch_template_file: Path,
        sample_batch_records: list[dict[str, Any]],
    ) -> None:
        """Test batch content formatting with multiple records.

        Args:
            batch_template_file: fixture providing a batch template file.
            sample_batch_records: fixture providing a list of sample records for batch processing.
        """
        builder = GenericPromptBuilder(batch_template_file)

        # First validate and clean the records
        cleaned_records = [
            builder._validate_and_clean_record(record)  # pyright: ignore[reportPrivateUsage]
            for record in sample_batch_records
        ]

        batch_content = builder._format_batch_content(cleaned_records)  # pyright: ignore[reportPrivateUsage]

        assert "RECORD 1:" in batch_content
        assert "Brevid: 001" in batch_content
        assert (
            'Text: """Ollum monnum þæim sæm þetta bref sea æder høyra sændir'
            in batch_content
        )

        assert "RECORD 2:" in batch_content
        assert "Brevid: 002" in batch_content
        assert 'Text: """Second record with special: æ, ø, å, ð, þ"""' in batch_content

        assert "RECORD 3:" in batch_content
        assert "Brevid: 003" in batch_content
        assert (
            'Text: """Ollom monnom þeim sæm þetta bref sea ædher høyra sænda'
            in batch_content
        )

    def test_format_batch_content_preserves_unicode(
        self,
        batch_template_file: Path,
        sample_unicode_record: dict[str, Any],
    ) -> None:
        """Test batch content formatting preserves unicode characters.

        Args:
            batch_template_file: fixture providing a batch template file.
            sample_unicode_record: fixture providing a sample record with unicode characters.
        """
        builder = GenericPromptBuilder(batch_template_file)

        # First validate and clean the record
        cleaned_record = builder._validate_and_clean_record(sample_unicode_record)  # pyright: ignore[reportPrivateUsage]

        log.debug("Cleaned unicode record: %s", cleaned_record)

        batch_content = builder._format_batch_content([cleaned_record])  # pyright: ignore[reportPrivateUsage]

        log.debug("Formatted batch content with unicode:\n%s", batch_content)

        assert "Hákon konungr réð fyrir Nóregi með æ, ø, å, ð, þ" in batch_content

    # ====================
    # Error Message Tests
    # ====================
    def test_prompt_error_includes_template_file(self, tmp_path: Path) -> None:
        """Test PromptError includes template file in error message.

        Args:
            tmp_path: temporary directory provided by pytest.
        """
        template = tmp_path / "test_template.txt"
        error = PromptError("Test error", template_file=template, operation="test")

        assert str(template) in str(error)
        assert "Test error" in str(error)
        assert "operation: test" in str(error)

    def test_template_not_found_error_format(self, tmp_path: Path) -> None:
        """Test TemplateNotFoundError has correct format.

        Args:
            tmp_path: temporary directory provided by pytest.
        """
        template = tmp_path / "nonexistent.txt"
        error = TemplateNotFoundError(template)

        assert str(template) in str(error)
        assert "template file not found" in str(error).lower()
        assert error.operation == "load"

    def test_prompt_build_error_includes_data_type(self, tmp_path: Path) -> None:
        """Test PromptBuildError includes data type in error message.

        Args:
            tmp_path: temporary directory provided by pytest.
        """
        template = tmp_path / "test_template.txt"
        error = PromptBuildError(
            "Test build error",
            template_file=template,
            data_type="batch",
        )

        assert "Test build error" in str(error)
        assert "data_type: batch" in str(error)
        assert str(template) in str(error)
        assert error.operation == "build"

    # ====================
    # Edge Case and Error Coverage Tests
    # ====================
    def test_build_template_format_error_with_bad_placeholder(
        self, tmp_path: Path
    ) -> None:
        """Test handling of template formatting error with malformed format specification.

        Args:
            tmp_path: temporary directory provided by pytest.
        """
        # Create a template with invalid format specification that will raise ValueError
        template = tmp_path / "bad_template.txt"
        template.write_text("Brevid: {brevid:invalid_spec}\nText: {text}")

        builder = GenericPromptBuilder(template)
        record = {self.SRC_KEY_BREVID: "001", self.SRC_KEY_TEXT: "Test text"}

        with pytest.raises(PromptBuildError) as exc_info:
            builder.build(record)

        assert "template formatting failed" in str(exc_info.value).lower()
        assert exc_info.value.operation == "build"

    def test_batch_template_format_error_with_bad_placeholder(
        self, tmp_path: Path
    ) -> None:
        """Test handling of batch template formatting error.

        Args:
            tmp_path: temporary directory provided by pytest.
        """
        # Create a batch template with invalid format specification
        template = tmp_path / "bad_batch_template.txt"
        template.write_text(
            "Processing {num_records:invalid_spec} records:\n{batch_content}"
        )

        builder = GenericPromptBuilder(template)
        records = [{self.SRC_KEY_BREVID: "001", self.SRC_KEY_TEXT: "Test text"}]

        with pytest.raises(PromptBuildError) as exc_info:
            builder.build(records)

        assert "template formatting failed" in str(exc_info.value).lower()
        assert exc_info.value.operation == "build"
