"""Unit tests for CSVReader class.

Tests cover:
- CSV file validation
- Header validation with required_headers parameter
- Record streaming functionality
- Error handling for malformed CSV files
"""

from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING

import pytest

from ai_ner_system.file_io.csv_reader import CSVReader
from ai_ner_system.file_io.exceptions import (
    CSVError,
    EncodingError,
    FileValidationError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


log = logging.getLogger(__name__)


class TestCSVReader:
    """Unit tests for CSVReader."""

    def test_init_valid_file_success(
        self,
        tmp_input_file: Path,
    ) -> None:
        """Test initializing CSVReader with a valid CSV file.

        Args:
            tmp_input_file: A temporary CSV file path fixture defined in conftest.py.
        """
        csv_file: Path = tmp_input_file

        reader = CSVReader(file_path=str(csv_file))
        assert reader.file_path == csv_file
        assert reader.delimiter == ";"
        assert reader.encoding == "utf-8"
        assert reader.required_headers is None

    def test_init_file_not_exist(self) -> None:
        """Test initializing CSVReader with a non-existent file."""
        non_existent_file = "/nonexistent/path/to/non_existent_file.csv"

        with pytest.raises(FileValidationError) as exc_info:
            CSVReader(file_path=non_existent_file)

        assert "CSV file does not exist" in str(exc_info.value)

    def test_init_file_not_file(self, tmp_path: Path) -> None:
        """Test initializing CSVReader with a path that is not a file.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        dir_path: Path = tmp_path / "a_directory"
        dir_path.mkdir()

        with pytest.raises(FileValidationError) as exc_info:
            CSVReader(file_path=str(dir_path))

        assert "is not a file" in str(exc_info.value)

    def test_init_file_empty(self, tmp_path: Path) -> None:
        """Test initializing CSVReader with an empty file.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        empty_file: Path = tmp_path / "empty.csv"
        empty_file.touch()

        with pytest.raises(FileValidationError) as exc_info:
            CSVReader(file_path=str(empty_file))

        log.debug("Exception message: %s", exc_info.value)
        assert "CSV file is empty" in str(exc_info.value)

    def test_init_with_custom_delimiter_and_encoding(self, tmp_path: Path) -> None:
        """Test CSVReader with custom delimiter and encoding.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        csv_content = "col1,col2,col3\nvalue1,value2,value3\n"
        csv_file = tmp_path / "comma_delimited.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        reader = CSVReader(str(csv_file), delimiter=",", encoding="utf-8")
        records = list(reader.stream_records())

        assert len(records) == 1
        assert records[0]["col1"] == "value1"

    def test_stream_records_with_generic_headers(self, tmp_path: Path) -> None:
        """Test streaming records from a CSV file with generic headers.

        When required_headers is None, CSVReader should accept any headers
        and successfully read the records.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        csv_content = "col1;col2;col3\nvalue1;value2;value3\nvalue4;value5;value6\n"
        csv_file: Path = tmp_path / "generic_headers.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        # No required headers - should accept any CSV
        reader = CSVReader(file_path=str(csv_file))

        # Should successfully read records even with different header names
        records = list(reader.stream_records())
        assert len(records) == 2
        assert records[0]["col1"] == "value1"
        assert records[0]["col2"] == "value2"
        assert records[0]["col3"] == "value3"

    def test_stream_records_missing_required_headers(self, tmp_path: Path) -> None:
        """Test that CSVReader raises error when required headers are missing.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        # CSV with wrong headers
        csv_content = "col1;col2;col3\nvalue1;value2;value3\n"
        csv_file: Path = tmp_path / "wrong_headers.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        # Specify required headers
        required_headers = frozenset({"Bindnr", "Brevid", "Tekst"})
        reader = CSVReader(file_path=str(csv_file), required_headers=required_headers)

        # Should raise CSVError when trying to stream records
        with pytest.raises(CSVError) as exc_info:
            list(reader.stream_records())

        log.debug("Exception message: %s", exc_info.value)

        assert "missing required headers" in str(exc_info.value)
        assert "Bindnr" in str(exc_info.value)
        assert "Brevid" in str(exc_info.value)
        assert "Tekst" in str(exc_info.value)

    def test_stream_records_with_expected_headers(self, tmp_input_file: Path) -> None:
        """Test streaming records from a CSV with expected headers.

        Args:
            tmp_input_file: A temporary CSV file path fixture defined in conftest.py.
        """
        required_headers = frozenset({"Bindnr", "Brevid", "Tekst"})
        reader = CSVReader(
            file_path=str(tmp_input_file), required_headers=required_headers
        )

        records = list(reader.stream_records())

        # Should successfully read all records
        assert len(records) == 3
        assert all("Bindnr" in record for record in records)
        assert all("Brevid" in record for record in records)
        assert all("Tekst" in record for record in records)

    def test_stream_records_with_extra_headers(self, tmp_path: Path) -> None:
        """Test that CSVReader accepts files with more headers than required.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        csv_content = "Bindnr;Brevid;Tekst;Extra1;Extra2\n1;001;Text;foo;bar\n"
        csv_file = tmp_path / "extra_headers.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        # Only require subset of headers
        required_headers = frozenset({"Bindnr", "Tekst"})
        reader = CSVReader(str(csv_file), required_headers=required_headers)

        records = list(reader.stream_records())
        assert len(records) == 1
        assert records[0]["Bindnr"] == "1"
        assert records[0]["Extra1"] == "foo"  # Extra columns still accessible
        assert records[0]["Extra2"] == "bar"  # All extra columns accessible

    def test_stream_records_with_more_values_than_headers(self, tmp_path: Path) -> None:
        """Test that CSVReader handles rows with more values than headers.

        When a row has more values than headers, csv.DictReader assigns
        extra values to None key. This is expected behavior but worth testing
        as it's a common data quality issue.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        csv_content = "Bindnr;Brevid;Tekst\n1;001;Text;extra_value\n"
        csv_file = tmp_path / "extra_values.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        reader = CSVReader(str(csv_file))
        records = list(reader.stream_records())

        assert len(records) == 1
        assert records[0]["Bindnr"] == "1"
        assert records[0]["Brevid"] == "001"
        assert records[0]["Tekst"] == "Text"
        # Extra value is assigned to None key by csv.DictReader
        assert None in records[0]
        assert records[0].get(None) == "['extra_value']"  # type: ignore[call-overload]

    @pytest.mark.parametrize(
        ("csv_content"),
        [
            "Bindnr;Brevid;Tekst\n\t\n1;001;Some text\n",
            "Bindnr;Brevid;Tekst\n;;\n1;001;Some text\n",
        ],
    )
    def test_stream_records_skip_empty_row(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        csv_content: str,
    ) -> None:
        """Test that CSVReader skips empty rows with a warning.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            caplog: Pytest fixture to capture log output.
            csv_content: CSV content string with an empty row.
        """
        csv_file: Path = tmp_path / "empty_line.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        reader = CSVReader(file_path=str(csv_file))  # with restval="" in your CSVReader

        with caplog.at_level(logging.WARNING):
            rows = list(reader.stream_records())

        for record in rows:
            log.debug("Record: %s", record)

        assert len(rows) == 1
        assert rows == [{"Bindnr": "1", "Brevid": "001", "Tekst": "Some text"}]
        # confirm a warning was logged about skipping the empty row
        assert any(
            "Skipping empty row at line 2" in rec.message for rec in caplog.records
        )

    def test_stream_records_skip_multiple_empty_rows(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that CSVReader skips multiple consecutive empty rows.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            caplog: Pytest fixture to capture log output.
        """
        csv_content = "Bindnr;Brevid;Tekst\n;;\n\t\n  \n1;001;Text\n"
        csv_file = tmp_path / "multiple_empty.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        reader = CSVReader(str(csv_file))

        with caplog.at_level(logging.WARNING):
            records = list(reader.stream_records())

        assert len(records) == 1
        # Should have 3 warnings for lines 2, 3, 4
        warnings = [r for r in caplog.records if "Skipping empty row" in r.message]
        assert len(warnings) == 3

    def test_stream_records_handles_validation_error(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test that validation errors are wrapped in CSVError with context.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            mocker: Pytest fixture for mocking.
        """
        csv_content = "Bindnr;Brevid;Tekst\n1;001;Some text\n2;002;Another text\n"
        csv_file: Path = tmp_path / "validation_error.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        def _fake_clean_row(row: dict[str, str]) -> dict[str, str]:
            if row["Bindnr"] == "2":
                raise ValueError("Simulated validation error")
            return {
                key: str(value).strip() if value else "" for key, value in row.items()
            }

        reader = CSVReader(file_path=str(csv_file))

        # Mock _clean_row to raise an exception
        mocker.patch.object(
            reader,
            "_clean_row",
            side_effect=_fake_clean_row,
        )

        with pytest.raises(CSVError) as exc_info:
            list(reader.stream_records())

        log.debug("Exception message: %s", exc_info.value)
        assert "Error processing row at line 3" in str(exc_info.value)
        assert "Simulated validation error" in str(exc_info.value)

    def test_stream_records_handles_unicode_decode_error(self, tmp_path: Path) -> None:
        """Test that UnicodeDecodeError is wrapped in EncodingError.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        csv_file: Path = tmp_path / "invalid_encoding.csv"
        # Create a file with latin-1 encoding
        csv_file.write_bytes("Bindnr;Brevid;Tekst\n1;001;Åæø\n".encode("latin-1"))

        # Try to read with utf-8
        reader = CSVReader(str(csv_file), encoding="utf-8")

        with pytest.raises(EncodingError) as exc_info:
            list(reader.stream_records())

        log.debug("Exception message: %s", exc_info.value)
        assert "Encoding error while reading CSV file" in str(exc_info.value)
        assert "utf-8" in str(exc_info.value)

    def test_stream_records_handles_os_error(self, tmp_path: Path) -> None:
        """Test that OSError during file read is wrapped in CSVError.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        csv_file: Path = tmp_path / "permission_denied.csv"
        csv_file.write_text("Bindnr;Brevid;Tekst\n1;001;Some text\n", encoding="utf-8")

        # Permission denied
        csv_file.chmod(0o000)  # Remove all permissions
        reader = CSVReader(str(csv_file), encoding="utf-8")

        try:
            with pytest.raises(CSVError) as exc_info:
                list(reader.stream_records())

            log.debug("Exception message: %s", exc_info.value)
            assert "Error reading CSV file" in str(exc_info.value)
        finally:
            # Restore permissions for cleanup
            csv_file.chmod(0o755)

    def test_stream_records_handles_csv_error(self, tmp_path: Path) -> None:
        """Test that csv.Error during parsing is wrapped in CSVError.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        csv_file: Path = tmp_path / "field_too_large.csv"
        # Create CSV with a field that exceeds the field size limit
        large_field = "x" * 200000  # 200KB field
        csv_file.write_text(
            f"Bindnr;Brevid;Tekst\n1;001;{large_field}\n", encoding="utf-8"
        )

        # Set a small field size limit to trigger csv.Error
        original_limit = csv.field_size_limit()
        try:
            csv.field_size_limit(1000)  # Set limit to 1KB
            reader = CSVReader(str(csv_file), encoding="utf-8")

            with pytest.raises(CSVError) as exc_info:
                list(reader.stream_records())

            log.debug("Exception message: %s", exc_info.value)
            assert "CSV parsing error" in str(exc_info.value)
        finally:
            # Restore original field size limit
            csv.field_size_limit(original_limit)

    def test_stream_records_trims_whitespace(self, tmp_path: Path) -> None:
        """Test that CSVReader trims whitespace from values.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        csv_content = "Bindnr;Brevid;Tekst\n  1  ;  001  ;  Some text  \n"
        csv_file = tmp_path / "whitespace.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        reader = CSVReader(str(csv_file))
        records = list(reader.stream_records())

        assert records[0]["Bindnr"] == "1"  # trimmed
        assert records[0]["Brevid"] == "001"  # trimmed
        assert records[0]["Tekst"] == "Some text"  # trimmed
