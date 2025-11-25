"""Unit tests for CSVReader class.

Tests cover:
- CSV file validation
- Header validation with required_headers parameter
- Record streaming functionality
- Error handling for malformed CSV files
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from ai_ner_system.file_io.csv_reader import CSVReader
from ai_ner_system.file_io.exceptions import CSVError, FileValidationError

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


class TestCSVReader:
    """Unit tests for CSVReader."""

    def test_init_valid_file_success(
        self,
        tmp_input_file: Path,
    ) -> None:
        """Test initializing CSVReader with a valid CSV file."""
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
        """Test initializing CSVReader with a path that is not a file."""
        dir_path: Path = tmp_path / "a_directory"
        dir_path.mkdir()

        with pytest.raises(FileValidationError) as exc_info:
            CSVReader(file_path=str(dir_path))

        assert "is not a file" in str(exc_info.value)

    def test_init_file_empty(self, tmp_path: Path) -> None:
        """Test initializing CSVReader with an empty file."""
        empty_file: Path = tmp_path / "empty.csv"
        empty_file.touch()

        with pytest.raises(FileValidationError) as exc_info:
            CSVReader(file_path=str(empty_file))

        log.debug("Exception message: %s", exc_info.value)
        assert "CSV file is empty" in str(exc_info.value)

    def test_stream_records_with_generic_headers(self, tmp_path: Path) -> None:
        """Test streaming records from a CSV file with generic headers.

        When required_headers is None, CSVReader should accept any headers
        and successfully read the records.
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
        """Test that CSVReader raises error when required headers are missing."""
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
        """Test streaming records from a CSV with expected headers."""
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
