"""CSV reader with validation and streaming capabilities for AI NER System.

This module is the streaming, validating input adapter that turns a CSV file into
cleaned record dictionaries for the rest of the pipeline.

"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .exceptions import CSVError, EncodingError, FileValidationError

if TYPE_CHECKING:
    from collections.abc import Iterator


class CSVReader:
    """CSV reader with validation and streaming capabilities.

    Provides methods for reading CSV files with proper error handling,
    validation, and memory-efficient streaming for large files.

    Attributes:
        file_path: Path to the CSV file.
        delimiter: Delimiter used in the CSV file.
        encoding: Encoding of the CSV file.
        required_headers: Optional set of headers that must be present in the CSV.
    """

    def __init__(
        self,
        file_path: str,
        *,
        delimiter: str = ";",
        encoding: str = "utf-8",
        required_headers: frozenset[str] | None = None,
    ) -> None:
        """Initialize the CSVReader with file path, delimiter, and encoding.

        Args:
            file_path: Path to the CSV file.
            delimiter: Delimiter used in the CSV file.
            encoding: Encoding of the CSV file.
            required_headers: Optional frozenset of headers that must be present in the CSV.
                If provided, will validate that all these headers exist.
                If None (default), no header validation is performed.

        Raises:
            FileValidationError: If file validation fails.
        """
        self.file_path = Path(file_path)
        self.delimiter = delimiter
        self.encoding = encoding
        self.required_headers = required_headers
        self._headers: list[str] | None = None

        self._validate_file()
        logging.info(
            "Initialized CSV reader for %s with delimiter %s and encoding %s",
            self.file_path,
            self.delimiter,
            self.encoding,
        )

    def _validate_file(self) -> None:
        """Validate that the CSV file exists and is readable.

        Raises:
            FileValidationError: If file validation fails.
        """
        if not self.file_path.exists():
            msg = f"CSV file does not exist: {self.file_path}"
            raise FileValidationError(
                msg,
                file_path=str(self.file_path),
                validation_type="existence",
            )

        if not self.file_path.is_file():
            msg = f"Path is not a file: {self.file_path}"
            raise FileValidationError(
                msg,
                file_path=str(self.file_path),
                validation_type="file_type",
            )

        if self.file_path.stat().st_size == 0:
            msg = f"CSV file is empty: {self.file_path}"
            raise FileValidationError(
                msg,
                file_path=str(self.file_path),
                validation_type="file_size",
            )

    def stream_records(self) -> Iterator[dict[str, str]]:
        """Stream CSV records as dictionaries.

        Yields:
            Dictionary representing each CSV row with column headers as keys

        Raises:
            CSVError: If reading fails.
        """
        logging.info("Starting to stream records from: %s", self.file_path)
        record_count = 0

        try:
            with self.file_path.open(encoding=self.encoding, newline="") as file:
                reader = csv.DictReader(
                    file, delimiter=self.delimiter, restval=""
                )  # make sure missing values are empty strings with restval=""

                # Store headers (csv.DictReader always uses first row as headers)
                self._headers = list(reader.fieldnames) if reader.fieldnames else []
                logging.debug("CSV headers detected: %s", self._headers)

                # Validate required headers if specified
                if self.required_headers:
                    self._validate_required_headers(self._headers)

                # Stream records with proper error handling
                # Start at 2 (header is row 1)
                for row_number, row in enumerate(reader, start=2):
                    try:
                        # skip empty rows but log them
                        if self._is_empty_row(row):
                            logging.warning(
                                "Skipping empty row at line %d",
                                row_number,
                            )
                            continue

                        # Clean row data
                        cleaned_row = self._clean_row(row)
                        logging.debug("Cleaned row %d: %s", row_number, cleaned_row)
                        record_count += 1
                        yield cleaned_row

                    except CSVError:
                        # Re-raise CSV-specific exceptions
                        raise
                    except Exception as e:
                        msg = f"Error processing row at line {row_number}: {e}"
                        raise CSVError(
                            msg,
                            file_path=str(self.file_path),
                            line_number=row_number,
                        ) from e

                logging.info(
                    "Successfully streamed %d records from: %s",
                    record_count,
                    self.file_path,
                )

        except UnicodeDecodeError as e:
            msg = f"Encoding error while reading CSV file: {e}"
            raise EncodingError(
                msg,
                file_path=str(self.file_path),
                encoding=self.encoding,
            ) from e
        except OSError as e:
            msg = f"Error reading CSV file {self.file_path}: {e}"
            raise CSVError(
                msg,
                file_path=str(self.file_path),
            ) from e
        except csv.Error as e:
            msg = f"CSV parsing error: {e}"
            raise CSVError(
                msg,
                file_path=str(self.file_path),
            ) from e

    def _clean_row(self, row: dict[str, str]) -> dict[str, str]:
        """Clean a CSV row.

        Args:
            row: Dictionary representing a CSV row.

        Returns:
            Cleaned row dictionary.
        """
        # Strip whitespace from all values and return cleaned row
        return {key: str(value).strip() if value else "" for key, value in row.items()}

    def _validate_required_headers(self, headers: list[str]) -> None:
        """Validate that CSV has all required headers.

        Args:
            headers: List of header names from the CSV file.

        Raises:
            CSVError: If required headers are missing.
        """
        if not self.required_headers:
            return

        headers_set = set(headers)
        missing_headers = self.required_headers - headers_set

        if missing_headers:
            msg = (
                f"CSV file is missing required headers: {sorted(missing_headers)}. "
                f"Required: {sorted(self.required_headers)}. "
                f"Found: {sorted(headers_set)}"
            )
            raise CSVError(
                msg,
                file_path=str(self.file_path),
                line_number=1,
            )

    @staticmethod
    def _is_empty_row(row: dict[str, str]) -> bool:
        """Check if a row contains only empty values.

        Args:
            row: Dictionary representing a CSV row.

        Returns:
            True if all values in the row are empty or whitespace-only.
        """
        return all(not str(value).strip() for value in row.values())
