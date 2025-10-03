"""CSV reader with validation and streaming capabilities for AI NER System."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterator

from .exceptions import CSVError, EncodingError, FileValidationError


class CSVReader:
    """CSV reader with validation and streaming capabilities.

    Provides methods for reading CSV files with proper error handling,
    validation, and memory-efficient streaming for large files.

    Attributes:
        file_path: Path to the CSV file.
        delimiter: Delimiter used in the CSV file.
        encoding: Encoding of the CSV file.
    """

    def __init__(
        self,
        file_path: str,
        delimiter: str = ';',
        encoding: str = 'utf-8'
    ) -> None:
        """Initialize the CSVReader with file path, delimiter, and encoding.

        Args:
            file_path: Path to the CSV file.
            delimiter: Delimiter used in the CSV file.
            encoding: Encoding of the CSV file.

        Raises:
            CSVError: If file validation fails.
        """
        self.file_path = Path(file_path)
        self.delimiter = delimiter
        self.encoding = encoding
        self._headers: list[str] | None = None

        self._validate_file()
        logging.info(
            f'Initialized CSV reader for {self.file_path} '
            f'with delimiter {self.delimiter} and encoding {self.encoding}'
        )

    def _validate_file(self) -> None:
        """Validate that the CSV file exists and is readable.

        Raises:
            FileValidationError: If file validation fails.
        """
        if not self.file_path.exists():
            raise FileValidationError(
                f'CSV file does not exist: {self.file_path}',
                file_path=str(self.file_path),
                validation_type='existence'
            )

        if not self.file_path.is_file():
            raise FileValidationError(
                f'Path is not a file: {self.file_path}',
                file_path=str(self.file_path),
                validation_type='file_type'
            )

        if self.file_path.stat().st_size == 0:
            raise FileValidationError(
                f'CSV file is empty: {self.file_path}',
                file_path=str(self.file_path),
                validation_type='file_size'
            )

    def stream_records(self) -> Iterator[dict[str, str]]:
        """Stream CSV records as dictionaries.

        Yields:
            Dictionary representing each CSV row with column headers as keys

        Raises:
            CSVError: If reading fails.
        """
        logging.info(f'Starting to stream records from: {self.file_path}')
        record_count = 0

        try:
            with open(self.file_path, 'r', encoding=self.encoding, newline='') as file:
                reader = csv.DictReader(file, delimiter=self.delimiter)

                # Validate that the CSV has headers
                if not reader.fieldnames:
                    raise CSVError(
                        f'CSV file does not have headers: {self.file_path}',
                        file_path=str(self.file_path),
                        line_number=1
                    )

                self._headers = list(reader.fieldnames)
                logging.debug(f'CSV headers detected: {self._headers}')

                # Stream records with proper error handling
                # Start at 2 (header is row 1)
                for row_number, row in enumerate(reader, start=2):
                    try:
                        # skip empty rows but log them
                        if self._is_empty_row(row):
                            logging.warning(
                                f'Skipping empty row at line {row_number}')
                            continue

                        # Validate row data
                        validated_row = self._validate_row(row, row_number)
                        record_count += 1
                        yield validated_row

                    except CSVError:
                        # Re-raise CSV-specific exceptions
                        raise
                    except Exception as e:
                        raise CSVError(
                            f'Error processing row at line {row_number}: {e}',
                            file_path=str(self.file_path),
                            line_number=row_number
                        ) from e

                logging.info(f'Successfully streamed {record_count} records from: {self.file_path}')

        except UnicodeDecodeError as e:
            raise EncodingError(
                f'Encoding error while reading CSV file: {e}',
                file_path=str(self.file_path),
                encoding=self.encoding
            ) from e
        except OSError as e:
            raise CSVError(
                f'Error reading CSV file {self.file_path}: {e}',
                file_path=str(self.file_path)
            ) from e
        except csv.Error as e:
            raise CSVError(
                f'CSV parsing error: {e}',
                file_path=str(self.file_path)
            ) from e

    def _validate_row(self, row: dict[str, str], row_number: int) -> dict[str, str]:
        """Validate and clean a CSV row.

        Args:
            row: Dictionary representing a CSV row.
            row_number: Line number of the row.

        Returns:
            Validated and cleaned row dictionary.

        Raises:
            CSVError: If row validation fails.
        """
        # Check for completely empty row (handled separately)
        if self._is_empty_row(row):
            raise CSVError(
                f'Empty row encountered at line {row_number}',
                file_path=str(self.file_path),
                line_number=row_number
            )

        # Strip whitespace from all values
        cleaned_row = {key: str(value).strip(
        ) if value else '' for key, value in row.items()}

        return cleaned_row

    @staticmethod
    def _is_empty_row(row: dict[str, str]) -> bool:
        """Check if a row contains only empty values.

        Args:
            row: Dictionary representing a CSV row.

        Returns:
            True if all values in the row are empty or whitespace-only.
        """
        return all(not str(value).strip() for value in row.values())
