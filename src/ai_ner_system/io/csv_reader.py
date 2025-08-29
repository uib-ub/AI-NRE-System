"""CSV reading operations for AI NER System."""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Iterator, Generator

from .exceptions import CSVError, CSVValidationError

class CSVReader:
    """CSV reader with validation and streaming capabilities.

    Provides methods for reading CSV files with proper error handling,
    validation, and memory-efficient streaming for large files.
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

        self._validate_file()
        logging.info(
            f'Initialized CSV reader for {self.file_path} '
            f'with delimiter {self.delimiter} and encoding {self.encoding}'
        )

    def _validate_file(self) -> None:
        """Validate that the CSV file exists and is readable.

        Raises:
            CSVError: If file validation fails.
        """
        if not self.file_path.exists():
            raise CSVError(
                f'CSV file does not exist: {self.file_path}',
                file_path=str(self.file_path)
            )

        if not self.file_path.is_file():
            raise CSVError(
                f'Path is not a file: {self.file_path}',
                file_path=str(self.file_path)
            )

        if self.file_path.stat().st_size == 0:
            raise CSVError(
                f'CSV file is empty: {self.file_path}',
                file_path=str(self.file_path)
            )

    def stream_records(self) -> Iterator[Dict[str, str]]: # Generator[Dict[str, str], None, None]:
        """Stream CSV records as dictionaries.

        Yields:
            Dictionary representing each CSV row with column headers as keys

        Raises:
            CSVError: If reading fails.
        """

        logging.info(f'Starting to stream records from: {self.file_path}')
        record_count = 0

        try:
            with open(self.file_path, 'r', encoding=self.encoding) as file:
                reader = csv.DictReader(file, delimiter=self.delimiter)

                # Validate that the CSV has headers
                if not reader.fieldnames:
                    raise IOError(f'CSV file does not have headers: {self.file_path}')

                for row_number, row in enumerate(reader, start=2): # Start at 2 (header is row 1)
                    try:
                        # Validate record completeness and skip empty rows
                        if self._is_empty_row(row):
                            logging.warning(f'Skipping empty row at line {row_number}')
                            continue

                        record_count += 1
                        yield row

                    except (CSVError, CSVValidationError):
                        # Re-raise CSV-specific exceptions
                        raise
                    except Exception as e:
                        raise CSVError(
                            f'Error processing row at line {row_number}: {e}',
                            file_path=str(self.file_path),
                            line_number=row_number
                        ) from e

                logging.info('Successfully streamed %d records from: %s', record_count, self.file_path)

        except (OSError, UnicodeDecodeError) as e:
            raise CSVError(
                f'Error reading CSV file {self.file_path}: {e}',
                file_path=str(self.file_path)
            ) from e

    @staticmethod
    def _is_empty_row(row: Dict[str, str]) -> bool:
        """Check if a row contains only empty values.

        Args:
            row: Dictionary representing a CSV row.

        Returns:
            True if all values in the row are empty or whitespace-only.
        """
        return all(not str(value).strip() for value in row.values())

    # TODO: this method is not in use, consider removing or implementing
    def get_headers(self) -> List[str]:
        """Get the column headers from the CSV file.

        Returns:
            List of column header names.

        Raises:
            IOError: If headers cannot be read.
        """
        try:
            with open(self.file_path, encoding=self.encoding) as file:
                reader = csv.DictReader(file, delimiter=self.delimiter)
                headers = list(reader.fieldnames or [])

                if not headers:
                    raise IOError(f'No headers found in CSV file: {self.file_path}')

                return headers

        except csv.Error as e:
            raise IOError(f'CSV parsing error reading headers from {self.file_path}: {e}') from e
        except OSError as e:
            raise IOError(f'Failed to read headers from {self.file_path}: {e}') from e
