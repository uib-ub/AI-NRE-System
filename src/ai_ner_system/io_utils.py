"""Input/Output utilities for LLL processing medieval texts.

This module provides functions to read CSV files, stream records,
and write annotated text and metadata outputs.
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Generator, Union, Any


class IOError(Exception):
    """Custom exception for I/O errors."""

class CSVReader:
    """CSV file reader with streaming capabilities."""

    def __init__(self, file_path: str, delimiter: str = ';', encoding: str = 'utf-8') -> None:
        """Initialize the CSVReader with file path, delimiter, and encoding.

        Args:
            file_path: Path to the CSV file.
            delimiter: Delimiter used in the CSV file.
            encoding: Encoding of the CSV file.

        Raises:
            IOError: If file path is invalid or file does not exist.
        """
        self.file_path = Path(file_path)
        self.delimiter = delimiter
        self.encoding = encoding
        self._validate_file()

    def _validate_file(self) -> None:
        """Validate that the CSV file exists and is readable.

        Raises:
            IOError: If file validation fails.
        """
        if not self.file_path.exists():
            raise IOError(f'CSV file does not exist: {self.file_path}')

        if not self.file_path.is_file():
            raise IOError(f'Path is not a file: {self.file_path}')

        if self.file_path.stat().st_size == 0:
            raise IOError(f'CSV file is empty: {self.file_path}')

    def stream_records(self) -> Generator[Dict[str, str], None, None]:
        """Stream records from the CSV file.

        Yields:
            Dictionary representing each CSV row with column headers as keys

        Raises:
            IOError: If file cannot be read or CSV parsing fails.
        """
        try:
            logging.info('Starting to stream records from: %s', self.file_path)

            with open(self.file_path, encoding=self.encoding) as file:
                reader = csv.DictReader(file, delimiter=self.delimiter)

                # Validate that the CSV has headers
                if not reader.fieldnames:
                    raise IOError(f'CSV file does not have headers: {self.file_path}')

                record_count = 0
                for row_number, record in enumerate(reader, start=2): # Start at 2 (header is row 1)
                    # Validate record completeness
                    if not any(record.values()):
                        logging.warning('Empty record found at row %d', row_number)
                        continue

                    record_count += 1
                    yield record

                logging.info('Finished streaming %d records from: %s', record_count, self.file_path)

        except csv.Error as e:
            raise IOError(f'Error parsing CSV file {self.file_path}: {e}') from e
        except OSError as e:
            raise IOError(f'Error reading CSV file {self.file_path}: {e}') from e
        except Exception as e:
            raise IOError(f'Unexpected error reading CSV file {self.file_path}: {e}') from e

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


class OutputWriter:
    """Output file writer for annotated text and metadata."""

    def __init__(self, encoding: str = 'utf-8') -> None:
        """Initialize the OutputWriter.

        Args:
            encoding: Encoding for output files.
        """
        self.encoding = encoding
        # Track which files have headers written for incremental writing output
        self._headers_written = {
            'text': False,
            'metadata': False
        }

    def _ensure_output_directory(self, file_path: Union[str, Path]) -> Path:
        """Ensure the output directory exists.

        Args:
            file_path: Path to the output file.

        Returns:
            Path object of the output file.

        Raises:
            IOError: If the output directory cannot be created.
        """
        path = Path(file_path)
        directory = path.parent

        try:
            directory.mkdir(parents=True, exist_ok=True)
            return path
        except OSError as e:
            raise IOError(f'Failed to create output directory {directory}: {e}') from e

    def write_text_output(
            self,
            file_path: str,
            header: str,
            annotations: List[str]
    ) -> None:
        """Write annotated text output to a file.

        Args:
            file_path: Output file path.
            header: Header line for the file.
            annotations: List of annotated text records (strings).

        Raises:
            IOError: If writing to the file fails.
            ValueError: If annotations list is empty.
        """

        if not annotations:
            raise ValueError('Annotations list cannot empty.')

        # Ensure output directory exists
        output_path = self._ensure_output_directory(file_path)

        try:
            logging.info('Writing annotated text output to %s', output_path)

            with open(output_path, "w", encoding=self.encoding) as file:
                file.write(header)
                if header and not header.endswith("\n"):
                    file.write("\n")

                content = "\n".join(annotations)
                file.write(content)

            logging.info('Annotated text output written to %s successfully', output_path)

        except OSError as e:
            raise IOError(f'Error writing annotated text output to {output_path}: {e}') from e

    def write_metadata_output(
            self,
            file_path: str,
            header: str,
            metadata: List[str]
    ) -> None:
        """Write metadata table output to a file.

        Args:
            file_path: Output file path.
            header: Header line for the file.
            metadata: List of metadata rows (strings).

        Raises:
            IOError: If writing to the file fails.
            ValueError: If metadata list is empty.
        """

        if not metadata:
            raise ValueError('Metadata list cannot be empty.')

        # Ensure output directory exists
        output_path = self._ensure_output_directory(file_path)

        try:
            logging.info('Writing metadata output to %s', output_path)

            with open(output_path, "w", encoding=self.encoding) as file:
                file.write(header)
                if header and not header.endswith("\n"):
                    file.write("\n")

                content = "\n".join(metadata)
                file.write(content)

            logging.info('Metadata output written to %s successfully', output_path)

        except OSError as e:
            raise IOError(f'Error writing metadata output to {output_path}: {e}') from e

    def append_text_output(
        self,
        file_path: str,
        header: str,
        annotations: List[str]
    ) -> None:
        """Append annotated text output to a file.

         Args:
             file_path: Output file path.
             header: Header line for the file (written only once).
             annotations: List of annotated text records (strings).

         Raises:
             IOError: If writing to the file fails.
             ValueError: If annotations list is empty.
         """
        if not annotations:
            raise ValueError('Annotations list cannot be empty.')

        # Ensure output directory exists
        output_path = self._ensure_output_directory(file_path)

        try:
            # Check if the file exists to determine if we need to write the header
            file_exists = output_path.exists()
            needs_header = not file_exists and not self._headers_written['text']

            mode = 'a' if file_exists else 'w'

            with open(output_path, mode, encoding=self.encoding) as file:
                # Write header only if the file is new or header has not been written yet
                if needs_header:
                    file.write(header)
                    if header and not header.endswith('\n'):
                        file.write('\n')
                    self._headers_written['text'] = True

                # Add newline before content if file already has content to avoid sticking
                # new annotations directly after old content without spacing.
                if file_exists and output_path.stat().st_size > 0:
                    file.write('\n')

                # Write the new annotations
                content = "\n".join(annotations)
                file.write(content)

            logging.info('Appended %d annotations to %s', len(annotations), output_path)

        except OSError as e:
            raise IOError(f'Error appending text output to {output_path}: {e}') from e

    def append_metadata_output(
        self,
        file_path: str,
        header: str,
        metadata: List[str]
    ) -> None:
        """Append metadata table output to a file.

         Args:
             file_path: Output file path.
             header: Header line for the file (written only once).
             metadata: List of metadata rows (strings).

         Raises:
             IOError: If writing to the file fails.
             ValueError: If metadata list is empty.
         """
        if not metadata:
            raise ValueError('Metadata list cannot be empty.')

        # Ensure output directory exists
        output_path = self._ensure_output_directory(file_path)

        try:
            # Check if file exists and if header needs to be written
            file_exists = output_path.exists()
            needs_header = not file_exists and not self._headers_written['metadata']

            mode = 'a' if file_exists else 'w'

            with open(output_path, mode, encoding=self.encoding) as file:
                # Write header only if the file is new or header has not been written yet
                if needs_header:
                    file.write(header)
                    if header and not header.endswith('\n'):
                        file.write('\n')
                    self._headers_written['metadata'] = True

                # Add newline before content if file already has content to avoid sticking
                # new annotations directly after old content without spacing.
                if file_exists and output_path.stat().st_size > 0:
                    file.write('\n')

                # Write the new metadata
                content = "\n".join(metadata)
                file.write(content)

            logging.info('Appended %d metadata rows to %s', len(metadata), output_path)

        except OSError as e:
            raise IOError(f'Error appending metadata output to {output_path}: {e}') from e

    def write_stats_output(self, file_path: str, stats_data: Dict[str, Any]) -> None:
        """Write processing statistics to a JSON file.

        Args:
            file_path: Output file path for the statistics.
            stats_data: Dictionary containing processing statistics.

        Raises:
            IOError: If writing to the file fails.
        """
        try:
            with open(file_path, "w", encoding="utf-8") as stats_file:
                json.dump(stats_data, stats_file, indent=2, ensure_ascii=False)

            logging.info(f'Processing statistics written to: {file_path}')
        except Exception as e:
            logging.error(f"Error writing stats output to {file_path}: {e}", exc_info=True)
            raise e

    def clean_output_files(self, *file_paths: str) -> None:
        """Clean up (delete) existing output files.

        Args:
            *file_paths: Variable number of file paths to clean up.

        Raises:
            IOError: If file deletion fails for any critical reason.
        """
        for file_path in file_paths:
            if not file_path: # Skip empty/None file paths
                continue

            try:
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    logging.info(f'Cleaned up existing output file: {file_path}')
                else:
                    logging.warning(f'Output file does not exist, skipping cleanup: {file_path}')

            except OSError as e:
                # Log error but don't fail the entire process for file cleanup issues
                logging.warning('Failed to clean up output file %s: %s', file_path, e)

        # Reset header tracking since we're starting fresh
        self._headers_written = {
            'text': False,
            'metadata': False
        }

        logging.info('Output file cleanup completed')