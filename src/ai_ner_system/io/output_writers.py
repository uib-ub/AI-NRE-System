
"""Output writing operations for AI NER System."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Union, Any

from .exceptions import OutputError

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

    @staticmethod
    def _ensure_output_directory(file_path: Union[str, Path]) -> Path:
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
            raise OutputError(
                f"Failed to create output directory {directory}: {e}",
                file_path=str(file_path)
            ) from e

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

        except (OSError, UnicodeEncodeError) as e:
            raise OutputError(
                f'Failed to write annotated text output to {output_path}: {e}',
                file_path=str(output_path),
                output_type="annotation"
            ) from e

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

        except (OSError, UnicodeEncodeError) as e:
            raise OutputError(
                f'Failed to write metadata output to {output_path}: {e}',
                file_path=str(output_path),
                output_type="metadata"
            ) from e

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

        except (OSError, UnicodeEncodeError) as e:
            raise OutputError(
                f"Failed to append annotated text output to {output_path}: {e}",
                file_path=str(output_path),
                output_type="annotation"
            ) from e

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

        except (OSError, UnicodeEncodeError) as e:
            raise OutputError(
                f"Failed to append metadata output to {output_path}: {e}",
                file_path=str(output_path),
                output_type="metadata"
            ) from e


    @staticmethod
    def write_stats_output(file_path: str, stats_data: Dict[str, Any]) -> None:
        """Write processing statistics to a JSON file.

        Args:
            file_path: Output file path for the statistics.
            stats_data: Dictionary containing processing statistics.

        Raises:
            IOError: If writing to the file fails.
        """
        try:
            # Ensure output directory exists
            output_path = OutputWriter._ensure_output_directory(file_path)

            with open(output_path, "w", encoding="utf-8") as stats_file:
                json.dump(stats_data, stats_file, indent=2, ensure_ascii=False)

            logging.info(f'Processing statistics written to: {output_path}')

        # except Exception as e:
        #     logging.error(f"Error writing stats output to {file_path}: {e}", exc_info=True)
        #     raise e
        except (OSError, UnicodeEncodeError, TypeError) as e:
            logging.error(f"Error writing stats output to {file_path}: {e}", exc_info=True)
            raise OutputError(
                f"Failed to write stats output to {file_path}: {e}",
                file_path=str(file_path),
                output_type="stats"
            ) from e

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