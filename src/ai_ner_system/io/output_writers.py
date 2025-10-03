"""Output writing operations for AI NER System.

This module provides utilities to write text-based outputs produced by the
pipeline:

* Atomic full-file writes for text and metadata (tempfile + os.replace).
* Flock-locked appends that serialize concurrent writers (POSIX only).
* JSON stats writing via atomic replace.

Concurrency:
* Appends use an exclusive `flock()` on the target file to avoid interleaving.
* When appending, a header is emitted only if the file is empty at lock time.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, ClassVar, cast

import fcntl  # POSIX-only lock

from .exceptions import OutputError

Pathish = str | Path  # Type alias for path-like objects


class OutputWriter:
    """Output file writer for annotated text, metadata, and JSON stats.

    Note: This implementation uses POSIX file locking (fcntl) and is not
    compatible with Windows. For cross-platform support, consider using
    portalocker or similar libraries.

    Notes:
        * `write_*` methods are atomic (tempfile + os.replace).
        * `append_*` methods take an exclusive `flock()` on the target file to
          serialize concurrent appenders across processes.
        * Header emission on append is determined solely by "is the file empty?"
          (robust to file rotation/truncation).
    """

    # Class constants for default settings
    DEFAULT_ENCODING: ClassVar[str] = 'utf-8'
    NEWLINE: ClassVar[str] = '\n'

    def __init__(self, encoding: str = DEFAULT_ENCODING) -> None:
        """Initialize the OutputWriter.

        Args:
            encoding: Text encoding used for all writes.
        """
        self.encoding = encoding
        logging.debug(f'OutputWriter initialized with encoding: {self.encoding}')

    @staticmethod
    def _ensure_output_directory(file_path: Pathish) -> Path:
        """Ensure the output directory exists.

        Args:
            file_path: Path to the output file.

        Returns:
            Path object of the output file.

        Raises:
            OutputError: If the output directory cannot be created.
        """
        path = Path(file_path)
        directory = path.parent
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logging.debug(f'Ensured output directory exists: {directory}')
            return path
        except OSError as e:
            raise OutputError(
                f"Failed to create output directory {directory}: {e}",
                file_path=str(file_path),
            ) from e

    @staticmethod
    def _atomic_write(file_path: Path, content: str, encoding: str) -> None:
        """Atomically write content to a file (POSIX).

        Args:
            file_path: Path to the output file.
            content: Content to write to the file.
            encoding: Text encoding used for writing.

        Raises:
            OutputError: If the atomic write fails.
        """
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w',
                    delete=False,
                    dir=file_path.parent,
                    encoding=encoding,
                    newline='',
                    suffix='.tmp'
            ) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)
            # Atomically replace the target file
            temp_path.replace(file_path)
            logging.debug('Atomic write completed for: %s', file_path)
        except(OSError, UnicodeEncodeError) as e:
            # Clean up temp file if it exists
            if 'temp_path' in locals() and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass  # Ignore cleanup errors
            raise OutputError(
                f'Atomic write failed for {file_path}: {e}',
                file_path=str(file_path),
                output_type="atomic_write"
            ) from e

    @staticmethod
    def _build_content(header: str, data: list[str]) -> str:
        """Build full-file content with header and lines.

        Args:
            header: Header line for the file.
            data: List of data lines.

        Returns:
            Text with normalized newlines and a trailing newline if non-empty.
        """
        content_parts: list[str] = []
        # Add header if provided
        if header:
            content_parts.append(header.rstrip('\r\n'))
        # Add lines
        content_parts.extend(line.rstrip('\r\n') for line in data)
        # Join with newlines and ensure final newline
        content = OutputWriter.NEWLINE.join(content_parts)
        if content and not content.endswith(OutputWriter.NEWLINE):
            content += OutputWriter.NEWLINE
        return content

    def _write_lines(
        self,
        *,
        file_path: Pathish,
        header: str,
        lines: list[str],
        log_label: str,
        output_type: str
    ) -> None:
        """Writes 'lines' to 'file_path' atomically (replacing the file).

        Args:
          file_path: Output file path.
          header: Header line.
          lines: Lines to write (one record per element).
          log_label: Human-readable label for logs ('annotations'/'metadata').
          output_type: Error tag.

        Raises:
          ValueError: If 'lines' is empty.
          OutputError: If writing fails due to I/O or encoding errors.
        """
        if not lines:
            raise ValueError(f'{log_label.capitalize()} list cannot be empty.')
        # Ensure output directory exists
        output_path = self._ensure_output_directory(file_path)
        try:
            logging.info(f'Writing {log_label} output to {output_path}')
            content = self._build_content(header, lines)
            self._atomic_write(output_path, content, self.encoding)
            logging.info(f'{log_label.capitalize()} output written to {output_path} successfully')
        except (OSError, UnicodeEncodeError) as e:
            raise OutputError(
                f'Failed to write {log_label} output to {output_path}: {e}',
                file_path=str(output_path),
                output_type=output_type
            ) from e

    def write_text_output(
            self,
            file_path: Pathish,
            header: str,
            annotation_lines: list[str]
    ) -> None:
        """Writes annotated text output atomically.

        Thin wrapper over '_write_lines'.

        Args:
            file_path: Output file path.
            header: Header line for the file.
            annotation_lines: List of annotated text records (strings, one per element).
        """
        self._write_lines(
            file_path=file_path,
            header=header,
            lines=annotation_lines,
            log_label='annotations',
            output_type='write_annotation'
        )

    def write_metadata_output(
            self,
            file_path: Pathish,
            header: str,
            metadata: list[str]
    ) -> None:
        """Writes metadata table output atomically.

        Thin wrapper over '_write_lines'.

        Args:
            file_path: Output file path.
            header: Header line for the file.
            metadata: List of metadata rows (strings, one per element).
        """
        self._write_lines(
            file_path=file_path,
            header=header,
            lines=metadata,
            log_label='metadata',
            output_type='write_metadata'
        )

    @staticmethod
    def _file_size_and_trailing_newline(file: BinaryIO) -> tuple[int, bool]:
        """Returns file size and whether the file ends with LF.

         The file pointer ends at EOF on return.

        Args:
            file: Opened file object in binary mode.

        Returns:
            A 2-tuple '(size_in_bytes, ends_with_newline)'.
        """
        file.seek(0, os.SEEK_END)  # Move to end of file
        size = file.tell()  # Get file size (current byte offset)
        if size <= 0:  # Empty file -> doesn’t end with a newline.
            return 0, False
        file.seek(-1, os.SEEK_END)  # Move to last byte of file
        return size, file.read(1) == b'\n'  # Check if last byte is newline

    @staticmethod
    def _compose_chunk(
        *,  # everything after this * must be passed by keyword
        header: str,
        data: list[str],
        add_header: bool,
        needs_leading_newline: bool,
        newline: str,
    ) -> str:
        """Composes an append chunk with optional header and leading newline.

        Args:
          header: Header line to write when the file is empty..
          data: Data records (one line each).
          add_header: Whether to emit the header first at the start of the chunk.
          needs_leading_newline: Whether to prepend a single blank line to avoid sticking.
          newline: Line separator to use (typically "\\n").

        Returns:
          The chunk text to write. If non-empty, the chunk always ends with a newline.
        """
        content_parts: list[str] = []
        # Add header if needed
        if add_header and header:
            content_parts.append(header.rstrip('\r\n'))
        if needs_leading_newline:
            content_parts.append('')  # exactly one extra newline
        # Add annotations
        content_parts.extend(line.rstrip('\r\n') for line in data)
        # Build content
        content = newline.join(content_parts)
        if content and not content.endswith(newline):
            content += newline
        return content

    def _append_lines(
        self,
        *,
        file_path: Pathish,
        header: str,
        lines: list[str],
        log_label: str,
        output_type: str
    ) -> None:
        """Append 'lines' to 'file_path' (locked, non-atomic).

        This method:
            * Takes an exclusive `flock()` on the target file to serialize appenders.
            * Emits `header` only if the file is empty at lock time.
            * Adds a single separator newline if the existing file does not end
              with LF to avoid sticking new content to the last line.

        Args:
          file_path: Output file path.
          header: Header line (written only once when the file is empty).
          lines: Lines to append (one record per element).
          log_label: Human-readable label for logs ('annotations'/'metadata').
          output_type: Error tag.

        Raises:
          ValueError: If 'lines' is empty.
          OutputError: If writing fails due to I/O or encoding errors.
        """
        if not lines:
            raise ValueError(f'{log_label.capitalize()} list cannot be empty.')
        # Ensure output directory exists
        output_path = self._ensure_output_directory(file_path)
        try:
            # Open/create in binary append/update so we can check last byte reliably.
            with cast(BinaryIO, open(output_path, "a+b")) as file:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
                try:
                    size, ends_with_newline = self._file_size_and_trailing_newline(file)
                    needs_header = size == 0
                    needs_leading_newline = size > 0 and not ends_with_newline

                    chunk: str = self._compose_chunk(
                        header=header,
                        data=lines,
                        add_header=needs_header,
                        needs_leading_newline=needs_leading_newline,
                        newline=self.NEWLINE,
                    )
                    if chunk:
                        data: bytes = chunk.encode(self.encoding)
                        file.write(data)
                        file.flush()
                finally:
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)

            logging.info(f'Appended {len(lines)} {log_label} to {output_path}')
        except (OSError, UnicodeEncodeError) as e:
            if isinstance(e, UnicodeEncodeError):
                error_msg = f'Encoding error writing {log_label} to {output_path}: {e}'
            else:
                error_msg = f'I/O error writing {log_label} to {output_path}: {e}'
            raise OutputError(
                error_msg,
                file_path=str(output_path),
                output_type=output_type
            ) from e

    def append_text_output(
        self,
        file_path: Pathish,
        header: str,
        annotation_lines: list[str]
    ) -> None:
        """Appends annotated text output with an exclusive 'flock()' (locked, non-atomic).

        Thin wrapper over `_append_lines`.

         Args:
             file_path: Output file path.
             header: Header line for the file (written only once when the file is empty).
             annotation_lines: List of annotated text records (strings, one per element).
         """
        self._append_lines(
            file_path=file_path,
            header=header,
            lines=annotation_lines,
            log_label='annotations',
            output_type='append_annotation'
        )

    def append_metadata_output(
        self,
        file_path: Pathish,
        header: str,
        metadata: list[str]
    ) -> None:
        """Appends metadata table output with an exclusive 'flock()' (locked, non-atomic).

        Thin wrapper over `_append_lines`

        Args:
            file_path: Output file path.
            header: Header line for the file (written only once when the file is empty).
            metadata: List of metadata rows (strings, one per element).
        """
        self._append_lines(
            file_path=file_path,
            header=header,
            lines=metadata,
            log_label='metadata',
            output_type='append_metadata'
        )

    @staticmethod
    def write_stats_output(file_path: Pathish, stats_data: dict[str, Any]) -> None:
        """Write processing statistics to a JSON file (atomic).

        Args:
            file_path: Output file path for the statistics.
            stats_data: Dictionary containing processing statistics.

        Raises:
            ValueError: If stats_data is not a dictionary.
            OutputError: If writing to the file fails.
        """
        if not isinstance(stats_data, dict):
            raise ValueError('Stats data must be a dictionary.')
        # Ensure output directory exists
        output_path = OutputWriter._ensure_output_directory(file_path)
        try:
            logging.info(f'Writing processing statistics to {output_path}')
            content = json.dumps(stats_data, indent=2, ensure_ascii=False)
            OutputWriter._atomic_write(output_path, content, OutputWriter.DEFAULT_ENCODING)
            logging.info(f'Processing statistics written to: {output_path}')
        except (OSError, UnicodeEncodeError, TypeError) as e:
            logging.error(
                f"Error writing stats output to {file_path}: {e}", exc_info=True)
            raise OutputError(
                f"Failed to write stats output to {file_path}: {e}",
                file_path=str(file_path),
                output_type="stats"
            ) from e

    @staticmethod
    def clean_output_files(*file_paths: Pathish) -> None:
        """Clean up (delete) existing output files.

        Args:
            *file_paths: Variable number of file paths to clean up.

        Raises:
            IOError: If file deletion fails for any critical reason.
        """
        for file_path in file_paths:
            if not file_path:  # Skip empty/None file paths
                continue
            try:
                path = Path(file_path)
                if path.exists() and path.is_file():
                    path.unlink()
                    logging.info(f'Cleaned up existing output file: {file_path}')
                else:
                    logging.debug(f'Output file does not exist, skipping cleanup: {file_path}')
            except OSError as e:
                # Log error but don't fail the entire process for file cleanup issues
                logging.debug(f'Failed to clean up output file {file_path}: {e}')
        logging.info('Output file cleanup completed')