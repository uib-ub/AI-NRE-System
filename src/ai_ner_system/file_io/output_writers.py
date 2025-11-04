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
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO, ClassVar, cast

# fcntl is POSIX-only; gracefully handle Windows
try:
    import fcntl

    _has_fcntl = True
except ImportError:
    _has_fcntl = False
    fcntl = None  # type: ignore[assignment]

from .exceptions import OutputError

Pathish = str | Path  # Type alias for path-like objects


class OutputWriter:
    """Output file writer for annotated text, metadata, and JSON stats.

    Cross-platform compatibility:
        * `write_*` methods (atomic writes) work on all platforms.
        * `append_*` methods require POSIX file locking (fcntl) and will raise
          OutputError on Windows. For cross-platform append support, consider
          using atomic writes or installing portalocker.

    Notes:
        * `write_*` methods are atomic (tempfile + os.replace).
        * `append_*` methods take an exclusive `flock()` on the target file to
          serialize concurrent appenders across processes (POSIX only).
        * Header emission on append is determined solely by "is the file empty?"
          (robust to file rotation/truncation).
    """

    # Class constants for default settings
    DEFAULT_ENCODING: ClassVar[str] = "utf-8"
    NEWLINE: ClassVar[str] = "\n"

    def __init__(self, encoding: str = "utf-8") -> None:
        """Initialize the OutputWriter.

        Args:
            encoding: Text encoding used for all writes.
        """
        self.encoding = encoding
        logging.debug(
            "OutputWriter initialized with encoding: %s",
            self.encoding,
        )

    @staticmethod
    def _ensure_output_directory(file_path: Pathish) -> Path:
        """Ensure the output directory exists.

        Note: Settings.initialize() should have already created output directories.
        This method provides a safety check and creates directories if needed
        (e.g., when OutputWriter is used standalone in tests).

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
            logging.debug("Ensured output directory exists: %s", directory)
        except OSError as e:
            msg = f"Failed to create output directory {directory}: {e}"
            raise OutputError(
                msg,
                file_path=str(file_path),
            ) from e
        else:
            return path

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
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                dir=file_path.parent,
                encoding=encoding,
                newline="",
                suffix=".tmp",
            ) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)
            # Atomically replace the target file
            temp_path.replace(file_path)
            logging.debug("Atomic write completed for: %s", file_path)
        except (OSError, UnicodeEncodeError) as e:
            # Clean up temp file if it exists
            if temp_path is not None and temp_path.exists():
                with suppress(OSError):
                    temp_path.unlink()
            msg = f"Error during atomic write to {file_path}: {e}"
            raise OutputError(
                msg,
                file_path=str(file_path),
                output_type="atomic_write",
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
            content_parts.append(header.rstrip("\r\n"))
        # Add lines
        content_parts.extend(line.rstrip("\r\n") for line in data)
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
        output_type: str,
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
            msg = f"{log_label.capitalize()} list cannot be empty."
            raise ValueError(msg)
        # Ensure output directory exists
        output_path = self._ensure_output_directory(file_path)
        try:
            logging.info("Writing %s output to %s", log_label, output_path)
            content = self._build_content(header, lines)
            self._atomic_write(output_path, content, self.encoding)
            logging.info(
                "%s output written to %s successfully",
                log_label.capitalize(),
                output_path,
            )
        except (OSError, UnicodeEncodeError) as e:
            msg = f"Error writing {log_label} output to {output_path}: {e}"
            raise OutputError(
                msg,
                file_path=str(output_path),
                output_type=output_type,
            ) from e

    def write_text_output(
        self,
        file_path: Pathish,
        header: str,
        annotation_lines: list[str],
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
            log_label="annotations",
            output_type="write_annotation",
        )

    def write_metadata_output(
        self,
        file_path: Pathish,
        header: str,
        metadata: list[str],
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
            log_label="metadata",
            output_type="write_metadata",
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
        if size <= 0:  # Empty file -> doesn't end with a newline.
            return 0, False
        file.seek(-1, os.SEEK_END)  # Move to last byte of file
        return size, file.read(1) == b"\n"  # Check if last byte is newline

    @staticmethod
    def _compose_chunk(
        *,  # everything after this * must be passed by keyword
        header: str,
        data: list[str],
        add_header: bool,
        needs_leading_newline: bool,
        newline: str,
    ) -> str:
        r"""Composes an append chunk with optional header and leading newline.

        Args:
          header: Header line to write when the file is empty..
          data: Data records (one line each).
          add_header: Whether to emit the header first at the start of the chunk.
          needs_leading_newline: Whether to prepend a single blank line to avoid sticking.
          newline: Line separator to use (typically '\n').

        Returns:
          The chunk text to write. If non-empty, the chunk always ends with a newline.
        """
        content_parts: list[str] = []
        # Add header if needed
        if add_header and header:
            content_parts.append(header.rstrip("\r\n"))
        if needs_leading_newline:
            content_parts.append("")  # exactly one extra newline
        # Add annotations
        content_parts.extend(line.rstrip("\r\n") for line in data)
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
        output_type: str,
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
            msg = f"{log_label.capitalize()} list cannot be empty."
            raise ValueError(msg)

        # Check if fcntl is available (POSIX-only)
        if not _has_fcntl:
            msg = (
                "Append operations require fcntl (POSIX file locking), "
                "which is not available on this platform. "
                "Consider using atomic write operations instead, "
                "or install portalocker for cross-platform file locking."
            )
            raise OutputError(msg, output_type=output_type)

        # Ensure output directory exists
        output_path = self._ensure_output_directory(file_path)
        try:
            # Open/create in binary append/update so we can check last byte reliably.
            with cast("BinaryIO", output_path.open("a+b")) as file:
                # fcntl is guaranteed to be available at this point (checked above)
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # pyright: ignore[reportOptionalMemberAccess]
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
                    # fcntl is guaranteed to be available at this point (checked above)
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # pyright: ignore[reportOptionalMemberAccess]
            logging.info(
                "Appended %d %s to %s",
                len(lines),
                log_label,
                output_path,
            )
        except (OSError, UnicodeEncodeError) as e:
            error_prefix = (
                "Encoding error" if isinstance(e, UnicodeEncodeError) else "I/O error"
            )
            error_msg = f"{error_prefix} writing {log_label} to {output_path}: {e}"
            raise OutputError(
                error_msg,
                file_path=str(output_path),
                output_type=output_type,
            ) from e

    def append_text_output(
        self,
        file_path: Pathish,
        header: str,
        annotation_lines: list[str],
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
            log_label="annotations",
            output_type="append_annotation",
        )

    def append_metadata_output(
        self,
        file_path: Pathish,
        header: str,
        metadata: list[str],
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
            log_label="metadata",
            output_type="append_metadata",
        )

    @staticmethod
    def write_stats_output(file_path: Pathish, stats_data: dict[str, Any]) -> None:
        """Write processing statistics to a JSON file (atomic).

        Args:
            file_path: Output file path for the statistics.
            stats_data: Dictionary containing processing statistics.

        Raises:
            OutputError: If writing to the file fails.
        """
        # Ensure output directory exists
        output_path = OutputWriter._ensure_output_directory(file_path)
        try:
            logging.info("Writing processing statistics to %s", output_path)
            content = json.dumps(stats_data, indent=2, ensure_ascii=False)
            OutputWriter._atomic_write(
                output_path,
                content,
                OutputWriter.DEFAULT_ENCODING,
            )
            logging.info("Processing statistics written to: %s", output_path)
        except (OSError, UnicodeEncodeError, TypeError) as e:
            logging.exception("Error writing stats output to %s", file_path)
            msg = f"Error writing stats output to {file_path}: {e}"
            raise OutputError(
                msg,
                file_path=str(file_path),
                output_type="stats",
            ) from e

    @staticmethod
    def clean_output_files(*file_paths: Pathish) -> None:
        """Clean up (delete) existing output files.

        Args:
            *file_paths: Variable number of file paths to clean up.

        Raises:
            FileIOError: If file deletion fails for any critical reason.
        """
        for file_path in file_paths:
            if not file_path:  # Skip empty/None file paths
                continue
            try:
                path = Path(file_path)
                if path.exists() and path.is_file():
                    path.unlink()
                    logging.info(
                        "Cleaned up existing output file: %s",
                        file_path,
                    )
                else:
                    logging.debug(
                        "Output file does not exist, skipping cleanup: %s",
                        file_path,
                    )
            except OSError as e:
                # Log error but don't fail the entire process for file cleanup issues
                logging.debug(
                    "Failed to clean up output file %s: %s",
                    file_path,
                    e,
                    exc_info=True,
                )
        logging.info("Output file cleanup completed")
