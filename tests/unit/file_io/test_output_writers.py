"""Unit tests for OutputWriter class.

Tests cover:
- Text output writing (atomic)
- Metadata output writing (atomic)
- Statistics output writing (JSON, atomic)
- Append operations (POSIX file locking)
- Error handling for file I/O operations
- Directory creation and cleanup
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
from typing import TYPE_CHECKING

import pytest

from ai_ner_system.file_io.exceptions import OutputError
from ai_ner_system.file_io.output_writers import OutputWriter

if TYPE_CHECKING:
    from pathlib import Path


log = logging.getLogger(__name__)


class TestOutputWriter:
    """Unit tests for OutputWriter."""

    def test_init_default_encoding(self) -> None:
        """Test OutputWriter initialization with default encoding."""
        writer = OutputWriter()
        assert writer.encoding == "utf-8"

    def test_init_custom_encoding(self) -> None:
        """Test OutputWriter initialization with custom encoding."""
        writer = OutputWriter(encoding="latin-1")
        assert writer.encoding == "latin-1"

    def test_write_text_output_success(
        self, tmp_path: Path, sample_header: str, sample_annotated_lines: list[str]
    ) -> None:
        """Test writing text output successfully.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            sample_header: Sample CSV header string.
            sample_annotated_lines: List of sample annotated text lines.
        """
        output_file = tmp_path / "output.txt"
        writer = OutputWriter()

        writer.write_text_output(output_file, sample_header, sample_annotated_lines)

        # Verify file exists and content is correct
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        log.debug("Written content:\n%s", content)
        expected = (
            f"{sample_header}\n"
            f"{sample_annotated_lines[0]}\n"
            f"{sample_annotated_lines[1]}\n"
            f"{sample_annotated_lines[2]}\n"
        )
        log.debug("Expected content:\n%s", expected)
        assert content == expected, f"Got:\n{content}\nexpected:\n{expected}"

    def test_write_text_output_creates_directory(self, tmp_path: Path) -> None:
        """Test that write_text_output creates parent directories.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        output_file = tmp_path / "nested" / "dir" / "output.txt"
        writer = OutputWriter()

        header = "Header"
        lines = ["Line 1"]

        writer.write_text_output(output_file, header, lines)

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_write_text_output_empty_lines_raises_error(self, tmp_path: Path) -> None:
        """Test that write_text_output raises ValueError for empty lines.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        output_file = tmp_path / "output.txt"
        writer = OutputWriter()

        with pytest.raises(ValueError, match="cannot be empty") as exc_info:
            writer.write_text_output(output_file, "Header", [])

        log.debug("Caught expected ValueError: %s", exc_info.value)
        assert "Annotations list cannot be empty" in str(exc_info.value)

    def test_write_metadata_output_success(
        self,
        tmp_path: Path,
        sample_metadata_header: str,
        sample_metadata_lines: list[str],
    ) -> None:
        """Test writing metadata output successfully.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            sample_metadata_header: Sample metadata CSV header string.
            sample_metadata_lines: List of sample metadata lines.
        """
        output_file = tmp_path / "metadata.txt"
        writer = OutputWriter()

        writer.write_metadata_output(
            output_file, sample_metadata_header, sample_metadata_lines
        )

        # Verify file exists and content is correct
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        log.debug("Written metadata content:\n%s", content)
        expected = (
            f"{sample_metadata_header}\n"
            f"{sample_metadata_lines[0]}\n"
            f"{sample_metadata_lines[1]}\n"
            f"{sample_metadata_lines[2]}\n"
        )
        log.debug("Expected metadata content:\n%s", expected)
        assert content == expected

    def test_write_metadata_output_empty_lines_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that write_metadata_output raises ValueError for empty metadata.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        output_file = tmp_path / "metadata.txt"
        writer = OutputWriter()

        with pytest.raises(ValueError, match="cannot be empty") as exc_info:
            writer.write_metadata_output(output_file, "Header", [])

        log.debug("Caught expected ValueError: %s", exc_info.value)
        assert "Metadata list cannot be empty" in str(exc_info.value)

    def test_write_stats_output_success(self, tmp_path: Path) -> None:
        """Test writing statistics output successfully.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        output_file = tmp_path / "stats.json"

        stats_data = {
            "total_records": 13,
            "processed_records": 13,
            "failed_records": 0,
            "success_rate": 100.0,
            "processing_time": 217.33317613601685,
            "throughput": 0.05981599418518605,
            "batch_info": None,
            "start_time": 1760042636.40566,
            "end_time": 1760042853.738836,
            "timestamp": 1760042853.738881,
            "processing_mode": "async",
        }

        OutputWriter.write_stats_output(output_file, stats_data)

        # Verify file exists and content is correct
        assert output_file.exists()
        loaded_data = json.loads(output_file.read_text(encoding="utf-8"))
        assert loaded_data == stats_data

    def test_write_stats_output_handles_json_serialization_error(
        self, tmp_path: Path
    ) -> None:
        """Test that TypeError from JSON serialization is wrapped in OutputError.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        output_file = tmp_path / "stats.json"

        # Create data with non-serializable object (e.g., a set or custom object)
        class NonSerializable:
            pass

        stats_data = {
            "total_records": 10,
            "non_serializable": NonSerializable(),  # This will cause TypeError
        }

        with pytest.raises(OutputError) as exc_info:
            OutputWriter.write_stats_output(output_file, stats_data)

        log.debug("Caught expected OutputError: %s", exc_info.value)

        assert "Error serializing stats data to JSON" in str(exc_info.value)
        assert str(output_file) in str(exc_info.value)

    def test_atomic_write_replaces_existing_file(self, tmp_path: Path) -> None:
        """Test that atomic write replaces existing file content.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        output_file = tmp_path / "output.txt"
        writer = OutputWriter()

        # Write initial content
        writer.write_text_output(output_file, "Header1", ["Line1"])
        content = output_file.read_text()
        log.debug("Initial content:\n%s", content)
        assert "Header1" in content

        # Overwrite with new content
        writer.write_text_output(output_file, "Header2", ["Line2"])
        content = output_file.read_text()
        log.debug("Replaced content:\n%s", content)
        assert "Header2" in content
        assert "Header1" not in content

    def test_clean_output_files_removes_existing_files(self, tmp_path: Path) -> None:
        """Test clean_output_files removes existing files.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("content1")
        file2.write_text("content2")

        assert file1.exists()
        assert file2.exists()

        OutputWriter.clean_output_files(file1, file2)

        assert not file1.exists()
        assert not file2.exists()

    def test_clean_output_files_skips_nonexistent_files(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test clean_output_files handles nonexistent files gracefully.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            caplog: Pytest fixture for capturing log output.
        """
        nonexistent_file = tmp_path / "nonexistent.txt"

        with caplog.at_level(logging.DEBUG):
            OutputWriter.clean_output_files(nonexistent_file)

        # Should not raise error
        log.debug("Log records:\n%s", "\n".join(rec.message for rec in caplog.records))
        assert any("does not exist" in rec.message for rec in caplog.records)

    def test_clean_output_files_skips_empty_paths(self) -> None:
        """Test clean_output_files skips None and empty string paths."""
        # Should not raise any errors
        OutputWriter.clean_output_files("", None, "")  # type: ignore[arg-type]

    def test_clean_output_files_handles_deletion_error(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that OSError during cleanup is logged but doesn't raise.

        This tests the defensive error handling where cleanup failures
        are logged but don't stop the cleanup process for other files.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            caplog: Pytest fixture for capturing log messages.
            monkeypatch: Pytest fixture for monkeypatching.
        """
        # Create test files
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file3 = tmp_path / "file3.txt"

        file1.write_text("content1")
        file2.write_text("content2")
        file3.write_text("content3")

        # Store original unlink method
        original_unlink = pathlib.Path.unlink

        # Create a wrapper that raises OSError for file2
        def mock_unlink(self: pathlib.Path, missing_ok: bool = False) -> None:  # noqa: FBT002
            if "file2.txt" in str(self):
                raise OSError("Permission denied")
            original_unlink(self, missing_ok=missing_ok)

        # Patch Path.unlink
        monkeypatch.setattr(pathlib.Path, "unlink", mock_unlink)

        with caplog.at_level(logging.DEBUG):
            # clean_output_files should NOT raise even if one file fails to delete
            OutputWriter.clean_output_files(file1, file2, file3)

        # The key is that no exception was raised and cleanup completed
        assert "Output file cleanup completed" in caplog.text
        assert "Failed to clean up output file" in caplog.text

        # file1 and file3 should be deleted, file2 should still exist (deletion failed)
        assert not file1.exists()
        assert file2.exists()  # Should still exist because unlink failed
        assert not file3.exists()

    def test_encoding_error_handling(self, tmp_path: Path) -> None:
        """Test handling of encoding errors during write.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        output_file = tmp_path / "output.txt"
        writer = OutputWriter(encoding="ascii")

        # Try to write non-ASCII characters with ASCII encoding
        header = "Header"
        lines = ["Text with unicode: Åæø"]

        with pytest.raises(OutputError) as exc_info:
            writer.write_text_output(output_file, header, lines)

        log.debug("Caught expected OutputError: %s", exc_info.value)

        assert "output" in str(exc_info.value).lower()

    def test_atomic_write_error_handling(self, tmp_path: Path) -> None:
        """Test handling of I/O errors during write.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        # Create a directory where a file is expected
        output_file = tmp_path / "output.txt"
        output_file.mkdir()  # Create a directory instead of a file

        writer = OutputWriter()

        header = "Header"
        lines = ["Line 1"]

        with pytest.raises(OutputError) as exc_info:
            writer.write_text_output(output_file, header, lines)

        log.debug("Caught expected OutputError: %s", exc_info.value)

        assert "Error during atomic write" in str(exc_info.value)

    def test_write_ensure_output_directory_error_handling(self, tmp_path: Path) -> None:
        """Test error handling when creating output directories fails.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        # Create a file where a directory is expected
        output_dir = tmp_path / "output_dir"
        output_dir.write_text("I am a file, not a directory")

        output_file = output_dir / "output.txt"
        writer = OutputWriter()

        header = "Header"
        lines = ["Line 1"]

        with pytest.raises(OutputError) as exc_info:
            writer.write_text_output(output_file, header, lines)

        log.debug("Caught expected OutputError: %s", exc_info.value)

        assert "Failed to create output directory" in str(exc_info.value)

    def test_write_text_output_with_custom_encoding(self, tmp_path: Path) -> None:
        """Test writing text output with custom encoding.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        output_file = tmp_path / "output.txt"
        writer = OutputWriter(encoding="latin-1")

        header = "Header"
        lines = ["Text with latin-1: café"]

        writer.write_text_output(output_file, header, lines)

        # Read back with correct encoding
        content = output_file.read_text(encoding="latin-1")
        log.debug("Written content with latin-1 encoding:\n%s", content)
        assert "café" in content

    @pytest.mark.skipif(
        sys.platform == "win32", reason="fcntl not available on Windows"
    )
    def test_append_text_output_creates_file_with_header(
        self,
        tmp_path: Path,
        sample_header: str,
        sample_annotated_lines: list[str],
    ) -> None:
        """Test append_text_output creates new file with header.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            sample_header: Sample CSV header string.
            sample_annotated_lines: List of sample annotated text lines.
        """
        output_file = tmp_path / "output.txt"
        writer = OutputWriter()

        writer.append_text_output(output_file, sample_header, sample_annotated_lines)

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        log.debug("Appended content:\n%s", content)
        expected = (
            f"{sample_header}\n"
            f"{sample_annotated_lines[0]}\n"
            f"{sample_annotated_lines[1]}\n"
            f"{sample_annotated_lines[2]}\n"
        )
        log.debug("Expected content:\n%s", expected)
        assert content == expected

    @pytest.mark.skipif(
        sys.platform == "win32", reason="fcntl not available on Windows"
    )
    def test_append_text_output_appends_without_duplicate_header(
        self,
        tmp_path: Path,
        sample_header: str,
    ) -> None:
        """Test append_text_output appends to existing file without duplicate header.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            sample_header: Sample CSV header string.
        """
        output_file = tmp_path / "output.txt"
        writer = OutputWriter()

        lines1 = ["1;001;Text1"]
        lines2 = ["2;002;Text2"]

        # First append - should write header
        writer.append_text_output(output_file, sample_header, lines1)

        # Second append - should NOT write header again
        writer.append_text_output(output_file, sample_header, lines2)

        content = output_file.read_text(encoding="utf-8")
        log.debug("Appended content:\n%s", content)

        # Header should appear only once
        assert content.count(sample_header) == 1
        assert "Text1" in content
        assert "Text2" in content

    @pytest.mark.skipif(
        sys.platform == "win32", reason="fcntl not available on Windows"
    )
    def test_append_metadata_output_success(
        self,
        tmp_path: Path,
        sample_metadata_header: str,
        sample_metadata_lines: list[str],
    ) -> None:
        """Test append_metadata_output appends metadata successfully.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
            sample_metadata_header: Sample metadata CSV header string.
            sample_metadata_lines: List of sample metadata lines.
        """
        output_file = tmp_path / "metadata.txt"
        writer = OutputWriter()

        for metadata_line in sample_metadata_lines:
            writer.append_metadata_output(
                output_file, sample_metadata_header, [metadata_line]
            )

        content = output_file.read_text(encoding="utf-8")
        log.debug("Appended metadata content:\n%s", content)

        expected = f"{sample_metadata_header}\n{sample_metadata_lines[0]}\n{sample_metadata_lines[1]}\n{sample_metadata_lines[2]}\n"
        log.debug("Expected metadata content:\n%s", expected)

        assert content.count(sample_metadata_header) == 1
        assert expected == content

    @pytest.mark.skipif(
        sys.platform == "win32", reason="fcntl not available on Windows"
    )
    def test_append_adds_separator_newline_when_file_missing_trailing_newline(
        self, tmp_path: Path
    ) -> None:
        """Test append adds separator newline when existing file lacks trailing newline.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        output_file = tmp_path / "output.txt"
        writer = OutputWriter()

        # Create file without trailing newline
        output_file.write_text(
            "Header\nExisting line without newline", encoding="utf-8"
        )

        lines = ["New line"]

        writer.append_text_output(output_file, "Header", lines)

        content = output_file.read_text(encoding="utf-8")

        log.debug("Appended content with separator newline:\n%s", content)
        # Should have separator newline between old and new content
        assert "without newline\nNew line\n" in content

    def test_append_empty_lines_raises_error(self, tmp_path: Path) -> None:
        """Test that append operations raise ValueError for empty lines."""
        output_file = tmp_path / "output.txt"
        writer = OutputWriter()

        with pytest.raises(ValueError, match="cannot be empty") as exc_info:
            writer.append_text_output(output_file, "Header", [])

        log.debug("Caught expected ValueError: %s", exc_info.value)

        assert "Annotations list cannot be empty" in str(exc_info.value)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="fcntl not available on Windows"
    )
    def test_append_text_output_handles_encoding_error(self, tmp_path: Path) -> None:
        """Test that UnicodeEncodeError in append is wrapped in OutputError.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        output_file = tmp_path / "output.txt"
        # Use ASCII encoding which cannot encode non-ASCII characters
        writer = OutputWriter(encoding="ascii")

        header = "Header"
        lines = ["Text with unicode: café"]

        with pytest.raises(OutputError) as exc_info:
            writer.append_text_output(output_file, header, lines)

        log.debug("Caught expected OutputError: %s", exc_info.value)

        assert "Encoding error" in str(exc_info.value)
        assert "annotations" in str(exc_info.value)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="fcntl not available on Windows"
    )
    def test_append_text_output_handles_permission_error(self, tmp_path: Path) -> None:
        """Test that OSError in append is wrapped in OutputError.

        Args:
            tmp_path: Pytest fixture providing a temporary directory.
        """
        output_file = tmp_path / "output.txt"
        # Create file and make it read-only
        output_file.write_text("existing content\n")
        output_file.chmod(0o444)  # Read-only

        writer = OutputWriter()
        header = "Header"
        lines = ["Line 1"]

        try:
            with pytest.raises(OutputError) as exc_info:
                writer.append_text_output(output_file, header, lines)

            log.debug("Caught expected OutputError: %s", exc_info.value)

            assert "I/O error" in str(exc_info.value)
            assert "annotations" in str(exc_info.value)
        finally:
            # Restore permissions for cleanup
            output_file.chmod(0o644)
