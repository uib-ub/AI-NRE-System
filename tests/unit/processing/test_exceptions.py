"""Unit tests for processing.exceptions module.

Tests cover:
- Exception creation with various parameters
- String formatting (__str__) for each exception type
- Context appending logic (brevid, operation)
- Truncation behaviour for long content (LLMResponseError, ParseError)
- The "endswith ')'" guard that controls context insertion
- Exception inheritance hierarchy
- Edge cases (None values, empty strings, boundary lengths)
"""

from __future__ import annotations

import logging

import pytest

from ai_ner_system.processing.exceptions import (
    BatchProcessingError,
    LLMResponseError,
    ParseError,
    ProcessingError,
    ValidationError,
)

log = logging.getLogger(__name__)

# Mirror the truncation threshold used by processing exceptions for assertions.
MAX_CONTENT_LENGTH = 100


class TestProcessingError:
    """Tests for base ProcessingError exception."""

    def test_basic_creation(self) -> None:
        """Test creating ProcessingError with message only."""
        error = ProcessingError("Something went wrong")

        log.debug("Created ProcessingError: %s", error)

        assert str(error) == "Something went wrong"
        assert error.brevid is None
        assert error.operation is None

    @pytest.mark.parametrize(
        ("err_msg", "brevid", "operation"),
        [
            ("Failed", "DN1_001", None),
            ("Failed", None, "test_operation"),
            ("Failed", "DN1_001", "test_operation"),
            ("Failed", None, None),
            ("Failed", "", ""),
            ("Failed", "", None),
            ("Failed", None, ""),
        ],
    )
    def test_basic_creation_with_params(
        self,
        err_msg: str,
        brevid: str | None,
        operation: str | None,
    ) -> None:
        """Test creating ProcessingError with various parameters."""
        error = ProcessingError(err_msg, brevid=brevid, operation=operation)

        log.debug(
            "Created ProcessingError with params: err_msg=%s, brevid=%s, operation=%s",
            err_msg,
            brevid,
            operation,
        )

        log.debug("ProcessingError string representation: %s", str(error))

        assert err_msg in str(error)
        assert (
            f"brevid: {brevid}" not in str(error)
            if not brevid
            else f"brevid: {brevid}" in str(error)
        )
        assert (
            f"operation: {operation}" not in str(error)
            if not operation
            else f"operation: {operation}" in str(error)
        )
        assert error.brevid == brevid
        assert error.operation == operation

    def test_inheritance(self) -> None:
        """Test ProcessingError inherits from Exception."""
        error = ProcessingError("Test")
        assert isinstance(error, Exception)


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_basic_creation(self) -> None:
        """Test creating ValidationError with message only."""
        error = ValidationError("Validation failed")

        log.debug("Created ValidationError: %s", error)

        assert str(error) == "Validation failed"
        assert error.brevid is None
        assert error.operation is None
        assert error.missing_fields == []

    @pytest.mark.parametrize(
        ("err_msg", "brevid", "operation", "missing_fields"),
        [
            ("Validation failed", "DN1_001", None, None),
            ("Validation failed", None, "test_operation", None),
            ("Validation failed", None, None, ["Bindnr", "Tekst"]),
            ("Validation failed", "", "", []),
            ("Validation failed", "DN1_001", "", []),
            ("Validation failed", "", "test_operation", []),
            ("Validation failed", "DN1_001", "test_operation", []),
            ("Validation failed", "DN1_001", "test_operation", ["Bindnr"]),
            ("Validation failed", "DN1_001", "test_operation", ["Bindnr", "Tekst"]),
        ],
    )
    def test_basic_creation_with_params(
        self,
        err_msg: str,
        brevid: str | None,
        operation: str | None,
        missing_fields: list[str] | None,
    ) -> None:
        """Test creating ValidationError with various parameters."""
        error = ValidationError(
            err_msg,
            brevid=brevid,
            operation=operation,
            missing_fields=missing_fields,
        )

        log.debug(
            "Created ValidationError with params: err_msg=%s, brevid=%s, operation=%s, missing_fields=%s",
            err_msg,
            brevid,
            operation,
            missing_fields,
        )

        log.debug("ValidationError string representation: %s", str(error))

        assert err_msg in str(error)
        assert (
            f"brevid: {brevid}" not in str(error)
            if not brevid
            else f"brevid: {brevid}" in str(error)
        )
        assert (
            f"operation: {operation}" not in str(error)
            if not operation
            else f"operation: {operation}" in str(error)
        )
        assert error.brevid == brevid
        assert error.operation == operation
        assert (
            error.missing_fields == missing_fields
            if missing_fields is not None
            else error.missing_fields == []
        )

        # Verify missing_fields appears in __str__ only when context ')'  exists
        has_context = bool(brevid) or bool(operation)
        if missing_fields and has_context:
            fields_str = ", ".join(missing_fields)
            assert f"missing_fields: [{fields_str}]" in str(error)
        elif missing_fields:
            fields_str = ", ".join(missing_fields)
            assert f"missing_fields: [{fields_str}]" not in str(error)

    def test_inheritance(self) -> None:
        """Test ValidationError inherits from ProcessingError."""
        error = ValidationError("Test")
        assert isinstance(error, ProcessingError)
        assert isinstance(error, Exception)


class TestLLMResponseError:
    """Tests for LLMResponseError exception."""

    def test_basic_creation(self) -> None:
        """Test creating LLMResponseError with message only."""
        error = LLMResponseError("LLM response error")

        log.debug("Created LLMResponseError: %s", error)

        assert str(error) == "LLM response error"
        assert error.brevid is None
        assert error.operation is None

    @pytest.mark.parametrize(
        ("err_msg", "brevid", "operation", "response_text"),
        [
            ("LLM response error", "DN1_001", None, None),
            ("LLM response error", None, "test_operation", None),
            ("LLM response error", None, None, "short response"),
            ("LLM response error", "", "", ""),
            ("LLM response error", "DN1_001", "", ""),
            ("LLM response error", "", "test_operation", ""),
            ("LLM response error", "DN1_001", "test_operation", ""),
            ("LLM response error", "DN1_001", "test_operation", "short response"),
            ("LLM response error", "DN1_001", "test_operation", "x" * 100),
            ("LLM response error", "DN1_001", "test_operation", "x" * 101),
            ("LLM response error", "DN1_001", "test_operation", "x" * 150),
        ],
    )
    def test_basic_creation_with_params(
        self,
        err_msg: str,
        brevid: str | None,
        operation: str | None,
        response_text: str | None,
    ) -> None:
        """Test creating LLMResponseError with various parameters."""
        error = LLMResponseError(
            err_msg,
            brevid=brevid,
            operation=operation,
            response_text=response_text,
        )

        log.debug(
            "Created LLMResponseError with params: err_msg=%s, brevid=%s, operation=%s, response_text=%s",
            err_msg,
            brevid,
            operation,
            response_text,
        )

        log.debug("LLMResponseError string representation: %s", str(error))

        assert err_msg in str(error)
        assert (
            f"brevid: {brevid}" not in str(error)
            if not brevid
            else f"brevid: {brevid}" in str(error)
        )
        assert (
            f"operation: {operation}" not in str(error)
            if not operation
            else f"operation: {operation}" in str(error)
        )
        assert error.brevid == brevid
        assert error.operation == operation
        assert error.response_text == response_text

        # Verify response_text appears in __str__ (with truncation) only when
        # context ')' exists
        has_context = bool(brevid) or bool(operation)
        if response_text and has_context:
            if len(response_text) > MAX_CONTENT_LENGTH:
                truncated = response_text[:MAX_CONTENT_LENGTH] + "..."
            else:
                truncated = response_text
            assert f"response_text: '{truncated}'" in str(error)
        elif response_text:
            assert "response_text:" not in str(error)

    def test_inheritance(self) -> None:
        """Test LLMResponseError inherits from ProcessingError."""
        error = LLMResponseError("Test")
        assert isinstance(error, ProcessingError)
        assert isinstance(error, Exception)


class TestParseError:
    """Tests for ParseError exception."""

    def test_basic_creation(self) -> None:
        """Test creating ParseError with message only."""
        error = ParseError("Parse error")

        log.debug("Created ParseError: %s", error)

        assert str(error) == "Parse error"
        assert error.brevid is None
        assert error.operation is None

    @pytest.mark.parametrize(
        ("err_msg", "brevid", "operation", "parse_type", "content"),
        [
            ("Invalid format", "DN1_001", None, None, None),
            ("Invalid format", "DN1_001", None, "json", None),
            ("Could not parse", "DN1_001", None, None, "{bad json}"),
            ("Parsing error", "DN1_001", "parse_response", "json", '{"broken": true'),
            ("Parsing error", "DN1_001", None, None, "x" * 120),
            ("Parsing error", "", None, "json", "some content"),
        ],
    )
    def test_basic_creation_with_params(
        self,
        err_msg: str,
        brevid: str | None,
        operation: str | None,
        parse_type: str | None,
        content: str | None,
    ) -> None:
        """Test creating ParseError with various parameters."""
        error = ParseError(
            err_msg,
            brevid=brevid,
            operation=operation,
            parse_type=parse_type,
            content=content,
        )

        log.debug(
            "Created ParseError with params: err_msg=%s, brevid=%s, operation=%s, parse_type=%s, content=%s",
            err_msg,
            brevid,
            operation,
            parse_type,
            content,
        )

        log.debug("ParseError string representation: %s", str(error))

        assert err_msg in str(error)
        assert (
            f"brevid: {brevid}" not in str(error)
            if not brevid
            else f"brevid: {brevid}" in str(error)
        )
        assert (
            f"operation: {operation}" not in str(error)
            if not operation
            else f"operation: {operation}" in str(error)
        )
        assert error.brevid == brevid
        assert error.operation == operation
        assert error.parse_type == parse_type
        assert error.content == content

        # Verify parse_type/content appear in __str__ (with truncation) only
        # when context ')' exists
        has_context = bool(brevid) or bool(operation)
        if has_context:
            if parse_type:
                assert f"parse_type: {parse_type}" in str(error)
            if content:
                if len(content) > MAX_CONTENT_LENGTH:
                    truncated = content[:MAX_CONTENT_LENGTH] + "..."
                else:
                    truncated = content
                assert f"content: '{truncated}'" in str(error)
        else:
            if parse_type:
                assert "parse_type:" not in str(error)
            if content:
                assert "content:" not in str(error)

    def test_inheritance(self) -> None:
        """Test ParseError inherits from ProcessingError."""
        error = ParseError("Test")
        assert isinstance(error, ProcessingError)
        assert isinstance(error, Exception)


class TestBatchProcessingError:
    """Tests for BatchProcessingError exception."""

    def test_basic_creation(self) -> None:
        """Test creating BatchProcessingError with message only."""
        error = BatchProcessingError("Batch processing failed")

        log.debug("Created BatchProcessingError: %s", error)

        assert str(error) == "Batch processing failed"
        assert error.brevid is None
        assert error.operation is None

    @pytest.mark.parametrize(
        ("err_msg", "operation", "batch_id"),
        [
            ("Batch processing failed", "process_batch", None),
            ("Batch processing failed", None, "batch_123"),
            ("Batch processing failed", "process_batch", "batch_123"),
        ],
    )
    def test_basic_creation_with_params(
        self,
        err_msg: str,
        operation: str | None,
        batch_id: str | None,
    ) -> None:
        """Test creating BatchProcessingError with various parameters."""
        error = BatchProcessingError(
            err_msg,
            operation=operation,
            batch_id=batch_id,
        )

        log.debug(
            "Created BatchProcessingError with params: err_msg=%s, operation=%s, batch_id=%s",
            err_msg,
            operation,
            batch_id,
        )

        log.debug("BatchProcessingError string representation: %s", str(error))

        assert err_msg in str(error)
        assert (
            f"operation: {operation}" not in str(error)
            if not operation
            else f"operation: {operation}" in str(error)
        )
        assert (
            f"batch_id: {batch_id}" not in str(error)
            if not batch_id or not str(error).endswith(")")
            else f"batch_id: {batch_id}" in str(error)
        )
        assert error.operation == operation
        assert error.batch_id == batch_id

    def test_inheritance(self) -> None:
        """Test BatchProcessingError inherits from ProcessingError."""
        error = BatchProcessingError("Test")
        assert isinstance(error, ProcessingError)
        assert isinstance(error, Exception)
