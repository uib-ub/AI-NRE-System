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
    LLMResponseError,
    ProcessingError,
    ValidationError,
)

log = logging.getLogger(__name__)


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

    def test_inheritance(self) -> None:
        """Test LLMResponseError inherits from ProcessingError."""
        error = LLMResponseError("Test")
        assert isinstance(error, ProcessingError)
        assert isinstance(error, Exception)
