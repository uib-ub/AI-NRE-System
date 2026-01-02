"""Unit tests for llm.batch_models module.

Tests cover:
- BatchStatus enum values
- BatchRequest dataclass creation and validation
- BatchResponse dataclass creation and validation
- BatchProgress dataclass creation and validation
- Edge cases and validation errors
"""

from __future__ import annotations

import logging

import pytest

from ai_ner_system.llm.batch_models import (
    BatchProgress,
    BatchRequest,
    BatchResponse,
    BatchStatus,
)

log = logging.getLogger(__name__)


class TestBatchStatus:
    """Tests for BatchStatus enum."""

    def test_in_progress_value(self) -> None:
        """Test IN_PROGRESS enum value."""
        assert BatchStatus.IN_PROGRESS.value == "in_progress"

    def test_ended_value(self) -> None:
        """Test ENDED enum value."""
        assert BatchStatus.ENDED.value == "ended"

    def test_canceling_value(self) -> None:
        """Test CANCELING enum value."""
        assert BatchStatus.CANCELING.value == "canceling"

    def test_all_statuses_count(self) -> None:
        """Test that there are exactly 3 batch statuses."""
        assert len(BatchStatus) == 3

    @pytest.mark.parametrize(
        ("status_str", "expected_status"),
        [
            ("in_progress", BatchStatus.IN_PROGRESS),
            ("ended", BatchStatus.ENDED),
            ("canceling", BatchStatus.CANCELING),
        ],
    )
    def test_from_string(self, status_str: str, expected_status: BatchStatus) -> None:
        """Test creating BatchStatus from string value.

        Args:
            status_str: The string value to convert.
            expected_status: The expected BatchStatus enum.
        """
        assert BatchStatus(status_str) == expected_status

    def test_invalid_status_raises_error(self) -> None:
        """Test that invalid status string raises ValueError."""
        with pytest.raises(ValueError, match=r"'invalid'"):
            BatchStatus("invalid")


class TestBatchRequest:
    """Tests for BatchRequest dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating BatchRequest with required fields only."""
        request = BatchRequest(
            custom_id="req-001",
            prompt="Test prompt",
        )

        log.debug("Created BatchRequest: %s", request)

        assert request.custom_id == "req-001"
        assert request.prompt == "Test prompt"
        assert request.max_tokens == 20000  # default
        assert request.temperature == 0.0  # default

    @pytest.mark.parametrize(
        ("custom_id", "prompt", "max_tokens", "temperature"),
        [
            ("req-001", "Simple prompt", 1000, 0.5),
            ("req-002", "Another prompt", 50000, 1.0),
            ("batch-123-record-456", "Long prompt " * 100, 100, 0.0),
        ],
    )
    def test_creation_with_all_params(
        self,
        custom_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> None:
        """Test creating BatchRequest with all parameters.

        Args:
            custom_id: Unique identifier for the request.
            prompt: The input prompt text.
            max_tokens: Maximum tokens in response.
            temperature: Temperature for generation.
        """
        request = BatchRequest(
            custom_id=custom_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        log.debug("Created BatchRequest with params: %s", request)

        assert request.custom_id == custom_id
        assert request.prompt == prompt
        assert request.max_tokens == max_tokens
        assert request.temperature == temperature

    @pytest.mark.parametrize(
        ("custom_id", "prompt", "match_pattern"),
        [
            ("", "Valid prompt", r"(?i)custom_id cannot be empty"),
            ("   ", "Valid prompt", r"(?i)custom_id cannot be empty"),
            ("\t\n", "Valid prompt", r"(?i)custom_id cannot be empty"),
            ("valid-id", "", r"(?i)prompt cannot be empty"),
            ("valid-id", "   ", r"(?i)prompt cannot be empty"),
            ("valid-id", "\n\t", r"(?i)prompt cannot be empty"),
        ],
    )
    def test_validation_errors(
        self, custom_id: str, prompt: str, match_pattern: str
    ) -> None:
        """Test validation raises ValueError for invalid inputs.

        Args:
            custom_id: The custom ID to test.
            prompt: The prompt to test.
            match_pattern: Regex pattern to match error message.
        """
        with pytest.raises(ValueError, match=match_pattern) as exc_info:
            BatchRequest(custom_id=custom_id, prompt=prompt)

        log.debug("Validation error as expected: %s", exc_info.value)


class TestBatchResponse:
    """Tests for BatchResponse dataclass."""

    def test_successful_response(self) -> None:
        """Test creating a successful BatchResponse."""
        response = BatchResponse(
            custom_id="req-001",
            response_text="Generated text output",
            success=True,
        )

        log.debug("Created successful BatchResponse: %s", response)

        assert response.custom_id == "req-001"
        assert response.response_text == "Generated text output"
        assert response.success is True
        assert response.error_message is None

    def test_failed_response(self) -> None:
        """Test creating a failed BatchResponse."""
        response = BatchResponse(
            custom_id="req-002",
            response_text="",
            success=False,
            error_message="API rate limit exceeded",
        )

        log.debug("Created failed BatchResponse: %s", response)

        assert response.custom_id == "req-002"
        assert response.response_text == ""
        assert response.success is False
        assert response.error_message == "API rate limit exceeded"

    @pytest.mark.parametrize(
        ("custom_id", "response_text", "success", "error_message", "match_pattern"),
        [
            # Empty custom_id
            ("", "Text", True, None, r"(?i)custom_id cannot be empty"),
            # Successful response with empty response_text
            ("req-001", "", True, None, r"(?i)successful response cannot have empty"),
            (
                "req-001",
                "   ",
                True,
                None,
                r"(?i)successful response cannot have empty",
            ),
            # Failed response without error_message
            (
                "req-001",
                "",
                False,
                None,
                r"(?i)failed response must have error_message",
            ),
            ("req-001", "", False, "", r"(?i)failed response must have error_message"),
        ],
    )
    def test_validation_errors(
        self,
        custom_id: str,
        response_text: str,
        success: bool,
        error_message: str | None,
        match_pattern: str,
    ) -> None:
        """Test validation raises ValueError for invalid inputs.

        Args:
            custom_id: The custom ID to test.
            response_text: The response text to test.
            success: Whether the response is successful.
            error_message: The error message to test.
            match_pattern: Regex pattern to match error message.
        """
        with pytest.raises(ValueError, match=match_pattern) as exc_info:
            BatchResponse(
                custom_id=custom_id,
                response_text=response_text,
                success=success,
                error_message=error_message,
            )

        log.debug("Validation error as expected: %s", exc_info.value)

    def test_failed_response_can_have_partial_text(self) -> None:
        """Test that failed response can have partial response_text."""
        response = BatchResponse(
            custom_id="req-003",
            response_text="Partial output before error",
            success=False,
            error_message="Connection lost",
        )

        assert response.response_text == "Partial output before error"
        assert response.success is False
        assert response.error_message == "Connection lost"


class TestBatchProgress:
    """Tests for BatchProgress dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating BatchProgress with all required fields."""
        progress = BatchProgress(
            batch_num=1,
            batch_id="batch_abc123",
            status=BatchStatus.IN_PROGRESS,
            elapsed_time=120.5,
            request_counts={"succeeded": 5, "errored": 0, "processing": 10},
            created_at="2026-01-02T10:00:00Z",
            expires_at="2026-01-03T10:00:00Z",
        )

        log.debug("Created BatchProgress: %s", progress)

        assert progress.batch_num == 1
        assert progress.batch_id == "batch_abc123"
        assert progress.status == BatchStatus.IN_PROGRESS
        assert progress.elapsed_time == 120.5
        assert progress.request_counts == {
            "succeeded": 5,
            "errored": 0,
            "processing": 10,
        }
        assert progress.created_at == "2026-01-02T10:00:00Z"
        assert progress.expires_at == "2026-01-03T10:00:00Z"

    @pytest.mark.parametrize(
        "status",
        [
            BatchStatus.IN_PROGRESS,
            BatchStatus.ENDED,
            BatchStatus.CANCELING,
        ],
    )
    def test_all_status_values(self, status: BatchStatus) -> None:
        """Test BatchProgress can be created with any BatchStatus.

        Args:
            status: The BatchStatus to test.
        """
        progress = BatchProgress(
            batch_num=1,
            batch_id="batch_123",
            status=status,
            elapsed_time=0.0,
            request_counts={},
            created_at="2026-01-02T10:00:00Z",
            expires_at="2026-01-03T10:00:00Z",
        )

        assert progress.status == status

    @pytest.mark.parametrize(
        ("batch_id", "elapsed_time", "match_pattern"),
        [
            ("", 0.0, r"(?i)batch_id cannot be empty"),
            ("   ", 0.0, r"(?i)batch_id cannot be empty"),
            ("\t\n", 0.0, r"(?i)batch_id cannot be empty"),
            ("valid-id", -1.0, r"(?i)elapsed_time cannot be negative"),
            ("valid-id", -0.001, r"(?i)elapsed_time cannot be negative"),
        ],
    )
    def test_validation_errors(
        self, batch_id: str, elapsed_time: float, match_pattern: str
    ) -> None:
        """Test validation raises ValueError for invalid inputs.

        Args:
            batch_id: The batch ID to test.
            elapsed_time: The elapsed time to test.
            match_pattern: Regex pattern to match error message.
        """
        with pytest.raises(ValueError, match=match_pattern) as exc_info:
            BatchProgress(
                batch_num=1,
                batch_id=batch_id,
                status=BatchStatus.IN_PROGRESS,
                elapsed_time=elapsed_time,
                request_counts={},
                created_at="2026-01-02T10:00:00Z",
                expires_at="2026-01-03T10:00:00Z",
            )

        log.debug("Validation error as expected: %s", exc_info.value)

    def test_zero_elapsed_time_valid(self) -> None:
        """Test that zero elapsed_time is valid."""
        progress = BatchProgress(
            batch_num=0,
            batch_id="batch_new",
            status=BatchStatus.IN_PROGRESS,
            elapsed_time=0.0,
            request_counts={"processing": 100},
            created_at="2026-01-02T10:00:00Z",
            expires_at="2026-01-03T10:00:00Z",
        )

        assert progress.elapsed_time == 0.0

    def test_empty_request_counts_valid(self) -> None:
        """Test that empty request_counts dict is valid."""
        progress = BatchProgress(
            batch_num=1,
            batch_id="batch_123",
            status=BatchStatus.IN_PROGRESS,
            elapsed_time=0.0,
            request_counts={},
            created_at="2026-01-02T10:00:00Z",
            expires_at="2026-01-03T10:00:00Z",
        )

        assert progress.request_counts == {}

    def test_large_request_counts(self) -> None:
        """Test BatchProgress with large request counts."""
        counts = {
            "succeeded": 10000,
            "errored": 50,
            "processing": 0,
            "canceled": 100,
        }
        progress = BatchProgress(
            batch_num=5,
            batch_id="batch_large",
            status=BatchStatus.ENDED,
            elapsed_time=7200.0,
            request_counts=counts,
            created_at="2026-01-02T10:00:00Z",
            expires_at="2026-01-03T10:00:00Z",
        )

        assert progress.request_counts["succeeded"] == 10000
        assert progress.request_counts["errored"] == 50
