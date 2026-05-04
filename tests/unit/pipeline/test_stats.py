"""Unit tests for pipeline.stats module.

Tests cover:
- ApplicationError: creation, inheritance, message, chaining
- AsyncProcessingStats: creation, defaults, __post_init__ validation,
  computed properties (success_rate, is_complete, throughput),
  summary() dict, __str__ formatting
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from ai_ner_system.pipeline.stats import ApplicationError, AsyncProcessingStats
from ai_ner_system.processing.entities import ProcessingResult

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


# ===================================================================
# ApplicationError
# ===================================================================
class TestApplicationError:
    """Tests for ApplicationError exception class."""

    def test_creation(self) -> None:
        """Test basic creation with a message."""
        error = ApplicationError("Test error message")
        assert str(error) == "Test error message"

    def test_inherits_exception(self) -> None:
        """Test ApplicationError is a subclass of Exception."""
        assert issubclass(ApplicationError, Exception)
        assert isinstance(ApplicationError("Test"), Exception)

    def test_empty_message(self) -> None:
        """Test creation with empty message."""
        error = ApplicationError("")
        assert str(error) == ""

    def test_chaining(self) -> None:
        """Test exception chaining preserves __cause__."""
        cause = ValueError("Underlying cause")
        with pytest.raises(ApplicationError) as exc_info:
            raise ApplicationError("wrapper") from cause
        log.debug("Caught ApplicationError: %s", exc_info.value)
        log.debug("Original cause: %s", exc_info.value.__cause__)
        assert exc_info.value.__cause__ is cause
        assert str(exc_info.value) == "wrapper"


# ===================================================================
# AsyncProcessingStats — creation & defaults
# ===================================================================
class TestAsyncProcessingStatsCreation:
    """Tests for AsyncProcessingStats dataclass creation."""

    def test_default_values(self) -> None:
        """Test all defaults are set correctly."""
        stats = AsyncProcessingStats()
        assert stats.total_records == 0
        assert stats.processed_records == 0
        assert stats.failed_records == 0
        assert stats.start_time == 0.0
        assert stats.end_time is None
        assert stats.processing_time == 0.0
        assert stats.batch_info is None
        assert stats.results == []

    def test_default_results_lists_are_distinct(self) -> None:
        """Test separate instances do not share the default results list."""
        stats1 = AsyncProcessingStats()
        stats2 = AsyncProcessingStats()

        stats1.results.append(
            ProcessingResult(
                record_id="record-1",
                brevid="brevid-1",
                annotated_text="annotated text",
                processing_time=1.23,
            )
        )

        assert stats1.results is not stats2.results
        assert len(stats1.results) == 1
        assert stats2.results == []

    def test_with_all_fields(self) -> None:
        """Test creation with all fields specified."""
        supplied_results = [
            ProcessingResult(
                record_id="record-1",
                brevid="brevid-1",
                annotated_text="annotated text",
                processing_time=1.23,
            )
        ]
        stats = AsyncProcessingStats(
            total_records=100,
            processed_records=80,
            failed_records=20,
            start_time=1620000000.0,
            end_time=1620003600.0,
            processing_time=3600.0,
            batch_info={"batch_size": 10, "num_batches": 10},
            results=supplied_results,
        )
        assert stats.total_records == 100
        assert stats.processed_records == 80
        assert stats.failed_records == 20
        assert stats.start_time == 1620000000.0
        assert stats.end_time == 1620003600.0
        assert stats.processing_time == 3600.0
        assert stats.batch_info == {"batch_size": 10, "num_batches": 10}
        assert stats.results is supplied_results
        assert stats.results == supplied_results


# ===================================================================
# AsyncProcessingStats — __post_init__ validation
# ===================================================================
class TestAsyncProcessingStatsValidation:
    """Tests for AsyncProcessingStats.__post_init__() validation."""

    @pytest.mark.parametrize(
        ("stats_factory", "match"),
        [
            (
                lambda: AsyncProcessingStats(total_records=-1),
                r"Total records cannot be negative, got -1",
            ),
            (
                lambda: AsyncProcessingStats(processed_records=-5),
                r"Processed records cannot be negative, got -5",
            ),
            (
                lambda: AsyncProcessingStats(failed_records=-3),
                r"Failed records cannot be negative, got -3",
            ),
            (
                lambda: AsyncProcessingStats(processing_time=-0.01),
                r"Processing time cannot be negative, got -0.01",
            ),
        ],
        ids=[
            "negative_total_records",
            "negative_processed_records",
            "negative_failed_records",
            "negative_processing_time",
        ],
    )
    def test_negative_value_raises(
        self, stats_factory: Callable[[], AsyncProcessingStats], match: str
    ) -> None:
        """Test negative values raise ValueError with descriptive message."""
        with pytest.raises(ValueError, match=match):
            stats_factory()

    def test_zero_values_accepted(self) -> None:
        """Test zero boundary is valid (not negative)."""
        stats = AsyncProcessingStats(
            total_records=0,
            processed_records=0,
            failed_records=0,
            processing_time=0.0,
        )
        assert stats.total_records == 0
        assert stats.processed_records == 0
        assert stats.failed_records == 0
        assert stats.processing_time == 0.0


# ===================================================================
# AsyncProcessingStats — properties
# ===================================================================
class TestAsyncProcessingStatsProperties:
    """Tests for computed properties on AsyncProcessingStats."""

    # --- success_rate ---
    @pytest.mark.parametrize(
        ("total_records", "processed_records", "expected_rate"),
        [
            (0, 0, 0.0),  # avoid division by zero
            (100, 0, 0.0),
            (100, 50, 50.0),
            (100, 100, 100.0),
            (200, 150, 75.0),
        ],
    )
    def test_success_rate_with_records(
        self, total_records: int, processed_records: int, expected_rate: float
    ) -> None:
        """Test success_rate percentage calculation."""
        stats = AsyncProcessingStats(
            total_records=total_records, processed_records=processed_records
        )
        assert stats.success_rate == expected_rate

    # --- is_complete ---
    @pytest.mark.parametrize(
        ("end_time", "expected_complete"),
        [
            (None, False),
            (1060.0, True),
        ],
    )
    def test_is_complete(self, end_time: float | None, expected_complete: bool) -> None:
        """Test is_complete based on end time."""
        stats = AsyncProcessingStats(end_time=end_time)
        assert stats.is_complete == expected_complete

    # --- throughput ---
    @pytest.mark.parametrize(
        ("processed_records", "processing_time", "expected_throughput"),
        [
            (0, 1.0, 0.0),  # zero processed records
            (100, 0.0, 0.0),  # avoid division by zero
            (100, 10.0, 10.0),
            (150, 30.0, 5.0),
        ],
    )
    def test_throughput(
        self,
        processed_records: int,
        processing_time: float,
        expected_throughput: float,
    ) -> None:
        """Test throughput calculation."""
        stats = AsyncProcessingStats(
            processed_records=processed_records, processing_time=processing_time
        )
        assert stats.throughput == expected_throughput


# ===================================================================
# AsyncProcessingStats — summary()
# ===================================================================
class TestAsyncProcessingStatsSummary:
    """Tests for AsyncProcessingStats.summary() method."""

    def test_return_all_keys(self) -> None:
        """Test summary dict contains all expected keys."""
        stats = AsyncProcessingStats()
        result = stats.summary()
        expected_keys = {
            "total_records",
            "processed_records",
            "failed_records",
            "success_rate",
            "processing_time",
            "throughput",
            "is_complete",
            "start_time",
            "end_time",
        }
        assert set(result.keys()) == expected_keys

    def test_value_match_properties(self) -> None:
        """Test summary values are consistent with instance properties."""
        stats = AsyncProcessingStats(
            total_records=100,
            processed_records=80,
            failed_records=20,
            start_time=1620000000.0,
            end_time=1620003600.0,
            processing_time=3600.0,
        )
        summary = stats.summary()
        assert summary["total_records"] == 100
        assert summary["processed_records"] == 80
        assert summary["failed_records"] == 20
        assert summary["success_rate"] == 80.0
        assert summary["processing_time"] == 3600.0
        assert summary["throughput"] == (80 / 3600.0)
        assert summary["is_complete"] is True
        assert summary["start_time"] == 1620000000.0
        assert summary["end_time"] == 1620003600.0
