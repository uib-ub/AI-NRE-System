"""Shared fixtures and helpers for pipeline module tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypeAlias

import pytest

from ai_ner_system.processing import (
    BatchProcessingResult,
    EntityRecord,
    ProcessingResult,
)

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable, Iterable, Iterator

    from ai_ner_system.llm import BatchProgress


Record: TypeAlias = dict[str, str]
ProcessorResult: TypeAlias = tuple[list[str], list[str]]
ProcessorOutcome: TypeAlias = ProcessorResult | Exception
AsyncRecordOutcome: TypeAlias = ProcessingResult | Exception
AsyncBatchOutcome: TypeAlias = BatchProcessingResult | Exception


def _record_results_factory() -> dict[str, ProcessorOutcome]:
    """Create an empty per-record outcome mapping with concrete types."""
    return {}


def _batch_results_factory() -> dict[tuple[str, ...], ProcessorOutcome]:
    """Create an empty per-batch outcome mapping with concrete types."""
    return {}


def _record_calls_factory() -> list[Record]:
    """Create an empty record call history list with concrete types."""
    return []


def _batch_calls_factory() -> list[list[Record]]:
    """Create an empty batch call history list with concrete types."""
    return []


def _async_record_results_factory() -> dict[str, AsyncRecordOutcome]:
    """Create an empty async per-record outcome mapping with concrete types."""
    return {}


def _async_batch_results_factory() -> dict[tuple[str, ...], AsyncBatchOutcome]:
    """Create an empty async per-batch outcome mapping with concrete types."""
    return {}


def _writer_calls_factory() -> list[WriterCall]:
    """Create an empty writer call history list with concrete types."""
    return []


def _async_batch_calls_factory() -> list[FakeAsyncBatchCall]:
    """Create an empty async batch call history list with concrete types."""
    return []


def make_record(brevid: str, bindnr: str) -> Record:
    """Create a deterministic fake CSV record for pipeline tests."""
    return {
        "Bindnr": bindnr,
        "Brevid": brevid,
        "Tekst": f"Text for {brevid}",
    }


def make_entity_record(
    brevid: str,
    *,
    name: str | None = None,
    order: int = 1,
) -> EntityRecord:
    """Create a deterministic fake entity for async output tests."""
    return EntityRecord(
        name=name or f"Entity-{brevid}",
        entity_type="PERSON",
        preposition="N/A",
        order=order,
        brevid=brevid,
        description=f"Description for {brevid}",
        gender="N/A",
        language="la",
    )


def make_processing_result(
    brevid: str,
    bindnr: str,
    *,
    success: bool = True,
    annotated_text: str | None = None,
    entities: list[EntityRecord] | None = None,
    error_message: str | None = None,
    processing_time: float = 0.25,
) -> ProcessingResult:
    """Create a deterministic async ProcessingResult for tests."""
    return ProcessingResult(
        record_id=f"{bindnr}_{brevid}",
        brevid=brevid,
        annotated_text=annotated_text
        if annotated_text is not None
        else f"ann-{brevid}",
        entities=list(entities or []),
        processing_time=processing_time,
        success=success,
        error_message=error_message,
    )


def make_batch_processing_result(
    batch_num: int,
    results: list[ProcessingResult],
    *,
    total_processing_time: float = 1.0,
) -> BatchProcessingResult:
    """Create a deterministic async BatchProcessingResult for tests."""
    successful_count = sum(1 for result in results if result.success)
    failed_count = len(results) - successful_count
    return BatchProcessingResult(
        batch_id=f"batch_{batch_num}",
        results=list(results),
        total_processing_time=total_processing_time,
        successful_count=successful_count,
        failed_count=failed_count,
    )


@dataclass
class FakeReader:
    """Minimal CSV reader double for SyncProcessor tests.

    Yields the configured records in order and a fixed file path.

    Example usage:
        reader = FakeReader(records=[
            {"Bindnr": "1", "Brevid": "601", "Tekst": "abc"},
            {"Bindnr": "2", "Brevid": "602", "Tekst": "def"}
        ])

        reader.stream_records()  # yields the records in order
    """

    records: Iterable[Record]
    file_path: Path = Path("input/test-records.csv")

    def stream_records(self) -> Iterator[Record]:
        """Yield the configured records in order."""
        yield from self.records


@dataclass
class FakeWriter:
    """Writer double that records incremental output calls."""

    text_calls: list[WriterCall] = field(default_factory=_writer_calls_factory)
    metadata_calls: list[WriterCall] = field(default_factory=_writer_calls_factory)
    text_error: Exception | None = None
    metadata_error: Exception | None = None

    def append_text_output(
        self,
        file_path: str,
        header: str,
        annotation_lines: list[str],
    ) -> None:
        """Record appended annotation output or raise a configured failure."""
        if self.text_error is not None:
            raise self.text_error
        self.text_calls.append(
            WriterCall(
                file_path=file_path,
                header=header,
                rows=list(annotation_lines),
            ),
        )

    def append_metadata_output(
        self,
        file_path: str,
        header: str,
        metadata: list[str],
    ) -> None:
        """Record appended metadata output or raise a configured failure."""
        if self.metadata_error is not None:
            raise self.metadata_error
        self.metadata_calls.append(
            WriterCall(
                file_path=file_path,
                header=header,
                rows=list(metadata),
            ),
        )


@dataclass
class WriterCall:
    """Captured output-writer invocation."""

    file_path: str
    header: str
    rows: list[str]


@dataclass
class FakeLLMClient:
    """LLM client double with configurable async batch support."""

    supports_async_batch_value: bool = True

    def supports_async_batch(self) -> bool:
        """Return whether async batch processing is supported."""
        return self.supports_async_batch_value


@dataclass
class FakeAsyncBatchCall:
    """Captured async batch processor invocation."""

    records: list[Record]
    batch_num: int
    progress_callback: Callable[[BatchProgress], None] | None
    max_wait_time: float | None
    poll_interval: float | None


@dataclass
class FakeProcessor:
    """Configurable processing double for record and batch workflows."""

    record_results: dict[str, ProcessorOutcome] = field(
        default_factory=_record_results_factory,
    )
    batch_results: dict[tuple[str, ...], ProcessorOutcome] = field(
        default_factory=_batch_results_factory,
    )
    async_record_results: dict[str, AsyncRecordOutcome] = field(
        default_factory=_async_record_results_factory,
    )
    async_batch_results: dict[tuple[str, ...], AsyncBatchOutcome] = field(
        default_factory=_async_batch_results_factory,
    )
    record_calls: list[Record] = field(default_factory=_record_calls_factory)
    batch_calls: list[list[Record]] = field(default_factory=_batch_calls_factory)
    async_record_calls: list[Record] = field(default_factory=_record_calls_factory)
    async_batch_calls: list[FakeAsyncBatchCall] = field(
        default_factory=_async_batch_calls_factory,
    )

    def process_record(self, record: Record) -> ProcessorResult:
        """Return or raise the configured per-record outcome."""
        self.record_calls.append(record)
        brevid = record.get("Brevid", "unknown")
        outcome = self.record_results.get(
            brevid,
            ([f"ann-{brevid}"], [f"meta-{brevid}"]),
        )
        return self._resolve(outcome)

    def process_batch(self, records: list[Record]) -> ProcessorResult:
        """Return or raise the configured per-batch outcome."""
        self.batch_calls.append(list(records))
        batch_key = tuple(record.get("Brevid", "unknown") for record in records)
        outcome = self.batch_results.get(
            batch_key,
            (
                [f"ann-{brevid}" for brevid in batch_key],
                [f"meta-{brevid}" for brevid in batch_key],
            ),
        )
        return self._resolve(outcome)

    @staticmethod
    def _resolve(outcome: ProcessorOutcome) -> ProcessorResult:
        """Convert a configured outcome into a return value or exception."""
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def process_record_async(self, record: Record) -> ProcessingResult:
        """Return or raise the configured async per-record outcome."""
        self.async_record_calls.append(record)
        brevid = record.get("Brevid", "unknown")
        outcome = self.async_record_results.get(
            brevid,
            make_processing_result(brevid, record.get("Bindnr", "unknown")),
        )
        return self._resolve_async_record(outcome)

    async def process_batch_async(
        self,
        records: list[Record],
        batch_num: int,
        progress_callback: Callable[[BatchProgress], None] | None = None,
        max_wait_time: float | None = None,
        poll_interval: float | None = None,
    ) -> BatchProcessingResult:
        """Return or raise the configured async per-batch outcome."""
        self.async_batch_calls.append(
            FakeAsyncBatchCall(
                records=list(records),
                batch_num=batch_num,
                progress_callback=progress_callback,
                max_wait_time=max_wait_time,
                poll_interval=poll_interval,
            ),
        )
        batch_key = tuple(record.get("Brevid", "unknown") for record in records)
        outcome = self.async_batch_results.get(
            batch_key,
            make_batch_processing_result(
                batch_num,
                [
                    make_processing_result(
                        record.get("Brevid", "unknown"),
                        record.get("Bindnr", "unknown"),
                    )
                    for record in records
                ],
            ),
        )
        return self._resolve_async_batch(outcome)

    @staticmethod
    def _resolve_async_record(outcome: AsyncRecordOutcome) -> ProcessingResult:
        """Convert a configured async record outcome into a return value or exception."""
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @staticmethod
    def _resolve_async_batch(
        outcome: AsyncBatchOutcome,
    ) -> BatchProcessingResult:
        """Convert a configured async batch outcome into a return value or exception."""
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class FakeMainProcessor:
    """Protocol-shaped main processor double for SyncProcessor tests."""

    ANNOTATED_HEADER: ClassVar[str] = "Annotated Text"
    METADATA_HEADER: ClassVar[str] = "Metadata"

    args: Namespace
    reader: FakeReader
    writer: FakeWriter
    processor: FakeProcessor
    llm_client: FakeLLMClient = field(default_factory=FakeLLMClient)
    incremental_mode: bool = False
    output_text_file: str = "output/annotations.txt"
    output_table_file: str = "output/metadata.txt"
    output_stats_file: str = "output/stats.json"


@pytest.fixture
def sample_records() -> list[Record]:
    """Provide a small deterministic record set for pipeline tests."""
    return [
        make_record("B1", "1"),
        make_record("B2", "1"),
        make_record("B3", "2"),
        make_record("B4", "2"),
    ]
