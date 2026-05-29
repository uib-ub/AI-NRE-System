"""Unit tests for pipeline.sync_processor module."""

from __future__ import annotations

import logging
from argparse import Namespace
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, TypeAlias, cast

import pytest

from ai_ner_system.pipeline import sync_processor as sync_processor_module
from ai_ner_system.pipeline.stats import ApplicationError
from ai_ner_system.processing import (
    BatchProcessingError,
    LLMResponseError,
    ParseError,
    ProcessingError,
    ValidationError,
)

from .conftest import (
    FakeMainProcessor,
    FakeProcessor,
    FakeReader,
    FakeWriter,
    ProcessorOutcome,
    Record,
)

SyncProcessor: TypeAlias = sync_processor_module.SyncProcessor

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from types import TracebackType

    from ai_ner_system.pipeline.processor_protocol import ProcessorContext


log = logging.getLogger(__name__)


class SyncProcessorFactory(Protocol):
    """Callable fixture type for constructing sync processor test contexts.

    This factory creates a SyncProcessor instance along with a fake main processor
    context for testing. It accepts parameters to configure the test scenario, such
    as whether to use batch processing and the specific record and batch results to
    simulate. The returned SyncProcessor is ready to be invoked in test cases, and the
    FakeMainProcessor allows inspection of the processing context and outcomes.
    """

    def __call__(
        self,
        *,
        records: Iterable[Record] | None = None,
        use_batch: bool = False,
        batch_size: int = 1,
        record_results: dict[str, ProcessorOutcome] | None = None,
        batch_results: dict[tuple[str, ...], ProcessorOutcome] | None = None,
    ) -> tuple[SyncProcessor, FakeMainProcessor]:
        """Create a SyncProcessor and its fake main processor context."""
        ...


class UnexpectedProcessingError(ProcessingError):
    """Custom exception for simulating unexpected processing errors in tests."""


class SyncProcessorProbe(SyncProcessor):
    """Expose focused internal hooks needed for otherwise unreachable branches.

    This is used only for private branches that are hard to reach through
    process_all_records(), like _process_final_batch and _handle_individual_error.
    """

    def process_final_batch(
        self,
        batch_records: list[Record],
        batch_count: int,
        batch_size: int,
        processing_mode: Literal["batch", "individual"],
    ) -> tuple[list[str], list[str]]:
        """Expose _process_final_batch that is not reachable publicly."""
        return self._process_final_batch(
            batch_records,
            batch_count,
            batch_size,
            processing_mode,
        )

    def handle_individual_error(
        self,
        record: Record,
        exception: ProcessingError,
    ) -> None:
        """Expose _handle_individual_error that is not reachable publicly."""
        return self._handle_individual_error(record, exception)


class FailingStreamingSyncProcessor(SyncProcessor):
    """SyncProcessor variant that fails below the public entry-point boundary."""

    def _process_records_streaming(
        self,
        batch_size: int,
        processing_mode: Literal["batch", "individual"],
    ) -> tuple[list[str], list[str]]:
        assert batch_size == 3
        assert processing_mode == "batch"
        raise RuntimeError("boom")


class ApplicationErrorStreamingSyncProcessor(SyncProcessor):
    """Processor variant that raises an already-wrapped streaming error."""

    def __init__(
        self,
        main_processor: ProcessorContext,
        error: ApplicationError,
    ) -> None:
        super().__init__(main_processor)
        self.error = error

    def _process_records_streaming(
        self,
        batch_size: int,
        processing_mode: Literal["batch", "individual"],
    ) -> tuple[list[str], list[str]]:
        assert batch_size == 3
        assert processing_mode == "batch"
        raise self.error


class NoopTqdm:
    """Small tqdm stand-in for deterministic sync processor tests."""

    def __init__(self, iterable: Iterable[Any] | None = None, **_kwargs: Any) -> None:
        self.iterable = iterable

    def __iter__(self) -> Iterator[Any]:
        """Yield the wrapped iterable without rendering a progress bar."""
        return iter([] if self.iterable is None else self.iterable)

    def __enter__(self) -> Self:
        """Return self for ``with tqdm(...) as bar`` usage."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Do not suppress exceptions."""

    def set_description(self, _description: str) -> None:
        """Accept progress description updates without recording them."""

    def update(self, _amount: int) -> None:
        """Accept progress increments without recording them."""


@pytest.fixture
def make_sync_processor() -> SyncProcessorFactory:
    """Create a SyncProcessor with a protocol-shaped fake context.

    This fixture returns a factory function that constructs a SyncProcessor with a
    FakeMainProcessor which matches the expected protocol shape.
    The factory accepts parameters to customize the records, batch setting,
    and simulated processor outcomes for testing various scenarios.
    By using this factory, tests can easily set up a SyncProcessor with
    controlled behavior and dependencies.
    """

    def _make(
        *,
        records: Iterable[Record] | None = None,
        use_batch: bool = False,
        batch_size: int = 1,
        record_results: dict[str, ProcessorOutcome] | None = None,
        batch_results: dict[tuple[str, ...], ProcessorOutcome] | None = None,
    ) -> tuple[SyncProcessor, FakeMainProcessor]:
        context = FakeMainProcessor(
            args=Namespace(use_batch=use_batch, batch_size=batch_size),
            reader=FakeReader(records=[] if records is None else records),
            writer=FakeWriter(),
            processor=FakeProcessor(
                record_results=dict(record_results or {}),
                batch_results=dict(batch_results or {}),
            ),
        )
        return SyncProcessor(cast("ProcessorContext", context)), context

    return _make


@pytest.fixture
def deterministic_sync_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disable progress rendering and real sleeps in sync processor tests.

    patches:
    - tqdm to NoopTqdm to avoid rendering progress bars during tests, ensuring deterministic behavior.
    - logging_redirect_tqdm to nullcotext.
    - BATCH_PROCESSING_DELAY to 0 to eliminate artificial delays in batch processing tests.
    """
    monkeypatch.setattr(sync_processor_module, "tqdm", NoopTqdm)
    monkeypatch.setattr(sync_processor_module, "logging_redirect_tqdm", nullcontext)
    monkeypatch.setattr(sync_processor_module, "BATCH_PROCESSING_DELAY", 0)


def test_constructor_forwards_context_dependencies(
    make_sync_processor: SyncProcessorFactory,
) -> None:
    """Test SyncProcessor stores and exposes the supplied context dependencies.

    This test verifies that the processor keeps access to the expected main processor
    dependencies.
    """
    # Create a SyncProcessor with a custom context using the factory fixture
    sync_processor, context = make_sync_processor()
    expected_main_processor: object = context
    expected_args: object = context.args
    expected_reader: object = context.reader
    expected_writer: object = context.writer
    expected_processor: object = context.processor

    assert sync_processor.main_processor is expected_main_processor
    assert sync_processor.args is expected_args
    assert sync_processor.reader is expected_reader
    assert sync_processor.writer is expected_writer
    assert sync_processor.processor is expected_processor


def test_process_all_records_processes_individual_records(
    make_sync_processor: SyncProcessorFactory,
    sample_records: list[Record],
) -> None:
    """Test individual mode processes each streamed record independently.

    This test verifies that when batch processing is disabled, the SyncProcessor
    processes each record one by one, and that the expected annotations and metadata are
    returned, and that the processor's record_calls reflect the individual processing.
    """
    sync_processor, context = make_sync_processor(
        records=sample_records[:2],
        use_batch=False,
        batch_size=10,
    )

    annotations, metadata = sync_processor.process_all_records()

    log.debug("Annotations: %s", annotations)
    log.debug("Metadata: %s", metadata)
    for batch_call in context.processor.batch_calls:
        log.debug("Batch call: %s", batch_call)
    for record_call in context.processor.record_calls:
        log.debug("Record call: %s", record_call)

    assert annotations == ["ann-B1", "ann-B2"]
    assert metadata == ["meta-B1", "meta-B2"]
    assert context.processor.record_calls == sample_records[:2]
    assert context.processor.batch_calls == []


def test_process_all_records_consumes_one_shot_generator(
    make_sync_processor: SyncProcessorFactory,
) -> None:
    """Test streaming does not pre-load or iterate records before processing begins."""
    iteration_allowed = False

    def one_shot_records() -> Iterator[Record]:
        nonlocal iteration_allowed
        assert iteration_allowed, "records iterable was consumed before processing"
        yield {"Bindnr": "1", "Brevid": "G1", "Tekst": "Text for G1"}
        yield {"Bindnr": "1", "Brevid": "G2", "Tekst": "Text for G2"}

    sync_processor, context = make_sync_processor(records=one_shot_records())
    iteration_allowed = True

    annotations, metadata = sync_processor.process_all_records()

    log.debug("Annotations: %s", annotations)
    log.debug("Metadata: %s", metadata)

    for record_call in context.processor.record_calls:
        log.debug("Record call: %s", record_call)

    assert annotations == ["ann-G1", "ann-G2"]
    assert metadata == ["meta-G1", "meta-G2"]
    assert [record["Brevid"] for record in context.processor.record_calls] == [
        "G1",
        "G2",
    ]


def test_process_all_records_applies_configured_batch_delay(
    make_sync_processor: SyncProcessorFactory,
    sample_records: list[Record],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test configured sync delay is applied after each processed full batch."""
    sleep_calls: list[float] = []
    monkeypatch.setattr(SyncProcessor, "BATCH_PROCESSING_DELAY", 0.05)
    monkeypatch.setattr(sync_processor_module.time, "sleep", sleep_calls.append)
    sync_processor, _ = make_sync_processor(records=sample_records[:2])

    annotations, metadata = sync_processor.process_all_records()

    log.debug("Annotations: %s", annotations)
    log.debug("Metadata: %s", metadata)

    assert annotations == ["ann-B1", "ann-B2"]
    assert metadata == ["meta-B1", "meta-B2"]
    assert sleep_calls == [0.05, 0.05]


def test_process_all_records_batches_records_and_final_partial_batch(
    make_sync_processor: SyncProcessorFactory,
    sample_records: list[Record],
) -> None:
    """Test batch mode groups full batches and processes a final partial batch."""
    # 3 records with batch size 2 should yield one full batch and one final partial batch
    sync_processor, context = make_sync_processor(
        records=sample_records[:3],
        use_batch=True,
        batch_size=2,
    )

    for record in sample_records[:3]:
        log.debug("Record: %s", record)

    annotations, metadata = sync_processor.process_all_records()

    log.debug("Annotations: %s", annotations)
    log.debug("Metadata: %s", metadata)

    for batch_call in context.processor.batch_calls:
        log.debug("Batch call: %s", batch_call)
    for record_call in context.processor.record_calls:
        log.debug("Record call: %s", record_call)

    assert annotations == ["ann-B1", "ann-B2", "ann-B3"]
    assert metadata == ["meta-B1", "meta-B2", "meta-B3"]
    assert context.processor.batch_calls == [
        sample_records[:2],
        sample_records[2:3],
    ]


def test_process_all_records_falls_back_when_final_partial_batch_fails(
    make_sync_processor: SyncProcessorFactory,
    sample_records: list[Record],
) -> None:
    """Test a failed final partial batch falls back to per-record processing."""
    sync_processor, context = make_sync_processor(
        records=sample_records[:3],
        use_batch=True,
        batch_size=2,
        batch_results={("B3",): BatchProcessingError("final batch failed")},
    )

    annotations, metadata = sync_processor.process_all_records()

    log.debug("Annotations: %s", annotations)
    log.debug("Metadata: %s", metadata)
    for batch_call in context.processor.batch_calls:
        log.debug("Batch call: %s", batch_call)
    for record_call in context.processor.record_calls:
        log.debug("Record call: %s", record_call)

    assert annotations == ["ann-B1", "ann-B2", "ann-B3"]
    assert metadata == ["meta-B1", "meta-B2", "meta-B3"]
    assert context.processor.batch_calls == [sample_records[:2], sample_records[2:3]]
    assert context.processor.record_calls == [sample_records[2]]


def test_process_all_records_falls_back_when_batch_processing_fails(
    make_sync_processor: SyncProcessorFactory,
    sample_records: list[Record],
) -> None:
    """Test failed sync batch processing falls back to per-record processing."""
    sync_processor, context = make_sync_processor(
        records=sample_records[:2],
        use_batch=True,
        batch_size=2,
        batch_results={("B1", "B2"): BatchProcessingError("batch failed")},
    )

    annotations, metadata = sync_processor.process_all_records()

    log.debug("Annotations: %s", annotations)
    log.debug("Metadata: %s", metadata)
    for batch_call in context.processor.batch_calls:
        log.debug("Batch call: %s", batch_call)
    for record_call in context.processor.record_calls:
        log.debug("Record call: %s", record_call)

    assert annotations == ["ann-B1", "ann-B2"]
    assert metadata == ["meta-B1", "meta-B2"]
    assert context.processor.batch_calls == [sample_records[:2]]
    assert context.processor.record_calls == sample_records[:2]


def test_process_all_records_skips_records_that_fail_during_batch_fallback(
    caplog: pytest.LogCaptureFixture,
    make_sync_processor: SyncProcessorFactory,
    sample_records: list[Record],
) -> None:
    """Test failed sync batch processing falls back to per-record processing."""
    sync_processor, context = make_sync_processor(
        records=sample_records[:2],
        use_batch=True,
        batch_size=2,
        batch_results={("B1", "B2"): BatchProcessingError("batch failed")},
        record_results={"B1": ValidationError("fallback failed")},
    )

    with caplog.at_level(logging.ERROR):
        annotations, metadata = sync_processor.process_all_records()

    log.debug("Annotations: %s", annotations)
    log.debug("Metadata: %s", metadata)
    for batch_call in context.processor.batch_calls:
        log.debug("Batch call: %s", batch_call)
    for record_call in context.processor.record_calls:
        log.debug("Record call: %s", record_call)

    log.debug("Captured log records: %s", caplog.text)

    assert annotations == ["ann-B2"]
    assert metadata == ["meta-B2"]
    assert context.processor.batch_calls == [sample_records[:2]]
    assert context.processor.record_calls == sample_records[:2]
    assert "Validation error for Brevid B1 (Bindnr: 1): fallback failed" in caplog.text


def test_process_all_records_does_not_log_full_individual_record(
    caplog: pytest.LogCaptureFixture,
    make_sync_processor: SyncProcessorFactory,
    sample_records: list[Record],
) -> None:
    """Test individual processing logs identifiers without emitting raw text."""
    sync_processor, _ = make_sync_processor(records=sample_records[:1])

    with caplog.at_level(logging.INFO):
        annotations, metadata = sync_processor.process_all_records()

    log.debug("Annotations: %s", annotations)
    log.debug("Metadata: %s", metadata)
    log.debug("Captured log records: %s", caplog.text)

    assert annotations == ["ann-B1"]
    assert metadata == ["meta-B1"]
    assert "Processing Record (Brevid: B1)" in caplog.text
    assert sample_records[0]["Tekst"] not in caplog.text
    assert str(sample_records[0]) not in caplog.text


def test_process_all_records_skips_failed_individual_records(
    caplog: pytest.LogCaptureFixture,
    make_sync_processor: SyncProcessorFactory,
    sample_records: list[Record],
) -> None:
    """Test individual ProcessingError is logged, skipped, and does not stop later records."""
    sync_processor, context = make_sync_processor(
        records=sample_records[:2],
        record_results={"B1": ValidationError("missing text")},
    )

    with caplog.at_level(logging.ERROR):
        annotations, metadata = sync_processor.process_all_records()

    log.debug("Annotations: %s", annotations)
    log.debug("Metadata: %s", metadata)
    log.debug("Captured log records: %s", caplog.text)

    assert annotations == ["ann-B2"]
    assert metadata == ["meta-B2"]
    assert context.processor.record_calls == sample_records[:2]
    assert "Validation error for Brevid B1 (Bindnr: 1): missing text" in caplog.text


def test_process_all_records_wraps_public_entry_point_failures(
    make_sync_processor: SyncProcessorFactory,
) -> None:
    """Test the public entry point wraps unexpected non-streaming failures."""
    sync_processor, _ = make_sync_processor(use_batch=True, batch_size=3)
    failing_processor = FailingStreamingSyncProcessor(sync_processor.main_processor)

    with pytest.raises(
        ApplicationError,
        match="Critical error during file processing",
    ) as exc_info:
        failing_processor.process_all_records()

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_process_all_records_reraises_streaming_application_errors(
    make_sync_processor: SyncProcessorFactory,
) -> None:
    """Test existing streaming ApplicationError is re-raised without wrapping."""
    expected_error = ApplicationError("streaming already wrapped")
    sync_processor, _ = make_sync_processor(use_batch=True, batch_size=3)
    failing_processor = ApplicationErrorStreamingSyncProcessor(
        sync_processor.main_processor,
        expected_error,
    )

    with pytest.raises(ApplicationError, match="streaming already wrapped") as exc_info:
        failing_processor.process_all_records()

    log.debug("Captured exception: %s", exc_info.value)

    assert exc_info.value is expected_error


def test_process_all_records_wraps_unexpected_streaming_errors(
    make_sync_processor: SyncProcessorFactory,
    sample_records: list[Record],
) -> None:
    """Test unexpected streaming failures are wrapped with batch-count context."""
    sync_processor, _ = make_sync_processor(
        records=sample_records[:1],
        record_results={"B1": RuntimeError("boom")},
    )

    with pytest.raises(
        ApplicationError,
        match="Streaming processing failed after 1 batches",
    ) as exc_info:
        sync_processor.process_all_records()

    log.debug("Captured exception: %s", exc_info.value)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "boom"


def test_process_final_batch_handles_individual_records(
    make_sync_processor: SyncProcessorFactory,
    sample_records: list[Record],
) -> None:
    """Test final individual-batch helper processes and tracks each record."""
    sync_processor, context = make_sync_processor(records=sample_records[:2])
    sync_processor_probe = SyncProcessorProbe(sync_processor.main_processor)

    annotations, metadata = sync_processor_probe.process_final_batch(
        sample_records[:2],
        batch_count=1,
        batch_size=1,
        processing_mode="individual",
    )

    log.debug("Annotations: %s", annotations)
    log.debug("Metadata: %s", metadata)
    for record_call in context.processor.record_calls:
        log.debug("Record call: %s", record_call)

    assert annotations == ["ann-B1", "ann-B2"]
    assert metadata == ["meta-B1", "meta-B2"]
    assert context.processor.record_calls == sample_records[:2]


def test_process_final_batch_skips_failed_individual_records(
    caplog: pytest.LogCaptureFixture,
    make_sync_processor: SyncProcessorFactory,
    sample_records: list[Record],
) -> None:
    """Test final individual-batch processing skips records with ProcessingError."""
    sync_processor, context = make_sync_processor(
        records=sample_records[:2],
        record_results={"B1": ValidationError("final record failed")},
    )
    sync_processor_probe = SyncProcessorProbe(sync_processor.main_processor)

    with caplog.at_level(logging.ERROR):
        annotations, metadata = sync_processor_probe.process_final_batch(
            sample_records[:2],
            batch_count=1,
            batch_size=1,
            processing_mode="individual",
        )

    log.debug("Annotations: %s", annotations)
    log.debug("Metadata: %s", metadata)
    log.debug("Captured log records: %s", caplog.text)

    assert annotations == ["ann-B2"]
    assert metadata == ["meta-B2"]
    assert context.processor.record_calls == sample_records[:2]
    assert "Validation error for Brevid B1 (Bindnr: 1): final record failed" in (
        caplog.text
    )


@pytest.mark.parametrize(
    ("exception", "expected_log"),
    [
        (
            ValidationError("missing field"),
            "Validation error for Brevid B1 (Bindnr: 1): missing field",
        ),
        (
            LLMResponseError("bad llm response"),
            "LLM Processing error for Brevid B1 (Bindnr: 1): bad llm response",
        ),
        (
            ParseError("parse failed"),
            "LLM Processing error for Brevid B1 (Bindnr: 1): parse failed",
        ),
        (
            BatchProcessingError("batch failed"),
            "LLM Processing error for Brevid B1 (Bindnr: 1): batch failed",
        ),
        (
            UnexpectedProcessingError("unexpected failure"),
            "Unexpected error processing Brevid B1 (Bindnr: 1): unexpected failure",
        ),
    ],
    ids=["validation", "llm_response", "parse", "batch_processing", "unexpected"],
)
def test_handle_individual_error_uses_expected_log_category(
    caplog: pytest.LogCaptureFixture,
    make_sync_processor: SyncProcessorFactory,
    sample_records: list[Record],
    exception: ProcessingError,
    expected_log: str,
) -> None:
    """Test individual error logging keeps the coarse error categories stable."""
    sync_processor, _ = make_sync_processor()
    sync_processor_probe = SyncProcessorProbe(sync_processor.main_processor)

    with caplog.at_level(logging.ERROR):
        sync_processor_probe.handle_individual_error(sample_records[0], exception)

    assert expected_log in caplog.text
