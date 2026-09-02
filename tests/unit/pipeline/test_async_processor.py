"""Unit tests for pipeline.async_processor module."""

from __future__ import annotations

import asyncio
import logging
from argparse import Namespace
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

import pytest

from ai_ner_system.config import Settings
from ai_ner_system.llm import BatchProgress, BatchStatus
from ai_ner_system.pipeline import async_processor as async_processor_module
from ai_ner_system.pipeline.stats import (
    ApplicationError,
    AsyncProcessingStats,
    FailedBatchInfo,
)

from .conftest import (
    AsyncBatchOutcome,
    AsyncRecordOutcome,
    FakeLLMClient,
    FakeMainProcessor,
    FakeProcessor,
    FakeReader,
    FakeWriter,
    make_batch_processing_result,
    make_entity_record,
    make_processing_result,
    make_record,
)

AsyncProcessor: TypeAlias = async_processor_module.AsyncProcessor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable

    from pytest_mock import MockerFixture

    from ai_ner_system.pipeline.processor_protocol import ProcessorContext
    from ai_ner_system.processing import BatchProcessingResult, ProcessingResult

    from .conftest import Record

log = logging.getLogger(__name__)


def _batch_result_for_records(
    batch_num: int,
    records: list[Record],
) -> BatchProcessingResult:
    """Build a batch result that preserves the supplied record order."""
    return make_batch_processing_result(
        batch_num,
        [
            make_processing_result(record["Brevid"], record["Bindnr"])
            for record in records
        ],
    )


async def _yield_records(records: list[Record]) -> AsyncIterator[Record]:
    """Yield the supplied records one by one to create a deterministic async record stream."""
    for record in records:
        yield record


def patch_async_records(
    monkeypatch: pytest.MonkeyPatch,
    async_processor: AsyncProcessor,
    records: list[Record],
) -> None:
    """Monkeypatch the async CSV record generator to yield specified records.

    This patch can be used to test processing algorithm.
    """
    monkeypatch.setattr(
        async_processor, "_async_stream_csv_records", lambda: _yield_records(records)
    )


def patch_to_thread_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Any, tuple[Any, ...]]]:
    """Run asyncio.to_thread calls inline and return captured call details.

    This patch can be used to test _async_stream_csv_records() or incremental writing itself.
    """
    calls: list[tuple[Any, tuple[Any, ...]]] = []

    async def _to_thread(function: Any, *args: Any) -> Any:
        calls.append((function, args))
        return function(*args)

    monkeypatch.setattr(async_processor_module.asyncio, "to_thread", _to_thread)
    return calls


class AsyncProcessorProbe(AsyncProcessor):
    """Expose focused AsyncProcessor hooks used by async pipeline unit tests."""

    @property
    def next_expected_batch_num(self) -> int:
        """Expose queued incremental output order state."""
        return self._next_expected_batch_num

    @property
    def batch_result_queue(self) -> dict[int, BatchProcessingResult]:
        """Expose queued batch results for order-preservation tests."""
        return self._batch_result_queue

    @batch_result_queue.setter
    def batch_result_queue(self, value: dict[int, BatchProcessingResult]) -> None:
        """Seed queued batch results without direct private access in tests."""
        self._batch_result_queue = value

    async def process_records_streaming_async(
        self,
        stats: AsyncProcessingStats,
        progress_callback: Callable[[BatchProgress], None] | None,
        max_wait_time: float,
        poll_interval: float,
    ) -> None:
        """Expose streaming batch processing for branch-level async tests."""
        await self._process_records_streaming_async(
            stats,
            progress_callback,
            max_wait_time,
            poll_interval,
        )

    async def async_stream_csv_records(self) -> list[Record]:
        """Consume the async CSV adapter and return its yielded records."""
        return [record async for record in self._async_stream_csv_records()]

    async def process_batch_with_order_async(
        self,
        batch_records: list[Record],
        batch_num: int,
        progress_callback: Callable[[BatchProgress], None] | None,
        max_wait_time: float,
        poll_interval: float,
    ) -> BatchProcessingResult:
        """Expose ordered batch processing for fallback and cancellation tests."""
        return await self._process_batch_with_order_async(
            batch_records,
            batch_num,
            progress_callback,
            max_wait_time,
            poll_interval,
        )

    async def collect_completed_batch_results_async(
        self,
        stats: AsyncProcessingStats,
        batch_tasks: dict[int, asyncio.Task[BatchProcessingResult]],
        completed_batch_results: dict[int, BatchProcessingResult],
        next_batch_num_to_add: int,
    ) -> int:
        """Expose completed-task collection for ordered flush edge cases."""
        return await self._collect_completed_batch_results_async(
            stats,
            batch_tasks,
            completed_batch_results,
            next_batch_num_to_add,
        )

    async def add_batch_results_in_order(
        self,
        stats: AsyncProcessingStats,
        batch_result: BatchProcessingResult,
        batch_num: int,
    ) -> None:
        """Expose ordered result accumulation for standard/incremental modes."""
        await self._add_batch_results_in_order(stats, batch_result, batch_num)

    async def flush_queued_batch_results_async(
        self,
        stats: AsyncProcessingStats,
    ) -> None:
        """Expose queued incremental batch flushing."""
        await self._flush_queued_batch_results_async(stats)

    async def write_batch_results_incremental_async(
        self, batch_result: BatchProcessingResult, batch_num: int
    ) -> None:
        """Expose incremental output writes for writer behavior tests."""
        await self._write_batch_results_incremental_async(batch_result, batch_num)

    async def process_records_individual_async(
        self,
        stats: AsyncProcessingStats,
    ) -> None:
        """Expose individual streaming processing for concurrency tests."""
        await self._process_records_individual_async(stats)

    async def process_task_chunk(
        self,
        tasks: list[asyncio.Task[ProcessingResult]],
        chunk_records: list[dict[str, str]],
        stats: AsyncProcessingStats,
    ) -> None:
        """Expose task chunk normalization for result conversion tests."""
        await self._process_task_chunk(tasks, chunk_records, stats)

    async def fallback_to_individual_async_streaming(
        self,
        batch_records: list[dict[str, str]],
        stats: AsyncProcessingStats,
    ) -> None:
        """Expose batch fallback logic for ordering and cancellation tests."""
        await self._fallback_to_individual_async_streaming(batch_records, stats)

    def create_batch_progress_callback(
        self,
        batch_num: int,
        total_batches: int | None,
        user_callback: Callable[[BatchProgress], None] | None = None,
    ) -> Callable[[BatchProgress], None]:
        """Expose progress callback creation for callback behavior tests."""
        return self._create_batch_progress_callback(
            batch_num,
            total_batches,
            user_callback,
        )

    def create_failed_result(
        self,
        brevid: str,
        error: Exception,
        bindnr: str = "unknown",
    ) -> ProcessingResult:
        """Expose failed-result factory for consistency tests."""
        return self._create_failed_result(brevid, error, bindnr)


class AsyncProcessorProbeFactory(Protocol):
    """Callable fixture type for constructing async processor test contexts."""

    def __call__(
        self,
        *,
        records: Iterable[Record] | None = None,
        batch_size: int = 1,
        max_concurrent_batches: int | None = None,
        max_concurrent_individual: int | None = None,
        fallback_concurrency: int | None = None,
        chunk_size: int | None = None,
        supports_async_batch: bool = True,
        incremental_mode: bool = False,
        async_record_results: dict[str, AsyncRecordOutcome] | None = None,
        async_batch_results: dict[tuple[str, ...], AsyncBatchOutcome] | None = None,
        writer: FakeWriter | None = None,
    ) -> tuple[AsyncProcessorProbe, FakeMainProcessor]:
        """Create an AsyncProcessorProbe and its fake main processor context."""
        ...


@pytest.fixture
def make_async_processor() -> AsyncProcessorProbeFactory:
    """Create an AsyncProcessor probe with a protocol-shaped fake context."""

    def _make(
        *,
        records: Iterable[Record] | None = None,
        batch_size: int = 1,
        max_concurrent_batches: int | None = None,
        max_concurrent_individual: int | None = None,
        fallback_concurrency: int | None = None,
        chunk_size: int | None = None,
        supports_async_batch: bool = True,
        incremental_mode: bool = False,
        async_record_results: dict[str, AsyncRecordOutcome] | None = None,
        async_batch_results: dict[tuple[str, ...], AsyncBatchOutcome] | None = None,
        writer: FakeWriter | None = None,
    ) -> tuple[AsyncProcessorProbe, FakeMainProcessor]:
        args_kwargs: dict[str, Any] = {"batch_size": batch_size}
        if max_concurrent_batches is not None:
            args_kwargs["max_concurrent_batches"] = max_concurrent_batches
        if max_concurrent_individual is not None:
            args_kwargs["max_concurrent_individual"] = max_concurrent_individual
        if fallback_concurrency is not None:
            args_kwargs["fallback_concurrency"] = fallback_concurrency
        if chunk_size is not None:
            args_kwargs["chunk_size"] = chunk_size

        context = FakeMainProcessor(
            args=Namespace(**args_kwargs),
            reader=FakeReader(records=[] if records is None else records),
            writer=FakeWriter() if writer is None else writer,
            processor=FakeProcessor(
                async_record_results=dict(async_record_results or {}),
                async_batch_results=dict(async_batch_results or {}),
            ),
            llm_client=FakeLLMClient(
                supports_async_batch_value=supports_async_batch,
            ),
            incremental_mode=incremental_mode,
        )
        return AsyncProcessorProbe(cast("ProcessorContext", context)), context

    return _make


class TestAsyncProcessorConstruction:
    """Tests for AsyncProcessor construction and forwarded properties."""

    def test_init_sets_queue_state_and_forwards_properties(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
    ) -> None:
        """Test __init__ stores queue state and forwards context properties."""
        async_processor, context = make_async_processor()
        expected_main_processor: object = context
        expected_reader: object = context.reader
        expected_writer: object = context.writer
        expected_processor: object = context.processor
        expected_llm_client: object = context.llm_client

        assert async_processor.main_processor is expected_main_processor
        assert async_processor.args is context.args
        assert async_processor.reader is expected_reader
        assert async_processor.writer is expected_writer
        assert async_processor.processor is expected_processor
        assert async_processor.llm_client is expected_llm_client
        assert async_processor.next_expected_batch_num == 1
        assert async_processor.batch_result_queue == {}


class TestProcessAllRecordsAsync:
    """Tests for the public async entry point."""

    @pytest.mark.asyncio
    async def test_process_all_records_async_uses_batch_mode(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test batch-capable configurations delegate to streaming batch processing."""
        async_processor, _ = make_async_processor(
            batch_size=3, supports_async_batch=True
        )

        def _progress_callback(_progress: BatchProgress) -> None:
            return None

        async def _capture_stream(
            stats: AsyncProcessingStats,
            callback: Any,
            wait_time: float,
            poll_interval: float,
        ) -> None:
            stats.total_records = 4
            stats.processed_records = 3
            stats.failed_records = 1
            assert callback is _progress_callback
            assert wait_time == 12.0
            assert poll_interval == 1.5

        stream = mocker.AsyncMock(side_effect=_capture_stream)
        individual = mocker.AsyncMock()
        monkeypatch.setattr(async_processor, "_process_records_streaming_async", stream)
        monkeypatch.setattr(
            async_processor,
            "_process_records_individual_async",
            individual,
        )
        monkeypatch.setattr(
            async_processor_module.time,
            "monotonic",
            MockMonotonic([10.0, 13.25]),
        )

        stats = await async_processor.process_all_records_async(
            progress_callback=_progress_callback,
            max_batch_wait_time=12.0,
            poll_interval=1.5,
        )

        assert stats.total_records == 4
        assert stats.processed_records == 3
        assert stats.failed_records == 1
        assert stats.start_time == 10.0
        assert stats.end_time == 13.25
        assert stats.processing_time == 3.25
        stream.assert_awaited_once()
        individual.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_process_all_records_async_uses_default_timeouts(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test omitted timeout options use configured async defaults."""
        async_processor, _ = make_async_processor(
            batch_size=2,
            supports_async_batch=True,
        )

        async def _capture_stream(
            _stats: AsyncProcessingStats,
            _callback: Any,
            wait_time: float,
            poll_interval: float,
        ) -> None:
            assert wait_time == Settings.DEFAULT_MAX_WAIT_TIME
            assert poll_interval == Settings.DEFAULT_POLL_INTERVAL

        monkeypatch.setattr(
            async_processor,
            "_process_records_streaming_async",
            mocker.AsyncMock(side_effect=_capture_stream),
        )
        monkeypatch.setattr(
            async_processor_module.time,
            "monotonic",
            MockMonotonic([11.0, 11.5]),
        )

        await async_processor.process_all_records_async()

    @pytest.mark.asyncio
    async def test_process_all_records_async_logs_partial_success_summary(
        self,
        caplog: pytest.LogCaptureFixture,
        make_async_processor: AsyncProcessorProbeFactory,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test a run that completed with failed batch writes emits a warning summary."""
        async_processor, _ = make_async_processor(
            batch_size=2,
            supports_async_batch=True,
        )

        async def _capture_stream(
            stats: AsyncProcessingStats,
            _callback: Any,
            _wait_time: float,
            _poll_interval: float,
        ) -> None:
            stats.failed_batch_writes.append(
                FailedBatchInfo(
                    batch_num=47,
                    record_ids=["1_B100", "1_B101"],
                    error_type="RuntimeError",
                    error_message="disk full",
                ),
            )

            stats.failed_batch_writes.append(
                FailedBatchInfo(
                    batch_num=89,
                    record_ids=["2_B500"],
                    error_type="OSError",
                    error_message="permission denied",
                )
            )

        stream = mocker.AsyncMock(side_effect=_capture_stream)
        monkeypatch.setattr(
            async_processor,
            "_process_records_streaming_async",
            stream,
        )
        monkeypatch.setattr(
            async_processor,
            "_process_records_individual_async",
            mocker.AsyncMock(),
        )
        monkeypatch.setattr(
            async_processor_module.time,
            "monotonic",
            MockMonotonic([0.0, 1.0]),
        )

        with caplog.at_level(logging.WARNING):
            stats = await async_processor.process_all_records_async()

        assert len(stats.failed_batch_writes) == 2
        warning_records = [
            record for record in caplog.records if record.levelno == logging.WARNING
        ]
        assert any(
            "Run completed with 2 failed batch write(s)" in rec.getMessage()
            and "[47, 89]" in rec.getMessage()
            and "3 record(s) missing" in rec.getMessage()
            for rec in warning_records
        )

    @pytest.mark.asyncio
    async def test_process_all_records_async_preserves_explicit_zero_timeouts(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test explicit ``0.0`` overrides are forwarded unchanged."""
        async_processor, _ = make_async_processor(
            batch_size=2,
            supports_async_batch=True,
        )

        async def _capture_stream(
            _stats: AsyncProcessingStats,
            _callback: Any,
            wait_time: float,
            poll_interval: float,
        ) -> None:
            assert wait_time == 0.0
            assert poll_interval == 0.0

        monkeypatch.setattr(
            async_processor,
            "_process_records_streaming_async",
            mocker.AsyncMock(side_effect=_capture_stream),
        )
        monkeypatch.setattr(
            async_processor_module.time,
            "monotonic",
            MockMonotonic([14.0, 14.0]),
        )

        await async_processor.process_all_records_async(
            max_batch_wait_time=0.0,
            poll_interval=0.0,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("batch_size", "supports_async_batch"),
        [(1, True), (4, False)],
        ids=["single-record-mode", "batch-not-supported"],
    )
    async def test_process_all_records_async_uses_individual_mode_when_needed(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        batch_size: int,
        supports_async_batch: bool,
    ) -> None:
        """Test non-batch configurations fall back to individual processing."""
        async_processor, _ = make_async_processor(
            batch_size=batch_size,
            supports_async_batch=supports_async_batch,
        )

        async def _capture_individual(
            stats: AsyncProcessingStats,
        ) -> None:
            stats.total_records = 2
            stats.processed_records = 2

        stream = mocker.AsyncMock()
        individual = mocker.AsyncMock(side_effect=_capture_individual)
        monkeypatch.setattr(
            async_processor,
            "_process_records_streaming_async",
            stream,
        )
        monkeypatch.setattr(
            async_processor,
            "_process_records_individual_async",
            individual,
        )
        monkeypatch.setattr(
            async_processor_module.time,
            "monotonic",
            MockMonotonic([20.0, 22.0]),
        )

        stats = await async_processor.process_all_records_async()

        assert stats.total_records == 2
        assert stats.processed_records == 2
        assert stats.processing_time == 2.0
        stream.assert_not_awaited()
        individual.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_all_records_async_requires_initialized_components(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
    ) -> None:
        """Test missing reader or processor raises ApplicationError immediately."""
        async_processor, context = make_async_processor()
        context.reader = None  # type: ignore[assignment]

        with pytest.raises(
            ApplicationError,
            match="Components not properly initialized for async processing",
        ):
            await async_processor.process_all_records_async()

    @pytest.mark.asyncio
    async def test_process_all_records_async_propagates_cancellation_and_finalizes_stats(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test cancellation is re-raised and still updates final timing stats."""
        async_processor, _ = make_async_processor(batch_size=2)
        captured_stats: list[AsyncProcessingStats] = []

        async def _cancel(stats: AsyncProcessingStats, *_args: Any) -> None:
            captured_stats.append(stats)
            raise asyncio.CancelledError

        monkeypatch.setattr(
            async_processor,
            "_process_records_streaming_async",
            mocker.AsyncMock(side_effect=_cancel),
        )
        monkeypatch.setattr(
            async_processor_module.time,
            "monotonic",
            MockMonotonic([30.0, 33.5]),
        )

        with pytest.raises(asyncio.CancelledError):
            await async_processor.process_all_records_async()

        assert len(captured_stats) == 1
        assert captured_stats[0].end_time == 33.5
        assert captured_stats[0].processing_time == 3.5

    @pytest.mark.asyncio
    async def test_process_all_records_async_wraps_timeout_failures(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test unexpected streaming failures are wrapped in ApplicationError."""
        async_processor, _ = make_async_processor(batch_size=2)
        monkeypatch.setattr(
            async_processor,
            "_process_records_streaming_async",
            mocker.AsyncMock(side_effect=TimeoutError("took too long")),
        )
        monkeypatch.setattr(
            async_processor_module.time,
            "monotonic",
            MockMonotonic([40.0, 44.0]),
        )

        with pytest.raises(
            ApplicationError,
            match="Async streaming processing failed",
        ) as exc_info:
            await async_processor.process_all_records_async()

        log.debug("Captured exception: %s", exc_info.value)
        assert isinstance(exc_info.value.__cause__, TimeoutError)


class TestStreamingBatchProcessing:
    """Tests for async streaming batch workflows."""

    @pytest.mark.asyncio
    async def test_process_records_streaming_async_batches_records_and_updates_stats(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test streaming creates ordered batches, including a final partial batch."""
        async_processor, _ = make_async_processor(
            batch_size=2,
            max_concurrent_batches=2,
        )
        records = sample_records[:3]
        stats = AsyncProcessingStats()
        batch_calls: list[tuple[int, list[str]]] = []

        async def _process_batch(
            batch_records: list[Record],
            batch_num: int,
            *_args: Any,
        ) -> BatchProcessingResult:
            batch_calls.append(
                (batch_num, [record["Brevid"] for record in batch_records]),
            )
            if batch_num == 1:
                await asyncio.sleep(0.01)  # Simulate a longer-running batch
            return _batch_result_for_records(batch_num, batch_records)

        patch_async_records(monkeypatch, async_processor, records)

        monkeypatch.setattr(
            async_processor,
            "_process_batch_with_order_async",
            _process_batch,
        )

        await async_processor.process_records_streaming_async(
            stats,
            progress_callback=None,
            max_wait_time=12.0,
            poll_interval=1.0,
        )

        assert batch_calls == [(1, ["B1", "B2"]), (2, ["B3"])]
        assert stats.total_records == 3
        assert stats.processed_records == 3
        assert stats.failed_records == 0
        assert [result.brevid for result in stats.results] == ["B1", "B2", "B3"]

    @pytest.mark.asyncio
    async def test_process_records_streaming_async_harvests_completed_batches(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test out-of-order completions are backpressured until ordering catches up."""
        async_processor, _ = make_async_processor(
            batch_size=1,
            max_concurrent_batches=2,
        )
        stats = AsyncProcessingStats()
        # control when batch 1 may finish
        release_batch_one = asyncio.Event()
        # batch 2 tells the test that it finished first
        batch_two_finished = asyncio.Event()
        # detect whether the processor starts batch 3 too early
        batch_three_started = asyncio.Event()
        started_batches: list[int] = []

        # the controlled batch processor
        async def _process_batch(
            batch_records: list[Record],
            batch_num: int,
            *_args: Any,
        ) -> BatchProcessingResult:
            started_batches.append(batch_num)

            if batch_num == 1:
                await release_batch_one.wait()
            elif batch_num == 2:
                batch_two_finished.set()
            elif batch_num == 3:
                batch_three_started.set()
            return _batch_result_for_records(batch_num, batch_records)

        patch_async_records(monkeypatch, async_processor, sample_records[:4])

        # replaces the CSV-streaming method with an in-memory async generator
        monkeypatch.setattr(
            async_processor,
            "_process_batch_with_order_async",
            _process_batch,
        )

        processing_task = asyncio.create_task(
            async_processor.process_records_streaming_async(
                stats,
                progress_callback=None,
                max_wait_time=12.0,
                poll_interval=1.0,
            ),
        )

        # waits until batch 2 announces that it finished.
        await asyncio.wait_for(batch_two_finished.wait(), timeout=1.0)
        """
        sleep(0) does not introduce a meaningful time delay.
        It yields control to the event loop for one scheduling cycle.
        That gives the streaming task an opportunity to:
        1. notice that batch 2 completed;
        2. remove batch 2 from batch_tasks;
        3. put its result in completed_batch_results;
        4. discover that batch 2 cannot be appended because batch 1 is missing;
        5. wait for batch 1 instead of reading record 3.
        """
        await asyncio.sleep(0)

        assert not batch_three_started.is_set()
        assert started_batches == [1, 2]

        # release batch 1
        release_batch_one.set()
        await asyncio.wait_for(processing_task, timeout=1.0)
        # batch 3 was eventually allowed to start
        assert batch_three_started.is_set()
        assert started_batches == [1, 2, 3, 4]
        assert [result.brevid for result in stats.results] == ["B1", "B2", "B3", "B4"]

    @pytest.mark.asyncio
    async def test_collect_completed_batch_results_async_flushes_ready_prefix_before_later_failure(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
    ) -> None:
        """Test a later failure does not discard an earlier success from the same wait."""
        async_processor, _ = make_async_processor(incremental_mode=False)
        stats = AsyncProcessingStats()
        completed_batch_results: dict[int, BatchProcessingResult] = {}

        async def _succeed() -> BatchProcessingResult:
            return _batch_result_for_records(1, sample_records[:1])

        async def _fail() -> BatchProcessingResult:
            raise RuntimeError("batch 2 exploded")

        # scheduling two coroutines
        first_task = asyncio.create_task(_succeed())
        second_task = asyncio.create_task(_fail())
        await asyncio.sleep(0)  # yield to let tasks start

        batch_tasks = {1: first_task, 2: second_task}
        collector = async_processor.collect_completed_batch_results_async

        with pytest.raises(RuntimeError, match="batch 2 exploded"):
            await collector(
                stats,
                batch_tasks,
                completed_batch_results,
                next_batch_num_to_add=1,
            )

        assert batch_tasks == {}
        assert completed_batch_results == {}
        assert stats.processed_records == 1
        assert stats.failed_records == 0
        assert [result.brevid for result in stats.results] == ["B1"]

    @pytest.mark.asyncio
    async def test_process_records_streaming_async_cancels_in_flight_tasks(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test streaming cancellation cancels and drains in-flight batch tasks."""
        async_processor, _ = make_async_processor(
            batch_size=1,
            max_concurrent_batches=2,
        )
        stats = AsyncProcessingStats()
        started: dict[int, asyncio.Event] = {
            1: asyncio.Event(),
            2: asyncio.Event(),
        }
        cancelled: dict[int, asyncio.Event] = {
            1: asyncio.Event(),
            2: asyncio.Event(),
        }

        async def _process_batch(
            _batch_records: list[Record],
            batch_num: int,
            *_args: Any,
        ) -> BatchProcessingResult:
            started[batch_num].set()  # start the batch
            try:
                # test wait until both child coroutines are definitely running
                # waiting forever, since no future.set_result(...) gets called.
                # batch remains suspended at this await until cancellation arrives.
                await asyncio.Future[None]()
            except asyncio.CancelledError as err:  # observing cancellation
                logging.debug(
                    "Batch %d received cancellation: %s",
                    batch_num,
                    err,
                )
                # when the task is cancelled, Python delivers CancelledError at the suspended await
                cancelled[batch_num].set()
                raise  # Re-raise the exception currently being handled.
            # unreachable guard
            pytest.fail("Batch task should have been cancelled")

        patch_async_records(monkeypatch, async_processor, sample_records[:2])

        monkeypatch.setattr(
            async_processor,
            "_process_batch_with_order_async",
            _process_batch,
        )
        # create background task for streaming processing
        streaming_task = asyncio.create_task(
            async_processor.process_records_streaming_async(
                stats,
                progress_callback=None,
                max_wait_time=12.0,
                poll_interval=1.0,
            ),
        )

        # Waiting until both batches start
        await asyncio.gather(*(event.wait() for event in started.values()))
        streaming_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await streaming_task

        assert stats.total_records == 2
        assert cancelled[1].is_set()
        assert cancelled[2].is_set()

    @pytest.mark.asyncio
    async def test_process_records_streaming_async_wraps_failures_and_cleans_up_tasks(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test streaming failure cancel pending tasks and raise ApplicationError."""
        async_processor, _ = make_async_processor(
            batch_size=1,
            max_concurrent_batches=2,
        )
        stats = AsyncProcessingStats()
        second_batch_started = asyncio.Event()
        second_batch_cancelled = asyncio.Event()
        failure = RuntimeError("batch 1 exploded")

        async def _process_batch(
            _batch_records: list[Record],
            batch_num: int,
            *_args: Any,
        ) -> BatchProcessingResult:
            if batch_num == 1:
                await second_batch_started.wait()
                raise failure

            second_batch_started.set()
            try:
                await asyncio.Future[None]()
            except asyncio.CancelledError:
                second_batch_cancelled.set()
                raise
            pytest.fail("Second batch task should have been cancelled")

        patch_async_records(monkeypatch, async_processor, sample_records[:2])

        monkeypatch.setattr(
            async_processor,
            "_process_batch_with_order_async",
            _process_batch,
        )

        with pytest.raises(
            ApplicationError,
            match="Async streaming processing failed",
        ) as exc:
            await async_processor.process_records_streaming_async(
                stats,
                progress_callback=None,
                max_wait_time=12.0,
                poll_interval=1.0,
            )

        assert exc.value.__cause__ is failure
        assert second_batch_cancelled.is_set()

    @pytest.mark.asyncio
    async def test_async_stream_csv_records_yields_all_records(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test the synchronous CSV iterator is wrapped with asyncio.to_thread."""
        async_processor, _ = make_async_processor(records=sample_records[:2])
        to_thread_called = patch_to_thread_inline(monkeypatch)

        records = await async_processor.async_stream_csv_records()

        for record in records:
            log.debug("Record: %s", record)

        assert records == sample_records[:2]
        assert len(to_thread_called) == 3
        assert all(call[0] is next for call in to_thread_called)

    @pytest.mark.asyncio
    async def test_process_batch_with_order_async_returns_successful_batch(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
    ) -> None:
        """Test batch processing passes through the successful batch result."""
        records = sample_records[:2]  # two records to form a single batch
        batch_result = _batch_result_for_records(5, records)

        async_processor, context = make_async_processor(
            async_batch_results={("B1", "B2"): batch_result},
        )

        result = await async_processor.process_batch_with_order_async(
            records,
            batch_num=5,
            progress_callback=None,
            max_wait_time=99.0,
            poll_interval=3.0,
        )

        log.debug("Batch_result: %s", batch_result)
        log.debug("Batch result: %s", result)

        assert result is batch_result
        assert len(context.processor.async_batch_calls) == 1
        async_call = context.processor.async_batch_calls[0]
        assert async_call.batch_num == 5
        assert async_call.max_wait_time == 99.0
        assert async_call.poll_interval == 3.0
        assert async_call.progress_callback is not None

    @pytest.mark.asyncio
    async def test_process_batch_with_order_async_falls_back_to_individual_results(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test batch failures are converted into fallback batch results."""
        async_processor, _ = make_async_processor(
            async_batch_results={("B1", "B2"): RuntimeError("batch failed")},
        )
        fallback_results: list[ProcessingResult] = [
            make_processing_result("B1", "1"),
            make_processing_result(
                "B2",
                "1",
                success=False,
                annotated_text="",
                error_message="failed individual",
            ),
        ]

        async def _fallback(
            batch_records: list[Record],
            stats: AsyncProcessingStats,
        ) -> None:
            assert batch_records == sample_records[:2]
            stats.results.extend(fallback_results)
            stats.processed_records = 1
            stats.failed_records = 1

        monkeypatch.setattr(
            async_processor,
            "_fallback_to_individual_async_streaming",
            _fallback,
        )

        result = await async_processor.process_batch_with_order_async(
            sample_records[:2],
            batch_num=7,
            progress_callback=None,
            max_wait_time=10.0,
            poll_interval=2.0,
        )

        log.debug("Result %s", result)
        log.debug("Fallback results: %s", fallback_results)
        log.debug("Batch result: %s", result.results)
        log.debug("Batch ID: %s", result.batch_id)
        log.debug("Successful count: %d", result.successful_count)
        log.debug("Failed count: %d", result.failed_count)

        assert result.batch_id == "batch_7"
        assert result.results == fallback_results
        assert result.successful_count == 1
        assert result.failed_count == 1

    @pytest.mark.asyncio
    async def test_process_batch_with_order_async_propagates_cancellation(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test cancelled batch processing in re-raised without fallback."""
        async_processor, context = make_async_processor()
        fallback = mocker.AsyncMock()

        async def _cancel(
            *_args: Any,
            **_kwargs: Any,
        ) -> BatchProcessingResult:
            raise asyncio.CancelledError

        monkeypatch.setattr(
            context.processor,
            "process_batch_async",
            _cancel,
        )
        monkeypatch.setattr(
            async_processor,
            "_fallback_to_individual_async_streaming",
            fallback,
        )

        with pytest.raises(asyncio.CancelledError):
            await async_processor.process_batch_with_order_async(
                sample_records[:2],
                batch_num=3,
                progress_callback=None,
                max_wait_time=10.0,
                poll_interval=2.0,
            )

        fallback.assert_not_awaited()


class TestOrderedBatchAccumulation:
    """Tests for batch result accumulation and flushing."""

    @pytest.mark.asyncio
    async def test_add_batch_results_in_order_standard_mode_accumulates_results(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
    ) -> None:
        """Test standard mode extends stats results directly."""
        async_processor, _ = make_async_processor(incremental_mode=False)
        stats = AsyncProcessingStats()
        batch_result = _batch_result_for_records(1, sample_records[:2])

        await async_processor.add_batch_results_in_order(stats, batch_result, 1)

        assert stats.processed_records == 2
        assert stats.failed_records == 0
        assert [result.brevid for result in stats.results] == ["B1", "B2"]
        assert async_processor.batch_result_queue == {}

    @pytest.mark.asyncio
    async def test_add_batch_results_in_order_incremental_mode_queues_and_flushes(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test incremental mode queues the batch before attempting a flush."""
        async_processor, _ = make_async_processor(incremental_mode=True)
        stats = AsyncProcessingStats()
        batch_result = _batch_result_for_records(2, sample_records[:2])
        flush = mocker.AsyncMock()
        monkeypatch.setattr(
            async_processor,
            "_flush_queued_batch_results_async",
            flush,
        )

        await async_processor.add_batch_results_in_order(stats, batch_result, 2)

        assert stats.processed_records == 2
        assert stats.results == []
        assert async_processor.batch_result_queue == {2: batch_result}
        flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_flush_queued_batch_results_async_preserves_order(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test queued incremental batches are written in batch-number order."""
        async_processor, _ = make_async_processor(incremental_mode=True)
        write_order: list[int] = []
        async_processor.batch_result_queue = {
            2: _batch_result_for_records(2, sample_records[2:4]),
            1: _batch_result_for_records(1, sample_records[:2]),
            3: _batch_result_for_records(3, sample_records[:1]),
        }

        async def _write(_batch_result: BatchProcessingResult, batch_num: int) -> None:
            write_order.append(batch_num)

        monkeypatch.setattr(
            async_processor,
            "_write_batch_results_incremental_async",
            _write,
        )

        stats = AsyncProcessingStats()
        await async_processor.flush_queued_batch_results_async(stats)

        assert write_order == [1, 2, 3]
        assert async_processor.next_expected_batch_num == 4
        assert async_processor.batch_result_queue == {}
        assert stats.failed_batch_writes == []

    @pytest.mark.asyncio
    async def test_flush_queued_batch_results_async_logs_and_continues_on_write_failure(
        self,
        caplog: pytest.LogCaptureFixture,
        make_async_processor: AsyncProcessorProbeFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test a per-batch write failure does not abort subsequent flushes.

        Batch 2's write raises; the flush loop must capture a ``FailedBatchInfo``
        on ``stats.failed_batch_writes``, log a record-count error, and continue
        flushing batches 1 and 3 successfully.
        """
        async_processor, _ = make_async_processor(incremental_mode=True)
        write_attempts: list[int] = []
        records_b1_b2 = [
            make_record(bindnr="1", brevid="B1"),
            make_record(bindnr="1", brevid="B2"),
        ]
        records_b3_b4 = [
            make_record(bindnr="1", brevid="B3"),
            make_record(bindnr="1", brevid="B4"),
        ]
        records_b5 = [make_record(bindnr="1", brevid="B5")]
        async_processor.batch_result_queue = {
            1: _batch_result_for_records(1, records_b1_b2),
            2: _batch_result_for_records(2, records_b3_b4),
            3: _batch_result_for_records(3, records_b5),
        }

        async def _write(_batch_result: BatchProcessingResult, batch_num: int) -> None:
            log.debug("Writing batch %d", batch_num)
            write_attempts.append(batch_num)
            if batch_num == 2:
                raise RuntimeError("disk full")

        monkeypatch.setattr(
            async_processor,
            "_write_batch_results_incremental_async",
            _write,
        )

        stats = AsyncProcessingStats()
        with caplog.at_level(logging.DEBUG):
            await async_processor.flush_queued_batch_results_async(stats)

        log.debug("failed_batch_writes: %s", stats.failed_batch_writes)

        # All three batches were attempted; batch 2 raised but did not abort the loop.
        assert write_attempts == [1, 2, 3]
        # Queue fully drained, ordering counter advanced past every batch.
        assert async_processor.batch_result_queue == {}
        assert async_processor.next_expected_batch_num == 4
        # Failed batch captured with full record-id detail.
        assert len(stats.failed_batch_writes) == 1
        failed_batch_write = stats.failed_batch_writes[0]
        assert failed_batch_write.batch_num == 2
        assert failed_batch_write.record_ids == ["1_B3", "1_B4"]
        assert failed_batch_write.error_type == "RuntimeError"
        assert "disk full" in failed_batch_write.error_message
        # Log line records the count, not the IDs.
        assert "Batch 2 incremental write failed" in caplog.text
        assert "2 record(s) missing" in caplog.text


class TestIncrementalBatchWriting:
    """Tests for incremental output writing."""

    @pytest.mark.asyncio
    async def test_write_batch_results_incremental_async_writes_successful_rows(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test successful incremental writes append annotations and metadata."""
        writer = FakeWriter()
        async_processor, context = make_async_processor(
            incremental_mode=True,
            writer=writer,
        )
        entity = make_entity_record("B1")
        log.debug("Entity: %s", entity)
        batch_result = make_batch_processing_result(
            1,
            [
                make_processing_result("B1", "1", entities=[entity]),
                make_processing_result(
                    "B2",
                    "1",
                    success=False,
                    annotated_text="",
                    error_message="bad",
                ),
            ],
        )

        log.debug("Batch_result: %s", batch_result)

        patch_to_thread_inline(monkeypatch)

        await async_processor.write_batch_results_incremental_async(
            batch_result,
            batch_num=1,
        )

        log.debug("context output text file: %s", context.output_text_file)
        log.debug("context output table file: %s", context.output_table_file)
        log.debug("context annotated header: %s", context.ANNOTATED_HEADER)
        log.debug("context metadata header: %s", context.METADATA_HEADER)

        assert len(writer.text_calls) == 1
        assert len(writer.metadata_calls) == 1
        assert writer.text_calls[0].file_path == context.output_text_file
        assert writer.text_calls[0].header == context.ANNOTATED_HEADER
        assert writer.text_calls[0].rows == ["ann-B1"]
        assert writer.metadata_calls[0].file_path == context.output_table_file
        assert writer.metadata_calls[0].header == context.METADATA_HEADER
        assert writer.metadata_calls[0].rows == [entity.to_csv_row()]

    @pytest.mark.asyncio
    async def test_write_batch_results_incremental_async_skips_empty_successes(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
    ) -> None:
        """Test batches without successful annotated output do not write anything."""
        writer = FakeWriter()
        async_processor, _ = make_async_processor(
            incremental_mode=True,
            writer=writer,
        )
        batch_result = make_batch_processing_result(
            2,
            [
                make_processing_result(
                    "B2",
                    "1",
                    success=False,
                    annotated_text="",
                    error_message="failed",
                )
            ],
        )

        await async_processor.write_batch_results_incremental_async(
            batch_result,
            batch_num=2,
        )

        assert writer.text_calls == []
        assert writer.metadata_calls == []

    @pytest.mark.asyncio
    async def test_write_batch_results_incremental_async_raises_failures(
        self,
        caplog: pytest.LogCaptureFixture,
        make_async_processor: AsyncProcessorProbeFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test incremental output failures are logged and surfaced (Bug 1 fix).

        Writes use ``asyncio.TaskGroup``, which wraps child exceptions in an
        `ÈxceptionGroup``, The underlying ``RuntimeError``must still be recoverable
        from the group and the failure must propagate rather than be swallowed
        silently.
        """
        writer = FakeWriter(text_error=RuntimeError("disk full"))
        async_processor, _ = make_async_processor(
            incremental_mode=True,
            writer=writer,
        )
        batch_result = make_batch_processing_result(
            3,
            [
                make_processing_result(
                    "B3",
                    "2",
                    entities=[make_entity_record("B3")],
                )
            ],
        )

        patch_to_thread_inline(monkeypatch)

        with caplog.at_level(logging.DEBUG), pytest.raises(ExceptionGroup) as exc_info:
            await async_processor.write_batch_results_incremental_async(
                batch_result,
                batch_num=3,
            )

        runtime_errors = [
            exc for exc in exc_info.value.exceptions if isinstance(exc, RuntimeError)
        ]
        assert len(runtime_errors) == 1
        assert "disk full" in str(runtime_errors[0])
        assert "Failed to write batch 3 results incrementally" in caplog.text


class TestIndividualAsyncProcessing:
    """Tests for individual-record async workflows."""

    @pytest.mark.asyncio
    async def test_process_records_individual_async_uses_chunking_and_semaphore_limits(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test per-record async processing respects chunking and concurrency limits."""
        records = [*sample_records, make_record(bindnr="3", brevid="B5")]
        async_processor, context = make_async_processor(
            batch_size=1,
            max_concurrent_individual=2,
            chunk_size=3,
        )
        stats = AsyncProcessingStats()
        active = 0
        max_active = 0

        async def _process_record(record: Record) -> Any:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01 if record["Brevid"] == "B1" else 0.0)
            active -= 1
            return make_processing_result(record["Brevid"], record["Bindnr"])

        patch_async_records(monkeypatch, async_processor, records)

        monkeypatch.setattr(
            context.processor,
            "process_record_async",
            _process_record,
        )

        chunk_spy = mocker.spy(
            async_processor,
            "_process_task_chunk",
        )

        await async_processor.process_records_individual_async(stats)

        assert max_active <= 2
        assert chunk_spy.call_count == 2
        assert stats.total_records == 5
        assert stats.processed_records == 5
        assert [result.brevid for result in stats.results] == [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
        ]

    @pytest.mark.asyncio
    async def test_process_records_individual_async_wraps_unexpected_failures(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test unexpected individual-processing failures are wrapped."""
        async_processor, context = make_async_processor(chunk_size=2)
        stats = AsyncProcessingStats()
        failure = RuntimeError("chunk bookkeeping broke")

        async def _process_record(record: Record) -> Any:
            return make_processing_result(record["Brevid"], record["Bindnr"])

        async def _raise_after_draining(
            tasks: list[asyncio.Task[Any]],
            _chunk_records: list[Record],
            _stats: AsyncProcessingStats,
        ) -> None:
            await asyncio.gather(*tasks, return_exceptions=True)
            raise failure

        patch_async_records(monkeypatch, async_processor, sample_records[:2])

        monkeypatch.setattr(context.processor, "process_record_async", _process_record)

        monkeypatch.setattr(
            async_processor,
            "_process_task_chunk",
            _raise_after_draining,
        )

        with pytest.raises(
            ApplicationError,
            match="Individual async streaming processing failed: chunk bookkeeping broke",
        ) as exc:
            await async_processor.process_records_individual_async(stats)

        assert exc.value.__cause__ is failure

    @pytest.mark.asyncio
    async def test_process_task_chunk_converts_exceptions_and_preserves_order(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
    ) -> None:
        """Test raw task exceptions become failed results in input order."""
        async_processor, _ = make_async_processor()
        stats = AsyncProcessingStats()

        async def _success() -> Any:
            return make_processing_result("B1", "1")

        async def _crash() -> Any:
            raise RuntimeError("unhandled failure")

        async def _failed_result() -> Any:
            return make_processing_result(
                "B3",
                "2",
                success=False,
                annotated_text="",
                error_message="returned failure",
            )

        tasks = [
            asyncio.create_task(_success()),
            asyncio.create_task(_crash()),
            asyncio.create_task(_failed_result()),
        ]

        await async_processor.process_task_chunk(
            tasks,
            sample_records[:3],
            stats,
        )

        assert stats.processed_records == 1
        assert stats.failed_records == 2
        assert [result.brevid for result in stats.results] == ["B1", "B2", "B3"]
        assert stats.results[1].success is False
        assert "unhandled failure" in str(stats.results[1].error_message)

    @pytest.mark.asyncio
    async def test_process_task_chunk_propagates_cancellation(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
    ) -> None:
        """Test any cancelled child task propagates cancellation."""
        async_processor, _ = make_async_processor()
        stats = AsyncProcessingStats()

        async def _cancel() -> Any:
            raise asyncio.CancelledError

        async def _success() -> Any:
            return make_processing_result("B2", "1")

        tasks = [
            asyncio.create_task(_cancel()),
            asyncio.create_task(_success()),
        ]

        with pytest.raises(asyncio.CancelledError):
            await async_processor.process_task_chunk(
                tasks,
                sample_records[:2],
                stats,
            )

    @pytest.mark.asyncio
    async def test_fallback_to_individual_async_streaming_preserves_order_and_limits_concurrency(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test fallback individual processing preserves order despite concurrency."""
        async_processor, context = make_async_processor(fallback_concurrency=2)
        stats = AsyncProcessingStats()
        active = 0
        max_active = 0

        async def _process_record(record: Record) -> Any:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            if record["Brevid"] == "B1":
                await asyncio.sleep(0.01)
            active -= 1
            if record["Brevid"] == "B2":
                raise RuntimeError("fallback failure")
            return make_processing_result(record["Brevid"], record["Bindnr"])

        monkeypatch.setattr(context.processor, "process_record_async", _process_record)

        await async_processor.fallback_to_individual_async_streaming(
            sample_records[:3],
            stats,
        )

        assert max_active <= 2
        assert stats.processed_records == 2
        assert stats.failed_records == 1
        assert [result.brevid for result in stats.results] == ["B1", "B2", "B3"]
        assert stats.results[1].success is False
        assert "fallback failure" in str(stats.results[1].error_message)

    @pytest.mark.asyncio
    async def test_fallback_to_individual_async_streaming_propagates_cancellation(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
        sample_records: list[Record],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test fallback cancellation is propagated to the caller."""
        async_processor, context = make_async_processor()
        stats = AsyncProcessingStats()

        async def _process_record(record: Record) -> Any:
            if record["Brevid"] == "B1":
                raise asyncio.CancelledError
            return make_processing_result(record["Brevid"], record["Bindnr"])

        monkeypatch.setattr(context.processor, "process_record_async", _process_record)

        with pytest.raises(asyncio.CancelledError):
            await async_processor.fallback_to_individual_async_streaming(
                sample_records[:2],
                stats,
            )


class TestAsyncProcessorHelpers:
    """Tests for smaller AsyncProcessor helper methods."""

    def test_create_batch_progress_callback_logs_and_shields_user_errors(
        self,
        caplog: pytest.LogCaptureFixture,
        make_async_processor: AsyncProcessorProbeFactory,
    ) -> None:
        """Test progress callback logs status and isolates user callback failures."""
        async_processor, _ = make_async_processor()
        received: list[BatchProgress] = []

        def _user_callback(progress: BatchProgress) -> None:
            received.append(progress)
            raise RuntimeError("callback failed")

        callback = async_processor.create_batch_progress_callback(
            3,
            6,
            _user_callback,
        )
        progress = BatchProgress(
            batch_num=3,
            batch_id="batch_123",
            status=BatchStatus.IN_PROGRESS,
            elapsed_time=9.5,
            request_counts={"processing": 2, "succeeded": 1, "errored": 0},
            created_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-02T00:00:00Z",
        )

        with caplog.at_level(logging.DEBUG):
            callback(progress)

        assert received == [progress]
        assert "Batch 3/6 (ID: batch_123): in_progress" in caplog.text
        assert "Error in user progress callback: callback failed" in caplog.text

    def test_create_failed_result_formats_record_id_and_error_message(
        self,
        make_async_processor: AsyncProcessorProbeFactory,
    ) -> None:
        """Test failed-result helper produces consistent IDs and messages."""
        async_processor, _ = make_async_processor()

        result = async_processor.create_failed_result(
            "B3",
            RuntimeError("broken"),
            "12",
        )

        assert result.record_id == "12_B3"
        assert result.brevid == "B3"
        assert result.success is False
        assert result.annotated_text == ""
        assert result.error_message == "Processing failed for Brevid B3: broken"


class MockMonotonic:
    """Deterministic callable used to replace time.monotonic in tests."""

    def __init__(self, values: list[float]) -> None:
        self._values = values
        self._index = 0

    def __call__(self) -> float:
        """Return the next configured monotonic value."""
        if self._index >= len(self._values):
            return self._values[-1]
        value = self._values[self._index]
        self._index += 1
        return value
