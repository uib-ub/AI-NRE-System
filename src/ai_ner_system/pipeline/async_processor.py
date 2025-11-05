"""Asynchronous processing module for medieval text processing pipeline.

This module handles all asynchronous processing workflows, including individual
record processing, batch processing, streaming modes, and incremental output
with order preservation. It provides comprehensive error handling and progress
monitoring for async operations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, cast

from ai_ner_system.config import Settings
from ai_ner_system.processing import BatchProcessingResult, ProcessingResult

from .stats import ApplicationError, AsyncProcessingStats

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import AsyncIterator, Callable

    from ai_ner_system.file_io import CSVReader, OutputWriter
    from ai_ner_system.llm import BatchProgress, Client
    from ai_ner_system.processing import RecordProcessor

    from .processor_protocol import ProcessorContext


class AsyncProcessor:
    """Handles asynchronous processing workflows for medieval text processing.

    This class is responsible for executing async processing pipelines,
    including batch processing with fallback, streaming modes, incremental
    output, and comprehensive progress monitoring.

    All default values for concurrency, timeouts, and intervals are pulled from
    Settings, ensuring a single source of truth for configuration. These can be
    overridden via command-line arguments.
    """

    def __init__(self, main_processor: ProcessorContext) -> None:
        """Initialize async processor with reference to main processor.

        Args:
           main_processor: Main MedievalTextProcessor instance.
        """
        self.main_processor = main_processor

        # Initialize tracking for incremental output order preservation
        self._next_expected_batch_num = 1
        self._batch_result_queue: dict[int, BatchProcessingResult] = {}

    @property
    def args(self) -> Namespace:
        """Access to command line arguments via main processor."""
        return self.main_processor.args

    @property
    def reader(self) -> CSVReader:
        """Access to input reader via main processor."""
        return self.main_processor.reader

    @property
    def writer(self) -> OutputWriter:
        """Access to output writer via main processor."""
        return self.main_processor.writer

    @property
    def processor(self) -> RecordProcessor:
        """Access to record processor via main processor."""
        return self.main_processor.processor

    @property
    def llm_client(self) -> Client:
        """Access to LLM client via main processor."""
        return self.main_processor.llm_client

    @property
    def _incremental_mode(self) -> bool:
        """Check if incremental output mode is enabled."""
        return bool(self.main_processor.incremental_mode)

    @property
    def _batch_wait_time(self) -> float:
        """Get default batch wait time."""
        return Settings.DEFAULT_MAX_WAIT_TIME

    @property
    def _poll_interval(self) -> float:
        """Get default poll interval."""
        return Settings.DEFAULT_POLL_INTERVAL

    @property
    def _max_concurrent_batches(self) -> int:
        """Get max concurrent batches from args or default."""
        return getattr(
            self.args,
            "max_concurrent_batches",
            Settings.DEFAULT_MAX_CONCURRENT_BATCHES,
        )

    @property
    def _max_concurrent_individual(self) -> int:
        """Get max concurrent individual tasks from args or default."""
        return getattr(
            self.args,
            "max_concurrent_individual",
            Settings.DEFAULT_MAX_CONCURRENT_INDIVIDUAL,
        )

    @property
    def _fallback_concurrency(self) -> int:
        """Get fallback concurrency from args or default."""
        return getattr(
            self.args,
            "fallback_concurrency",
            Settings.DEFAULT_FALLBACK_CONCURRENCY,
        )

    @property
    def _chunk_size(self) -> int:
        """Get chunk size from args or default."""
        return getattr(
            self.args,
            "chunk_size",
            Settings.DEFAULT_CHUNK_SIZE,
        )

    async def process_all_records_async(
        self,
        progress_callback: Callable[[BatchProgress], None] | None = None,
        max_batch_wait_time: float | None = None,
        poll_interval: float | None = None,
    ) -> AsyncProcessingStats:
        """Process all records asynchronously with batch operations.

        This method provides async batch processing with real-time progress monitoring
        and comprehensive error handling. It automatically falls back to individual
        processing if batch processing is not available.

        Args:
            progress_callback: Optional callback for batch progress updates.
            max_batch_wait_time: Maximum time in seconds to wait for batch completion.
                Defaults to DEFAULT_BATCH_WAIT_TIME (24 hours).
            poll_interval: Time in seconds between progress checks.
                Defaults to DEFAULT_POLL_INTERVAL (30 seconds).

        Returns:
            AsyncProcessingStats with detailed processing information.

        Raises:
            ApplicationError: If processing fails completely.
        """
        if not self.reader or not self.processor:
            msg = "Components not properly initialized for async processing"
            raise ApplicationError(msg)

        # Initialize statistics
        stats = AsyncProcessingStats(start_time=time.monotonic())
        success = False

        try:
            logging.info("Starting async streaming processing...")
            # Use defaults if not specified
            wait_time = max_batch_wait_time or self._batch_wait_time
            poll_time = poll_interval or self._poll_interval

            # check if the LLM client supports async batch processing and if batch processing is enabled
            if self.args.batch_size > 1 and self.llm_client.supports_async_batch():
                # Use async batch processing with streaming
                await self._process_records_streaming_async(
                    stats,
                    progress_callback,
                    wait_time,
                    poll_time,
                )
            else:
                # Use individual async processing with streaming
                await self._process_records_individual_async(stats)

            success = True
        except asyncio.CancelledError:
            logging.info("Async processing cancelled")
            raise
        except Exception as e:
            error_msg = "Async streaming processing failed"
            logging.exception(error_msg)
            raise ApplicationError(error_msg) from e
        finally:
            # Finalize statistics
            stats.end_time = time.monotonic()
            stats.processing_time = stats.end_time - stats.start_time

        if success:
            logging.info(
                "Async streaming processing completed: %d/%d records (%.1f%% success rate) in %.2fs",
                stats.processed_records,
                stats.total_records,
                stats.success_rate,
                stats.processing_time,
            )
        return stats

    async def _process_records_streaming_async(
        self,
        stats: AsyncProcessingStats,
        progress_callback: Callable[[BatchProgress], None] | None,
        max_wait_time: float,
        poll_interval: float,
    ) -> None:
        """Process records using async streaming approach with batching.

        Args:
            stats: Statistics object to update.
            progress_callback: Optional callback for progress updates.
            max_wait_time: Maximum time to wait for batch completion.
            poll_interval: Time between progress checks.
        """
        logging.info(
            "Starting async streaming with batch processing, batch size: %d",
            self.args.batch_size,
        )

        # a batch of records to process together
        batch_records: list[dict[str, str]] = []
        record_count = 0
        batch_num = 0
        # Track batch tasks with their order information using a map (batch_num -> task)
        batch_tasks: dict[int, asyncio.Task[BatchProcessingResult]] = {}

        # Limit to default 5 (configured in settings) concurrent batch processing tasks,
        # otherwise it can reach 50 batch request limitation of Anthropic API
        max_concurrent_batches = self._max_concurrent_batches
        logging.info(
            "Using max concurrent batches: %d",
            max_concurrent_batches,
        )

        try:
            # Stream records asynchronously in batches and
            # concurrently (coroutines) process them with order preservation
            async for record in self._async_stream_csv_records():
                batch_records.append(record)
                # count number of records
                record_count += 1
                # Update total count as we discover records
                stats.total_records = record_count

                # Process batch when it reaches the specified size, eg: 10, 100
                if len(batch_records) >= self.args.batch_size:
                    batch_num += 1  # batch_num starts from 1
                    # Create and start coroutine task with batch number tracking
                    batch_tasks[batch_num] = asyncio.create_task(
                        self._process_batch_with_order_async(
                            batch_records.copy(),
                            batch_num,  # Use batch_num for tracking the order
                            progress_callback,
                            max_wait_time,
                            poll_interval,
                        ),
                    )
                    # Clear batch records after processing a batch
                    batch_records.clear()

                    # Limit concurrent batch processing tasks by
                    # keep up max_concurrent_batches (5 as default) tasks running at any time
                    if len(batch_tasks) >= max_concurrent_batches:
                        # Wait for the OLDEST(smallest batch_num) batch to complete (maintain order)
                        oldest_batch_num = min(batch_tasks.keys())
                        oldest_task = batch_tasks.pop(oldest_batch_num)
                        # Process results in order
                        batch_result = await oldest_task
                        # Add results to stats in order
                        await self._add_batch_results_in_order(
                            stats,
                            batch_result,
                            oldest_batch_num,
                        )

            # Process final batch if there are any remaining records
            if batch_records:
                batch_num += 1
                batch_tasks[batch_num] = asyncio.create_task(
                    self._process_batch_with_order_async(
                        batch_records.copy(),
                        batch_num,
                        progress_callback,
                        max_wait_time,
                        poll_interval,
                    ),
                )

            # Process any remaining batch tasks in ORDER
            for batch_num in sorted(batch_tasks.keys()):
                batch_result = await batch_tasks[batch_num]
                # Add results to stats in order
                await self._add_batch_results_in_order(stats, batch_result, batch_num)

            # Final flush of any queued results (for incremental mode)
            if self._incremental_mode:
                await self._flush_queued_batch_results_async()

            logging.info(
                "Async streaming processing completed with preserved order: %d records",
                record_count,
            )
        except asyncio.CancelledError:
            # Cancel & drain all children, then propagate
            for task in batch_tasks.values():
                task.cancel()
            await asyncio.gather(*batch_tasks.values(), return_exceptions=True)
            logging.info("Async streaming processing cancelled")
            raise
        except Exception as e:
            # Cancel remaining tasks
            for task in batch_tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*batch_tasks.values(), return_exceptions=True)
            error_msg = "Async streaming processing failed"
            logging.exception(error_msg)
            raise ApplicationError(error_msg) from e

    async def _async_stream_csv_records(self) -> AsyncIterator[dict[str, str]]:
        """Asynchronously stream records from the CSV input file.

        Wraps the synchronous CSV reader in an async iterator to allow
        non-blocking I/O operations during record processing. Uses
        asyncio.to_thread to run the blocking next() call in a thread pool.

        Yields:
            Record dictionaries from the CSV file.
        """
        iterator = self.reader.stream_records()

        while True:
            # Run synchronous next() in thread pool to avoid blocking event loop
            # next(iterator, None) returns None at end instead of raising StopIteration
            record = await asyncio.to_thread(next, iterator, None)
            if record is None:
                break
            yield record

    async def _process_batch_with_order_async(
        self,
        batch_records: list[dict[str, str]],
        batch_num: int,
        progress_callback: Callable[[BatchProgress], None] | None,
        max_wait_time: float,
        poll_interval: float,
    ) -> BatchProcessingResult:
        """Process a batch of records asynchronously with order tracking.

        Args:
            batch_records: List of records to process in this batch.
            batch_num: Current batch number (starting from number 1).
            progress_callback: Optional callback for progress updates.
            max_wait_time: Maximum time to wait for batch completion.
            poll_interval: Time between progress checks.

        Returns:
            BatchProcessingResult.
        """
        logging.info(
            "Processing batch %d with %d records (order-preserving)",
            batch_num,
            len(batch_records),
        )

        # Create progress callback for this batch
        batch_progress_callback = self._create_batch_progress_callback(
            # None for total_batches since we don't know yet
            batch_num,
            None,
            progress_callback,
        )

        try:
            # Process batch asynchronously
            batch_result = await self.processor.process_batch_async(
                batch_records,
                batch_num,
                progress_callback=batch_progress_callback,
                max_wait_time=max_wait_time,
                poll_interval=poll_interval,
            )

        except asyncio.CancelledError:
            logging.info("Batch %d cancelled", batch_num)
            raise
        except Exception:
            logging.exception("Batch %d failed.", batch_num)
            # Fallback to per-record async processing; do not raise
            fallback_stats = AsyncProcessingStats()
            await self._fallback_to_individual_async_streaming(
                batch_records,
                fallback_stats,
            )
            # Convert to BatchProcessingResult format
            return BatchProcessingResult(
                batch_id=f"fallback_batch_{batch_num}",
                results=fallback_stats.results,
                total_processing_time=0.0,
                successful_count=fallback_stats.processed_records,
                failed_count=fallback_stats.failed_records,
            )
        else:
            logging.info(
                "Async batch %d completed: %d successful, %d failed in %.2fs",
                batch_num,
                batch_result.successful_count,
                batch_result.failed_count,
                batch_result.total_processing_time,
            )

            return batch_result

    async def _add_batch_results_in_order(
        self,
        stats: AsyncProcessingStats,
        batch_result: BatchProcessingResult,
        batch_num: int,
    ) -> None:
        """Add batch results to stats while preserving order and handle incremental output.

        In incremental mode, results are queued until all earlier batches have been
        written, ensuring output appears in the correct order. In standard mode,
        results are accumulated in memory for final writing.

        Args:
            stats: Statistics object to update.
            batch_result: Result of the processed batch.
            batch_num: The batch number for tracking.
        """
        logging.info(
            "Adding results for batch %d (expected: %d)",
            batch_num,
            self._next_expected_batch_num,
        )

        stats.processed_records += batch_result.successful_count
        stats.failed_records += batch_result.failed_count

        if self._incremental_mode:
            # Incremental mode: queue results until we can write them in order
            self._batch_result_queue[batch_num] = batch_result
            await self._flush_queued_batch_results_async()
        else:
            # Standard mode: accumulate all results in memory
            # Add results in batch order (they're already in record order within batch)
            if batch_result.results:
                stats.results.extend(batch_result.results)
            logging.info(
                "Added results from batch %d to stats in order",
                batch_num,
            )

        logging.info(
            "Added batch %d results: %d successful, %d failed",
            batch_num,
            batch_result.successful_count,
            batch_result.failed_count,
        )

    async def _flush_queued_batch_results_async(self) -> None:
        """Write queued batch results in order and remove from queue."""
        while self._next_expected_batch_num in self._batch_result_queue:
            # pop the next expected batch result
            batch_result = self._batch_result_queue.pop(
                self._next_expected_batch_num,
            )
            # Write this batch's results immediately
            await self._write_batch_results_incremental_async(
                batch_result,
                self._next_expected_batch_num,
            )
            self._next_expected_batch_num += 1

            logging.info(
                "Flushed batch %d results to output files",
                self._next_expected_batch_num - 1,
            )

    async def _write_batch_results_incremental_async(
        self,
        batch_result: BatchProcessingResult,
        batch_num: int,
    ) -> None:
        """Write batch results to output files incrementally.

        Args:
            batch_result: Result of the processed batch.
            batch_num: The batch number for tracking.
        """
        try:
            successful_results = [
                r
                for r in batch_result.results
                if r.success and r.annotated_text.strip()
            ]

            if not successful_results:
                logging.info("Batch %d: No successful results to write", batch_num)
                return

            # Prepare annotated data and entity metadata
            annotated_rows: list[str] = [
                res.annotated_text for res in successful_results
            ]

            metadata_rows: list[str] = [
                entity.to_csv_row()
                for res in successful_results
                for entity in res.entities
            ]

            # Determine output file paths from main processor
            output_text_file = self.main_processor.output_text_file
            output_table_file = self.main_processor.output_table_file

            # Use headers from main processor to maintain consistency
            annotated_header = self.main_processor.ANNOTATED_HEADER
            metadata_header = self.main_processor.METADATA_HEADER

            # Use TaskGroup for better async task management concurrently, and it
            # automatically waits for all tasks to complete when exiting the context manager
            async with asyncio.TaskGroup() as tg:
                # write output files asynchronously using OutputWriter methods
                # Write annotated text output
                tg.create_task(
                    asyncio.to_thread(
                        self.writer.append_text_output,
                        output_text_file,
                        annotated_header,
                        annotated_rows,
                    ),
                )

                # Write metadata if we have entities
                if metadata_rows:
                    tg.create_task(
                        asyncio.to_thread(
                            self.writer.append_metadata_output,
                            output_table_file,
                            metadata_header,
                            metadata_rows,
                        ),
                    )
            logging.info(
                "Batch %d: Wrote %d annotations and %d entities incrementally",
                batch_num,
                len(annotated_rows),
                len(metadata_rows),
            )
        # Re-raise cancellation; log-and-continue for other failures.
        except asyncio.CancelledError:
            logging.info("Incremental write for batch %d cancelled", batch_num)
            raise
        except Exception:
            logging.exception(
                "Failed to write batch %d results incrementally",
                batch_num,
            )
            # non-fatal, don't raise - this is not critical enough to stop processing

    async def _process_records_individual_async(
        self,
        stats: AsyncProcessingStats,
    ) -> None:
        """Process records individually asynchronously using streaming.

        Args:
            stats: Statistics object to update.
        """
        logging.info("Starting individual async streaming processing")

        try:
            # Process records with limited concurrency to avoid overwhelming the API
            # Use configured concurrency limit (5 as default)
            max_concurrent = self._max_concurrent_individual
            semaphore = asyncio.Semaphore(max_concurrent)
            logging.info(
                "Using max concurrent individual tasks: %d",
                max_concurrent,
            )

            async def process_single_record(record: dict[str, str]) -> ProcessingResult:
                async with semaphore:
                    return await self.processor.process_record_async(record)

            # Create tasks for streaming records
            tasks: list[asyncio.Task[ProcessingResult]] = []
            # Keep track of current chunk records
            current_chunk_records: list[dict[str, str]] = []
            record_count = 0

            async for record in self._async_stream_csv_records():
                record_count += 1
                stats.total_records = record_count
                tasks.append(asyncio.create_task(process_single_record(record)))
                current_chunk_records.append(record)

                # Process in chunks to avoid memory issues with large files
                # Use configured chunk size (50 as default)
                chunk_size = self._chunk_size
                if len(tasks) >= chunk_size:
                    await self._process_task_chunk(tasks, current_chunk_records, stats)
                    tasks.clear()
                    current_chunk_records.clear()

            # Process remaining tasks
            if tasks:
                await self._process_task_chunk(tasks, current_chunk_records, stats)

        except asyncio.CancelledError:
            logging.info("Individual async streaming cancelled")
            raise
        except Exception as e:
            logging.exception("Individual async streaming processing failed")
            raise ApplicationError(
                f"Individual async streaming processing failed: {e}",
            ) from e

    async def _process_task_chunk(
        self,
        tasks: list[asyncio.Task[ProcessingResult]],
        chunk_records: list[dict[str, str]],
        stats: AsyncProcessingStats,
    ) -> None:
        """Process a chunk of async tasks.

        Args:
            tasks: List of asyncio tasks to process.
            chunk_records: Original records corresponding to the tasks.
            stats: Statistics object to update.
        """
        # Gather results in the same order tasks were created
        # return_exceptions=True ensures all tasks complete even if some fail
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # If any child was cancelled, propagate cancellation before entering try
        if any(isinstance(r, asyncio.CancelledError) for r in results):
            raise asyncio.CancelledError

        try:
            # Process results in the same order as input
            for i, result in enumerate(results):
                # Get original record by index and extract brevid
                brevid = chunk_records[i].get("Brevid", "unknown")

                # Any other exception => convert to failed ProcessingResult
                if isinstance(result, Exception):
                    # Handle failed task
                    stats.failed_records += 1
                    # Create a failed ProcessingResult for consistency
                    failed_result = ProcessingResult(
                        # TODO: check record_id format, with "failed" prefix?
                        record_id=f"failed_{brevid}",
                        brevid=brevid,
                        success=False,
                        error_message=f"Processing failed for Brevid {brevid}: {result}",
                    )
                    stats.results.append(failed_result)
                    logging.error(
                        "Task failed for Brevid %s with exception: %s",
                        brevid,
                        result,
                    )
                    continue

                # Handle successful task
                result = cast("ProcessingResult", result)
                stats.results.append(result)
                if result.success:
                    stats.processed_records += 1
                else:
                    stats.failed_records += 1
                    logging.warning(
                        "Record %s and Brevid %s failed: %s",
                        result.record_id,
                        result.brevid,
                        result.error_message,
                    )

            logging.info("Processed chunk: %d tasks completed", len(results))
        except asyncio.CancelledError:
            logging.info("Processing of task chunk cancelled")
            raise
        except Exception:
            logging.exception("Error processing task chunk")
            # Update stats for failed chunk
            stats.failed_records += len(tasks)
            raise

    async def _fallback_to_individual_async_streaming(
        self,
        batch_records: list[dict[str, str]],
        stats: AsyncProcessingStats,
    ) -> None:
        """Fallback to individual async processing for a batch of records.

        Args:
            batch_records: Records to process individually.
            stats: Statistics object to update.
        """
        logging.info(
            "Falling back to individual async processing for %d records",
            len(batch_records),
        )

        # Process records with limited concurrency
        # Use configured fallback concurrency (3 as default)
        fallback_concurrency = self._fallback_concurrency
        semaphore = asyncio.Semaphore(fallback_concurrency)
        logging.info(
            "Using fallback concurrency: %d",
            fallback_concurrency,
        )

        async def process_one(record: dict[str, str]) -> ProcessingResult:
            async with semaphore:
                return await self.processor.process_record_async(record)

        # Create tasks for all records in this batch
        tasks = [process_one(record) for record in batch_records]

        # Process all tasks by asyncio.gather() to preserve order
        # return_exceptions=True ensures all records get processed even if some fail
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results in original order and update statistics
        for i, result in enumerate(results):
            # Get original record by index and extract brevid
            brevid = batch_records[i].get("Brevid", "unknown")
            if isinstance(result, asyncio.CancelledError):
                # Cooperate with cancellation policy
                raise result
            if isinstance(result, Exception):
                # Handle failed task
                stats.failed_records += 1
                # Create a failed ProcessingResult for the exception
                failed_result = ProcessingResult(
                    record_id=f"failed_{brevid}",
                    brevid=brevid,
                    success=False,
                    error_message=f"Processing failed for Brevid {brevid}: {result}",
                )
                stats.results.append(failed_result)
                logging.warning(
                    "Fallback processing exception for Brevid %s: %s",
                    brevid,
                    result,
                )
                continue

            result = cast("ProcessingResult", result)
            # Handle successful task
            stats.results.append(result)  # Results added in original order
            if result.success:
                stats.processed_records += 1
            else:
                stats.failed_records += 1
                logging.warning(
                    "Fallback processing failed for record %s and Brevid %s: %s",
                    result.record_id,
                    result.brevid,
                    result.error_message,
                )

    def _create_batch_progress_callback(
        self,
        batch_num: int,
        total_batches: int | None,
        user_callback: Callable[[BatchProgress], None] | None = None,
    ) -> Callable[[BatchProgress], None]:
        """Create a progress callback for batch monitoring.

        Args:
            batch_num: Current batch number (starting from number 1).
            total_batches: Total number of batches (None if unknown).
            user_callback: Optional user-defined callback for progress updates.

        Returns:
            Progress callback function that can be used to report batch progress.
        """

        def progress_callback(progress: BatchProgress) -> None:
            # Log batch progress
            counts: dict[str, int] = progress.request_counts
            if total_batches:
                batch_info = f"Batch {batch_num}/{total_batches}"
            else:
                batch_info = f"Batch {batch_num}"

            logging.info(
                "%s (ID: %s): %s - Processing: %d, Succeeded: %d, Errored: %d, Elapsed: %.1fs",
                batch_info,
                progress.batch_id,
                progress.status.value,
                counts.get("processing", 0),
                counts.get("succeeded", 0),
                counts.get("errored", 0),
                progress.elapsed_time,
            )

            # Call user-defined callback if available
            if user_callback:
                try:
                    user_callback(progress)
                except Exception as e:  # noqa: BLE001 - user callback is untrusted code
                    # Robustness - batch processing continues even if user callbacks fail
                    logging.warning(
                        "Error in user progress callback: %s",
                        e,
                        exc_info=True,
                    )

        return progress_callback
