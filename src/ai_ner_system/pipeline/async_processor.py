"""Asynchronous processing module for medieval text processing pipeline.

This module handles all asynchronous processing workflows, including individual
record processing, batch processing, streaming modes, and incremental output
with order preservation. It provides comprehensive error handling and progress
monitoring for async operations.
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Callable, AsyncIterator, TYPE_CHECKING

from ai_ner_system.config import Settings
from ai_ner_system.llm import BatchProgress
from ai_ner_system.processing import ProcessingResult, BatchProcessingResult

from .stats import AsyncProcessingStats, ApplicationError

if TYPE_CHECKING:
    from .main_processor import MedievalTextProcessor


class AsyncProcessor:
    """Handles asynchronous processing workflows for medieval text processing.

    This class is responsible for executing async processing pipelines,
    including batch processing with fallback, streaming modes, incremental
    output, and comprehensive progress monitoring.
    """

    def  __init__(self, main_processor: 'MedievalTextProcessor') -> None:
        """Initialize async processor with reference to main processor.

        Args:
           main_processor: Main MedievalTextProcessor instance.
        """
        self.main_processor = main_processor

        # Initialize tracking for incremental output order preservation
        self._next_expected_batch_num = 1
        self._batch_result_queue: Dict[int, BatchProcessingResult] = {}

    @property
    def args(self):
        """Access to command line arguments via main processor."""
        return self.main_processor.args

    @property
    def reader(self):
        """Access to input reader via main processor."""
        return self.main_processor.reader

    @property
    def writer(self):
        """Access to output writer via main processor."""
        return self.main_processor.writer

    @property
    def processor(self):
        """Access to record processor via main processor."""
        return self.main_processor.processor

    @property
    def llm_client(self):
        """Access to LLM client via main processor."""
        return self.main_processor.llm_client

    @property
    def _incremental_mode(self) -> bool:
        """Check if incremental output mode is enabled."""
        return self.main_processor.incremental_mode


    async def process_all_records_async(
        self,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
        max_batch_wait_time: int = 86400, # 24 hours
        poll_interval: int = 30
    ) -> AsyncProcessingStats:
        """Process all records asynchronously with batch operations.

        This method provides async batch processing with real-time progress monitoring
        and comprehensive error handling. It automatically falls back to individual
        processing if batch processing is not available.

        Args:
            progress_callback: Optional callback for batch progress updates.
            max_batch_wait_time: Maximum time to wait for batch completion.
            poll_interval: Time between progress checks.

        Returns:
            AsyncProcessingStats with detailed processing information.

        Raises:
            ApplicationError: If processing fails completely.
        """
        if not self.reader or not self.processor:
            raise ApplicationError('Components not properly initialized for async processing')

        # Initialize statistics
        stats = AsyncProcessingStats(start_time=time.time())

        try:
            logging.info("Starting async streaming processing...")

            # check if the LLM client supports async batch processing and if batch processing is enabled
            if self.args.batch_size > 1 and self.llm_client.supports_async_batch():
                # Use async batch processing with streaming
                await self._process_records_streaming_async(
                    stats, progress_callback, max_batch_wait_time, poll_interval
                )
            else:
                # Use individual async processing with streaming
                await self._process_records_individual_async(stats)

            # Finalize statistics
            stats.end_time = time.time()
            stats.processing_time = stats.end_time - stats.start_time

            logging.info(
                f'Async streaming processing completed: {stats.processed_records}/{stats.total_records} '
                f'records ({stats.success_rate:.1f}% success rate) in {stats.processing_time:.2f}s'
            )

            return stats

        except Exception as e:
            stats.end_time = time.time()
            stats.processing_time = stats.end_time - stats.start_time
            error_msg = f"Async streaming processing failed: {e}"
            logging.error(error_msg, exc_info=True)
            raise ApplicationError(error_msg) from e

    async def _process_records_streaming_async(
        self,
        stats: AsyncProcessingStats,
        progress_callback: Optional[Callable[[BatchProgress], None]],
        max_wait_time: int,
        poll_interval: int
    ) -> None:
        """Process records using async streaming approach with batching

        Args:
            stats: Statistics object to update.
            progress_callback: Optional callback for progress updates.
            max_wait_time: Maximum time to wait for batch completion.
            poll_interval: Time between progress checks.
        """
        logging.info("Starting async streaming with batch processing, batch size: %d", self.args.batch_size)

        batch_records: List[Dict[str, str]] = [] # a batch of records to process together
        record_count = 0
        batch_num = 0

        # Track batch tasks with their order information using a map
        batch_tasks: Dict[int, asyncio.Task] = {}  # batch_num -> task
        # Limit to 5 concurrent batch processing tasks, otherwise it can reach 50 batch request limitation
        max_concurrent_batches = 5

        try:
            # Stream records asynchronously in batches and concurrently (coroutines) process them with order preservation
            async for record in self._async_stream_csv_records():
                batch_records.append(record)
                # count number of records
                record_count += 1

                # Update total count as we discover records
                stats.total_records = record_count

                # Process batch when it reaches the specified size, eg: 10, 100
                if len(batch_records) >= self.args.batch_size:
                    batch_num += 1 # batch_num starts from 1

                    # Create and start coroutine task with batch number tracking
                    batch_task = asyncio.create_task(
                        self._process_batch_with_order_async(
                            batch_records.copy(),
                            batch_num, # Use batch_num for tracking the order
                            progress_callback,
                            max_wait_time,
                            poll_interval
                        )
                    )

                    # Add the task to the tracking dictionary
                    batch_tasks[batch_num] = batch_task
                    batch_records = [] # Clear batch records after processing a batch

                    # Limit concurrent batch processing tasks by
                    # keep up max_concurrent_batches (5) tasks running at any time
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
                            oldest_batch_num
                        )

            # Process final batch if there are any remaining records
            if batch_records:
                batch_num += 1
                final_task = asyncio.create_task(
                    self._process_batch_with_order_async(
                        batch_records.copy(), batch_num, progress_callback,
                        max_wait_time, poll_interval
                    )
                )
                batch_tasks[batch_num] = final_task

            # Process any remaining batch tasks in ORDER
            for batch_num in sorted(batch_tasks.keys()):
                batch_task = batch_tasks[batch_num]
                batch_result = await batch_task
                # Add results to stats in order
                await self._add_batch_results_in_order(stats, batch_result, batch_num)

            # Final flush of any queued results (for incremental mode)
            if self._incremental_mode:
                await self._flush_queued_batch_results_async()

            logging.info('Async streaming processing completed with preserved order: %d records', record_count)

        except Exception as e:
            # Cancel remaining tasks
            for task in batch_tasks.values():
                if not task.done():
                    task.cancel()
            error_msg = f'Async streaming processing failed: {e}'
            logging.error(error_msg, exc_info=True)
            raise ApplicationError(error_msg) from e

    async def _async_stream_csv_records(self) -> AsyncIterator[Dict[str, str]]:
        """Asynchronously stream records from the CSV input file.

        Returns:
            AsyncIterator yielding records as dictionaries.
        """
        iterator = self.reader.stream_records()  # synchronous generator

        while True:
            # next(it, None) never raises StopIteration—it returns None at end
            record = await asyncio.to_thread(lambda it: next(it, None), iterator)
            if record is None:
                break
            yield record

    async def _process_batch_with_order_async(
            self,
            batch_records: List[Dict[str, str]],
            batch_num: int,
            progress_callback: Optional[Callable[[BatchProgress], None]],
            max_wait_time: int,
            poll_interval: int
    ) -> BatchProcessingResult:
        """Process a batch of records asynchronously with order tracking

        Args:
            batch_records: List of records to process in this batch.
            batch_num: Current batch number (starting from number 1).
            progress_callback: Optional callback for progress updates.
            max_wait_time: Maximum time to wait for batch completion.
            poll_interval: Time between progress checks.

        Returns:
            BatchProcessingResult.
        """

        logging.info('Processing batch %d with %d records (order-preserving)',
                     batch_num, len(batch_records))

        # Create progress callback for this batch
        batch_progress_callback = self._create_batch_progress_callback(
            batch_num, None, progress_callback  # None for total_batches since we don't know yet
        )

        try:
            # Process batch asynchronously
            batch_result = await self.processor.process_batch_async(
                batch_records,
                progress_callback=batch_progress_callback,
                max_wait_time=max_wait_time,
                poll_interval=poll_interval
            )

            logging.info(
                f'Async batch {batch_num} completed: {batch_result.successful_count} successful, '
                f'{batch_result.failed_count} failed in {batch_result.total_processing_time:.2f}s'
            )

            return batch_result

        except Exception as e:
            logging.error(f'Batch {batch_num} failed: {e}')
            # Create fallback results for this batch
            fallback_stats = AsyncProcessingStats()
            await self._fallback_to_individual_async_streaming(batch_records, fallback_stats)

            # Convert to BatchProcessingResult format
            fallback_batch_result = BatchProcessingResult(
                batch_id = f"fallback_batch_{batch_num}",
                results=fallback_stats.results,
                total_processing_time=0.0,
                successful_count=fallback_stats.processed_records,
                failed_count=fallback_stats.failed_records
            )

            return fallback_batch_result

    async def _add_batch_results_in_order(
        self,
        stats: AsyncProcessingStats,
        batch_result: BatchProcessingResult,
        batch_num: int
    ) -> None:
        """Add batch results to stats while preserving order and handle incremental output.

        Args:
            stats: Statistics object to update.
            batch_result: Result of the processed batch.
            batch_num: The batch number for tracking.
        """

        logging.info('Adding results for batch %d (expected: %d)',
                     batch_num, self._next_expected_batch_num)

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
            logging.info(f'Added results from batch {batch_num} to stats in order')

        logging.info('Added batch %d results: %d successful, %d failed',
                     batch_num, batch_result.successful_count, batch_result.failed_count)

    async def _flush_queued_batch_results_async(self) -> None:
        """Write queued batch results in order and remove from queue."""

        while self._next_expected_batch_num in self._batch_result_queue:
            # pop the next expected batch result
            batch_result = self._batch_result_queue.pop(self._next_expected_batch_num)
            # Write this batch's results immediately
            await self._write_batch_results_incremental_async(
                batch_result,
                self._next_expected_batch_num
            )
            self._next_expected_batch_num += 1

            logging.info(f'Flushed batch {self._next_expected_batch_num - 1} results to output files')

    async def _write_batch_results_incremental_async(
        self,
        batch_result: BatchProcessingResult,
        batch_num: int
    ) -> None:
        """Write batch results to output files incrementally

        Args:
            batch_result: Result of the processed batch.
            batch_num: The batch number for tracking.
        """
        try:
            successful_results = [r for r in batch_result.results if r.success and r.annotated_text.strip()]

            if not successful_results:
                logging.info('Batch %d: No successful results to write', batch_num)
                return

            # Prepare annotated data and entity metadata
            annotated_rows = []
            metadata_rows = []

            for result in successful_results:
                annotated_rows.append(result.annotated_text)
                # Convert entities to metadata records
                for entity in result.entities:
                    metadata_rows.append(entity.to_csv_row())

            # Determine output file paths
            output_text_file = self.args.output_text or Settings.OUTPUT_TEXT_FILE
            output_table_file = self.args.output_table or Settings.OUTPUT_TABLE_FILE

            # Define headers
            annotated_header = "Bindnr;Brevid;Tekst"
            metadata_header = (
                "Proper Noun;Type of Proper Noun;Preposition;Order of Occurrence in Doc;"
                "Brevid;Status/Occupation/Description;Gender;Language"
            )

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
                       annotated_rows
                   )
                )

                # Write metadata if we have entities
                if metadata_rows:
                    tg.create_task(
                        asyncio.to_thread(
                            self.writer.append_metadata_output,
                            output_table_file,
                            metadata_header,
                            metadata_rows
                        )
                    )
            logging.info('Batch %d: Wrote %d annotations and %d entities incrementally',
                     batch_num, len(annotated_rows), len(metadata_rows))

        except Exception as e:
            logging.error('Failed to write batch %d results incrementally: %s',
                          batch_num, e)
            # Don't raise - this is not critical enough to stop processing

    async def _process_records_individual_async(self, stats: AsyncProcessingStats) -> None:
        """Process records individually asynchronously using streaming

        Args:
            stats: Statistics object to update.
        """
        logging.info("Starting individual async streaming processing")

        try:
            # Process records with limited concurrency to avoid overwhelming the API
            semaphore = asyncio.Semaphore(5) # Limit to 5 concurrent requests

            async def process_single_record(record: Dict[str, str]) -> ProcessingResult:
                async with semaphore:
                    return await self.processor.process_record_async(record)

            # Create tasks for streaming records
            tasks: List[asyncio.Task] = []
            # Keep track of current chunk records
            current_chunk_records = []
            record_count = 0

            async for record in self._async_stream_csv_records():
                record_count += 1
                stats.total_records = record_count
                task = asyncio.create_task(process_single_record(record))
                tasks.append(task)
                current_chunk_records.append(record)

                # Process in chunks to avoid memory issues with large files
                if len(tasks) >= 50: # Process 50 records at a time
                    await self._process_task_chunk(tasks, current_chunk_records, stats)
                    tasks.clear()
                    current_chunk_records.clear()

            # Process remaining tasks
            if tasks:
                await self._process_task_chunk(tasks, current_chunk_records, stats)

        except Exception as e:
            logging.error(f'Individual async streaming processing failed: {e}', exc_info=True)
            raise ApplicationError(f'Individual async streaming processing failed: {e}') from e

    async def _process_task_chunk(
        self,
        tasks: List[asyncio.Task],
        chunk_records: List[Dict[str, str]],
        stats: AsyncProcessingStats
    ) -> None:
        """Process a chunk of async tasks

        Args:
            tasks: List of asyncio tasks to process.
            chunk_records: Original records corresponding to the tasks.
            stats: Statistics object to update.
        """
        try:
            # Gather results in the same order tasks were created
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results in the same order as input
            for i, result in enumerate(results):
                original_record = chunk_records[i]  # Get original record by index
                brevid = original_record.get("Brevid", "")  # Extract brevid

                if isinstance(result, Exception):
                    # Handle failed task
                    stats.failed_records += 1
                    # Create a failed ProcessingResult for consistency
                    failed_result = ProcessingResult(
                        record_id=f'failed_{brevid}',
                        brevid=brevid,
                        success=False,
                        error_message=f'Processing failed for Brevid {brevid}: {result}'
                    )
                    stats.results.append(failed_result)
                    logging.error(f'Task failed for Brevid {brevid} with exception: {result}')
                elif isinstance(result, ProcessingResult):
                    # Handle successful task
                    stats.results.append(result)
                    if result.success:
                        stats.processed_records += 1
                    else:
                        stats.failed_records += 1
                        logging.warning(f'Record {result.record_id} and Brevid {result.brevid} failed: {result.error_message}')

            logging.info(f'Processed chunk: {len(results)} tasks completed')

        except Exception as e:
            logging.error(f"Error processing task chunk: {e}", exc_info=True)
            # Update stats for failed chunk
            stats.failed_records += len(tasks)
            raise

    async def _fallback_to_individual_async_streaming(
        self,
        batch_records: List[Dict[str, str]],
        stats: AsyncProcessingStats
    ) -> None:
        """Fallback to individual async processing for a batch of records

        Args:
            batch_records: Records to process individually.
            stats: Statistics object to update.
        """
        logging.info(f"Falling back to individual async processing for {len(batch_records)} records")

        # Process records with limited concurrency
        semaphore = asyncio.Semaphore(3) # Lower concurrency for fallback

        async def process_single_record(record: Dict[str, str]) -> ProcessingResult:
            async with semaphore:
                return await self.processor.process_record_async(record)

        # Create tasks for all records in this batch
        tasks = [process_single_record(record) for record in batch_records]

        # Process all tasks by asyncio.gather() to preserve order
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results in original order and update statistics
        for i, result in enumerate(results):
            original_record = batch_records[i] # Get original record by index
            brevid = original_record.get("Brevid", "")  # Extract brevid

            if isinstance(result, Exception):
                # Handle failed task
                stats.failed_records += 1
                # Create a failed ProcessingResult for the exception
                failed_result = ProcessingResult(
                    record_id=f"failed_{brevid}",
                    brevid=brevid,
                    success=False,
                    error_message=f"Processing failed for Brevid {brevid}: {result}"
                )
                stats.results.append(failed_result)
                logging.warning(f'Fallback processing exception for Brevid {brevid}: {result}')
            elif isinstance(result, ProcessingResult):
                # Handle successful task
                stats.results.append(result) # Results added in original order
                if result.success:
                    stats.processed_records += 1
                else:
                    stats.failed_records += 1
                    logging.warning(f'Fallback processing failed for record {result.record_id} and Brevid {result.brevid}: {result.error_message}')

    def _create_batch_progress_callback(
        self,
        batch_num: int,
        total_batches: Optional[int],
        user_callback: Optional[Callable[[BatchProgress], None]] = None
    ) -> Callable[[BatchProgress], None]:
        """Create a progress callback for batch monitoring

        Args:
            batch_num: Current batch number (starting from number 1).
            total_batches: Total number of batches (None if unknown).
            user_callback: Optional user-defined callback for progress updates.

        Returns:
            Progress callback function that can be used to report batch progress.
        """
        def progress_callback(progress: BatchProgress) -> None:
            # Log batch progress
            counts = progress.request_counts
            if total_batches:
                batch_info = f'Batch {batch_num}/{total_batches}'
            else:
                batch_info = f'Batch {batch_num}'

            logging.info(
                f'{batch_info} ({progress.batch_id}): {progress.status.value} - '
                f'Processing: {counts.get("processing", 0)}, '
                f'Succeeded: {counts.get("succeeded", 0)}, '
                f'Errored: {counts.get("errored", 0)}, '
                f'Elapsed: {progress.elapsed_time:.1f}s'
            )

            # Call user-defined callback if available
            if user_callback:
                try:
                    user_callback(progress)
                except Exception as e:
                    logging.warning(f"Error in user progress callback: {e}")

        return progress_callback

