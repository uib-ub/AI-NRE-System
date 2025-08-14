"""Main application module for medieval text processing using Large Language Models.

This module provides the primary entry point and orchestration logic for processing
medieval texts with Named Entity Recognition capabilities. It supports both
synchronous and asynchronous processing modes with comprehensive error handling
and progress monitoring.
"""

import argparse
import asyncio
import time
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable, Any, AsyncIterator
from dataclasses import dataclass

from tqdm import tqdm

from ai_ner_system.config import Config, ConfigError
from ai_ner_system.llm_clients import (
    create_llm_client,
    Client,
    LLMClientError,
    BatchProgress
)
from ai_ner_system.prompts import PromptBuilder, GenericPromptBuilder, PromptError
from ai_ner_system.processing import (
    RecordProcessor,
    ValidationError,
    ProcessingError,
    LLMResponseError,
    ProcessingResult,
    create_progress_logger, BatchProcessingResult
)
from ai_ner_system.io_utils import CSVReader, OutputWriter, IOError


class ApplicationError(Exception):
    """Custom exception for application-level errors."""

@dataclass
class AsyncProcessingStats:
    """Statistics for async processing operations

    This class tracks comprehensive statistics during asynchronous processing,
    including timing, success rates, and detailed batch information.

    Attributes:
        total_records: Total number of records to process.
        processed_records: Number of successfully processed records.
        failed_records: Number of failed records.
        start_time: Processing start time.
        end_time: Processing end time (None if still running).
        processing_time: Total processing time in seconds.
        batch_info: Information about batch processing (if used).
        results: List of ProcessingResult objects for detailed tracking.
    """
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    processing_time: float = 0.0
    batch_info: Optional[Dict[str, Any]] = None
    results: List[ProcessingResult] = None

    def __post_init__(self):
        """Initialize results list if not provided."""
        if self.results is None:
            self.results = []

    @property
    def success_rate(self) -> float:
        """Calculate success rate of processing as percentage.

        Returns:
            Success rate as a percentage of processed records over total records.
        """
        if self.total_records == 0:
            return 0.0
        return (self.processed_records / self.total_records) * 100.0

    @property
    def is_complete(self) -> bool:
        """Check if processing is complete.

        Returns:
            True if processing has ended (end_time is not None), False otherwise.
        """
        return self.end_time is not None

    @property
    def throughput(self) -> float:
        """Calculate records processed per second.

        Returns:
            Throughput as records per second. Returns 0 if processing time is zero.
        """
        if self.processing_time == 0:
            return 0.0
        return self.processed_records / self.processing_time

class MedievalTextProcessor:
    """Main processor for medieval text analysis using Large Language Models.

    This class orchestrates the entire processing pipeline, from reading input CSV
    files to generating annotated output. It supports both individual record
    processing and batch processing, with automatic fallback mechanisms and
    comprehensive error handling.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the medieval text processor.

        Args:
            args: Parsed command line arguments.

        Raises:
            ApplicationError: If initialization fails.
        """
        self.args = args

        # Initialize components
        self.llm_client: Optional[Client] = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self.processor: Optional[RecordProcessor] = None
        self.reader: Optional[CSVReader] = None
        self.writer: Optional[OutputWriter] = None

        # Initialize incremental mode based on args
        self._incremental_mode = getattr(args, 'incremental_output', False)
        # Initialize tracking for incremental output order preservation
        self._next_expected_batch_num = 1
        self._batch_result_queue: Dict[int, BatchProcessingResult] = {}

        try:
            self.llm_client = self._initialize_llm_client()
            self.prompt_builder = self._initialize_prompt_builder()
            self.processor = RecordProcessor(self.llm_client, self.prompt_builder)

            self.reader = self._initialize_csv_reader()
            self.writer = OutputWriter()

            logging.info('MedievalTextProcessor initialized successfully')
            logging.info('Incremental output mode: %s', self._incremental_mode)

        except Exception as e:
            raise ApplicationError(f'Failed to initialize MedievalTextProcessor: {e}') from e

    def _initialize_llm_client(self) -> Client:
        """Initialize the LLM client based on command line arguments.

        Returns:
            Configured LLM client instance.

        Raises:
            ApplicationError: If LLM client initialization fails.
        """
        try:
            client_type = self.args.client.lower()

            if client_type not in ['claude', 'ollama']:
                raise ApplicationError(f'Unsupported client type: {client_type}')

            llm_client = create_llm_client(client_type)
            logging.info('LLM client initialized: %s', client_type)
            return llm_client

        except LLMClientError as e:
            raise ApplicationError(f'Failed to initialize LLM client "{self.args.client}": {e}') from e
        except Exception as e:
            raise ApplicationError(f'Unexpected error initializing LLM client: {e}') from e

    def _initialize_prompt_builder(self) -> PromptBuilder:
        """Initialize the prompt builder with template file.

        Returns:
            Configured PromptBuilder instance that can handle both single and batch processing.

        Raises:
            ApplicationError: If prompt builder initialization fails.
        """
        try:
            # Determine which template to use based on whether batch processing is enabled
            if self.args.use_batch:
                template_file = self.args.batch_template or Config.BATCH_TEMPLATE_FILE
                logging.info('Using batch prompt template: %s', template_file)
            else:
                template_file = self.args.prompt_template or Config.PROMPT_TEMPLATE_FILE
                logging.info('Using single prompt template: %s', template_file)

            prompt_builder = GenericPromptBuilder(template_file)
            logging.info('Prompt builder initialized with template: %s', template_file)
            return prompt_builder
        except PromptError as e:
            raise ApplicationError(f'Failed to initialize PromptBuilder: {e}') from e

    def _initialize_csv_reader(self) -> CSVReader:
        """Initialize the CSV reader for input file.

        Returns:
            Configured CSVReader instance.

        Raises:
            ApplicationError: If CSV reader initialization fails.
        """
        try:
            input_file = self.args.input or Config.INPUT_FILE

            if not Path(input_file).exists():
                raise ApplicationError(f"Input file does not exist: {input_file}")

            reader = CSVReader(input_file, delimiter=';', encoding='utf-8')
            logging.info('CSV reader initialized for input file: %s', input_file)
            return reader

        except IOError as e:
            raise ApplicationError(f'Failed to initialize CSV reader: {e}') from e
        except Exception as e:
            raise ApplicationError(f"Unexpected error initializing CSV reader: {e}") from e

    def _cleanup_output_files(self) -> None:
        """Cleanup existing output files before processing.

        This method removes existing output files to ensure a clean slate for new processing.
        It is called at the start of the run/run_async method.
        """
        try:
            output_text_file = self.args.output_text or Config.OUTPUT_TEXT_FILE
            output_table_file = self.args.output_table or Config.OUTPUT_TABLE_FILE
            output_stats_file = self.args.output_stats or Config.OUTPUT_STATS_FILE

            # Clean up all output files if they exist
            self.writer.clean_output_files(
                output_text_file,
                output_table_file,
                output_stats_file
            )

        except Exception as e:
            logging.warning('Error during output file cleanup: %s', e)
            # Don't fail the entire process for cleanup issues

    # ============================================================================
    # SYNCHRONOUS METHODS -  sync batch processing capabilities
    # ============================================================================
    def process_all_records(self) -> Tuple[List[str], List[str]]:
        """Process all records from the input CSV file using streaming approach.

        Returns:
            Tuple of (all_annotations, all_metadata).

        Raises:
            ApplicationError: If critical processing error occurs.
        """
        try:
            logging.info('Starting to process records from: %s', self.reader.file_path)

            batch_size = self.args.batch_size if self.args.use_batch else 1

            # Process records with unified streaming approach
            all_annotations, all_metadata = self._process_records_streaming(batch_size)
            logging.info('Completed processing all records')
            return all_annotations, all_metadata

        except Exception as e:
            raise ApplicationError(f'Critical error during file processing: {e}') from e

    def _process_records_streaming(self, batch_size: int) -> Tuple[List[str], List[str]]:
        """Process records using streaming approach with configurable batch size.

        Args:
            batch_size: Number of records to process together (1 = individual processing)

        Returns:
            Tuple of (all_annotations, all_metadata).
        """
        all_annotations: List[str] = []
        all_metadata: List[str] = []

        batch_records: List[Dict[str, str]] = [] # a batch of records to process together
        batch_count = 0 # counter for the number of batches processed

        processing_mode = "batch" if batch_size > 1 else "individual"

        logging.info('Using %s processing (batch_size=%d)', processing_mode, batch_size)

        try:
            for record in tqdm(self.reader.stream_records(), desc=f'Processing Records ({processing_mode} mode)'):
                batch_records.append(record)

                # Process when batch is full or for individual processing (batch_size=1)
                if len(batch_records) >= batch_size:
                    batch_count += 1

                    annotated_records, metadata_records = self._process_batch(batch_records, batch_count, batch_size)
                    # Collect results
                    all_annotations.extend(annotated_records)
                    all_metadata.extend(metadata_records)

                    logging.debug('Successfully processed %s: %d annotations, %d metadata',
                                f'Brevid {batch_records[0].get("Brevid", "unknown")}' if batch_size == 1 else f'batch {batch_count}',
                                len(annotated_records), len(metadata_records))

                    # Clear batch records after processing
                    batch_records = []
                    time.sleep(0.2)

            # Process any remaining records in the final partial batch
            if batch_records:
                batch_count += 1
                annotations, metadata = self._process_final_batch(batch_records, batch_count, batch_size, processing_mode)
                all_annotations.extend(annotations)
                all_metadata.extend(metadata)

            return all_annotations, all_metadata
        except Exception as e:
            logging.error(f"Streaming processing failed: {e}", exc_info=True)
            raise

    def _process_batch(
        self,
        batch_records: List[Dict[str, str]],
        batch_count: int,
        batch_size: int
    ) -> Tuple[List[str], List[str]]:
        """Process a batch of records, handling both individual and batch modes.

        Args:
            batch_records: The list of records to process.
            batch_count: The current batch number.
            batch_size: The size of the batch.

        Returns:
            A tuple containing a list of annotation strings and a list of metadata strings.

        Raises:
            Any exception raised by the underlying processor will be handled by _handle_batch_exception.
        """
        try:
            # Process the batch/record
            if batch_size == 1:
                # Individual processing
                individual_record = batch_records[0]
                brevid = individual_record.get("Brevid", "unknown")
                logging.info('Processing Record with Brevid: %s', brevid)
                logging.debug('Individual record data: %s', individual_record)
                return self.processor.process_record(individual_record)
            else:
                # Batch processing
                logging.info('Processing batch %d with %d records', batch_count, len(batch_records))
                return self.processor.process_batch(batch_records)
        except Exception as e:
            return self._handle_batch_exception(batch_records, batch_count, batch_size, e)

    def _handle_batch_exception(
        self,
        batch_records: List[Dict[str, str]],
        batch_count: int,
        batch_size: int,
        error: Exception
    ) -> Tuple[List[str], List[str]]:
        """Handle exceptions during batch processing, with fallback to individual processing if needed.

        Args:
            batch_records: The list of records in the batch.
            batch_count: The current batch number.
            batch_size: The size of the batch.
            error: The exception that was raised.

        Returns:
            A tuple containing a list of annotation strings and a list of metadata strings.
        """
        if batch_size == 1:
            # Individual processing error
            brevid = batch_records[0].get("Brevid", "unknown")
            logging.error('Error processing Brevid %s: %s', brevid, error)
            self._handle_individual_error(batch_records[0], error)
            return [], []
        else:
            # Batch processing error - fallback to individual processing
            logging.error('Error processing batch %d: %s', batch_count, error)
            logging.info('Falling back to individual processing for batch %d', batch_count)
            annotations: List[str] = []
            metadata: List[str] = []
            # Process each record in the failed batch individually
            for record in batch_records:
                annotated_record, metadata_record = self._fallback_to_individual_processing(record)
                annotations.extend(annotated_record)
                metadata.extend(metadata_record)
            return annotations, metadata

    def _process_final_batch(
        self,
        batch_records: List[Dict[str, str]],
        batch_count: int,
        batch_size: int,
        processing_mode: str
    ) -> Tuple[List[str], List[str]]:
        """Process the final (possibly partial) batch of records.

        Args:
            batch_records: The list of records in the final batch.
            batch_count: The current batch number.
            batch_size: The size of the batch.
            processing_mode: The processing mode ("batch" or "individual").

        Returns:
            A tuple containing a list of annotation strings and a list of metadata strings.
        """
        annotations: List[str] = []
        metadata: List[str] = []
        with tqdm(total=len(batch_records), desc=f'Final {processing_mode} batch') as final_pbar:
            try:
                if batch_size == 1:
                    # This shouldn't happen since we process immediately, but we still handle it
                    for record in batch_records:
                        brevid = record.get("Brevid", "unknown")
                        annotated_records, metadata_records = self.processor.process_record(record)
                        annotations.extend(annotated_records)
                        metadata.extend(metadata_records)
                        final_pbar.set_description(f'Final record: {brevid}')
                        final_pbar.update(1)
                else:
                    # Process the final batch
                    logging.info('Processing final batch %d with %d records', batch_count, len(batch_records))
                    annotated_records, metadata_records = self.processor.process_batch(batch_records)
                    annotations.extend(annotated_records)
                    metadata.extend(metadata_records)
                    final_pbar.set_description(f'Final batch ({len(batch_records)} records)')
                    final_pbar.update(len(batch_records))
                    logging.debug('Successfully processed final batch: %d annotations, %d metadata',
                                  len(annotated_records), len(metadata_records))
            except Exception as e:
                logging.error('Error processing final batch: %s', e)
                # Fallback to individual processing for remaining records
                final_pbar.set_description('Final batch fallback')
                for record in batch_records:
                    annotated_record, metadata_record = self._fallback_to_individual_processing(record)
                    annotations.extend(annotated_record)
                    metadata.extend(metadata_record)
                    final_pbar.update(1)
        return annotations, metadata

    def _fallback_to_individual_processing(self, record: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """Fallback to individual processing when doing a batch of records.

        Args:
            record: record that failed to process in a batch processing.

        Returns:
            Tuple of (annotated_record, metadata_record) for the individual record.
        """
        logging.info('Falling back to individual processing for record: %s', record)

        annotated_record, metadata_record = [], []

        try:
            annotated_record, metadata_record = self.processor.process_record(record)
        except Exception as e:
            brevid = record.get("Brevid", "unknown")
            logging.error('Error in fallback processing for Brevid %s: %s', brevid, e)
            self._handle_individual_error(record, e)

        return annotated_record, metadata_record

    def _handle_individual_error(self, record: Dict[str, str], exception: Exception) -> None:
        """Handle errors that occur during individual record processing.

        Args:
            record: The record that failed to process.
            exception: The exception that occurred.
        """
        brevid = record.get("Brevid", "unknown")
        bindnr = record.get('Bindnr', 'unknown')

        if isinstance(exception, ValidationError):
            logging.error('Validation error for Brevid %s (Bindnr: %s): %s', brevid, bindnr, exception)
        elif isinstance(exception, (ProcessingError, LLMResponseError)):
            logging.error('LLM Processing error for Brevid %s (Bindnr: %s): %s', brevid, bindnr, exception)
        else:
            logging.error('Unexpected error processing Brevid %s (Bindnr: %s): %s', brevid, bindnr, exception, exc_info=True)

    def write_output(self, annotations: List[str], metadata: List[str]) -> None:
        """Write processed data to output files.

        Args:
            annotations: List of annotated text records.
            metadata: List of metadata records.

        Raises:
            ApplicationError: If output writing fails.
        """

        try:
            # Determine output file paths
            output_text = self.args.output_text or Config.OUTPUT_TEXT_FILE
            output_table = self.args.output_table or Config.OUTPUT_TABLE_FILE

            # Write annotated text output
            if annotations:
                annotated_header = "Bindnr;Brevid;Tekst"
                self.writer.write_text_output(output_text, annotated_header, annotations)
                logging.info('Annotated text written to: %s (%d records)', output_text, len(annotations))
            else:
                logging.warning('No annotated text output to write')

            # Write metadata table output
            if metadata:
                metadata_header = (
                    "Proper Noun;Type of Proper Noun;Preposition;Order of Occurrence in Doc;"
                    "Brevid;Status/Occupation/Description;Gender;Language"
                )
                self.writer.write_metadata_output(output_table, metadata_header, metadata)
                logging.info('Metadata written to: %s (%d records)', output_table, len(metadata))
            else:
                logging.warning('No metadata output to write')

        except IOError as e:
            raise ApplicationError(f'Failed to write outputs: {e}') from e
        except Exception as e:
            raise ApplicationError(f'Unexpected error during output writing: {e}') from e

    def run(self) -> int:
        """Run the complete processing pipeline.

        Raises:
            ApplicationError: If any step of the pipeline fails.

        Returns:
            Exit code (0 for success, 1 for failure).
        """
        try:
            logging.info('Starting medieval text processing...')

            # Clean up existing output files first
            self._cleanup_output_files()

            # Process all records
            annotations, metadata = self.process_all_records()

            # Write output files
            self.write_output(annotations, metadata)

            logging.info('Processing completed successfully')

            # Final output paths
            output_text = self.args.output_text or Config.OUTPUT_TEXT_FILE
            output_table = self.args.output_table or Config.OUTPUT_TABLE_FILE
            print(f'\nOutputs written to:')
            print(f'  Annotated text: {output_text}')
            print(f'  Metadata table: {output_table}')

            return 0

        except ApplicationError as e:
            logging.error('Application error: %s', e, exc_info=True)
            return 1
        except KeyboardInterrupt:
            logging.info('Processing interrupted by user.')
            return 1
        except Exception as e:
            logging.error('Unexpected error: %s', e, exc_info=True)
            return 1

    # ============================================================================
    # ASYNCHRONOUS METHODS -  async batch processing capabilities
    # ============================================================================
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
            # TODO: need to revise code for self.args.use_batch
            if ( # self.args.use_batch and
                self.args.batch_size > 1 and
                self.llm_client.supports_async_batch()):
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
            output_text_file = self.args.output_text or Config.OUTPUT_TEXT_FILE
            output_table_file = self.args.output_table or Config.OUTPUT_TABLE_FILE

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

    async def write_output_async(self, stats: AsyncProcessingStats) -> None:
        """Write processing results from async operations

        Args:
            stats: Processing statistics containing results

        Raises:
            ApplicationError: If output writing fails.
        """
        # handle incremental output mode
        if self._incremental_mode:
            # In incremental mode, most output is already written
            # Just ensure final stats are written
            logging.info('Finalizing incremental output mode')
            await self._write_stats_async(stats)
            return

        try:
            logging.info('Writing output files...')

            # Determine output file paths
            output_text = self.args.output_text or Config.OUTPUT_TEXT_FILE
            output_table = self.args.output_table or Config.OUTPUT_TABLE_FILE

            # Define headers
            annotated_header = "Bindnr;Brevid;Tekst"
            metadata_header = (
                "Proper Noun;Type of Proper Noun;Preposition;Order of Occurrence in Doc;"
                "Brevid;Status/Occupation/Description;Gender;Language"
            )

            # Extract annotated texts and metadata from results
            annotated_records = []
            metadata_records = []

            for result in stats.results:
                if result.success and result.annotated_text:
                    annotated_records.append(result.annotated_text)
                    # Convert entities to metadata records
                    for entity in result.entities:
                        metadata_records.append(entity.to_csv_row())

            # Use TaskGroup for better async task management, and it
            # automatically waits for all tasks to complete when exiting the context manager
            async with asyncio.TaskGroup() as tg:
                # write output files asynchronously using OutputWriter methods
                tg.create_task(asyncio.to_thread(
                    self.writer.write_text_output,
                    output_text,
                    annotated_header,
                    annotated_records
                ))

                tg.create_task(asyncio.to_thread(
                    self.writer.write_metadata_output,
                    output_table,
                    metadata_header,
                    metadata_records
                ))

                # Write processing statistics, ASYNC method, no asyncio.to_thread needed
                tg.create_task(self._write_stats_async(stats))

            logging.info(f'Text output written to: {output_text} ({len(annotated_records)} records)')
            logging.info(f'Metadata output written to: {output_table} ({len(metadata_records)} records)')

        except* Exception as eg: # Exception groups handling
            # Handle exception group
            errors = [str(e) for e in eg.exceptions]
            raise ApplicationError(f'Failed to write async output. Errors: {errors}') from eg

    async def _write_stats_async(self, stats: AsyncProcessingStats) -> None:
        """Write processing statistics to file

        Args:
            stats: Processing statistics to write.
        """
        try:
            stats_output_file = self.args.output_stats or Config.OUTPUT_STATS_FILE

            stats_data = {
                "total_records": stats.total_records,
                "processed_records": stats.processed_records,
                "failed_records": stats.failed_records,
                "success_rate": stats.success_rate,
                "processing_time": stats.processing_time,
                "throughput": stats.throughput,
                "batch_info": stats.batch_info,
                "start_time": stats.start_time,
                "end_time": stats.end_time,
                "timestamp": time.time(),
                "processing_mode": "async" if hasattr(self.args, 'async_mode') and self.args.async_mode else "sync"
            }

            await asyncio.to_thread(
                self.writer.write_stats_output,
                stats_output_file,
                stats_data
            )
        except Exception as e:
            logging.warning(f"Failed to write processing statistics: {e}")
            # Don't raise the exception - stats writing is not critical

    async def run_async(
        self,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
    ) -> int:
        """Run the medieval text processor asynchronously

        This method provides the async entry point for the application,
        with comprehensive error handling and progress monitoring.

        Args:
            progress_callback: Optional callback for batch progress updates.

        returns:
            Exit code (0 for success, 1 for failure).
        """
        try:
            logging.info("Starting async medieval text processing...")

            # Clean up existing output files first
            self._cleanup_output_files()

            # Use async context manager for better resource cleanup with timeout
            async with asyncio.timeout(3600 * 24): # 24-hour timeout
                # Process all records asynchronously using streaming
                stats = await self.process_all_records_async(progress_callback)

                # Write output files asynchronously
                await self.write_output_async(stats)

            logging.info(
                'Async processing completed successfully: '
                f'{stats.processed_records}/{stats.total_records} records '
                f'({stats.success_rate:.1f}% success) in {stats.processing_time:.2f}s'
            )
            return 0

        except asyncio.TimeoutError:
            logging.error("Processing timed out after 24 hours")
            return 1
        except ApplicationError as e:
            logging.error('Application error: %s', e, exc_info=True)
            return 1
        except KeyboardInterrupt:
            logging.info('Processing interrupted by user.')
            return 1
        except Exception as e:
            logging.error('Unexpected error during async processing: %s', e, exc_info=True)
            return 1

# ============================================================================
# Utility functions
# ============================================================================

def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments.

    Args:
        args: Parsed command line arguments.

    Raises:
        ValueError: If arguments are invalid.
    """
    # Validate input files
    input_file = args.input or Config.INPUT_FILE
    input_path = Path(input_file)

    if not input_path.exists():
        raise ApplicationError(f'Input file does not exist: {input_path}')
    if not input_path.is_file():
        raise ApplicationError(f'Input path is not a file: {input_path}')

    # if input_file and not Path(input_file).exists():
    #     raise ValueError(f'Input file does not exist: {input_file}')

    # Validate output directories
    for output_file in [args.output_text, args.output_table]:
        output_path = Path(output_file)
        output_dir = output_path.parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logging.info('Created output directory: %s', output_dir)
            except OSError as e:
                raise ApplicationError(f'Failed to create output directory {output_dir}: {e}') from e

    # Check if prompt template exists
    if args.prompt_template and not Path(args.prompt_template).exists():
        raise ValueError(f'Prompt template file does not exist: {args.prompt_template}')

    # Check batch template if batch processing is enabled
    if args.use_batch:
        if args.batch_template and not Path(args.batch_template).exists():
            raise ValueError(f'Batch template file does not exist: {args.batch_template}')

    # Validate client type
    if args.client.lower() not in ['claude', 'ollama']:
        raise ApplicationError(f'Unsupported client type: {args.client}')

    # Validate async-specific arguments
    if hasattr(args, 'async_mode') and args.async_mode:
        if hasattr(args, 'max_wait_time') and args.max_wait_time <= 60:
            raise ApplicationError(f'Max wait time must be at least 60 seconds for async mode, got {args.max_wait_time} seconds')
        if hasattr(args, 'poll_interval') and args.poll_interval <= 5:
            raise ApplicationError(f'Poll interval must be at least 5 seconds for async mode, got {args.poll_interval} seconds')

    logging.info('Command line arguments validated successfully')


def validate_configuration() -> None:
    """Validate application configuration.

    Raises:
        ConfigError: If configuration is invalid.
    """
    try:
        if not Config.is_valid():
            Config.validate_required_config()
            Config.validate_file_paths()
    except ConfigError as e:
        raise ApplicationError(f'Configuration validation failed: {e}') from e

    logging.info('Configuration validated successfully')

def setup_logging(level: str = 'INFO') -> None:
    """Setup application logging

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure logging format
    log_format = '%(asctime)s %(name)s [%(levelname)s]: %(message)s'
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

    logging.info('Logging configured (level=%s)', level)

    # Set specific loggers to appropriate levels
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description='Medieval Text Processor - Process medieval texts using Large Language Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
                Examples:
                    # Process with sync mode
                    python -m ai_ner_system.main --client claude/ollama --input input/input.txt --output-text output/annotated_output.txt --output-table output/metadata_table.txt --use-batch --batch-size 10 -l DEBUG
                
                    # Process with async batch processing
                    python -m ai_ner_system.main --client claude --input data.csv --batch-size 20 --async
                    
                    uv run src/ai_ner_system/main.py --client claude --output-text output/annotated_output_claude_batch_100R_B100_async.txt --output-table output/metadata_table_claude_batch_100R_B100_async.txt --output-stats output/stats_claude_batch_100R_B100_async.txt  -l DEBUG --batch-size 100 --async
                    uv run src/ai_ner_system/main.py --client claude --output-text output/annotated_output_claude_batch_13R_B2_async.txt --output-table output/metadata_table_claude_batch_13R_B2_async.txt --output-stats output/stats_claude_batch_13R_B2_async.txt  -l DEBUG --batch-size 2 --async --incremental-output
                """
    )

    # Model client selection
    parser.add_argument(
        '--client', '-c',
        type=str,
        choices=['claude', 'ollama'],
        default='claude',
        help='Select LLM Client (default: claude)'
    )

    # IO File file arguments
    parser.add_argument(
        '--input',
        type=str,
        default=Config.INPUT_FILE,
        help='Path to the input file'
    )

    parser.add_argument(
        '--output-text',
        type=str,
        default=Config.OUTPUT_TEXT_FILE,
        help='Path for annotated text output'
    )

    parser.add_argument(
        '--output-table',
        type=str,
        default=Config.OUTPUT_TABLE_FILE,
        help='Path for metadata table output'
    )

    parser.add_argument(
        "--output-stats",
        type=str,
        default=Config.OUTPUT_STATS_FILE,
        help="Output file for processing statistics (JSON format)"
    )

    # Batch processing options
    parser.add_argument(
        '--use-batch',
        action='store_true',
        help='Enable batch processing for better performance'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Number of records to process in each batch (default: 5)'
    )

    parser.add_argument(
        '--prompt-template',
        type=str,
        default=Config.PROMPT_TEMPLATE_FILE,
        help='Path to the prompt template file'
    )

    parser.add_argument(
        '--batch-template',
        type=str,
        default=Config.BATCH_TEMPLATE_FILE,
        help='Path to the batch template file'
    )

    # Async processing arguments
    parser.add_argument(
        '--async-mode', '-a',
        action='store_true',
        dest='async_mode',
        help='Enable asynchronous processing for batch operations'
    )

    # Maximum number of concurrent batches
    parser.add_argument(
        '--max-concurrent-batches',
        type=int,
        default=5,
        help='Maximum number of concurrent batches (default: 5)'
    )

    # Incremental output option
    parser.add_argument(
        '--incremental-output',
        action='store_true',
        help='Write outputs incrementally after each batch (useful for large datasets)'
    )

    parser.add_argument(
        '--max-wait-time',
        type=int,
        default=86400,  # 24 hours
        help='Maximum time to wait for async batch completion (default: 86400 seconds, i.e. 24 hours)'
    )

    parser.add_argument(
        '--poll-interval',
        type=int,
        default=30,
        help='Time between progress checks for async processing (default: 30s)'
    )

    # Utility arguments
    # Logging
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )

    # Additional options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and inputs without processing'
    )

    return parser

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------

def main() -> int:
    """Main application entry point.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # Add this at the very beginning
    # root_logger = logging.getLogger()
    # print(f"Handlers before setup_logging: {root_logger.handlers}")
    # print(f"Root logger level before setup_logging: {root_logger.level}")

    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()

        # Setup logging
        setup_logging(args.log_level)

        # Validate configuration and arguments
        validate_arguments(args)
        validate_configuration()

        # Handle dry run
        if args.dry_run:
            print('✓ Configuration validated successfully')
            print('✓ Command line arguments validated')
            print('✓ Input files exist and are accessible')
            print('Dry run completed successfully - no processing performed')
            return 0

        # Initialize and run processor instance
        processor = MedievalTextProcessor(args)
        # processor.run()

        # Choose sync or async execution
        match getattr(args, 'async_mode', False):
            case True:
                logging.info('Using asynchronous processing mode')

                # Create progress callback
                progress_callback = create_progress_logger(60) # Log every 60 seconds

                # Run async processing
                return asyncio.run(processor.run_async(progress_callback))
            case False:
                logging.info('Using synchronous processing mode')
                # Run synchronous processing
                return processor.run()
            case _:
                logging.error(
                    f"Invalid value for async_mode: {args.async_mode}. "
                    "Please set it to either True or False."
                )
                return 1  # Non-zero exit code to indicate configuration error

    except KeyboardInterrupt:
        logging.error('Processing interrupted by user')
        return 1
    except ApplicationError as e:
        logging.error('Application error: %s', e)
        return 1
    except Exception as e:
        logging.error('Unexpected error: %s', e, exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
