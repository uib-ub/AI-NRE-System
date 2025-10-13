"""Core processor class for medieval text processing pipeline.

This module contains the main MedievalTextProcessor class that orchestrates
the medieval text processing pipeline. It handles component initialization,
configuration, and provides the main entry points for both synchronous and
asynchronous processing workflows.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, ClassVar, Literal
from collections.abc import Callable

from ai_ner_system.config import Settings
from ai_ner_system.io import CSVReader, OutputWriter, CSVError, OutputError
from ai_ner_system.llm import create_llm_client, Client, LLMClientError, BatchProgress
from ai_ner_system.prompt import PromptBuilder, GenericPromptBuilder, PromptError
from ai_ner_system.processing import RecordProcessor

from .stats import ApplicationError, AsyncProcessingStats


class MedievalTextProcessor:
    """Main processor for medieval text analysis using Large Language Models.

    This class orchestrates the entire processing pipeline, from reading input CSV
    files to generating annotated output. It supports both individual record
    processing and batch processing, with automatic fallback mechanisms and
    comprehensive error handling.
    """

    # Class constants for output headers
    ANNOTATED_HEADER: ClassVar[str] = 'Bindnr;Brevid;Tekst'
    METADATA_HEADER: ClassVar[str] = (
        'Proper Noun;Type of Proper Noun;Preposition;Order of Occurrence in Doc;'
        'Brevid;Status/Occupation/Description;Gender;Language'
    )

    # Default timeout for async processing (24 hours in seconds)
    DEFAULT_ASYNC_TIMEOUT: ClassVar[int] = 3600 * 24

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the medieval text processor.

        Args:
            args: Parsed command line arguments.

        Raises:
            ApplicationError: If initialization fails.
        """
        self.args = args

        # Initialize incremental mode based on args
        self.incremental_mode = getattr(args, 'incremental_output', False)

        try:
            # Initialize all components (these will always be non-None after successful init)
            self.llm_client: Client = self._initialize_llm_client()
            self.prompt_builder: PromptBuilder = self._initialize_prompt_builder()
            self.processor: RecordProcessor = RecordProcessor(self.llm_client, self.prompt_builder)
            self.reader: CSVReader = self._initialize_csv_reader()
            self.writer: OutputWriter = OutputWriter()

            logging.info('MedievalTextProcessor initialized successfully')
            logging.info('Incremental output mode: %s', self.incremental_mode)

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

            # Use constant from Settings for supported clients validation
            if client_type not in Settings.SUPPORTED_CLIENTS:
                supported = ', '.join(sorted(Settings.SUPPORTED_CLIENTS))
                raise ApplicationError(
                    f'Unsupported client type: {client_type}. '
                    f'Supported types: {supported}'
                )

            llm_client = create_llm_client(client_type)
            logging.info('LLM client initialized: %s', client_type)
            return llm_client

        except LLMClientError as e:
            raise ApplicationError(
                f'Failed to initialize LLM client "{self.args.client}": {e}'
            ) from e
        except Exception as e:
            raise ApplicationError(
                f'Unexpected error initializing LLM client: {e}'
            ) from e

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
                template_file = self.args.batch_template or Settings.BATCH_TEMPLATE_FILE
                logging.info('Using batch prompt template: %s', template_file)
            else:
                template_file = self.args.prompt_template or Settings.PROMPT_TEMPLATE_FILE
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
            input_file = self.args.input or Settings.INPUT_FILE

            if not Path(input_file).exists():
                raise ApplicationError(f'Input file does not exist: {input_file}')

            reader = CSVReader(input_file, delimiter=';', encoding='utf-8')
            logging.info('CSV reader initialized for input file: %s', input_file)
            return reader

        except CSVError as e:
            raise ApplicationError(f'Failed to initialize CSV reader: {e}') from e
        except Exception as e:
            raise ApplicationError(f'Unexpected error initializing CSV reader: {e}') from e


    def _cleanup_output_files(self) -> None:
        """Cleanup existing output files before processing.

        This method removes existing output files to ensure a clean slate for new processing.
        It is called at the start of the run/run_async method.
        """
        try:
            output_text_file = self.args.output_text or Settings.OUTPUT_TEXT_FILE
            output_table_file = self.args.output_table or Settings.OUTPUT_TABLE_FILE
            output_stats_file = self.args.output_stats or Settings.OUTPUT_STATS_FILE

            # Clean up all output files if they exist
            self.writer.clean_output_files(
                output_text_file,
                output_table_file,
                output_stats_file
            )

        except Exception as e:
            logging.warning('Error during output file cleanup: %s', e)
            # Don't fail the entire process for cleanup issues


    def write_output(self, annotations: list[str], metadata: list[str]) -> None:
        """Write processed data to output files.

        Args:
            annotations: List of annotated text records.
            metadata: List of metadata records.

        Raises:
            ApplicationError: If output writing fails.
        """
        try:
            # Determine output file paths
            output_text = self.args.output_text or Settings.OUTPUT_TEXT_FILE
            output_table = self.args.output_table or Settings.OUTPUT_TABLE_FILE

            # Write annotated text output
            if annotations:
                # annotated_header = 'Bindnr;Brevid;Tekst'
                annotated_header = self.ANNOTATED_HEADER
                self.writer.write_text_output(
                    output_text, 
                    annotated_header, 
                    annotations
                )
                logging.info(
                    'Annotated text written to: %s (%d records)', 
                    output_text, 
                    len(annotations)
                )
            else:
                logging.warning('No annotated text output to write')

            # Write metadata table output
            if metadata:
                metadata_header = self.METADATA_HEADER
                self.writer.write_metadata_output(
                    output_table, 
                    metadata_header, 
                    metadata
                )
                logging.info(
                    'Metadata written to: %s (%d records)', 
                    output_table, 
                    len(metadata)
                )
            else:
                logging.warning('No metadata output to write')

        except OutputError as e:
            raise ApplicationError(f'Failed to write outputs: {e}') from e
        except Exception as e:
            raise ApplicationError(f'Unexpected error during output writing: {e}') from e


    async def write_output_async(self, stats: AsyncProcessingStats) -> None:
        """Write processing results from async operations

        Args:
            stats: Processing statistics containing results

        Raises:
            ApplicationError: If output writing fails.
        """
        # handle incremental output mode
        if self.incremental_mode:
            # In incremental mode, most output is already written
            # Just ensure final stats are written
            logging.info('Finalizing incremental output mode')
            await self._write_stats_async(stats)
            return

        try:
            logging.info('Writing output files asynchronously...')

            # Determine output file paths
            output_text = self.args.output_text or Settings.OUTPUT_TEXT_FILE
            output_table = self.args.output_table or Settings.OUTPUT_TABLE_FILE

            # Extract annotated texts and metadata from results
            annotated_records: list[str] = []
            metadata_records: list[str] = []

            for result in stats.results:
                if result.success and result.annotated_text:
                    annotated_records.append(result.annotated_text)
                    # Convert entities to metadata records
                    for entity in result.entities:
                        metadata_records.append(entity.to_csv_row())

            # Define headers
            annotated_header = self.ANNOTATED_HEADER 
            metadata_header = self.METADATA_HEADER

            # Use TaskGroup for better async task management
            # It automatically waits for all tasks to complete when exiting the context manager
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

            logging.info(
                'Text output written to: %s (%d records)', 
                output_text, 
                len(annotated_records)
            )
            logging.info(
                'Metadata output written to: %s (%d records)', 
                output_table, 
                len(metadata_records)
            )

        except* (OutputError, OSError) as eg:
            # Handle expected exception types specifically
            errors = [f'{type(e).__name__}: {e}' for e in eg.exceptions]
            raise ApplicationError(
                f'Failed to write async output. Errors: {"; ".join(errors)}'
            ) from eg
        except* Exception as eg:
            # Handle unexpected exceptions
            errors = [f'{type(e).__name__}: {e}' for e in eg.exceptions]
            logging.error('Unexpected errors during async output: %s', errors)
            raise ApplicationError(
                f'Unexpected errors writing async output: {"; ".join(errors)}'
            ) from eg


    async def _write_stats_async(self, stats: AsyncProcessingStats) -> None:
        """Write processing statistics to file

        Args:
            stats: Processing statistics to write.

        Note:
            Failures in stats writing are logged but not raised, as statistics
            writing is not critical to the main processing pipeline.
        """
        try:
            stats_output_file = self.args.output_stats or Settings.OUTPUT_STATS_FILE

            stats_data: dict[str, Any] = {
                'total_records': stats.total_records,
                'processed_records': stats.processed_records,
                'failed_records': stats.failed_records,
                'success_rate': stats.success_rate,
                'processing_time': stats.processing_time,
                'throughput': stats.throughput,
                'batch_info': stats.batch_info,
                'start_time': stats.start_time,
                'end_time': stats.end_time,
                'timestamp': time.time(),
                'processing_mode': 'async' if hasattr(self.args, 'async_mode') and self.args.async_mode else 'sync'
            }

            await asyncio.to_thread(
                self.writer.write_stats_output,
                stats_output_file,
                stats_data
            )
            logging.info('Statistics written to: %s', stats_output_file)

        except Exception as e:
            logging.warning('Failed to write processing statistics: %s', e)
            # Don't raise the exception - stats writing is not critical


    def run(self) -> Literal[0, 1]:
        """Run the complete processing pipeline synchronously.

        Returns:
            Exit code: 0 for success, 1 for failure.

        Raises:
            ApplicationError: If any step of the pipeline fails.
        """
        try:
            logging.info('Starting medieval text processing...')

            # Clean up existing output files first
            self._cleanup_output_files()

            # Process all records using synchronous processor
            from .sync_processor import SyncProcessor
            sync_processor = SyncProcessor(self)
            annotations, metadata = sync_processor.process_all_records()

            # Write output files
            self.write_output(annotations, metadata)

            logging.info('Processing completed successfully')

            # Final output paths
            output_text = self.args.output_text or Settings.OUTPUT_TEXT_FILE
            output_table = self.args.output_table or Settings.OUTPUT_TABLE_FILE
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


    async def run_async(
        self,
        progress_callback: Callable[[BatchProgress], None] | None = None,
        *,
        timeout: int | None = None,
    ) -> Literal[0, 1]:
        """Run the medieval text processor asynchronously.

        This method provides the async entry point for the application,
        with comprehensive error handling and progress monitoring.

        Args:
            progress_callback: Optional callback for batch progress updates.
                Called with BatchProgress objects during processing.
            timeout: Optional timeout in seconds. Defaults to 24 hours.
                Set to None for no timeout.

        Returns:
            Exit code: 0 for success, 1 for failure.
        """
        try:
            logging.info('Starting async medieval text processing...')

            # Clean up existing output files first
            self._cleanup_output_files()

            # Use timeout if specified (default to 24 hours)
            timeout_seconds = timeout if timeout is not None else self.DEFAULT_ASYNC_TIMEOUT

            # Use async context manager for better resource cleanup with timeout
            async with asyncio.timeout(timeout_seconds):  # 24-hour timeout
                # Process all records asynchronously using async processor
                from .async_processor import AsyncProcessor
                async_processor = AsyncProcessor(self)
                stats = await async_processor.process_all_records_async(progress_callback)

                # Write output files asynchronously
                await self.write_output_async(stats)

            logging.info(
                'Async processing completed successfully: %d/%d records (%d%% success) in %.2fs',
                stats.processed_records, 
                stats.total_records, 
                stats.success_rate, 
                stats.processing_time
            )
            return 0

        except asyncio.TimeoutError:
            logging.error('Processing timed out after 24 hours')
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