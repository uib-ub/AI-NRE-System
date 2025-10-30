"""Core processor class for medieval text processing pipeline.

This module contains the main MedievalTextProcessor class that orchestrates
the medieval text processing pipeline. It handles component initialization,
configuration, and provides the main entry points for both synchronous and
asynchronous processing workflows.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NoReturn

from ai_ner_system.config import Settings
from ai_ner_system.file_io import CSVError, CSVReader, OutputError, OutputWriter
from ai_ner_system.llm import BatchProgress, Client, LLMClientError, create_llm_client
from ai_ner_system.processing import RecordProcessor
from ai_ner_system.prompt import GenericPromptBuilder, PromptBuilder, PromptError

from .stats import ApplicationError, AsyncProcessingStats

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable


class MedievalTextProcessor:
    """Main processor for medieval text analysis using Large Language Models.

    This class orchestrates the entire processing pipeline, from reading input CSV
    files to generating annotated output. It supports both individual record
    processing and batch processing, with automatic fallback mechanisms and
    comprehensive error handling.
    """

    # Class constants for output headers
    ANNOTATED_HEADER: ClassVar[str] = "Bindnr;Brevid;Tekst"
    METADATA_HEADER: ClassVar[str] = (
        "Proper Noun;Type of Proper Noun;Preposition;Order of Occurrence in Doc;"
        "Brevid;Status/Occupation/Description;Gender;Language"
    )
    # Default timeout for async processing (24 hours in seconds)
    DEFAULT_ASYNC_TIMEOUT: ClassVar[float] = 3600.0 * 24.0

    @property
    def output_text_file(self) -> str:
        """Get output text file path."""
        return self.args.output_text or Settings.OUTPUT_TEXT_FILE

    @property
    def output_table_file(self) -> str:
        """Get output table file path."""
        return self.args.output_table or Settings.OUTPUT_TABLE_FILE

    @property
    def output_stats_file(self) -> str:
        """Get output stats file path."""
        return self.args.output_stats or Settings.OUTPUT_STATS_FILE

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the medieval text processor.

        Args:
            args: Parsed command line arguments.

        Raises:
            ApplicationError: If initialization fails.
        """
        self.args = args

        # Initialize incremental mode based on args
        self.incremental_mode = args.incremental_output

        try:
            # Initialize all components (these will always be non-None after successful init)
            self.llm_client: Client = self._initialize_llm_client()
            self.prompt_builder: PromptBuilder = self._initialize_prompt_builder()
            self.processor: RecordProcessor = RecordProcessor(
                self.llm_client,
                self.prompt_builder,
            )
            self.reader: CSVReader = self._initialize_csv_reader()
            self.writer: OutputWriter = OutputWriter()

            logging.info("MedievalTextProcessor initialized successfully")
            logging.info("Incremental output mode: %s", self.incremental_mode)

        except Exception as e:
            raise ApplicationError(
                f"Failed to initialize MedievalTextProcessor: {e}",
            ) from e

    def _initialize_llm_client(self) -> Client:
        """Initialize the LLM client based on command line arguments.

        Returns:
            Configured LLM client instance.

        Raises:
            ApplicationError: If LLM client initialization fails.
        """

        def _fail(msg: str, *, cause: Exception | None = None) -> NoReturn:
            if cause is None:
                raise ApplicationError(msg)
            raise ApplicationError(msg) from cause

        try:
            llm_client = create_llm_client(self.args.client)
        except LLMClientError as e:
            _fail(f'Failed to initialize LLM client "{self.args.client}"', cause=e)
        except Exception as e:  # noqa: BLE001 — boundary: convert unknown init errors
            _fail("Unexpected error got when initializing LLM client", cause=e)
        else:
            logging.info("LLM client initialized: %s", self.args.client)
            return llm_client

    def _initialize_prompt_builder(self) -> PromptBuilder:
        """Initialize the prompt builder with template file.

        Returns:
            Configured PromptBuilder instance that can handle both single and batch processing.

        Raises:
            ApplicationError: If prompt builder initialization fails.
        """
        # Determine which template to use based on whether batch processing is enabled
        if self.args.use_batch:
            template_file = self.args.batch_template or Settings.BATCH_TEMPLATE_FILE
            logging.info("Using batch prompt template: %s", template_file)
        else:
            template_file = self.args.prompt_template or Settings.PROMPT_TEMPLATE_FILE
            logging.info("Using single prompt template: %s", template_file)

        try:
            prompt_builder = GenericPromptBuilder(template_file)
        except PromptError as e:
            msg = "Failed to initialize PromptBuilder"
            raise ApplicationError(msg) from e

        logging.info("Prompt builder initialized with template: %s", template_file)
        return prompt_builder

    def _initialize_csv_reader(self) -> CSVReader:
        """Initialize the CSV reader for input file.

        Returns:
            Configured CSVReader instance.

        Raises:
            ApplicationError: If CSV reader initialization fails.
        """

        def _fail(msg: str, *, cause: Exception | None = None) -> NoReturn:
            if cause is None:
                raise ApplicationError(msg)
            raise ApplicationError(msg) from cause

        input_file = self.args.input or Settings.INPUT_FILE

        if not Path(input_file).exists():
            _fail(f"Input file does not exist: {input_file}")

        try:
            reader = CSVReader(input_file, delimiter=";", encoding="utf-8")
        except CSVError as e:
            _fail("Failed to initialize CSV reader", cause=e)
        except Exception as e:  # noqa: BLE001 — boundary: convert unknown init errors
            _fail("Unexpected error got when initializing CSV reader", cause=e)
        else:
            logging.info("CSV reader initialized for input file: %s", input_file)
            return reader

    def _cleanup_output_files(self) -> None:
        """Cleanup existing output files before processing.

        This method removes existing output files to ensure a clean slate for new processing.
        It is called at the start of the run/run_async method.
        """
        # Clean up all output files if they exist
        self.writer.clean_output_files(
            self.output_text_file,
            self.output_table_file,
            self.output_stats_file,
        )

    def write_output(self, annotations: list[str], metadata: list[str]) -> None:
        """Write processed data to output files.

        Args:
            annotations: List of annotated text records.
            metadata: List of metadata records.

        Raises:
            ApplicationError: If output writing fails.
        """
        errors: list[tuple[str, Exception]] = []

        # Write annotated text output
        if annotations:
            try:
                self.writer.write_text_output(
                    self.output_text_file,
                    self.ANNOTATED_HEADER,
                    annotations,
                )
                logging.info(
                    "Annotated text written to: %s (%d records)",
                    self.output_text_file,
                    len(annotations),
                )
            except OutputError as e:
                logging.exception(
                    "Failed writing annotations to %s",
                    self.output_text_file,
                )
                errors.append(("annotations", e))
        else:
            logging.warning("No annotated text output to write")

        # Write metadata table output
        if metadata:
            try:
                self.writer.write_metadata_output(
                    self.output_table_file,
                    self.METADATA_HEADER,
                    metadata,
                )
                logging.info(
                    "Metadata written to: %s (%d records)",
                    self.output_table_file,
                    len(metadata),
                )
            except OutputError as e:
                logging.exception(
                    "Failed writing metadata to %s",
                    self.output_table_file,
                )
                errors.append(("metadata", e))
        else:
            logging.warning("No metadata output to write")

        if errors:
            # Build a compact message without interpolating exceptions into the raise
            if len(errors) == 1:
                error_type, exc = errors[0]
                msg = f"Failed to write {error_type} output"
                raise ApplicationError(msg) from exc
            # multi-failure case
            error_types = ", ".join(et for et, _ in errors)
            first_exc = errors[0][1]
            msg = f"Failed to write multiple outputs: {error_types}"
            raise ApplicationError(msg) from first_exc

    async def write_output_async(self, stats: AsyncProcessingStats) -> None:
        """Write processing results from async operations.

        Args:
            stats: Processing statistics containing results

        Raises:
            ApplicationError: If output writing fails.
        """
        # handle incremental output mode
        if self.incremental_mode:
            # In incremental mode, most output is already written
            # Just ensure final stats are written
            logging.info("Finalizing incremental output mode")
            await self._write_stats_async(stats)
            return

        logging.info("Writing output files asynchronously...")

        # Extract annotated texts and metadata from results
        annotated_records: list[str] = [
            res.annotated_text
            for res in stats.results
            if res.success and res.annotated_text
        ]

        metadata_records: list[str] = [
            entity.to_csv_row()
            for res in stats.results
            if res.success
            for entity in res.entities
        ]

        try:
            # Use TaskGroup for better async task management
            # It automatically waits for all tasks to complete when exiting the context manager
            async with asyncio.TaskGroup() as tg:
                # write output files asynchronously using OutputWriter methods
                tg.create_task(
                    asyncio.to_thread(
                        self.writer.write_text_output,
                        self.output_text_file,
                        self.ANNOTATED_HEADER,
                        annotated_records,
                    ),
                )
                tg.create_task(
                    asyncio.to_thread(
                        self.writer.write_metadata_output,
                        self.output_table_file,
                        self.METADATA_HEADER,
                        metadata_records,
                    ),
                )
                # Write processing statistics, ASYNC method, no asyncio.to_thread needed
                tg.create_task(self._write_stats_async(stats))
        except* asyncio.CancelledError:
            # Propagate cooperative cancellation
            logging.info("write_output_async cancelled")
            raise
        except* (OutputError, OSError, UnicodeEncodeError) as eg:
            # Handle expected exception types specifically
            details = "; ".join(f"{type(e).__name__}: {e}" for e in eg.exceptions)
            msg = "Failed to write async output"
            logging.exception("%s: %s", msg, details)
            raise ApplicationError(msg) from eg
        except* Exception as eg:
            # Handle unexpected exceptions
            details = "; ".join(f"{type(e).__name__}: {e}" for e in eg.exceptions)
            msg = "Unexpected errors writing async output"
            logging.exception("%s: %s", msg, details)
            raise ApplicationError(msg) from eg

        logging.info(
            "Text output written to: %s (%d records)",
            self.output_text_file,
            len(annotated_records),
        )
        logging.info(
            "Metadata output written to: %s (%d records)",
            self.output_table_file,
            len(metadata_records),
        )

    async def _write_stats_async(self, stats: AsyncProcessingStats) -> None:
        """Write processing statistics to file.

        Args:
            stats: Processing statistics to write.

        Note:
            Failures in stats writing are logged but not raised, as statistics
            writing is not critical to the main processing pipeline.
        """
        stats_data: dict[str, Any] = self._build_stats_data(stats)

        try:
            await asyncio.to_thread(
                self.writer.write_stats_output,
                self.output_stats_file,
                stats_data,
            )
        except asyncio.CancelledError:
            logging.info("Stats write cancelled")
            raise
        except (OutputError, OSError, UnicodeEncodeError, ValueError, TypeError) as e:
            # Non-critical: log and continue
            logging.warning(
                "Failed to write processing statistics to %s: %s",
                self.output_stats_file,
                e,
                exc_info=True,
            )
        else:
            logging.info(
                "Statistics written to: %s successfully",
                self.output_stats_file,
            )

    def _build_stats_data(self, stats: AsyncProcessingStats) -> dict[str, Any]:
        """Build statistics data dictionary from AsyncProcessingStats.

        Args:
            stats: Processing statistics.

        Returns:
            Dictionary representation of processing statistics.
        """
        return {
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
            "processing_mode": "async" if self.args.async_mode else "sync",
        }

    def run(self) -> Literal[0, 1]:
        """Run the complete processing pipeline synchronously.

        Returns:
            Exit code: 0 for success, 1 for failure.

        Raises:
            ApplicationError: If any step of the pipeline fails.
        """
        try:
            logging.info("Starting medieval text processing...")
            # Clean up existing output files first
            self._cleanup_output_files()

            # Process all records using synchronous processor
            # import here to avoid circular imports
            # TODO: refactor to avoid circular import if possible
            from .sync_processor import (  # noqa: PLC0415 — avoid circular import
                SyncProcessor,
            )

            sync_processor = SyncProcessor(self)

            annotations, metadata = sync_processor.process_all_records()

            # Write output files
            self.write_output(annotations, metadata)

        except ApplicationError:
            logging.exception("Application error occurred during sync processing")
            return 1
        except KeyboardInterrupt:
            logging.info("Processing interrupted by user.")
            return 1
        except Exception:
            logging.exception("Unexpected error")
            return 1
        else:
            logging.info("Processing completed successfully")
            print("\nOutputs written to:")
            print(f"  Annotated text: {self.output_text_file}")
            print(f"  Metadata table: {self.output_table_file}")
            return 0

    async def run_async(
        self,
        progress_callback: Callable[[BatchProgress], None] | None = None,
        *,
        timeout_seconds: float | None = None,
        max_batch_wait_time: float | None = None,
        poll_interval: float | None = None,
    ) -> Literal[0, 1]:
        """Run the medieval text processor asynchronously.

        This method provides the async entry point for the application,
        with comprehensive error handling and progress monitoring.

        Args:
            progress_callback: Optional callback for batch progress updates.
                Called with BatchProgress objects during processing.
            timeout_seconds: Optional overall timeout in seconds for the entire processing.
                Defaults to DEFAULT_ASYNC_TIMEOUT (24 hours). Set to None for no timeout.
            max_batch_wait_time: Maximum time in seconds to wait for individual batch completion.
                Passed to AsyncProcessor. Uses AsyncProcessor defaults if not specified.
            poll_interval: Time in seconds between batch progress checks.
                Passed to AsyncProcessor. Uses AsyncProcessor defaults if not specified.

        Returns:
            Exit code: 0 for success, 1 for failure.
        """
        logging.info("Starting async medieval text processing...")
        # Use timeout if specified (default to 24 hours)
        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.DEFAULT_ASYNC_TIMEOUT
        )

        try:
            # Clean up existing output files first
            self._cleanup_output_files()
            # Use async context manager for better resource cleanup with timeout
            async with asyncio.timeout(timeout):
                # Process all records asynchronously using async processor
                # import here to avoid circular imports
                # TODO: refactor to avoid circular import if possible
                from .async_processor import (  # noqa: PLC0415 — avoid circular import
                    AsyncProcessor,
                )

                async_processor = AsyncProcessor(self)
                stats = await async_processor.process_all_records_async(
                    progress_callback,
                    max_batch_wait_time=max_batch_wait_time,
                    poll_interval=poll_interval,
                )
                # Write output files asynchronously
                await self.write_output_async(stats)
        except TimeoutError:  # the policy (timeout → cancellation propagates)
            logging.exception("Processing timed out after %.0f seconds", timeout)
            return 1
        except ApplicationError:
            logging.exception("Application error occurred during async processing")
            return 1
        except asyncio.CancelledError:
            logging.info("Async processing cancelled")
            raise
        except KeyboardInterrupt:
            logging.info("Processing interrupted by user.")
            return 1
        except Exception:
            logging.exception("Unexpected error during async processing")
            return 1
        else:
            logging.info(
                "Async processing completed successfully: %d/%d records (%.1f%% success) in %.2fs",
                stats.processed_records,
                stats.total_records,
                stats.success_rate,
                stats.processing_time,
            )
            return 0
