"""Synchronous processing module for medieval text processing pipeline.

This module handles all synchronous processing workflows, including individual
record processing, batch processing, streaming modes, and comprehensive
error handling with progress monitoring for sync operations.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ai_ner_system.processing import (
    BatchProcessingError,
    LLMResponseError,
    ParseError,
    ProcessingError,
    ValidationError,
)

from .stats import ApplicationError

if TYPE_CHECKING:
    from argparse import Namespace

    from ai_ner_system.file_io import CSVReader, OutputWriter
    from ai_ner_system.processing import RecordProcessor

    from .main_processor import MedievalTextProcessor


class SyncProcessor:
    """Handles synchronous processing workflows for medieval text processing.

    This class is responsible for executing sync processing pipelines,
    including batch processing with fallback, streaming modes, and
    comprehensive error handling with progress monitoring.

    Class Attributes:
        BATCH_PROCESSING_DELAY: Delay in seconds between batch processing
            to avoid overwhelming external APIs with rapid requests.
    """

    # Class constant for the delay between batches
    BATCH_PROCESSING_DELAY: ClassVar[float] = 0.2

    def __init__(self, main_processor: MedievalTextProcessor) -> None:
        """Initialize sync processor with reference to main processor.

        Args:
            main_processor: Main MedievalTextProcessor instance.
        """
        self.main_processor = main_processor

    @property
    def args(self) -> Namespace:
        """Access to command line arguments via main processor.

        Returns:
            Command line arguments namespace.
        """
        return self.main_processor.args

    @property
    def reader(self) -> CSVReader:
        """Access to CSV reader via main processor.

        Returns:
            CSV reader instance for input file operations.
        """
        return self.main_processor.reader

    @property
    def writer(self) -> OutputWriter:
        """Access to output writer via main processor.

        Returns:
            Output writer instance for file writing operations.
        """
        return self.main_processor.writer

    @property
    def processor(self) -> RecordProcessor:
        """Access to the core processing logic via main processor.

        Returns:
            Record processor instance for text processing.
        """
        return self.main_processor.processor

    @staticmethod
    def _tqdm_kwargs(desc: str) -> dict[str, Any]:
        """Common tqdm kwargs with TTY-aware defaults.

        Args:
            desc: Description for the progress bar.
        """
        return {
            "desc": desc,
            # if don't want to leave finished bars in logs, set to False
            "leave": True,
            "disable": not sys.stderr.isatty(),  # auto-disable if not a TTY
            "dynamic_ncols": True,  # fit nicely to terminal width
            "mininterval": 0.5,  # reduce redraw chattiness
        }

    def process_all_records(self) -> tuple[list[str], list[str]]:
        """Process all records from the input CSV file using streaming approach.

        Returns:
            Tuple of (all_annotations, all_metadata) containing processed results.

        Raises:
            ApplicationError: If critical processing error occurs.
        """
        logging.info("Starting to process records from: %s", self.reader.file_path)

        batch_size = self.args.batch_size if self.args.use_batch else 1
        processing_mode: Literal["batch", "individual"] = (
            "batch" if batch_size > 1 else "individual"
        )
        logging.info("Using %s processing (batch_size=%d)", processing_mode, batch_size)

        try:
            # Process records with unified streaming approach
            all_annotations, all_metadata = self._process_records_streaming(
                batch_size,
                processing_mode,
            )
        except Exception as e:
            raise ApplicationError("Critical error during file processing") from e

        logging.info(
            "Completed processing all records: %d annotations, %d metadata entries",
            len(all_annotations),
            len(all_metadata),
        )
        return all_annotations, all_metadata

    def _process_records_streaming(
        self,
        batch_size: int,
        processing_mode: Literal["batch", "individual"],
    ) -> tuple[list[str], list[str]]:
        """Process records using streaming approach with configurable batch size.

        Args:
            batch_size: Number of records to process together (1 = individual processing).
            processing_mode: Processing mode ("batch" or "individual").

        Returns:
            Tuple of (all_annotations, all_metadata).

        Raises:
            ApplicationError: If streaming processing fails.
        """
        all_annotations: list[str] = []
        all_metadata: list[str] = []

        # a batch of records to process together
        batch_records: list[dict[str, str]] = []
        batch_count = 0  # counter for the number of batches processed

        # Redirect logging through tqdm to keep output tidy
        with logging_redirect_tqdm():
            iterable = self.reader.stream_records()
            desc = f"Processing Records ({processing_mode} mode)"
            try:
                for record in tqdm(iterable, **self._tqdm_kwargs(desc)):
                    batch_records.append(record)

                    # Process when batch is full or for individual processing (batch_size=1)
                    if len(batch_records) >= batch_size:
                        batch_count += 1

                        annotated_records, metadata_records = self._process_batch(
                            batch_records,
                            batch_count,
                            batch_size,
                        )

                        # Collect results
                        all_annotations.extend(annotated_records)
                        all_metadata.extend(metadata_records)

                        # Log progress
                        if batch_size == 1:
                            brevid = batch_records[0].get("Brevid", "unknown")
                            logging.debug(
                                "Successfully processed Brevid %s: %d annotations, %d metadata",
                                brevid,
                                len(annotated_records),
                                len(metadata_records),
                            )
                        else:
                            logging.debug(
                                "Successfully processed batch %d: %d annotations, %d metadata",
                                batch_count,
                                len(annotated_records),
                                len(metadata_records),
                            )

                        # Clear batch records after processing
                        batch_records.clear()

                        # Small delay to avoid overwhelming the API (rate limiting)
                        if self.BATCH_PROCESSING_DELAY > 0:
                            time.sleep(self.BATCH_PROCESSING_DELAY)

                # Process any remaining records in the final partial batch
                if batch_records:
                    batch_count += 1
                    annotations, metadata = self._process_final_batch(
                        batch_records,
                        batch_count,
                        batch_size,
                        processing_mode,
                    )
                    all_annotations.extend(annotations)
                    all_metadata.extend(metadata)

            except Exception as e:
                logging.exception("Streaming processing failed")
                # Avoid embedding `e` in the message; preserve via chaining
                raise ApplicationError(
                    f"Streaming processing failed after {batch_count} batches",
                ) from e

        return all_annotations, all_metadata

    def _process_batch(
        self,
        batch_records: list[dict[str, str]],
        batch_count: int,
        batch_size: int,
    ) -> tuple[list[str], list[str]]:
        """Process a batch of records, handling both individual and batch modes.

        Args:
            batch_records: The list of records to process.
            batch_count: The current batch number.
            batch_size: The size of the batch.

        Returns:
            A tuple containing a list of annotation strings and a list of metadata strings.

        Note:
            Any exception raised by the underlying processor will be handled
            by _handle_batch_exception, which implements fallback logic.
        """
        try:
            # Process the batch/record
            if batch_size == 1:
                # Individual processing
                individual_record = batch_records[0]
                brevid = individual_record.get("Brevid", "unknown")
                logging.info("Processing Record (Brevid: %s)", brevid)
                logging.debug("Individual record data: %s", individual_record)
                return self.processor.process_record(individual_record)

            # Batch processing
            logging.info(
                "Processing batch %d with %d records",
                batch_count,
                len(batch_records),
            )
            return self.processor.process_batch(batch_records)

        except ProcessingError as e:
            return self._handle_batch_exception(
                batch_records,
                batch_count,
                batch_size,
                e,
            )

    def _handle_batch_exception(
        self,
        batch_records: list[dict[str, str]],
        batch_count: int,
        batch_size: int,
        error: ProcessingError,
    ) -> tuple[list[str], list[str]]:
        """Handle exceptions during batch processing, with fallback to individual processing if needed.

        For batch mode: falls back to individual processing of each record.
        For individual mode: logs and returns empty outputs.

        Args:
            batch_records: The list of records in the batch.
            batch_count: The current batch number.
            batch_size: The size of the batch.
            error: The exception that was raised.

        Returns:
            A tuple containing a list of annotation strings and a list of metadata strings.
            Returns empty lists if all processing fails.
        """
        if batch_size == 1:
            # Individual processing error - no fallback available
            brevid = batch_records[0].get("Brevid", "unknown")
            logging.error("Error processing Brevid %s: %s", brevid, error)
            self._handle_individual_error(batch_records[0], error)
            return [], []

        # Batch processing error - fallback to individual processing
        logging.error("Error processing batch %d: %s", batch_count, error)
        logging.info(
            "Falling back to individual processing for batch %d (%d records)",
            batch_count,
            len(batch_records),
        )

        annotations: list[str] = []
        metadata: list[str] = []

        with logging_redirect_tqdm():
            # Process each record in the failed batch individually
            for record in tqdm(
                batch_records,
                **self._tqdm_kwargs(
                    desc=f"Fallback Batch {batch_count} Individual Processing",
                ),
            ):
                annotated_record, metadata_record = (
                    self._fallback_to_individual_processing(
                        record,
                    )
                )
                annotations.extend(annotated_record)
                metadata.extend(metadata_record)

        logging.info(
            "Fallback processing completed for batch %d: %d annotations, %d metadata",
            batch_count,
            len(annotations),
            len(metadata),
        )
        return annotations, metadata

    def _process_final_batch(
        self,
        batch_records: list[dict[str, str]],
        batch_count: int,
        batch_size: int,
        processing_mode: Literal["batch", "individual"],
    ) -> tuple[list[str], list[str]]:
        """Process the final (possibly partial) batch of records.

        Args:
            batch_records: The list of records in the final batch.
            batch_count: The current batch number.
            batch_size: The size of the batch.
            processing_mode: The processing mode ("batch" or "individual").

        Returns:
            A tuple containing a list of annotation strings and a list of metadata strings.
        """
        annotations: list[str] = []
        metadata: list[str] = []

        with (
            logging_redirect_tqdm(),
            tqdm(
                total=len(batch_records),
                **self._tqdm_kwargs(f"Final {processing_mode} batch"),
            ) as final_pbar,
        ):
            if batch_size == 1:
                # Individual processing for remaining records
                for record in batch_records:
                    brevid = record.get("Brevid", "unknown")
                    try:
                        annotated_records, metadata_records = (
                            self.processor.process_record(record)
                        )
                    except ProcessingError:
                        logging.exception("Final record %s failed", brevid)
                        annotated_records, metadata_records = (
                            self._fallback_to_individual_processing(
                                record,
                            )
                        )
                    annotations.extend(annotated_records)
                    metadata.extend(metadata_records)
                    final_pbar.set_description(f"Final record: {brevid}")
                    final_pbar.update(1)
            else:
                # Process the final batch
                logging.info(
                    "Processing final batch %d with %d records",
                    batch_count,
                    len(batch_records),
                )
                try:
                    annotated_records, metadata_records = self.processor.process_batch(
                        batch_records,
                    )
                except ProcessingError:
                    # Fallback to individual processing for remaining records
                    logging.exception(
                        "Final batch %d failed; falling back per record",
                        batch_count,
                    )
                    final_pbar.set_description("Final batch fallback")
                    for record in batch_records:
                        annotated_records, metadata_records = (
                            self._fallback_to_individual_processing(
                                record,
                            )
                        )
                        annotations.extend(annotated_records)
                        metadata.extend(metadata_records)
                        final_pbar.update(1)
                else:
                    annotations.extend(annotated_records)
                    metadata.extend(metadata_records)
                    final_pbar.set_description(
                        f"Final batch ({len(batch_records)} records)",
                    )
                    final_pbar.update(len(batch_records))
                    logging.debug(
                        "Successfully processed final batch: %d annotations, %d metadata",
                        len(annotated_records),
                        len(metadata_records),
                    )

        return annotations, metadata

    def _fallback_to_individual_processing(
        self,
        record: dict[str, str],
    ) -> tuple[list[str], list[str]]:
        """Fallback to individual processing when batch processing fails.

        Args:
            record: Record that failed to process in batch processing.

        Returns:
            Tuple of (annotated_record, metadata_record) for the individual record.
            Returns empty lists if processing fails.
        """
        brevid = record.get("Brevid", "unknown")
        logging.debug("Attempting individual processing for Brevid %s", brevid)

        try:
            return self.processor.process_record(record)
        except ProcessingError as e:
            logging.exception("Error in fallback processing for Brevid %s", brevid)
            self._handle_individual_error(record, e)
            return [], []

    @staticmethod
    def _handle_individual_error(
        record: dict[str, str],
        exception: ProcessingError,
    ) -> None:
        """Handle errors that occur during individual record processing.

        Args:
            record: The record that failed to process.
            exception: The exception that occurred.

        Note:
            This method logs the error but does not raise exceptions,
            allowing processing to continue with other records.
        """
        brevid = record.get("Brevid", "unknown")
        bindnr = record.get("Bindnr", "unknown")

        if isinstance(exception, ValidationError):
            logging.error(
                "Validation error for Brevid %s (Bindnr: %s): %s",
                brevid,
                bindnr,
                exception,
            )
        elif isinstance(
            exception,
            LLMResponseError | ParseError | BatchProcessingError,
        ):
            logging.error(
                "LLM Processing error for Brevid %s (Bindnr: %s): %s",
                brevid,
                bindnr,
                exception,
            )
        else:
            logging.error(
                "Unexpected error processing Brevid %s (Bindnr: %s): %s",
                brevid,
                bindnr,
                exception,
                exc_info=True,
            )
