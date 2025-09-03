"""Synchronous processing module for medieval text processing pipeline.

This module handles all synchronous processing workflows, including individual
record processing, batch processing, streaming modes, and comprehensive
error handling with progress monitoring for sync operations.
"""

import logging
import time
from typing import Callable, Dict, List, Tuple, TYPE_CHECKING

from tqdm import tqdm

from ai_ner_system.processing import ValidationError, ProcessingError, LLMResponseError

from .stats import ApplicationError

if TYPE_CHECKING:
    from .main_processor import MedievalTextProcessor


class SyncProcessor:
    """Handles synchronous processing workflows for medieval text processing.

    This class is responsible for executing sync processing pipelines,
    including batch processing with fallback, streaming modes, and
    comprehensive error handling with progress monitoring.
    """

    def __init__(self, main_processor: 'MedievalTextProcessor') -> None:
        """Initialize sync processor with reference to main processor.

        Args:
            main_processor: Main MedievalTextProcessor instance.
        """
        self.main_processor = main_processor

    @property
    def args(self):
        """Access to command line arguments via main processor."""
        return self.main_processor.args

    @property
    def reader(self):
        """Access to CSV reader via main processor."""
        return self.main_processor.reader

    @property
    def writer(self):
        """Access to output writer via main processor."""
        return self.main_processor.writer

    @property
    def processor(self):
        """Access to the core processing logic via main processor."""
        return self.main_processor.processor


    def process_all_records(self) -> Tuple[List[str], List[str]]:
        """Process all records from the input CSV file using streaming approach.

        Returns:
            Tuple of (all_annotations, all_metadata). Raises:

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