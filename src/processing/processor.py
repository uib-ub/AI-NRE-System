"""Main processor for medieval text annotation with LLM services.

This module provides the core RecordProcessor class that orchestrates
the processing of medieval text records using LLM services.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import time
from collections.abc import Callable
from typing import ClassVar

from ..llm.base_client import Client
from ..llm.batch_models import BatchProgress, BatchRequest, BatchResponse
from ..prompt.builder import PromptBuilder

from .entities import EntityRecord, ProcessingResult, BatchProcessingResult
from .validator import RecordValidator
from .parser import ResponseParser
from .exceptions import ProcessingError, BatchProcessingError


class RecordProcessor:
    """Main processor for handling medieval text records through LLM services.

    This processor provides both synchronous and asynchronous methods for
    processing individual records and batches through LLM services.
    """

    # Class constants
    DEFAULT_MAX_WAIT_TIME: ClassVar[float] = 86400.0  # 24 hours
    DEFAULT_POLL_INTERVAL: ClassVar[float] = 30.0  # 30 seconds
    DEFAULT_LOG_INTERVAL: ClassVar[float] = 60.0  # 60 seconds
    DEFAULT_MAX_TOKENS: ClassVar[int] = 20000
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.0

    def __init__(self, llm_client: Client, prompt_builder: PromptBuilder) -> None:
        """Initialize the processor with LLM client and prompt builder.

        Args:
            llm_client: Instance of LLM client (ClaudeClient or OllamaClient)
            prompt_builder: Instance of PromptBuilder
        """
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder

    # ---------------------------------------------------------------------
    # Synchronous single-record processing
    # ---------------------------------------------------------------------
    def process_record(
        self,
        record: dict[str, str]
    ) -> tuple[list[str], list[str]]:
        """Process a single record through the LLM synchronously.

        Args:
            record: Dictionary with keys "Bindnr", "Brevid", and "Tekst"

        Return:
            Tuple (annotated_record, metadata_record)

        Raises:
            ProcessingError: If record processing fails.
        """
        # Extract required fields from the record
        bindnr = record.get('Bindnr', 'unknown')
        brevid = record.get('Brevid', 'unknown')

        try:
            # Validate required fields
            RecordValidator.validate_record(record)

            # Build prompt using the prompt builder (single record)
            prompt = self.prompt_builder.build(record)
            logging.debug('--- Prompt for Brevid %s ---\n%s', brevid, prompt)

            # Call LLM
            raw_response = self._call_llm(brevid, prompt)
            logging.debug('--- RAW RESPONSE for Brevid %s ---\n%s',
                          brevid, raw_response)

            # Parse response
            annotated_text, entities = ResponseParser.parse_llm_response(
                brevid, raw_response)

            # DEBUG: annotated text and entities
            logging.debug(
                '--- Annotated text for Brevid %s ---\n%s',
                brevid, annotated_text
            )
            logging.debug(
                '--- Entities for Brevid %s ---\n%s',
                brevid, entities
            )

            # Build output records
            annotated_record = self._build_annotated_record(
                bindnr, brevid, annotated_text
            )
            metadata_record = self._build_metadata_record(entities, brevid)

            logging.info(
                '--- Annotated record for Brevid %s ---\n%s',
                brevid, annotated_record
            )
            logging.info(
                '--- Metadata for Brevid %s ---\n%s',
                brevid, metadata_record
            )

            return [annotated_record], metadata_record

        except Exception as e:
            logging.error(
                'Error during LLM call for Brevid %s: %s',
                brevid, e, exc_info=True
            )
            raise ProcessingError(
                f'Failed to process record with Brevid {brevid}: {e}',
                brevid=brevid,
                operation='process_record',
            ) from e

    # ---------------------------------------------------------------------
    # Synchronous batch processing (single LLM call)
    # ---------------------------------------------------------------------
    def process_batch(
        self,
        records: list[dict[str, str]]
    ) -> tuple[list[str], list[str]]:
        """Process multiple records in a single LLM call synchronously.

        Args:
            records: List of record dictionaries to process.

        Returns:
            Tuple of (all_annotated_records, all_metadata_records).

        Raises:
            ProcessingError: If batch processing fails.
        """
        if not records:
            logging.warning('Empty records list provided to process_batch')
            return [], []

        try:
            # Use original sync batch processing logic
            # Validate all records
            RecordValidator.validate_records(records)

            # Build batch prompt using the prompt builder (list of records)
            batch_prompt = self.prompt_builder.build(records)
            logging.debug('--- Batch Prompt ---\n%s', batch_prompt)

            # Create batch identifier
            brevids = [record.get('Brevid', 'unknown') for record in records]
            batch_id = self._create_batch_id(brevids)

            # Call LLM with batch prompt
            raw_response = self._call_llm(batch_id, batch_prompt)
            logging.debug(
                'Received batch response (length: %d)',
                len(raw_response)
            )
            logging.debug(
                '--- RAW RESPONSE for batch %s ---\n%s',
                batch_id, raw_response
            )

            # Parse batch response
            annotated_records, metadata_records = ResponseParser.parse_batch_response(
                records, raw_response
            )

            logging.info(
                'Successfully processed batch of %d records: %d annotations, %d metadata',
                len(records),
                len(annotated_records),
                len(metadata_records),
            )
            return annotated_records, metadata_records

        except Exception as e:
            logging.error(
                'Error during batch processing: %s',
                e, exc_info=True
            )
            raise ProcessingError(
                f'Failed to process batch: {e}',
                operation='process_batch',
            ) from e

    # ---------------------------------------------------------------------
    # Asynchronous single-record processing
    # ---------------------------------------------------------------------
    async def process_record_async(
        self,
        record: dict[str, str]
    ) -> ProcessingResult:
        """Process a single record asynchronously

        Args:
            record: Dictionary containing record data with 'Brevid' and 'Tekst' keys.

        Returns:
            ProcessingResult containing the processed data.
        """
        start_time = time.monotonic()
        brevid = record.get('Brevid', 'unknown')
        bindnr = record.get('Bindnr', 'unknown')
        record_id = self._create_record_id(bindnr, brevid)

        try:
            # Validate record
            RecordValidator.validate_record(record)

            # Build prompt
            prompt = self.prompt_builder.build(record)

            # Call LLM asynchronously
            response = await self.llm_client.call_async(prompt)

            # Parse response
            annotated_text, entities = ResponseParser.parse_llm_response(
                brevid, response
            )

            # Build annotated text for result
            formatted_text = self._build_annotated_record(
                bindnr, brevid, annotated_text
            )

            processing_time = time.monotonic() - start_time

            return ProcessingResult(
                record_id=record_id,
                brevid=brevid,
                annotated_text=formatted_text,
                entities=entities,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            error_msg = f'Failed to process record: {record_id}: {e}'
            logging.error(error_msg, exc_info=True)
            processing_time = time.monotonic() - start_time
            return ProcessingResult(
                record_id=record_id,
                brevid=brevid,
                processing_time=processing_time,
                success=False,
                error_message=error_msg
            )

    # ---------------------------------------------------------------------
    # Asynchronous batch processing (via client.batch APIs when available)
    # ---------------------------------------------------------------------
    async def process_batch_async(
        self,
        records: list[dict[str, str]],
        batch_num: int,
        progress_callback: Callable[[BatchProgress], None] | None = None,
        max_wait_time: float = DEFAULT_MAX_WAIT_TIME,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> BatchProcessingResult:
        """Process multiple records as a batch asynchronously

        Args:
            records: List of record dictionaries to process.
            batch_num: A sequential batch number for display/IDs.
            progress_callback: Optional callback function to update progress.
            max_wait_time: Maximum time to wait for the batch to complete.
            poll_interval: Time between progress checks

        Return:
            BatchProcessingResult containing all processed records.

        Raises:
            ValueError: If records list is empty.
        """
        if not records:
            raise ValueError('Records list cannot be empty')

        if not self.llm_client.supports_async_batch():
            # Fallback to individual async processing
            logging.info(
                'LLM client does not support batch async, falling back to individual processing'
            )
            return await self._process_individual_async(records, batch_num, progress_callback)

        start_time = time.monotonic()

        try:
            # Prepare batch requests
            batch_requests: list[BatchRequest] = []

            for i, record in enumerate(records):
                try:
                    RecordValidator.validate_record(record)
                    prompt = self.prompt_builder.build(record)  # one record prompt

                    bindnr = record.get('Bindnr', 'unknown')
                    brevid = record.get('Brevid', 'unknown')
                    custom_id = self._create_custom_id(i, bindnr, brevid)

                    # Create a batch request
                    batch_request = BatchRequest(
                        custom_id=custom_id,
                        prompt=prompt,
                        max_tokens=self.DEFAULT_MAX_TOKENS,
                        temperature=self.DEFAULT_TEMPERATURE
                    )
                    # Append records into list
                    batch_requests.append(batch_request)

                except Exception as e:
                    logging.error(
                        'Failed to prepare batch request for record index %d, Bindnr %s, Brevid %s, %s',
                        i,
                        record.get('Bindnr', 'unknown'),
                        record.get('Brevid', 'unknown'),
                        e,
                    )
                    continue

            if not batch_requests:
                raise BatchProcessingError(
                    'No valid requests to process',
                    operation='prepare_batch'
                )

            logging.info(
                'Starting async batch processing of %d records', 
                len(batch_requests)
            )

            # Processing batch using LLM client
            batch_responses = await self.llm_client.process_batch_requests_async(
                batch_requests,
                batch_num,
                max_wait_time=max_wait_time,
                poll_interval=poll_interval,
                progress_callback=progress_callback
            )

            # Parse batch responses and build results
            results = self._build_batch_results(records, batch_responses)

            total_processing_time = time.monotonic() - start_time
            successful_count = sum(1 for r in results if r.success)
            failed_count = len(results) - successful_count

            logging.info(
                'Batch processing completed: %d successful, %d failed, %.2f seconds total',
                successful_count,
                failed_count,
                total_processing_time,
            )

            return BatchProcessingResult(
                batch_id=f'batch_{batch_num}',
                results=results,  # Results in original order
                total_processing_time=total_processing_time,
                successful_count=successful_count,
                failed_count=failed_count
            )

        except Exception as e:
            total_processing_time = time.monotonic() - start_time
            error_msg = f'Batch processing failed: {e}'
            logging.error(error_msg, exc_info=True)

            return BatchProcessingResult(
                batch_id=f'batch_{batch_num}_failed',
                results=[],
                total_processing_time=total_processing_time,
                successful_count=0,
                failed_count=len(records)
            )

    # ---------------------------------------------------------------------
    # Fallback async
    # ---------------------------------------------------------------------
    async def _process_individual_async(
            self,
            records: list[dict[str, str]],
            batch_num: int,
            progress_callback: Callable[[BatchProgress], None] | None = None
    ) -> BatchProcessingResult:
        """Fallback to individual async processing when batch is not supported.

        Args:
            records: List of record dictionaries to process.
            batch_num: A sequential batch number for display/IDs.
            progress_callback: Optional callback (currently unused in fallback).

        Returns:
            A BatchProcessingResult containing all processed records.
        """
        start_time = time.monotonic()
        batch_id = f'batch_{batch_num}_individual'

        # Process records concurrently using individual async calls
        tasks = [self.process_record_async(record) for record in records]

        # Execute all tasks concurrently
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results: list[ProcessingResult] = []
        successful_count = 0
        failed_count = 0

        for i, result in enumerate(results_raw):
            if isinstance(result, Exception):
                brevid = records[i].get('Brevid', 'unknown')
                bindnr = records[i].get('Bindnr', 'unknown')
                record_id = self._create_record_id(bindnr, brevid)

                processed_results.append(
                    ProcessingResult(
                        record_id=record_id,
                        brevid=brevid,
                        success=False,
                        error_message=str(result)
                    )
                )
                failed_count += 1
            elif isinstance(result, ProcessingResult):
                processed_results.append(result)
                if result.success:
                    successful_count += 1
                else:
                    failed_count += 1

        total_processing_time = time.monotonic() - start_time

        logging.info(
            'Individual async processing completed: %d successful, %d failed, %.2f seconds total',
            successful_count,
            failed_count,
            total_processing_time
        )

        return BatchProcessingResult(
            batch_id=batch_id,
            results=processed_results,
            total_processing_time=total_processing_time,
            successful_count=successful_count,
            failed_count=failed_count
        )

    def _build_batch_results(
        self,
        records: list[dict[str, str]],
        batch_responses: list[BatchResponse]
    ) -> list[ProcessingResult]:
        """Build processing results from batch responses.

        Args:
            records: Original list of records.
            batch_responses: List of BatchResponse objects.

        Returns:
            List of ProcessingResult objects.
        """
        results: list[ProcessingResult] = []

        # Create mapping from custom_id to (index, record) for order preservation
        response_map: dict[int, BatchResponse] = {}

        for response in batch_responses:
            try:
                # Extract index i from custom_id: "record_{i}_{Bindnr}_{Brevid}"
                index = self._extract_index_from_custom_id(response.custom_id)
                response_map[index] = response
            except (ValueError, IndexError):
                logging.warning(
                    'Could not parse index from custom_id: %s', 
                    response.custom_id
                )

        # Process responses in original order
        for i, record in enumerate(records):
            brevid = record.get('Brevid', 'unknown')
            bindnr = record.get('Bindnr', 'unknown')
            record_id = self._create_custom_id(i, bindnr, brevid)

            response = response_map.get(i)
            if not response:
                # No response found for this record
                logging.warning(
                    'No response found for record index %d, Bindnr %s Brevid %s', i, bindnr, brevid
                )
                results.append(
                    ProcessingResult(
                        record_id=record_id,
                        brevid=brevid,
                        success=False,
                        error_message=f'No response received for record index {i} with Bindnr {bindnr} Brevid {brevid}'
                    )
                )
                continue

            if not response.success:
                results.append(
                    ProcessingResult(
                        record_id=response.custom_id,
                        brevid=brevid,
                        success=False,
                        error_message=response.error_message
                    )
                )
                continue

            # Success path:
            try:
                annotated_text, entities = ResponseParser.parse_llm_response(
                    brevid,
                    response.response_text,
                )

                # formatted_text = f'{bindnr};{brevid};{annotated_text}'
                formatted_text = self._build_annotated_record(
                    bindnr, brevid, annotated_text
                )

                results.append(
                    ProcessingResult(
                        record_id=response.custom_id,
                        brevid=brevid,
                        annotated_text=formatted_text,
                        entities=entities,
                        success=True
                    )
                )
            except Exception as e:
                logging.error(
                    'Failed to parse LLM response for custom id %s with Brevid %s: %s',
                    response.custom_id, brevid, e
                )
                results.append(
                    ProcessingResult(
                        record_id=response.custom_id,
                        brevid=brevid,
                        success=False,
                        error_message=(
                            f'Failed to parse LLM response for custom id {response.custom_id} '
                            f'with Brevid {brevid}: {e}'
                        ),
                    )
                )
        return results

    def _call_llm(self, identifier: str, prompt: str) -> str:
        """Call the LLM service with the prompt.

        Args:
            identifier: Identifier for logging (brevid or batch_id).
            prompt: The prompt to send.

        Returns:
            Raw response from the LLM.

        Raises:
            ProcessingError: If LLM call fails.
        """
        try:
            logging.debug('Calling LLM for %s', identifier)
            raw_response = self.llm_client.call(prompt)

            if not raw_response or raw_response.strip() in [
                'Claude API call failed',
                'Ollama API call failed',
            ]:
                raise ProcessingError(
                    f'LLM returned error response for {identifier}',
                    operation='call_llm'
                )

            logging.debug(
                'Received LLM response for %s (length: %d)',
                identifier, len(raw_response)
            )
            return raw_response

        except Exception as e:
            raise ProcessingError(
                f'Error during LLM call for {identifier}: {e}',
                operation='call_llm',
            ) from e

    @staticmethod
    def _build_annotated_record(bindnr: str, brevid: str, annotated_text: str) -> str:
        """Build annotated text records for output.

        Args:
            bindnr: The Bindnr identifier.
            brevid: The Brevid identifier.
            annotated_text: The annotated text.

        Returns:
            Formatted annotated record.
        """
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=';', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([bindnr, brevid, annotated_text])
        return buf.getvalue().rstrip('\r\n')

    @staticmethod
    def _build_metadata_record(entities: list[EntityRecord], brevid: str) -> list[str]:
        """Build metadata records from entities.

        Args:
            entities: List of EntityRecord objects.
            brevid: The Brevid identifier for logging.

        Returns:
            List of metadata record strings.
        """
        metadata_record = [entity.to_csv_row() for entity in entities]
        logging.debug(
            'Built %d metadata records for Brevid %s',
            len(metadata_record), brevid
        )
        return metadata_record

    @staticmethod
    def _create_batch_id(brevids: list[str], max_display: int = 3) -> str:
        """Create a batch identifier from a list of brevids.

        Args:
            brevids: List of Brevid strings.
            max_display: Maximum number of Brevids to display in the ID.

        Returns:
            Batch identifier string.
        """
        if len(brevids) > max_display:
            displayed = '-'.join(brevids[:max_display])
            return f'BATCH-{displayed}...'
        return f'BATCH-{"-".join(brevids)}'

    @staticmethod
    def _create_record_id(bindnr: str, brevid: str) -> str:
        """Create a unique record identifier from Bindnr and Brevid.

        Args:
            bindnr: The Bindnr string.
            brevid: The Brevid string.

        Returns:
            Unique record identifier string.
        """
        return f'{bindnr}_{brevid}'

    @staticmethod
    def _create_custom_id(index: int, bindnr: str, brevid: str) -> str:
        """Create a custom ID for batch requests.

        Args:
            index: Index of the record in the batch.
            bindnr: The Bindnr string.
            brevid: The Brevid string.

        Returns:
            Custom ID string.
        """
        return f'record_{index}_{bindnr}_{brevid}'

    @staticmethod
    def _extract_index_from_custom_id(custom_id: str) -> int:
        """Extract the index from a custom ID.

        Args:
            custom_id: Custom ID string in format "record_{index}_{bindnr}_{brevid}".

        Returns:
            Extracted index as integer.

        Raise:
            ValueError: If custom_id format is invalid.
        """
        if not custom_id.startswith('record_'):
            raise ValueError(f'Invalid custom_id format: {custom_id}')

        parts = custom_id.split('_')
        if len(parts) < 2:
            raise ValueError(f'Invalid custom_id format: {custom_id}')
        try:
            return int(parts[1])
        except (ValueError) as e:
            raise ValueError(
                f'Could not extract index from custom_id: {custom_id}'
            ) from e


# -------------------------------------------------------------------------
# Utility functions for batch processing monitoring
# -------------------------------------------------------------------------
def create_progress_logger(
    log_interval: float = RecordProcessor.DEFAULT_LOG_INTERVAL
) -> Callable[[BatchProgress], None]:
    """Create a progress callback that logs batch status.
    Args:
        log_interval: Minimum seconds between log messages.

    Returns:
        Callback function that logs batch progress.
    """
    last_log_time: float = 0.0

    def log_progress(progress: BatchProgress) -> None:
        nonlocal last_log_time
        current_time = time.monotonic()

        if current_time - last_log_time > log_interval:
            counts = progress.request_counts
            logging.info(
                'Batch %d (ID: %s) progress updates every %.1fs: %s - Processing: %d, Succeeded: %d, Errored: %d, Elapsed: %.1fs',
                progress.batch_num,
                progress.batch_id,
                log_interval,
                progress.status.value,
                counts.get('processing', 0),
                counts.get('succeeded', 0),
                counts.get('errored', 0),
                progress.elapsed_time,
            )
            last_log_time = current_time

    return log_progress
