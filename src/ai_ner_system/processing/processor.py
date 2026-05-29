"""Main processor for medieval text annotation with LLM services.

This module provides the core RecordProcessor class that orchestrates
the processing of medieval text records using LLM services.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, ClassVar, NoReturn

from ai_ner_system.llm.base_client import Client
from ai_ner_system.llm.batch_models import BatchProgress, BatchRequest, BatchResponse
from ai_ner_system.llm.exceptions import LLMClientError

from .entities import BatchProcessingResult, EntityRecord, ProcessingResult
from .exceptions import (
    BatchProcessingError,
    LLMResponseError,
    ParseError,
    ProcessingError,
    ValidationError,
)
from .parser import ResponseParser
from .validator import RecordValidator

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from ai_ner_system.llm.base_client import Client
    from ai_ner_system.prompt.builder import PromptBuilder


class RecordProcessor:
    """Main processor for handling medieval text records through LLM services.

    This processor provides both synchronous and asynchronous methods for
    processing individual records and batches through LLM services.
    """

    # Class constants
    DEFAULT_MAX_WAIT_TIME: ClassVar[float] = 86400.0  # 24 hours
    DEFAULT_POLL_INTERVAL: ClassVar[float] = 30.0  # 30 seconds
    DEFAULT_MAX_TOKENS: ClassVar[int] = 20000
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.0
    MIN_CUSTOM_ID_PARTS: ClassVar[int] = 2  # Minimum parts in custom_id format

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
        record: dict[str, str],
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
        bindnr = record.get("Bindnr", "unknown")
        brevid = record.get("Brevid", "unknown")

        try:
            # Validate required fields
            RecordValidator.validate_record(record)
            # Build prompt using the prompt builder (single record)
            prompt = self.prompt_builder.build(record)
            logging.debug("--- Prompt for Brevid %s ---\n%s", brevid, prompt)
            # Call LLM
            raw_response = self._call_llm(brevid, prompt)
            logging.debug(
                "--- RAW RESPONSE for Brevid %s ---\n%s",
                brevid,
                raw_response,
            )
            # Parse response
            annotated_text, entities = ResponseParser.parse_llm_response(
                brevid,
                raw_response,
            )
            # Build output records
            annotated_record = self._build_annotated_record(
                bindnr,
                brevid,
                annotated_text,
            )
            metadata_record = self._build_metadata_record(entities, brevid)
        except ProcessingError:
            # Already our domain error; keep original context
            logging.exception("Error during processing for Brevid %s", brevid)
            raise
        except Exception as e:
            logging.exception("Error during LLM call for Brevid %s", brevid)
            raise ProcessingError(
                f"Failed to process record with Brevid {brevid}: {e}",
                brevid=brevid,
                operation="process_record",
            ) from e
        else:
            # DEBUG: annotated text and entities
            logging.debug(
                "--- Annotated text for Brevid %s ---\n%s",
                brevid,
                annotated_text,
            )
            logging.debug("--- Entities for Brevid %s ---\n%s", brevid, entities)
            logging.debug(
                "--- Annotated record for Brevid %s ---\n%s",
                brevid,
                annotated_record,
            )
            logging.debug("--- Metadata for Brevid %s ---\n%s", brevid, metadata_record)
            return [annotated_record], metadata_record

    # ---------------------------------------------------------------------
    # Synchronous batch processing (single LLM call)
    # ---------------------------------------------------------------------
    def process_batch(
        self,
        records: list[dict[str, str]],
    ) -> tuple[list[str], list[str]]:
        """Process multiple records in a single LLM call synchronously.

        This is a batch processing in the sync path: one prompt (prompt-batch.txt),
        one LLM call, one parser pass over a multi-record response.
        So when the sync pipeline says batch processing, it means 'combine several
        records into a single model request with the prompt-batch.txt prompt template.'

        Args:
            records: List of record dictionaries to process.

        Returns:
            Tuple of (all_annotated_records, all_metadata_records).

        Raises:
            ProcessingError: If batch processing fails.
        """
        if not records:
            logging.warning("Empty records list provided to process_batch")
            return [], []

        # Create batch identifier
        brevids = [record.get("Brevid", "unknown") for record in records]
        batch_id = self._create_batch_id(brevids)

        try:
            # Use original sync batch processing logic
            # Validate all records
            RecordValidator.validate_records(records)
            # Build batch prompt using the prompt builder (list of records)
            batch_prompt = self.prompt_builder.build(records)
            # Call LLM with batch prompt
            raw_response = self._call_llm(batch_id, batch_prompt)
            # Parse batch response
            annotated_records, metadata_records = ResponseParser.parse_batch_response(
                records,
                raw_response,
            )
        except ProcessingError:
            # Keep original ProcessingError intact (no double-wrapping)
            logging.exception("Error during batch processing for %s", batch_id)
            raise
        except Exception as e:
            logging.exception("Error during batch processing for %s", batch_id)
            raise ProcessingError(
                f"Failed to process batch: {e}",
                operation="process_batch",
            ) from e
        else:
            logging.debug(
                "--- Batch Prompt for batch %s ---\n%s",
                batch_id,
                batch_prompt,
            )
            logging.debug(
                "Received batch response for batch %s (length: %d)",
                batch_id,
                len(raw_response),
            )
            logging.debug(
                "--- RAW RESPONSE for batch %s ---\n%s",
                batch_id,
                raw_response,
            )
            logging.info(
                "Successfully processed batch of %d records: %d annotations, %d metadata",
                len(records),
                len(annotated_records),
                len(metadata_records),
            )
            return annotated_records, metadata_records

    # ---------------------------------------------------------------------
    # Asynchronous single-record processing
    # ---------------------------------------------------------------------
    async def process_record_async(
        self,
        record: dict[str, str],
    ) -> ProcessingResult:
        """Process a single record asynchronously.

        Args:
            record: Dictionary containing record data with 'Brevid' and 'Tekst' keys.

        Returns:
            ProcessingResult containing the processed data.
        """
        start_time = time.monotonic()
        brevid = record.get("Brevid", "unknown")
        bindnr = record.get("Bindnr", "unknown")
        record_id = self.create_record_id(bindnr, brevid)

        try:
            # Validate record
            RecordValidator.validate_record(record)
            # Build prompt
            prompt = self.prompt_builder.build(record)
            # Call LLM asynchronously
            response = await self.llm_client.call_async(prompt)
            # Parse response
            annotated_text, entities = ResponseParser.parse_llm_response(
                brevid,
                response,
            )
            # Build annotated text for result
            formatted_text = self._build_annotated_record(
                bindnr,
                brevid,
                annotated_text,
            )
        except ValidationError as e:
            return self._handle_async_record_error(
                e,
                record_id,
                brevid,
                start_time,
                "Validation failed",
            )
        except LLMClientError as e:
            return self._handle_async_record_error(
                e,
                record_id,
                brevid,
                start_time,
                "LLM client error",
            )
        except (LLMResponseError, ParseError) as e:
            return self._handle_async_record_error(
                e,
                record_id,
                brevid,
                start_time,
                "Response parse error",
            )
        else:
            # Success path
            processing_time = time.monotonic() - start_time
            return self._create_processing_result(
                record_id=record_id,
                brevid=brevid,
                success=True,
                processing_time=processing_time,
                annotated_text=formatted_text,
                entities=entities,
            )

    # ---------------------------------------------------------------------
    # Asynchronous batch processing (via client.batch APIs when available)
    # ---------------------------------------------------------------------
    async def process_batch_async(
        self,
        records: list[dict[str, str]],
        batch_num: int,
        progress_callback: Callable[[BatchProgress], None] | None = None,
        max_wait_time: float | None = None,
        poll_interval: float | None = None,
    ) -> BatchProcessingResult:
        """Process multiple records as a batch asynchronously.

        This method processes one async batch, and produces one BatchProcessingResult.

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
            raise ValueError("Records list cannot be empty")

        # Use class defaults if not provided
        if max_wait_time is None:
            max_wait_time = self.DEFAULT_MAX_WAIT_TIME
        if poll_interval is None:
            poll_interval = self.DEFAULT_POLL_INTERVAL

        # Fallback to individual async processing if batch not supported
        if not self.llm_client.supports_async_batch():
            logging.info(
                "LLM client does not support batch async, falling back to individual processing",
            )
            return await self._process_individual_async(
                records,
                batch_num,
                progress_callback,
            )

        start_time = time.monotonic()

        # Prepare batch requests
        batch_requests = self._prepare_batch_requests(records)
        if not batch_requests:
            raise BatchProcessingError(
                "No valid requests to process",
                operation="prepare_batch",
                batch_id=f"batch_{batch_num}",
            )

        logging.info(
            "Starting async batch processing of %d records",
            len(batch_requests),
        )

        # Execute batch processing
        try:
            batch_responses = await self.llm_client.process_batch_requests_async(
                batch_requests,
                batch_num,
                max_wait_time=max_wait_time,
                poll_interval=poll_interval,
                progress_callback=progress_callback,
            )
            results = self._build_batch_results(records, batch_responses)
        except asyncio.CancelledError:
            logging.info("Batch processing was cancelled")
            raise
        except Exception:
            # Re-raise so the caller's fallback path (per-record async processing)
            # in AsyncProcessor._process_batch_with_order_async can run.
            # Swallowing here would silently lose every record in the batch.
            logging.exception(
                "Batch %d processing failed; allowing caller to fall back",
                batch_num,
            )
            raise
        else:
            total_processing_time = time.monotonic() - start_time
            return self._create_batch_result(
                batch_num,
                results=results,
                total_processing_time=total_processing_time,
            )

    def _handle_async_record_error(
        self,
        error: Exception,
        record_id: str,
        brevid: str,
        start_time: float,
        error_context: str,
    ) -> ProcessingResult:
        """Handle errors during async record processing.

        Args:
            error: The exception that occurred.
            record_id: Unique identifier for the record.
            brevid: Brevid identifier.
            start_time: Processing start time.
            error_context: Context description for logging.

        Returns:
            ProcessingResult configured for failure.
        """
        logging.exception("%s for %s", error_context, record_id)
        processing_time = time.monotonic() - start_time
        return self._create_processing_result(
            record_id=record_id,
            brevid=brevid,
            success=False,
            processing_time=processing_time,
            error_msg=str(error),
        )

    def _prepare_batch_requests(
        self,
        records: list[dict[str, str]],
    ) -> list[BatchRequest]:
        """Prepare batch requests from records.

        This method converts raw input records into BatchRequest objects
        for the LLM batch API.

        Args:
            records: List of record dictionaries to process.

        Returns:
            List of BatchRequest objects.
        """
        batch_requests: list[BatchRequest] = []
        for i, record in enumerate(records):
            try:
                RecordValidator.validate_record(record)
                prompt = self.prompt_builder.build(record)

                bindnr = record.get("Bindnr", "unknown")
                brevid = record.get("Brevid", "unknown")
                # create custom_id in the format: "record_{i}_{bindnr}_{brevid}"
                custom_id = self._create_custom_id(i, bindnr, brevid)

                batch_request = BatchRequest(
                    custom_id=custom_id,
                    prompt=prompt,
                    max_tokens=self.DEFAULT_MAX_TOKENS,
                    temperature=self.DEFAULT_TEMPERATURE,
                )
                batch_requests.append(batch_request)

            except Exception:
                logging.exception(
                    "Failed to prepare batch request for record index %d, Bindnr %s, Brevid %s",
                    i,
                    record.get("Bindnr", "unknown"),
                    record.get("Brevid", "unknown"),
                )
                continue

        return batch_requests

    def _create_batch_result(
        self,
        batch_num: int,
        results: list[ProcessingResult],
        total_processing_time: float,
        *,
        failed: bool = False,
    ) -> BatchProcessingResult:
        """Create a BatchProcessingResult object.

        Args:
            batch_num: Batch number.
            results: List of ProcessingResult objects.
            total_processing_time: Total processing time.
            failed: Whether the entire batch failed.

        Returns:
            BatchProcessingResult object.
        """
        successful_count = sum(1 for r in results if r.success)
        failed_count = len(results) - successful_count

        batch_id = f"batch_{batch_num}"

        if not failed:
            logging.info(
                "Batch processing completed: %d successful, %d failed, %.2f seconds total",
                successful_count,
                failed_count,
                total_processing_time,
            )

        return BatchProcessingResult(
            batch_id=batch_id,
            results=results,
            total_processing_time=total_processing_time,
            successful_count=successful_count,
            failed_count=failed_count,
        )

    # ---------------------------------------------------------------------
    # Fallback async
    # ---------------------------------------------------------------------
    async def _process_individual_async(
        self,
        records: list[dict[str, str]],
        batch_num: int,
        _progress_callback: Callable[[BatchProgress], None] | None = None,
    ) -> BatchProcessingResult:
        """Fallback to individual async processing when batch is not supported.

        This method creates one _process_individual_async(...) coroutine per record,
        and runs them concurrently with asyncio.gather(..., return_exceptions=True).

        Args:
            records: List of record dictionaries to process.
            batch_num: A sequential batch number for display/IDs.
            _progress_callback: Optional callback (currently unused in fallback).

        Returns:
            A BatchProcessingResult (a list of ProcessingResult objects) containing
            all processed records.
        """
        start_time = time.monotonic()
        batch_id = f"batch_{batch_num}"

        # Process records concurrently using individual async calls
        tasks = [self.process_record_async(record) for record in records]

        # Execute all tasks concurrently, waiting for their completion.
        # The reason that we do not use modern asyncio.TaskGroup is that
        # 1. we need to preserve the order of results as they correspond to
        # the original records, but TaskGroup does not guarantee order.
        # 2. we want to process as many records as possible, even if
        # some fail (Fault-tolerant), but TaskGroup() cancels all tasks
        # on the first failure.
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results: list[ProcessingResult] = []
        successful_count = 0
        failed_count = 0

        for i, result in enumerate(results_raw):
            if isinstance(result, BaseException):
                brevid = records[i].get("Brevid", "unknown")
                bindnr = records[i].get("Bindnr", "unknown")
                record_id = self.create_record_id(bindnr, brevid)

                processed_results.append(
                    self._create_processing_result(
                        record_id=record_id,
                        brevid=brevid,
                        success=False,
                        error_msg=str(result),
                    ),
                )
                failed_count += 1
            else:
                processed_results.append(result)
                if result.success:
                    successful_count += 1
                else:
                    failed_count += 1

        total_processing_time = time.monotonic() - start_time

        logging.info(
            "Individual async processing completed: %d successful, %d failed, %.2f seconds total",
            successful_count,
            failed_count,
            total_processing_time,
        )

        return BatchProcessingResult(
            batch_id=batch_id,
            results=processed_results,
            total_processing_time=total_processing_time,
            successful_count=successful_count,
            failed_count=failed_count,
        )

    def _create_response_map(
        self,
        batch_responses: list[BatchResponse],
    ) -> dict[int, BatchResponse]:
        """Create mapping from index to BatchResponse for order preservation.

        Args:
            batch_responses: List of BatchResponse objects.

        Returns:
            Dictionary mapping record index to BatchResponse.
        """
        response_map: dict[int, BatchResponse] = {}
        for response in batch_responses:
            try:
                # Extract index i from custom_id: "record_{i}_{Bindnr}_{Brevid}"
                index = self._extract_index_from_custom_id(response.custom_id)
                response_map[index] = response
            except (ValueError, IndexError):
                logging.warning(
                    "Could not parse index from custom_id: %s",
                    response.custom_id,
                )
        return response_map

    def _process_single_batch_response(
        self,
        i: int,
        record: dict[str, str],
        response: BatchResponse | None,
    ) -> ProcessingResult:
        """Process a single batch API response into a ProcessingResult.

        Args:
            i: Record index, used for logging and mapping to original record.
            record: Original record dictionary.
            response: BatchResponse object or None if no response.

        Returns:
            ProcessingResult for this record.
        """
        brevid = record.get("Brevid", "unknown")
        bindnr = record.get("Bindnr", "unknown")
        record_id = self._create_custom_id(i, bindnr, brevid)

        # Handle missing response
        if not response:
            logging.warning(
                "No response found for record index %d, Bindnr %s Brevid %s",
                i,
                bindnr,
                brevid,
            )
            # No response found for that record -> failed ProcessingResult
            return self._create_processing_result(
                record_id=record_id,
                brevid=brevid,
                success=False,
                error_msg=f"No response received for record index {i} with Bindnr {bindnr} Brevid {brevid}",
            )

        # Handle failed response
        if not response.success:
            # Response exists but is marked unsuccessful -> failed ProcessingResult
            return self._create_processing_result(
                record_id=response.custom_id,
                brevid=brevid,
                success=False,
                error_msg=response.error_message,
            )

        # Response successful -> parse response text and create and return success ProcessingResult
        try:
            annotated_text, entities = ResponseParser.parse_llm_response(
                brevid,
                response.response_text,
            )
            formatted_text = self._build_annotated_record(
                bindnr,
                brevid,
                annotated_text,
            )

            return self._create_processing_result(
                record_id=response.custom_id,
                brevid=brevid,
                success=True,
                annotated_text=formatted_text,
                entities=entities,
            )
        except Exception as e:
            logging.exception(
                "Failed to parse LLM response for custom id %s with Brevid %s",
                response.custom_id,
                brevid,
            )
            return self._create_processing_result(
                record_id=response.custom_id,
                brevid=brevid,
                success=False,
                error_msg=(
                    f"Failed to parse LLM response for custom id {response.custom_id} "
                    f"with Brevid {brevid}: {e}"
                ),
            )

    def _build_batch_results(
        self,
        records: list[dict[str, str]],
        batch_responses: list[BatchResponse],
    ) -> list[ProcessingResult]:
        """Build processing results from batch responses while preserving order of original records.

        This method:

        1. builds response_map to map from record index to BatchResponse
        2. iterates over original records with their index, looks up corresponding BatchResponse by response_map.get(i)
        3. create one ProcessingResult per record, preserving the original order of records in the results list.

        This guarantees the resulting list is in original input order, even if the batch API returned responses out of order.

        Args:
            records: Original list of records.
            batch_responses: List of BatchResponse objects.

        Returns:
            List of ProcessingResult objects.
        """
        # Create mapping for order preservation
        response_map = self._create_response_map(batch_responses)

        # List comprehension processes each record in original order automatically
        return [
            self._process_single_batch_response(i, record, response_map.get(i))
            for i, record in enumerate(records)
        ]

    def _call_llm(self, identifier: str, prompt: str) -> str:
        """This method is the sync LLM-call wrapper to call the LLM service with the prompt.

        Args:
            identifier: Identifier for logging (brevid or batch_id).
            prompt: The prompt to send.

        Returns:
            Raw response from the LLM.

        Raises:
            ProcessingError: If LLM call fails.
        """

        def _fail(
            msg: str,
            *,
            operation: str,
            cause: Exception | None = None,
        ) -> NoReturn:
            if cause is None:
                raise ProcessingError(msg, operation=operation)
            raise ProcessingError(msg, operation=operation) from cause

        logging.debug("Calling LLM for %s", identifier)
        try:
            raw_response = self.llm_client.call(prompt)
        except LLMClientError as e:
            _fail(
                f"Error during LLM call for {identifier}: {e}",
                operation="call_llm",
                cause=e,
            )

        stripped = raw_response.strip() if raw_response else ""
        if not stripped or stripped in self.llm_client.ERROR_RESPONSE_SENTINELS:
            _fail(f"LLM returned error response for {identifier}", operation="call_llm")

        logging.debug(
            "Received LLM response for %s (length: %d)",
            identifier,
            len(raw_response),
        )
        return raw_response

    @staticmethod
    def _build_annotated_record(bindnr: str, brevid: str, annotated_text: str) -> str:
        """Build annotated text records for output.

        This delegates to ResponseParser.format_csv_row() to ensure consistent
        CSV formatting across all processing paths (sync, batch, async).

        Args:
            bindnr: The Bindnr identifier.
            brevid: The Brevid identifier.
            annotated_text: The annotated text.

        Returns:
            Formatted annotated record.
        """
        return ResponseParser.format_csv_row(bindnr, brevid, annotated_text)

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
            "Built %d metadata records for Brevid %s",
            len(metadata_record),
            brevid,
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
            displayed = "-".join(brevids[:max_display])
            return f"BATCH-{displayed}..."
        return f"BATCH-{'-'.join(brevids)}"

    @staticmethod
    def create_record_id(bindnr: str, brevid: str) -> str:
        """Create a unique record identifier from Bindnr and Brevid.

        Args:
            bindnr: The Bindnr string.
            brevid: The Brevid string.

        Returns:
            Unique record identifier string.
        """
        return f"{bindnr}_{brevid}"

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
        return f"record_{index}_{bindnr}_{brevid}"

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
        if not custom_id.startswith("record_"):
            raise ValueError(f"Invalid custom_id format: {custom_id}")

        parts = custom_id.split("_")
        try:
            return int(parts[1])
        except ValueError as e:
            raise ValueError(
                f"Could not extract index from custom_id: {custom_id}",
            ) from e

    @staticmethod
    def _create_processing_result(
        *,
        record_id: str,
        brevid: str,
        success: bool,
        processing_time: float = 0.0,
        annotated_text: str | None = None,
        entities: list[EntityRecord] | None = None,
        error_msg: str | None = None,
    ) -> ProcessingResult:
        """Create a ProcessingResult object.

        Args:
            record_id: Unique identifier for the record.
            brevid: Brevid identifier.
            success: Whether processing was successful.
            processing_time: Time taken to process (default: 0.0).
            annotated_text: Annotated text (required if success=True).
            entities: List of extracted entities (required if success=True).
            error_msg: Error message (required if success=False).

        Returns:
            ProcessingResult object configured for success or failure case.
        """
        if success:
            # Type assertions for success case
            if annotated_text is None:
                msg = "annotated_text is required when success=True"
                raise ValueError(msg)
            if entities is None:
                msg = "entities is required when success=True"
                raise ValueError(msg)
            return ProcessingResult(
                record_id=record_id,
                brevid=brevid,
                annotated_text=annotated_text,
                entities=entities,
                processing_time=processing_time,
                success=True,
            )

        # Failure case
        return ProcessingResult(
            record_id=record_id,
            brevid=brevid,
            processing_time=processing_time,
            success=False,
            error_message=error_msg or "Unknown error",
        )


# -------------------------------------------------------------------------
# Utility functions for batch processing monitoring
# -------------------------------------------------------------------------
def create_progress_logger(
    log_interval: float = 60.0,
) -> Callable[[BatchProgress], None]:
    """Create a progress callback that logs batch status.

    Args:
        log_interval: Minimum seconds between log messages (default: 60.0).

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
                "Batch %d (ID: %s) progress updates every %.1fs: %s - Processing: %d, Succeeded: %d, Errored: %d, Elapsed: %.1fs",
                progress.batch_num,
                progress.batch_id,
                log_interval,
                progress.status.value,
                counts.get("processing", 0),
                counts.get("succeeded", 0),
                counts.get("errored", 0),
                progress.elapsed_time,
            )
            last_log_time = current_time

    return log_progress
