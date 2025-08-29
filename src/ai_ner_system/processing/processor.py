"""Main processor for medieval text annotation with LLM services.

This module provides the core RecordProcessor class that orchestrates
the processing of medieval text records using LLM services.
"""

import asyncio
import logging
import time
from typing import Dict, List, Tuple, Optional, Callable, Union

from ..llm.base_client import Client
from ..llm.batch_models import BatchProgress, BatchRequest, BatchResponse
from ..prompt.builder import PromptBuilder

from .entities import EntityRecord, ProcessingResult, BatchProcessingResult
from .validator import RecordValidator
from .parser import ResponseParser
from .exceptions import ProcessingError, BatchProcessingError

class RecordProcessor:
    """Main processor for handling medieval text records through LLM services."""

    def __init__(self, llm_client: Client, prompt_builder: PromptBuilder) -> None:
        """Initialize the processor with LLM client and prompt builder.

        Args:
            llm_client: Instance of LLM client (ClaudeClient or OllamaClient)
            prompt_builder: Instance of PromptBuilder
        """
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder

    def process_record(self, record: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """Process a single record through the LLM synchronously.

        Args:
            record: Dictionary with keys "Bindnr", "Brevid", and "Tekst"

        Return:
            Tuple (annotated_record, metadata_record)
        """
        # Extract required fields from the record
        bindnr = record.get("Bindnr", "unknown")
        brevid = record.get("Brevid", "unknown")

        try:
            # Validate required fields
            RecordValidator.validate_record(record)

            # Build prompt using the prompt builder (single record)
            prompt = self.prompt_builder.build(record)
            logging.debug('--- Prompt ---\n%s', prompt)

            # Call LLM
            raw_response = self._call_llm(brevid, prompt)
            logging.debug('--- RAW RESPONSE for Brevid %s ---\n%s', brevid, raw_response)

            # Parse response
            annotated_text, entities = ResponseParser.parse_llm_response(brevid, raw_response)

            # DEBUG: annotated text and entities
            logging.debug('--- annotated text ---\n%s', annotated_text)
            logging.debug('--- entities ---\n%s', entities)

            # Build output records
            annotated_record = self._build_annotated_record(bindnr, brevid, annotated_text)
            metadata_record = self._build_metadata_record(entities, brevid)

            logging.info('--- annotated record ---\n%s', annotated_record)
            logging.info('--- metadata ---\n%s', metadata_record)

            return annotated_record, metadata_record

        except Exception as e:
            logging.error('Error during LLM call for Brevid %s: %s', brevid, e, exc_info=True)
            return [], []

    def process_batch(self, records: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
        """Process multiple records in a single LLM call synchronously.

        Args:
            records: List of record dictionaries to process.

        Returns:
            Tuple of (all_annotated_records, all_metadata_records).
        """
        if not records:
            return [], []

        try:
            # Use original sync batch processing logic
            # Validate all records
            RecordValidator.validate_records(records)

            # Build batch prompt using the prompt builder (list of records)
            batch_prompt = self.prompt_builder.build(records)
            logging.debug('--- Prompt ---\n%s', batch_prompt)

            # Call LLM with batch prompt
            brevids = [record['Brevid'] for record in records]
            batch_id = f"BATCH-{'-'.join(brevids[:3])}..." if len(brevids) > 3 else f"BATCH-{'-'.join(brevids)}"

            raw_response = self._call_llm(batch_id, batch_prompt)
            logging.debug('Received batch response (length: %d)', len(raw_response))
            logging.debug('--- RAW RESPONSE for batch %s ---\n%s', batch_id, raw_response)

            # Parse batch response
            annotated_records, metadata_records = ResponseParser.parse_batch_response(records, raw_response)

            logging.info('Successfully processed batch of %d records: %d annotations, %d metadata',
                         len(records), len(annotated_records), len(metadata_records))

            return annotated_records, metadata_records

        except Exception as e:
            logging.error('Error during batch processing: %s', e, exc_info=True)
            raise  # Let the caller handle fallback

    async def process_record_async(self, record: Dict[str, str]) -> ProcessingResult:
        """Process a single record asynchronously

        Args:
            record: Dictionary containing record data with 'Brevid' and 'Tekst' keys.

        Returns:
            ProcessingResult containing the processed data.
        """
        start_time = time.time()
        record_id = f'{record.get("Bindnr", "unknown")}_{record.get("Brevid", "unknown")}'
        brevid = record.get("Brevid", "unknown")
        bindnr = record.get("Bindnr", "unknown")

        try:
            # Validate record
            RecordValidator.validate_record(record)

            # Build prompt
            prompt = self.prompt_builder.build(record)

            # Call LLM asynchronously
            response = await self.llm_client.call_async(prompt)

            # Parse response
            annotated_text, entities = ResponseParser.parse_llm_response(brevid, response)

            # Build annotated text for result
            formatted_text = f'{bindnr};{brevid};{annotated_text}'

            processing_time = time.time() - start_time

            return ProcessingResult(
                record_id = record_id,
                brevid = brevid,
                annotated_text = formatted_text,
                entities = entities,
                processing_time = processing_time,
                success = True
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f'Failed to process record having Bindnr_Brevid: {record_id}: {e}'
            logging.error(error_msg, exc_info=True)

            return ProcessingResult(
                record_id = record_id,
                brevid = brevid,
                processing_time = processing_time,
                success = False,
                error_message = error_msg
            )

    async def process_batch_async(
            self,
            records: List[Dict[str, str]],
            progress_callback: Optional[Callable[[BatchProgress], None]] = None,
            max_wait_time: int = 86400,
            poll_interval: int = 30,
    ) -> BatchProcessingResult:
        """Process multiple records as a batch asynchronously

        Args:
            records: List of record dictionaries to process.
            progress_callback: Optional callback function to update progress.
            max_wait_time: Maximum time to wait for the batch to complete.
            poll_interval: Time between progress checks

        Return:
            BatchProcessingResult containing all processed records.
        """
        if not records:
            raise ValueError("Records list cannot be empty")

        if not self.llm_client.supports_async_batch():
            # Fallback to individual async processing
            return await self._process_individual_async(records, progress_callback)

        start_time = time.time()

        try:
            # Prepare batch requests
            batch_requests = []
            for i, record in enumerate(records):
                try:
                    RecordValidator.validate_record(record)
                    prompt = self.prompt_builder.build(record) # one record prompt

                    # Create a batch request
                    batch_request = BatchRequest(
                        custom_id = f'record_{i}_{record.get("Bindnr", "unknown")}_{record.get("Brevid", "unknown")}',
                        prompt = prompt,
                        max_tokens = 20000,
                        temperature = 0.0
                    )
                    # Append records into list
                    batch_requests.append(batch_request)

                except Exception as e:
                    logging.error(f'Failed to prepare batch request for record Bindnr {record.get("Bindnr", "unknown")} Brevid {record.get("Brevid", "unknown")}: {e}')
                    continue

            if not batch_requests:
                raise BatchProcessingError('No valid requests to process')

            logging.info(f'Starting async batch processing of {len(batch_requests)} records')

            # Processing batch using LLM client
            batch_responses = await self.llm_client.process_batch_requests_async(
                batch_requests,
                max_wait_time=max_wait_time,
                poll_interval=poll_interval,
                progress_callback=progress_callback
            )

            # # Parse batch responses
            # results = []
            # successful_count = 0
            # failed_count = 0

            # # Create mapping from custom_id to (index, record) for order preservation
            # response_map = {}
            # for response in batch_responses:
            #     if 'record_' in response.custom_id:
            #         try:
            #             # Extract index i from custom_id: "record_{i}_{Bindnr}_{Brevid}"
            #             parts = response.custom_id.split('_')
            #             if len(parts) >= 2:
            #                 index = int(parts[1])
            #                 response_map[index] = response
            #         except (ValueError, IndexError):
            #             logging.warning(f'Could not parse index from custom_id: {response.custom_id}')

            # # Process responses in original order
            # for i, record in enumerate(records):
            #     response = response_map.get(i)

            #     if not response:
            #         # No response found for this record
            #         logging.warning(f'No response found for record index {i}, Bindnr {record.get("Bindnr", "unknown")} Brevid {record.get("Brevid", "unknown")}')
            #         results.append(ProcessingResult(
            #             record_id = f'record_{i}_{record.get("Bindnr", "unknown")}_{record.get("Brevid", "unknown")}',
            #             brevid=record.get("Brevid", "unknown"),
            #             success=False,
            #             error_message=f'No response received for record index {i} with Bindnr {record.get("Bindnr", "unknown")} Brevid {record.get("Brevid", "unknown")}'
            #         ))
            #         failed_count += 1
            #         continue

            #     if response.success:
            #         try:
            #             annotated_text, entities = self._parse_llm_response(
            #                 record["Brevid"], response.response_text
            #             )

            #             results.append(ProcessingResult(
            #                 record_id=response.custom_id,
            #                 brevid=record.get("Brevid", "unknown"),
            #                 annotated_text=f'{record.get("Bindnr")};{record.get("Brevid")};{annotated_text}',
            #                 entities=entities,
            #                 success=True
            #             ))
            #             successful_count += 1


            #        except Exception as e:
            #            logging.error(
            #                f'Failed to parse LLM response for custom id {response.custom_id} with Brevid {record["Brevid"]}: {e}')
            #            results.append(ProcessingResult(
            #                record_id=response.custom_id,
            #                brevid=record.get("Brevid", "unknown"),
            #                success=False,
            #                error_message=f"Failed to parse LLM response for custom id {response.custom_id} with Brevid {record['Brevid']}: {e}"
            #            ))
            #            failed_count += 1
            #    else:
            #        results.append(ProcessingResult(
            #            record_id=response.custom_id,
            #            brevid=record.get("Brevid", "unknown"),
            #            success=False,
            #            error_message=response.error_message
            #        ))
            #        failed_count += 1

            # Parse batch responses and build results
            results = self._build_batch_results(records, batch_responses)

            total_processing_time = time.time() - start_time

            successful_count = sum(1 for r in results if r.success)
            failed_count = len(results) - successful_count

            logging.info(f'Batch processing completed: {successful_count} successful,'
                         f'{failed_count} failed, {total_processing_time:.2f} seconds total')

            return BatchProcessingResult(
                batch_id = f'batch_{int(start_time)}',
                results = results, # Results in original order
                total_processing_time = total_processing_time,
                successful_count = successful_count,
                failed_count = failed_count
            )

        except Exception as e:
            total_processing_time = time.time() - start_time
            error_msg = f"Batch processing failed: {e}"
            logging.error(error_msg, exc_info=True)

            return BatchProcessingResult(
                batch_id=f"batch_{int(start_time)}_failed",
                results=[],
                total_processing_time=total_processing_time,
                successful_count=0,
                failed_count=len(records)
            )

    async def _process_individual_async(
            self,
            records: List[Dict[str, str]],
            progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ) -> BatchProcessingResult:
        """Fallback to individual async processing when batch is not supported."""
        start_time = time.time()
        # TODO: check batch_id definition
        batch_id = f'individual_batch_{int(start_time)}'

        # Process records concurrently using individual async calls
        async_tasks = []
        for record in records:
            async_task = self.process_record_async(record)
            async_tasks.append(async_task)

        # Execute all tasks concurrently
        results: List[Union[ProcessingResult, Exception]] = await asyncio.gather(*async_tasks, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results = []
        successful_count = 0
        failed_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    record_id=f'record_{i}_{records[i].get("Bindnr", "unkown")}_{records[i].get("Brevid", "unknown")}',
                    brevid=records[i].get("Brevid", 'unknown'),
                    success=False,
                    error_message=str(result)
                ))
                failed_count += 1
            elif isinstance(result, ProcessingResult):
                processed_results.append(result)
                if result.success:
                    successful_count += 1
                else:
                    failed_count += 1

        total_processing_time = time.time() - start_time

        logging.info(f'Individual async processing completed: {successful_count} successful, '
                     f'{failed_count} failed, {total_processing_time:.2f} seconds total')

        return BatchProcessingResult(
            batch_id=batch_id,
            results=processed_results,
            total_processing_time=total_processing_time,
            successful_count=successful_count,
            failed_count=failed_count
        )

    @staticmethod
    def _build_batch_results(
            records: List[Dict[str, str]],
            batch_responses: List[BatchResponse]
    ) -> List[ProcessingResult]:
        """Build processing results from batch responses.

        Args:
            records: Original list of records.
            batch_responses: List of BatchResponse objects.

        Returns:
            List of ProcessingResult objects.
        """
        results = []

        # Create mapping from custom_id to (index, record) for order preservation
        response_map = {}
        for response in batch_responses:
            if 'record_' in response.custom_id:
                try:
                    # Extract index i from custom_id: "record_{i}_{Bindnr}_{Brevid}"
                    parts = response.custom_id.split('_')
                    if len(parts) >= 2:
                        index = int(parts[1])
                        response_map[index] = response
                except (ValueError, IndexError):
                    logging.warning(f'Could not parse index from custom_id: {response.custom_id}')

        # Process responses in original order
        for i, record in enumerate(records):
            response = response_map.get(i)
            brevid = record.get("Brevid", "unknown")
            bindnr = record.get("Bindnr", "unknown")

            if not response:
                # No response found for this record
                logging.warning(
                    f'No response found for record index {i}, Bindnr {bindnr} Brevid {brevid}')
                results.append(ProcessingResult(
                    record_id=f'record_{i}_{bindnr}_{brevid}',
                    brevid=brevid,
                    success=False,
                    error_message=f'No response received for record index {i} with Bindnr {bindnr} Brevid {brevid}'
                ))
                continue

            if response.success:
                try:
                    annotated_text, entities = ResponseParser.parse_llm_response(
                        brevid, response.response_text
                    )

                    formatted_text = f'{bindnr};{brevid};{annotated_text}'

                    results.append(ProcessingResult(
                        record_id=response.custom_id,
                        brevid=brevid,
                        annotated_text=formatted_text,
                        entities=entities,
                        success=True
                    ))

                except Exception as e:
                    logging.error(
                        f'Failed to parse LLM response for custom id {response.custom_id} with Brevid {brevid}: {e}')
                    results.append(ProcessingResult(
                        record_id=response.custom_id,
                        brevid=brevid,
                        success=False,
                        error_message=f"Failed to parse LLM response for custom id {response.custom_id} with Brevid {brevid}: {e}"
                    ))
            else:
                results.append(ProcessingResult(
                    record_id=response.custom_id,
                    brevid=brevid,
                    success=False,
                    error_message=response.error_message
                ))
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

            if not raw_response or raw_response.strip() in ['Claude API call failed', 'Ollama API call failed']:
                raise ProcessingError(
                    'LLM returned error response',
                    brevid=identifier
                )

            logging.debug('Received LLM response for %s (length: %d)', identifier, len(raw_response))
            return raw_response

        except Exception as e:
            raise ProcessingError(
                f'Error during LLM call for {identifier}: {e}',
                brevid=identifier
            ) from e

    @staticmethod
    def _build_annotated_record(bindnr: str, brevid: str, annotated_text: str) -> List[str]:
        """Build annotated text records for output.

        Args:
            bindnr: The Bindnr identifier.
            brevid: The Brevid identifier.
            annotated_text: The annotated text.

        Returns:
            List containing the formatted annotated record.
        """
        return [";".join([bindnr, brevid, annotated_text])]

    @staticmethod
    def _build_metadata_record(entities: List[EntityRecord], brevid: str) -> List[str]:
        """Build metadata records from entities.

        Args:
            entities: List of EntityRecord objects.
            brevid: The Brevid identifier for logging.

        Returns:
            List of metadata record strings.
        """
        metadata_record = [entity.to_csv_row() for entity in entities]
        logging.debug('Built %d metadata records for Brevid %s', len(metadata_record), brevid)
        return metadata_record

# Utility functions for batch processing monitoring
def create_progress_logger(log_interval: int = 60) -> Callable[[BatchProgress], None]:
    """Create a progress callback that logs batch status."""
    last_log_time = 0

    def log_progress(progress: BatchProgress) -> None:
        nonlocal last_log_time
        current_time = time.time()

        if current_time - last_log_time > log_interval:
            counts = progress.request_counts
            logging.info(
                f'Batch {progress.batch_id} progress: {progress.status.value} '
                f'(Processing: {counts.get("processing", 0)}), '
                f'Succeeded: {counts.get("succeeded", 0)},  '
                f'Errored: {counts.get("errored", 0)}, '
                f'Elapsed: {progress.elapsed_time:.1f}s, '
            )
            last_log_time = current_time

    return log_progress