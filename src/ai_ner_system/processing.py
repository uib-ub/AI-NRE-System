"""Processing module for medieval text annotation with LLM services.

This module provides classes and functions for processing medieval text records
using Large Language Models, with support for both individual and batch processing.
"""
import asyncio
import json
import logging
import time

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable, Union, Any

from ai_ner_system.prompts import PromptBuilder
from ai_ner_system.llm_clients import Client, BatchProgress, BatchRequest

class ProcessingError(Exception):
    """Base exception for processing-related errors."""


class ValidationError(ProcessingError):
    """Exception raised when data validation fails."""


class LLMResponseError(ProcessingError):
    """Exception raised when LLM response parsing fails."""


class BatchProcessingError(ProcessingError):
    """Exception for batch processing failures."""


@dataclass
class EntityRecord:
    """Data class representing an entity record extracted from medieval text by LLM response.

    Attributes:
        name: The proper noun itself.
        type: Type of proper noun (Person Name, Place Name, etc.).
        preposition: Preposition used with the proper noun (if applicable, otherwise use “N/A”),
        order: Order of occurrence in the text.
        brevid: The Brevid identifier from the source record.
        description: Status/occupation for people, type for places.
        gender: Gender information, "Male", "Female", or "N/A" for non-persons.
        language: Language code (ISO 639-3) (e.g., "lat", "non").
    """
    name: str
    type: str
    preposition: str
    order: int
    brevid: str
    description: str = ""
    gender: str = ""
    language: str = ""

    def to_csv_row(self) -> str:
        """Convert entity record to CSV row format.

        # Returns:
            Semicolon-separated string representation.
        """
        return ";".join([
            self.name,
            self.type,
            self.preposition,
            str(self.order),
            self.brevid,
            self.description,
            self.gender,
            self.language
        ])

    @classmethod
    def create_entity_record(cls, entity_data: Dict[str, str], brevid: str) -> 'EntityRecord':
        """Create an EntityRecord from dictionary data.

        Args:
            entity_data: Dictionary containing entity information.
            brevid: Brevid identifier for the record.

        Returns:
            An instance of EntityRecord.

        Raises:
            ValidationError: If required fields are missing or invalid.
        """
        try:
            return cls(
                name=str(entity_data.get("name", "")).strip(),
                type=str(entity_data.get("type", "")).strip(),
                preposition=str(entity_data.get("preposition", "N/A")).strip(),
                order=int(entity_data.get("order", 0)),
                brevid=str(entity_data.get("brevid", brevid)).strip(),
                description=str(entity_data.get("description", "")).strip(),
                gender=str(entity_data.get("gender", "")).strip(),
                language=str(entity_data.get("language", "")).strip()
            )
        except (ValueError, TypeError) as e:
            raise ValidationError(f'Invalid entity data: {e}') from e

@dataclass
class ProcessingResult:
    """Represents the result of processing a single record (for async methods).

    Attributes:
        record_id: Unique identifier for the record.
        brevid: Brevid ID from the source record.
        annotated_text: Text with proper nouns marked up.
        entities: List of extracted entities.
        processing_time: Time taken to process in seconds.
        success: Whether processing was successful.
        error_message: Error message if processing failed.
    """
    record_id: str
    brevid: str
    annotated_text: str = ""
    entities: List[EntityRecord] = field(default_factory=list)
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class BatchProcessingResult:
    """Represents the result of processing a batch of records (for async methods).

    Attributes:
        batch_id: Unique identifier for the batch.
        results: List of individual processing results.
        total_processing_time: Total time for batch processing.
        successful_count: Number of successfully processed records.
        failed_count: Number of failed records.
        batch_info: Additional batch information from the API.
    """
    batch_id: str
    results: List[ProcessingResult] = field(default_factory=list)
    total_processing_time: float = 0.0
    successful_count: int = 0
    failed_count: int = 0
    batch_info: Optional[Dict[str, Any]] = None


class RecordProcessor:
    """Handles processing of individual CSV records through LLM services."""

    def __init__(self, llm_client: Client, prompt_builder: PromptBuilder) -> None:
        """Initialize the RecordProcessor with LLM client and prompt builder.

        Args:
            llm_client: Instance of LLM client (ClaudeClient or OllamaClient)
            prompt_builder: Instance of PromptBuilder
        """
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder

    def process_record(self, record: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """
        Send one record to the LLM, parse results and return
        annotated text plus metadata rows.

        Args:
            record: Dict with keys "Brevid" and "Tekst"
        Return:
            Tuple (annotated_record, metadata)
        """
        # Extract required fields from the record
        bindnr = record["Bindnr"]
        brevid = record["Brevid"]

        try:
            # Validate required fields
            self._validate_record(record)

            # Build prompt using the prompt builder (single record)
            prompt = self.prompt_builder.build(record)
            logging.debug('--- Prompt ---\n%s', prompt)

            # Call LLM
            raw_response = self._call_llm(brevid, prompt)
            logging.debug('--- RAW RESPONSE for Brevid %s ---\n%s', brevid, raw_response)

            # Parse response
            annotated_text, entities = self._parse_llm_response(brevid, raw_response)

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
        """Process multiple records in a single LLM call.

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
            for record in records:
                self._validate_record(record)

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
            annotated_records, metadata_records = self._parse_batch_response(records, raw_response)

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

        try:
            # Validate record
            self._validate_record(record)

            # Build prompt
            prompt = self.prompt_builder.build(record)

            # Call LLM asynchronously
            response = await self.llm_client.call_async(prompt)

            # Parse response
            annotated_text, entities = self._parse_llm_response(record["Brevid"], response)

            processing_time = time.time() - start_time

            return ProcessingResult(
                record_id = record_id,
                brevid = record["Brevid"],
                annotated_text = annotated_text,
                entities = entities,
                processing_time = processing_time,
                success = True
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to process record having Bindnr_Brevid: {record_id}: {e}"
            logging.error(error_msg, exc_info=True)

            return ProcessingResult(
                record_id = record_id,
                brevid = record.get("Brevid", "unknown"),
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
                    self._validate_record(record)
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

            # Parse batch responses
            results = []
            successful_count = 0
            failed_count = 0

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

                if not response:
                    # No response found for this record
                    logging.warning(f'No response found for record index {i}, Bindnr {record.get("Bindnr", "unknown")} Brevid {record.get("Brevid", "unknown")}')
                    results.append(ProcessingResult(
                        record_id = f'record_{i}_{record.get("Bindnr", "unknown")}_{record.get("Brevid", "unknown")}',
                        brevid=record.get("Brevid", "unknown"),
                        success=False,
                        error_message=f'No response received for record index {i} with Bindnr {record.get("Bindnr", "unknown")} Brevid {record.get("Brevid", "unknown")}'
                    ))
                    failed_count += 1
                    continue

                if response.success:
                    try:
                        annotated_text, entities = self._parse_llm_response(
                            record["Brevid"], response.response_text
                        )

                        results.append(ProcessingResult(
                            record_id=response.custom_id,
                            brevid=record.get("Brevid", "unknown"),
                            annotated_text=f'{record.get("Bindnr")};{record.get("Brevid")};{annotated_text}',
                            entities=entities,
                            success=True
                        ))
                        successful_count += 1

                    except Exception as e:
                        logging.error(
                            f'Failed to parse LLM response for custom id {response.custom_id} with Brevid {record["Brevid"]}: {e}')
                        results.append(ProcessingResult(
                            record_id=response.custom_id,
                            brevid=record.get("Brevid", "unknown"),
                            success=False,
                            error_message=f"Failed to parse LLM response for custom id {response.custom_id} with Brevid {record['Brevid']}: {e}"
                        ))
                        failed_count += 1
                else:
                    results.append(ProcessingResult(
                        record_id=response.custom_id,
                        brevid=record.get("Brevid", "unknown"),
                        success=False,
                        error_message=response.error_message
                    ))
                    failed_count += 1

            total_processing_time = time.time() - start_time

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
                    record_id = f'record_{i}_{records[i].get("Bindnr", "unkown")}_{records[i].get("Brevid", "unknown")}',
                    brevid = records[i].get("Brevid", 'unknown'),
                    success = False,
                    error_message = str(result)
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
            batch_id = batch_id,
            results = processed_results,
            total_processing_time = total_processing_time,
            successful_count = successful_count,
            failed_count = failed_count
        )

    def _parse_batch_response(self, records: List[Dict[str, str]],
                              raw_response: str) -> Tuple[List[str], List[str]]:
        """Parse batch LLM response into individual record results.

        Args:
            records: Original records list for reference.
            raw_response: Raw response string from LLM.

        Returns:
            Tuple of (annotated_records, metadata_records).
        """
        all_annotated_records = []
        all_metadata_records = []

        try:
            # Split response by RECORD markers
            record_sections = raw_response.split('RECORD ')[1:]  # Skip empty first element

            if len(record_sections) != len(records):
                logging.warning('Expected %d record sections, found %d. Processing available sections.',
                                len(records), len(record_sections))

            # Process each record section
            for i, section in enumerate(record_sections):
                logging.debug('record index %d, section: %s', i, section)

                if i >= len(records):
                    break

                try:
                    record = records[i]
                    bindnr = record["Bindnr"]
                    brevid = record["Brevid"]

                    logging.debug('Processing record %d: Bindnr=%s, Brevid=%s', i, bindnr, brevid)
                    # Extract result content (after "RESULT:")
                    if 'RESULT:' in section:
                        logging.debug('Found RESULT marker in section for Brevid %s and record index %d', brevid, i)
                        result_content = section.split('RESULT:', 1)[1]
                    else:
                        logging.warning('No RESULT marker found in section for Brevid %s and record index %d', brevid, i)
                        result_content = section

                    # Parse as single record response
                    annotated_text, entities = self._parse_llm_response(brevid, result_content)

                    # Build output records
                    annotated_record = self._build_annotated_record(bindnr, brevid, annotated_text)
                    metadata_record = self._build_metadata_record(entities, brevid)

                    logging.info('--- annotated record ---\n%s', annotated_record)
                    logging.info('--- metadata ---\n%s', metadata_record)

                    all_annotated_records.extend(annotated_record)
                    all_metadata_records.extend(metadata_record)

                except Exception as e:
                    logging.error('Error parsing record %d in batch: %s', i + 1, e)
                    # Add empty record to maintain order
                    record = records[i]
                    empty_record = f"{record['Bindnr']};{record['Brevid']};{record['Tekst']}"
                    all_annotated_records.append(empty_record)

            return all_annotated_records, all_metadata_records

        except Exception as e:
            logging.error('Error parsing batch response: %s', e)
            # Return original records as fallback
            fallback_records = []
            for record in records:
                fallback_record = f"{record['Bindnr']};{record['Brevid']};{record['Tekst']}"
                fallback_records.append(fallback_record)
            return fallback_records, []

    @staticmethod
    def _validate_record(record: Dict[str, str]) -> None:
        """Validate that the record contains required fields.

        Args:
            record: Record dictionary to validate.

        Raises:
            ValidationError: If required fields are missing or invalid.
        """
        required_fields = ["Bindnr", "Brevid", "Tekst"]
        missing_fields = [field for field in required_fields if not record.get(field)]

        if missing_fields:
            raise ValidationError(f"Missing required fields: {missing_fields}")

        # Validate field content
        if not record["Brevid"].strip():
            raise ValidationError('Brevid cannot be empty')

        if not record["Tekst"].strip():
            raise ValidationError('Tekst cannot be empty')

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
                raise ProcessingError('LLM returned error response')

            logging.debug('Received LLM response for %s (length: %d)', identifier, len(raw_response))
            return raw_response
        except Exception as e:
            raise ProcessingError(f'Error during LLM call for {identifier}: {e}') from e

    def _parse_llm_response(self, brevid: str, raw_response: str) -> Tuple[str, List[EntityRecord]]:
        """Parse the raw LLM response into annotated text and entities.

        Args:
            brevid: The Brevid identifier.
            raw_response: The raw response string from the LLM.

        Returns:
            Tuple of (annotated_text, entities_list).

        Raises:
            LLMResponseError: If response parsing fails.
        """
        try:
            # Split response into annotated text and JSON structure
            if "===JSON===" in raw_response:
                annotated_text, json_text = raw_response.split("===JSON===", 1)
            else:
                logging.warning('No JSON marker found in response for Brevid %s', brevid)
                annotated_text, json_text = raw_response, '{"entities":[]}'

            annotated_text = annotated_text.strip()

            # Parse JSON entities
            entities = self._parse_entities_json(json_text, brevid)

            return annotated_text, entities

        except Exception as e:
            raise LLMResponseError(f'Failed to parse LLM response: {e}') from e

    @staticmethod
    def _parse_entities_json(json_text: str, brevid: str) -> List[EntityRecord]:
        """Parse the JSON entities section of the LLM response.

        Args:
            json_text: JSON string containing entities.
            brevid: The Brevid identifier for logging.

        Returns:
            List of EntityRecord objects.

        Raises:
            LLMResponseError: If JSON parsing fails.
        """
        try:
            # Clean up JSON text
            json_text = json_text.strip()

            # Extract entities
            entities_data = json.loads(json_text).get("entities", [])
            logging.debug('Parsed %d entities for Brevid %s', len(entities_data), brevid)

            entities = []
            for entity_data in entities_data:
                try:
                    entity = EntityRecord.create_entity_record(entity_data, brevid)
                    logging.info('Created entity record for Brevid %s: %s', brevid, entity)
                    entities.append(entity)
                except ValidationError as e:
                    logging.warning('Invalid entity data for Brevid %s: %s', brevid, e)
                    continue
            logging.info('Parsed %d valid entities for Brevid %s', len(entities), brevid)
            return entities

        except Exception as e:
            raise LLMResponseError(f'Failed to parse entities JSON for Brevid {brevid}: {e}') from e

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


def create_progress_tracker() -> Tuple[Callable[[BatchProgress], None], Callable[[], Dict[str, Any]]]:
    """Create a progress tracker that stores batch information."""
    stats = {
        'start_time': time.time(),
        'last_update': time.time(),
        'status': 'starting',
        'request_counts': {},
        'elapsed_time': 0
    }

    def update_progress(progress: BatchProgress) -> None:
        stats.update({
            'last_update': time.time(),
            'status': progress.status.value,
            'request_counts': progress.request_counts.copy(),
            'elapsed_time': progress.elapsed_time
        })

    def get_stats() -> Dict[str, Any]:
        return stats.copy()

    return update_progress, get_stats