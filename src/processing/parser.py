"""Response parsing for AI NER System processing.

This module provides functions for parsing LLM responses into structured data
for medieval text annotation tasks.
"""

from __future__ import annotations

import io
import csv
import json
import logging
from typing import Any, cast, ClassVar

from .entities import EntityRecord
from .exceptions import LLMResponseError, ParseError


class ResponseParser:
    """Parses LLM responses into structured data.

    This parser handles both single-record and batch responses from LLM services,
    extracting annotated text and entity information.
    """

    # Class constants for markers
    JSON_MARKER: ClassVar[str] = '===JSON==='
    RECORD_MARKER: ClassVar[str] = 'RECORD '
    RESULT_MARKER: ClassVar[str] = 'RESULT:'

    _MAX_SNIPPET: ClassVar[int] = 200  # for logging snippets only

    @staticmethod
    def parse_llm_response(
        brevid: str,
        raw_response: str
    ) -> tuple[str, list[EntityRecord]]:
        """Parse the raw LLM response into annotated text and entities.

        Args:
            brevid: The Brevid identifier for the record.
            raw_response: The raw response string from LLM.

        Returns:
            Tuple of (annotated_text, entities_list).

        Raises:
            LLMResponseError: If response parsing fails.
        """

        if not raw_response:
            raise LLMResponseError(
                'Empty response from LLM',
                brevid=brevid,
                operation='parse_llm_response',
            )

        try:
            # Split response into annotated text and JSON structure
            if ResponseParser.JSON_MARKER in raw_response:
                pre, post = raw_response.split(ResponseParser.JSON_MARKER, 1)
                annotated_text, json_text = pre.strip(), post.strip()
            else:
                logging.warning(
                    'No JSON marker found in response, for Brevid=%s',
                    brevid
                )
                annotated_text, json_text = raw_response.strip(
                ), '{"entities":[]}'

            # Parse entities from JSON
            entities = ResponseParser.parse_entities_json(json_text, brevid)

            logging.debug(
                'Parsed response for brevid=%s: %d entities',
                brevid, len(entities)
            )

            return annotated_text, entities

        except (ParseError, LLMResponseError):
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            raise LLMResponseError(
                f'Failed to parse LLM response for Brevid {brevid}: {e}',
                brevid=brevid,
                operation='parse_llm_response',
                response_text=raw_response[:ResponseParser._MAX_SNIPPET] if raw_response else None,
            ) from e

    @staticmethod
    def parse_entities_json(
        json_text: str,
        brevid: str
    ) -> list[EntityRecord]:
        """Parse the JSON entities section from LLM response.

        Args:
            json_text: JSON string containing entities data.
            brevid: The Brevid identifier for error reporting.

        Returns:
            List of EntityRecord objects.

        Raises:
            ParseError: If JSON parsing fails or entities are invalid.
        """
        if not json_text or not json_text.strip():
            logging.warning('No JSON content to parse for brevid=%s', brevid)
            return []

        try:
            json_text = json_text.strip()

            # Parse JSON - return Any type
            try:
                data: Any = json.loads(json_text)
            except json.JSONDecodeError as e:
                raise ParseError(
                    f'Invalid JSON format: {e}',
                    brevid=brevid,
                    operation='parse_entities_json',
                    parse_type='json',
                    # Truncate for error message
                    content=json_text[:ResponseParser._MAX_SNIPPET]
                ) from e

            # Validate data structure
            if not isinstance(data, dict):
                raise ParseError(
                    f'Expected JSON object for Brevid {brevid}, got {type(data).__name__}',
                    brevid=brevid,
                    operation='parse_entities_json',
                    parse_type='json_structure',
                    # Truncate for error message
                    content=json_text[:ResponseParser._MAX_SNIPPET]
                )

            # Cast to dict for type checker (runtime check done above)
            data = cast(dict[str, Any], data)

            # Extract entities list with default empty list
            entities_data: Any = data.get('entities', [])

            # Validate it's a list (could be different type if JSON is malformed)
            if not isinstance(entities_data, list):
                raise ParseError(
                    f'Entities must be a list, got {type(entities_data).__name__}',
                    brevid=brevid,
                    operation='parse_entities_json',
                    parse_type='entities_structure',
                )

            # Type narrow after validation
            entities_data = cast(list[Any], entities_data)

            logging.debug(
                'Parsed %d entities for Brevid=%s',
                len(entities_data), brevid
            )

            # Create EntityRecord objects
            entities: list[EntityRecord] = []
            failed_count = 0
            for entity_data in entities_data:
                try:
                    entity = EntityRecord.create_entity_record(
                        entity_data, brevid
                    )
                    logging.info(
                        'Created entity record for Brevid=%s: %s',
                        brevid, entity
                    )
                    entities.append(entity)
                except Exception as e:
                    failed_count += 1
                    logging.warning(
                        'Invalid entity data for Brevid=%s: %s',
                        brevid, e
                    )
                    continue

            if failed_count > 0:
                logging.warning(
                    'Parsed %d/%d valid entities for Brevid=%s (%d entities failed).',
                    len(entities),
                    len(entities_data),
                    brevid,
                    failed_count
                )
            else:
                logging.info(
                    'Parsed all %d entities successfully for Brevid=%s',
                    len(entities), brevid
                )

            return entities

        except ParseError:
            raise  # Re-raise ParseError as-is
        except Exception as e:
            raise ParseError(
                f'Failed to parse entities JSON for Brevid {brevid}: {e}',
                brevid=brevid,
                operation='parse_entities_json',
                parse_type='parse_json',
                # Truncate for error message
                content=json_text[:ResponseParser._MAX_SNIPPET]
            ) from e

    @staticmethod
    def parse_batch_response(
        records: list[dict[str, str]],
        raw_response: str
    ) -> tuple[list[str], list[str]]:
        """Parse batch LLM response into individual record results.

        Args:
            records: Original records list for reference.
            raw_response: Raw response string from LLM.

        Returns:
            Tuple of (annotated_records, metadata_records).
        """
        if not raw_response:
            logging.error('Empty batch response')
            return ResponseParser._create_fallback_records(records)

        all_annotated_records: list[str] = []
        all_metadata_records: list[str] = []

        # TODO: DEBUG: log full response for now
        logging.debug('Full batch response:\n%s', raw_response)

        try:
            # Split response by RECORD markers
            parts = raw_response.split(ResponseParser.RECORD_MARKER)
            record_sections = [part for part in parts if part.strip()]

            if len(record_sections) != len(records):
                logging.warning(
                    'Expected %d record sections, found %d. Processing available sections.',
                    len(records),
                    len(record_sections)
                )

            # Process each record section
            for i, section in enumerate(record_sections):
                logging.debug('record index %d, section: %s', i, section)

                if i >= len(records):
                    logging.warning(
                        'More sections (%d) than records (%d). Stopping processing.',
                        len(record_sections),
                        len(records)
                    )
                    break

                record = records[i]

                try:
                    # Safe access Bindnr and Brevid with defaults
                    bindnr = record.get('Bindnr', 'unknown')
                    brevid = record.get('Brevid', 'unknown')

                    logging.debug(
                        'Processing record, Index=%d: Bindnr=%s, Brevid=%s',
                        i, bindnr, brevid
                    )

                    # Extract result content (after "RESULT:")
                    if ResponseParser.RESULT_MARKER in section:
                        logging.debug(
                            'Found RESULT marker in section for Brevid %s and record index %d', 
                            brevid, i
                        )
                        result_content = section.split(
                            ResponseParser.RESULT_MARKER, 1)[1].strip()
                    else:
                        logging.warning(
                            'No RESULT marker found in section for Brevid %s and record index %d',
                            brevid, i
                        )
                        result_content = section.strip()

                    # Parse as single record response
                    annotated_text, entities = ResponseParser.parse_llm_response(
                        brevid, result_content
                    )

                    # Build output records using CSV writer
                    annotated_record = ResponseParser._format_csv_row(
                        bindnr, brevid, annotated_text
                    )
                    logging.info(
                        '--- Annotated record for Brevid %s ---\n%s',
                        brevid, annotated_record
                    )
                    all_annotated_records.append(annotated_record)

                    # Add entity metadata records
                    for entity in entities:
                        metadata_record = entity.to_csv_row()
                        logging.info(
                            '--- Metadata for Brevid %s ---\n%s',
                            brevid, metadata_record
                        )
                        all_metadata_records.append(metadata_record)

                    logging.info(
                        'Parsed record %d/%d successfully for Brevid=%s: %d entities',
                        i + 1,
                        len(records),
                        brevid,
                        len(entities),
                    )

                except Exception as e:
                    logging.error(
                        'Error parsing record %d in batch: %s',
                        i + 1, e, exc_info=True
                    )
                    # Add fallback record to maintain order
                    fallback_record = ResponseParser._format_csv_row(
                        record.get('Bindnr', 'unknown'),
                        record.get('Brevid', 'unknown'),
                        record.get('Tekst', 'unknown')
                    )
                    all_annotated_records.append(fallback_record)

            return all_annotated_records, all_metadata_records

        except Exception as e:
            logging.error(
                f'Critical error parsing batch response: {e}',
                exc_info=True
            )
            # Return original records as fallback
            return ResponseParser._create_fallback_records(records)

    @staticmethod
    def _format_csv_row(bindnr: str, brevid: str, text: str) -> str:
        """Format a CSV row with proper quoting.

        Args:
            bindnr: The Bindnr number.
            brevid: The Brevid identifier.
            text: The annotated text.

        Returns:
            Semicolon-separated CSV row string.
        """
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=';', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([bindnr, brevid, text])
        return buf.getvalue().rstrip('\r\n')

    @staticmethod
    def _create_fallback_records(
        records: list[dict[str, str]]
    ) -> tuple[list[str], list[str]]:
        """Create fallback records when parsing fails.

        Args:
            records: Original records to convert to fallback format.

        Returns:
            Tuple of (annotated_records, empty_metadata_list).
        """
        fallback_records = [
            ResponseParser._format_csv_row(
                record.get('Bindnr', 'unknown'),
                record.get('Brevid', 'unknown'),
                record.get('Tekst', 'unknown'),
            )
            for record in records
        ]
        return fallback_records, []
