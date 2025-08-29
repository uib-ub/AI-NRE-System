"""Response parsing for AI NER System processing.

This module provides functions for parsing LLM responses into structured data
for medieval text annotation tasks.
"""
import json
import logging
from typing import Tuple, List, Dict

from .exceptions import LLMResponseError, ParseError, ValidationError
from .entities import EntityRecord

class ResponseParser:
    """Parses LLM responses into structured data."""

    @staticmethod
    def parse_llm_response(brevid: str, raw_response: str) -> Tuple[str, List[EntityRecord]]:
        """Parse the raw LLM response into annotated text and entities.

        Args:
            brevid: The Brevid identifier.
            raw_response: The raw response string from LLM.

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

            # Parse entities from JSON
            entities = ResponseParser.parse_entities_json(json_text, brevid)

            return annotated_text, entities

        except Exception as e:
            raise LLMResponseError(
                f'Failed to parse LLM response for Brevid {brevid}: {e}',
                brevid=brevid,
                response_text=raw_response
            ) from e

    @staticmethod
    def parse_entities_json(json_text: str, brevid: str) -> List[EntityRecord]:
        """Parse the JSON entities section from LLM response.

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

            # Parse JSON
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as e:
                raise ParseError(
                    f'Failed to parse JSON for Brevid {brevid}: {e}',
                    brevid=brevid,
                    parse_type='json',
                    content=json_text
                ) from e

            # Extract entities
            entities_data = data.get("entities", [])
            if not isinstance(entities_data, list):
                raise ParseError(
                    f'Entities must be a list for Brevid {brevid}',
                    brevid=brevid,
                    parse_type='entities',
                )

            logging.debug('Parsed %d entities for Brevid %s', len(entities_data), brevid)

            # Create EntityRecord objects
            entities = []
            for entity_data in entities_data:
                try:
                    entity = EntityRecord.create_entity_record(entity_data, brevid)
                    logging.info('Created entity record for Brevid %s: %s', brevid, entity)
                    entities.append(entity)
                except Exception as e:
                    logging.warning('Invalid entity data for Brevid %s: %s', brevid, e)
                    continue

            logging.info('Parsed %d valid entities for Brevid %s', len(entities), brevid)
            return entities

        except ParseError:
            raise  # Re-raise ParseError as-is
        except Exception as e:
            raise ParseError(
                f'Failed to parse entities JSON for Brevid {brevid}: {e}',
                brevid=brevid,
                parse_type='json',
                content=json_text
            ) from e

    @staticmethod
    def parse_batch_response(records: List[Dict[str, str]],
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

                    logging.debug('Processing record, Index=%d: Bindnr=%s, Brevid=%s', i, bindnr, brevid)
                    # Extract result content (after "RESULT:")
                    if 'RESULT:' in section:
                        logging.debug('Found RESULT marker in section for Brevid %s and record index %d', brevid, i)
                        result_content = section.split('RESULT:', 1)[1]
                    else:
                        logging.warning('No RESULT marker found in section for Brevid %s and record index %d', brevid, i)
                        result_content = section

                    # Parse as single record response
                    annotated_text, entities = ResponseParser.parse_llm_response(brevid, result_content)

                    # Build output records
                    annotated_record = f'{bindnr};{brevid};{annotated_text}'
                    logging.info('--- annotated record ---\n%s', annotated_record)
                    all_annotated_records.append(annotated_record)

                    for entity in entities:
                        metadata_record = entity.to_csv_row()
                        logging.info('--- metadata ---\n%s', metadata_record)
                        all_metadata_records.append(metadata_record)

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