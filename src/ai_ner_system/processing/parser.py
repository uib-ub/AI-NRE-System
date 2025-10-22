"""Response parsing for AI NER System processing.

This module provides functions for parsing LLM responses into structured data
for medieval text annotation tasks.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any, ClassVar, cast

from .entities import EntityRecord
from .exceptions import LLMResponseError, ParseError


class ResponseParser:
    """Parses LLM responses into structured data.

    This parser handles both single-record and batch responses from LLM services,
    extracting annotated text and entity information.
    """

    # Class constants for markers
    JSON_MARKER: ClassVar[str] = "===JSON==="
    RECORD_MARKER: ClassVar[str] = "RECORD "
    RESULT_MARKER: ClassVar[str] = "RESULT:"

    _MAX_SNIPPET: ClassVar[int] = 200  # for logging snippets only

    @staticmethod
    def parse_llm_response(
        brevid: str,
        raw_response: str,
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
                "Empty response from LLM",
                brevid=brevid,
                operation="parse_llm_response",
            )

        # Split response into annotated text and JSON structure
        if ResponseParser.JSON_MARKER in raw_response:
            pre, post = raw_response.split(ResponseParser.JSON_MARKER, 1)
            annotated_text, json_text = pre.strip(), post.strip()
        else:
            logging.warning(
                "No JSON marker found in response, for Brevid=%s",
                brevid,
            )
            annotated_text, json_text = raw_response.strip(), '{"entities":[]}'

        try:
            # Parse entities from JSON
            entities = ResponseParser.parse_entities_json(json_text, brevid)

        except (ParseError, LLMResponseError):
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            raise LLMResponseError(
                f"Failed to parse LLM response for Brevid {brevid}: {e}",
                brevid=brevid,
                operation="parse_llm_response",
                response_text=raw_response[: ResponseParser._MAX_SNIPPET] if raw_response else None,
            ) from e
        else:
            logging.debug(
                "Parsed response for brevid=%s: %d entities",
                brevid,
                len(entities),
            )
            return annotated_text, entities

    @staticmethod
    def parse_entities_json(
        json_text: str,
        brevid: str,
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
            logging.warning("No JSON content to parse for brevid=%s", brevid)
            return []

        data = ResponseParser._parse_json_structure(json_text, brevid)
        entities_data = ResponseParser._validate_entities_structure(data, brevid)
        logging.debug("Parsed %d entities for Brevid=%s", len(entities_data), brevid)

        return ResponseParser._create_entity_records(entities_data, brevid)

    @staticmethod
    def _parse_json_structure(json_text: str, brevid: str) -> dict[str, Any]:
        """Parse and validate JSON structure.

        Args:
            json_text: JSON string to parse.
            brevid: The Brevid identifier for error reporting.

        Returns:
            Parsed JSON data as dictionary.

        Raises:
            ParseError: If JSON parsing fails or structure is invalid.
        """
        # Parse JSON and validate structure
        try:
            data: Any = json.loads(json_text.strip())
        except json.JSONDecodeError as e:
            raise ParseError(
                f"Invalid JSON format: {e}",
                brevid=brevid,
                operation="parse_entities_json",
                parse_type="json",
                # Truncate for error message
                content=json_text[: ResponseParser._MAX_SNIPPET],
            ) from e
        # Validate structure of data
        if not isinstance(data, dict):
            raise ParseError(
                f"Expected JSON object for Brevid {brevid}, got {type(data).__name__}",
                brevid=brevid,
                operation="parse_entities_json",
                parse_type="json_structure",
                # Truncate for error message
                content=json_text[: ResponseParser._MAX_SNIPPET],
            )
        # Cast to dict for type checker (runtime check done above) and return
        return cast("dict[str, Any]", data)

    @staticmethod
    def _validate_entities_structure(data: dict[str, Any], brevid: str) -> list[Any]:
        """Validate and extract entities list from parsed JSON.

        Args:
            data: Parsed JSON data dictionary.
            brevid: The Brevid identifier for error reporting.

        Returns:
            List of entity data objects.

        Raises:
            ParseError: If entities structure is invalid.
        """
        entities_data: Any = data.get("entities", [])

        if not isinstance(entities_data, list):
            raise ParseError(
                f"Entities must be a list, got {type(entities_data).__name__}",
                brevid=brevid,
                operation="parse_entities_json",
                parse_type="entities_structure",
            )
        # Type narrow and return entities data list
        return cast("list[Any]", entities_data)

    @staticmethod
    def _create_entity_records(
        entities_data: list[Any],
        brevid: str,
    ) -> list[EntityRecord]:
        """Create EntityRecord objects from entity data, skipping invalid ones.

        Args:
            entities_data: List of entity data objects.
            brevid: The Brevid identifier for error reporting.

        Returns:
            List of valid EntityRecord objects.
        """
        entities: list[EntityRecord] = []
        failed_count = 0

        for entity_data in entities_data:
            try:
                entity = EntityRecord.create_entity_record(entity_data, brevid)
                logging.info("Created entity record for Brevid=%s: %s", brevid, entity)
                entities.append(entity)
            except Exception as e:  # noqa: BLE001
                failed_count += 1
                logging.warning("Invalid entity data for Brevid=%s: %s", brevid, e)

        ResponseParser._log_entity_creation_results(entities, entities_data, brevid, failed_count)
        return entities

    @staticmethod
    def _log_entity_creation_results(
        entities: list[EntityRecord],
        entities_data: list[Any],
        brevid: str,
        failed_count: int,
    ) -> None:
        """Log the results of entity creation.

        Args:
            entities: Successfully created EntityRecord objects.
            entities_data: Original entity data list.
            brevid: The Brevid identifier.
            failed_count: Number of entities that failed to create.
        """
        if failed_count:
            logging.warning(
                "Parsed %d/%d valid entities for Brevid=%s (%d failed)",
                len(entities),
                len(entities_data),
                brevid,
                failed_count,
            )
        else:
            logging.info("Parsed all %d entities successfully for Brevid=%s", len(entities), brevid)

    @staticmethod
    def parse_batch_response(
        records: list[dict[str, str]],
        raw_response: str,
    ) -> tuple[list[str], list[str]]:
        """Parse batch LLM response into individual record results.

        Args:
            records: Original records list for reference.
            raw_response: Raw response string from LLM.

        Returns:
            Tuple of (annotated_records, metadata_records).
        """
        if not raw_response:
            logging.error("Empty batch response")
            return ResponseParser._create_fallback_records(records)

        logging.debug("Full batch response:\n%s", raw_response)

        try:
            record_sections = ResponseParser._split_batch_response(raw_response, records)
            return ResponseParser._process_record_sections(record_sections, records)
        except Exception:
            logging.exception("Critical error parsing batch response")
            return ResponseParser._create_fallback_records(records)

    @staticmethod
    def _split_batch_response(
        raw_response: str,
        records: list[dict[str, str]],
    ) -> list[str]:
        """Split batch response into individual record sections.

        Args:
            raw_response: Raw response string from LLM.
            records: Original records list for size validation.

        Returns:
            List of record section strings.
        """
        parts = raw_response.split(ResponseParser.RECORD_MARKER)
        record_sections = [part for part in parts if part.strip()]

        if len(record_sections) != len(records):
            logging.warning(
                "Expected %d record sections, found %d. Processing available sections.",
                len(records),
                len(record_sections),
            )

        return record_sections

    @staticmethod
    def _process_record_sections(
        record_sections: list[str],
        records: list[dict[str, str]],
    ) -> tuple[list[str], list[str]]:
        """Process each record section and extract annotated text and entities.

        Args:
            record_sections: List of record section strings.
            records: Original records list for reference.

        Returns:
            Tuple of (annotated_records, metadata_records).
        """
        all_annotated_records: list[str] = []
        all_metadata_records: list[str] = []

        for i, section in enumerate(record_sections):
            logging.debug("record index %d, section: %s", i, section)

            if i >= len(records):
                logging.warning(
                    "More sections (%d) than records (%d). Stopping processing.",
                    len(record_sections),
                    len(records),
                )
                break

            record = records[i]

            try:
                annotated_record, metadata_records = ResponseParser._process_single_record_section(
                    section,
                    record,
                    i,
                    len(records),
                )
                all_annotated_records.append(annotated_record)
                all_metadata_records.extend(metadata_records)
            except Exception:
                logging.exception("Error parsing record %d in batch", i + 1)
                fallback_record = ResponseParser._format_csv_row(
                    record.get("Bindnr", "unknown"),
                    record.get("Brevid", "unknown"),
                    record.get("Tekst", "unknown"),
                )
                all_annotated_records.append(fallback_record)

        return all_annotated_records, all_metadata_records

    @staticmethod
    def _process_single_record_section(
        section: str,
        record: dict[str, str],
        index: int,
        total_records: int,
    ) -> tuple[str, list[str]]:
        """Process a single record section and extract annotated text and entities.

        Args:
            section: Record section string.
            record: Original record dictionary.
            index: Current record index.
            total_records: Total number of records.

        Returns:
            Tuple of (annotated_record, metadata_records).
        """
        bindnr = record.get("Bindnr", "unknown")
        brevid = record.get("Brevid", "unknown")

        logging.debug("Processing record, Index=%d: Bindnr=%s, Brevid=%s", index, bindnr, brevid)

        result_content = ResponseParser._extract_result_content(section, brevid, index)
        annotated_text, entities = ResponseParser.parse_llm_response(brevid, result_content)

        annotated_record = ResponseParser._format_csv_row(bindnr, brevid, annotated_text)
        logging.info("--- Annotated record for Brevid %s ---\n%s", brevid, annotated_record)

        metadata_records = ResponseParser._format_entity_metadata(entities, brevid)

        logging.info(
            "Parsed record %d/%d successfully for Brevid=%s: %d entities",
            index + 1,
            total_records,
            brevid,
            len(entities),
        )

        return annotated_record, metadata_records

    @staticmethod
    def _extract_result_content(section: str, brevid: str, index: int) -> str:
        """Extract result content from record section.

        Args:
            section: Record section string.
            brevid: The Brevid identifier.
            index: Current record index.

        Returns:
            Extracted result content string.
        """
        if ResponseParser.RESULT_MARKER in section:
            logging.debug(
                "Found RESULT marker in section for Brevid %s and record index %d",
                brevid,
                index,
            )
            return section.split(ResponseParser.RESULT_MARKER, 1)[1].strip()

        logging.warning(
            "No RESULT marker found in section for Brevid %s and record index %d",
            brevid,
            index,
        )
        return section.strip()

    @staticmethod
    def _format_entity_metadata(entities: list[EntityRecord], brevid: str) -> list[str]:
        """Format entity metadata records.

        Args:
            entities: List of EntityRecord objects.
            brevid: The Brevid identifier.

        Returns:
            List of formatted metadata record strings.
        """
        metadata_records: list[str] = []
        for entity in entities:
            metadata_record = entity.to_csv_row()
            logging.info("--- Metadata for Brevid %s ---\n%s", brevid, metadata_record)
            metadata_records.append(metadata_record)
        return metadata_records

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
        writer = csv.writer(buf, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow([bindnr, brevid, text])
        return buf.getvalue().rstrip("\r\n")

    @staticmethod
    def _create_fallback_records(
        records: list[dict[str, str]],
    ) -> tuple[list[str], list[str]]:
        """Create fallback records when parsing fails.

        Args:
            records: Original records to convert to fallback format.

        Returns:
            Tuple of (annotated_records, empty_metadata_list).
        """
        fallback_records = [
            ResponseParser._format_csv_row(
                record.get("Bindnr", "unknown"),
                record.get("Brevid", "unknown"),
                record.get("Tekst", "unknown"),
            )
            for record in records
        ]
        return fallback_records, []
