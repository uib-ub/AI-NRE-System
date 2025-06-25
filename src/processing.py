"""Processing module for medieval text annotation with LLM services.

This module provides functionality to process individual records from CSV files,
send them to LLM services for annotation, and parse the results into structured
data for output generation.
"""

import json
import logging

from dataclasses import dataclass
from typing import List, Tuple, Dict
from prompts import PromptBuilder, PromptError
from llm_clients import Client

class ProcessingError(Exception):
    """Custom exception for processing-related errors."""

class ValidationError(ProcessingError):
    """Exception raised when data validation fails."""

class LLMResponseError(ProcessingError):
    """Exception raised when LLM response parsing fails."""

@dataclass
class EntityRecord:
    """Data class representing an entity record extracted from medieval text by LLM response.

    Attributes:
        name: The proper noun itself.
        type: Type of proper noun (Person Name, Place Name, etc.).
        order: Order of occurrence in the text.
        brevid: The Brevid identifier.
        description: Status/occupation or description.
        gender: Gender information.
        language: Language code (ISO 639-3).
    """
    name: str
    type: str
    order: int
    brevid: str
    description: str = ""
    gender: str = ""
    language: str = ""

    def to_csv_row(self) -> str:
        """Convert entity to CSV row format.

        Returns:
            Semicolon-separated string representation.
        """
        return ";".join([
            self.name,
            self.type,
            str(self.order),
            self.brevid,
            self.description,
            self.gender,
            self.language
        ])

    @classmethod
    def create_entity_record(cls, data: Dict[str, str], brevid: str) -> 'EntityRecord':
        """Create an EntityRecord from dictionary data.

        Args:
            data: Dictionary containing entity data.
            brevid: The Brevid identifier.

        Returns:
            An instance of EntityRecord.

        Raises:
            ValidationError: If required fields are missing or invalid.
        """
        try:
            return cls(
                name=str(data.get("name", "")).strip(),
                type=str(data.get("type", "")).strip(),
                order=int(data.get("order", 0)),
                brevid=str(data.get("brevid", brevid)).strip(),
                description=str(data.get("description", "")).strip(),
                gender=str(data.get("gender", "")).strip(),
                language=str(data.get("language", "")).strip()
            )
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid entity data: {e}") from e


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
        text = record["Tekst"]

        try:
            # Validate required fields
            self._validate_record(record)

            # Build prompt
            prompt = self._build_prompt(brevid, text)
            # DEBUG: prompt
            logging.debug("--- Prompt ---\n%s", prompt)

            # Call LLM
            raw_response = self._call_llm(brevid, prompt)
            # raw_data = self.llm_client.call(prompt)
            # DEBUG: raw data
            logging.debug("--- RAW RESPONSE for Brevid %s ---\n%s", brevid, raw_response)

            # Parse response
            annotated_text, entities = self._parse_llm_response(brevid, raw_response)

            # DEBUG: annotated text and entities
            logging.debug("--- annotated text ---\n%s", annotated_text)
            logging.debug("--- entities ---\n%s", entities)

            # Build output records
            annotated_record = self._build_annotated_record(bindnr, brevid, annotated_text)
            # INFO: annotated record
            logging.info("--- annotated record ---\n%s", annotated_record)

            metadata_record = self._build_metadata_record(entities, brevid)
            # INFO: metadata
            logging.info("--- metadata ---\n%s", metadata_record)

            return annotated_record, metadata_record

        except Exception as e:
            logging.error("Error during LLM call for Brevid %s: %s", brevid, e, exc_info=True)
            return [], []

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
            raise ValidationError("Brevid cannot be empty")

        if not record["Tekst"].strip():
            raise ValidationError("Tekst cannot be empty")

    def _build_prompt(self, brevid: str, text: str) -> str:
        """Build prompt using the prompt builder.

        Args:
            brevid: The Brevid identifier.
            text: The medieval text.

        Returns:
            Formatted prompt string.

        Raises:
            ProcessingError: If prompt building fails.
        """
        try:
            prompt = self.prompt_builder.build(brevid, text)
            logging.debug("Built prompt for Brevid %s (length: %d)", brevid, len(prompt))
            return prompt
        except (PromptError, ValueError) as e:
            raise ProcessingError(f"Failed to build prompt: {e}") from e

    def _call_llm(self, brevid: str, prompt: str) -> str:
        """Call the LLM service with the prompt.

        Args:
            brevid: The Brevid identifier for logging.
            prompt: The prompt to send.

        Returns:
            Raw response from the LLM.

        Raises:
            ProcessingError: If LLM call fails.
        """
        try:
            logging.debug("Calling LLM for Brevid %s", brevid)
            raw_response = self.llm_client.call(prompt)

            if not raw_response or raw_response.strip() in ["Claude API call failed", "Ollama API call failed"]:
                raise ProcessingError("LLM returned error response")

            logging.debug("Received LLM response for Brevid %s (length: %d)", brevid, len(raw_response))
            return raw_response
        except Exception as e:
            raise ProcessingError(f"Error during LLM call for Brevid {brevid}: {e}") from e

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
                annotated_text, json_text = raw_response.split("===JSON===")
            else:
                logging.warning("No JSON marker found in response for Brevid %s", brevid)
                annotated_text, json_text = raw_response, '{"entities":[]}'

            annotated_text = annotated_text.strip()

            # Parse JSON entities
            entities = self._parse_entities_json(json_text, brevid)

            return annotated_text, entities

        except Exception as e:
            raise LLMResponseError(f"Failed to parse LLM response: {e}") from e

    @staticmethod
    def _parse_entities_json(json_text: str, brevid: str) -> List[EntityRecord]:
        """Parse the JSON entities section of the LLM response.

        Args:
            json_text: JSON string containing entities.
            brevid: The Brevid identifier for logging.

        Returns:
            # List of entity dictionaries.
            List of EntityRecord objects.

        Raises:
            LLMResponseError: If JSON parsing fails.
        """
        try:
            # Clean up JSON text
            json_text = json_text.strip()

            # Extract entities
            entities_data = json.loads(json_text).get("entities", [])
            logging.debug("Parsed %d entities for Brevid %s", len(entities_data), brevid)

            entities = []
            for entity_data in entities_data:
                try:
                    entity = EntityRecord.create_entity_record(entity_data, brevid)
                    # TODO: DEBUG: log entity
                    logging.info("Created entity record for Brevid %s: %s", brevid, entity)
                    entities.append(entity)
                except ValidationError as e:
                    logging.warning("Invalid entity data for Brevid %s: %s", brevid, e)
                    continue
            # TODO: DEBUG: log entities
            logging.info("Parsed %d valid entities for Brevid %s", len(entities), brevid)

            return entities

        except Exception as e:
            raise LLMResponseError(f"Failed to parse entities JSON for Brevid {brevid}: {e}") from e

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
        logging.debug("Built %d metadata records for Brevid %s", len(metadata_record), brevid)
        return metadata_record
