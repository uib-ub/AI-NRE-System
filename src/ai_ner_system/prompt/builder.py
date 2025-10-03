"""Prompt building and template management for AI NER System.

This module provides abstract and concrete implementations for building
prompts from templates, with robust validation and error handling for
medieval text annotation tasks.

Template expectations:
  * Single-record prompts require "Brevid" and "Tekst" placeholders.
  * Batch prompts require "num_records" and "batch_content" placeholders.

Note: Input records are expected to carry "Brevid" and "Tekst" (case-sensitive);
they are normalized to lower-case keys internally as {"brevid", "text"}.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from string import Formatter
from typing import ClassVar

from .exceptions import (
    PromptError,
    TemplateNotFoundError,
    PromptBuildError
)

# Type aliases
Pathish = str | Path  # for path-like objects
RecordData = dict[str, str]
BatchData = list[RecordData]
PromptData = RecordData | BatchData


class PromptBuilder(ABC):
    """Abstract base class for prompt builders.

    This class defines the interface for building prompts from templates.
    Subclasses must implement the `build` method to create a formatted prompt.
    """

    DEFAULT_ENCODING: ClassVar[str] = "utf-8"
    # External record keys (source data)
    SRC_KEY_BREVID: ClassVar[str] = "Brevid"
    SRC_KEY_TEXT: ClassVar[str] = "Tekst"

    def __init__(self, template_file: Pathish) -> None:
        """Initialize the PromptBuilder with a template file.

        Args:
            template_file: Path to the template file.
        """
        self.template_file = Path(template_file)
        self.template: str | None = None
        self._load_template()

    def _load_template(self) -> None:
        """Load the template from the specified file.

        Raises:
            TemplateNotFoundError: If the template file doesn't exist.
            PromptError: If the template file cannot be read or is empty.
        """
        if not self.template_file.exists():
            raise TemplateNotFoundError(self.template_file)
        if not self.template_file.is_file():
            raise PromptError(
                f'Template path is not a file: {self.template_file}',
                template_file=self.template_file,
                operation='load'
            )

        try:
            self.template = self.template_file.read_text(
                encoding=self.DEFAULT_ENCODING
            )
            if not self.template.strip():
                raise PromptError(
                    f'Template file is empty: {self.template_file}',
                    template_file=self.template_file,
                    operation='load'
                )
            logging.info('Template loaded successfully from %s', self.template_file)
        except OSError as e:
            raise PromptError(
                f'Error reading template file {self.template_file}: {e}',
                template_file=self.template_file,
                operation='load'
            ) from e

    @staticmethod
    def _extract_placeholders(template: str) -> set[str]:
        """Extracts top-level placeholder names from a format string.

        Args:
          template: The template string using str.format placeholders.

        Returns:
          A set of placeholder field names (root names only).
        """
        fields: set[str] = set()
        for _, field_name, _, _ in Formatter().parse(template):
            if field_name is None:
                continue
            # Strip attribute/index access: "a.b[0]" -> "a"
            before_dot, _, _ = field_name.partition(".")
            root, _, _ = before_dot.partition("[")
            if root:
                fields.add(root)
        return fields

    @staticmethod
    def _require_template_fields(
        present: set[str], required: set[str], template_file: Path
    ) -> None:
        """Ensures `required` is a subset of `present` placeholders.

        Args:
          present: Fields present in the template.
          required: Required field names.
          template_file: For error context.

        Raises:
          PromptBuildError: If any required fields are missing.
        """
        missing = required - present
        if missing:
            raise PromptBuildError(
                f"Template is missing required fields: {sorted(missing)}",
                template_file=template_file,
            )

    @abstractmethod
    def build(self, data: PromptData) -> str:
        """Build a formatted prompt from the template.

        Args:
            data: Either a single record dictionary or a list of record dictionaries.

        Returns:
            Formatted prompt string.

        Raises:
            PromptBuildError: If prompt building fails.
        """

class GenericPromptBuilder(PromptBuilder):
    """Generic prompt builder for single-record and batch prompts.

    This class handles loading a prompt template from a file and formatting
    it with specific parameters such as Brevid and text content for medieval
    text annotation tasks.
    """

    # Required template fields by mode
    REQUIRED_SINGLE: ClassVar[set[str]] = {"brevid", "text"}
    REQUIRED_BATCH: ClassVar[set[str]] = {"num_records", "batch_content"}

    def __init__(self, template_file: Pathish) -> None:
        """Initialize the GenericPromptBuilder with a template file.

        Args:
            template_file: Path to the prompt template file.
        """
        super().__init__(template_file)
        logging.info(
            'GenericPromptBuilder initialized with template: %s',
            template_file
        )

    def build(self, data: PromptData) -> str:
        """Build a formatted prompt from the template with provided data.

        Args:
            data: Either a single record dictionary with "Brevid" and "Tekst" keys,
                  or a list of such dictionaries for synchronous batch processing.
.
        Returns:
            Formatted prompt string.

        Raises:
            PromptBuildError: If data format or template fields are invalid..
        """
        if isinstance(data, dict):
            return self._build_single_record_prompt(data)
        elif isinstance(data, list):
            return self._build_sync_batch_prompt(data)
        else:
            raise PromptBuildError(
                f"Expected dict or list, got {type(data).__name__}",
                template_file=self.template_file,
                data_type=type(data).__name__
            )

    def _build_single_record_prompt(self, record: RecordData) -> str:
        """Build a formatted prompt for a single record.

        Args:
            record: Dict with "Brevid" and "Tekst" keys

        Returns:
            The formatted prompt string.

        Raises:
            PromptBuildError: If template formatting fails.
        """
        if not self.template:
            raise PromptBuildError(
                "Template is not loaded or is invalid.",
                template_file=self.template_file
            )

        # Validate template has required fields for single-record mode.
        present = self._extract_placeholders(self.template)
        self._require_template_fields(
            present, self.REQUIRED_SINGLE, self.template_file
        )

        # Validate and clean record data, normalize to {"brevid","text"}
        cleaned_record = self._validate_and_clean_record(record)

        try:
            prompt = self.template.format(**cleaned_record).strip()
            if not prompt:
                raise PromptBuildError(
                    'Formatted prompt is empty after processing.',
                    template_file=self.template_file
                )
            logging.info(
                'Built single prompt for brevid: %s (text length: %d)',
                cleaned_record['brevid'],
                len(cleaned_record['text'])
            )
            return prompt
        except (KeyError, ValueError, TypeError ) as e:
            raise PromptBuildError(
                f'Template formatting failed: {e}',
                template_file=self.template_file
            ) from e

    def _build_sync_batch_prompt(self, records: BatchData) -> str:
        """Build a formatted prompt for a batch of records processing synchronously.

        Args:
            records: List of record dictionaries with Brevid and Tekst fields.

        Returns:
            Formatted batch prompt string.

        Raises:
            PromptBuildError: If template or data are invalid.
        """
        if not self.template:
            raise PromptBuildError(
                'Batch prompt template is not loaded',
                template_file=self.template_file
            )

        if not records:
            raise PromptBuildError(
                'Records list cannot be empty for batch processing',
                template_file=self.template_file
            )

        # Validate template has required fields for batch mode.
        present = self._extract_placeholders(self.template)
        self._require_template_fields(
            present, self.REQUIRED_BATCH, self.template_file
        )

        # Validate and clean all records
        cleaned_records: list[RecordData] = []
        for i, record in enumerate(records):
            try:
                cleaned_record = self._validate_and_clean_record(record)
                cleaned_records.append(cleaned_record)
            except ValueError as e:
                raise PromptBuildError(
                    f'Record {i + 1} validation failed: {e}',
                    template_file=self.template_file
                ) from e

        # Build batch content
        batch_content = self._format_batch_content(cleaned_records)

        logging.info('Batch content:\n%s', batch_content)

        # Prepare template data
        template_data = {
            'num_records': len(cleaned_records),
            'batch_content': batch_content
        }

        try:
            # Format the template with actual data
            batch_prompt = self.template.format(**template_data).strip()
            logging.info(
                'Built batch prompt for %d records (total length: %d)',
                len(records),
                len(batch_prompt)
            )
            return batch_prompt
        except (KeyError, ValueError, TypeError) as e:
            raise PromptError(
                f'Batch template formatting failed: {e}',
                template_file=self.template_file
            ) from e

    @classmethod
    def _validate_and_clean_record(cls, record: RecordData) -> RecordData:
        """Validates and normalizes a single record.

        Args:
            record: Raw record dictionary with 'Brevid' and 'Tekst'.

        Returns:
            Cleaned record with standardized keys: {'brevid', 'text'}.

        Raises:
            ValueError: If record validation fails.
        """
        raw_brevid = record.get(cls.SRC_KEY_BREVID, '')
        raw_text = record.get(cls.SRC_KEY_TEXT, '')
        brevid = str(raw_brevid).strip()
        text = str(raw_text).strip()

        if not brevid:
            raise ValueError('Brevid must be a non-empty string')
        if not text:
            raise ValueError('Text must be a non-empty string')

        return {"brevid": brevid, "text": text}

    @staticmethod
    def _format_batch_content(records: BatchData) -> str:
        """Format multiple records into batch content.

        Args:
            records: List of cleaned record dictionaries.
                (each with 'brevid' and 'text').

        Returns:
            A formatted batch content string.
        """
        record_sections: list[str] = []
        for i, record in enumerate(records, start=1):
            section = (
                f'RECORD {i}:\n'
                f'Brevid: {record["brevid"]}\n'
                f'Text: """{record["text"]}"""'
            )
            record_sections.append(section)
        return '\n\n'.join(record_sections)