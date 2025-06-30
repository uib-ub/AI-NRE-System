import logging
from pathlib import Path
from typing import Optional, Dict, List, Union

from abc import abstractmethod, ABC

class PromptError(Exception):
    """Custom exception for prompt-related errors."""

class PromptBuilder(ABC):
    """Abstract base class for prompt builders.

    This class defines the interface for building prompts from templates.
    Subclasses must implement the `build` method to create a formatted prompt.
    """

    def __init__(self, template_file: str) -> None:
        """Initialize the PromptBuilder with a template file.

        Args:
            template_file: Path to the template file.
        """

        self.template_file = template_file
        self.template: Optional[str] = None
        self._load_template()

    def _load_template(self) -> None:
        """Load the template from the specified file.

        Raises:
            PromptError: If the template file cannot be read or is empty.
        """
        try:
            template_path = Path(self.template_file)

            if not template_path.exists():
                raise PromptError(f'Template file does not exist: {self.template_file}')

            if not template_path.is_file():
                raise PromptError(f'Template path is not a file: {self.template_file}')

            self.template = template_path.read_text(encoding="utf-8")

            if not self.template.strip():
                raise PromptError(f'Template file is empty: {self.template_file}')

            logging.info('Template loaded from %s', self.template_file)

        except OSError as e:
            raise PromptError(f'Error reading template file {self.template_file}: {e}') from e

    @abstractmethod
    def build(self, data: Union[Dict[str, str], List[Dict[str, str]]]) -> str:
        """Build a formatted prompt from the template."""

class GenericPromptBuilder(PromptBuilder):
    """GenericPromptBuilder is responsible for constructing a prompt from a template file.

    This class handles loading a prompt template from a file and formatting
    it with specific parameters such as Brevid and text content for medieval
    text annotation tasks.
    """

    def __init__(self, prompt_template_file: str) -> None:
        """Initialize the GenericPromptBuilder with a template file.
        Args:
            prompt_template_file: Path to the prompt template file.
        """
        super().__init__(prompt_template_file)

    def build(self, data: Union[Dict[str, str], List[Dict[str, str]]]) -> str:
        """Build a formatted prompt from the template with provided data.
        Args:
            data: Either a single record dictionary with keys "Brevid" and "Tekst",
                  or a list of such dictionaries.
        Returns:
            Formatted prompt string based on the template and provided data.
        Raises:
            TypeError: If data is not a dict or list of dicts.
        """

        if isinstance(data, dict):
            return self._build_simple_prompt(data)
        elif isinstance(data, list):
            return self._build_batch_prompt(data)
        else:
            raise TypeError(f'Expected Dict[str, str] or List[Dict[str, str]], got {type(data)}')

    def _build_batch_prompt(self, records: List[Dict[str, str]]) -> str:
        """Build a formatted prompt from the template with a batch of records.

        Args:
            records: List of record dictionaries with Brevid and Tekst fields.

        Returns:
            Formatted batch prompt string.

        Raises:
            ValueError: If records list is empty or contains invalid records.
            PromptError: If template is not loaded or formatting fails.
        """

        if not self.template:
            raise PromptError('Batch prompt template is not loaded')

        if not records:
            raise ValueError('Records list cannot be empty')

        # Validate all records first
        for i, record in enumerate(records):
            brevid = record.get('Brevid', '').strip()
            text = record.get('Tekst', '').strip()

            if not brevid or not text:
                raise ValueError(f'Record {i + 1}: Brevid and Tekst must be non-empty')

        # Build individual record sections
        record_sections = []
        for i, record in enumerate(records, 1):
            brevid = record['Brevid'].strip()
            text = record['Tekst'].strip()

            record_sections.append(f"""RECORD {i}: 
Brevid: {brevid}
Text: \"\"\"{text}\"\"\"""")

        # Create batch content
        batch_content = "\n\n".join(record_sections)

        logging.info('Batch content:\n%s', batch_content)

        try:
            # Format the template with actual data
            batch_prompt = self.template.format(
                num_records=len(records),
                batch_content=batch_content
            )

            logging.info('Built batch prompt for %d records (total length: %d)',
                         len(records), len(batch_prompt))
            return batch_prompt

        except (KeyError, ValueError) as e:
            raise PromptError(f'Failed to format batch prompt template: {e}') from e

    def _build_simple_prompt(self, record: Dict[str, str]) -> str:
        """Build a formatted prompt from the template with a single record.

        Args:
            record: Dict with keys "Brevid" and "Tekst"

        Returns:
            The formatted prompt string.

        Raises:
            PromptError: If the template is not loaded or is invalid.
            ValueError: If the template is missing required placeholders.
        """

        if not self.template:
            raise PromptError('Prompt template is not loaded or is invalid.')

        brevid = record.get('Brevid', '').strip()
        text = record.get('Tekst', '').strip()

        if not brevid:
            raise ValueError('Brevid must be a non-empty string or whitespace.')

        if not text:
            raise ValueError('Text must be a non-empty string or whitespace.')

        try:
            prompt = self.template.format(
                brevid=brevid.strip(),
                text=text.strip()
            ).strip()

            if not prompt:
                raise PromptError('Formatted prompt is empty after processing.')

            logging.info('Built prompt for brevid: %s (text length: %d)',
                          brevid, len(text))

            return prompt

        except KeyError as e:
            raise PromptError(
                f'Template is missing required placeholders: {e}'
            ) from e
        except Exception as e:
            raise PromptError(
                f'Unexpected error during building prompt for brevid {brevid}: {e}'
            ) from e
