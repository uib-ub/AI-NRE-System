import logging
from pathlib import Path
from typing import Optional

class PromptError(Exception):
    """Custom exception for prompt-related errors."""


class PromptBuilder:
    """PromptBuilder is responsible for constructing prompts from a template file.

    This class handles loading a prompt template from a file and formatting
    it with specific parameters such as Brevid and text content for medieval
    text annotation tasks.

    Attributes:
        prompt_template_file (str): Path to the prompt template file.
        template (str): The content of the prompt template loaded from the file.
    """

    def __init__(self, prompt_template_file: str):
        self.prompt_template_file = prompt_template_file
        self.template: Optional[str] = None
        self._load_prompt_template()
        # self.template = Path(self.prompt_template_file).read_text(encoding="utf-8")

    def _load_prompt_template(self) -> None:
        """Load the prompt template from the specified file.

        Raises:
            PromptError: If the template file cannot be read or is empty.
        """
        try:
            template_path = Path(self.prompt_template_file)

            if not template_path.exists():
                raise PromptError(
                    f'Prompt template file does not exist: {self.prompt_template_file}'
                )

            if not template_path.is_file():
                raise PromptError(
                    f'Prompt template path is not a file: {self.prompt_template_file}'
                )

            self.template = template_path.read_text(encoding="utf-8")

            if not self.template.strip():
                raise PromptError(
                    f'Prompt template file is empty: {self.prompt_template_file}'
                )

            logging.info('Prompt template loaded from %s', self.prompt_template_file)

        except OSError as e:
            raise PromptError(
                f'Error reading prompt template file {self.prompt_template_file}: {e}'
            ) from e

    def build(self, brevid: str, text: str) -> str:
        """Build a formatted prompt from the template.

        Args:
            brevid: The Brevid identifier for the medieval text.
            text: The medieval text to be annotated.

        Returns:
            The formatted prompt string.

        Raises:
            PromptError: If the template is not loaded or is invalid.
            ValueError: If the template is missing required placeholders.
        """

        if not self.template:
            raise PromptError('Prompt template is not loaded or is invalid.')

        if not brevid or not brevid.strip():
            raise ValueError('Brevid must be a non-empty string or whitespace.')

        if not text or not text.strip():
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
