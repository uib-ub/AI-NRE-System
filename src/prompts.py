from pathlib import Path

class PromptBuilder:
    def __init__(self, prompt_template_file: str):
        self.prompt_template_file = prompt_template_file
        self.template = Path(self.prompt_template_file).read_text(encoding="utf-8")

    def build(self, brevid: str, text: str) -> str:
        """
        Fill the prompt template for a given Brevid and text from a record.
        Args:
            brevid: The Brevid identifier
            text: The medieval text to annotate
        Returns:
            The formatted prompt string.
        """
        return self.template.format(brevid=brevid, text=text).strip()
        