
class PromptError(Exception):
    """Base exception for prompt-related errors."""

    def __init__(self, message: str, template_file: str = None) -> None:
        """Initialize PromptError.

        Args:
            message: Error message.
            template_file: Optional template file path related to the error.
        """
        super().__init__(message)
        self.template_file = template_file

