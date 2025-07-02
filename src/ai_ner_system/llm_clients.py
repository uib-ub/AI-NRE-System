"""LLM client implementations for medieval text processing application.

This module provides client implementations for various LLM services including
Anthropic Claude and Ollama.
"""

import anthropic
import tiktoken
import requests
import logging
import json

from abc import abstractmethod, ABC
from ai_ner_system.config import Config

class LLMClientError(Exception):
    """Custom exception for LLM client errors."""

class APIError(LLMClientError):
    """Exception raised for API errors."""

class Client(ABC):
    """Abstract base class for LLM clients"""

    def __init__(self, model: str) -> None:
        """
        Initialize the client with the LLM name.

        Args:
            model (str): The name of the LLM to use.
        """
        self.model = model

    @abstractmethod
    def call(self, prompt: str) -> str:
        """Call the LLM with the given prompt.

        Args:
            prompt: The input prompt to send to the LLM.

        Returns:
            The response text from the LLM.

        Raises:
            LLMClientError: If the API call fails.
        """
        pass

class ClaudeClient(Client):
    """
    Client for interacting with Anthropic Claude API
    """

    def __init__(self,
        api_key: str,
        model: str,
        max_tokens: int = 20000,
        temperature: float = 0.0,
    ) -> None:
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key.
            model: Claude model to use.
            max_tokens: Maximum tokens in response.
            temperature: Response randomness (0.0 = deterministic).

        Raises:
            ValueError: If required parameters are missing (for backward compatibility).
            LLMClientError: If client initialization fails.
        """
        if not api_key:
            raise ValueError('API key must be provided for ClaudeClient.')

        if not model:
            raise ValueError('Model must be provided for ClaudeClient.')

        super().__init__(model)

        # self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.tokenizer = tiktoken.get_encoding('cl100k_base')
        except Exception as e:
            raise LLMClientError(f'Failed to initialize Claude client: {e}') from e

        logging.info('Claude Client initialized with model=%s', model)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in the given text.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens in the text.
        """

        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logging.warning('Token counting failed: %s', e)
            return 0

    def call(self, prompt: str) -> str:
        """Call Claude API with the given prompt.

        Args:
            prompt (str): The input prompt to send to the Claude model.

        Returns:
            str: The response text from the Claude model.

        Raises:
            ValueError: If the prompt is empty.
        """
        if not prompt or not prompt.strip():
            raise ValueError('Prompt must not be empty for ClaudeClient.')

        try:
            token_count = self._count_tokens(prompt)
            logging.info('Prompt Token Count: %d ', token_count)

            system_message = (
                'You are an expert for understanding and analyzing medieval texts and manuscripts, '
                'and you can markup all PROPER NOUNS in all kinds of medieval texts'
            )

            messages = [
                {"role": "user", "content": prompt}
            ]

            response = self.client.messages.create(
                model=self.model,
                system=system_message,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=1,
                top_p=1.0,
                stream=False
            )

            if not response.content or not response.content[0].text:
                logging.error('Empty response received from Claude API')
                return 'Claude API call failed'

            return response.content[0].text

        except anthropic.RateLimitError as e:
            logging.error('Claude API rate limit exceeded: %s', e, exc_info=True)
            return 'Claude API rate limit exceeded'
        except anthropic.APIError as e:
            logging.error('Claude API error: %s', e, exc_info=True)
            return 'Claude API call failed'
        except Exception as e:
            logging.error('Claude API call failed: %s', e, exc_info=True)
            return 'Claude API call failed'

class OllamaClient(Client):
    """Client for interacting with Ollama via OpenWebUI API
    """

    def __init__(
        self,
        endpoint: str,
        token: str,
        model: str,
        timeout: int = 10800,
        temperature: float = 0.0
    ) -> None:
        """Initialize Ollama client.

        Args:
            endpoint: OpenWebUI endpoint URL.
            token: Authentication token.
            model: Ollama model to use.
            temperature: Response randomness (0.0 = deterministic).

        Raises:
            ValueError: If required parameters are missing (for backward compatibility).
            LLMClientError: If client initialization fails.
        """

        if not endpoint:
            raise ValueError('Endpoint must be provided for OllamaClient.')

        if not token:
            raise ValueError('Token must be provided for OllamaClient.')

        if not model:
            raise ValueError('Model must be provided for OllamaClient.')

        super().__init__(model)

        self.endpoint = endpoint
        self.token = token
        self.timeout = timeout
        self.temperature = temperature

        logging.info('Ollama Client initialized with model=%s',  model)

    def call(self, prompt: str) -> str:
        """Call Ollama API with the given prompt.

        Args:
            prompt (str): The input prompt to send to the Ollama model.

        Returns:
            The response text from the Ollama model.

        Raises:
            ValueError: If the prompt is empty.
        """
        if not prompt.strip():
            raise ValueError('Prompt must not be empty for OllamaClient.')

        try:
            logging.info('Sending prompt to Ollama (length: %d)', len(prompt))

            # Prepare API payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": 1.0,
                    "top_k": 1,
                    "seed": 42
                }
            }

            # Prepare headers with authentication
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            }

            # Send request to OpenWebUI/Ollama
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            response_text = response.json().get("response", "")

            if not response_text:
                logging.error('Empty response received from Ollama API')
                return 'Ollama API call failed'

            logging.info('Received Ollama response (length: %d)', len(response_text))

            return response_text

        except requests.exceptions.Timeout as e:
            logging.error('llama API call timed out after %ds: %s', self.timeout, e, exc_info=True)
            return 'Ollama API call timed out'
        except requests.exceptions.RequestException as e:
            logging.error('Ollama API request failed: %s', e, exc_info=True)
            return 'Ollama API request failed'
        except json.JSONDecodeError as e:
            logging.error('Invalid JSON response from Ollama API: %s', e, exc_info=True)
            return 'Ollama API call failed'
        except Exception as e:
            logging.error('Ollama API call failed: %s', e, exc_info=True)
            return "Ollama API call failed"

def create_llm_client(client_type: str, **kwargs) -> Client:
    """Factory function to create LLM clients.

    Args:
        client_type: Type of client ('claude' or 'ollama').
        **kwargs: Additional arguments for client initialization.

    Returns:
        Initialized LLM client.

    Raises:
        LLMClientError: If client type is unsupported or initialization fails.
    """

    client_type = client_type.lower()
    try:
        match client_type:
            case 'claude':
                return ClaudeClient(
                    api_key=Config.ANTHROPIC_API_KEY,
                    model=Config.CLAUDE_MODEL
                )
            case 'ollama':
                return OllamaClient(
                    endpoint=Config.OPENWEBUI_ENDPOINT,
                    token=Config.OPENWEBUI_TOKEN,
                    model=Config.OLLAMA_MODEL
                )
            case _:
                raise LLMClientError(f'Unsupported client type: {client_type}')
    except (ValueError, LLMClientError):
        raise
    except Exception as e:
        raise LLMClientError(f'Failed to create {client_type} client: {e}') from e



