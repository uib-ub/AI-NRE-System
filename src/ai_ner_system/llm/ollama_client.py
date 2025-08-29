"""Ollama client implementation."""

import asyncio
import aiohttp
import logging
import json
import requests

from .base_client import Client
from .exceptions import APIError, LLMClientError


class OllamaClient(Client):
    """Client for interacting with Ollama via OpenWebUI API

    Note: This client does not support async batch processing as Ollama
    doesn't have a batch API similar to Claude.
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
            timeout: Request timeout in seconds, default 10800 seconds (7.5 days).
            temperature: Response randomness (0.0 = deterministic).

        Raises:
            ValueError: If required parameters are missing (for backward compatibility).
            LLMClientError: If client initialization fails.
        """
        # Validate required parameters
        if not endpoint:
            raise ValueError('Endpoint must be provided for OllamaClient.')
        if not token:
            raise ValueError('Token must be provided for OllamaClient.')
        if not model:
            raise ValueError('Model must be provided for OllamaClient.')

        # Initialize base class
        super().__init__(model)

        # Set instance attributes
        self.endpoint = endpoint.rstrip('/')
        self.token = token
        self.timeout = timeout
        self.temperature = temperature

        logging.info('Ollama Client initialized with model=%s',  model)


    def _prepare_request_data(self, prompt: str) -> tuple[dict, dict]:
        """Prepare headers and payload for API request.

        Args:
            prompt: Input prompt text.

        Returns:
            Tuple of (headers, payload) dictionaries.
        """
        # Prepare headers with authentication
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

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

        return headers, payload

    @staticmethod
    def _validate_prompt(prompt: str) -> None:
        """Validate prompt input.

        Args:
            prompt: Input prompt to validate.

        Raises:
            ValueError: If prompt is empty or invalid.
        """
        if not prompt.strip() or not prompt.strip():
            raise ValueError('Prompt must not be empty for OllamaClient.')


    def call(self, prompt: str) -> str:
        """Call Ollama API through OpenWebUI with the given prompt.

        Args:
            prompt (str): The input prompt to send to the Ollama model.

        Returns:
            The response text from the Ollama model.

        Raises:
            ValueError: If the prompt is empty.
        """

        self._validate_prompt(prompt)

        headers, payload = self._prepare_request_data(prompt)

        try:
            logging.info('Sending request to Ollama (prompt length: %d)', len(prompt))

            # Send request to OpenWebUI/Ollama
            response = requests.post(
                self.endpoint,
                json = payload,
                headers = headers,
                timeout = self.timeout
            )

            response.raise_for_status()
            response_data = response.json()

            response_text = response_data.get("response", "")

            if not response_text:
                logging.error('Empty response received from Ollama API')
                raise APIError(
                    'Empty response received from Ollama API',
                    client_type="ollama",
                    status_code=response.status_code
                )

            logging.info('Received Ollama response (length: %d)', len(response_text))
            return response_text

        except requests.exceptions.Timeout as e:
            error_msg = f'Ollama API call timed out after {self.timeout}s'
            logging.error(f'{error_msg}: {e}', exc_info=True)
            raise APIError(error_msg, client_type="ollama") from e
        except requests.exceptions.RequestException as e:
            error_msg = f'Ollama API request failed: {e}'
            logging.error(error_msg, exc_info=True)
            raise APIError(error_msg, client_type="ollama") from e
        except json.JSONDecodeError as e:
            error_msg = f'Invalid JSON response from Ollama API: {e}'
            logging.error(error_msg, exc_info=True)
            raise APIError(error_msg, client_type="ollama") from e
        except Exception as e:
            error_msg = f'Ollama API call failed: {e}'
            logging.error(error_msg, exc_info=True)
            raise LLMClientError(error_msg, client_type="ollama") from e

    async def call_async(self, prompt: str) -> str:
        """Call Ollama API asynchronously through OpenWebUI.

        Args:
            prompt: The input prompt to send to the Ollama model.

        Returns:
            The response text from the Ollama model.

        Raises:
            ValueError: If the prompt is empty.
        """
        self._validate_prompt(prompt)

        headers, payload = self._prepare_request_data(prompt)

        try:
            logging.info('Sending async request to Ollama (prompt length: %d)', len(prompt))

            # Send request to OpenWebUI/Ollama
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    if 'response' not in result:
                        logging.error('Invalid response format from Ollama')
                        raise APIError(
                            'Invalid response format from Ollama API',
                            client_type="ollama",
                            status_code=response.status
                        )

                    response_text = result.get("response", "")

                    if not response_text:
                        logging.error('Empty response received from Ollama API')
                        raise APIError(
                            "Empty response received from Ollama API",
                            client_type="ollama",
                            status_code=response.status
                        )

                    logging.info('Received async Ollama response (length: %d)', len(response_text))
                    return response_text

        except asyncio.TimeoutError as e:
            error_msg = f'Ollama API request timed out after {self.timeout}s'
            logging.error(f'{error_msg}: {e}', exc_info=True)
            raise APIError(error_msg, client_type="ollama") from e
        except aiohttp.ClientError as e:
            error_msg = f'Ollama API client error: {e}'
            logging.error(error_msg, exc_info=True)
            raise APIError(error_msg, client_type="ollama") from e
        except Exception as e:
            error_msg = f'Ollama async API call failed: {e}'
            logging.error(error_msg, exc_info=True)
            raise LLMClientError(error_msg, client_type="ollama") from e