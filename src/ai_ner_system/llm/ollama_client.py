"""Ollama client implementation for interacting with Ollama via OpenWebUI API."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TYPE_CHECKING

import aiohttp
import requests

from .base_client import Client
from .exceptions import APIError, LLMClientError, LLMConnectionError

if TYPE_CHECKING:  # Only for type-checkers; not needed at runtime.
    from aiohttp import ClientTimeout


class OllamaClient(Client):
    """Client for interacting with Ollama via OpenWebUI API.

    Supports synchronous and asynchronous single-call operations. Async batch
    operations are not supported by this client.
    """

    def __init__(
        self,
        endpoint: str,
        token: str,
        model: str,
        *,
        timeout: int = 10800,  # 3 hours default
        temperature: float = 0.0
    ) -> None:
        """Initialize Ollama client.

        Args:
            endpoint: OpenWebUI endpoint URL.
            token: Authentication token for OpenWebUI.
            model: Ollama model to use.
            timeout: Request timeout in seconds, default 10800 seconds (3 hours).
            temperature: Response randomness (0.0 = deterministic).

        Raises:
            ValueError: If required parameters are missing.
            LLMClientError: If client initialization fails.
        """
        # Validate required parameters
        if not endpoint:
            raise ValueError('Endpoint must be provided for OllamaClient.')
        if not token:
            raise ValueError('Token must be provided for OllamaClient.')
        if not model:
            raise ValueError('Model must be provided for OllamaClient.')
        if timeout <= 0:
            raise ValueError("timeout must be > 0.")
        if not (0.0 <= temperature <= 1.0):
            raise ValueError("temperature must be in [0.0, 1.0].")

        # Initialize base class
        super().__init__(model)

        # Set instance attributes
        self.endpoint = endpoint.rstrip('/')
        self.token = token
        self.timeout = timeout
        self.temperature = temperature

        logging.info(
            'Ollama Client initialized with model=%s, endpoint=%s, timeout=%ds',
            model,
            self.endpoint,
            self.timeout,
        )

    def _build_headers(self) -> dict[str, str]:
        """Build headers for API request.

        Returns:
            Dictionary of headers.
        """
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        """Build payload for API request.

        Args:
            prompt: Input prompt text.

        Return:
           Dictionary of payload
        """
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": 1.0,
                "top_k": 1,
                "seed": 42,
            },
        }

    @staticmethod
    def _validate_prompt(prompt: str) -> None:
        """Validate prompt input.

        Args:
            prompt: Input prompt to validate.

        Raises:
            ValueError: If prompt is empty or invalid.
        """
        if not prompt or not prompt.strip():
            raise ValueError('Prompt must not be empty for OllamaClient.')

    def _extract_text_from_json(self, obj: dict[str, Any]) -> str:
        """Extracts the model text from a JSON response.

        OpenWebUI commonly returns {"response": "..."} for non-streamed calls.

        Args:
          obj: Parsed JSON.

        Returns:
          The textual response.

        Raises:
          APIError: If the response does not contain the expected field.
        """
        text = obj.get('response', '')
        if not isinstance(text, str) or not text:
            raise APIError(
                "Invalid or empty response payload.",
                client_type = self.client_type,
                status_code=None,
            )
        return text

    # ----------------------------------------------------------------------
    # Synchronous single-call
    # ----------------------------------------------------------------------
    def call(self, prompt: str) -> str:
        """Call Ollama API through OpenWebUI with the given prompt.

        Args:
            prompt: The input prompt to send to the Ollama model.

        Returns:
            The response text from the Ollama model.

        Raises:
            ValueError: If the prompt is empty.
            APIError: For HTTP/API errors.
            LLMConnectionError: On network failures.
            LLMClientError: If processing fails.
        """

        self._validate_prompt(prompt)

        headers = self._build_headers()
        payload = self._build_payload(prompt)

        logging.info(
            'Sending request to Ollama (prompt length: %d)', len(prompt)
        )

        try:
            # Send request to OpenWebUI/Ollama
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )

            response.raise_for_status()
            response_data = response.json()

            response_text = self._extract_text_from_json(response_data)

            logging.info('Received Ollama response (length: %d)', len(response_text))
            return response_text

        except requests.exceptions.Timeout as e:
            error_msg = f'Ollama API call timed out after {self.timeout}s'
            logging.error(f'{error_msg}: {e}', exc_info=True)
            raise APIError(
                error_msg,
                client_type=self.client_type,
                operation='single_call'
            ) from e
        except requests.exceptions.ConnectionError as e:
            error_msg = f'Failed to connect to Ollama endpoint: {self.endpoint}'
            logging.error(f'{error_msg}: {e}', exc_info=True)
            raise LLMConnectionError(
                error_msg,
                client_type=self.client_type,
                operation='single_call',
                endpoint=self.endpoint
            ) from e
        except requests.exceptions.RequestException as e:
            error_msg = f'Ollama API request failed: {e}'
            logging.error(f'{error_msg}', exc_info=True)
            raise APIError(
                error_msg,
                client_type=self.client_type,
                operation='single_call',
            ) from e
        except json.JSONDecodeError as e:
            error_msg = f'Invalid JSON response from Ollama API: {e}'
            logging.error(f'{error_msg}', exc_info=True)
            raise APIError(
                error_msg,
                client_type=self.client_type,
                operation='single_call',
            ) from e
        except Exception as e:
            error_msg = f'Ollama API call failed: {e}'
            logging.error(f'{error_msg}', exc_info=True)
            raise LLMClientError(
                error_msg,
                client_type=self.client_type,
                operation='single_call'
            ) from e

    # ----------------------------------------------------------------------
    # Asynchronous single-call
    # ----------------------------------------------------------------------
    async def call_async(self, prompt: str) -> str:
        """Call Ollama API asynchronously through OpenWebUI.

        Args:
            prompt: The input prompt to send to the Ollama model.

        Returns:
            The response text from the Ollama model.

        Raises:
            ValueError: If the prompt is empty.
            APIError: If API call fails.
            LLMConnectionError: On network failures.
            LLMClientError: If processing fails.
        """
        self._validate_prompt(prompt)

        headers = self._build_headers()
        payload = self._build_payload(prompt)

        # Configure proper timeout for aiohttp, annotated only for type-checkers;
        # thanks to postponed annotations, OK at runtime.
        timeout_config: ClientTimeout = aiohttp.ClientTimeout(total=self.timeout)

        logging.info(
            'Sending async request to Ollama (prompt length: %d)',
            len(prompt)
        )

        try:
            # Send request to OpenWebUI/Ollama
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                async with session.post(
                    url=self.endpoint,
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

            response_text = self._extract_text_from_json(data)

            logging.info(
                'Received async Ollama response (length: %d)',
                len(response_text)
            )
            return response_text

        except asyncio.TimeoutError as e:
            error_msg = f'Ollama API request timed out after {self.timeout}s'
            logging.error(f'{error_msg}: {e}', exc_info=True)
            raise APIError(
                error_msg,
                client_type="ollama",
                operation = 'async_single_call',
            ) from e
        except aiohttp.ClientConnectorError as e:
            error_msg = f'Failed to connect to Ollama endpoint: {self.endpoint}'
            logging.error(f'{error_msg}: {e}', exc_info=True)
            raise LLMConnectionError(
                error_msg,
                client_type=self.client_type,
                operation="async_single_call",
                endpoint=self.endpoint,
            ) from e
        except aiohttp.ClientError as e:
            error_msg = f'Ollama API client error: {e}'
            logging.error(f'{error_msg}', exc_info=True)
            raise APIError(
                error_msg,
                client_type=self.client_type,
                operation="async_single_call",
            ) from e
        except Exception as e:
            error_msg = f'Ollama async API call failed: {e}'
            logging.error(f'{error_msg}', exc_info=True)
            raise LLMClientError(
                error_msg,
                client_type=self.client_type,
                operation="async_single_call",
            ) from e