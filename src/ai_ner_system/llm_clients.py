"""LLM client implementations for medieval text processing application.

This module provides client implementations for various LLM services including
Anthropic Claude and Ollama.
"""

import anthropic
import tiktoken
import requests
import logging
import json
import time
import asyncio
import aiohttp

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, AsyncIterator, Dict, List, Any
from abc import abstractmethod, ABC
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from ai_ner_system.config import Config

class LLMClientError(Exception):
    """Base exception for LLM client errors."""


class APIError(Exception):
    """Exception for API-related errors."""


class BatchTimeoutError(LLMClientError):
    """Exception raised when batch processing times out."""


class BatchStatus(Enum):
    """Enum for batch processing status.

    Based on Anthropic's Message Batches API documentation:
    - in_progress: The batch is currently being processed
    - ended: The batch has completed processing (success or failure)
    """
    IN_PROGRESS = 'in_progress'
    ENDED = 'ended'
    # COMPLETED = 'COMPLETED'
    # FAILED = 'FAILED'
    # EXPIRED = 'EXPIRED'
    # CANCELLED = 'CANCELLED'

@dataclass
class BatchRequest:
    """Represents a single request in a batch using Claude Batches API.

    Attributes:
        custom_id: Unique identifier for this request within the batch.
        prompt: The input prompt text to process.
        max_tokens: Maximum number of tokens in the response.
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative).
    """
    custom_id: str
    prompt: str
    max_tokens: int = 20000
    temperature: float = 0.0

@dataclass
class BatchResponse:
    """Represents a response from batch processing using Claude Batches API.

    Attributes:
        custom_id: The unique identifier from the original request.
        response_text: The generated response text.
        success: Whether the request was processed successfully.
        error_message: Error message if success is False.
    """
    custom_id: str
    response_text: str
    success: bool
    error_message: Optional[str] = None

@dataclass
class BatchProgress:
    """Represents progress information for a batch job.

    Attributes:
        batch_id: The unique identifier for the batch job.
        status: Current status of the batch.
        elapsed_time: Time elapsed since batch creation (seconds).
        # estimated_remaining: Estimated time remaining (seconds, if available).
        request_counts: Dictionary with request counts by status.
        created_at: When the batch was created.
        expires_at: When the batch will expire (24 hours from creation).
    """
    batch_id: str
    status: BatchStatus
    elapsed_time: float
    # estimated_remaining: Optional[float] = None
    request_counts: Dict[str, int]
    created_at: str
    expires_at: str


class Client(ABC):
    """Abstract base class for LLM clients

    This class defines the interface for both synchronous and asynchronous
    operations with LLM APIs. Subclasses should implement the abstract methods
    and optionally override batch processing methods if supported.
    """

    def __init__(self, model: str) -> None:
        """Initialize the client with the LLM name.

        Args:
            model (str): The name of the LLM to use.
        """
        self.model = model

    @abstractmethod
    def call(self, prompt: str) -> str:
        """Make a synchronous API Call the LLM with the given prompt.

        Args:
            prompt: The input prompt to send to the LLM.

        Returns:
            The response text from the LLM.
        """

    @abstractmethod
    async def call_async(self, prompt: str) -> str:
        """Make an asynchronous API Call the LLM with the given prompt.

        Args:
            prompt: The input prompt to send to the LLM.

        Returns:
            The response text from the LLM.
        """

    def supports_batch(self) -> bool:
        """Check if the client supports batch processing.

        Returns:
            True if batch processing is supported, False otherwise.
        """
        return False

    async def create_batch_async(self, requests: list[BatchRequest]) -> str:
        """Create a batch processing job asynchronously.

        Args:
            requests: List of batch requests.

        Returns:
            Batch job ID.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError('Async batch processing not supported by this client')

    async def get_batch_status_async(self, batch_id: str) -> BatchStatus:
        """Get the status of a batch job asynchronously.

        Args:
            batch_id: The batch job ID.

        Returns:
            Current batch status.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError('Async batch processing is not supported by this client')

    async def get_batch_info_async(self, batch_id: str) -> Dict[str, Any]:
        """Get detailed information about a batch job asynchronously.

        Args:
            batch_id: The batch job ID.

        Returns:
            Batch information dictionary.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError("Async batch processing not supported by this client")


    async def get_batch_results_async(self, batch_id: str) -> list[BatchResponse]:
        """Get results from a completed batch job asynchronously.

        Args:
            batch_id: The batch job ID.

        Returns:
            List of batch responses.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError('Async batch processing not supported by this client')

    async def cancel_batch_async(self, batch_id: str) -> bool:
        """Cancel a batch job asynchronously.

        Args:
            batch_id: The batch job ID.

        Returns:
            True if batch was cancelled successfully.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError("Async batch processing not supported by this client")

    async def monitor_batch_progress_async(
            self,
            batch_id: str,
            progress_callback: Optional[Callable[[BatchProgress], None]] = None,
            poll_interval: int = 30
    ) -> AsyncIterator[BatchProgress]:
        """Monitor batch progress asynchronously with real-time updates.

        Args:
            batch_id: The batch job ID to monitor.
            progress_callback: Optional callback function for process updates.
            poll_interval: Time between status checks in seconds, default is 30 seconds.

        Yields:
            BatchProgress object with current status and timing information.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError("Async batch monitoring not supported by this client")

    async def wait_for_batch_completion_async(
            self,
            batch_id: str,
            max_wait_time: int = 86400,  # 24 hours default (batch expires after 24h)
            poll_interval: int = 30,
            progress_callback: Optional[Callable[[BatchProgress], None]] = None,
    ) -> BatchStatus:
        """Wait for a batch job to complete asynchronously.

        This method is non-blocking and allows other coroutines to run while
        waiting for batch completion.

        Args:
            batch_id: The batch job ID.
            max_wait_time: Maximum time to wait in seconds (default: 24 hours).
            poll_interval: Time between status checks in seconds, default is 30 seconds.

        Returns:
            Final batch status.

        Raises:
            BatchTimeoutError: If batch doesn't complete within max_wait_time.
        """
        start_time = time.time()

        async for progress in self.monitor_batch_progress_async(
            batch_id, progress_callback, poll_interval
        ):
            # Check for completion
            if progress.status == BatchStatus.ENDED:
                return progress.status
            # if progress.status in [BatchStatus.COMPLETED, BatchStatus.FAILED,
            #                        BatchStatus.EXPIRED, BatchStatus.CANCELLED]:
            #     return progress.status

            # Check for timeout
            if time.time() - start_time > max_wait_time:
                raise BatchTimeoutError(
                    f"Batch job {batch_id} did not complete within {max_wait_time} seconds"
                )

        # Fallback status check
        return await self.get_batch_status_async(batch_id)

    async def process_batch_requests_async(
            self,
            requests: list[BatchRequest],
            max_wait_time: int = 86400,  # 24 hours default
            poll_interval: int = 30,
            progress_callback: Optional[Callable[[BatchProgress], None]] = None,
    ) -> list[BatchResponse]:
        """Process batch requests end-to-end asynchronously.

        This is the main entry point for async batch processing, handling
        creation, monitoring, and result retrieval.

        Args:
            requests: List of batch requests to process.
            max_wait_time: Maximum time to wait for completion in seconds.
            poll_interval: Time between status checks in seconds.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of batch responses.

        Raises:
            LLMClientError: If batch processing fails.
            BatchTimeoutError: If batch doesn't complete in time.
        """
        if not self.supports_batch():
            raise LLMClientError(f'Client {self.__class__.__name__} does not support batch processing')

        try:
            # Create batch job.
            batch_id = await self.create_batch_async(requests)
            logging.info(f'Created batch job {batch_id} with {len(requests)} requests')

            # Wait for completion with progress monitoring
            final_status = await self.wait_for_batch_completion_async(
                batch_id, max_wait_time, poll_interval, progress_callback
            )

            # if final_status == BatchStatus.COMPLETED:
            if final_status == BatchStatus.ENDED:
                # Get and return results
                results = await self.get_batch_results_async(batch_id)
                logging.info(f'Batch job {batch_id} completed successfully with {len(results)} results')
                return results
            else:
                raise LLMClientError(f'Batch job {batch_id} failed with status {final_status.value}')

        except Exception as e:
            if isinstance(e, (LLMClientError, BatchTimeoutError)):
                raise
            raise LLMClientError(f"Batch processing failed: {e}") from e

class ClaudeClient(Client):
    """
    Client for interacting with Anthropic Claude API.

    This client supports both synchronous and asynchronous operations, with
    comprehensive batch processing capabilities using the Claude Batches API.
    """

    def __init__(
        self,
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
            ValueError: If required parameters are missing.
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
            # Synchronous client
            self.client = anthropic.Anthropic(api_key=api_key)
            # Asynchronous client
            self.async_client = anthropic.Anthropic(api_key=api_key)

            self.tokenizer = tiktoken.get_encoding('cl100k_base')
        except Exception as e:
            raise LLMClientError(f'Failed to initialize Claude client: {e}') from e

        logging.info('Claude Client initialized with model=%s', model)

    def supports_batch(self) -> bool:
        """Check if the client supports batch processing.

        Returns:
            True, as Claude supports batch processing.
        """
        return True

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

    def _get_system_message(self) -> str:
        """Get the system message for medieval text processing.

        Returns:
            System message string.
        """
        return (
            'You are an expert for understanding and analyzing medieval texts and manuscripts, '
            'and you can markup all PROPER NOUNS in all kinds of medieval texts'
        )

    def call(self, prompt: str) -> str:
        """Call Claude API synchronously with the given prompt.

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

            messages = [{"role": "user", "content": prompt}]

            response = self.client.messages.create(
                model=self.model,
                system=self._get_system_message(),
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

    async def call_async(self, prompt: str) -> str:
        """Call Claude API asynchronously with the given prompt.

        Args:
            prompt: The input prompt to send to the Claude model.

        Returns:
            The response text from the Claude model.

        Raises:
            ValueError: If the prompt is empty.
        """
        if not prompt or not prompt.strip():
            raise ValueError('Prompt must not be empty for ClaudeClient.')

        try:
            token_count = self._count_tokens(prompt)
            logging.info('Async prompt Token Count: %d ', token_count)

            messages = [{"role": "user", "content": prompt}]

            response = await self.async_client.messages.create(
                model = self.model,
                system = self._get_system_message(),
                messages = messages,
                max_tokens = self.max_tokens,
                temperature = self.temperature,
                top_k = 1,
                top_p = 1.0,
                stream = False
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


    async def create_batch_async(self, requests: list[BatchRequest]) -> str:
        """Create a batch processing job using Claude Message Batches API asynchronously.

        Args:
            requests: List of batch requests to process.

        Returns:
            Batch job ID.

        Raises:
            LLMClientError: If batch creation fails.
        """
        if not requests:
            raise ValueError("Batch requests list cannot be empty")

        try:
            # Prepare batch requests in the format expected by Claude Message Batches API
            batch_requests: List[Request] = []

            for request in requests:
                # Create properly typed message parameters
                message_params = MessageCreateParamsNonStreaming(
                    model = self.model,
                    system = self._get_system_message(),
                    messages = [{
                        "role": "user",
                        "content": request.prompt,
                    }],
                    max_tokens = request.max_tokens,
                    temperature = request.temperature,
                    top_k = 1,
                    top_p = 1.0,
                    stream = False
                )

                # Create properly typed batch request
                batch_request = Request(
                    custom_id = request.custom_id,
                    params = message_params
                )

                batch_requests.append(batch_request)

            # Create the batch job using the correct API endpoint with typed requests
            message_batch = await self.async_client.messages.batches.create(
                requests = batch_requests
            )

            logging.info(f"Created batch job with ID: {message_batch.id}, {len(requests)} requests")
            return message_batch.id

        except Exception as e:
            raise LLMClientError(f"Failed to create batch job: {e}") from e

    async def get_batch_status_async(self, batch_id: str) -> BatchStatus:
        """Get the current status of a batch job asynchronously.

        Args:
            batch_id: The batch job ID.

        Returns:
            Current batch status.

        Raises:
            LLMClientError: If status retrieval fails.
        """
        try:
            message_batch = await self.async_client.messages.batches.retrieve(batch_id)

            # Map Claude batch processing_status to our enum
            if message_batch.processing_status == "in_progress":
                return BatchStatus.IN_PROGRESS
            elif message_batch.processing_status == "ended":
                return BatchStatus.ENDED
            else:
                # Handle any unexpected status
                logging.warning(f"Unexpected batch status: {message_batch.processing_status}")
                return BatchStatus.ENDED

        except Exception as e:
            raise LLMClientError(f"Failed to get batch status: {e}") from e

    async def get_batch_info_async(self, batch_id: str) -> Dict[str, any]:
        """Get detailed information about a batch job asynchronously.

        Args:
            batch_id: The batch job ID.

        Returns:
            Batch information dictionary.

        Raises:
            LLMClientError: If info retrieval fails.
        """
        try:
            message_batch = await self.async_client.messages.batches.retrieve(batch_id)

            # Extract detailed information from the batch object
            batch_info = {
                "id": message_batch.id,
                "type": message_batch.type,
                "processing_status": message_batch.processing_status,
                "request_counts": {
                    "processing": message_batch.request_counts.processing,
                    "succeeded": message_batch.request_counts.succeeded,
                    "errored": message_batch.request_counts.errored,
                    "canceled": message_batch.request_counts.canceled,
                    "expired": message_batch.request_counts.expired
                },
                "created_at": message_batch.created_at,
                "expires_at": message_batch.expires_at,
                "ended_at": message_batch.ended_at,
                "cancel_initiated_at": message_batch.cancel_initiated_at,
                "results_url": message_batch.results_url
            }

            return batch_info

        except Exception as e:
            raise LLMClientError(f"Failed to get batch info: {e}") from e


    async def get_batch_results_async(self, batch_id: str) -> List[BatchResponse]:
        """
        Get results from a completed batch job asynchronously.

        Args:
            batch_id: The batch job ID.

        Returns:
            List of batch responses.

        Raises:
            LLMClientError: If results retrieval fails.
        """
        try:
            # First check if batch is completed
            status = await self.get_batch_status_async(batch_id)
            if status != BatchStatus.ENDED:
                raise LLMClientError(f"Batch {batch_id} is not completed yet, current status: {status.value}")

            # Get batch information to access results_url
            batch_info = await self.get_batch_info_async(batch_id)

            if not batch_info.get("results_url"):
                raise LLMClientError(f"No results URL found for batch {batch_id}")

            results_iter  = self.async_client.messages.batches.results(batch_id)

            # Process results from the async iterator
            results = []
            async for result in results_iter:
                custom_id = result.custom_id

                # Check if the request was successful
                if result.result.status == "succeeded":
                    message = result.result.message
                    if message.content and len(message.content) > 0:
                        response_text = message.content[0].text
                        results.append(BatchResponse(
                            custom_id = custom_id,
                            response_text = response_text,
                            success = True
                        ))
                    else:
                        results.append(BatchResponse(
                            custom_id = custom_id,
                            response_text = "",
                            success = False,
                            error_message = "Empty response content"
                        ))
                else:
                    # Handle error case
                    error_message = getattr(result.result, "error", {}).get("message", "Unknown error")
                    results.append(BatchResponse(
                        custom_id = custom_id,
                        response_text = "",
                        success = False,
                        error_message = error_message
                    ))

            logging.info(f"Retrieved {len(results)} results from batch {batch_id}")
            return results

        except Exception as e:
            raise LLMClientError(f"Failed to get batch results: {e}") from e

    async def cancel_batch_async(self, batch_id: str) -> bool:
        """Cancel a batch job asynchronously.

        Args:
            batch_id: The batch job ID.

        Returns:
            True if batch was cancelled successfully.

        Raises:
            LLMClientError: If cancellation fails.
        """
        try:
            await self.async_client.messages.batches.cancel(batch_id)
            logging.info(f"Batch {batch_id} cancelled successfully")
            return True

        except Exception as e:
            raise LLMClientError(f"Failed to cancel batch: {e}") from e

    async def monitor_batch_progress_async(
            self,
            batch_id: str,
            progress_callback: Optional[Callable[[BatchProgress], None]] = None,
            poll_interval: int = 30
    ) -> AsyncIterator[BatchProgress]:
        """Monitor batch progress asynchronously with real-time updates.

        This method implements a Go-like channel pattern using async generators,
        yielding progress updates as they become available.

        Args:
            batch_id: The batch job ID to monitor.
            progress_callback: Optional callback function for progress updates.
            poll_interval: Time between status checks in seconds.

        Yields:
            BatchProgress objects with current status and timing information.
        """
        start_time = time.time()

        while True:
            try:
                # Get current status and detailed information
                status = await self.get_batch_status_async(batch_id)
                batch_info = await self.get_batch_info_async(batch_id)

                elapsed_time = time.time() - start_time

                # Create progress object with detailed request counts
                progress = BatchProgress(
                    batch_id = batch_id,
                    status = status,
                    elapsed_time = elapsed_time,
                    request_counts = batch_info.get("request_counts", {}),
                    created_at = batch_info.get("created_at", ""),
                    expires_at = batch_info.get("expires_at", ""),
                )

                # Call progress callback if provides
                if progress_callback:
                    try:
                        progress_callback(progress)
                    except Exception as e:
                        logging.warning(f"Error in progress callback: {e}")

                # Yield progress update (similar to sending to a Go channel)
                yield progress

                # Check for terminal state
                if status == BatchStatus.ENDED:
                    logging.info(f"Batch {batch_id} reached terminal state: {status.value}")
                    break

                # Wait before next poll (non-blocking)
                await asyncio.sleep(poll_interval)

            except Exception as e:
                logging.error(f"Error monitoring batch {batch_id}: {e}")
                # Yield error state and break
                yield BatchProgress(
                    batch_id=batch_id,
                    status=BatchStatus.ENDED,
                    elapsed_time=time.time() - start_time,
                    request_counts={},
                    created_at="",
                    expires_at=""
                )
                break

class OllamaClient(Client):
    """Client for interacting with Ollama via OpenWebUI API

    Note: This client does not support batch processing as Ollama
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
            timeout: Request timeout in seconds.
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

        self.endpoint = endpoint.rstrip('/')
        self.token = token
        self.timeout = timeout
        self.temperature = temperature

        logging.info('Ollama Client initialized with model=%s',  model)

    def call(self, prompt: str) -> str:
        """
        Call Ollama API through OpenWebUI with the given prompt.

        Args:
            prompt (str): The input prompt to send to the Ollama model.

        Returns:
            The response text from the Ollama model.

        Raises:
            ValueError: If the prompt is empty.
        """
        if not prompt.strip() or not prompt.strip():
            raise ValueError('Prompt must not be empty for OllamaClient.')

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

            response_text = response.json().get("response", "")

            if not response_text:
                logging.error('Empty response received from Ollama API')
                return 'Ollama API call failed'

            logging.info('Received Ollama response (length: %d)', len(response_text))

            return response_text

        except requests.exceptions.Timeout as e:
            logging.error('Ollama API call timed out after %ds: %s', self.timeout, e, exc_info=True)
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

    async def call_async(self, prompt: str) -> str:
        """
        Call Ollama API asynchronously through OpenWebUI.

        Args:
            prompt: The input prompt to send to the Ollama model.

        Returns:
            The response text from the Ollama model.

        Raises:
            ValueError: If the prompt is empty.
        """
        if not prompt.strip() or not prompt.strip():
            raise ValueError('Prompt must not be empty for OllamaClient.')

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

        try:
            logging.info('Sending async request to Ollama (prompt length: %d)', len(prompt))

            # Send request to OpenWebUI/Ollama
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                ) as response:

                    response.raise_for_status()
                    result = await response.json()

                    if 'response' not in result:
                        logging.error('Invalid response format from Ollama')
                        return 'Invalid response format from Ollama API'

                    response_text = response.json().get("response", "")

                    if not response_text or not response_text.strip():
                        logging.error('Empty response received from Ollama')
                        return 'Empty response received from Ollama'

                    return response_text

        except asyncio.TimeoutError as e:
            logging.error('Ollama API request timed out after %ds: %s', self.timeout, e, exc_info=True)
            return 'Ollama API request timed out'
        except aiohttp.ClientError as e:
            logging.error('Ollama API request failed: %s', e, exc_info=True)
            return  f'Ollama API client error: {e}'
        except Exception as e:
            logging.error('Ollama API call failed: %s', e, exc_info=True)
            return 'Ollama API call failed'


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



