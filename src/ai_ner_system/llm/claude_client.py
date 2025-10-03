"""Claude client implementation using Anthropic Claude API."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import anthropic
import tiktoken
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from .base_client import Client
from .batch_models import BatchRequest, BatchResponse, BatchProgress, BatchStatus
from .exceptions import APIError, AuthenticationError, LLMClientError, RateLimitError


class ClaudeClient(Client):
    """Client for interacting with Claude using Anthropic Claude API.

    This client supports both synchronous and asynchronous operations, with
    comprehensive batch processing capabilities using the Claude Batches API.
    """

    # Reasonable bounds for the constructor validation
    MAX_ALLOWED_TOKENS: ClassVar[int] = 20000
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.0

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        """Initialize a Claude client.

        Args:
            api_key: Anthropic API key.
            model: Claude model to use.
            max_tokens: Maximum tokens in response (defaults to MAX_ALLOWED_TOKENS)
            temperature: Response randomness (0.0-1.0, defaults to DEFAULT_TEMPERATURE)

        Raises:
            ValueError: If required parameters are missing or invalid.
            LLMClientError: If client initialization fails.
        """
        if not api_key:
            raise ValueError('API key must be provided for ClaudeClient.')
        if not model:
            raise ValueError('Model must be provided for ClaudeClient.')
        # Resolve defaults from class variables.
        if max_tokens is None:
            max_tokens = self.MAX_ALLOWED_TOKENS
        if not (1 <= max_tokens <= self.MAX_ALLOWED_TOKENS):
            raise ValueError(
                f'max_tokens must be between 1 and {self.MAX_ALLOWED_TOKENS}'
            )
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if not (0.0 <= temperature <= 1.0):
            raise ValueError("temperature must be between 0.0 and 1.0")

        # Initialize base class
        super().__init__(model)

        # Set instance attributes
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            # Initialize Synchronous client
            self.client = anthropic.Anthropic(api_key=api_key)
            # Initialize Asynchronous client
            self.async_client = anthropic.AsyncAnthropic(api_key=api_key)
            # Initialize tokenizer for token counting
            self.tokenizer = tiktoken.get_encoding('cl100k_base')
        except Exception as e:
            raise LLMClientError(
                f'Failed to initialize Claude client: {e}',
                client_type=self.client_type,
                operation='initialization'
            ) from e
        logging.info(
            'Claude Client initialized with model=%s, max_tokens=%d, temperature=%.2f',
            model,
            max_tokens,
            temperature
        )

    @staticmethod
    def supports_async_batch() -> bool:
        """Check if the client supports batch processing.

        Returns:
            True, as Claude supports batch processing.
        """
        return True

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens in the text.
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logging.debug('Token counting failed: %s', e, exc_info=True)
            return 0

    @staticmethod
    def _system_message() -> str:
        """Get the system message for medieval text processing.

        Returns:
            System message string.
        """
        return (
            'You are an expert for understanding and analyzing medieval texts and manuscripts, '
            'and you can markup all PROPER NOUNS in all kinds of medieval texts'
        )

    @staticmethod
    def _validate_prompt(prompt: str) -> None:
        """Validate prompt input.

        Args:
            prompt: Input prompt to validate.

        Raises:
            ValueError: If prompt is empty or invalid.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty for ClaudeClient.")

    def _message_payload(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Build a non-streaming Messages.create params object.

        Args:
            prompt: The input prompt to send to the Claude model.
            max_tokens: Optional override for the maximum response tokens.
            temperature: Optional override for the sampling temperature.

        Returns:
          A JSON-serializable dictionary matching the Messages API schema.
        """
        return {
            "model": self.model,
            "system": self._system_message(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_k": 1,
            "top_p": 1.0,
            "stream": False
        }

    def _handle_auth_error(self, exc: Exception, *, operation: str) -> AuthenticationError:
        """Handle authentication errors uniformly.

        Args:
            exc: The caught exception.
            operation: The operation being performed when the error occurred.

        Returns:
            An AuthenticationError
        """
        return AuthenticationError(
            f'Claude authentication failed: {exc}',
            client_type=self.client_type,
            operation=operation
        )

    def _handle_rate_limit_error(self, exc: Exception, *, operation: str) -> RateLimitError:
        """Handle rate limit errors uniformly.

        Args:
            exc: The caught exception.
            operation: The operation being performed when the error occurred.

        Returns:
            A RateLimitError
        """
        return RateLimitError(
            f'Claude API rate limit exceeded: {exc}',
            client_type=self.client_type,
            operation=operation,
            limit_type="requests",
        )

    def _handle_api_error(self, exc: Exception, *, operation: str, status_code: int | None) -> APIError:
        """Handle generic API errors uniformly.

        Args:
            exc: The caught exception.
            operation: The operation being performed when the error occurred.

        Returns:
            An APIError
        """
        return APIError(
            f'Claude API error: {exc}',
            client_type=self.client_type,
            operation=operation,
            status_code=status_code,
        )

    # ------------------------------------------------------------------ #
    # Sync single call
    # ------------------------------------------------------------------ #
    def call(self, prompt: str) -> str:
        """Call Claude API synchronously with the given prompt.

        Args:
            prompt: The input prompt to send to the Claude model.

        Returns:
            The response text from the Claude model.

        Raises:
            ValueError: If the prompt is empty.
            APIError: If API call fails.
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limit is exceeded.
            LLMClientError: If processing fails.
        """
        self._validate_prompt(prompt)

        try:
            token_count = self._count_tokens(prompt)
            logging.info('Prompt Token Count: %d ', token_count)

            payload = self._message_payload(prompt)
            response = self.client.messages.create(**payload)

            # messages = [{"role": "user", "content": prompt}]

            # response = self.client.messages.create(
            #     model=self.model,
            #     system=self._system_message(),
            #     messages=messages,
            #     max_tokens=self.max_tokens,
            #     temperature=self.temperature,
            #     top_k=1,
            #     top_p=1.0,
            #     stream=False,
            # )

            # if not response.content or not response.content[0].text:
            #     # logging.error('Empty response received from Claude API')
            #     # return 'Claude API call failed'
            #     raise APIError(
            #         'Empty response received from Claude API',
            #         client_type=self.client_type,
            #         operation='single_call'
            #     )

            # return response.content[0].text

            text = self._extract_response_text_from_message(response)
            if not text:
                raise APIError(
                    'Empty response received from Claude API',
                    client_type=self.client_type,
                    operation='single_call'
                )
            return text

        except anthropic.AuthenticationError as e:
            raise self._handle_auth_error(e, operation='single_call') from e
        except anthropic.RateLimitError as e:
            raise self._handle_rate_limit_error(e, operation='single_call') from e
        except anthropic.APIError as e:
            sc = getattr(e, "status_code", None)
            raise self._handle_api_error(e, operation='single_call', status_code=sc) from e
        except Exception as e:
            raise LLMClientError(
                f'Claude API call failed: {e}',
                client_type=self.client_type,
                operation='single_call'
            ) from e

    # ------------------------------------------------------------------ #
    # Async single call
    # ------------------------------------------------------------------ #
    async def call_async(self, prompt: str) -> str:
        """Call Claude API asynchronously with the given prompt.

        Args:
            prompt: The input prompt to send to the Claude model.

        Returns:
            The response text from the Claude model.

        Raises:
            ValueError: If the prompt is empty.
            APIError: If API call fails.
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limit is exceeded.
            LLMClientError: If processing fails.
        """
        self._validate_prompt(prompt)

        try:
            token_count = self._count_tokens(prompt)
            logging.info('Async prompt Token Count: %d ', token_count)

            payload = self._message_payload(prompt)
            response = await self.async_client.messages.create(**payload)

            # messages = [{"role": "user", "content": prompt}]

            # response = await self.async_client.messages.create(
            #     model=self.model,
            #     system=self._system_message(),
            #     messages=messages,
            #     max_tokens=self.max_tokens,
            #     temperature=self.temperature,
            #     top_k=1,
            #     top_p=1.0,
            #     stream=False
            # )

            # if not response.content or not response.content[0].text:
            #     raise APIError(
            #         'Empty response received from Claude API',
            #         client_type=self.client_type,
            #         operation='async_single_call'
            #     )

            # return response.content[0].text

            text = self._extract_response_text_from_message(response)
            if not text:
                raise APIError(
                    'Empty response received from Claude API',
                    client_type=self.client_type,
                    operation='async_single_call'
                )
            return text

        except anthropic.AuthenticationError as e:
            raise self._handle_auth_error(e, operation='async_single_call') from e
        except anthropic.RateLimitError as e:
            raise self._handle_rate_limit_error(e, operation='async_single_call') from e
        except anthropic.APIError as e:
            status_code = getattr(e, "status_code", None)
            raise self._handle_api_error(
                e, operation='async_single_call', status_code=status_code
            ) from e
        except Exception as e:
            raise LLMClientError(
                f'Claude API call failed: {e}',
                client_type=self.client_type,
                operation='async_single_call'
            ) from e

    # ------------------------------------------------------------------ #
    # Async batch APIs
    # ------------------------------------------------------------------ #
    async def create_batch_async(self, requests: list[BatchRequest]) -> str:
        """Create a batch processing job using Claude Message Batches API, asynchronously.

        Args:
            requests: List of batch requests to process.

        Returns:
            Batch job ID.

        Raises:
            ValueError: If requests list is empty.
            LLMClientError: If batch creation fails.
        """
        if not requests:
            raise ValueError("Batch requests list cannot be empty")

        try:
            # Prepare batch requests in the format expected by Claude Message Batches API
            batch_requests: list[Request] = []
            for request in requests:
                payload = self._message_payload(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
                message_params = MessageCreateParamsNonStreaming(**payload)

                # message_params = MessageCreateParamsNonStreaming(
                #     model=self.model,
                #     system=self._system_message(),
                #     messages=[{
                #         "role": "user",
                #         "content": request.prompt,
                #     }],
                #     max_tokens=request.max_tokens,
                #     temperature=request.temperature,
                #     top_k=1,
                #     top_p=1.0,
                #     stream=False
                # )

                # Create properly typed batch request
                batch_requests.append(
                    Request(custom_id=request.custom_id, params=message_params)
                )
            # Use AsyncAnthropic client for proper async batch creation
            message_batch = await self.async_client.messages.batches.create(
                requests=batch_requests
            )
            logging.info(
                f'Created batch job with ID: {message_batch.id}, {len(requests)} requests'
            )
            return message_batch.id

        except Exception as e:
            raise LLMClientError(
                f'Failed to create batch job: {e}',
                client_type=self.client_type,
                operation='async_create_batch'
            ) from e

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
            raise LLMClientError(
                f'Failed to get batch status: {e}',
                client_type=self.client_type,
                operation='async_get_batch_status'
            ) from e

    async def get_batch_info_async(self, batch_id: str) -> dict[str, Any]:
        """Return detailed async batch information

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
            batch_info: dict[str, Any] = {
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
            raise LLMClientError(
                f'Failed to get batch info: {e}',
                client_type=self.client_type,
                operation='async_get_batch_info'
            ) from e

    async def get_batch_results_async(self, batch_id: str) -> list[BatchResponse]:
        """Fetch results from a completed batch job asynchronously.

        Args:
            batch_id: Identifier of the completed batch job.

        Returns:
            A list of per-request BatchResponse objects.

        Raises:
            LLMClientError: If the batch is not completed or retrieval fails.
        """
        try:
            # Ensure the batch is actually completed
            status = await self.get_batch_status_async(batch_id)
            if status != BatchStatus.ENDED:
                raise LLMClientError(
                    f'Batch {batch_id} is not completed yet, current status: {status.value}',
                    client_type=self.client_type,
                    operation='async_get_batch_results'
                )

            # Fatch batch information to access results_url
            batch_info = await self.get_batch_info_async(batch_id)
            if not batch_info.get("results_url"):
                raise LLMClientError(
                    f'No results URL found for batch {batch_id}',
                    client_type=self.client_type,
                    operation='async_get_batch_results'
                )

            # Process results from the async iterator
            results: list[BatchResponse] = []
            # counters are for logging and debug purposes
            counters = {
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
                "parse_errors": 0,
                "other": 0,
            }

            # Fetch the async iterator of results
            results_iter  = await self.async_client.messages.batches.results(batch_id)
            async for result in results_iter:
                custom_id = getattr(result, 'custom_id', 'unknown_custom_id')
                try:
                    result_obj = getattr(result, "result", None)
                    if result_obj is None:
                        results.append(
                            BatchResponse(
                                custom_id=custom_id,
                                response_text='',
                                success=False,
                                error_message='Missing result object.'
                            )
                        )
                        counters["other"] += 1
                        continue

                    result_type = getattr(result_obj, "type", None)

                    # Success path
                    if result_type == "succeeded":
                        message = getattr(result_obj, "message", None)
                        response_text = self._extract_response_text_from_message(message)
                        results.append(
                            BatchResponse(
                                custom_id=custom_id,
                                response_text=response_text,
                                success=bool(response_text),
                                error_message='' if response_text else 'Empty response content',
                            )
                        )
                        counters["succeeded"] += 1
                        # continue

                    # errored, canceled, or expired path
                    elif result_type in {"errored", "canceled", "expired"}:
                        error_message = self._extract_error_message(result_obj, result_type)
                        results.append(
                            BatchResponse(
                                custom_id=custom_id,
                                response_text='',
                                success=False,
                                error_message=error_message,
                            )
                        )
                        counters[str(result_type)] += 1

                    # Unknown/undocumented type: treat as failure but attempt to extract.
                    else:
                        error_message = self._extract_error_message(result_obj, result_type)
                        results.append(
                            BatchResponse(
                                custom_id=custom_id,
                                response_text='',
                                success=False,
                                error_message=error_message,
                            )
                        )
                        counters["other"] += 1

                except Exception as result_exc:
                    # Never let one malformed result crash the whole batch
                    logging.error(
                        f'Failed to parse batch result for custom_id {custom_id}: {result_exc}',
                        exc_info=True
                    )
                    results.append(
                        BatchResponse(
                            custom_id=custom_id,
                            response_text='',
                            success=False,
                            error_message=f'Failed to parse result: {result_exc}',
                        )
                    )
                    counters["parse_errors"] += 1

            logging.info(
                'Batch %s parsed. total=%d, succeeded=%d, errored=%d, '
                'canceled=%d, expired=%d, other=%d, parse_errors=%d',
                batch_id,
                len(results),
                counters["succeeded"],
                counters["errored"],
                counters["canceled"],
                counters["expired"],
                counters["other"],
                counters["parse_errors"],
            )
            return results

        except Exception as e:
            raise LLMClientError(
                f'Failed to get batch results: {e}',
                client_type=self.client_type,
                operation='async_get_batch_results',
            ) from e

    # ------------------------------------------------------------------ #
    # Message parsing helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_response_text_from_message(msg: Any) -> str:
        """Extract plain text from an Anthropic message object.
        
        Args:
            msg: Anthropic message object from a successful batch result.
            response_text = message.content[0].text

        Returns:
            The extracted text content, or empty string if not found.
        """
        if msg is None:
            return ""

        # Get content
        content = ClaudeClient._get_field(msg, "content")

        # Handle string content if it is already a string, just return it.
        if isinstance(content, str):
            return content

        # If content is a list of blocks, collect text from blocks with type == "text".
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                # Resolve type block
                block_type = ClaudeClient._get_field(block, "type")
                # Only consume text blocks; ignore tool/thinking/etc per docs
                if block_type == "text":
                    text = ClaudeClient._get_field(block, "text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
            if text_parts:
                return "".join(text_parts)

        # Fallback to direct text attribute
        text = getattr(msg, "text", None)
        if isinstance(text, str):
            return text

        return ""

    @staticmethod
    def _get_field(obj: Any, field_name: str) -> Any:
        """Helper to safely retrieve a field from an object or dict.

        Tries attribute access first (SDK/Pydantic objects),
        then dict key lookup.

        Args:
            obj: The object or dict to extract from.
            field_name: The field name to get.

        Returns:
            The field value, or None if not found.
        """
        val = getattr(obj, field_name, None)
        if val is None and isinstance(obj, dict):
            val = obj.get(field_name)
        return val

    @staticmethod
    def _extract_error_message(result_obj: Any, result_type: str | None) -> str:
        """Extract an error message for errored/canceled/expired/other cases.

        Args:
            result_obj: The result object from a batch-response result.
            result_type: The type of the result (e.g., "errored", "canceled", "expired".)

        Returns:
            A human-readable error message string.
        """
        if result_type == "canceled":
            # NOTE: Claude API does not provide an output result for cancellation, and
            # canceled requests “will not be billed”, since they never executed
            return 'Request was canceled before execution.'

        if result_type == "expired":
            # NOTE: Claude API does not provide an output result for expiration (request timed out), and
            # expired requests “will not be billed”, since the request expired before it could be processed.
            return 'Request expired (not processed within the batch time window).'

        # Generic/errored: inspect error objects.
        error_obj = getattr(result_obj, "error", None)

        # Dict-like error, use .get
        if isinstance(error_obj, dict):
            msg = error_obj.get("message")
            if msg:
                return str(msg)
            err = error_obj.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
                if msg:
                    return str(msg)
            return str(error_obj)

        # SDK/Pydantic path: use attribute access
        msg =  getattr(error_obj, "message", None)
        if msg:
            return str(msg)
        err = getattr(error_obj, "error", None)
        if isinstance(err, dict):
            msg = err.get("message")
            if msg:
                return str(msg)
        else:
            msg = getattr(err, "message", None)
            if msg:
                return str(msg)

        # Direct message at result level
        msg = getattr(result_obj, "message", None)
        if msg:
            return str(msg)

        return f'Request failed with unknown error type: {result_type}'

    async def cancel_batch_async(self, batch_id: str) -> bool:
        """Cancel a batch job asynchronously.

        Args:
            batch_id: The batch job ID.

        Returns:
            True if batch was canceled successfully.

        Raises:
            LLMClientError: If cancellation fails.
        """
        try:
            await self.async_client.messages.batches.cancel(batch_id)
            logging.info(f'Batch {batch_id} cancelled successfully')
            return True
        except Exception as e:
            raise LLMClientError(
                f'Failed to cancel batch: {e}',
                client_type=self.client_type,
                operation='async_cancel_batch'
            ) from e

    # ------------------------------------------------------------------ #
    # Batch monitoring (async iterator)
    # Claude-specific batch monitoring implementation
    # ------------------------------------------------------------------ #
    async def monitor_batch_progress_async(
        self,
        batch_id: str,
        poll_interval: float = Client.DEFAULT_POLL_INTERVAL
    ) -> AsyncIterator[BatchProgress]:
        """Yield progress updates for a batch job.

        This async generator polls the Anthropic batches API at a fixed interval and
        yields 'BatchProgress' objects until a terminal status is reached. The
        orchestration method in the base class is responsible for invoking any
        optional progress callback supplied by the caller.

        Args:
            batch_id: The batch job ID to monitor.
            poll_interval: Time between status checks in seconds.

        Yields:
            BatchProgress instances with current status and timing information.

        Raises:
            ValueError: If poll_interval is not positive.
        """
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0.")

        start_time = time.monotonic()

        while True:
            try:
                # Get current status and detailed information
                status = await self.get_batch_status_async(batch_id)
                batch_info = await self.get_batch_info_async(batch_id)

                elapsed_time = time.monotonic() - start_time
                # Defensive extraction/typing
                req_counts = batch_info.get('request_counts') or {}
                if not isinstance(req_counts, dict):
                    req_counts = {}

                created_at = str(batch_info.get('created_at', ''))
                expires_at = str(batch_info.get('expires_at', ''))

                # Create and yield progress to the caller
                yield BatchProgress(
                    batch_id=batch_id,
                    status=status,
                    elapsed_time=elapsed_time,
                    request_counts=req_counts,
                    created_at=created_at,
                    expires_at=expires_at,
                )

                # Check for terminal state
                if status == BatchStatus.ENDED:
                    logging.info(
                        f'Batch {batch_id} reached terminal state: {status.value}'
                    )
                    # break
                    return

                # Wait before next poll (non-blocking)
                await asyncio.sleep(poll_interval)

            except Exception as e:
                logging.error(f"Error monitoring batch {batch_id}: {e}", exc_info=True)
                # Emit a final ended state so the caller can unwind cleanly.
                yield BatchProgress(
                    batch_id=batch_id,
                    status=BatchStatus.ENDED,
                    elapsed_time=time.monotonic() - start_time,
                    request_counts={},
                    created_at="",
                    expires_at=""
                )
                # break
                return