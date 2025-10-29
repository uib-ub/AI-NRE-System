"""Claude client implementation using Anthropic Claude API."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar, cast

import anthropic
import tiktoken
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from anthropic.types import Message
    from anthropic.types.messages import (
        MessageBatch,
        MessageBatchErroredResult,
        MessageBatchResult,
    )

from .base_client import Client as LLMBaseClient
from .batch_models import BatchProgress, BatchRequest, BatchResponse, BatchStatus
from .exceptions import AuthenticationError, RateLimitError


class ClaudeClient(LLMBaseClient):
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
            msg = "API key must be provided for ClaudeClient."
            raise ValueError(msg)
        if not model:
            msg = "Model must be provided for ClaudeClient."
            raise ValueError(msg)
        # Resolve defaults from class variables.
        if max_tokens is None:
            max_tokens = self.MAX_ALLOWED_TOKENS
        if not (1 <= max_tokens <= self.MAX_ALLOWED_TOKENS):
            msg = f"max_tokens must be between 1 and {self.MAX_ALLOWED_TOKENS}"
            raise ValueError(msg)
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if not (0.0 <= temperature <= 1.0):
            msg = "temperature must be between 0.0 and 1.0"
            raise ValueError(msg)

        # Initialize base class
        super().__init__(model)

        # Set instance attributes
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            # Initialize Synchronous client
            self.client = Anthropic(api_key=api_key)
            # Initialize Asynchronous client
            self.async_client = AsyncAnthropic(api_key=api_key)
            # Initialize tokenizer for token counting
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:  # noqa: BLE001
            # Catch unexpected errors during client/tokenizer initialization
            self._raise_llm_client_error(
                "Failed to initialize Claude client",
                operation="initialization",
                cause=e,
            )
        else:
            logging.info(
                "Claude Client initialized with model=%s, max_tokens=%d, temperature=%.2f",
                model,
                max_tokens,
                temperature,
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
        except Exception:  # noqa: BLE001
            # Non-critical operation: gracefully degrade to 0 on any tokenizer error
            logging.debug("Token counting failed", exc_info=True)
            return 0

    @staticmethod
    def _system_message() -> str:
        """Get the system message for medieval text processing.

        Returns:
            System message string.
        """
        return (
            "You are an expert for understanding and analyzing medieval texts and manuscripts, "
            "and you can markup all PROPER NOUNS in all kinds of medieval texts"
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
            msg = "Prompt must not be empty for ClaudeClient."
            raise ValueError(msg)

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
            "stream": False,
        }

    def _handle_auth_error(
        self,
        exc: Exception,
        *,
        operation: str,
    ) -> AuthenticationError:
        """Handle authentication errors uniformly.

        Args:
            exc: The caught exception.
            operation: The operation being performed when the error occurred.

        Returns:
            An AuthenticationError
        """
        return AuthenticationError(
            f"Claude authentication failed: {exc}",
            client_type=self.client_type,
            operation=operation,
        )

    def _handle_rate_limit_error(
        self,
        exc: Exception,
        *,
        operation: str,
    ) -> RateLimitError:
        """Handle rate limit errors uniformly.

        Args:
            exc: The caught exception.
            operation: The operation being performed when the error occurred.

        Returns:
            A RateLimitError
        """
        return RateLimitError(
            f"Claude API rate limit exceeded: {exc}",
            client_type=self.client_type,
            operation=operation,
            limit_type="requests",
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
            logging.info("Prompt Token Count: %d ", token_count)

            payload = self._message_payload(prompt)
            response: Message = cast("Message", self.client.messages.create(**payload))
            text = self._extract_response_text_from_message(response)
            if not text:
                msg = "Empty response received from Claude API"
                self._raise_api_error(msg, operation="single_call")

        except anthropic.AuthenticationError as e:
            raise self._handle_auth_error(e, operation="single_call") from e
        except anthropic.RateLimitError as e:
            raise self._handle_rate_limit_error(e, operation="single_call") from e
        except anthropic.APIError as e:
            status_code = getattr(e, "status_code", None)
            msg = f"Claude API error: {e}"
            self._raise_api_error(
                msg,
                operation="single_call",
                status_code=status_code,
                cause=e,
            )
        except Exception as e:  # noqa: BLE001
            msg = f"Claude API call failed: {e}"
            self._raise_llm_client_error(msg, operation="single_call", cause=e)
        else:
            return text

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

        token_count = self._count_tokens(prompt)
        logging.info("Async prompt Token Count: %d ", token_count)

        try:
            payload = self._message_payload(prompt)
            response: Message = cast(
                "Message",
                await self.async_client.messages.create(**payload),
            )
            text = self._extract_response_text_from_message(response)
            if not text:
                msg = "Empty response received from Claude API"
                self._raise_api_error(msg, operation="async_single_call")

        except asyncio.CancelledError:
            logging.debug("async_single_call cancelled")
            raise
        except anthropic.AuthenticationError as e:
            raise self._handle_auth_error(e, operation="async_single_call") from e
        except anthropic.RateLimitError as e:
            raise self._handle_rate_limit_error(e, operation="async_single_call") from e
        except anthropic.APIError as e:
            status_code = getattr(e, "status_code", None)
            msg = f"Claude API error: {e}"
            self._raise_api_error(
                msg,
                operation="async_single_call",
                status_code=status_code,
                cause=e,
            )
        except Exception as e:  # noqa: BLE001
            msg = f"Claude API call failed: {e}"
            self._raise_llm_client_error(msg, operation="async_single_call", cause=e)
        else:
            return text

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

        # Prepare batch requests in the format expected by Claude Message Batches API
        batch_requests: list[Request] = []
        for request in requests:
            payload = self._message_payload(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            message_params = MessageCreateParamsNonStreaming(**payload)  # type: ignore[typeddict-item]

            # Create properly typed batch request
            batch_requests.append(
                Request(custom_id=request.custom_id, params=message_params),
            )

        try:
            # Use AsyncAnthropic client for proper async batch creation
            message_batch = await self.async_client.messages.batches.create(
                requests=batch_requests,
            )
        except asyncio.CancelledError:
            logging.debug("async_create_batch cancelled")
            raise
        except anthropic.AuthenticationError as e:
            raise self._handle_auth_error(e, operation="async_create_batch") from e
        except anthropic.RateLimitError as e:
            raise self._handle_rate_limit_error(
                e,
                operation="async_create_batch",
            ) from e
        except anthropic.APIError as e:
            status_code = getattr(e, "status_code", None)
            self._raise_api_error(
                f"Claude API error: {e}",
                operation="async_create_batch",
                status_code=status_code,
                cause=e,
            )
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to create batch job: {e}"
            self._raise_llm_client_error(msg, operation="async_create_batch", cause=e)
        else:
            return message_batch.id

    async def get_batch_status_async(self, batch_id: str) -> BatchStatus:
        """Get the current status of a batch job asynchronously.

        Args:
            batch_id: The batch job ID.

        Returns:
            Current batch status.

        Raises:
            LLMClientError: If status retrieval fails.
        """
        if not batch_id:
            raise ValueError("batch_id cannot be empty")

        try:
            message_batch = await self.async_client.messages.batches.retrieve(batch_id)

        except asyncio.CancelledError:
            logging.debug("async_get_batch_status cancelled")
            raise
        except anthropic.AuthenticationError as e:
            raise self._handle_auth_error(e, operation="async_get_batch_status") from e
        except anthropic.RateLimitError as e:
            raise self._handle_rate_limit_error(
                e,
                operation="async_get_batch_status",
            ) from e
        except anthropic.APIError as e:
            status_code = getattr(e, "status_code", None)
            self._raise_api_error(
                f"Claude API error: {e}",
                operation="async_get_batch_status",
                status_code=status_code,
                cause=e,
            )
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to get batch status: {e}"
            self._raise_llm_client_error(
                msg,
                operation="async_get_batch_status",
                cause=e,
            )
        else:
            # Map Claude batch processing_status to our enum
            ps = getattr(message_batch, "processing_status", None)
            if ps == "in_progress":
                return BatchStatus.IN_PROGRESS
            if ps == "ended":
                return BatchStatus.ENDED
            # Handle any unexpected status
            logging.warning("Unexpected batch status: %s", ps)
            return BatchStatus.ENDED

    async def get_batch_info_async(self, batch_id: str) -> dict[str, Any]:
        """Return detailed async batch information.

        Args:
            batch_id: The batch job ID.

        Returns:
            Batch information dictionary.

        Raises:
            LLMClientError: If info retrieval fails.
        """
        if not batch_id:
            raise ValueError("batch_id cannot be empty")
        try:
            message_batch: MessageBatch = (
                await self.async_client.messages.batches.retrieve(
                    batch_id,
                )
            )
        except asyncio.CancelledError:
            logging.debug("async_get_batch_info cancelled")
            raise
        except anthropic.AuthenticationError as e:
            raise self._handle_auth_error(e, operation="async_get_batch_info") from e
        except anthropic.RateLimitError as e:
            raise self._handle_rate_limit_error(
                e,
                operation="async_get_batch_info",
            ) from e
        except anthropic.APIError as e:
            status_code = getattr(e, "status_code", None)
            self._raise_api_error(
                f"Claude API error: {e}",
                operation="async_get_batch_info",
                status_code=status_code,
                cause=e,
            )
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to get batch info: {e}"
            self._raise_llm_client_error(msg, operation="async_get_batch_info", cause=e)
        else:
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
                    "expired": message_batch.request_counts.expired,
                },
                "created_at": message_batch.created_at,
                "expires_at": message_batch.expires_at,
                "ended_at": message_batch.ended_at,
                "cancel_initiated_at": message_batch.cancel_initiated_at,
                "results_url": message_batch.results_url,
            }
            return batch_info

    async def get_batch_results_async(self, batch_id: str) -> list[BatchResponse]:
        """Fetch results from a completed batch job asynchronously.

        Args:
            batch_id: Identifier of the completed batch job.

        Returns:
            A list of per-request BatchResponse objects.

        Raises:
            LLMClientError: If the batch is not completed or retrieval fails.
        """
        if not batch_id:
            raise ValueError("batch_id cannot be empty")
        try:
            # Validate batch is completed and has results
            await self._validate_batch_ready(batch_id)

            # Process results from the async iterator
            results: list[BatchResponse] = []
            counters = self._create_result_counters()

            # Fetch and process each result
            results_iter = await self.async_client.messages.batches.results(batch_id)
            async for result in results_iter:
                custom_id: str = getattr(result, "custom_id", "unknown_custom_id")
                batch_response = self._process_single_batch_result(
                    result,
                    custom_id,
                    counters,
                )
                results.append(batch_response)

            self._log_batch_summary(batch_id, results, counters)

        except asyncio.CancelledError:
            logging.debug("async_get_batch_results cancelled")
            raise
        except anthropic.AuthenticationError as e:
            raise self._handle_auth_error(e, operation="async_get_batch_results") from e
        except anthropic.RateLimitError as e:
            raise self._handle_rate_limit_error(
                e,
                operation="async_get_batch_results",
            ) from e
        except anthropic.APIError as e:
            status_code = getattr(e, "status_code", None)
            self._raise_api_error(
                f"Claude API error: {e}",
                operation="async_get_batch_results",
                status_code=status_code,
                cause=e,
            )
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to get batch results: {e}"
            self._raise_llm_client_error(
                msg,
                operation="async_get_batch_results",
                cause=e,
            )
        else:
            return results

    async def _validate_batch_ready(self, batch_id: str) -> None:
        """Validate that a batch is completed and ready for result retrieval.

        Args:
            batch_id: The batch job ID to validate.

        Raises:
            LLMClientError: If batch is not ready or missing results URL.
        """
        # Ensure the batch is actually completed
        status = await self.get_batch_status_async(batch_id)
        if status != BatchStatus.ENDED:
            msg = (
                f"Batch {batch_id} is not completed yet, current status: {status.value}"
            )
            self._raise_llm_client_error(msg, operation="async_get_batch_results")

        # Fetch batch information to access results_url
        batch_info = await self.get_batch_info_async(batch_id)
        if not batch_info.get("results_url"):
            msg = f"Batch {batch_id} has no results URL available."
            self._raise_llm_client_error(msg, operation="async_get_batch_results")

    @staticmethod
    def _create_result_counters() -> dict[str, int]:
        """Create a dictionary for tracking result type counts.

        Returns:
            Dictionary with counters initialized to 0.
        """
        return {
            "succeeded": 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
            "parse_errors": 0,
            "other": 0,
        }

    def _process_single_batch_result(
        self,
        result: object,
        custom_id: str,
        counters: dict[str, int],
    ) -> BatchResponse:
        """Process a single batch result item.

        Args:
            result: Raw result item from the batch results iterator.
            custom_id: The custom ID for this result.
            counters: Dictionary to update with result type counts.

        Returns:
            A BatchResponse object for this result.
        """
        try:
            result_obj: MessageBatchResult | None = getattr(
                result,
                "result",
                None,
            )
            if result_obj is None:
                counters["other"] += 1
                return BatchResponse(
                    custom_id=custom_id,
                    response_text="",
                    success=False,
                    error_message="Missing result object.",
                )

            return self._create_batch_response_for_result_type(
                result_obj,
                custom_id,
                counters,
            )

        except Exception as result_exc:
            # Never let one malformed result crash the whole batch
            logging.exception(
                "Failed to parse batch result for custom_id %s",
                custom_id,
            )
            counters["parse_errors"] += 1
            return BatchResponse(
                custom_id=custom_id,
                response_text="",
                success=False,
                error_message=f"Failed to parse result: {result_exc}",
            )

    def _create_batch_response_for_result_type(
        self,
        result_obj: MessageBatchResult,
        custom_id: str,
        counters: dict[str, int],
    ) -> BatchResponse:
        """Create a BatchResponse based on the result type.

        Uses type narrowing with Literal types from MessageBatchResult union.

        Args:
            result_obj: The result object with a discriminated type field.
            custom_id: The custom ID for this result.
            counters: Dictionary to update with result type counts.

        Returns:
            A BatchResponse object appropriate for the result type.
        """
        # Success path: MessageBatchSucceededResult
        if result_obj.type == "succeeded":
            counters["succeeded"] += 1
            response_text = self._extract_response_text_from_message(
                result_obj.message,
            )
            return BatchResponse(
                custom_id=custom_id,
                response_text=response_text,
                success=bool(response_text),
                error_message="" if response_text else "Empty response content",
            )

        # Errored path: MessageBatchErroredResult
        if result_obj.type == "errored":
            counters["errored"] += 1
            error_message = self._extract_error_from_errored_result(result_obj)
            return BatchResponse(
                custom_id=custom_id,
                response_text="",
                success=False,
                error_message=error_message,
            )

        # Canceled path: MessageBatchCanceledResult
        if result_obj.type == "canceled":
            counters["canceled"] += 1
            return BatchResponse(
                custom_id=custom_id,
                response_text="",
                success=False,
                error_message="Request was canceled before execution.",
            )

        # Expired path: MessageBatchExpiredResult
        if result_obj.type == "expired":
            counters["expired"] += 1
            return BatchResponse(
                custom_id=custom_id,
                response_text="",
                success=False,
                error_message="Request expired (not processed within the batch time window).",
            )

        # All known types are exhaustively handled above
        # This is unreachable but kept for runtime safety
        raise AssertionError(f"Unhandled result type: {result_obj.type}")

    @staticmethod
    def _log_batch_summary(
        batch_id: str,
        results: list[BatchResponse],
        counters: dict[str, int],
    ) -> None:
        """Log a summary of batch processing results.

        Args:
            batch_id: The batch job ID.
            results: List of all batch responses.
            counters: Dictionary with result type counts.
        """
        logging.info(
            "Batch (ID: %s) parsed. total=%d, succeeded=%d, errored=%d, "
            "canceled=%d, expired=%d, other=%d, parse_errors=%d",
            batch_id,
            len(results),
            counters["succeeded"],
            counters["errored"],
            counters["canceled"],
            counters["expired"],
            counters["other"],
            counters["parse_errors"],
        )

    # ------------------------------------------------------------------ #
    # Message parsing helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_response_text_from_message(msg: Message) -> str:
        """Extract plain text from an Anthropic Message object.

        Args:
            msg: Anthropic Message object from a successful batch result.

        Returns:
            The extracted text content from all text blocks.
        """
        # Message.content is a list of content blocks
        # Collect text from blocks with type == "text"
        # Each block has a 'type' field (Literal type from SDK)
        # Here we only consume text blocks; other blocks such as tool_use, thinking are ignored.
        # Text blocks have a 'text' attribute containing the actual text.
        # Type narrowing: when block.type == "text", block is TextBlock with a 'text' attribute
        text_parts = [
            block.text for block in msg.content if block.type == "text" and block.text
        ]
        return "".join(text_parts)

    @staticmethod
    def _extract_error_from_errored_result(
        errored_result: MessageBatchErroredResult,
    ) -> str:
        """Extract error message from MessageBatchErroredResult.

        Args:
            errored_result: The errored result from batch processing.

        Returns:
            A human-readable error message string.
        """
        # MessageBatchErroredResult has an 'error' attribute of type ErrorResponse
        error = errored_result.error

        # Try to extract message from the error response
        # ErrorResponse has an 'error' attribute of type ErrorObject, which is an Union type,
        # so use defensive access
        if hasattr(error, "error") and hasattr(error.error, "message"):
            return str(error.error.message)
        # Fallback to string representation
        return f"Batch request failed: {error}"

    async def cancel_batch_async(self, batch_id: str) -> bool:
        """Cancel a batch job asynchronously.

        Args:
            batch_id: The batch job ID.

        Returns:
            True if batch was canceled successfully.

        Raises:
            LLMClientError: If cancellation fails.
        """
        if not batch_id:
            raise ValueError("batch_id cannot be empty")

        try:
            await self.async_client.messages.batches.cancel(batch_id)

        except asyncio.CancelledError:
            logging.debug("async_cancel_batch cancelled")
            raise
        except anthropic.AuthenticationError as e:
            raise self._handle_auth_error(e, operation="async_cancel_batch") from e
        except anthropic.RateLimitError as e:
            raise self._handle_rate_limit_error(
                e,
                operation="async_cancel_batch",
            ) from e
        except anthropic.APIError as e:
            status_code = getattr(e, "status_code", None)
            self._raise_api_error(
                f"Claude API error: {e}",
                operation="async_cancel_batch",
                status_code=status_code,
                cause=e,
            )
        except Exception as e:  # noqa: BLE001
            msg = f"Failed to cancel batch: {e}"
            self._raise_llm_client_error(msg, operation="async_cancel_batch", cause=e)
        else:
            logging.info("Batch %s cancelled successfully", batch_id)
            return True

    # ------------------------------------------------------------------ #
    # Batch monitoring (async iterator)
    # Claude-specific batch monitoring implementation
    # ------------------------------------------------------------------ #
    async def monitor_batch_progress_async(
        self,
        batch_num: int,
        batch_id: str,
        poll_interval: float | None = None,
    ) -> AsyncIterator[BatchProgress]:
        """Yield progress updates for a batch job.

        This async generator polls the Anthropic batches API at a fixed interval and
        yields 'BatchProgress' objects until a terminal status is reached. The
        orchestration method in the base class is responsible for invoking any
        optional progress callback supplied by the caller.

        Args:
            batch_num: The batch number for tracking multiple batches.
            batch_id: The batch job ID to monitor.
            poll_interval: Time between status checks in seconds (default: 30 seconds).

        Yields:
            BatchProgress instances with current status and timing information.

        Raises:
            ValueError: If poll_interval is not positive or batch_id is empty.
        """
        if not batch_id:
            raise ValueError("batch_id cannot be empty")

        if poll_interval is None:
            poll_interval = self.DEFAULT_POLL_INTERVAL  # base class constant

        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0.")

        start_time = time.monotonic()

        while True:
            try:
                # Get current status and detailed information
                status = await self.get_batch_status_async(batch_id)
                batch_info: dict[str, Any] = await self.get_batch_info_async(batch_id)

                elapsed_time = time.monotonic() - start_time
                # Defensive extraction/typing
                req_counts: dict[str, int] = batch_info.get("request_counts") or {}

                created_at = str(batch_info.get("created_at", ""))
                expires_at = str(batch_info.get("expires_at", ""))

                # Create and yield progress to the caller
                yield BatchProgress(
                    batch_num=batch_num,
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
                        "Batch %s reached terminal state: %s",
                        batch_id,
                        status.value,
                    )
                    return

                # Wait before next poll (non-blocking)
                await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                logging.info("Batch monitor cancelled for %s", batch_id)
                raise
            except Exception:
                logging.exception(
                    "Error monitoring batch %s",
                    batch_id,
                )
                # Emit a final ended state so the caller can unwind cleanly.
                yield BatchProgress(
                    batch_num=batch_num,
                    batch_id=batch_id,
                    status=BatchStatus.ENDED,
                    elapsed_time=time.monotonic() - start_time,
                    request_counts={},
                    created_at="",
                    expires_at="",
                )
                return
