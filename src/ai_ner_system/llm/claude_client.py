"""Claude client implementation using Anthropic's API."""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Callable, AsyncIterator, Any

import tiktoken
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from .base_client import Client
from .batch_models import BatchRequest, BatchStatus, BatchResponse, BatchProgress
from .exceptions import LLMClientError


class ClaudeClient(Client):
    """Client for interacting with Claude using Anthropic Claude API.

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
            raise LLMClientError(f'Failed to initialize Claude client: {e}', client_type='claude') from e

        logging.info('Claude Client initialized with model=%s', model)

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
            logging.warning('Token counting failed: %s', e)
            return 0

    @staticmethod
    def _get_system_message() -> str:
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

    def call(self, prompt: str) -> str:
        """Call Claude API synchronously with the given prompt.

        Args:
            prompt (str): The input prompt to send to the Claude model.

        Returns:
            str: The response text from the Claude model.

        Raises:
            ValueError: If the prompt is empty.
        """

        self._validate_prompt(prompt)

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
        self._validate_prompt(prompt)

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

            # Use AsyncAnthropic client for proper async batch creation
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
        """Fetch results from a completed batch job asynchronously.

        Args:
            batch_id: The batch job ID.

        Returns:
            List of batch responses.

        Raises:
            LLMClientError: If results retrieval fails.
        """
        try:
            # Ensure the batch is actually completed
            status = await self.get_batch_status_async(batch_id)
            if status != BatchStatus.ENDED:
                raise LLMClientError(
                    f'Batch {batch_id} is not completed yet, current status: {status.value}'
                )

            # Fatch batch information to access results_url
            batch_info = await self.get_batch_info_async(batch_id)
            if not batch_info.get("results_url"):
                raise LLMClientError(f'No results URL found for batch {batch_id}')

            # Fetch the async iterator of results
            results_iter  = await self.async_client.messages.batches.results(batch_id)

            # Process results from the async iterator
            results: List[BatchResponse] = []
            # counters are for logging and debug purposes
            counters = {
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
                "parse_errors": 0,
                "other": 0,
            }

            async for result in results_iter:
                custom_id = getattr(result, "custom_id", "unknown_custom_id")
                try:
                    result_obj = getattr(result, "result", None)
                    if result_obj is None:
                        results.append(
                            BatchResponse(
                                custom_id = custom_id,
                                response_text = "",
                                success = False,
                                error_message = "Missing result object."
                            )
                        )
                        counters["other"] += 1
                        continue

                    result_type = getattr(result_obj, "type", None)

                    # Success path
                    if result_type == "succeeded":
                        message = getattr(result_obj, "message", None)
                        response_text = self._extract_response_text_from_message(message)
                        if response_text:
                            results.append(
                                BatchResponse(
                                    custom_id=custom_id,
                                    response_text=response_text,
                                    success=True,
                                )
                            )
                        else:
                            results.append(
                                BatchResponse(
                                    custom_id=custom_id,
                                    response_text="",
                                    success=False,
                                    error_message="Empty response content"
                                )
                            )
                        counters["succeeded"] += 1
                        continue

                    # errored, canceled, or expired path
                    if result_type in {"errored", "canceled", "expired"}:
                        error_message = self._extract_error_message(result_obj, result_type)
                        results.append(
                            BatchResponse(
                                custom_id = custom_id,
                                response_text = "",
                                success = False,
                                error_message = error_message,
                            )
                        )
                        counters[str(result_type)] += 1
                        continue

                    # Unknown/undocumented type: treat as failure but attempt to extract.
                    error_message = self._extract_error_message(result_obj, result_type)
                    results.append(
                        BatchResponse(
                            custom_id = custom_id,
                            response_text = "",
                            success = False,
                            error_message = error_message,
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
                            custom_id = custom_id,
                            response_text = "",
                            success = False,
                            error_message = f'Failed to parse result: {result_exc}',
                        )
                    )

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
            raise LLMClientError(f'Failed to get batch results: {e}') from e

    @staticmethod
    def _extract_response_text_from_message(msg: Any) -> str:
        """Extract plain text from an Anthropic message object.
        
        Args:
            msg: The message object from a successful batch result.
            response_text = message.content[0].text

        Returns:
            The extracted text content, or empty string if not found.
        """
        if msg is None:
            return ""

        # Get content
        content = ClaudeClient._get_field(msg, "content")

        # Just in case if content is already a string, just return it.
        if isinstance(content, str):
            return content

        # If content is a list of blocks, collect text from blocks with type == "text".
        if isinstance(content, list):
            text_parts: List[str] = []
            for block in content:
                # Resolve block type for both SDK objects and dicts
                block_type = ClaudeClient._get_field(block, "type")

                # Only consume text blocks; ignore tool/thinking/etc per docs
                if block_type == "text":
                    text = ClaudeClient._get_field(block, "text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
            if text_parts:
                # Newline-join to keep paragraph boundaries without overfusing content
                return "".join(text_parts)

        # Fallback: some SDKs expose a top-level .text
        text = getattr(msg, "text", None)
        if isinstance(text, str):
            return text

        return ""


    @staticmethod
    def _get_field(obj: Any, field_name: str) -> Any:
        """Helper to safely retrieve a field from an object or dict.

        Tries attribute access first (SDK/Pydantic objects), then dict key lookup.

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
    def _extract_error_message(result_obj: Any, result_type: Optional[str]) -> str:
        """Extract an error message for errored/canceled/expired/other cases.

        Args:
            result_obj: The result object from a batch-response result.
            result_type: The type of the result (e.g., "errored", "canceled", "expired".)

        Returns:
            A human-readable error message string.
        """
        if result_type == "canceled":
            # NOTE: Claude API does not provide a output result for cancellation, and
            # canceled requests “will not be billed”, since they never executed
            return 'Request was canceled before execution.'

        if result_type == "expired":
            # NOTE: Claude API does not provide a output result for expiration (request timed out), and
            # expired requests “will not be billed”, since the request expired before it could be processed.
            return 'Request expired (not processed within the batch time window).'

        # Generic/errored: inspect error objects.
        error_obj = getattr(result_obj, "error", None)

        # For dict-like error, use .get
        if isinstance(error_obj, dict):
            msg = error_obj.get("message")
            if msg:
                return msg
            err = error_obj.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
                if msg:
                    return msg
            return str(error_obj)

        # SDK/Pydantic path: use attribute access
        msg =  getattr(error_obj, "message", None)
        if msg:
            return msg
        err = getattr(error_obj, "error", None)
        if isinstance(err, dict):
            msg = err.get("message")
            if msg:
                return msg
        else:
            msg = getattr(err, "message", None)
            if msg:
                return msg

        # Some SDK shapes place message directly at the result level.
        msg = getattr(result_obj, "message", None)
        if msg:
            return msg

        return f'Request failed with unknown error type: {result_type}'


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
            logging.info(f'Batch {batch_id} cancelled successfully')
            return True
        except Exception as e:
            raise LLMClientError(f'Failed to cancel batch: {e}') from e

    # Claude-specific batch monitoring implementation
    async def monitor_batch_progress_async(
            self,
            batch_id: str,
            progress_callback: Optional[Callable[[BatchProgress], None]] = None,
            poll_interval: int = 30
    ) -> AsyncIterator[BatchProgress]:
        """Monitor batch progress asynchronously with real-time updates.

        This method implements Claude-specific progress monitoring using the
        Claude Message Batches API data format.

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
                    batch_id=batch_id,
                    status=status,
                    elapsed_time=elapsed_time,
                    request_counts=batch_info.get("request_counts", {}),
                    created_at=batch_info.get("created_at", ""),
                    expires_at=batch_info.get("expires_at", ""),
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
                logging.error(f"Error monitoring batch {batch_id}: {e}", exc_info=True)
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
