"""Base LLM client abstract class.

Defines the abstract interface for LLM clients, supporting both synchronous
and asynchronous operations. Async batch methods are optional and raise
NotImplementedError by default.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import Any, ClassVar

from .batch_models import BatchProgress, BatchRequest, BatchResponse, BatchStatus
from .exceptions import BatchTimeoutError, LLMClientError

# Type aliases
ProgressCallback = Callable[[BatchProgress], None]


class Client(ABC):
    """Abstract base class for LLM clients.

    This class defines the interface for both synchronous and asynchronous
    operations with LLM APIs. Subclasses implement sync/async single-call
    methods and, optionally, async batch methods.
    """

    # Sensible defaults for polling APIs (seconds)
    # 24 hours timeout default
    DEFAULT_MAX_WAIT_TIME: ClassVar[float] = 86400.0
    # 30 seconds polling default
    DEFAULT_POLL_INTERVAL: ClassVar[float] = 30.0

    def __init__(self, model: str) -> None:
        """Initialize the client with the LLM name.

        Args:
            model: The name of the LLM to use.

        Raises:
            ValueError: If model is empty or None.
        """
        if not model:
            msg = 'Model name cannot be empty or None'
            raise ValueError(msg)

        self.model = model
        logging.info(
            'Initializing LLM client %s with model %s', 
            self.__class__.__name__, self.model,
        )

    # ----------------------------------------------------------------------
    # Properties / helpers
    # ----------------------------------------------------------------------
    @property
    def client_type(self) -> str:
        """Return the type of LLM client (e.g., 'claude', 'ollama')."""
        return self.__class__.__name__.removesuffix('Client').lower()

    @staticmethod
    def supports_async_batch() -> bool:
        """Check if the client supports batch processing.

        Returns:
            True if batch processing is supported, False otherwise.
        """
        return False

    # ----------------------------------------------------------------------
    # Synchronous / asynchronous single-call APIs
    # ----------------------------------------------------------------------
    @abstractmethod
    def call(self, prompt: str) -> str:
        """Performs a synchronous API Call to the LLM.

        Args:
            prompt: The input prompt to send to the LLM.

        Returns:
            The response text from the model.

        Raises:
            ValueError: If prompt is empty.
            LLMClientError: If API call fails.
        """

    @abstractmethod
    async def call_async(self, prompt: str) -> str:
        """Performs an asynchronous API Call to the LLM.

        Args:
            prompt: The input prompt to send to the LLM.

        Returns:
            The response text from the model.

        Raises:
            ValueError: If prompt is empty.
            LLMClientError: If API call fails.
        """

    # ----------------------------------------------------------------------
    # Optional async batch API primitives (override in clients that support it).
    # Concrete subclasses that support async batch should override the methods below.
    # ----------------------------------------------------------------------
    async def create_batch_async(self, requests: list[BatchRequest]) -> str:
        """Create a batch processing job (async).

        Args:
            requests: List of batch requests.

        Returns:
            Batch job Identifier.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support async batch processing',
        )

    async def get_batch_status_async(self, batch_id: str) -> BatchStatus:
        """Get the status of a batch job (async).

        Args:
            batch_id: Batch job identifier.

        Returns:
            Current batch status.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support async batch processing',
        )

    async def get_batch_info_async(self, batch_id: str) -> dict[str, Any]:
        """Get detailed information about a batch job asynchronously.

        Args:
            batch_id: Batch job identifier.

        Returns:
            Batch information dictionary.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support async batch processing',
        )

    async def get_batch_results_async(self, batch_id: str) -> list[BatchResponse]:
        """Get results from a completed batch job asynchronously.

        Args:
            batch_id: The batch job identifier.

        Returns:
            List of batch responses for each request.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support async batch processing',
        )

    async def cancel_batch_async(self, batch_id: str) -> bool:
        """Cancel a batch job asynchronously.

        Args:
            batch_id: The batch job identifier.

        Returns:
            True if batch was canceled successfully.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support async batch processing',
        )

    # Abstract batch orchestration methods - must be implemented by concrete classes
    def monitor_batch_progress_async(
        self,
        batch_num: int,
        batch_id: str,
        poll_interval: float | None = None,
    ) -> AsyncIterator[BatchProgress]:
        # """Monitor batch progress asynchronously with real-time updates.
        """Yields progress updates for a batch job.

        Subclasses that support async batches should implement this as an
        **async generator** (using `async def` + `yield`) that polls status and
        yields `BatchProgress` at intervals.

        Args:
            batch_id: The batch job identifier to monitor.
            batch_num: The batch job number.
            poll_interval: Time between status checks in seconds, default is 30 seconds.

        Yields:
            BatchProgress instances.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} does not support async batch monitoring',
        )

    # ----------------------------------------------------------------------
    # Shared async orchestration helpers (built on top of the primitives)
    # ----------------------------------------------------------------------
    async def wait_for_batch_completion_async(
        self,
        batch_num: int,
        batch_id: str,
        *,
        max_wait_time: float | None = None,
        poll_interval: float | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> BatchStatus:
        """Waits asynchronously for a batch job to complete.

        This method is non-blocking and allows other coroutines to run while
        waiting for batch completion. It uses `monitor_batch_progress_async`
        to receive progress updates. This method owns the `progress_callback`
        invocation to avoid duplicate calls.

        Args:
            batch_id: The batch job identifier.
            max_wait_time: Maximum time to wait in seconds before timing out (default: 24 hours).
            poll_interval: Time between status checks in seconds, default is 30 seconds.
            progress_callback: Optional callback invoked on each progress update.

        Returns:
            Final batch status: BatchStatus.

        Raises:
            BatchTimeoutError: If batch doesn't complete within max_wait_time.
            LLMClientError: If batch processing fails.
        """
        if not self.supports_async_batch():
            msg = f'Client {self.__class__.__name__} does not support async batch processing'
            raise LLMClientError(
                msg,
                client_type=self.client_type,
                operation='batch_waiting',
            )

        # Resolve defaults from class constants
        if max_wait_time is None:
            max_wait_time = self.DEFAULT_MAX_WAIT_TIME
        if poll_interval is None:
            poll_interval = self.DEFAULT_POLL_INTERVAL

        if poll_interval <= 0:
            raise ValueError('poll_interval must be > 0.')
        if max_wait_time <= 0:
            raise ValueError('max_wait_time must be > 0.')

        start_time = time.monotonic()

        async for progress in self.monitor_batch_progress_async(
            batch_num,
            batch_id,
            poll_interval,
        ):
            # Call progress callback if provides
            if progress_callback is not None:
                try:
                    progress_callback(progress)
                except Exception as e:
                    # Catch all exceptions from user-provided callback to prevent
                    # callback errors from breaking batch processing
                    logging.debug(
                        'Progress callback raised: %s',
                        e, exc_info=True,
                    )

            # Check for completion
            if progress.status == BatchStatus.ENDED:
                return progress.status
            # Check for timeout
            if time.monotonic() - start_time > max_wait_time:
                msg = f'Batch job {batch_id} did not complete within {max_wait_time} seconds'
                raise BatchTimeoutError(
                    msg,
                    client_type=self.client_type,
                    operation='batch_waiting',
                    batch_id=batch_id,
                    timeout_seconds=int(max_wait_time),
                )
        # Fallback status check; if monitor exits without ENDED, query once more.
        return await self.get_batch_status_async(batch_id)

    async def process_batch_requests_async(
        self,
        requests: list[BatchRequest],
        batch_num: int,
        *,
        max_wait_time: float | None = None,
        poll_interval: float | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[BatchResponse]:
        """Process batch requests end-to-end (create → wait → fetch results) asynchronously.

        This is the main entry point for async batch processing, handling
        creation, monitoring, and result retrieval.

        Args:
            requests: List of batch requests to process.
            max_wait_time: Maximum time to wait for completion in seconds (default: 24 hours).
            poll_interval: Time between status checks in seconds (default: 30 seconds).
            progress_callback: Optional callback for progress updates.

        Returns:
            The list of 'BatchResponse' items.

        Raises:
            ValueError: If 'requests' is empty.
            LLMClientError: If batch processing fails.
            BatchTimeoutError: If batch doesn't complete in time.
        """
        if not self.supports_async_batch():
            msg = f'Client {self.__class__.__name__} does not support async batch processing'
            raise LLMClientError(
                msg,
                client_type=self.client_type,
                operation='batch_processing',
            )

        if not requests:
            raise ValueError('Request list cannot be empty')

        # Resolve defaults from class constants
        if max_wait_time is None:
            max_wait_time = self.DEFAULT_MAX_WAIT_TIME
        if poll_interval is None:
            poll_interval = self.DEFAULT_POLL_INTERVAL

        try:
            # Create batch job.
            batch_id = await self.create_batch_async(requests)
            logging.info(
                'Created batch %d (ID: %s) with %d requests',
                batch_num, batch_id, len(requests),
            )

            # Wait for completion with progress monitoring
            final_status = await self.wait_for_batch_completion_async(
                batch_num,
                batch_id,
                max_wait_time=max_wait_time,
                poll_interval=poll_interval,
                progress_callback=progress_callback,
            )

            if final_status == BatchStatus.ENDED:
                # Get and return results
                results = await self.get_batch_results_async(batch_id)
                logging.info(
                    'Batch %d (ID: %s) completed successfully with %d results', 
                    batch_num, batch_id, len(results)
                )
                return results
            
            msg = f'Batch {batch_num} (ID: {batch_id}) failed with status {final_status.value}'
            raise LLMClientError(
                msg,
                client_type=self.client_type,
                operation='batch_processing',
            )
        except (LLMClientError, BatchTimeoutError):
            # Preserve domain-specific exceptions.
            raise
        except Exception as e:
            msg = f'Batch processing failed: {e}'
            raise LLMClientError(
                msg,
                client_type=self.client_type,
                operation='batch_processing',
            ) from e
