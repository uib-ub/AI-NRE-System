"""Base LLM client abstract class."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, AsyncIterator, Dict, Any

from .batch_models import BatchStatus, BatchRequest, BatchResponse, BatchProgress
from .exceptions import LLMClientError, BatchTimeoutError


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

        Raises:
            ValueError: If model is empty or None.
        """
        if not model:
            raise ValueError('Model name cannot be empty or None')

        self.model = model
        logging.info(f'Initializing LLM client {self.__class__.__name__} with model {self.model}')

    # Synchronous methods
    @abstractmethod
    def call(self, prompt: str) -> str:
        """Make a synchronous API Call to the LLM with the given prompt.

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
        """Make an asynchronous API Call to the LLM with the given prompt.

        Args:
            prompt: The input prompt to send to the LLM.

        Returns:
            The response text from the model.

        Raises:
            ValueError: If prompt is empty.
            LLMClientError: If API call fails.
        """

    # Async batch processing methods (optional - not all clients support this)
    @staticmethod
    def supports_async_batch() -> bool:
        """Check if the client supports batch processing.

        Returns:
            True if batch processing is supported, False otherwise.
        """
        return False

    async def create_batch_async(
        self,
        requests: list[BatchRequest]
    ) -> str:
        """Create a batch processing job (async).

        Args:
            requests: List of batch requests.

        Returns:
            Batch job ID.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support async batch processing')

    async def get_batch_status_async(self, batch_id: str) -> BatchStatus:
        """Get the status of a batch job (async).

        Args:
            batch_id: Batch job identifier.

        Returns:
            Current batch status.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support async batch processing')

    async def get_batch_info_async(self, batch_id: str) -> Dict[str, Any]:
        """Get detailed information about a batch job asynchronously.

        Args:
            batch_id: Batch job identifier.

        Returns:
            Batch information dictionary.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError('Async batch processing not supported by this client')


    async def get_batch_results_async(self, batch_id: str) -> list[BatchResponse]:
        """Get results from a completed batch job asynchronously.

        Args:
            batch_id: The batch job ID.

        Returns:
            List of batch responses.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support async batch processing')

    async def cancel_batch_async(self, batch_id: str) -> bool:
        """Cancel a batch job asynchronously.

        Args:
            batch_id: The batch job ID.

        Returns:
            True if batch was cancelled successfully.

        Raises:
            NotImplementedError: If batch processing is not supported.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support async batch processing')

    # Abstract batch orchestration methods - must be implemented by concrete classes
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
        raise NotImplementedError('Async batch monitoring not supported by this client')

    # Shared batch orchestration methods
    async def wait_for_batch_completion_async(
            self,
            batch_id: str,
            max_wait_time: int = 86400,  # 24-hours default (batch expires after 24h)
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
            progress_callback: Optional callback for progress updates.

        Returns:
            Final batch status.

        Raises:
            BatchTimeoutError: If batch doesn't complete within max_wait_time.
        """
        if not self.supports_async_batch():
            raise LLMClientError(f'Client {self.__class__.__name__} does not support async batch processing')

        start_time = time.time()

        async for progress in self.monitor_batch_progress_async(
            batch_id, progress_callback, poll_interval
        ):
            # Check for completion
            if progress.status == BatchStatus.ENDED:
                return progress.status
            # Check for timeout
            if time.time() - start_time > max_wait_time:
                raise BatchTimeoutError(
                    f'Batch job {batch_id} did not complete within {max_wait_time} seconds',
                    batch_id=batch_id
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
        if not self.supports_async_batch():
            raise LLMClientError(f'Client {self.__class__.__name__} does not support async batch processing')

        try:
            # Create batch job.
            batch_id = await self.create_batch_async(requests)
            logging.info(f'Created batch job {batch_id} with {len(requests)} requests')

            # Wait for completion with progress monitoring
            final_status = await self.wait_for_batch_completion_async(
                batch_id, max_wait_time, poll_interval, progress_callback
            )

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
            raise LLMClientError(f'Batch processing failed: {e}') from e