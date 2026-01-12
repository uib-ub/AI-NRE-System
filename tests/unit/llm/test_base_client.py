"""Unit tests for llm.base_client module.

Tests cover:
- Client initialization and validation
- client_type property
- supports_async_batch method
- Error helper methods (_raise_llm_client_error, _raise_api_error)
- Default NotImplementedError for async batch methods
- wait_for_batch_completion_async orchestration
- process_batch_requests_async orchestration
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from typing import TYPE_CHECKING

import pytest

from ai_ner_system.llm.base_client import Client
from ai_ner_system.llm.batch_models import (
    BatchProgress,
    BatchRequest,
    BatchResponse,
    BatchStatus,
)
from ai_ner_system.llm.exceptions import APIError, BatchTimeoutError, LLMClientError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

log = logging.getLogger(__name__)


class ConcreteClient(Client):
    """Concrete implementation of Client for testing purposes."""

    def call(self, prompt: str) -> str:
        """Synchronous call implementation."""
        return f"Response to: {prompt}"

    async def call_async(self, prompt: str) -> str:
        """Asynchronous call implementation."""
        return f"Async response to: {prompt}"


class BatchEnabledClient(Client):
    """Concrete implementation with batch support for testing."""

    def __init__(self, model: str) -> None:
        """Initialize with batch support enabled."""
        super().__init__(model)
        self._batch_results: list[BatchResponse] = []
        self._batch_status = BatchStatus.IN_PROGRESS

    @staticmethod
    def supports_async_batch() -> bool:
        """Return True to enable batch support."""
        return True

    def call(self, prompt: str) -> str:
        """Synchronous call implementation."""
        return f"Response to: {prompt}"

    async def call_async(self, prompt: str) -> str:
        """Asynchronous call implementation."""
        return f"Async response to: {prompt}"

    async def create_batch_async(
        self,
        requests: list[BatchRequest],  # noqa: ARG002
    ) -> str:
        """Create batch and return batch ID."""
        # Generate realistic Anthropic batch ID format: msgbatch_<24 chars>
        random_id = secrets.token_hex(12)  # 12 bytes = 24 hex chars
        return f"msgbatch_{random_id}"

    async def get_batch_status_async(
        self,
        batch_id: str,  # noqa: ARG002
    ) -> BatchStatus:
        """Return configured batch status."""
        return self._batch_status

    async def get_batch_results_async(
        self,
        batch_id: str,  # noqa: ARG002
    ) -> list[BatchResponse]:
        """Return configured batch results."""
        return self._batch_results

    async def monitor_batch_progress_async(
        self,
        batch_num: int,
        batch_id: str,
        poll_interval: float | None = None,  # noqa: ARG002
    ) -> AsyncGenerator[BatchProgress]:
        """Yield a single progress update then complete."""
        yield BatchProgress(
            batch_num=batch_num,
            batch_id=batch_id,
            status=self._batch_status,
            elapsed_time=1.0,
            request_counts={"succeeded": 1, "errored": 0},
            created_at="2026-01-02T10:00:00Z",
            expires_at="2026-01-03T10:00:00Z",
        )


class TestClientInit:
    """Tests for Client initialization."""

    def test_init_with_valid_model(self) -> None:
        """Test client initialization with valid model name."""
        client = ConcreteClient(model="test-model")

        log.debug("Created ConcreteClient with model: %s", client.model)

        assert client.model == "test-model"

    @pytest.mark.parametrize(
        ("model", "match_pattern"),
        [
            ("", r"(?i)model name cannot be empty"),
            (None, r"(?i)model name cannot be empty"),
        ],
    )
    def test_init_with_invalid_model(
        self, model: str | None, match_pattern: str
    ) -> None:
        """Test client initialization raises ValueError for invalid model.

        Args:
            model: The model name to test.
            match_pattern: Regex pattern to match error message.
        """
        with pytest.raises(ValueError, match=match_pattern) as exc_info:
            ConcreteClient(model=model)  # type: ignore[arg-type]

        assert "model name cannot be empty" in str(exc_info.value).lower()


class TestClientProperties:
    """Tests for Client properties and helper methods."""

    def test_client_type_concrete(self) -> None:
        """Test client_type returns class name without 'Client' suffix."""
        client = ConcreteClient(model="test-model")
        assert client.client_type == "concrete"

    def test_client_type_batch_enabled(self) -> None:
        """Test client_type for BatchEnabledClient."""
        client = BatchEnabledClient(model="test-model")
        assert client.client_type == "batchenabled"

    def test_supports_async_batch_default_false(self) -> None:
        """Test supports_async_batch returns False by default."""
        client = ConcreteClient(model="test-model")
        assert client.supports_async_batch() is False

    def test_supports_async_batch_enabled(self) -> None:
        """Test supports_async_batch returns True when overridden."""
        client = BatchEnabledClient(model="test-model")
        assert client.supports_async_batch() is True


class TestErrorHelpers:
    """Tests for error helper methods."""

    def test_raise_llm_client_error(self) -> None:
        """Test _raise_llm_client_error raises LLMClientError."""
        client = ConcreteClient(model="test-model")

        with pytest.raises(LLMClientError, match=r"Test error") as exc_info:
            client._raise_llm_client_error("Test error", operation="test_op")  # pyright: ignore[reportPrivateUsage]

        log.debug("LLMClientError raised: %s", exc_info.value)

        assert "Test error" in str(exc_info.value)
        assert exc_info.value.client_type == "concrete"
        assert exc_info.value.operation == "test_op"

    def test_raise_llm_client_error_with_cause(self) -> None:
        """Test _raise_llm_client_error preserves exception cause."""
        client = ConcreteClient(model="test-model")
        original = ValueError("original error")

        with pytest.raises(LLMClientError) as exc_info:
            client._raise_llm_client_error("Wrapped", operation="wrap", cause=original)  # pyright: ignore[reportPrivateUsage]

        assert exc_info.value.__cause__ is original

    def test_raise_api_error(self) -> None:
        """Test _raise_api_error raises APIError."""
        client = ConcreteClient(model="test-model")

        with pytest.raises(APIError, match=r"API failed") as exc_info:
            client._raise_api_error("API failed", operation="api_call", status_code=500)  # pyright: ignore[reportPrivateUsage]

        log.debug("APIError raised: %s", exc_info.value)

        assert exc_info.value.client_type == "concrete"
        assert exc_info.value.operation == "api_call"
        assert exc_info.value.status_code == 500

    def test_raise_api_error_with_cause(self) -> None:
        """Test _raise_api_error preserves exception cause."""
        client = ConcreteClient(model="test-model")
        original = ConnectionError("connection lost")

        with pytest.raises(APIError) as exc_info:
            client._raise_api_error("Failed", operation="call", cause=original)  # pyright: ignore[reportPrivateUsage]

        assert exc_info.value.__cause__ is original


class TestAsyncBatchNotImplemented:
    """Tests for default NotImplementedError behavior of async batch methods."""

    @pytest.mark.asyncio
    async def test_create_batch_async_not_implemented(self) -> None:
        """Test create_batch_async raises NotImplementedError by default."""
        client = ConcreteClient(model="test-model")

        with pytest.raises(
            NotImplementedError, match=r"ConcreteClient.*async batch"
        ) as exc_info:
            await client.create_batch_async([])

        log.debug("NotImplementedError raised: %s", exc_info.value)
        assert "ConcreteClient" in str(exc_info.value)
        assert "does not support async batch processing" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_batch_status_async_not_implemented(self) -> None:
        """Test get_batch_status_async raises NotImplementedError by default."""
        client = ConcreteClient(model="test-model")

        with pytest.raises(
            NotImplementedError, match=r"ConcreteClient.*async batch"
        ) as exc_info:
            await client.get_batch_status_async("batch_123")

        log.debug("NotImplementedError raised: %s", exc_info.value)
        assert "ConcreteClient" in str(exc_info.value)
        assert "does not support async batch processing" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_batch_info_async_not_implemented(self) -> None:
        """Test get_batch_info_async raises NotImplementedError by default."""
        client = ConcreteClient(model="test-model")

        with pytest.raises(
            NotImplementedError, match=r"ConcreteClient.*async batch"
        ) as exc_info:
            await client.get_batch_info_async("batch_123")

        log.debug("NotImplementedError raised: %s", exc_info.value)
        assert "ConcreteClient" in str(exc_info.value)
        assert "does not support async batch processing" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_batch_results_async_not_implemented(self) -> None:
        """Test get_batch_results_async raises NotImplementedError by default."""
        client = ConcreteClient(model="test-model")

        with pytest.raises(
            NotImplementedError, match=r"ConcreteClient.*async batch"
        ) as exc_info:
            await client.get_batch_results_async("batch_123")

        log.debug("NotImplementedError raised: %s", exc_info.value)
        assert "ConcreteClient" in str(exc_info.value)
        assert "does not support async batch processing" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cancel_batch_async_not_implemented(self) -> None:
        """Test cancel_batch_async raises NotImplementedError by default."""
        client = ConcreteClient(model="test-model")

        with pytest.raises(
            NotImplementedError, match=r"ConcreteClient.*async batch"
        ) as exc_info:
            await client.cancel_batch_async("batch_123")

        log.debug("NotImplementedError raised: %s", exc_info.value)
        assert "ConcreteClient" in str(exc_info.value)
        assert "does not support async batch processing" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_monitor_batch_progress_async_not_implemented(self) -> None:
        """Test monitor_batch_progress_async raises NotImplementedError by default."""
        client = ConcreteClient(model="test-model")

        with pytest.raises(
            NotImplementedError, match=r"ConcreteClient.*async batch"
        ) as exc_info:
            async for _ in client.monitor_batch_progress_async(1, "batch_123"):
                pass

        log.debug("NotImplementedError raised: %s", exc_info.value)
        assert "ConcreteClient" in str(exc_info.value)
        assert "does not support async batch monitoring" in str(exc_info.value)


class TestWaitForBatchCompletionAsync:
    """Tests for wait_for_batch_completion_async method."""

    @pytest.mark.asyncio
    async def test_raises_error_when_batch_not_supported(self) -> None:
        """Test raises LLMClientError when client doesn't support batch."""
        client = ConcreteClient(model="test-model")

        with pytest.raises(
            LLMClientError, match=r"does not support async batch"
        ) as exc_info:
            await client.wait_for_batch_completion_async(1, "batch_123")

        log.debug("LLMClientError raised: %s", exc_info.value)
        assert exc_info.value.client_type == "concrete"
        assert "ConcreteClient" in str(exc_info.value)
        assert "does not support async batch processing" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_successful_completion(self) -> None:
        """Test successful batch completion."""
        client = BatchEnabledClient(model="test-model")
        client._batch_status = BatchStatus.ENDED  # pyright: ignore[reportPrivateUsage]

        status = await client.wait_for_batch_completion_async(1, "batch_123")
        assert status == BatchStatus.ENDED

    @pytest.mark.asyncio
    async def test_with_progress_callback(self) -> None:
        """Test progress callback is invoked."""
        client = BatchEnabledClient(model="test-model")
        client._batch_status = BatchStatus.ENDED  # pyright: ignore[reportPrivateUsage]
        callback_called: list[BatchProgress] = []

        def progress_callback(progress: BatchProgress) -> None:
            callback_called.append(progress)

        status = await client.wait_for_batch_completion_async(
            1, "batch_123", progress_callback=progress_callback
        )
        assert len(callback_called) == 1
        assert callback_called[0].batch_id == "batch_123"
        assert status == BatchStatus.ENDED

    @pytest.mark.asyncio
    async def test_callback_exception_isolated(self) -> None:
        """Test callback exception doesn't break batch processing."""
        client = BatchEnabledClient(model="test-model")
        client._batch_status = BatchStatus.ENDED  # pyright: ignore[reportPrivateUsage]

        def bad_callback(progress: BatchProgress) -> None:  # noqa: ARG001
            raise RuntimeError("Callback failed")

        # Should not raise despite callback error
        status = await client.wait_for_batch_completion_async(
            1, "batch_123", progress_callback=bad_callback
        )

        assert status == BatchStatus.ENDED

    @pytest.mark.asyncio
    async def test_invalid_poll_interval(self) -> None:
        """Test raises ValueError for invalid poll_interval."""
        client = BatchEnabledClient(model="test-model")

        with pytest.raises(ValueError, match=r"poll_interval must be > 0") as exc_info:
            await client.wait_for_batch_completion_async(
                1, "batch_123", poll_interval=0
            )

        log.debug("ValueError raised: %s", exc_info.value)
        assert "poll_interval must be > 0" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_max_wait_time(self) -> None:
        """Test raises ValueError for invalid max_wait_time."""
        client = BatchEnabledClient(model="test-model")

        with pytest.raises(ValueError, match=r"max_wait_time must be > 0") as exc_info:
            await client.wait_for_batch_completion_async(
                1, "batch_123", max_wait_time=-1
            )

        log.debug("ValueError raised: %s", exc_info.value)
        assert "max_wait_time must be > 0" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_raises_batch_timeout_error(self) -> None:
        """Test timeout raises BatchTimeoutError."""
        client = BatchEnabledClient(model="test-model")
        client._batch_status = BatchStatus.IN_PROGRESS  # pyright: ignore[reportPrivateUsage]

        # Override monitor to yield IN_PROGRESS indefinitely
        async def mock_monitor(
            batch_num: int,
            batch_id: str,
            poll_interval: float | None = None,  # noqa: ARG001
        ) -> AsyncGenerator[BatchProgress]:
            while True:
                yield BatchProgress(
                    batch_num=batch_num,
                    batch_id=batch_id,
                    status=BatchStatus.IN_PROGRESS,
                    elapsed_time=1.0,
                    request_counts={},
                    created_at="2026-01-02T10:00:00Z",
                    expires_at="2026-01-03T10:00:00Z",
                )
                await asyncio.sleep(0.01)

        client.monitor_batch_progress_async = mock_monitor  # type: ignore[method-assign]

        with pytest.raises(
            BatchTimeoutError, match=r"did not complete within"
        ) as exc_info:
            await client.wait_for_batch_completion_async(
                1, "batch_123", max_wait_time=0.05, poll_interval=0.01
            )

        log.debug("BatchTimeoutError raised: %s", exc_info.value)

        assert exc_info.value.client_type == "batchenabled"
        assert exc_info.value.batch_id == "batch_123"
        assert exc_info.value.operation == "batch_waiting"
        assert exc_info.value.timeout_seconds == 0.05


class TestProcessBatchRequestsAsync:
    """Tests for process_batch_requests_async method."""

    @pytest.mark.asyncio
    async def test_raises_error_when_batch_not_supported(self) -> None:
        """Test raises LLMClientError when client doesn't support batch."""
        client = ConcreteClient(model="test-model")
        requests = [BatchRequest(custom_id="req-1", prompt="Test")]

        with pytest.raises(
            LLMClientError, match=r"does not support async batch"
        ) as exc_info:
            await client.process_batch_requests_async(requests, batch_num=1)

        log.debug("LLMClientError raised: %s", exc_info.value)

        assert exc_info.value.client_type == "concrete"
        assert exc_info.value.operation == "batch_processing"
        assert "does not support async batch processing" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_error_for_empty_requests(self) -> None:
        """Test raises ValueError for empty request list."""
        client = BatchEnabledClient(model="test-model")

        with pytest.raises(
            ValueError, match=r"Request list cannot be empty"
        ) as exc_info:
            await client.process_batch_requests_async([], batch_num=1)

        log.debug("ValueError raised: %s", exc_info.value)

        assert "Request list cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_successful_batch_processing(self) -> None:
        """Test successful end-to-end batch processing."""
        client = BatchEnabledClient(model="test-model")
        client._batch_status = BatchStatus.ENDED  # pyright: ignore[reportPrivateUsage]
        client._batch_results = [  # pyright: ignore[reportPrivateUsage]
            BatchResponse(custom_id="req-1", response_text="Result 1", success=True),
            BatchResponse(custom_id="req-2", response_text="Result 2", success=True),
        ]

        requests = [
            BatchRequest(custom_id="req-1", prompt="Prompt 1"),
            BatchRequest(custom_id="req-2", prompt="Prompt 2"),
        ]

        results = await client.process_batch_requests_async(requests, batch_num=1)

        assert len(results) == 2
        for idx, result in enumerate(results):
            assert result.custom_id == requests[idx].custom_id

    @pytest.mark.asyncio
    async def test_with_progress_callback(self) -> None:
        """Test progress callback is passed through to wait method."""
        client = BatchEnabledClient(model="test-model")
        client._batch_status = BatchStatus.ENDED  # pyright: ignore[reportPrivateUsage]
        client._batch_results = [  # pyright: ignore[reportPrivateUsage]
            BatchResponse(custom_id="req-1", response_text="Result", success=True),
        ]
        callback_invoked: list[BatchProgress] = []

        def callback(progress: BatchProgress) -> None:
            callback_invoked.append(progress)

        requests = [BatchRequest(custom_id="req-1", prompt="Test")]
        await client.process_batch_requests_async(
            requests, batch_num=1, progress_callback=callback
        )

        assert len(callback_invoked) == 1

    @pytest.mark.asyncio
    async def test_preserves_llm_client_error(self) -> None:
        """Test LLMClientError is preserved and not wrapped."""
        client = BatchEnabledClient(model="test-model")
        client._batch_status = BatchStatus.ENDED  # pyright: ignore[reportPrivateUsage]

        async def failing_create(
            requests: list[BatchRequest],  # noqa: ARG001
        ) -> str:
            raise LLMClientError(
                "Create failed", client_type="batchenabled", operation="create"
            )

        client.create_batch_async = failing_create  # type: ignore[method-assign]

        requests = [BatchRequest(custom_id="req-1", prompt="Test")]

        with pytest.raises(LLMClientError, match=r"Create failed") as exc_info:
            await client.process_batch_requests_async(requests, batch_num=1)

        log.debug("LLMClientError raised: %s", exc_info.value)

        assert "Create failed" in str(exc_info.value)
        assert exc_info.value.client_type == "batchenabled"
        assert exc_info.value.operation == "create"

    @pytest.mark.asyncio
    async def test_preserves_batch_timeout_error(self) -> None:
        """Test BatchTimeoutError is preserved and not wrapped."""
        client = BatchEnabledClient(model="test-model")

        async def timeout_wait(
            *args: object,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> BatchStatus:
            raise BatchTimeoutError(
                "Timeout", batch_id="batch_123", timeout_seconds=100
            )

        client.wait_for_batch_completion_async = timeout_wait  # type: ignore[method-assign]

        requests = [BatchRequest(custom_id="req-1", prompt="Test")]

        with pytest.raises(BatchTimeoutError, match=r"Timeout") as exc_info:
            await client.process_batch_requests_async(requests, batch_num=1)

        log.debug("BatchTimeoutError raised: %s", exc_info.value)

        assert "Timeout" in str(exc_info.value)
        assert exc_info.value.batch_id == "batch_123"
        assert exc_info.value.timeout_seconds == 100

    @pytest.mark.asyncio
    async def test_wraps_unexpected_exception(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test unexpected exceptions are wrapped in LLMClientError."""
        client = BatchEnabledClient(model="test-model")

        async def failing_create(
            requests: list[BatchRequest],  # noqa: ARG001
        ) -> str:
            raise RuntimeError("Unexpected error")

        client.create_batch_async = failing_create  # type: ignore[method-assign]

        requests = [BatchRequest(custom_id="req-1", prompt="Test")]

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(LLMClientError, match=r"Batch processing failed") as exc_info,
        ):
            await client.process_batch_requests_async(requests, batch_num=1)

        log.debug("LLMClientError raised: %s", exc_info.value)

        # Verify the error was logged
        assert "Batch processing failed" in caplog.text
        assert "RuntimeError" in caplog.text
        assert "Unexpected error" in caplog.text

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self) -> None:
        """Test asyncio.CancelledError propagates without wrapping."""
        client = BatchEnabledClient(model="test-model")

        async def cancelled_create(
            requests: list[BatchRequest],  # noqa: ARG001
        ) -> str:
            raise asyncio.CancelledError

        client.create_batch_async = cancelled_create  # type: ignore[method-assign]

        requests = [BatchRequest(custom_id="req-1", prompt="Test")]

        with pytest.raises(asyncio.CancelledError):
            await client.process_batch_requests_async(requests, batch_num=1)

    @pytest.mark.asyncio
    async def test_non_ended_status_raises_error(self) -> None:
        """Test non-ENDED final status raises LLMClientError."""
        client = BatchEnabledClient(model="test-model")
        client._batch_status = BatchStatus.CANCELING  # pyright: ignore[reportPrivateUsage]

        requests = [BatchRequest(custom_id="req-1", prompt="Test")]

        with pytest.raises(
            LLMClientError, match=r"failed with status canceling"
        ) as exc_info:
            await client.process_batch_requests_async(requests, batch_num=1)

        log.debug("LLMClientError raised: %s", exc_info.value)

        assert exc_info.value.client_type == "batchenabled"
        assert exc_info.value.operation == "batch_processing"
        assert "failed with status canceling" in str(exc_info.value)


class TestClassConstants:
    """Tests for class-level constants."""

    def test_default_max_wait_time(self) -> None:
        """Test DEFAULT_MAX_WAIT_TIME is 24 hours."""
        assert Client.DEFAULT_MAX_WAIT_TIME == 86400.0

    def test_default_poll_interval(self) -> None:
        """Test DEFAULT_POLL_INTERVAL is 30 seconds."""
        assert Client.DEFAULT_POLL_INTERVAL == 30.0

    def test_error_response_sentinels(self) -> None:
        """Test ERROR_RESPONSE_SENTINELS contains expected values."""
        sentinels = Client.ERROR_RESPONSE_SENTINELS
        assert "Claude API call failed" in sentinels
        assert "Ollama API call failed" in sentinels
        assert len(sentinels) == 2
