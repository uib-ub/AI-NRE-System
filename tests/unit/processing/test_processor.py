"""Unit tests for processing.processor module.

Tests cover:
- RecordProcessor initialization
- Sync single-record: process_record()
- Sync batch: process_batch()
- Async single-record: process_record_async()
- Async batch: process_batch_async() (with batch support + fallback)
- Helper methods: _call_llm, _create_batch_id, create_record_id,
  _create_custom_id, _extract_index_from_custom_id, _create_processing_result,
  _prepare_batch_requests, _build_batch_results, _process_single_batch_response,
  _create_response_map
- create_progress_logger() utility function
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from ai_ner_system.processing.entities import ProcessingResult

from ai_ner_system.llm.batch_models import (
    BatchProgress,
    BatchRequest,
    BatchResponse,
    BatchStatus,
)
from ai_ner_system.llm.exceptions import LLMClientError
from ai_ner_system.processing.entities import (
    BatchProcessingResult,
    EntityRecord,
)
from ai_ner_system.processing.exceptions import (
    BatchProcessingError,
    ProcessingError,
)
from ai_ner_system.processing.processor import RecordProcessor, create_progress_logger
from tests.unit.processing.conftest import (
    ANNOTATED_TEXT,
    BINDNR,
    BREVID,
    EXPECTED_ENTITY,
    GENERATED_PROMPT,
    METADATA_TEXT,
    SAMPLE_BATCH_LLM_RESPONSE,
    SAMPLE_LLM_RESPONSE,
    VALID_ENTITY_DATA,
    VALID_RECORD,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aliases for protected static helpers (avoid repeated type-ignore comments)
# ---------------------------------------------------------------------------
_create_batch_id = RecordProcessor._create_batch_id  # pyright: ignore[reportPrivateUsage]
_create_custom_id = RecordProcessor._create_custom_id  # pyright: ignore[reportPrivateUsage]
_extract_index_from_custom_id = RecordProcessor._extract_index_from_custom_id  # pyright: ignore[reportPrivateUsage]
_create_processing_result = RecordProcessor._create_processing_result  # pyright: ignore[reportPrivateUsage]


# ===================================================================
# Initialization
# ===================================================================
class TestInit:
    """Tests for RecordProcessor.__init__()."""

    def test_init(
        self,
        mock_llm_client: Any,
        mock_prompt_builder: Any,
    ) -> None:
        """Test processor initialization stores dependencies."""
        record_processor = RecordProcessor(
            llm_client=mock_llm_client,
            prompt_builder=mock_prompt_builder,
        )
        assert record_processor.llm_client is mock_llm_client
        assert record_processor.prompt_builder is mock_prompt_builder

    def test_class_constants(self) -> None:
        """Test class-level constants have expected values."""
        assert RecordProcessor.DEFAULT_MAX_WAIT_TIME == 86400.0  # 24 hours
        assert RecordProcessor.DEFAULT_POLL_INTERVAL == 30.0  # 30 seconds
        assert RecordProcessor.DEFAULT_MAX_TOKENS == 20000
        assert RecordProcessor.DEFAULT_TEMPERATURE == 0.0
        assert RecordProcessor.MIN_CUSTOM_ID_PARTS == 2  # Minimum parts for custom ID


# ===================================================================
# process_record (sync single-record)
# ===================================================================
class TestProcessRecord:
    """Tests for RecordProcessor.process_record()."""

    def test_success(
        self,
        processor: Any,
    ) -> None:
        """Test successful single-record processing."""
        annotated, metadata = processor.process_record(VALID_RECORD)

        log.debug("Annotated text: %s", annotated)
        log.debug("Metadata: %s", metadata)

        assert len(annotated) == 1
        assert len(metadata) == 1
        assert annotated[0] == f'{BINDNR};{BREVID};"{ANNOTATED_TEXT}"'
        assert metadata[0] == METADATA_TEXT
        processor.llm_client.call.assert_called_once()  # LLM should be called once
        processor.prompt_builder.build.assert_called_once_with(
            VALID_RECORD
        )  # Prompt should be built with the input record

    def test_validation_error_reraises(
        self,
        processor: Any,
    ) -> None:
        """Test ValidationError is re-raised as ProcessingError."""
        invalid_record = {"Bindnr": "1", "Brevid": ""}
        with pytest.raises(ProcessingError) as exc_info:
            processor.process_record(invalid_record)

        log.debug("Caught exception: %s", exc_info.value)
        assert "Record missing required fields" in str(exc_info.value)

    def test_processing_error_reraises(self, processor: Any) -> None:
        """Test ProcessingError from LLM call is re-raised as-is."""
        processor.llm_client.call.side_effect = LLMClientError(
            "LLM failure",
            operation="call_llm",
        )
        with pytest.raises(ProcessingError, match="Error during LLM call") as exc_info:
            processor.process_record(VALID_RECORD)

        log.debug("Caught exception: %s", exc_info.value)
        assert "Error during LLM call" in str(exc_info.value)
        assert "LLM failure" in str(exc_info.value)
        assert exc_info.value.operation == "call_llm"

    def test_unexpected_error_wraps(
        self,
        processor: Any,
    ) -> None:
        """Test unexpected exception is wrapped in ProcessingError."""
        processor.prompt_builder.build.side_effect = RuntimeError("Unexpected error")
        with pytest.raises(
            ProcessingError, match="Failed to process record with Brevid"
        ) as exc_info:
            processor.process_record(VALID_RECORD)

        log.debug("Caught exception: %s", exc_info.value)

        assert f"Failed to process record with Brevid {BREVID}" in str(exc_info.value)
        assert "Unexpected error" in str(exc_info.value)
        assert exc_info.value.brevid == BREVID
        assert exc_info.value.operation == "process_record"
        assert isinstance(exc_info.value.__cause__, RuntimeError)


# ===================================================================
# process_batch (sync batch)
# ===================================================================
class TestProcessBatch:
    """Tests for RecordProcessor.process_batch()."""

    def test_success(
        self,
        processor: Any,
    ) -> None:
        """Test successful batch processing."""
        processor.llm_client.call.return_value = SAMPLE_BATCH_LLM_RESPONSE
        records = [VALID_RECORD]
        annotated, metadata = processor.process_batch(records)
        log.debug("Annotated batch: %s", annotated)
        log.debug("Metadata batch: %s", metadata)

        assert len(annotated) == 1
        assert len(metadata) == 1
        assert annotated[0] == f'{BINDNR};{BREVID};"{ANNOTATED_TEXT}"'
        assert metadata[0] == METADATA_TEXT
        processor.llm_client.call.assert_called_once()  # LLM should be called once for the batch

    def test_empty_records_return_empty(
        self,
        processor: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test empty records list returns empty results."""
        with caplog.at_level(logging.WARNING):
            annotated, metadata = processor.process_batch([])

        log.debug("Annotated batch: %s", annotated)
        log.debug("Metadata batch: %s", metadata)
        assert annotated == []
        assert metadata == []
        assert "Empty records list provided to process_batch" in caplog.text

    def test_validation_error_reraises(
        self,
        processor: Any,
    ) -> None:
        """Test ValidationError from validate is re-raised."""
        invalid_records = [{"Bindnr": "1", "Brevid": ""}]
        with pytest.raises(ProcessingError) as exc_info:
            processor.process_batch(invalid_records)

        log.debug("Caught exception: %s", exc_info.value)
        assert "Record missing required fields" in str(exc_info.value)

    def test_processing_error_reraises(
        self,
        processor: Any,
    ) -> None:
        """Test ProcessingError is re-raised without double-wrapping."""
        processor.llm_client.call.side_effect = LLMClientError(
            "LLM failure",
            operation="call_llm",
        )
        with pytest.raises(
            ProcessingError,
            match="Error during LLM call",
        ) as exc_info:
            processor.process_batch([VALID_RECORD])

        log.debug("Caught exception: %s", exc_info.value)

        assert "Error during LLM call" in str(exc_info.value)
        assert "LLM failure" in str(exc_info.value)
        assert exc_info.value.operation == "call_llm"

    def test_unexpected_error_wraps(
        self,
        processor: Any,
    ) -> None:
        """Test unexpected exception is wrapped in ProcessingError."""
        processor.prompt_builder.build.side_effect = RuntimeError("Unexpected")
        with pytest.raises(
            ProcessingError, match="Failed to process batch"
        ) as exc_info:
            processor.process_batch([VALID_RECORD])

        log.debug("Caught exception: %s", exc_info.value)

        assert "Failed to process batch" in str(exc_info.value)
        assert "Unexpected" in str(exc_info.value)
        assert exc_info.value.operation == "process_batch"
        assert isinstance(exc_info.value.__cause__, RuntimeError)


# ===================================================================
# process_record_async
# ===================================================================
class TestProcessRecordAsync:
    """Tests for RecordProcessor.process_record_async()."""

    @pytest.mark.asyncio
    async def test_success(
        self,
        processor: Any,
    ) -> None:
        """Test successful async single-record processing."""
        result = await processor.process_record_async(VALID_RECORD)

        log.debug("Async processing result: %s", result)
        assert result.success is True
        assert result.brevid == BREVID
        assert result.annotated_text == f'{BINDNR};{BREVID};"{ANNOTATED_TEXT}"'
        log.debug("Metadata: %s", result.entities[0])
        assert result.entities[0] == EXPECTED_ENTITY

    @pytest.mark.asyncio
    async def test_validation_error(
        self,
        processor: Any,
    ) -> None:
        """Test ValidationError returns failed ProcessingResult."""
        # Brevid must be non-empty so ProcessingResult creation succeeds
        invalid_record: dict[str, str] = {"Bindnr": "1", "Brevid": "999", "Tekst": ""}
        result = await processor.process_record_async(invalid_record)

        log.debug("Async processing result: %s", result)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_llm_client_error(
        self,
        processor: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test LLMClientError returns failed ProcessingResult."""
        processor.llm_client.call_async = mocker.AsyncMock(
            side_effect=LLMClientError("LLM failure", operation="call_async")
        )

        result = await processor.process_record_async(VALID_RECORD)

        log.debug("Async processing result: %s", result)
        assert result.success is False
        assert result.brevid == BREVID
        assert result.annotated_text == ""
        assert result.entities == []
        assert "LLM failure" in str(result.error_message)

    @pytest.mark.asyncio
    async def test_parse_error(
        self,
        processor: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test ParseError returns failed ProcessingResult."""
        processor.llm_client.call_async = mocker.AsyncMock(
            return_value="text\n===JSON===\n{invalid json}"
        )

        result = await processor.process_record_async(VALID_RECORD)

        log.debug("Async processing result: %s", result)
        assert result.success is False
        assert result.brevid == BREVID
        assert result.annotated_text == ""
        assert result.entities == []
        assert "Invalid JSON format" in str(result.error_message)

    @pytest.mark.asyncio
    async def test_llm_response_error(
        self,
        processor: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test LLMResponseError (empty response) returns failed ProcessingResult."""
        processor.llm_client.call_async = mocker.AsyncMock(return_value="")
        result = await processor.process_record_async(VALID_RECORD)

        log.debug("Async processing result: %s", result)
        assert result.success is False
        assert result.brevid == BREVID
        assert result.annotated_text == ""
        assert result.entities == []
        assert "Empty response from LLM" in str(result.error_message)


# ===================================================================
# process_batch_async
# ===================================================================
class TestProcessBatchAsync:
    """Tests for RecordProcessor.process_batch_async()."""

    @pytest.mark.asyncio
    async def test_empty_records_raises(
        self,
        processor: Any,
    ) -> None:
        """Test empty records raises ValueError."""
        with pytest.raises(
            ValueError, match="Records list cannot be empty"
        ) as exc_info:
            await processor.process_batch_async([], batch_num=1)

        log.debug("Caught exception: %s", exc_info.value)
        assert "Records list cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fallback_to_individual(
        self,
        processor: Any,
    ) -> None:
        """Test fallback to individual processing when batch not supported."""
        processor.llm_client.supports_async_batch.return_value = False
        result = await processor.process_batch_async([VALID_RECORD], batch_num=1)

        log.debug("Fallback processing result: %s", result)
        assert isinstance(result, BatchProcessingResult)
        assert len(result.results) == 1
        assert result.results[0].brevid == BREVID
        assert result.results[0].success is True
        assert (
            result.results[0].annotated_text == f'{BINDNR};{BREVID};"{ANNOTATED_TEXT}"'
        )
        assert result.results[0].entities[0] == EXPECTED_ENTITY

    @pytest.mark.asyncio
    async def test_with_batch_support(
        self,
        processor: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test async batch processing when client supports it."""
        processor.llm_client.supports_async_batch.return_value = True
        processor.llm_client.process_batch_requests_async = mocker.AsyncMock(
            return_value=[
                BatchResponse(
                    custom_id=f"record_0_{BINDNR}_{BREVID}",
                    response_text=SAMPLE_LLM_RESPONSE,
                    success=True,
                )
            ]
        )

        result = await processor.process_batch_async([VALID_RECORD], batch_num=1)
        log.debug("Batch processing result: %s", result)
        assert isinstance(result, BatchProcessingResult)
        assert result.batch_id == "batch_1"
        assert len(result.results) == 1
        assert result.results[0].brevid == BREVID
        assert result.results[0].success is True
        assert (
            result.results[0].annotated_text == f'{BINDNR};{BREVID};"{ANNOTATED_TEXT}"'
        )
        assert result.results[0].entities[0] == EXPECTED_ENTITY
        assert result.results[0].record_id == f"record_0_{BINDNR}_{BREVID}"

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(
        self,
        processor: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test CancelledError propagates without being caught."""
        processor.llm_client.supports_async_batch.return_value = True
        processor.llm_client.process_batch_requests_async = mocker.AsyncMock(
            side_effect=asyncio.CancelledError()
        )

        with pytest.raises(asyncio.CancelledError) as exc_info:
            await processor.process_batch_async([VALID_RECORD], batch_num=1)

        log.debug("Caught exception: %s", exc_info.value)

    @pytest.mark.asyncio
    async def test_generic_error_returns_failed(
        self,
        processor: Any,
        caplog: pytest.LogCaptureFixture,
        mocker: MockerFixture,
    ) -> None:
        """Test generic exception returns failed BatchProcessingResult."""
        processor.llm_client.supports_async_batch.return_value = True
        processor.llm_client.process_batch_requests_async = mocker.AsyncMock(
            side_effect=RuntimeError("Unexpected error")
        )

        with caplog.at_level(logging.ERROR):
            result = await processor.process_batch_async([VALID_RECORD], batch_num=1)

        log.debug("Batch processing result: %s", result)
        log.debug("Captured logs: %s", caplog.text)
        assert isinstance(result, BatchProcessingResult)
        assert result.batch_id == "batch_1"
        assert result.results == []  # No results should be returned on error
        assert "Batch processing failed" in caplog.text

    @pytest.mark.asyncio
    async def test_no_valid_requests_raises(
        self,
        processor: Any,
    ) -> None:
        """Test BatchProcessingError raised when no valid requests."""
        processor.llm_client.supports_async_batch.return_value = True
        # All records invalid -> no batch requests
        invalid_records = [{"Bindnr": "1", "Brevid": ""}]
        with pytest.raises(BatchProcessingError, match="No valid requests") as exc_info:
            await processor.process_batch_async(invalid_records, batch_num=1)

        log.debug("Caught exception: %s", exc_info.value)
        assert "No valid requests to process" in str(exc_info.value)
        assert exc_info.value.operation == "prepare_batch"
        assert exc_info.value.batch_id == "batch_1"

    @pytest.mark.asyncio
    async def test_default_wait_and_poll(
        self,
        processor: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test default max_wait_time and poll_interval are used."""
        processor.llm_client.supports_async_batch.return_value = True
        processor.llm_client.process_batch_requests_async = mocker.AsyncMock(
            return_value=[
                BatchResponse(
                    custom_id=f"record_0_{BINDNR}_{BREVID}",
                    response_text=SAMPLE_LLM_RESPONSE,
                    success=True,
                )
            ]
        )

        result = await processor.process_batch_async([VALID_RECORD], batch_num=1)
        log.debug("Batch processing result: %s", result)

        # Verify defaults were passed through
        call_kwargs = processor.llm_client.process_batch_requests_async.call_args
        assert (
            call_kwargs.kwargs["max_wait_time"] == RecordProcessor.DEFAULT_MAX_WAIT_TIME
        )
        assert (
            call_kwargs.kwargs["poll_interval"] == RecordProcessor.DEFAULT_POLL_INTERVAL
        )

    @pytest.mark.asyncio
    async def test_custom_wait_and_poll(
        self,
        processor: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test custom max_wait_time and poll_interval override defaults."""
        processor.llm_client.supports_async_batch.return_value = True
        processor.llm_client.process_batch_requests_async = mocker.AsyncMock(
            return_value=[
                BatchResponse(
                    custom_id=f"record_0_{BINDNR}_{BREVID}",
                    response_text=SAMPLE_LLM_RESPONSE,
                    success=True,
                ),
            ]
        )

        await processor.process_batch_async(
            [VALID_RECORD],
            batch_num=1,
            max_wait_time=120.0,
            poll_interval=5.0,
        )

        call_kwargs = processor.llm_client.process_batch_requests_async.call_args
        assert call_kwargs.kwargs["max_wait_time"] == 120.0
        assert call_kwargs.kwargs["poll_interval"] == 5.0

    @pytest.mark.asyncio
    async def test_with_progress_callback(
        self,
        processor: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test progress callback is passed to client."""
        processor.llm_client.supports_async_batch.return_value = True
        processor.llm_client.process_batch_requests_async = mocker.AsyncMock(
            return_value=[
                BatchResponse(
                    custom_id=f"record_0_{BINDNR}_{BREVID}",
                    response_text=SAMPLE_LLM_RESPONSE,
                    success=True,
                ),
            ]
        )

        callback = mocker.MagicMock()

        await processor.process_batch_async(
            [VALID_RECORD],
            batch_num=1,
            progress_callback=callback,
        )

        call_kwargs = processor.llm_client.process_batch_requests_async.call_args
        assert call_kwargs.kwargs["progress_callback"] is callback


# ===================================================================
# Call LLM helper
# ===================================================================
class TestCallLLM:
    """Tests for RecordProcessor._call_llm()."""

    def test_success(
        self,
        processor: Any,
    ) -> None:
        """Test successful LLM call returns response."""
        result = processor._call_llm(BREVID, GENERATED_PROMPT)
        log.debug("LLM call result: %s", result)
        assert result == SAMPLE_LLM_RESPONSE
        processor.llm_client.call.assert_called_once_with(GENERATED_PROMPT)

    def test_client_error_raises(
        self,
        processor: Any,
    ) -> None:
        """Test LLMClientError is wrapped in ProcessingError."""
        processor.llm_client.call.side_effect = LLMClientError(
            "API error",
            operation="call_llm",
        )

        with pytest.raises(
            ProcessingError,
            match="Error during LLM call",
        ) as exc_info:
            processor._call_llm(BREVID, GENERATED_PROMPT)

        log.debug("Caught exception: %s", exc_info.value)
        assert "Error during LLM call" in str(exc_info.value)

    @pytest.mark.parametrize(
        "return_value",
        ["", "   ", "Claude API call failed"],
        ids=["empty_response", "whitespace_only", "error_sentinel"],
    )
    def test_error_response_raises(
        self,
        processor: Any,
        return_value: str,
    ) -> None:
        """Test empty or sentinel LLM response raises ProcessingError."""
        processor.llm_client.call.return_value = return_value

        with pytest.raises(
            ProcessingError,
            match="LLM returned error response",
        ) as exc_info:
            processor._call_llm(BREVID, GENERATED_PROMPT)

        log.debug("Caught exception: %s", exc_info.value)
        assert "LLM returned error response" in str(exc_info.value)
        assert exc_info.value.operation == "call_llm"


# ===================================================================
# Prepare batch requests
# ===================================================================
class TestPrepareBatchRequests:
    """Test for RecordProcessor._prepare_batch_requests()."""

    def test_success(
        self,
        processor: Any,
    ) -> None:
        """Test successful batch request preparation."""
        requests = processor._prepare_batch_requests([VALID_RECORD])

        log.debug("Prepared batch requests: %s", requests)

        assert len(requests) == 1
        assert isinstance(requests[0], BatchRequest)
        assert requests[0].prompt == GENERATED_PROMPT
        assert requests[0].custom_id == f"record_0_{BINDNR}_{BREVID}"
        assert requests[0].max_tokens == RecordProcessor.DEFAULT_MAX_TOKENS
        assert requests[0].temperature == RecordProcessor.DEFAULT_TEMPERATURE

    def test_skips_invalid(
        self,
        processor: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test invalid records are skipped with logging."""
        records = [
            VALID_RECORD,
            {"Bindnr": "2", "Brevid": ""},  # Invalid record
        ]
        with caplog.at_level(logging.ERROR):
            requests = processor._prepare_batch_requests(records)

        log.debug("Requests: %s", requests)
        log.debug("Captured logs: %s", caplog.text)

        assert len(requests) == 1
        assert "Failed to prepare batch request" in caplog.text

    def test_all_invalid_returns_empty(
        self,
        processor: Any,
    ) -> None:
        """Test all invalid records returns empty list."""
        records = [
            {"Bindnr": "1", "Brevid": ""},  # Invalid record
        ]
        requests = processor._prepare_batch_requests(records)
        log.debug("Requests: %s", requests)
        assert requests == []


# ===================================================================
# Build batch results
# ===================================================================
class TestBuildBatchResults:
    """Tests for RecordProcessor._build_batch_results()."""

    def test_success(
        self,
        processor: Any,
    ) -> None:
        """Test all responses successful built."""
        responses = [
            BatchResponse(
                custom_id=f"record_0_{BINDNR}_{BREVID}",
                response_text=SAMPLE_LLM_RESPONSE,
                success=True,
            )
        ]
        results = processor._build_batch_results([VALID_RECORD], responses)
        log.debug("Built batch results: %s", results)
        assert len(results) == 1
        assert results[0].brevid == BREVID
        assert results[0].record_id == f"record_0_{BINDNR}_{BREVID}"
        assert results[0].success is True
        assert results[0].annotated_text == f'{BINDNR};{BREVID};"{ANNOTATED_TEXT}"'
        assert results[0].entities[0] == EXPECTED_ENTITY

    def test_with_missing_response(
        self,
        processor: Any,
    ) -> None:
        """Test missing response produces failed result."""
        results = processor._build_batch_results(
            [VALID_RECORD],
            [],  # No responses
        )
        log.debug("Built batch results: %s", results)
        assert len(results) == 1
        assert results[0].brevid == BREVID
        assert results[0].record_id == f"record_0_{BINDNR}_{BREVID}"
        assert results[0].success is False
        assert results[0].annotated_text == ""
        assert results[0].entities == []
        assert "No response received for record" in results[0].error_message

    def test_with_failed_response(
        self,
        processor: Any,
    ) -> None:
        """Test failed response produces failed result."""
        responses = [
            BatchResponse(
                custom_id=f"record_0_{BINDNR}_{BREVID}",
                response_text="",
                success=False,
                error_message="API overloaded",
            ),
        ]
        results = processor._build_batch_results([VALID_RECORD], responses)
        log.debug("Built batch results: %s", results)
        assert len(results) == 1
        assert results[0].brevid == BREVID
        assert results[0].record_id == f"record_0_{BINDNR}_{BREVID}"
        assert results[0].success is False
        assert results[0].annotated_text == ""
        assert results[0].entities == []
        assert "API overloaded" in results[0].error_message

    def test_parse_error(
        self,
        processor: Any,
    ) -> None:
        """Test parse error produces failed result."""
        responses = [
            BatchResponse(
                custom_id=f"record_0_{BINDNR}_{BREVID}",
                response_text="not parseable\n===JSON===\n{invalid}",
                success=True,
            ),
        ]
        results = processor._build_batch_results([VALID_RECORD], responses)
        log.debug("Built batch results: %s", results)
        assert len(results) == 1
        assert results[0].brevid == BREVID
        assert results[0].record_id == f"record_0_{BINDNR}_{BREVID}"
        assert results[0].success is False
        assert results[0].annotated_text == ""
        assert results[0].entities == []
        assert "Failed to parse LLM response" in results[0].error_message
        assert "Invalid JSON format" in results[0].error_message

    def test_order_preserved_with_shuffled_responses(
        self,
        processor: Any,
    ) -> None:
        """Test results follow original record order even when responses arrive out of order."""
        records = [
            {"Bindnr": "1", "Brevid": "601", "Tekst": "text A"},
            {"Bindnr": "1", "Brevid": "602", "Tekst": "text B"},
            {"Bindnr": "1", "Brevid": "603", "Tekst": "text C"},
        ]
        # Responses arrive out of order (index 2, 0, 1)
        responses = [
            BatchResponse(
                custom_id="record_2_1_603",
                response_text=SAMPLE_LLM_RESPONSE,
                success=True,
            ),
            BatchResponse(
                custom_id="record_0_1_601",
                response_text=SAMPLE_LLM_RESPONSE,
                success=True,
            ),
            BatchResponse(
                custom_id="record_1_1_602",
                response_text=SAMPLE_LLM_RESPONSE,
                success=True,
            ),
        ]
        results = processor._build_batch_results(records, responses)

        log.debug("Results: %s", results)
        assert len(results) == 3
        # Results must follow original record order, not response order
        assert results[0].brevid == "601"
        assert results[1].brevid == "602"
        assert results[2].brevid == "603"
        assert all(r.success for r in results)


# ===================================================================
# Process single batch response
# ===================================================================
class TestProcessSingleBatchResponse:
    """Tests for RecordProcessor._process_single_batch_response()."""

    def test_success(
        self,
        processor: Any,
    ) -> None:
        """Test successful single batch response processing."""
        response = BatchResponse(
            custom_id=f"record_0_{BINDNR}_{BREVID}",
            response_text=SAMPLE_LLM_RESPONSE,
            success=True,
        )
        result = processor._process_single_batch_response(
            0,
            VALID_RECORD,
            response,
        )
        log.debug("Processed batch response result: %s", result)
        assert result.record_id == f"record_0_{BINDNR}_{BREVID}"
        assert result.brevid == BREVID
        assert result.success is True
        assert result.annotated_text == f'{BINDNR};{BREVID};"{ANNOTATED_TEXT}"'
        assert result.entities[0] == EXPECTED_ENTITY

    @pytest.mark.parametrize(
        ("response", "expected_error"),
        [
            (None, "No response received for record"),
            (
                BatchResponse(
                    custom_id=f"record_0_{BINDNR}_{BREVID}",
                    response_text="",
                    success=False,
                    error_message="Server error",
                ),
                "Server error",
            ),
            (
                BatchResponse(
                    custom_id=f"record_0_{BINDNR}_{BREVID}",
                    response_text="not parseable\n===JSON===\n{invalid json}",
                    success=True,
                ),
                "Failed to parse LLM response",
            ),
        ],
        ids=["no_response", "failed_response", "parse_exception"],
    )
    def test_failure_cases(
        self,
        processor: Any,
        response: BatchResponse | None,
        expected_error: str,
    ) -> None:
        """Test failure scenarios produce failed results with expected errors."""
        result = processor._process_single_batch_response(
            0,
            VALID_RECORD,
            response,
        )
        log.debug("Processed batch response result: %s", result)
        assert result.record_id == f"record_0_{BINDNR}_{BREVID}"
        assert result.brevid == BREVID
        assert result.success is False
        assert result.annotated_text == ""
        assert result.entities == []
        assert expected_error in result.error_message


# ===================================================================
# Create response map
# ===================================================================
class TestCreateResponseMap:
    """Tests for RecordProcessor._create_response_map()."""

    def test_success(
        self,
        processor: Any,
    ) -> None:
        """Test valid responses are mapped by index."""
        responses = [
            BatchResponse(
                custom_id="record_0_1_601",
                response_text=SAMPLE_LLM_RESPONSE,
                success=True,
            ),
            BatchResponse(
                custom_id="record_1_1_602",
                response_text=SAMPLE_LLM_RESPONSE,
                success=True,
            ),
        ]
        response_map = processor._create_response_map(responses)
        log.debug("Created response map: %s", response_map)
        assert len(response_map) == 2
        assert 0 in response_map
        assert response_map[0].custom_id == "record_0_1_601"
        assert 1 in response_map
        assert response_map[1].custom_id == "record_1_1_602"

    def test_invalid_custom_id_skips(
        self,
        processor: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test invalid custom_id is skipped with warning."""
        responses = [
            BatchResponse(
                custom_id="invalid_id",
                response_text=SAMPLE_LLM_RESPONSE,
                success=False,
                error_message="Invalid custom_id format",
            )
        ]
        with caplog.at_level(logging.WARNING):
            response_map = processor._create_response_map(responses)

        log.debug("Created response map: %s", response_map)
        log.debug("Captured logs: %s", caplog.text)

        assert len(response_map) == 0
        assert "Could not parse index from custom_id" in caplog.text

    def test_duplicate_index_keeps_last(
        self,
        processor: Any,
    ) -> None:
        """Test duplicate indices are overwritten; last response wins."""
        first = BatchResponse(
            custom_id="record_0_1_601",
            response_text="first",
            success=True,
        )
        second = BatchResponse(
            custom_id="record_0_1_601",
            response_text="second",
            success=True,
        )
        response_map = processor._create_response_map([first, second])

        log.debug("Response map: %s", response_map)
        assert len(response_map) == 1
        assert response_map[0].response_text == "second"

    def test_mixed_valid_and_invalid(
        self,
        processor: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test valid responses are kept while invalid ones are skipped."""
        responses = [
            BatchResponse(
                custom_id="record_0_1_601",
                response_text=SAMPLE_LLM_RESPONSE,
                success=True,
            ),
            BatchResponse(
                custom_id="bad_id",
                response_text="",
                success=False,
                error_message="Bad ID",
            ),
            BatchResponse(
                custom_id="record_2_1_603",
                response_text=SAMPLE_LLM_RESPONSE,
                success=True,
            ),
        ]
        with caplog.at_level(logging.WARNING):
            response_map = processor._create_response_map(responses)

        log.debug("Response map: %s", response_map)
        assert len(response_map) == 2
        assert 0 in response_map
        assert 2 in response_map
        assert "Could not parse index from custom_id" in caplog.text


# ===================================================================
# Static helpers
# ===================================================================
class TestStaticHelpers:
    """Tests for static helper methods on RecordProcessor."""

    @pytest.mark.parametrize(
        ("brevids", "max_display", "expected"),
        [
            (["601", "602"], 3, "BATCH-601-602"),
            (["601", "602", "603"], 3, "BATCH-601-602-603"),
            (["601", "602", "603", "604", "605"], 3, "BATCH-601-602-603..."),
        ],
        ids=["short_list", "exact_max", "long_list_truncates"],
    )
    def test_create_batch_id(
        self,
        brevids: list[str],
        max_display: int,
        expected: str,
    ) -> None:
        """Test batch ID generation with varying list lengths."""
        result = _create_batch_id(brevids, max_display=max_display)
        log.debug("Created batch ID: %s", result)
        assert result == expected

    def test_create_record_id(self) -> None:
        """Test record ID creation."""
        result = RecordProcessor.create_record_id("1", "601")
        log.debug("Created record ID: %s", result)
        assert result == "1_601"

    def test_create_custom_id(self) -> None:
        """Test custom ID creation."""
        result = _create_custom_id(0, "1", "601")
        log.debug("Created custom ID: %s", result)
        assert result == "record_0_1_601"

    def test_extract_index_from_custom_id_success(self) -> None:
        """Test extracting index from valid custom ID."""
        result = _extract_index_from_custom_id("record_0_1_601")
        log.debug("Extracted index: %s", result)
        assert result == 0

    @pytest.mark.parametrize(
        ("custom_id", "match_text"),
        [
            ("test_0_1_601", "Invalid custom_id format"),
            ("record", "Invalid custom_id format"),
            ("record_", "Could not extract index"),
            ("record_abc_1_601", "Could not extract index"),
        ],
        ids=["invalid_prefix", "single_part", "empty_index", "non_numeric"],
    )
    def test_extract_index_from_custom_id_raises(
        self,
        custom_id: str,
        match_text: str,
    ) -> None:
        """Test invalid custom_id inputs raise ValueError."""
        with pytest.raises(ValueError, match=match_text) as exc_info:
            _extract_index_from_custom_id(custom_id)

        log.debug("Caught exception: %s", exc_info.value)
        assert match_text in str(exc_info.value)


# ===================================================================
# Create processing result
# ===================================================================
class TestCreateProcessingResult:
    """Test for _create_processing_result() helper function."""

    def test_success(self) -> None:
        """Test creating a success ProcessingResult."""
        entity = EntityRecord(
            name=VALID_ENTITY_DATA["name"],
            entity_type=VALID_ENTITY_DATA["type"],
            preposition=VALID_ENTITY_DATA["preposition"],
            order=VALID_ENTITY_DATA["order"],
            brevid="601",
            description=VALID_ENTITY_DATA["description"],
            gender=VALID_ENTITY_DATA["gender"],
            language=VALID_ENTITY_DATA["language"],
        )
        result = _create_processing_result(
            record_id="record_0_1_601",
            brevid=BREVID,
            success=True,
            processing_time=1.23,
            annotated_text=f'{BINDNR};{BREVID};"{ANNOTATED_TEXT}"',
            entities=[entity],
        )

        log.debug("Created ProcessingResult: %s", result)
        assert result.record_id == "record_0_1_601"
        assert result.brevid == BREVID
        assert result.success is True
        assert result.processing_time == 1.23
        assert result.annotated_text == f'{BINDNR};{BREVID};"{ANNOTATED_TEXT}"'
        assert result.entities == [entity]

    def test_failure(self) -> None:
        """Test creating a failure ProcessingResult."""
        result = _create_processing_result(
            record_id="record_0_1_601",
            brevid=BREVID,
            success=False,
            error_msg="LLM call failed",
        )

        log.debug("Created ProcessingResult: %s", result)
        assert result.record_id == "record_0_1_601"
        assert result.brevid == BREVID
        assert result.success is False
        assert result.error_message == "LLM call failed"
        assert result.annotated_text == ""
        assert result.entities == []

    def test_success_missing_annotated_text_raises(self) -> None:
        """Test ValueError when success=True but annotated_text is None."""
        with pytest.raises(ValueError, match="annotated_text is required") as exc_info:
            _create_processing_result(
                record_id="record_0_1_601",
                brevid=BREVID,
                success=True,
                entities=[EXPECTED_ENTITY],
            )

        log.debug("Caught exception: %s", exc_info.value)
        assert "annotated_text is required when success=True" in str(exc_info.value)

    def test_success_missing_entities_raises(self) -> None:
        """Test ValueError when success=True but entities is None."""
        with pytest.raises(ValueError, match="entities is required") as exc_info:
            _create_processing_result(
                record_id="record_0_1_601",
                brevid=BREVID,
                success=True,
                annotated_text=f'{BINDNR};{BREVID};"{ANNOTATED_TEXT}"',
            )

        log.debug("Caught exception: %s", exc_info.value)
        assert "entities is required when success=True" in str(exc_info.value)


# ===================================================================
# Process individual async (fallback)
# ===================================================================
class TestProcessIndividualAsync:
    """Test for RecordProcessor._process_individual_async()."""

    @pytest.mark.asyncio
    async def test_success(
        self,
        processor: Any,
    ) -> None:
        """Test all records processed successfully."""
        records = [VALID_RECORD, {**VALID_RECORD, "Brevid": "602"}]
        result = await processor._process_individual_async(
            records,
            batch_num=1,
        )

        log.debug("Individual async processing result: %s", result)
        assert isinstance(result, BatchProcessingResult)
        assert result.batch_id == "batch_1"
        assert len(result.results) == 2
        assert result.results[0].brevid == BREVID
        assert result.results[1].brevid == "602"
        assert result.successful_count == 2
        assert result.failed_count == 0

    @pytest.mark.asyncio
    async def test_with_exceptions(
        self,
        processor: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test records that raise exceptions are handled as failures."""
        processor.llm_client.call_async = mocker.AsyncMock(
            side_effect=LLMClientError("LLM failure", operation="call_async")
        )

        result = await processor._process_individual_async(
            [VALID_RECORD],
            batch_num=1,
        )

        log.debug("Individual async processing result: %s", result)
        assert result.successful_count == 0
        assert result.failed_count == 1
        assert result.batch_id == "batch_1"
        assert result.results[0].brevid == BREVID
        assert result.results[0].success is False
        assert "LLM failure" in result.results[0].error_message

    @pytest.mark.asyncio
    async def test_mixed_results(self, processor: Any, mocker: MockerFixture) -> None:
        """Test mixed successful and failed results."""
        # First call succeeds, second raises
        processor.llm_client.call_async = mocker.AsyncMock(
            side_effect=[
                SAMPLE_LLM_RESPONSE,
                LLMClientError("LLM failure", operation="call_async"),
            ]
        )
        records = [VALID_RECORD, {**VALID_RECORD, "Brevid": "602"}]
        result = await processor._process_individual_async(records, batch_num=1)

        log.debug("Result: %s", result)

        assert result.successful_count == 1
        assert result.failed_count == 1
        assert result.batch_id == "batch_1"
        assert result.results[0].brevid == BREVID
        assert result.results[0].success is True
        assert result.results[1].brevid == "602"
        assert result.results[1].success is False

    @pytest.mark.asyncio
    async def test_raw_exception_from_gather(
        self,
        processor: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test raw Exception from asyncio.gather is converted to failed result."""

        # Monkeypatch process_record_async to raise directly (bypassing its own handlers)
        async def _raise_exception(_record: dict[str, str]) -> ProcessingResult:
            raise RuntimeError("unhandled crash")

        monkeypatch.setattr(processor, "process_record_async", _raise_exception)
        result = await processor._process_individual_async([VALID_RECORD], batch_num=1)
        log.debug("Individual async processing result: %s", result)

        assert result.successful_count == 0
        assert result.failed_count == 1
        assert result.batch_id == "batch_1"
        assert result.results[0].brevid == BREVID
        assert result.results[0].success is False
        assert "unhandled crash" in result.results[0].error_message


# ===================================================================
# create_progress_logger
# ===================================================================
class TestCreateProgressLogger:
    """Tests for create_progress_logger() utility function."""

    def test_logs_at_interval(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test progress is logged when interval has elapsed."""
        # the variable last_log_time starts at 0.0, so first call must be > log_interval
        # to trigger logging: 61.0 - 0.0 > 60.0 → True
        monkeypatch.setattr(
            "ai_ner_system.processing.processor.time.monotonic",
            lambda: 61.0,
        )

        progress = BatchProgress(
            batch_num=1,
            batch_id="batch_1",
            status=BatchStatus.IN_PROGRESS,
            elapsed_time=61.0,
            request_counts={"processing": 5, "succeeded": 3, "errored": 0},
            created_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-02T00:00:00Z",
        )

        logger_func = create_progress_logger(log_interval=60.0)
        with caplog.at_level(logging.INFO):
            logger_func(progress)

        log.debug("Captured logs: %s", caplog.text)

        assert "batch_1" in caplog.text
        assert "Processing: 5" in caplog.text
        assert "Succeeded: 3" in caplog.text

    def test_suppresses_within_interval(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test progress is suppressed when within interval."""
        # Both calls within the interval: t=100, t=110 (only 10s apart)
        time_values = iter([100.0, 110.0])
        monkeypatch.setattr(
            "ai_ner_system.processing.processor.time.monotonic",
            lambda: next(time_values),
        )

        progress = BatchProgress(
            batch_num=1,
            batch_id="batch_1",
            status=BatchStatus.IN_PROGRESS,
            elapsed_time=10.0,
            request_counts={"processing": 5, "succeeded": 3, "errored": 0},
            created_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-02T00:00:00Z",
        )

        logger_func = create_progress_logger(log_interval=60.0)
        with caplog.at_level(logging.INFO):
            logger_func(progress)  # first call at t=100 → logs (100 - 0 > 60)
            logger_func(progress)  # second call at t=110 → suppressed (110 - 100 < 60)

        # Assert before debug-logging caplog.text to avoid self-referential capture
        assert caplog.text.count("Batch 1") == 1
        log.debug("Captured logs: %s", caplog.text)
