"""Request-shape tests for ClaudeClient that run through the real Anthropic SDK.

The other Claude tests replace ``Anthropic`` and ``AsyncAnthropic`` with mocks,
so they cannot notice when the SDK stops accepting an argument. These tests
keep the real SDK classes and replace only the HTTP transport, so the SDK's own
argument checking and request serialization run. Nothing touches the network.

Tests cover:
- sync, async and batch requests reach the transport and carry no sampling keys
- every key of the request payload is a parameter the installed SDK accepts
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.resources.messages import Messages

from ai_ner_system.llm.batch_models import BatchRequest
from ai_ner_system.llm.claude_client import ClaudeClient

# Synthetic values; no real credential or model is involved.
TEST_API_KEY = "sk-ant-test-key-123456789"
TEST_MODEL = "claude-test-model"
BASE_URL = "https://api.anthropic.com"
SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k"})
RESPONSE_TEXT = 'annotated text\n===JSON===\n{"entities": []}'


def message_json(text: str = RESPONSE_TEXT) -> dict[str, Any]:
    """Return a minimal Messages API response body."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": TEST_MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }


def batch_json() -> dict[str, Any]:
    """Return a minimal Message Batches API creation response body."""
    return {
        "id": "msgbatch_test",
        "type": "message_batch",
        "processing_status": "in_progress",
        "request_counts": {
            "processing": 1,
            "succeeded": 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        },
        "created_at": "2026-01-01T00:00:00Z",
        "expires_at": "2026-01-02T00:00:00Z",
        "ended_at": None,
        "archived_at": None,
        "cancel_initiated_at": None,
        "results_url": None,
    }


class RecordingTransport:
    """HTTP handler that records requests and answers like the Claude API.

    Attributes:
        requests: Every request the handler received, in order.
    """

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        """Record the request and return a canned response for its path."""
        self.requests.append(request)
        if request.url.path.endswith("/batches"):
            return httpx2.Response(200, json=batch_json())
        return httpx2.Response(200, json=message_json())

    def body(self, index: int = 0) -> dict[str, Any]:
        """Return the JSON body of a recorded request."""
        loaded: dict[str, Any] = json.loads(self.requests[index].content)
        return loaded


@pytest.fixture
def transport() -> RecordingTransport:
    """Provide a fresh recording transport."""
    return RecordingTransport()


@pytest.fixture
def claude_http(transport: RecordingTransport) -> ClaudeClient:
    """Create a ClaudeClient whose real SDK clients use the fake transport."""
    client = ClaudeClient(api_key=TEST_API_KEY, model=TEST_MODEL)
    client.client = anthropic.Anthropic(
        api_key=TEST_API_KEY,
        base_url=BASE_URL,
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(
            transport=httpx2.MockTransport(transport),
        ),
    )
    client.async_client = anthropic.AsyncAnthropic(
        api_key=TEST_API_KEY,
        base_url=BASE_URL,
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(
            transport=httpx2.MockTransport(transport),
        ),
    )
    return client


class TestClaudeRequestShape:
    """The requests ClaudeClient builds are accepted by the installed SDK."""

    def test_call_reaches_transport_without_sampling_parameters(
        self,
        claude_http: ClaudeClient,
        transport: RecordingTransport,
    ) -> None:
        """A sync call passes SDK validation and sends no sampling keys."""
        text = claude_http.call("Test prompt")

        assert text == RESPONSE_TEXT
        assert len(transport.requests) == 1
        assert transport.requests[0].url.path == "/v1/messages"
        body = transport.body()
        assert body["model"] == TEST_MODEL
        assert body["max_tokens"] == ClaudeClient.MAX_ALLOWED_TOKENS
        assert body["messages"] == [{"role": "user", "content": "Test prompt"}]
        assert SAMPLING_KEYS.isdisjoint(body)

    @pytest.mark.asyncio
    async def test_call_async_reaches_transport_without_sampling_parameters(
        self,
        claude_http: ClaudeClient,
        transport: RecordingTransport,
    ) -> None:
        """An async call passes SDK validation and sends no sampling keys."""
        text = await claude_http.call_async("Test prompt")

        assert text == RESPONSE_TEXT
        assert len(transport.requests) == 1
        assert SAMPLING_KEYS.isdisjoint(transport.body())

    @pytest.mark.asyncio
    async def test_create_batch_sends_no_sampling_parameters(
        self,
        claude_http: ClaudeClient,
        transport: RecordingTransport,
    ) -> None:
        """Each request in a provider batch is free of sampling keys."""
        batch_id = await claude_http.create_batch_async(
            [
                BatchRequest(custom_id="record_0_1_601", prompt="Prompt A"),
                BatchRequest(custom_id="record_1_1_602", prompt="Prompt B"),
            ],
        )

        assert batch_id == "msgbatch_test"
        assert transport.requests[0].url.path == "/v1/messages/batches"
        requests = transport.body()["requests"]
        assert [r["custom_id"] for r in requests] == [
            "record_0_1_601",
            "record_1_1_602",
        ]
        for request in requests:
            assert SAMPLING_KEYS.isdisjoint(request["params"])
            assert request["params"]["model"] == TEST_MODEL

    def test_payload_keys_are_accepted_by_installed_sdk(
        self,
        claude_http: ClaudeClient,
    ) -> None:
        """Every payload key is a parameter of the SDK's ``messages.create``.

        This fails as soon as an SDK upgrade removes an argument the client
        still sends, which is how the sampling parameters broke unnoticed.
        """
        payload = claude_http._message_payload("Test prompt")
        accepted = set(inspect.signature(Messages.create).parameters)

        assert set(payload) <= accepted
