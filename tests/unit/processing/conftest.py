"""Shared constants, sample data, and fixtures for processing module tests.

Provides domain-relevant medieval NER test data used across multiple
test files (test_processor, test_parser, test_validator, etc.).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from ai_ner_system.llm.base_client import Client
from ai_ner_system.processing.processor import RecordProcessor
from ai_ner_system.prompt.builder import PromptBuilder

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# ---------------------------------------------------------------------------
# Shared constants / sample data
# ---------------------------------------------------------------------------

BREVID = "601"
BINDNR = "1"

VALID_RECORD: dict[str, str] = {
    "Bindnr": BINDNR,
    "Brevid": BREVID,
    "Tekst": "Ollum monnum þæim sæm þetta bref sea æder høyra sændir Olauer med gudz",
}

VALID_ENTITY_DATA: dict[str, Any] = {
    "name": "Olauer",
    "type": "Person Name",
    "preposition": "N/A",
    "order": 1,
    "description": "Abbot",
    "gender": "Male",
    "language": "non",
}

VALID_ENTITIES_JSON = json.dumps({"entities": [VALID_ENTITY_DATA]})

ANNOTATED_TEXT = (
    "Ollum monnum þæim sæm þetta bref sea æder høyra sændir"
    " < Olauer;Person Name;N/A;1;601 > med gudz"
)

SAMPLE_LLM_RESPONSE = f"{ANNOTATED_TEXT}\n===JSON===\n{VALID_ENTITIES_JSON}"

GENERATED_PROMPT = "Generated prompt for testing"


# ---------------------------------------------------------------------------
# Fixtures (processor-level, available to all processing tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_client(mocker: MockerFixture) -> Any:
    """Create a mock LLM client with spec=Client."""
    client = mocker.MagicMock(spec=Client)
    client.call.return_value = SAMPLE_LLM_RESPONSE
    client.call_async = mocker.AsyncMock(return_value=SAMPLE_LLM_RESPONSE)
    client.supports_async_batch.return_value = False
    client.ERROR_RESPONSE_SENTINELS = frozenset(
        {"Claude API call failed", "Ollama API call failed"}
    )
    return client


@pytest.fixture
def mock_prompt_builder(mocker: MockerFixture) -> Any:
    """Create a mock PromptBuilder."""
    builder = mocker.MagicMock(spec=PromptBuilder)
    builder.build.return_value = GENERATED_PROMPT
    return builder


@pytest.fixture
def processor(mock_llm_client: Any, mock_prompt_builder: Any) -> Any:
    """Create a RecordProcessor with mocked dependencies."""
    return RecordProcessor(mock_llm_client, mock_prompt_builder)
