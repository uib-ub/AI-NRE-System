"""Pytest configuration and shared fixtures.

This module provides core test infrastructure including:
- Session-level logging configuration
- Environment isolation (autouse)
- Settings singleton reset (autouse)
- Custom marker registration

Unit-test-specific fixtures live in ``tests/unit/conftest.py``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest

from ai_ner_system.config.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ai_ner_system.config.settings import Settings as SettingsType


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging() -> None:
    """Configure logging for test execution.

    Respects PYTEST_LOG_LEVEL (default INFO). Uses `force=True` to override
    any prior logging config from libraries/plugins.
    """
    level = os.getenv("PYTEST_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from external libraries
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


_ENV_KEYS_TO_CLEAR: tuple[str, ...] = (
    # API keys / tokens
    "ANTHROPIC_API_KEY",
    "OPENWEBUI_TOKEN",
    # Models
    "CLAUDE_MODEL",
    "OLLAMA_MODEL",
    # Endpoints
    "OPENWEBUI_ENDPOINT",
    # File/dir paths
    "INPUT_FILE",
    "OUTPUT_TEXT_FILE",
    "OUTPUT_TABLE_FILE",
    "OUTPUT_STATS_FILE",
    "PROMPT_TEMPLATE_FILE",
    "BATCH_TEMPLATE_FILE",
    "CACHE_DIR",
)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate test environment from system settings.

    Clears potentially interfering environment variables to ensure
    deterministic test behavior across different development machines.
    This runs BEFORE test-specific fixtures due to autouse.

    Args:
        monkeypatch: Pytest fixture for modifying environment.
    """
    for key in _ENV_KEYS_TO_CLEAR:
        monkeypatch.delenv(key, raising=False)


def _get_settings() -> type[SettingsType]:
    """Get Settings class for fixture use.

    Helper function to provide Settings class reference to fixtures.
    """
    return Settings


@pytest.fixture(autouse=True)
def reset_settings() -> Iterator[None]:
    """Reset Settings singleton before and after each test.

    Ensures test isolation by clearing any configuration state.
    Prevents test pollution from environment variables or previous runs.
    """
    settings_class = _get_settings()
    settings_class.reset()
    try:
        yield
    finally:
        settings_class.reset()


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers and options.

    Args:
        config: Pytest configuration object.
    """
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (slower)",
    )
    config.addinivalue_line(
        "markers",
        "system: mark test as system/acceptance test (slowest)",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow-running",
    )
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as async test requiring asyncio",
    )
