"""Pytest configuration and shared fixtures.

This module provides test infrastructure including:
- Session-level logging configuration
- Common fixtures for temporary files and directories
- Settings reset automation
- Helper utilities for test data generation

Uses Python 3.11+ features for better async test support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

import ai_ner_system.config.settings as settings_mod
from ai_ner_system.config.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ai_ner_system.config.settings import Settings as SettingsType


def _get_settings() -> type[SettingsType]:
    """Get Settings class for fixture use.

    Helper function to provide Settings class reference to fixtures.
    """
    return Settings


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging() -> None:
    """Configure logging for test execution.

    Sets up DEBUG-level logging with timestamps for all test runs.
    Helps diagnose test failures and understand execution flow.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from external libraries
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


@pytest.fixture(autouse=True)
def reset_settings() -> Iterator[None]:
    """Reset Settings singleton before and after each test.

    Ensures test isolation by clearing any configuration state.
    Prevents test pollution from environment variables or previous runs.
    """
    settings_class = _get_settings()
    settings_class.reset()
    yield
    settings_class.reset()


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate test environment from system settings.

    Clears potentially interfering environment variables to ensure
    deterministic test behavior across different development machines.
    This runs BEFORE test-specific fixtures due to autouse.

    Args:
        monkeypatch: Pytest fixture for modifying environment.
    """
    # Clear API keys to prevent accidental real API calls
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENWEBUI_TOKEN", raising=False)

    # Clear model settings
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    # Clear endpoint settings
    monkeypatch.delenv("OPENWEBUI_ENDPOINT", raising=False)

    # Clear file paths
    monkeypatch.delenv("INPUT_FILE", raising=False)
    monkeypatch.delenv("OUTPUT_TEXT_FILE", raising=False)
    monkeypatch.delenv("OUTPUT_TABLE_FILE", raising=False)
    monkeypatch.delenv("OUTPUT_STATS_FILE", raising=False)
    monkeypatch.delenv("PROMPT_TEMPLATE_FILE", raising=False)
    monkeypatch.delenv("BATCH_TEMPLATE_FILE", raising=False)
    monkeypatch.delenv("CACHE_DIR", raising=False)


@pytest.fixture
def no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable load_dotenv during test."""

    def _no_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False

    monkeypatch.setattr(settings_mod, "load_dotenv", _no_dotenv, raising=True)


@pytest.fixture
def tmp_input_file(tmp_path: Path) -> Path:
    """Create a temporary CSV input file with sample medieval text data.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to created input file with header and sample records.
    """
    input_file = tmp_path / "test_input.txt"
    input_file.write_text(
        "Bindnr;Brevid;Tekst\n"
        "B001;001;Henrik av Norge var konge i 1200-tallet.\n"
        "B002;002;Oslo og Bergen er viktige byer.\n"
        "B003;003;Håkon Håkonsson regjerte fra Nidaros.\n",
        encoding="utf-8",
    )
    return input_file


@pytest.fixture
def mock_env_claude(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set up Claude client environment variables.

    Args:
        monkeypatch: Pytest fixture for modifying environment.

    Returns:
        Dictionary of set environment variables.
    """
    env_vars = {
        "ANTHROPIC_API_KEY": "sk-ant-test-key-123456789",
        "CLAUDE_MODEL": "claude-3-opus-20240229",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def mock_env_ollama(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set up Ollama client environment variables.

    Args:
        monkeypatch: Pytest fixture for modifying environment.

    Returns:
        Dictionary of set environment variables.
    """
    env_vars = {
        "OPENWEBUI_ENDPOINT": "http://localhost:11434",
        "OPENWEBUI_TOKEN": "test-token-123",
        "OLLAMA_MODEL": "gemma2:27b",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


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
