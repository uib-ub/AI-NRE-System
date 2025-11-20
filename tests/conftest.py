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
import os
from typing import TYPE_CHECKING

import pytest

import ai_ner_system.config.settings as settings_mod
from ai_ner_system.config.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

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
        force=True,
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
        "1;601;Ollum monnum þæim sæm þetta bref sea æder høyra sændir Olauer med gudz nadh abote j Olafsklaustre j Tunsbergi q. g. ok sina kunnikt gerande at ek hæfuir samþykt med þesso mino opno brefue a Olafsklausters vægna þet jardarkaup sæm sira Æirikar Kolbiornason prester a Sondum hafde kœypt af Haluarde Þorgæirssyni ij marka bool j sydra Strandh er ligger j Sanda sokn a Væstfoldh æfter þui sæm jngifta bref vattar at Haluarder ok hans kona gafuo sik jn j sancte Olafsklauster med allu þui sem þau atto bæde j lauso ok fasto. serdæilis kenniz ek ok at ek hafuer vpboret af fyrnæmdom sira Æirike ij half stykki klædes sæm æfter stodo af jardar verdino ok ofuan a þet gaf han mik ok afhende a klaustrens vægna .j. half stykki j ifuir giof firir fyrnæmda jordh. Ok til sanynda her vm sætto þesser goder men er sua æita ok ner waro fyrnemdo giærdh ok samþykt sira Halbiorn Biornsson profaster j Tunsbergi. sira Hakon Gudþormsson prester j Nioterøy sin insigli med mino firir þetta bref er giort war j Raudenom j Tunsbergi a Botolf vaku æftan anno domini mo cccco vo ok a xvj are rikis wars wyrduliks herra herra Eriks med gudz nadh Noreks Dana ok Guta konongs.\n"
        "1;604;Veer Eskill meder gudes naad erchibiskuper j Nidaros kungerom allom mannom þæim sem þetta bref sea eda høyra at þet var skilordh j kaupmaala vaarom ok velborens manz Hac[onar Sigurdz] sonar vm æighner þær sem han hefuer os ok vaare kirkiu pansæt j Sennione ok j Trumpsar kirkiu sokn [firir] fiortaan læster skræidar til gilldz huoriar ver hafuum os ok vaara epterkomanda vnderbundit meder vaaro opno brefue at luca Titeke sæmm eda hans erfuingiom. jnnan þriu aar her epter a Haconar Sigurdzsonar væghna. at sidan ver ok vaar kirkia eda efterkomanda hafuum med fulnade. jam marghar læster fisk til gildz oc kostnadh varn mæder apter vpboret af hans Haconar æignom fyrnemdom. þa scula þær sama æighner allar. vera firir os oc vaare kirkiu. eda varom efterkomandom .quittar oc lydughar ok allungis aakiæralausar. en Hacone eder hans erfuingiom. frealsar oc heimhollar. til alz afrædes. sosom bref hans vaattar sæm ver hafuum vm fyrsagdan kaupmaala. Til meire vissu her vm. sættom ver vaart secretum firir þetta bref ær gort war j Berguin vigilia beati Bartholomei apostoli. anno domini millesimo. quadringentesimo quinto.\n"
        "1;611;Ollom monnom þeim sæm þetta bref sea ædher høyra sænda Hakon Amundason ok Arne Drængsson quædiu gudz ok sina kunnikt gerande at mit hafuum sælt Ælifui ok Alfue Olafssonom mærka bool j Lundaby sæm ligger j Sanda sokn j Sææms bygd sæm Sanda kirkiæ atte till vpbygninga mædh ollom lutum ok lunnyndom sæm till liggia ædher leghet (hafua) fra forno ok nyghiu jnnan gardz ok vttan frialst ok hæimholt ok akiæralaust firir huariom manne. kænnomzst mit at mit hafuum vpboret af fyrnæmfdom brødrom fyrsta pening ok øfsta ok alla þer j mellom æfter þui sæm j kaup vart kom sua at okker væl atnøgde. Ok till sannynda settom mit okor incigli firir þetta bref er gort var a Berghom j Sanda sokn a Blasius messo dagh a xviii aare okkars vyrdaligs herræ herræ Eriks mædh guds naadh Noregs konongs.\n",
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
        "CLAUDE_MODEL": "claude-sonnet-4",
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
        "OLLAMA_MODEL": "gemma3:12b",
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
