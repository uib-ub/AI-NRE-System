"""Unit tests for main_processor wiring, output, and application policies."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import pytest

from ai_ner_system.pipeline import main_processor as main_processor_module

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture, MockType

# Tests should not accidentally load .env configuration
pytestmark = pytest.mark.usefixtures("no_dotenv")


@dataclass
class MainProcessorDependencies:
    """Mocked constructor boundaries and their configured return values.

    "factory" means something called to obtain another object.
    """

    client_factory: MockType
    prompt_factory: MockType
    processor_factory: MockType
    reader_factory: MockType
    writer_factory: MockType
    path_factory: MockType

    @property
    def writer(self) -> MockType:
        """Return the writer mock installed by the real constructor."""
        return cast("MockType", self.writer_factory.return_value)


class MainProcessorFactory(Protocol):
    """Keyword-only factory for constructing the real main processor.

    This protocol describes the interface of the factory returned later.
    __call__ describes calling an object like a function.
    """

    def __call__(
        self,
        *,
        async_mode: bool = False,
        incremental_output: bool = False,
        use_batch: bool = False,
    ) -> main_processor_module.MedievalTextProcessor:
        """Construct a processor with independent command-line arguments."""
        ...


@pytest.fixture
def processor_args(tmp_path: Path) -> Namespace:
    """Provide the command-line fields consumed by the main processor."""
    return Namespace(
        client="claude",
        input=str(tmp_path / "records.csv"),
        output_text=str(tmp_path / "annotations.txt"),
        output_table=str(tmp_path / "metadata.txt"),
        output_stats=str(tmp_path / "stats.json"),
        prompt_template=str(tmp_path / "single-template.txt"),
        batch_template=str(tmp_path / "batch-template.txt"),
        use_batch=False,
        batch_size=2,
        async_mode=False,
        incremental_output=False,
    )


@pytest.fixture
def main_dependencies(mocker: MockerFixture) -> MainProcessorDependencies:
    """Replace external constructors without bypassing main initialization."""
    dependencies = MainProcessorDependencies(
        client_factory=mocker.patch.object(
            main_processor_module, "create_llm_client", autospec=True
        ),
        prompt_factory=mocker.patch.object(
            main_processor_module, "GenericPromptBuilder", autospec=True
        ),
        processor_factory=mocker.patch.object(
            main_processor_module, "RecordProcessor", autospec=True
        ),
        reader_factory=mocker.patch.object(
            main_processor_module, "CSVReader", autospec=True
        ),
        writer_factory=mocker.patch.object(
            main_processor_module, "OutputWriter", autospec=True
        ),
        path_factory=mocker.patch.object(main_processor_module, "Path", autospec=True),
    )
    dependencies.path_factory.return_value.exists.return_value = True
    return dependencies


@pytest.fixture
def make_main_processor(
    processor_args: Namespace,
    main_dependencies: MainProcessorDependencies,
) -> MainProcessorFactory:
    """Return a local factory that executes the real main constructor."""

    def _make(
        *,
        async_mode: bool = False,
        incremental_output: bool = False,
        use_batch: bool = False,
    ) -> main_processor_module.MedievalTextProcessor:
        args = Namespace(**vars(processor_args))
        args.async_mode = async_mode
        args.incremental_output = incremental_output
        args.use_batch = use_batch
        processor = main_processor_module.MedievalTextProcessor(args)
        assert processor.writer is main_dependencies.writer_factory.return_value
        return processor

    return _make
