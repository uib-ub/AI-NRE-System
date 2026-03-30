"""Protocol definition for processor dependencies.

This module defines the interface that async and sync processors depend on,
breaking the circular import between main_processor and the sub-processors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    from argparse import Namespace

    from ai_ner_system.file_io import CSVReader, OutputWriter
    from ai_ner_system.llm import Client
    from ai_ner_system.processing import RecordProcessor


class ProcessorContext(Protocol):
    """Protocol defining the interface that processors need from main processor.

    This protocol breaks the circular dependency by defining only the interface
    that AsyncProcessor and SyncProcessor actually use, without importing
    MedievalTextProcessor directly.

    Note: Properties must be declared with @property decorator in the Protocol.
    """

    # Constants needed by processors
    ANNOTATED_HEADER: ClassVar[str]
    METADATA_HEADER: ClassVar[str]

    # Instance attributes
    args: Namespace
    reader: CSVReader
    writer: OutputWriter
    processor: RecordProcessor
    llm_client: Client
    incremental_mode: bool

    # Output file path properties (resolved from CLI args or Settings)
    @property
    def output_text_file(self) -> str:  # pyright: ignore[reportReturnType]
        """Output text file path."""

    @property
    def output_table_file(self) -> str:  # pyright: ignore[reportReturnType]
        """Output table file path."""

    @property
    def output_stats_file(self) -> str:  # pyright: ignore[reportReturnType]
        """Output stats file path."""
