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

    Note: These can be either instance attributes or properties - Protocol accepts both.
    """

    # Constants needed by processors
    ANNOTATED_HEADER: ClassVar[str]
    METADATA_HEADER: ClassVar[str]

    # Instance attributes (or properties - both work with Protocol)
    args: Namespace
    reader: CSVReader
    writer: OutputWriter
    processor: RecordProcessor
    llm_client: Client
    incremental_mode: bool
