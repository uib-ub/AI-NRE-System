"""Main application module for medieval text processing using Large Language Models.

This module provides the primary entry point and orchestration logic for processing
medieval texts with Named Entity Recognition capabilities. It supports both
synchronous and asynchronous processing modes with comprehensive error handling
and progress monitoring.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Final, Literal

from ai_ner_system.config import ConfigError, ConfigValidator, Settings
from ai_ner_system.pipeline import ApplicationError, MedievalTextProcessor
from ai_ner_system.processing import create_progress_logger

# ============================================================================
# Constants
# ============================================================================

# Validation thresholds for async mode
MIN_MAX_WAIT_TIME: Final[float] = 60.0  # Minimum max wait time in seconds
MIN_POLL_INTERVAL: Final[float] = 5.0  # Minimum poll interval in seconds

# Default values for processing
DEFAULT_BATCH_SIZE: Final[int] = 5  # Records per batch
# Max concurrent batches in async mode
DEFAULT_MAX_CONCURRENT_BATCHES: Final[int] = 5

# Default values for async arguments
DEFAULT_MAX_WAIT_TIME: Final[float] = 86400.0  # 24 hours in seconds
DEFAULT_POLL_INTERVAL: Final[float] = 30.0  # 30 seconds

# Progress logging interval
PROGRESS_LOG_INTERVAL: Final[float] = 60.0  # Log progress every 60 seconds


# ============================================================================
# Utility functions
# ============================================================================
def setup_logging(level: str = "INFO") -> None:
    """Set up application logging.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        Defaults to 'INFO'.
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure logging format
    log_format = "%(asctime)s %(name)s [%(levelname)s]: %(message)s"
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Logging configured (level=%s)", level)

    # Set specific loggers to appropriate levels
    for logger_name in ["anthropic", "httpx", "requests"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _validate_input_file(input_file: str) -> None:
    """Validate the input file path.

    Args:
        input_file: Path to the input file.

    Raises:
        ApplicationError: If the input file is invalid.
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise ApplicationError(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise ApplicationError(f"Input path is not a file: {input_path}")

    # Check if file is readable
    try:
        with input_path.open("rb"):
            pass
    except OSError as e:
        raise ApplicationError(f"Input file is not readable: {e}") from e


def _validate_output_directories(output_files: list[str]) -> None:
    """Validate and create output directories if they do not exist.

    Args:
        output_files: List of output file paths.

    Raises:
        ApplicationError: If an output directory cannot be created.
    """
    for output_file in output_files:
        output_path = Path(output_file)
        output_dir = output_path.parent

        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logging.info("Created output directory: %s", output_dir)
            except OSError as e:
                raise ApplicationError(
                    f"Failed to create output directory {output_dir}: {e}",
                ) from e


def _validate_template_files(args: argparse.Namespace) -> None:
    """Validate template files exist if specified.

    Args:
        args: Parsed command line arguments.

    Raises:
        ValueError: If template files do not exist.
    """
    # Check if prompt template exists
    if args.prompt_template and not Path(args.prompt_template).exists():
        raise ApplicationError(
            f"Prompt template file does not exist: {args.prompt_template}",
        )

    # Check batch template if batch processing is enabled
    if (
        args.use_batch
        and args.batch_template
        and not Path(args.batch_template).exists()
    ):
        raise ApplicationError(
            f"Batch template file does not exist: {args.batch_template}",
        )


def _validate_async_arguments(args: argparse.Namespace) -> None:
    """Validate async-specific arguments.

    Args:
        args: Parsed command line arguments.

    Raises:
        ApplicationError: If arguments are invalid.
    """
    if not args.async_mode:
        return

    max_wait_time = args.max_wait_time
    if max_wait_time < MIN_MAX_WAIT_TIME:
        msg = (
            f"Max wait time must be at least {MIN_MAX_WAIT_TIME} seconds for async mode, "
            f"got {max_wait_time} seconds"
        )
        raise ApplicationError(msg)

    poll_interval = args.poll_interval
    if poll_interval < MIN_POLL_INTERVAL:
        msg = (
            f"Poll interval must be at least {MIN_POLL_INTERVAL} seconds for async mode, "
            f"got {poll_interval} seconds"
        )
        raise ApplicationError(msg)


def validate_arguments(args: argparse.Namespace) -> None:
    """Uses command-line overrides OR Settings defaults.

    Args:
        args: Parsed command line arguments.

    Raises:
        ApplicationError: If arguments are invalid.
    """
    # Validate input files
    input_file = args.input or Settings.INPUT_FILE
    _validate_input_file(input_file)

    # Validate output directories
    output_files = [
        args.output_text or Settings.OUTPUT_TEXT_FILE,
        args.output_table or Settings.OUTPUT_TABLE_FILE,
        args.output_stats or Settings.OUTPUT_STATS_FILE,
    ]
    _validate_output_directories(output_files)

    # Validate template files
    _validate_template_files(args)

    # Validate client type using constant from Settings
    if args.client not in Settings.SUPPORTED_CLIENTS:
        supported = ", ".join(sorted(Settings.SUPPORTED_CLIENTS))
        raise ApplicationError(
            f"Unsupported client type: {args.client}. Supported types: {supported}",
        )

    # Validate async-specific arguments
    _validate_async_arguments(args)

    logging.info("Command line arguments validated successfully")


def validate_configuration(args: argparse.Namespace) -> None:
    """Validate application configuration.

    Args:
        args: Parsed command line arguments.

    Raises:
        ConfigError: If configuration is invalid.
    """
    # Ensure Settings are initialized before applying overrides
    # (safe to call multiple times - no-op if already initialized)
    Settings.initialize()

    # Apply CLI overrides so configuration validation checks the same
    # effective paths accepted during argument validation.
    Settings.apply_cli_overrides(
        input_file=args.input,
        output_text_file=args.output_text,
        output_table_file=args.output_table,
        output_stats_file=args.output_stats,
        prompt_template_file=args.prompt_template,
        batch_template_file=args.batch_template,
    )

    try:
        ConfigValidator.validate_all(args.client)
        logging.info("Configuration validation completed successfully")
    except ConfigError as e:
        raise ApplicationError(f"Configuration validation failed: {e}") from e


def _get_example_text() -> str:
    """Get example text for argument parser epilog.

    Returns:
        Formatted example text with usage examples.
    """
    return """
Examples:
    # Process with sync mode
    uv run -m ai_ner_system.main --client ollama \\
        --input input/input.txt \\
        --output-text output/annotated_output.txt \\
        --output-table output/metadata_table.txt \\
        --use-batch --batch-size 10 -l DEBUG

    uv run -m ai_ner_system.main --client ollama \\
        --output-text output/annotated_output_gemma_batch_13R_B1.txt \\
        --output-table output/metadata_table_gemma_batch_13R_B1.txt \\
        -l DEBUG

    # Process with async batch processing
    uv run -m ai_ner_system.main --client claude \\
        --output-text output/annotated_output_claude_batch_100R_B100_async.txt \\
        --output-table output/metadata_table_claude_batch_100R_B100_async.txt \\
        --output-stats output/stats_claude_batch_100R_B100_async.txt  \\
        --batch-size 100 --async -l DEBUG

    uv run -m ai_ner_system.main --client claude \\
        --output-text output/annotated_output_claude_batch_13R_B2_async.txt \\
        --output-table output/metadata_table_claude_batch_13R_B2_async.txt \\
        --output-stats output/stats_claude_batch_13R_B2_async.txt \\
        --batch-size 2 --async --incremental-output -l DEBUG
"""


def _add_io_arguments(parser: argparse.ArgumentParser) -> None:
    """Add input/output file arguments to the parser.

    Args:
        parser: ArgumentParser instance to modify.
    """
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the input file (default: from .env or Settings.INPUT_FILE)",
    )

    parser.add_argument(
        "--output-text",
        type=str,
        default=None,
        help="Path for annotated text output (default: from .env or Settings.OUTPUT_TEXT_FILE)",
    )

    parser.add_argument(
        "--output-table",
        type=str,
        default=None,
        help="Path for metadata table output (default: from .env or Settings.OUTPUT_TABLE_FILE)",
    )

    parser.add_argument(
        "--output-stats",
        type=str,
        default=None,
        help="Output file for processing statistics (default: from .env or Settings.OUTPUT_STATS_FILE)",
    )


def _add_batch_arguments(parser: argparse.ArgumentParser) -> None:
    """Add batch processing arguments to the parser.

    Args:
        parser: ArgumentParser instance.
    """
    parser.add_argument(
        "--use-batch",
        action="store_true",
        help="Enable batch processing for better performance",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of records to process in each batch (default: {DEFAULT_BATCH_SIZE})",
    )


def _add_template_arguments(parser: argparse.ArgumentParser) -> None:
    """Add template file arguments to the parser.

    Args:
        parser: ArgumentParser instance.
    """
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help="Path to the prompt template file (default: from .env or Settings.PROMPT_TEMPLATE_FILE)",
    )

    parser.add_argument(
        "--batch-template",
        type=str,
        default=None,
        help="Path to the batch template file (default: from .env or Settings.BATCH_TEMPLATE_FILE)",
    )


def _add_async_arguments(parser: argparse.ArgumentParser) -> None:
    """Add async processing arguments to the parser.

    Args:
        parser: ArgumentParser instance.
    """
    parser.add_argument(
        "--async-mode",
        "-a",
        action="store_true",
        dest="async_mode",
        help="Enable asynchronous processing for batch operations",
    )

    parser.add_argument(
        "--max-concurrent-batches",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_BATCHES,
        help=f"Maximum number of concurrent batches (default: {DEFAULT_MAX_CONCURRENT_BATCHES})",
    )

    parser.add_argument(
        "--incremental-output",
        action="store_true",
        help="Write outputs incrementally after each batch (useful for large datasets)",
    )

    parser.add_argument(
        "--max-wait-time",
        type=float,
        default=DEFAULT_MAX_WAIT_TIME,
        help=f"Maximum time to wait for async batch completion in seconds (default: {DEFAULT_MAX_WAIT_TIME}, i.e. 24 hours)",
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Time between progress checks for async processing in seconds (default: {DEFAULT_POLL_INTERVAL})",
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Medieval Text Processor with AI NER System - Process medieval texts using Large Language Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=_get_example_text(),
    )

    # Model client selection
    parser.add_argument(
        "--client",
        "-c",
        type=str.lower,
        choices=sorted(Settings.SUPPORTED_CLIENTS),
        default="claude",
        help="Select LLM Client (default: claude)",
    )

    # Add argument groups
    _add_io_arguments(parser)
    _add_batch_arguments(parser)
    _add_template_arguments(parser)
    _add_async_arguments(parser)

    # Utility arguments
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and inputs without processing",
    )

    return parser


def _run_processor(
    processor: MedievalTextProcessor,
    args: argparse.Namespace,
) -> Literal[0, 1]:
    """Run the processor in the appropriate mode (sync or async).

    Args:
        processor: Instance of MedievalTextProcessor.
        args: Parsed command line arguments.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    # Choose execution mode based on async_mode argument
    async_mode = args.async_mode

    if async_mode:
        logging.info("Using asynchronous processing mode")
        # Create progress callback
        progress_callback = create_progress_logger(
            PROGRESS_LOG_INTERVAL,  # Log every 60 seconds
        )

        # Run async processing with parameters from command line arguments
        return asyncio.run(
            processor.run_async(
                progress_callback,
                timeout_seconds=args.max_wait_time,
                max_batch_wait_time=args.max_wait_time,
                poll_interval=args.poll_interval,
            ),
        )

    logging.info("Using synchronous processing mode")
    # Run synchronous processing
    return processor.run()


def _print_dry_run_success() -> None:
    """Print success message for dry run mode.

    Logs validation success to both console and logs.
    """
    success_messages = [
        "✓ Configuration validated successfully",
        "✓ Command line arguments validated",
        "✓ Input files exist and are accessible",
        "Dry run completed successfully - no processing performed",
    ]
    message = "\n".join(success_messages)
    print(message)
    logging.info("Dry run validation completed successfully")


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
def main() -> Literal[0, 1]:
    """Main application entry point.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()

        # Setup logging
        setup_logging(args.log_level)
        logging.info("AI NER System - Medieval Text Processing started")

        # Initialize settings (load .env file and create directories)
        Settings.initialize()

        # Validate configuration and arguments
        validate_arguments(args)
        validate_configuration(args)

        # Handle dry run
        if args.dry_run:
            _print_dry_run_success()
            return 0

        # Initialize processor
        processor = MedievalTextProcessor(args)
        return _run_processor(processor, args)

    except KeyboardInterrupt:
        logging.exception("Processing interrupted by user")
        return 1
    except ApplicationError:
        logging.exception("Application error: %s")
        return 1
    except Exception:
        # Unexpected errors - log with full traceback
        logging.exception("Unexpected error: %s")
        return 1


if __name__ == "__main__":
    sys.exit(main())
