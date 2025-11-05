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
from typing import Final, Literal

from ai_ner_system.config import (
    ConfigError,
    ConfigValidationError,
    ConfigValidator,
    DirectoryValidationError,
    FileValidationError,
    Settings,
)
from ai_ner_system.pipeline import ApplicationError, MedievalTextProcessor
from ai_ner_system.processing import create_progress_logger

# ============================================================================
# Constants
# ============================================================================

# Validation thresholds for async mode
MIN_MAX_WAIT_TIME: Final[float] = 60.0  # Minimum max wait time in seconds
MIN_POLL_INTERVAL: Final[float] = 5.0  # Minimum poll interval in seconds

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


def _validate_configuration(args: argparse.Namespace) -> None:
    """Validate application configuration and arguments.

    This function:
    1. Initializes Settings (loads .env, creates directories)
    2. Applies CLI overrides to Settings
    3. Delegates to ConfigValidator for comprehensive validation

    Args:
        args: Parsed command line arguments.

    Raises:
        ApplicationError: If validation fails.
    """
    try:
        # Initialize settings (load .env file and create directories)
        # Safe to call multiple times - no-op if already initialized
        Settings.initialize()

        # Apply CLI overrides so ConfigValidator checks the effective paths
        Settings.apply_cli_overrides(
            input_file=args.input,
            output_text_file=args.output_text,
            output_table_file=args.output_table,
            output_stats_file=args.output_stats,
            prompt_template_file=args.prompt_template,
            batch_template_file=args.batch_template,
        )

        # Comprehensive validation: files, paths, and client config
        ConfigValidator.validate_all(args.client)

        _log_configuration_summary(args)

        logging.info("Configuration validation completed successfully")

    except (
        ConfigError,
        ConfigValidationError,
        FileValidationError,
        DirectoryValidationError,
    ) as e:
        raise ApplicationError(f"Configuration validation failed: {e}") from e


def _log_configuration_summary(args: argparse.Namespace) -> None:
    """Log summary of effective configuration.

    Helps with debugging and provides audit trail.

    Args:
        args: Parsed command line arguments.
    """
    logging.info("Configuration Summary:")
    logging.info("  Client: %s", args.client)
    logging.info("  Input: %s", Settings.INPUT_FILE)
    logging.info("  Prompt Template: %s", Settings.PROMPT_TEMPLATE_FILE)
    logging.info("  Output Text: %s", Settings.OUTPUT_TEXT_FILE)
    logging.info("  Output Table: %s", Settings.OUTPUT_TABLE_FILE)
    logging.info("  Async Mode: %s", args.async_mode)

    if args.async_mode:
        logging.info("  Batch Size: %d", args.batch_size)
        logging.info("  Max Concurrent Batches: %d", args.max_concurrent_batches)
        logging.info("  Max Concurrent Individual: %d", args.max_concurrent_individual)
        logging.info("  Fallback Concurrency: %d", args.fallback_concurrency)
        logging.info("  Chunk Size: %d", args.chunk_size)
        logging.info("  Max Wait Time: %.1fs", args.max_wait_time)
        logging.info("  Poll Interval: %.1fs", args.poll_interval)
        logging.info("  Incremental Output: %s", args.incremental_output)
        logging.info("  Output Stats: %s", Settings.OUTPUT_STATS_FILE)


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
        default=Settings.DEFAULT_BATCH_SIZE,
        help=f"Number of records to process in each batch (default: {Settings.DEFAULT_BATCH_SIZE})",
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
        default=Settings.DEFAULT_MAX_CONCURRENT_BATCHES,
        help=f"Maximum number of concurrent batch processing tasks (default: {Settings.DEFAULT_MAX_CONCURRENT_BATCHES})",
    )

    parser.add_argument(
        "--max-concurrent-individual",
        type=int,
        default=Settings.DEFAULT_MAX_CONCURRENT_INDIVIDUAL,
        help=f"Maximum concurrent individual record processing tasks (default: {Settings.DEFAULT_MAX_CONCURRENT_INDIVIDUAL})",
    )

    parser.add_argument(
        "--fallback-concurrency",
        type=int,
        default=Settings.DEFAULT_FALLBACK_CONCURRENCY,
        help=f"Concurrency limit for fallback processing (default: {Settings.DEFAULT_FALLBACK_CONCURRENCY})",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=Settings.DEFAULT_CHUNK_SIZE,
        help=f"Number of records to process per chunk for memory management (default: {Settings.DEFAULT_CHUNK_SIZE})",
    )

    parser.add_argument(
        "--incremental-output",
        action="store_true",
        help="Write outputs incrementally after each batch (useful for large datasets)",
    )

    parser.add_argument(
        "--max-wait-time",
        type=float,
        default=Settings.DEFAULT_MAX_WAIT_TIME,
        help=f"Maximum time to wait for async batch completion in seconds (default: {Settings.DEFAULT_MAX_WAIT_TIME}, i.e. 24 hours)",
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=Settings.DEFAULT_POLL_INTERVAL,
        help=f"Time between progress checks for async processing in seconds (default: {Settings.DEFAULT_POLL_INTERVAL})",
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
    # Dynamically generate choices from Settings.SUPPORTED_CLIENTS
    supported_clients = sorted(Settings.SUPPORTED_CLIENTS)
    parser.add_argument(
        "--client",
        "-c",
        type=str.lower,
        choices=supported_clients,
        default="claude",
        help=f"Select LLM Client (choices: {', '.join(supported_clients)}, default: claude)",
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


def _validate_arguments(args: argparse.Namespace) -> None:
    """Quick validation of argument before full setup.

    This provides faster feedback for obvious errors.

    Args:
        args: Parsed command line arguments.

    Raises:
        ApplicationError: If arguments are invalid.
    """
    if args.batch_size <= 0:
        msg = f"Batch size must be a positive integer, got {args.batch_size}"
        raise ApplicationError(msg)

    # Only validate async-specific arguments if async mode is enabled
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

    if poll_interval > max_wait_time:
        msg = (
            f"Poll interval ({poll_interval}s) cannot be greater than "
            f"max wait time ({max_wait_time}s)"
        )
        raise ApplicationError(msg)

    max_concurrent_batches = args.max_concurrent_batches
    if max_concurrent_batches < 1:
        msg = (
            "Max concurrent batches must be at least 1 for async mode, "
            f"got {max_concurrent_batches}"
        )
        raise ApplicationError(msg)

    max_concurrent_individual = args.max_concurrent_individual
    if max_concurrent_individual < 1:
        msg = (
            "Max concurrent individual tasks must be at least 1 for async mode, "
            f"got {max_concurrent_individual}"
        )
        raise ApplicationError(msg)

    fallback_concurrency = args.fallback_concurrency
    if fallback_concurrency < 1:
        msg = (
            "Fallback concurrency must be at least 1 for async mode, "
            f"got {fallback_concurrency}"
        )
        raise ApplicationError(msg)

    chunk_size = args.chunk_size
    if chunk_size < 1:
        msg = f"Chunk size must be at least 1 for async mode, got {chunk_size}"
        raise ApplicationError(msg)


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

        # Quick argument validation before full setup
        _validate_arguments(args)

        # Validate configuration (includes Settings initialization and all validation)
        _validate_configuration(args)

        # Handle dry run
        if args.dry_run:
            _print_dry_run_success()
            return 0

        # Initialize processor
        processor = MedievalTextProcessor(args)
        return _run_processor(processor, args)

    except KeyboardInterrupt:
        logging.warning("Processing interrupted by user")
        return 1
    except (ConfigError, ApplicationError):
        logging.exception("Application error occurred")
        return 1
    except Exception:
        # Unexpected errors - log with full traceback
        logging.exception("Unexpected error occurred")
        return 1


if __name__ == "__main__":
    sys.exit(main())
