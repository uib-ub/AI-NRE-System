"""Main application module for medieval text processing using Large Language Models.

This module provides the primary entry point and orchestration logic for processing
medieval texts with Named Entity Recognition capabilities. It supports both
synchronous and asynchronous processing modes with comprehensive error handling
and progress monitoring.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from ai_ner_system.config import ConfigValidator, ConfigError, Settings
from ai_ner_system.pipeline import ApplicationError, MedievalTextProcessor
from ai_ner_system.processing import create_progress_logger


# ============================================================================
# Utility functions
# ============================================================================
def setup_logging(level: str = 'INFO') -> None:
    """Set up application logging.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure logging format
    log_format = '%(asctime)s %(name)s [%(levelname)s]: %(message)s'
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

    logging.info('Logging configured (level=%s)', level)

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
        raise ApplicationError(f'Input file does not exist: {input_path}')
    if not input_path.is_file():
        raise ApplicationError(f'Input path is not a file: {input_path}')


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
                logging.info('Created output directory: %s', output_dir)
            except OSError as e:
                raise ApplicationError(f'Failed to create output directory {output_dir}: {e}') from e


def _validate_template_files(args: argparse.Namespace) -> None:
    """Validate template files exist if specified.

    Args:
        args: Parsed command line arguments.

    Raises:
        ValueError: If template files do not exist.
    """
    # Check if prompt template exists
    if args.prompt_template and not Path(args.prompt_template).exists():
        raise ValueError(f'Prompt template file does not exist: {args.prompt_template}')

    # Check batch template if batch processing is enabled
    if args.use_batch:
        if args.batch_template and not Path(args.batch_template).exists():
            raise ValueError(f'Batch template file does not exist: {args.batch_template}')


def _validate_async_arguments(args: argparse.Namespace) -> None:
    """Validate async-specific arguments.

    Args:
        args: Parsed command line arguments.

    Raises:
        ApplicationError: If arguments are invalid.
    """
    if not getattr(args, 'async_mode', False):
        return

    max_wait_time = getattr(args, 'max_wait_time', 0)
    if max_wait_time <= 60:
        raise ApplicationError(
            f'Max wait time must be at least 60 seconds for async mode, '
            f'got {max_wait_time} seconds'
        )

    poll_interval = getattr(args, 'poll_interval', 0)
    if poll_interval <= 5:
        raise ApplicationError(
            f'Poll interval must be at least 5 seconds for async mode, '
            f'got {poll_interval} seconds'
        )


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments.

    Args:
        args: Parsed command line arguments.

    Raises:
        ApplicationError: If arguments are invalid.
    """
    # Validate input files
    input_file = args.input or Settings.INPUT_FILE
    _validate_input_file(input_file)

    # Validate output directories
    output_files = [args.output_text, args.output_table]
    _validate_output_directories(output_files)

    # Validate template files
    _validate_template_files(args)

    # Validate client type
    if args.client.lower() not in ('claude', 'ollama'):
        raise ApplicationError(f'Unsupported client type: {args.client}')

    # Validate async-specific arguments
    _validate_async_arguments(args)

    logging.info('Command line arguments validated successfully')


def validate_configuration(args: argparse.Namespace) -> None:
    """Validate application configuration.

    Raises:
        ConfigError: If configuration is invalid.
    """
    try:
        ConfigValidator.validate_all(args.client)
        logging.info('Configuration validation completed successfully')
    except ConfigError as e:
        raise ApplicationError(f'Configuration validation failed: {e}') from e


def _get_example_text() -> str:
    """Get example text for argument parser epilog."""
    return """
Examples:
    # Process with sync mode
    uv run src/ai_ner_system.main.py --client ollama \\
        --input input/input.txt \\ 
        --output-text output/annotated_output.txt \\
        --output-table output/metadata_table.txt \\ 
        --use-batch --batch-size 10 -l DEBUG
        
    uv run src/ai_ner_system/main.py --client ollama \\ 
        --output-text output/annotated_output_gemma_batch_13R_B1.txt \\
        --output-table output/metadata_table_gemma_batch_13R_B1.txt \\
        -l DEBUG
    
    # Process with async batch processing
    uv run src/ai_ner_system/main.py --client claude \\
        --output-text output/annotated_output_claude_batch_100R_B100_async.txt \\
        --output-table output/metadata_table_claude_batch_100R_B100_async.txt \\
        --output-stats output/stats_claude_batch_100R_B100_async.txt  \\
        --batch-size 100 --async -l DEBUG
        
    uv run src/ai_ner_system/main.py --client claude \\
        --output-text output/annotated_output_claude_batch_13R_B2_async.txt \\
        --output-table output/metadata_table_claude_batch_13R_B2_async.txt \\
        --output-stats output/stats_claude_batch_13R_B2_async.txt \\
        --batch-size 2 --async --incremental-output -l DEBUG
"""


def _add_io_arguments(parser: argparse.ArgumentParser) -> None:
    """Add input/output file arguments to the parser.

    Args:
        parser: ArgumentParser instance.
    """
    parser.add_argument(
        '--input',
        type=str,
        default=Settings.INPUT_FILE,
        help='Path to the input file'
    )

    parser.add_argument(
        '--output-text',
        type=str,
        default=Settings.OUTPUT_TEXT_FILE,
        help='Path for annotated text output'
    )

    parser.add_argument(
        '--output-table',
        type=str,
        default=Settings.OUTPUT_TABLE_FILE,
        help='Path for metadata table output'
    )

    parser.add_argument(
        "--output-stats",
        type=str,
        default=Settings.OUTPUT_STATS_FILE,
        help="Output file for processing statistics (JSON format)"
    )


def _add_batch_arguments(parser: argparse.ArgumentParser) -> None:
    """Add batch processing arguments to the parser.

    Args:
        parser: ArgumentParser instance.
    """
    parser.add_argument(
        '--use-batch',
        action='store_true',
        help='Enable batch processing for better performance'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Number of records to process in each batch (default: 5)'
    )


def _add_template_arguments(parser: argparse.ArgumentParser) -> None:
    """Add template file arguments to the parser.

    Args:
        parser: ArgumentParser instance.
    """
    parser.add_argument(
        '--prompt-template',
        type=str,
        default=Settings.PROMPT_TEMPLATE_FILE,
        help='Path to the prompt template file'
    )

    parser.add_argument(
        '--batch-template',
        type=str,
        default=Settings.BATCH_TEMPLATE_FILE,
        help='Path to the batch template file'
    )

def _add_async_arguments(parser: argparse.ArgumentParser) -> None:
    """Add async processing arguments to the parser.

    Args:
        parser: ArgumentParser instance.
    """
    parser.add_argument(
        '--async-mode', '-a',
        action='store_true',
        dest='async_mode',
        help='Enable asynchronous processing for batch operations'
    )

    parser.add_argument(
        '--max-concurrent-batches',
        type=int,
        default=5,
        help='Maximum number of concurrent batches (default: 5)'
    )

    parser.add_argument(
        '--incremental-output',
        action='store_true',
        help='Write outputs incrementally after each batch (useful for large datasets)'
    )

    parser.add_argument(
        '--max-wait-time',
        type=int,
        default=86400,  # 24 hours
        help='Maximum time to wait for async batch completion (default: 86400 seconds, i.e. 24 hours)'
    )

    parser.add_argument(
        '--poll-interval',
        type=int,
        default=30,
        help='Time between progress checks for async processing (default: 30s)'
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description='Medieval Text Processor with AI NER System - Process medieval texts using Large Language Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=_get_example_text()
    )

    # Model client selection
    parser.add_argument(
        '--client', '-c',
        type=str,
        choices=['claude', 'ollama'],
        default='claude',
        help='Select LLM Client (default: claude)'
    )

    # Add argument groups
    _add_io_arguments(parser)
    _add_batch_arguments(parser)
    _add_template_arguments(parser)
    _add_async_arguments(parser)

    # Utility arguments
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and inputs without processing'
    )

    return parser


def _run_processor(processor: MedievalTextProcessor, args: argparse.Namespace) -> int:
    """Run the processor in synchronous mode.

    Args:
        processor: Instance of MedievalTextProcessor.
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # Choose execution mode based on async_mode argument
    async_mode = getattr(args, 'async_mode', False)

    if async_mode:
        logging.info('Using asynchronous processing mode')
        # Create progress callback
        progress_callback = create_progress_logger(60)  # Log every 60 seconds
        # Run async processing
        return asyncio.run(processor.run_async(progress_callback))
    else:
        logging.info('Using synchronous processing mode')
        # Run synchronous processing
        return processor.run()


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
def main() -> int:
    """Main application entry point.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()

        # Setup logging
        setup_logging(args.log_level)
        logging.info('AI NER System - Medieval Text Processing started')

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
        logging.error('Processing interrupted by user')
        return 1
    except ApplicationError as e:
        logging.error('Application error: %s', e)
        return 1
    except Exception as e:
        logging.error('Unexpected error: %s', e, exc_info=True)
        return 1


def _print_dry_run_success() -> None:
    """Print success message for dry run."""
    success_messages = [
        '✓ Configuration validated successfully',
        '✓ Command line arguments validated',
        '✓ Input files exist and are accessible',
        'Dry run completed successfully - no processing performed'
    ]
    print('\n'.join(success_messages))


if __name__ == "__main__":
    sys.exit(main())