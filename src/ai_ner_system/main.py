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

from ai_ner_system.config import Settings, ConfigValidator, ConfigError
from ai_ner_system.pipeline import ApplicationError, MedievalTextProcessor
from ai_ner_system.processing import create_progress_logger


# ============================================================================
# Utility functions
# ============================================================================
def setup_logging(level: str = 'INFO') -> None:
    """Setup application logging

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
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments.

    Args:
        args: Parsed command line arguments.

    Raises:
        ValueError: If arguments are invalid.
    """
    # Validate input files
    input_file = args.input or Settings.INPUT_FILE
    input_path = Path(input_file)

    if not input_path.exists():
        raise ApplicationError(f'Input file does not exist: {input_path}')
    if not input_path.is_file():
        raise ApplicationError(f'Input path is not a file: {input_path}')

    # if input_file and not Path(input_file).exists():
    #     raise ValueError(f'Input file does not exist: {input_file}')

    # Validate output directories
    for output_file in [args.output_text, args.output_table]:
        output_path = Path(output_file)
        output_dir = output_path.parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logging.info('Created output directory: %s', output_dir)
            except OSError as e:
                raise ApplicationError(f'Failed to create output directory {output_dir}: {e}') from e

    # Check if prompt template exists
    if args.prompt_template and not Path(args.prompt_template).exists():
        raise ValueError(f'Prompt template file does not exist: {args.prompt_template}')

    # Check batch template if batch processing is enabled
    if args.use_batch:
        if args.batch_template and not Path(args.batch_template).exists():
            raise ValueError(f'Batch template file does not exist: {args.batch_template}')

    # Validate client type
    if args.client.lower() not in ['claude', 'ollama']:
        raise ApplicationError(f'Unsupported client type: {args.client}')

    # Validate async-specific arguments
    if hasattr(args, 'async_mode') and args.async_mode:
        if hasattr(args, 'max_wait_time') and args.max_wait_time <= 60:
            raise ApplicationError(f'Max wait time must be at least 60 seconds for async mode, got {args.max_wait_time} seconds')
        if hasattr(args, 'poll_interval') and args.poll_interval <= 5:
            raise ApplicationError(f'Poll interval must be at least 5 seconds for async mode, got {args.poll_interval} seconds')

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


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description='Medieval Text Processor with AI NER System - Process medieval texts using Large Language Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
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
    )

    # Model client selection
    parser.add_argument(
        '--client', '-c',
        type=str,
        choices=['claude', 'ollama'],
        default='claude',
        help='Select LLM Client (default: claude)'
    )

    # IO File file arguments
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

    # Batch processing options
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

    # Async processing arguments
    parser.add_argument(
        '--async-mode', '-a',
        action='store_true',
        dest='async_mode',
        help='Enable asynchronous processing for batch operations'
    )

    # Maximum number of concurrent batches
    parser.add_argument(
        '--max-concurrent-batches',
        type=int,
        default=5,
        help='Maximum number of concurrent batches (default: 5)'
    )

    # Incremental output option
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

    # Utility arguments
    # Logging
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )

    # Additional options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and inputs without processing'
    )

    return parser

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
def main() -> int:
    """Main application entry point.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # Add this at the very beginning
    # root_logger = logging.getLogger()
    # print(f"Handlers before setup_logging: {root_logger.handlers}")
    # print(f"Root logger level before setup_logging: {root_logger.level}")

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
            print('✓ Configuration validated successfully')
            print('✓ Command line arguments validated')
            print('✓ Input files exist and are accessible')
            print('Dry run completed successfully - no processing performed')
            return 0

        # Initialize processor
        processor = MedievalTextProcessor(args)

        # Choose execution mode based on async_mode argument
        match getattr(args, 'async_mode', False):
            case True:
                logging.info('Using asynchronous processing mode')

                # Create progress callback
                progress_callback = create_progress_logger(60) # Log every 60 seconds

                # Run async processing
                return asyncio.run(processor.run_async(progress_callback))
            case False:
                logging.info('Using synchronous processing mode')
                # Run synchronous processing
                return processor.run()
            case _:
                logging.error(
                    f"Invalid value for async_mode: {args.async_mode}. "
                    "Please set it to either True or False."
                )
                return 1  # Non-zero exit code to indicate configuration error

    except KeyboardInterrupt:
        logging.error('Processing interrupted by user')
        return 1
    except ApplicationError as e:
        logging.error('Application error: %s', e)
        return 1
    except Exception as e:
        logging.error('Unexpected error: %s', e, exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())