"""Medieval text processing application with LLM annotation.

This module provides the main entry point for processing medieval texts using
Large Language Models (LLMs) for proper noun annotation and metadata extraction.
"""

import argparse
import time
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass

from tqdm import tqdm

from ai_ner_system.config import Config, ConfigError
from ai_ner_system.prompts import PromptBuilder, GenericPromptBuilder, PromptError
from ai_ner_system.io_utils import CSVReader, OutputWriter, IOError
from ai_ner_system.processing import RecordProcessor, ValidationError, ProcessingError, LLMResponseError
from ai_ner_system.llm_clients import create_llm_client, Client, LLMClientError, BatchProgress


class ApplicationError(Exception):
    """Custom exception for application-level errors."""

@dataclass
class AsyncProcessingStats:
    pass

class MedievalTextProcessor:
    """Main application class for processing medieval texts with LLMs."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the medieval text processor.

        Args:
            args: Parsed command line arguments.

        Raises:
            ApplicationError: If initialization fails.
        """
        self.args = args

        # Initialize components
        self.llm_client: Optional[Client] = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self.processor: Optional[RecordProcessor] = None
        self.reader: Optional[CSVReader] = None
        self.writer: Optional[OutputWriter] = None

        try:
            self.llm_client = self._initialize_llm_client()
            self.prompt_builder = self._initialize_prompt_builder()
            self.processor = RecordProcessor(self.llm_client, self.prompt_builder)

            self.reader = self._initialize_csv_reader()
            self.writer = OutputWriter()

            logging.info('Medieval text processor initialized successfully')

        except Exception as e:
            raise ApplicationError(f'Failed to initialize MedievalTextProcessor: {e}') from e

    def _initialize_llm_client(self) -> Client:
        """Initialize the LLM client based on command line arguments.

        Returns:
            Configured LLM client instance.

        Raises:
            ApplicationError: If LLM client initialization fails.
        """
        try:
            client_type = self.args.client.lower()

            if client_type not in ['claude', 'ollama']:
                raise ApplicationError(f'Unsupported client type: {client_type}')

            llm_client = create_llm_client(client_type)
            logging.info('LLM client initialized: %s', client_type)
            return llm_client

        except LLMClientError as e:
            raise ApplicationError(f'Failed to initialize LLM client "{self.args.client}": {e}') from e
        except Exception as e:
            raise ApplicationError(f'Unexpected error initializing LLM client: {e}') from e

    def _initialize_prompt_builder(self) -> PromptBuilder:
        """Initialize the prompt builder with template file.

        Returns:
            Configured PromptBuilder instance that can handle both single and batch processing.

        Raises:
            ApplicationError: If prompt builder initialization fails.
        """
        try:
            # Determine which template to use based on whether batch processing is enabled
            if self.args.use_batch:
                template_file = self.args.batch_template or Config.BATCH_TEMPLATE_FILE
                logging.info('Using batch prompt template: %s', template_file)
            else:
                template_file = self.args.prompt_template or Config.PROMPT_TEMPLATE_FILE
                logging.info('Using single prompt template: %s', template_file)

            prompt_builder = GenericPromptBuilder(template_file)
            logging.info('Prompt builder initialized with template: %s', template_file)
            return prompt_builder
        except PromptError as e:
            raise ApplicationError(f'Failed to initialize PromptBuilder: {e}') from e

    def _initialize_csv_reader(self) -> CSVReader:
        """Initialize the CSV reader for input file.

        Returns:
            Configured CSVReader instance.

        Raises:
            ApplicationError: If CSV reader initialization fails.
        """
        try:
            input_file = self.args.input or Config.INPUT_FILE

            if not Path(input_file).exists():
                raise ApplicationError(f"Input file does not exist: {input_file}")

            reader = CSVReader(input_file, delimiter=';', encoding='utf-8')
            logging.info('CSV reader initialized for input file: %s', input_file)
            return reader

        except IOError as e:
            raise ApplicationError(f'Failed to initialize CSV reader: {e}') from e
        except Exception as e:
            raise ApplicationError(f"Unexpected error initializing CSV reader: {e}") from e

    # ============================================================================
    # SYNC METHODS -  sync batch processing capabilities
    # ============================================================================
    def process_all_records(self) -> Tuple[List[str], List[str]]:
        """Process all records from the input CSV file using streaming approach.

        Returns:
            Tuple of (all_annotations, all_metadata).

        Raises:
            ApplicationError: If critical processing error occurs.
        """
        try:
            logging.info('Starting to process records from: %s', self.reader.file_path)

            batch_size = self.args.batch_size if self.args.use_batch else 1

            # Process records with unified streaming approach
            all_annotations, all_metadata = self._process_records_streaming(batch_size)
            logging.info('Completed processing all records')
            return all_annotations, all_metadata

        except Exception as e:
            raise ApplicationError(f'Critical error during file processing: {e}') from e

    def _process_records_streaming(self, batch_size: int) -> Tuple[List[str], List[str]]:
        """Process records using streaming approach with configurable batch size.

        Args:
            batch_size: Number of records to process together (1 = individual processing)

        Returns:
            Tuple of (all_annotations, all_metadata).
        """
        all_annotations: List[str] = []
        all_metadata: List[str] = []

        batch_records: List[Dict[str, str]] = [] # a batch of records to process together
        batch_count = 0 # counter for the number of batches processed

        processing_mode = "batch" if batch_size > 1 else "individual"

        logging.info('Using %s processing (batch_size=%d)', processing_mode, batch_size)

        for record in tqdm(self.reader.stream_records(), desc=f'Processing Records ({processing_mode} mode)'):
            batch_records.append(record)

            # Process when batch is full or for individual processing (batch_size=1)
            if len(batch_records) >= batch_size:
                batch_count += 1

                annotated_records, metadata_records = self._process_batch(batch_records, batch_count, batch_size)
                # Collect results
                all_annotations.extend(annotated_records)
                all_metadata.extend(metadata_records)

                logging.debug('Successfully processed %s: %d annotations, %d metadata',
                            f'Brevid {batch_records[0].get("Brevid", "unknown")}' if batch_size == 1 else f'batch {batch_count}',
                            len(annotated_records), len(metadata_records))

                # Clear batch records after processing
                batch_records = []
                time.sleep(0.2)

        # Process any remaining records in the final partial batch
        if batch_records:
            batch_count += 1
            annotations, metadata = self._process_final_batch(batch_records, batch_count, batch_size, processing_mode)
            all_annotations.extend(annotations)
            all_metadata.extend(metadata)

        return all_annotations, all_metadata

    def _process_batch(
        self,
        batch_records: List[Dict[str, str]],
        batch_count: int,
        batch_size: int
    ) -> Tuple[List[str], List[str]]:
        """Process a batch of records, handling both individual and batch modes.

        Args:
            batch_records: The list of records to process.
            batch_count: The current batch number.
            batch_size: The size of the batch.

        Returns:
            A tuple containing a list of annotation strings and a list of metadata strings.

        Raises:
            Any exception raised by the underlying processor will be handled by _handle_batch_exception.
        """
        try:
            # Process the batch/record
            if batch_size == 1:
                # Individual processing
                individual_record = batch_records[0]
                brevid = individual_record.get("Brevid", "unknown")
                logging.info('Processing Record with Brevid: %s', brevid)
                logging.debug('Individual record data: %s', individual_record)
                return self.processor.process_record(individual_record)
            else:
                # Batch processing
                logging.info('Processing batch %d with %d records', batch_count, len(batch_records))
                return self.processor.process_batch(batch_records)
        except Exception as e:
            return self._handle_batch_exception(batch_records, batch_count, batch_size, e)

    def _handle_batch_exception(
        self,
        batch_records: List[Dict[str, str]],
        batch_count: int,
        batch_size: int,
        error: Exception
    ) -> Tuple[List[str], List[str]]:
        """Handle exceptions during batch processing, with fallback to individual processing if needed.

        Args:
            batch_records: The list of records in the batch.
            batch_count: The current batch number.
            batch_size: The size of the batch.
            error: The exception that was raised.

        Returns:
            A tuple containing a list of annotation strings and a list of metadata strings.
        """
        if batch_size == 1:
            # Individual processing error
            brevid = batch_records[0].get("Brevid", "unknown")
            logging.error('Error processing Brevid %s: %s', brevid, error)
            self._handle_individual_error(batch_records[0], error)
            return [], []
        else:
            # Batch processing error - fallback to individual processing
            logging.error('Error processing batch %d: %s', batch_count, error)
            logging.info('Falling back to individual processing for batch %d', batch_count)
            annotations: List[str] = []
            metadata: List[str] = []
            # Process each record in the failed batch individually
            for record in batch_records:
                annotated_record, metadata_record = self._fallback_to_individual_processing(record)
                annotations.extend(annotated_record)
                metadata.extend(metadata_record)
            return annotations, metadata

    def _process_final_batch(
        self,
        batch_records: List[Dict[str, str]],
        batch_count: int,
        batch_size: int,
        processing_mode: str
    ) -> Tuple[List[str], List[str]]:
        """Process the final (possibly partial) batch of records.

        Args:
            batch_records: The list of records in the final batch.
            batch_count: The current batch number.
            batch_size: The size of the batch.
            processing_mode: The processing mode ("batch" or "individual").

        Returns:
            A tuple containing a list of annotation strings and a list of metadata strings.
        """
        annotations: List[str] = []
        metadata: List[str] = []
        with tqdm(total=len(batch_records), desc=f'Final {processing_mode} batch') as final_pbar:
            try:
                if batch_size == 1:
                    # This shouldn't happen since we process immediately, but we still handle it
                    for record in batch_records:
                        brevid = record.get("Brevid", "unknown")
                        annotated_records, metadata_records = self.processor.process_record(record)
                        annotations.extend(annotated_records)
                        metadata.extend(metadata_records)
                        final_pbar.set_description(f'Final record: {brevid}')
                        final_pbar.update(1)
                else:
                    # Process the final batch
                    logging.info('Processing final batch %d with %d records', batch_count, len(batch_records))
                    annotated_records, metadata_records = self.processor.process_batch(batch_records)
                    annotations.extend(annotated_records)
                    metadata.extend(metadata_records)
                    final_pbar.set_description(f'Final batch ({len(batch_records)} records)')
                    final_pbar.update(len(batch_records))
                    logging.debug('Successfully processed final batch: %d annotations, %d metadata',
                                  len(annotated_records), len(metadata_records))
            except Exception as e:
                logging.error('Error processing final batch: %s', e)
                # Fallback to individual processing for remaining records
                final_pbar.set_description('Final batch fallback')
                for record in batch_records:
                    annotated_record, metadata_record = self._fallback_to_individual_processing(record)
                    annotations.extend(annotated_record)
                    metadata.extend(metadata_record)
                    final_pbar.update(1)
        return annotations, metadata

    def _fallback_to_individual_processing(self, record: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """Fallback to individual processing when doing a batch of records.

        Args:
            record: record that failed to process in a batch processing.

        Returns:
            Tuple of (annotated_record, metadata_record) for the individual record.
        """
        logging.info('Falling back to individual processing for record: %s', record)

        annotated_record, metadata_record = [], []

        try:
            annotated_record, metadata_record = self.processor.process_record(record)
        except Exception as e:
            brevid = record.get("Brevid", "unknown")
            logging.error('Error in fallback processing for Brevid %s: %s', brevid, e)
            self._handle_individual_error(record, e)

        return annotated_record, metadata_record

    def _handle_individual_error(self, record: Dict[str, str], error: Exception) -> None:
        """Handle errors that occur during individual record processing.

        Args:
            record: The record that failed to process.
            error: The exception that occurred.
        """
        brevid = record.get("Brevid", "unknown")
        bindnr = record.get('Bindnr', 'unknown')

        if isinstance(error, ValidationError):
            logging.error('Validation error for Brevid %s (Bindnr: %s): %s', brevid, bindnr, error)
        elif isinstance(error, (ProcessingError, LLMResponseError)):
            logging.error('LLM Processing error for Brevid %s (Bindnr: %s): %s', brevid, bindnr, error)
        else:
            logging.error('Unexpected error processing Brevid %s (Bindnr: %s): %s', brevid, bindnr, error, exc_info=True)

    def write_output(self, annotations: List[str], metadata: List[str]) -> None:
        """Write processed data to output files.

        Args:
            annotations: List of annotated text records.
            metadata: List of metadata records.

        Raises:
            ApplicationError: If output writing fails.
        """

        try:
            # Determine output file paths
            output_text = self.args.output_text or Config.OUTPUT_TEXT_FILE
            output_table = self.args.output_table or Config.OUTPUT_TABLE_FILE

            # Write annotated text output
            if annotations:
                annotated_header = "Bindnr;Brevid;Tekst"
                self.writer.write_text_output(output_text, annotated_header, annotations)
                logging.info('Annotated text written to: %s (%d records)', output_text, len(annotations))
            else:
                logging.warning('No annotated text output to write')

            # Write metadata table output
            if metadata:
                metadata_header = (
                    "Proper Noun;Type of Proper Noun;Order of Occurrence in Doc;"
                    "Brevid;Status/Occupation/Description;Gender;Language"
                )
                self.writer.write_metadata_output(output_table, metadata_header, metadata)
                logging.info('Metadata written to: %s (%d records)', output_table, len(metadata))
            else:
                logging.warning('No metadata output to write')

        except IOError as e:
            raise ApplicationError(f'Failed to write outputs: {e}') from e
        except Exception as e:
            raise ApplicationError(f'Unexpected error during output writing: {e}') from e

    def run(self) -> int:
        """Run the complete processing pipeline.

        Raises:
            ApplicationError: If any step of the pipeline fails.

        Returns:
            Exit code (0 for success, 1 for failure).
        """
        try:
            logging.info('Starting medieval text processing...')

            # Process all records
            annotations, metadata = self.process_all_records()

            # Write output files
            self.write_output(annotations, metadata)

            logging.info('Processing completed successfully')

            # Final output paths
            output_text = self.args.output_text or Config.OUTPUT_TEXT_FILE
            output_table = self.args.output_table or Config.OUTPUT_TABLE_FILE
            print(f'\nOutputs written to:')
            print(f'  Annotated text: {output_text}')
            print(f'  Metadata table: {output_table}')

            return 0

        except ApplicationError as e:
            logging.error('Application error: %s', e, exc_info=True)
            return 1
        except KeyboardInterrupt:
            logging.info('Processing interrupted by user.')
            return 1
        except Exception as e:
            logging.error('Unexpected error: %s', e, exc_info=True)
            return 1

    # ============================================================================
    # ASYNC METHODS -  async batch processing capabilities
    # ============================================================================
    # TODO: Implement process_all_records_async
    async def process_all_records_async(
        self,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
        max_batch_wait_time: int = 86400, # 24 hours
        poll_interval: int = 30
    ) -> AsyncProcessingStats:
        pass

    # TODO: Implement _async_stream_records
    # TODO: Implement _process_batch_async
    # TODO: Implement _process_individual_async
    # TODO: Implement _create_batch_progress_callback
    # TODO: Implement write_output_async
    # TODO: Implement run_async

def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments.

    Args:
        args: Parsed command line arguments.

    Raises:
        ValueError: If arguments are invalid.
    """
    # Validate input files
    input_file = args.input or Config.INPUT_FILE
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

    logging.info('Command line arguments validated successfully')


def validate_configuration() -> None:
    """Validate application configuration.

    Raises:
        ConfigError: If configuration is invalid.
    """
    try:
        if not Config.is_valid():
            Config.validate_required_config()
    except ConfigError as e:
        raise ApplicationError(f'Configuration validation failed: {e}') from e

    logging.info('Configuration validated successfully')

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

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description='Medieval Text Processor - Process medieval texts using Large Language Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
                Examples:
                    python main.py --client claude
                    python src/main.py --client ollama --input input/input.txt -l INFO
                    python src/main.py --client ollama --output-text output/annotated_output.txt --output-table output/metadata_table.txt
                    python src/main.py --client ollama --output-text output/annotated_output.txt --output-table output/metadata_table.txt --use-batch --batch-size 10 -l DEBUG
                """
    )
    # Model selection
    parser.add_argument(
        '--client', '-c',
        choices=['claude', 'ollama'],
        default='claude',
        help='Select LLM Client (default: claude)'
    )

    # IO File paths
    parser.add_argument(
        '--input',
        default=Config.INPUT_FILE,
        help='Path to the input file'
    )

    parser.add_argument(
        '--output-text',
        default=Config.OUTPUT_TEXT_FILE,
        help='Path for annotated text output'
    )

    parser.add_argument(
        '--output-table',
        default=Config.OUTPUT_TABLE_FILE,
        help='Path for metadata table output'
    )

    # Batch processing options
    parser.add_argument(
        '--prompt-template',
        type=str,
        default=Config.PROMPT_TEMPLATE_FILE,
        help='Path to the prompt template file'
    )

    parser.add_argument(
        '--batch-template',
        type=str,
        default=Config.BATCH_TEMPLATE_FILE,
        help='Path to the batch template file'
    )

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

    # TODO: Async processing arguments
    # '--async'
    # '--max-wait-time'
    # '--poll-interval'

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

        # Validate configuration and arguments
        validate_arguments(args)
        validate_configuration()

        # Handle dry run
        if args.dry_run:
            print('✓ Configuration validated successfully')
            print('✓ Command line arguments validated')
            print('✓ Input files exist and are accessible')
            print('Dry run completed successfully - no processing performed')
            return 0

        # Initialize and run processor
        processor = MedievalTextProcessor(args)
        processor.run()

        logging.info('Application completed successfully')

        # TODO: Choose sync or async execution
        return 0

    # except ConfigError as e:
    #     logging.error('Configuration error: %s', e)
    #     print(f'Configuration error: {e}', file=sys.stderr)
    #     sys.exit(1)

    # except ValueError as e:
    #     logging.error('Invalid arguments: %s', e)
    #     print(f'Invalid arguments: {e}', file=sys.stderr)
    #     sys.exit(1)

    except ApplicationError as e:
        logging.error('Application error: %s', e)
        return 1
    except KeyboardInterrupt:
        logging.error('Processing interrupted by user')
        return 1
    except Exception as e:
        logging.error('Unexpected error: %s', e, exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
