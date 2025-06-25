"""Medieval text processing application with LLM annotation.

This module provides the main entry point for processing medieval texts using
Large Language Models (LLMs) for proper noun annotation and metadata extraction.
"""

import argparse
import time
import logging
import sys
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from config import Config, ConfigError
from prompts import PromptBuilder, PromptError
from io_utils import CSVReader, OutputWriter, IOError
from processing import RecordProcessor, ValidationError, ProcessingError
from llm_clients import create_llm_client, Client, LLMClientError


class ApplicationError(Exception):
    """Custom exception for application-level errors."""

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

        try:
            # Initialize components
            self.llm_client = self._initialize_llm_client()
            self.prompt_builder = self._initialize_prompt_builder()
            self.processor = RecordProcessor(self.llm_client, self.prompt_builder)
            self.reader = self._initialize_csv_reader()
            self.writer = OutputWriter()

            logging.info("Medieval text processor initialized successfully")

        except Exception as e:
            raise ApplicationError(f"Failed to initialize MedievalTextProcessor: {e}") from e

    def _initialize_llm_client(self) -> Client:
        """Initialize the LLM client based on command line arguments.

        Returns:
            Configured LLM client instance.

        Raises:
            ApplicationError: If LLM client initialization fails.
        """
        try:
            client = create_llm_client(self.args.model)
            logging.info("LLM client initialized: %s", self.args.model)
            return client
        except LLMClientError as e:
            raise ApplicationError(f"Failed to initialize LLM client '{self.args.model}': {e}") from e

    def _initialize_prompt_builder(self) -> PromptBuilder:
        """Initialize the prompt builder with template file.

        Returns:
            Configured PromptBuilder instance.

        Raises:
            ApplicationError: If prompt builder initialization fails.
        """
        try:
            prompt_builder = PromptBuilder(self.args.prompt_template)
            logging.info("Prompt builder initialized with template: %s", self.args.prompt_template)
            return prompt_builder
        except PromptError as e:
            raise ApplicationError(f"Failed to initialize PromptBuilder: {e}") from e

    def _initialize_csv_reader(self) -> CSVReader:
        """Initialize the CSV reader for input file.

        Returns:
            Configured CSVReader instance.

        Raises:
            ApplicationError: If CSV reader initialization fails.
        """
        try:
            input_file = self.args.input or Config.INPUT_FILE
            reader = CSVReader(input_file, delimiter=";", encoding="utf-8")
            logging.info("CSV reader initialized for input file: %s", input_file)
            return reader
        except IOError as e:
            raise ApplicationError(f"Failed to initialize CSV reader: {e}") from e

    def process_all_records(self) -> Tuple[List[str], List[str]]:
        """Process all records from the input CSV file.

        Returns:
            Tuple of (all_annotations, all_metadata).

        Raises:
            ApplicationError: If critical processing error occurs.
        """
        all_annotations: List[str] = []
        all_metadata: List[str] = []

        try:
            logging.info("Starting to process records from: %s", self.reader.file_path)
            # Process each record with progress bar

            for record in tqdm(self.reader.stream_records(), desc="Processing Records"):
                try:
                    brevid = record.get("Brevid", "unknown")
                    logging.info("Processing Record with Brevid: %s", brevid)
                    logging.debug("Record data: %s", record)  # DEBUG: print each record

                    # Process the record
                    annotated_record, metadata_record = self.processor.process_record(record)

                    # Collect results
                    all_annotations.extend(annotated_record)
                    all_metadata.extend(metadata_record)

                    logging.debug("Successfully processed Brevid %s: %d annotations, %d metadata",
                                 brevid, len(annotated_record), len(metadata_record))

                except ValidationError as e:
                    brevid = record.get("Brevid", "unknown")
                    logging.error("Validation error for Brevid %s: %s", brevid, e)
                except ProcessingError as e:
                    brevid = record.get("Brevid", "unknown")
                    logging.error("Processing error for Brevid %s: %s", brevid, e)
                except Exception as e:
                    brevid = record.get("Brevid", "unknown")
                    logging.error("error processing record with Brevid %s: %s", brevid, e, exc_info=True)

                # Rate limiting
                time.sleep(0.2)

            logging.info("Completed processing all records")

            return all_annotations, all_metadata

        except Exception as e:
            raise ApplicationError(f"Critical error during file processing: {e}") from e

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
                logging.info("Annotated text written to: %s (%d records)", output_text, len(annotations))
            else:
               logging.warning("No annotated text output to write")

            # Write metadata table output
            if metadata:
                metadata_header = (
                    "Proper Noun;Type of Proper Noun;Order of Occurrence in Doc;"
                    "Brevid;Status/Occupation/Description;Gender;Language"
                )
                self.writer.write_metadata_output(output_table, metadata_header, metadata)
                logging.info("Metadata written to: %s (%d records)", output_table, len(metadata))
            else:
                logging.warning("No metadata output to write")

        except IOError as e:
            raise ApplicationError(f"Failed to write outputs: {e}") from e

    def run(self) -> None:
        """Run the complete processing pipeline.

        Raises:
            ApplicationError: If any step of the pipeline fails.
        """
        try:
            # Process all records
            annotations, metadata = self.process_all_records()

            # Write outputs
            self.write_output(annotations, metadata)

            logging.info("Processing completed successfully")

            # Final output paths
            output_text = self.args.output_text or Config.OUTPUT_TEXT_FILE
            output_table = self.args.output_table or Config.OUTPUT_TABLE_FILE
            print(f"\nOutputs written to:")
            print(f"  Annotated text: {output_text}")
            print(f"  Metadata table: {output_table}")

        except ApplicationError as e:
            logging.critical("Application error: %s", e, exc_info=True)
            raise
        except Exception as e:
            raise ApplicationError(f"Unexpected error in processing pipeline: {e}") from e

def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments.

    Args:
        args: Parsed command line arguments.

    Raises:
        ValueError: If arguments are invalid.
    """
    # Check if input file exists
    input_file = args.input or Config.INPUT_FILE
    if input_file and not Path(input_file).exists():
        raise ValueError(f"Input file does not exist: {input_file}")

    # Check if prompt template exists
    if args.prompt_template and not Path(args.prompt_template).exists():
        raise ValueError(f"Prompt template file does not exist: {args.prompt_template}")

    logging.info("Command line arguments validated successfully")


def validate_configuration() -> None:
    """Validate application configuration.

    Raises:
        ConfigError: If configuration is invalid.
    """
    if not Config.is_valid():
        raise ConfigError(
            "Configuration validation failed. "
            "Please check your .env file or environment variables."
        )
    logging.info("Configuration validated successfully")

def setup_logging(verbose: bool) -> None:
    """Setup application logging configuration.

    Args:
        verbose: Whether to enable verbose (DEBUG) logging.
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Clear any existing handlers to avoid conflicts
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force = True
    )

    logging.info("Logging configured (verbose=%s)", verbose)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Annotate medieval texts by Brevid records with LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
                Examples:
                    python process_medieval_llm.py --model claude
                    python src/process_medieval_llm.py --model ollama --input input/input.txt
                    python src/process_medieval_llm.py --model ollama --output_text output/annotated_output.txt --output_table output/metadata_table.txt
                """
    )
    # Model selection
    parser.add_argument(
        "--model",
        choices=["claude", "ollama"],
        default="claude",
        help="Select LLM backend (default: claude)"
    )

    # File paths
    parser.add_argument(
        "--prompt_template",
        default=Config.PROMPT_TEMPLATE_FILE,
        help="Path to the prompt template file"
    )
    parser.add_argument(
        "--input",
        default=Config.INPUT_FILE,
        help="Path to the input file"
    )
    parser.add_argument(
        "--output_text",
        default=Config.OUTPUT_TEXT_FILE,
        help="Path for annotated text output"
    )
    parser.add_argument(
        "--output_table",
        default=Config.OUTPUT_TABLE_FILE,
        help="Path for metadata table output"
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    # Additional options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and inputs without processing"
    )

    return parser

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------

def main() -> None:
    """Main application entry point."""
    # Add this at the very beginning
    root_logger = logging.getLogger()
    print(f"Handlers before setup_logging: {root_logger.handlers}")
    print(f"Root logger level before setup_logging: {root_logger.level}")

    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()

        # Setup logging
        setup_logging(args.verbose)

        # Validate configuration and arguments
        validate_configuration()
        validate_arguments(args)

        # Handle dry run
        if args.dry_run:
            print("✓ Configuration validated successfully")
            print("✓ Command line arguments validated")
            print("✓ Input files exist and are accessible")
            print("Dry run completed successfully - no processing performed")
            return

        # Initialize and run processor
        processor = MedievalTextProcessor(args)
        processor.run()

        logging.info("Application completed successfully")

    except ConfigError as e:
        logging.error("Configuration error: %s", e)
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    except ValueError as e:
        logging.error("Invalid arguments: %s", e)
        print(f"Invalid arguments: {e}", file=sys.stderr)
        sys.exit(1)

    except ApplicationError as e:
        logging.error("Application error: %s", e)
        print(f"Application error: {e}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        print("\nProcessing interrupted by user", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        logging.critical("Unexpected error: %s", e, exc_info=True)
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()