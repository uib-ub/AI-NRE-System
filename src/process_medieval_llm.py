import argparse
import time
import logging
from tqdm import tqdm

# from llm_clients import ClaudeClient, OllamaClient
from config import Config, ConfigError
from prompts import PromptBuilder
from io_utils import stream_csv_records, write_metadata_output, write_text_output, CSVReader, OutputWriter
from processing import process_record

from llm_clients import create_llm_client

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------

def main(args):
    """
    Read input (csv) txt file, process each record with LLM, and write outputs

    Args:
        args: Parsed command line arguments
    """
    if not Config.is_valid():
        raise ConfigError("Configuration validation failed. "
                          "Please check your .env file or environment variables.")
    all_annotations = []
    all_metadata= []

    # Initialize LLM client
    llm_client = create_llm_client(args.model)

    # match args.model:
    #    case "claude":
    #        llm_client = ClaudeClient(
    #            api_key=Config.ANTHROPIC_API_KEY,
    #            model=Config.CLAUDE_MODEL
    #        )
    #    case "ollama":
    #        llm_client = OllamaClient(
    #            endpoint=Config.OPENWEBUI_ENDPOINT,
    #            token=Config.OPENWEBUI_TOKEN,
    #            model=Config.OLLAMA_MODEL
    #        )
    #    case _:
    #        raise SystemExit("Please select LLM model by input option --model")

    # Initialize Prompt Builder
    prompt_builder = PromptBuilder(args.prompt_template)

    # Initialize CSVReader
    reader = CSVReader(args.input or Config.INPUT_FILE, delimiter=";", encoding="utf-8")
    logging.info(
        "Processing file: %s",
        args.input or Config.INPUT_FILE,
    )

    try:
        # Stream through each record in the CSV
        # for rec in tqdm(stream_csv_records(args.input or Config.INPUT_FILE), desc="Processing Records"):
        for rec in tqdm(reader.stream_records(), desc="Processing Records"):
            try:
                logging.info("Processing Record with Brevid: %s", rec.get("Brevid"))
                logging.debug(rec)  # DEBUG: print each record
                annotated_rec, metadata_rows = process_record(rec, llm_client, prompt_builder)
                all_annotations.extend(annotated_rec)
                all_metadata.extend(metadata_rows)
            except Exception as e:
                logging.error("Error processing record with Brevid %s: %s", rec.get("Brevid"), e, exc_info=True)
            time.sleep(0.2)
    except Exception as e:
        logging.critical("Critical error during file processing: %s", e, exc_info=True)
        raise

    # Initialize OutputWriter
    writer = OutputWriter()

    # Write annotated text
    annotated_header = "Bindnr;Brevid;Tekst"
    writer.write_text_output(args.output_text or Config.OUTPUT_TEXT_FILE, annotated_header, all_annotations)

    # try:
    #     write_text_output(args.output_text or Config.OUTPUT_TEXT_FILE, annotated_header, all_annotations)
    # except Exception as e:
    #     logging.error("Error writing annotated text output: %s", e, exc_info=True)

    # Write metadata table
    metadata_header = ("Proper Noun;Type of Proper Noun;Order of Occurrence in Doc;"
        "Brevid;Status/Occupation/Description;Gender;Language")
    writer.write_metadata_output(args.output_table or Config.OUTPUT_TABLE_FILE, metadata_header, all_metadata)

   # try:
   #     write_metadata_output(args.output_table or Config.OUTPUT_TABLE_FILE, metadata_header, all_metadata)
   # except Exception as e:
   #     logging.error("Error writing metadata output: %s", e, exc_info=True)

    print(f"Done! Outputs → {args.output_text or Config.OUTPUT_TEXT_FILE}, {args.output_table or Config.OUTPUT_TABLE_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate medieval texts by Brevid records with LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog = """
                Examples:
                    python process_medieval_llm.py --model claude
                    python process_medieval_llm.py --model ollama --input data/input.csv
                """
    )
    parser.add_argument(
        "--model", choices=["claude", "ollama"], default="claude",
        help="Select LLM backend (default: claude)"
    )
    parser.add_argument(
        "--prompt_template", default=Config.PROMPT_TEMPLATE_FILE,
        help="Path to the prompt template file"
    )
    parser.add_argument(
        "--input", default=Config.INPUT_FILE,
        help="Path to the input file"
    )
    parser.add_argument(
        "--output_text", default=Config.OUTPUT_TEXT_FILE,
        help="Path for annotated test output"
    )
    parser.add_argument(
        "--output_table", default=Config.OUTPUT_TABLE_FILE,
        help="Path for metadata table output"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # verbose logging for debugging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    main(args)