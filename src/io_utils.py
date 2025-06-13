import csv
import logging
from typing import List

def stream_csv_records(file_path: str, delimiter: str = ";"):
    """
    Stream records from a file
    Args:
        file_path: Path to the file.
        delimiter: Delimiter used in the file
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for record_row in reader:
                yield record_row
    except Exception as e:
        logging.error("Error reading CSV file %s: %s", file_path, e, exc_info=True)
        raise

def write_text_output(file_path: str, header: str, annotations: List[str]):
    """
    Write annotated text output to a file.
    Args:
        file_path: Output file path.
        header: Header line for the file.
        annotations: List of annotated text records (strings).
    """
    try:
        with open(file_path, "w", encoding="utf-8") as txt:
            txt.write(header + "\n")
            txt.write("\n".join(annotations))
    except Exception as e:
        logging.error("Error writing text output to %s: %s", file_path, e, exc_info=True)
        raise

def write_metadata_output(file_path: str, header: str, metadata: List[str]):
    """
    Write metadata table output to a file.
    Args:
        file_path: Output file path:
        header: Header line for the file.
        metadata: List of metadata rows (strings)
    """
    try:
        with open(file_path, "w", encoding="utf-8") as tbl:
            tbl.write(header + "\n")
            tbl.write("\n".join(metadata))
    except Exception as e:
        logging.error("Error writing metadata output to %s: %s", file_path, e, exc_info=True)
        raise

