import json
import logging

from typing import Any, List, Tuple
from prompts import PromptBuilder

def process_record(record: dict, llm_client: Any, prompt_builder: PromptBuilder) -> Tuple[List[str], List[str]]: 
    """
    Send one record to the LLM, parse results and return 
    annotated text plus metadata rows.

    Args:
        record: Dict with keys "Brevid" and "Tekst"
        llm_client: ClaudeClient or OllamaClient
        prompt_builder: PromptBuilder instance
    Return:
        Tuple (annotated_record, metadata)
    """
    bindnr = record["Bindnr"]
    brevid = record["Brevid"]
    text = record["Tekst"]

    # Build prompt
    prompt = prompt_builder.build(brevid, text)
    # DEBUG: prompt
    logging.debug("--- Prompt ---\n%s", prompt)
    # logging.debug(prompt)

    # Call LLM
    try:
        raw_data = llm_client.call(prompt)
        # DEBUG: raw data
        logging.debug("--- RAW DATA for Brevid %s ---\n%s", brevid, raw_data)
    except Exception as e:
        logging.error("Error during LLM call for Brevid %s: %s", brevid, e, exc_info=True)
        return [], []

    # Process raw data received from LLM
    if "===JSON===" in raw_data:
        annotated_text, json_text = raw_data.split("===JSON===")
    else:
        annotated_text, json_text = raw_data, '{"entities":[]}'

    annotated_text = annotated_text.strip()
    try: 
        entities = json.loads(json_text).get("entities", [])
    except Exception as e:
        logging.error("JSON decode error for Brevid %s: %s", brevid, e, exc_info=True)
        entities = []

    # DEBUG: annotated text and entities
    logging.debug("--- annotated text ---\n%s", annotated_text)
    logging.debug("--- entities ---\n%s", entities)

    annotated_record = []
    annotated_record.append(";".join([
        bindnr,
        brevid,
        annotated_text
    ]))
    # INFO: annotated record
    logging.info("--- annotated record ---\n%s", annotated_record)

    metadata = []
    for ent in entities:
        # Build metadata row directly from JSON
        metadata.append(";".join([
            ent.get("name", ""),
            ent.get("type", ""),
            str(ent.get("order", "")),
            ent.get("brevid", brevid),
            ent.get("description", ""),
            ent.get("gender", ""),
            ent.get("language", "")
        ]))

    # INFO: metadata
    logging.info("--- metadata ---\n%s", metadata)

    return annotated_record, metadata
