import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() # Loads variables from .env

class Config:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENWEBUI_TOKEN = os.getenv("OPENWEBUI_TOKEN")
    OPENWEBUI_ENDPOINT = os.getenv("OPENWEBUI_ENDPOINT")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL")
    INPUT_FILE = os.getenv("INPUT_FILE")
    OUTPUT_TEXT_FILE = os.getenv("OUTPUT_TEXT_FILE")
    OUTPUT_TABLE_FILE = os.getenv("OUTPUT_TABLE_FILE")
    PROMPT_TEMPLATE_FILE = os.getenv("PROMPT_TEMPLATE_FILE")
    CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache_llm"))
    CACHE_DIR.mkdir(exist_ok=True)