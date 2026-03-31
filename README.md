# AI-NER Historical Text System

A Named Entity Recognition (NER) system designed for medieval historical texts in languages with unstandardized orthography — including Old Norse, Latin, Middle Norse, and Middle Dutch. The system leverages Large Language Models (LLMs) to extract named entities (persons, places, institutions) and generate structured metadata from digitized historical records.

The system supports two LLM backends — the Anthropic Claude API and Ollama (via OpenWebUI) — and offers both synchronous and asynchronous batch processing modes to balance throughput, cost, and latency for large-scale corpora.

## System Architecture Overview

<img width="1021" height="1111" alt="AI-NER-System excalidraw dark" src="https://github.com/user-attachments/assets/82938153-5046-427a-8fc6-4b009e9d5d28" />

## Project Structure

### Input and Output

- **`input/`** — Contains semicolon-delimited CSV files of medieval historical texts to be processed. Each file has the required headers `Bindnr`, `Brevid`, and `Tekst`. Files range from small test sets (e.g., `Brevid-601-610.txt` with 10 records) to the full corpus (`Brevid-DN-AI.txt` with 18,559 records).
- **`output/`** — Stores processing results. Each run produces up to three output files: annotated text (`annotated_output_*.txt`), metadata tables (`metadata_table_*.txt`) with extracted entities, and processing statistics (`processing_stats.json` for async runs).
- **`prompt/`** — Prompt template files used by the prompt builder. Includes single-record templates (e.g., `prompt-v0.1.txt`) and batch templates (e.g., `prompt-batch.txt`).

## Module Overview

All source code lives under `src/ai_ner_system/`. The entry point is `main.py`, and the codebase is organized into six packages — each corresponding to a component in the architecture diagram above. Every package includes its own `exceptions.py` with a module-specific exception hierarchy.

### Application Core (`main.py` + `pipeline/`)

Entry point and orchestration layer.

- **`main.py`** — CLI entry point. Defines all command-line arguments via `argparse` (client type, input/output paths, batch size, async mode, concurrency limits, log level, `--dry-run`, etc.), sets up logging, validates configuration, and launches the processing pipeline. Returns exit code 0 on success, 1 on failure.
- **`pipeline/main_processor.py`** — `MedievalTextProcessor` orchestrator class. Initializes all components (LLM client via factory, prompt builder, CSV reader, output writer) and coordinates sync or async processing. Defines output file headers (`ANNOTATED_HEADER`, `METADATA_HEADER`) and manages output file cleanup before each run.
- **`pipeline/sync_processor.py`** — `SyncProcessor` processes records sequentially with `tqdm` progress bars. Supports both individual and batch modes. Includes automatic fallback from batch to individual processing on failure, with configurable rate-limiting delays (`BATCH_PROCESSING_DELAY = 0.2s`) between batches.
- **`pipeline/async_processor.py`** — `AsyncProcessor` uses `asyncio.TaskGroup` for concurrent batch processing. Features configurable concurrency limits (`max_concurrent_batches`, `max_concurrent_individual`, `fallback_concurrency`), order-preserving result queuing via `_batch_result_queue`, and incremental output (streaming results to files as batches complete). Falls back to individual processing with reduced concurrency on batch failure.
- **`pipeline/processor_protocol.py`** — `ProcessorContext` protocol that defines the interface `SyncProcessor` and `AsyncProcessor` depend on, breaking circular imports between pipeline modules. Specifies required attributes, class constants, and output file path properties.
- **`pipeline/stats.py`** — `AsyncProcessingStats` dataclass for tracking async run metrics: total/processed/failed records, timing, `success_rate`, `throughput` (records/sec), and `summary()` for JSON output.

### Configuration Management (`config/`)

Environment loading and validation.

- **`config/settings.py`** — `Settings` singleton that loads environment variables from `.env` via `python-dotenv`. Provides `DEFAULT_*` class constants (e.g., `DEFAULT_BATCH_SIZE=5`, `DEFAULT_MAX_CONCURRENT_BATCHES=5`) and dynamic class-level attributes for all configuration. Key methods: `initialize()` (loads `.env`, creates directories), `apply_cli_overrides()` (applies argparse args), `get_client_init_params()` (returns validated parameters for client construction), `reset()` (for testing). Maintains a `_CLIENT_CONFIG_REGISTRY` mapping client types to their required configuration.
- **`config/validation.py`** — `ConfigValidator` with static methods for pre-flight checks: `validate_for_client()` (client-specific validation), `validate_file_paths()` (input file, template files), `validate_all()` (comprehensive validation), `is_valid()` (non-throwing check). Validates that API keys are set, files are readable, and output directories are writable.
- **`config/exceptions.py`** — `ConfigError` (base, with optional `config_key`), `ConfigValidationError` (with `missing_keys` list), `FileValidationError` (with `file_path` and `validation_type`), `DirectoryValidationError` (with `directory_path`).

### Input/Output Layer (`file_io/`)

File reading and writing.

- **`file_io/csv_reader.py`** — `CSVReader` provides streaming, semicolon-delimited (`;`) CSV reading. Constructor accepts file path, delimiter, encoding, and optional `required_headers` frozenset. Key method `stream_records()` yields validated `dict` records. Validates file existence and size, required headers, and individual rows (skips empty rows with warnings).
- **`file_io/output_writers.py`** — `OutputWriter` supports two write modes. Atomic write methods (`write_text_output()`, `write_metadata_output()`) use tempfile + `os.replace` for safe full-file output. Append methods (`append_text_output()`, `append_metadata_output()`) use POSIX `fcntl` file locking for concurrent incremental writes. Static method `write_stats_output()` writes JSON stats atomically. Append operations automatically emit headers when files are empty.
- **`file_io/exceptions.py`** — `FileIOError` (base), `CSVError` (with `line_number`), `OutputError` (with `output_type`), `FileValidationError` (with `validation_type`), `EncodingError` (with `encoding`).

### Processing Engine (`processing/`)

Core NER processing logic.

- **`processing/processor.py`** — `RecordProcessor` orchestrates LLM calls for individual records and batches. Constructor takes an `llm_client` and `prompt_builder`. Provides sync (`process_record()`) and async (`aprocess_record()`) methods that validate records, build prompts, call the LLM, parse responses, and format output into `ProcessingResult` objects.
- **`processing/parser.py`** — `ResponseParser` with static methods to extract structured data from LLM responses. `parse_llm_response()` extracts annotated text and JSON entities (delimited by `===JSON===` markers). `parse_entities_json()` converts JSON into `EntityRecord` lists. `parse_batch_response()` handles multi-record batch results (split by `RECORD` markers). `format_csv_row()` produces canonical semicolon-delimited output.
- **`processing/validator.py`** — `RecordValidator` ensures data integrity. Checks that all `REQUIRED_FIELDS` (`Bindnr`, `Brevid`, `Tekst`) are present and non-empty. Supports single-record (`validate_record()`) and list (`validate_records()`) validation with record index tracking.
- **`processing/entities.py`** — Data models. `EntityRecord` dataclass with fields: `name`, `entity_type`, `preposition`, `order`, `brevid`, `description`, `gender` (constrained to `Male`/`Female`/`N/A`), `language`. Factory method `create_entity_record()` constructs from dicts with validation. `ProcessingResult` tracks individual record outcomes (`record_id`, `brevid`, `annotated_text`, `entities`, `processing_time`, `success`, `error_message`). `BatchProcessingResult` aggregates batch outcomes with success/failure counts.
- **`processing/exceptions.py`** — `ProcessingError` (base, with `brevid` and `operation`), `ValidationError` (with `missing_fields`), `LLMResponseError` (with `response_text`), `ParseError` (with `parse_type` and `content`), `BatchProcessingError` (with `batch_id`).

### Prompt Management (`prompt/`)

Template-based prompt construction.

- **`prompt/builder.py`** — `PromptBuilder` ABC and `GenericPromptBuilder` implementation. Loads prompt templates from files and formats them with record data using placeholders: `{Brevid}` and `{Tekst}` for single-record templates; `{num_records}` and `{batch_content}` for batch templates. Includes template validation via `_extract_placeholders()` and `_require_template_fields()` to ensure templates contain expected fields.
- **`prompt/exceptions.py`** — `PromptError` (base, with `template_file` and `operation`), `TemplateNotFoundError` (for missing template files), `PromptBuildError` (with `data_type` distinguishing `"single"` vs `"batch"` processing).

### LLM Integration Layer (`llm/`)

Abstract client interface and concrete implementations.

- **`llm/base_client.py`** — `Client` abstract base class defining the LLM interface. Requires `call(prompt)` (sync) and `call_async(prompt)` (async) for single requests. Provides optional batch API methods (`create_batch_async()`, `get_batch_status_async()`, `get_batch_results_async()`, `cancel_batch_async()`, `monitor_batch_progress_async()`) that raise `NotImplementedError` by default. Includes `wait_for_batch_completion_async()` and `process_batch_requests_async()` for end-to-end batch orchestration.
- **`llm/claude_client.py`** — `ClaudeClient` implements the Anthropic Claude API. Supports both sync and async single calls, plus full batch processing via the Message Batches API (`create_batch_async()`, `get_batch_results_async()`, `monitor_batch_progress_async()` as an async generator yielding `BatchProgress`). Configurable `max_tokens` (default 20000) and `temperature` (default 0.0). Handles authentication errors, rate limiting, and API errors.
- **`llm/ollama_client.py`** — `OllamaClient` connects to Ollama via OpenWebUI HTTP API. Supports sync (`call()` via `requests`) and async (`call_async()` via `aiohttp`) single calls. Does not support batch processing. Configurable `timeout` (default 3 hours) and `temperature`. Uses Bearer token authentication.
- **`llm/factory.py`** — `create_llm_client(client_type)` factory function. Maps `"claude"` to `ClaudeClient` and `"ollama"` to `OllamaClient` via a `_CLIENT_CLASSES` registry. Retrieves validated initialization parameters from `Settings.get_client_init_params()`.
- **`llm/batch_models.py`** — Data models for batch processing. `BatchStatus` enum (`IN_PROGRESS`, `ENDED`, `CANCELING`), `BatchRequest` (with `custom_id`, `prompt`, `max_tokens`, `temperature`), `BatchResponse` (with `custom_id`, `response_text`, `success`, `error_message`), `BatchProgress` (tracking `batch_num`, `batch_id`, `status`, `elapsed_time`, `request_counts`).
- **`llm/exceptions.py`** — `LLMClientError` (base), `APIError` (with `status_code`, `response_text`, `request_id`, `is_retryable()`), `LLMConnectionError` (with `endpoint`), `AuthenticationError`, `RateLimitError` (with `retry_after`, `limit_type`), `BatchTimeoutError` (with `batch_id`, `timeout_seconds`), `BatchProcessingError` (with `batch_id`, `failed_requests`).

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Install Dependencies

```bash
# Core dependencies
uv sync

# With development tools
uv sync --extra lint --extra test

# All optional groups (lint, test, security, docs)
uv sync --all-extras
```

Optional dependency groups:

- **lint** — ruff, mypy, codespell, vulture, validate-pyproject
- **test** — pytest, pytest-cov, pytest-asyncio, pytest-mock, allure-pytest
- **security** — bandit, safety
- **docs** — sphinx, sphinx-rtd-theme, myst-parser

### Environment Configuration

```bash
cp .env.example .env
# Edit .env and add your API keys / endpoints
```

Key variables in `.env.example`:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key (required for Claude) |
| `CLAUDE_MODEL` | Claude model name (default: `claude-sonnet-4-20250514`) |
| `OPENWEBUI_TOKEN` | Auth token for OpenWebUI/Ollama (required for Ollama) |
| `OPENWEBUI_ENDPOINT` | OpenWebUI API endpoint URL |
| `OLLAMA_MODEL` | Ollama model name (default: `gemma3:12b-it-qat`) |
| `INPUT_FILE` | Input CSV file path |
| `OUTPUT_TEXT_FILE` | Annotated text output path |
| `OUTPUT_TABLE_FILE` | Metadata table output path |
| `PROMPT_TEMPLATE_FILE` | Single-record prompt template |
| `BATCH_TEMPLATE_FILE` | Batch prompt template |
| `CACHE_DIR` | LLM response cache directory |

### Linting and Formatting

```bash
uv run ruff check src/ tests/          # Lint
uv run ruff check --fix src/ tests/    # Lint with auto-fix
uv run ruff format src/ tests/         # Format
uv run mypy src/                       # Type check
```

### Pre-commit Hooks

```bash
pre-commit install
```

The following hooks run automatically on each commit:

- **ruff** — linting (with `--fix`) and formatting
- **mypy** — strict type checking
- **bandit** — security scanning
- **pyupgrade** — Python 3.11+ syntax upgrades
- **General checks** — valid AST, merge conflict markers, trailing whitespace, line endings, private key detection, file size limits (500KB), debug statements

### CI/CD

GitHub Actions workflows in `.github/workflows/`:

- **`ci.yml`** — Runs on push to `main`/`master` and pull requests. Two jobs:
  1. **Code Quality** (Python 3.11): Ruff linter, Ruff format checker, and MyPy type checking.
  2. **Unit Tests** (Python 3.11 / 3.12 / 3.13 matrix): Runs pytest with coverage reporting (minimum 10% threshold) and uploads results to Codecov.

- **`security.yml`** — Runs weekly (Monday 9:00 UTC) and on pushes affecting `src/`, `pyproject.toml`, or `uv.lock`. Runs Bandit security scanning (fails on high-severity issues) and Safety vulnerability checks. Reports are uploaded as artifacts with 30-day retention.

- **`codeql.yml`** — Runs weekly (Monday 9:00 UTC), on push/PR to `main`/`master` affecting `src/` or `pyproject.toml`, and on manual dispatch. Performs GitHub CodeQL static analysis for Python using the `security-and-quality` query suite to detect vulnerabilities and code quality issues.

## Usage

The system supports two LLM backends. Configure your chosen backend in `.env` (see `.env.example`), then run via the CLI:

```bash
uv run python -m ai_ner_system.main --client {claude,ollama} [options]
```

Use `--dry-run` to validate configuration without processing.

### Example 1: Claude API with async batch processing

```bash
uv run python -m ai_ner_system.main \
    --client claude \
    --output-text output/annotated_output_claude_batch_13R_B2_async_incremental.txt \
    --output-table output/metadata_table_claude_batch_13R_B2_async_incremental.txt \
    --output-stats output/stats_claude_batch_13R_B2_async_incremental.txt \
    --batch-size 2 \
    --async \
    --incremental-output
```

### Example 2: Ollama with sync batch processing

```bash
uv run python -m ai_ner_system.main \
    --client ollama \
    --output-text output/annotated_output_gemma_batch_10R_B2.txt \
    --output-table output/metadata_table_gemma_batch_10R_B2.txt \
    -l DEBUG \
    --use-batch \
    --batch-size 2
```

### Example 3: Ollama with individual record processing

```bash
uv run python -m ai_ner_system.main \
    --client ollama \
    --output-text output/annotated_output_gemma_batch_13R_B1.txt \
    --output-table output/metadata_table_gemma_batch_13R_B1.txt \
    -l DEBUG
```

## Testing

Tests are organized under `tests/` mirroring the `src/ai_ner_system/` module structure.

### Running Tests

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest tests/unit/

# Tests for a specific module
uv run pytest tests/unit/config/
uv run pytest tests/unit/llm/
uv run pytest tests/unit/file_io/
uv run pytest tests/unit/processing/
uv run pytest tests/unit/prompt/

# A single test
uv run pytest tests/unit/config/test_settings.py::TestSettings::test_init -v

# Async tests only
uv run pytest -m asyncio

# Integration tests
uv run pytest tests/integration/
```

### Coverage

```bash
# Run with coverage report
uv run pytest --cov=src/ai_ner_system --cov-report=html

# View the HTML report
open htmlcov/index.html
```

### Test Structure

```text
tests/
├── conftest.py                    # Session-level fixtures, environment isolation
├── unit/
│   ├── conftest.py                # Shared unit fixtures (mock envs, temp files, templates)
│   ├── config/                    # Settings and ConfigValidator tests
│   ├── file_io/                   # CSVReader and OutputWriter tests
│   ├── llm/                       # Client, factory, and batch model tests
│   ├── processing/                # Processor, parser, validator, entity tests
│   │   └── conftest.py            # Processing-specific fixtures
│   ├── prompt/                    # PromptBuilder tests
│   └── pipeline/
│       └── conftest.py            # Pipeline-specific fixtures
└── integration/
    └── test_pipeline.py           # End-to-end pipeline tests
```

## Processing Modes

The system supports three processing modes, each offering different trade-offs between speed, cost, and complexity.

### Individual Processing

Each record is sent to the LLM in a separate API call. This is the default mode when no batch flags are set.

```bash
uv run python -m ai_ner_system.main --client claude
```

### Sync Batch Processing

Multiple records are grouped into batches and sent in a single API call. Batches are processed sequentially. If a batch fails, the system automatically falls back to individual processing for the records in that batch.

```bash
uv run python -m ai_ner_system.main --client claude --use-batch --batch-size 10
```

### Async Batch Processing

Batches are processed concurrently using `asyncio.TaskGroup` with configurable concurrency limits. Supports incremental output (results written to files as batches complete) and order-preserving result queuing. Available with the Claude client.

```bash
uv run python -m ai_ner_system.main --client claude --async --batch-size 10 --incremental-output
```

### Performance Benchmarks

All benchmarks below use the Claude API on a corpus of 10 sample records, with projections for the full corpus of 18,559 records.

#### Individual (baseline)

| Records | Time | Cost | Projected Time (18,559 records) |
|---------|------|------|---------------------------------|
| 10 | 6:15 | $0.31 | ~194 hours (8.1 days) |

#### Sync Batch

| Records | Batch Size | Time | Cost | Input Tokens | Output Tokens |
|---------|------------|------|------|--------------|---------------|
| 10 | 3 | 5:55 | $0.42 | 6,626 | 18,235 |
| 10 | 5 | 5:26 | $0.29 | 5,654 | 18,118 |
| 10 | 10 | 5:23 | $0.29 | 5,168 | 18,177 |

Compared to individual processing (batch size 10): ~14% faster (5:23 vs 6:15) and ~6.5% cheaper ($0.29 vs $0.31).

#### Async Batch

| Records | Batch Size | Time | Cost | Projected Time (18,559 records) |
|---------|------------|------|------|---------------------------------|
| 10 | 10 | 1:34 | $0.17 | ~48.5 hours (2.0 days) |
| 50 | 50 | 2:04 | $1.18 | ~12.8 hours (0.5 days) |
| 100 | 100 | 2:04 | $1.75 | ~8.0 hours (0.3 days) |
| 103 | 10 | 4:40 | — | ~11 hours |
| 551 | 100 | 13:47 | ~$12 | ~7.7 hours (0.3 days), ~$404 |
