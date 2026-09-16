# AI-NER Historical Text System

A Named Entity Recognition (NER) system designed for medieval historical texts in languages with unstandardized orthography — including Old Norse, Latin, etc. The system leverages Large Language Models (LLMs) to extract named entities (persons, places, institutions) and generate structured metadata from digitized historical records.

The system supports two LLM backends — the Anthropic Claude API and Ollama (via OpenWebUI) — and offers both synchronous and asynchronous batch processing modes to balance throughput, cost, and latency for large-scale corpora.

## System Architecture Overview

<img width="1021" height="1111" alt="AI-NER-System excalidraw dark" src="https://github.com/user-attachments/assets/82938153-5046-427a-8fc6-4b009e9d5d28" />

## Project Structure

### Input and Output

- **`input/`** — Contains semicolon-delimited CSV files of medieval historical texts to be processed. Each file has the required headers `Bindnr`, `Brevid`, and `Tekst`. The repository includes small sample inputs such as `Brevid-601-613.txt` and larger corpus files such as `Brevid-DN-AI-sorted.txt`.
- **`output/`** — Stores processing results. Synchronous runs write annotated text and metadata output files. Asynchronous runs also write a statistics file, and in incremental mode the text and metadata files are appended during processing while final statistics are written at the end.
- **`prompt/`** — Stores prompt template files used at runtime. `prompt.txt` is the single-record template used for sync single-record processing and async processing. `prompt-batch.txt` is the synchronous batch template. This top-level directory contains template files only; the prompt-building code itself lives in `src/ai_ner_system/prompt/`.

### Repository Layout

```text
.
├── input/                         # Input CSV data files
├── output/                        # Generated output files
├── prompt/                        # Prompt template files
├── src/
│   └── ai_ner_system/
│       ├── config/               # Configuration loading and validation
│       ├── file_io/              # CSV reading and output writing
│       ├── llm/                  # LLM client abstraction and providers
│       ├── pipeline/             # Sync/async orchestration
│       ├── processing/           # Core record processing and parsing
│       ├── prompt/               # Prompt-building code
│       ├── main.py               # CLI entry point
│       └── py.typed              # Typing marker for the package
├── tests/
│   ├── unit/                     # Unit tests mirroring package structure
│   └── integration/              # End-to-end/integration tests
├── .env.example                  # Example runtime configuration
├── .pre-commit-config.yaml       # Pre-commit hooks
├── .python-version               # Python version pin for local tooling
├── pyproject.toml                # Project metadata and tool configuration
└── uv.lock                       # Locked dependency resolution for uv
```

## Package Overview

All source code lives under `src/ai_ner_system/`. `main.py` is the CLI entry
module, and the rest of the codebase is organized into six packages. At a high
level, `config` loads and validates runtime settings, `pipeline` orchestrates
sync and async execution, `file_io` handles CSV input and output writing,
`prompt` builds prompt strings, `processing` owns the core NER business logic,
and `llm` hides provider-specific APIs behind a common client interface.

Most packages define their own `exceptions.py` module; the `pipeline` package
instead defines its application-level error in `pipeline/stats.py`.

### Application Core (`main.py` module + `pipeline/` package)

Entry point and workflow orchestration.

- **`main.py`** — Parses CLI arguments (`--client`, `--batch-size`, `--async-mode`, output paths, log level, `--dry-run`, etc.), sets up logging, validates configuration, and dispatches to sync or async execution.
- **`pipeline/main_processor.py`** — Defines `MedievalTextProcessor`, which wires together the shared components (`llm_client`, `prompt_builder`, `RecordProcessor`, `CSVReader`, `OutputWriter`) and exposes `run()` and `run_async()` as the main sync/async entry points. It also owns final output writing and async stats writing.
- **`pipeline/sync_processor.py`** — Implements synchronous workflows. It streams CSV records, decides between individual and sync-batch mode, processes batches sequentially, and falls back to per-record processing when a sync batch fails.
- **`pipeline/async_processor.py`** — Implements asynchronous workflows. If the selected client supports async batches, it schedules batches with `asyncio.create_task`, limits in-flight concurrency, and preserves ordered writes for incremental output. Otherwise it falls back to concurrent individual-record processing.
- **`pipeline/processor_protocol.py`** — Defines `ProcessorContext`, the protocol that `SyncProcessor` and `AsyncProcessor` depend on. This avoids circular imports between pipeline modules.
- **`pipeline/stats.py`** — Defines `AsyncProcessingStats` for async run metrics and `ApplicationError` for pipeline/application-level failures.

### Configuration Management (`config/`)

Environment loading and validation.

- **`config/settings.py`** — Loads `.env`, stores runtime settings, applies CLI overrides, creates output/cache directories, and returns validated client initialization parameters. `CACHE_DIR` is currently created but not used for runtime LLM response caching.
- **`config/validation.py`** — Runs pre-flight validation for client-specific settings, input/template files, and output paths before processing starts.
- **`config/exceptions.py`** — Configuration-specific exception hierarchy.

### Input/Output Layer (`file_io/`)

File reading and writing.

- **`file_io/csv_reader.py`** — Streams semicolon-delimited CSV input, validates required headers, and skips invalid or empty rows with warnings.
- **`file_io/output_writers.py`** — Provides atomic full-file output and incremental append mode. Incremental appends use POSIX `fcntl` file locking, so that path is naturally Unix-oriented.
- **`file_io/exceptions.py`** — File I/O-specific exception hierarchy.

### Processing Engine (`processing/`)

Core NER processing logic.

- **`processing/processor.py`** — `RecordProcessor` orchestrates LLM calls for sync single-record processing, sync batch processing through one combined prompt, async single-record processing, and async batch processing through the client batch API when available.
- **`processing/parser.py`** — Parses LLM responses into annotated text and structured entity records.
- **`processing/validator.py`** — Validates required input record fields before processing.
- **`processing/entities.py`** — Defines the main typed result models: `EntityRecord`, `ProcessingResult`, and `BatchProcessingResult`.
- **`processing/exceptions.py`** — Processing-specific exception hierarchy.

### Prompt Management (`prompt/`)

Template-based prompt construction.

- **`prompt/builder.py`** — Loads prompt templates and formats them with record data. Single-record templates use `{brevid}` and `{text}` and are used for sync single-record processing and async processing. Sync batch templates use `{num_records}` and `{batch_content}` and are used only for synchronous batch processing.
- **`prompt/exceptions.py`** — Prompt/template-specific exception hierarchy.

### LLM Integration Layer (`llm/`)

Abstract client interface and concrete implementations.

- **`llm/base_client.py`** — Defines the shared client interface for sync calls, async calls, and optional batch APIs.
- **`llm/claude_client.py`** — Anthropic Claude implementation, including async message-batch support.
- **`llm/ollama_client.py`** — OpenWebUI/Ollama implementation for sync and async single requests; it does not support async batch processing.
- **`llm/factory.py`** — Chooses the concrete client from the selected provider and validated settings.
- **`llm/batch_models.py`** — Shared batch request, response, status, and progress models.
- **`llm/exceptions.py`** — LLM/client/provider-specific exception hierarchy.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Install Dependencies

```bash
# Default local development environment
uv sync

# With selected development dependency groups
uv sync --group lint --group test

# With documentation dependencies
uv sync --group docs

# All dependency groups (lint, test, security, docs, profiling, dev)
uv sync --all-groups
```

Development dependency groups:

- **lint** — ruff, mypy, codespell, vulture, validate-pyproject
- **test** — pytest, pytest-cov, pytest-asyncio, pytest-mock, allure-pytest
- **security** — bandit, safety
- **docs** — sphinx, sphinx-rtd-theme, myst-parser
- **profiling** — memory-profiler, py-spy, line-profiler
- **dev** — lint, test, docs, IPython, Jupyter, pre-commit, build, twine

### Updating Dependencies After Security Alerts

When GitHub Dependabot reports a vulnerable package in `uv.lock`, update both the
project constraints and the lockfile as needed.

1. Identify whether the vulnerable package is a direct dependency or only appears
   transitively in `uv.lock`.
2. If it is a direct dependency, raise its minimum version in `pyproject.toml`
   to the patched release or later.
3. Refresh the lockfile with `uv lock --upgrade-package ...` so `uv.lock`
   resolves to non-vulnerable versions.
4. Run targeted tests for the affected area before committing.
5. Push the updated `pyproject.toml` and `uv.lock` to GitHub. Dependabot alerts
   should close after GitHub reprocesses the default branch.

Typical workflow:

```bash
# 1. Inspect the affected package and current resolved versions
rg -n "anthropic|cryptography" pyproject.toml uv.lock src tests

# 2. If needed, update the direct dependency floor in pyproject.toml
# Example:
# anthropic>=0.87.0

# 3. Refresh the lockfile for the affected packages
uv lock --upgrade-package anthropic --upgrade-package cryptography

# 4. Sync the environment if needed
uv sync --all-extras

# 5. Run targeted verification
uv run pytest tests/unit/llm/test_claude_client.py \
  tests/unit/llm/test_factory.py \
  tests/unit/config/test_settings.py

# 6. Review the resulting changes
git diff -- pyproject.toml uv.lock
```

Notes:

- If the vulnerable package is only transitive, start with `uv lock --upgrade-package <name>`.
- If the resolver cannot move the transitive package to a safe version, inspect
  which direct dependency is constraining it and bump that dependency instead.
- GitHub security alerts for Python dependencies in this project are driven by
  the resolved versions in `uv.lock`, not just by `pyproject.toml`.

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
| `OUTPUT_STATS_FILE` | Async processing statistics output path |
| `PROMPT_TEMPLATE_FILE` | Single-record prompt template |
| `BATCH_TEMPLATE_FILE` | Batch prompt template |
| `CACHE_DIR` | LLM response cache directory |

Note: `CACHE_DIR` currently is not used for runtime LLM response caching
but only for creating the directory.

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
    --async-mode \
    --incremental-output
```

**Exit codes for async incremental runs**:

- `0` — clean success; output is complete.
- `1` — fatal failure; inspect logs.
- `2` — partial success: one or more batches' incremental writes failed and were skipped. The dropped record IDs are listed in `failed_batch_writes` inside the stats JSON.

> **⚠️ On exit code 2, do not naïvely re-run with the same output paths.** Annotations and metadata are written concurrently per batch, so a "failed" batch may have one side already on disk; re-running would duplicate that side. Use the **fresh-output-paths + post-process merge** procedure in `docs/refactoring-docs/ASYNC_PROCESSOR_LARGE_RUN_RECOMMENDATIONS.md` ("Suggested Operational Workflow Now") before treating recovery as complete. This caveat goes away once the commit-state manifest and `--resume` mode land (Priorities 4–5 in `docs/refactoring-docs/LARGE_RUN_HARDENING_ANALYSIS_AND_PLAN.md`).

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

Tests are organized under `tests/` mirroring the `src/ai_ner_system/` package structure.

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

# Debug mode

uv run pytest tests/unit/config/test_settings.py::TestSettings::test_init --log-cli-level=DEBUG -v


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

Multiple records are grouped into batches and sent in a single synchronous LLM call. This is a pipeline-level mode, so it can use whichever configured client is selected, assuming the selected prompt template and model response format work for that client. Batches are processed sequentially. If a batch fails, the system automatically falls back to individual processing for the records in that batch.

```bash
uv run python -m ai_ner_system.main --client claude --use-batch --batch-size 10
```

### Async Batch Processing

Batches are processed concurrently with `asyncio.create_task` and a task dictionary that limits the number of in-flight batches and awaits the oldest batch when the concurrency limit is reached. `asyncio.TaskGroup` is used in related async paths, such as concurrent output writing and incremental write operations. Async batch processing supports incremental output (results written to files as batches complete) and order-preserving result queuing. Available with the Claude client.

```bash
uv run python -m ai_ner_system.main --client claude --async-mode --batch-size 10 --incremental-output
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

Compared to individual processing (10 records, batch size 10): ~75% faster (1:34 vs 6:15) and ~45% cheaper ($0.17 vs $0.31). Compared to sync batch (10 records, batch size 10): ~71% faster (1:34 vs 5:23) and ~41% cheaper ($0.17 vs $0.29). At scale (551 records, batch size 100), projected full corpus processing drops from ~194 hours (individual) to ~7.7 hours — a **96% reduction** in processing time.


Text
