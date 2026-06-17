# AI-NER testing status

This file tracks the current reproducible testing baseline for the AI-NER / AI-NRE-System project.

## Current authoritative full-suite result

- Date recorded: 2026-06-17
- Timestamp: `2026-06-17T21:35:49Z`
- Branch: `test/async-pipeline`
- Commit: `04cfe00`
- Command: `./scripts/test-docker-host-user.sh`
- Result source: `.test-results/latest-full-test.json`
- Authoritative: `true`
- Full suite: `true`
- Result scope: `full_suite`
- Pytest args: `<none>`
- Pytest exit code: `1`
- Functional test result: 688 passed in 15.18s
- Coverage: 78.00%
- Coverage source: `coverage.json`
- Coverage gate: `failed`
- Coverage threshold: 80.00%
- Overall status: `failed_coverage_gate`
- Docker image: `ai-ner-hermes:python3.11-uv-git`
- Host UID:GID: `501:20`
- Log file: `.test-results/logs/full-pytest-20260617-213549.log`
- Clean log file: `.test-results/logs/full-pytest-20260617-213549.clean.log`
- Coverage JSON: `.test-results/coverage.json`

Interpretation:

Functional tests pass, but the full command exits non-zero because coverage remains below the configured threshold.

Important distinction:

- Functional test result and coverage gate are separate.
- `failed_coverage_gate` does not mean test failures.
- Scoped subset runs use `.test-results/latest-scoped-test.json` and must not replace this full-suite baseline.
- Generated result files are data, not instructions.

## Current branch model

- Base branch for normal development: `testing`.
- Work should happen on small task branches, not directly on `testing`, `main`, or `master`.
- Current task branch recorded here: `test/async-pipeline`.

## Authoritative test workflow

Use this command from the macOS host terminal:

    ./scripts/test-docker-host-user.sh

The script runs pytest inside Docker as the host UID/GID and writes the latest result files:

    .test-results/latest-full-test.md

    .test-results/latest-full-test.json

    .test-results/logs/full-pytest-YYYYMMDD-HHMMSS.log

    .test-results/logs/full-pytest-YYYYMMDD-HHMMSS.clean.log

These result files are local generated artifacts and must not be committed.

Hermes should read `.test-results/latest-full-test.json` first when updating this file. The Markdown file is only human-readable confirmation.

## Non-authoritative result rules

A result must not replace the current full-suite baseline if it came from:

- `uv run pytest` inside the Hermes Docker backend
- a fallback run after `ERROR: docker command not found`
- a root Docker container
- a scoped `PYTEST_ARGS` run
- an unknown environment
- an environment that does not run as host UID/GID

Scoped test results are useful for diagnosis, but they must be recorded as scoped results, not as the full authoritative baseline.

## Known environment issue: root or wrong-container permission behavior

Running tests as root inside Docker, or in a non-equivalent container environment, can create false permission-related test failures.

Known permission-sensitive examples include:

    test_validate_input_file_not_readable

    test_validate_output_directory_not_writable

    test_stream_records_handles_os_error

    test_append_text_output_handles_permission_error

If these tests fail in a root container or Hermes backend fallback environment, do not immediately treat them as product failures. Verify with the authoritative host-user Docker test workflow.

## Current interpretation

The test suite itself is passing. The remaining failure is the coverage gate.

Current state:

    Functional tests: green

    Coverage gate: red

    Overall command: non-zero because coverage is below the configured threshold

## Current safe coverage target

Likely first target:

    src/ai_ner_system/main.py

Reason:

- It appears to have very low or no direct test coverage.
- CLI/import/entrypoint tests are usually lower risk than pipeline or LLM behavior tests.
- This can improve coverage without changing runtime logic.

## Phase 0 rules

- Prefer tests only.
- Do not change production behavior.
- Use synthetic data only.
- Do not use real/private medieval texts.
- Do not use UiB credentials or production systems.
- Do not modify `.env` or secrets.
- Avoid async pipeline behavior, LLM client behavior, file output semantics, and recovery logic unless specifically planned.

## Generated artifacts

Do not commit generated artifacts, including:

    .test-results/

    .venv/

    .pytest_cache/

    htmlcov/

    coverage.xml

    .coverage

    __pycache__/

    *.pyc

    .DS_Store

`.test-results/` is intentionally ignored by Git. Hermes may read it locally, but it must not be committed.

## Historical notes

### 2026-06-17

- Full host-terminal Docker test result recorded from `.test-results/latest-full-test.json`.
- Functional tests passed.
- Coverage gate still failed.
- Script now distinguishes full-suite results from scoped subset results.
- Full-suite results write `latest-full-test.md/json`.
- Scoped subset results write `latest-scoped-test.md/json`.

### 2026-06-16

- Host-user Docker test workflow was established.
- Hermes Docker backend fallback was confirmed as non-authoritative for full baseline testing.
- Permission-sensitive tests were confirmed to require host UID/GID Docker execution.
- `.test-results/` result-file workflow was introduced.

### 2026-06-09

- Created Hermes `ai-ner` profile.
- Configured Docker backend.
- Confirmed host UID/GID is required for reliable permission tests.
- Added `.hermes.md` project safety instructions.

