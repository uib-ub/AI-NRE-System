# AI-NER Testing Status

This file tracks the current reproducible testing baseline for the AI-NER / AI-NRE-System project.

## Current branch model

- Base branch for normal development: `testing`
- Work should happen on small task branches, not directly on `testing`.
- The Hermes setup branch `hermes/setup-sandbox` is only for initial Hermes safety setup.

## Current baseline

Date recorded: 2026-06-09

Environment:

- Local working copy: sandbox clone
- Docker image: `ai-ner-hermes:python3.11-uv-git`
- Python inside Docker: 3.11.14
- uv inside Docker: 0.9.30
- Test runner: pytest
- Test command: `uv run pytest`
- Important rule: run Docker as the host UID/GID, not root.

Result:

- Tests collected: 688
- Tests passed: 688
- Failed tests: 0
- Functional test status: green
- Coverage: 77.64%
- Required coverage: 80%
- Overall command status: fails because the coverage gate is not met

## Known environment issue

Running tests as root inside Docker can create false permission-related test failures.

Observed false-failure examples from root Docker run:

- `test_validate_input_file_not_readable`
- `test_validate_output_directory_not_writable`
- `test_stream_records_handles_os_error`
- `test_append_text_output_handles_permission_error`

These disappeared when the container was run as the host UID/GID.

## Recommended Docker test command

Use:

```bash
./scripts/test-docker-host-user.sh
```

If the script is not available, run:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -e UV_LINK_MODE=copy \
  -v "$PWD:/workspace/ai-ner:rw" \
  -w /workspace/ai-ner \
  ai-ner-hermes:python3.11-uv-git \
  bash -lc 'uv sync --frozen && uv run pytest'
```

## Current interpretation

The test suite itself is passing. The remaining failure is the coverage gate:

```text
77.64% < 80%
```

This means the next task should focus on safe coverage improvement, preferably by adding meaningful tests rather than changing production behavior.

## Current safe coverage target

Likely first target:

```text
src/ai_ner_system/main.py
```

Reason:

- It appears to have 0% coverage.
- CLI/import/entrypoint tests are usually lower risk than pipeline or LLM behavior tests.
- This can improve coverage without changing runtime logic.

## Phase 0 rules

- Prefer tests only.
- Do not change production behavior.
- Use synthetic data only.
- Do not use real/private medieval texts.
- Do not use UiB credentials or production systems.
- Do not modify `.env` or secrets.
- Avoid async pipeline, LLM client behavior, file output semantics, and recovery logic unless specifically planned.

## Latest updates

### 2026-06-09

- Created Hermes `ai-ner` profile.
- Configured Docker backend.
- Confirmed Docker test workflow.
- Confirmed host UID/GID is required for reliable permission tests.
- Added `.hermes.md` project safety instructions.
