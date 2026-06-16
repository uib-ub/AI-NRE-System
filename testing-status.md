# AI-NER Testing Status

This file tracks the current reproducible testing baseline for the AI-NER / AI-NRE-System project.

---

## Current branch model

- Base branch for normal development: `testing`.
- Work should happen on small task branches, not directly on `testing`, `main`, or `master`.
- The Hermes setup branch `hermes/setup-sandbox` was only for initial Hermes safety setup.
- Current task branch recorded here: `test/async-pipeline`.

---

## Current authoritative baseline

Date recorded: 2026-06-16

Authoritative result source:

```text
.test-results/latest-full-test.md
.test-results/latest-full-test.json
```

Environment:

- Local working copy: macOS host terminal, sandbox clone
- Sandbox path: `/Users/ruiwang/Workspace/uib/hermes-sandboxes/AI-NRE-System-Harness-Sandbox`
- Docker image: `ai-ner-hermes:python3.11-uv-git`
- Python inside Docker: 3.11.14
- uv inside Docker: not recorded in latest result file
- Test runner: pytest 9.0.3
- Test command: `./scripts/test-docker-host-user.sh`
- Important rule: run Docker as the host UID/GID, not root.
- Host UID:GID from latest result: `501:20`
- Commit from latest result: `b81aa69`

Latest authoritative result:

- Timestamp: `2026-06-16T20:00:34Z`
- Branch: `test/async-pipeline`
- Tests collected: 688
- Tests passed: 688
- Failed tests: 0
- Functional test status: green
- Coverage: 77.80%
- Required coverage: 80%
- Coverage gate: failed
- Overall command status: `failed_coverage_gate`
- Pytest exit code: 1

Interpretation:

```text
Functional tests pass, but the full command exits non-zero because the coverage gate fails.
```

---

## Authoritative test workflow

Use this command from the macOS host terminal:

```bash
./scripts/test-docker-host-user.sh
```

This script runs pytest inside Docker as the host UID/GID and writes the latest result files:

```text
.test-results/latest-full-test.md
.test-results/latest-full-test.json
.test-results/logs/full-pytest-YYYYMMDD-HHMMSS.log
.test-results/logs/full-pytest-YYYYMMDD-HHMMSS.clean.log
```

These result files are local generated artifacts and must not be committed.

Hermes should read:

```text
.test-results/latest-full-test.md
.test-results/latest-full-test.json
```

when updating this file.

Hermes should not require Rui to copy/paste terminal output if these files exist.

---

## Important Hermes Docker backend limitation

Hermes must not run the authoritative full Docker test command from inside its Docker backend:

```bash
./scripts/test-docker-host-user.sh
```

Reason:

- Hermes terminal commands run inside a Docker backend container.
- `scripts/test-docker-host-user.sh` itself calls the `docker` command.
- The Hermes Docker backend does not have access to the macOS Docker CLI/daemon.
- The project intentionally does not mount the host Docker socket into Hermes for security reasons.

If Hermes tries to run the script inside its Docker backend, it may fail with:

```text
ERROR: docker command not found.
```

That result is not an authoritative test baseline.

Correct workflow:

1. Hermes plans the test.
2. Rui runs `./scripts/test-docker-host-user.sh` from the macOS host terminal.
3. The script writes `.test-results/latest-full-test.md` and `.test-results/latest-full-test.json`.
4. Hermes reads those files.
5. Hermes updates `testing-status.md`.

---

## Non-authoritative test results

A result should not be used as the official baseline if it came from:

- `uv run pytest` inside the Hermes Docker backend
- a fallback run after `ERROR: docker command not found`
- a root Docker container
- a partial `PYTEST_ARGS` run presented as the full project baseline
- an unknown environment
- an environment that does not run as host UID/GID

Non-authoritative results may be useful for diagnosis, but they must be clearly labeled as non-authoritative.

---

## Known environment issue: root or wrong-container permission behavior

Running tests as root inside Docker, or in a non-equivalent container environment, can create false permission-related test failures.

Observed false-failure examples include:

```text
test_validate_input_file_not_readable
test_validate_output_directory_not_writable
test_stream_records_handles_os_error
test_append_text_output_handles_permission_error
```

These failures disappeared when the full suite was run through the host-user Docker script from the macOS host terminal.

If these tests fail in a root container or Hermes backend fallback environment, do not immediately treat them as product failures. Verify with the authoritative host-user Docker test workflow.

---

## Recommended Docker test command

Use this from the macOS host terminal:

```bash
./scripts/test-docker-host-user.sh
```

The script is preferred because it:

- runs Docker as the host UID/GID
- avoids root-container permission false failures
- records the current branch and commit
- writes human-readable and JSON result files
- preserves the pytest exit code
- provides a stable result source for Hermes

If the script is not available, use this equivalent manual command from the macOS host terminal:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -e UV_LINK_MODE=copy \
  -v "$PWD:/workspace/ai-ner:rw" \
  -w /workspace/ai-ner \
  ai-ner-hermes:python3.11-uv-git \
  bash -lc 'uv sync --frozen && uv run pytest --color=no'
```

Manual runs should still be summarized clearly before updating this file.

---

## Current interpretation

The test suite itself is passing. The remaining failure is the coverage gate:

```text
77.80% < 80%
```

This means the next task should focus on safe coverage improvement, preferably by adding meaningful tests rather than changing production behavior.

The current state should be understood as:

```text
Functional tests: green
Coverage gate: red
Overall command: non-zero because coverage is below 80%
```

---

## Current safe coverage target

Likely first target:

```text
src/ai_ner_system/main.py
```

Reason:

- It appears to have 0% coverage.
- CLI/import/entrypoint tests are usually lower risk than pipeline or LLM behavior tests.
- This can improve coverage without changing runtime logic.

---

## Phase 0 rules

- Prefer tests only.
- Do not change production behavior.
- Use synthetic data only.
- Do not use real/private medieval texts.
- Do not use UiB credentials or production systems.
- Do not modify `.env` or secrets.
- Avoid async pipeline, LLM client behavior, file output semantics, and recovery logic unless specifically planned.

---

## Generated artifacts

Do not commit generated artifacts, including:

```text
.test-results/
.venv/
.pytest_cache/
htmlcov/
coverage.xml
.coverage
__pycache__/
*.pyc
.DS_Store
```

`.test-results/` is intentionally ignored by Git. Hermes may read it locally, but it must not be committed.

---

## Latest updates

### 2026-06-16 authoritative Mac host Docker test run with result files

- Branch: `test/async-pipeline`
- Commit: `b81aa69`
- Timestamp: `2026-06-16T20:00:34Z`
- Command:

  ```bash
  ./scripts/test-docker-host-user.sh
  ```

- Authoritative result files:

  ```text
  .test-results/latest-full-test.md
  .test-results/latest-full-test.json
  ```

- Log files:

  ```text
  .test-results/logs/full-pytest-20260616-200034.log
  .test-results/logs/full-pytest-20260616-200034.clean.log
  ```

- Test result:

  ```text
  688 passed in 14.09s
  ```

- Coverage result:

  ```text
  FAIL Required test coverage of 80% not reached. Total coverage: 77.80%
  ```

- Coverage gate: failed
- Overall status: `failed_coverage_gate`
- Interpretation: Functional tests passed, but the full command exited non-zero because the coverage gate failed.
- Notes: This result comes from the generated `.test-results/latest-full-test.json` and `.test-results/latest-full-test.md` files. Rui ran the Docker host-user script from the macOS host terminal. The permission-sensitive tests passed in this run. The remaining failure is the 80% coverage gate.

### 2026-06-16 earlier Mac host Docker test run

- Branch: `test/async-pipeline`
- Command: `./scripts/test-docker-host-user.sh`
- Test result:

  ```text
  688 passed in 11.32s
  ```

- Coverage result:

  ```text
  FAIL Required test coverage of 80% not reached. Total coverage: 77.80%
  ```

- Overall status: failed because the coverage gate is not met.
- Notes: Rui ran this from the Mac host terminal using the host-user Docker script. The permission-sensitive tests passed in this run; the remaining failure is the 80% coverage gate.

### 2026-06-16 local Hermes backend fallback pytest run

- Branch: `test/async-pipeline`
- Command:

  ```bash
  HOME="$PWD/.cache/home" XDG_CACHE_HOME="$PWD/.cache/home/.cache" TMPDIR="$PWD/.cache/tmp" UV_CACHE_DIR="$PWD/.cache/uv" uv run pytest
  ```

- Test result:

  ```text
  4 failed, 684 passed in 12.15s
  ```

- Failed tests:

  ```text
  FAILED tests/unit/config/test_validation.py::TestConfigValidatorInputFileValidation::test_validate_input_file_not_readable
  FAILED tests/unit/config/test_validation.py::TestConfigValidatorOutputValidation::test_validate_output_directory_not_writable
  FAILED tests/unit/file_io/test_csv_reader.py::TestCSVReader::test_stream_records_handles_os_error
  FAILED tests/unit/file_io/test_output_writers.py::TestOutputWriter::test_append_text_output_handles_permission_error
  ```

- Coverage result:

  ```text
  FAIL Required test coverage of 80% not reached. Total coverage: 77.57%
  ```

- Overall status: non-authoritative failure.
- Notes: The first local attempt, `uv sync --frozen && uv run pytest`, failed before tests because `/tmp/uv-cache` had no space left. The full pytest run above used repo-local cache and temp directories. This was not the preferred host-user Docker command because `docker` is unavailable in the Hermes Docker backend. No source code or tests were changed. This result must not replace the authoritative baseline.

### 2026-06-16 Hermes backend Docker command attempt

- Branch: `test/async-pipeline`
- Command: `./scripts/test-docker-host-user.sh`
- Test result:

  ```text
  ERROR: docker command not found.
  ```

- Coverage result: not produced; the command failed before pytest or coverage ran.
- Overall status: failed before tests.
- Notes: This happened because the Hermes Docker backend environment does not have the `docker` command available. No source code or tests were changed. The authoritative baseline remains the Mac host terminal Docker run.

### 2026-06-09

- Created Hermes `ai-ner` profile.
- Configured Docker backend.
- Confirmed Docker test workflow.
- Confirmed host UID/GID is required for reliable permission tests.
- Added `.hermes.md` project safety instructions.
