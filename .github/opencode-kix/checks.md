<!-- The command list is specific to AI-NRE-System (as in its CI: paths, --no-cov for its 80% coverage threshold). The rules after the list are general. -->

# Project checks

This is a uv project, and its lint and test tools are installed. You may run these commands with the shell tool:

- Unit tests: `uv run --no-sync pytest tests/unit/ --no-cov -q`
- One test file: `uv run --no-sync pytest tests/unit/test_example.py --no-cov -q`
- Unit tests with coverage: `uv run --no-sync pytest tests/unit/ --cov-report=term-missing --cov-fail-under=0 -q`
- Lint: `uv run --no-sync ruff check src/ tests/`
- Formatting: `uv run --no-sync ruff format --check src/ tests/`
- Types: `uv run --no-sync mypy src/`
- Lock file up to date: `uv lock --check`

Run each command on its own and in this form: no pipes (`|`), redirections (`>`, `2>&1`), `&&` or `;`, and always `uv run --no-sync`. Other commands are rejected. The shell tool already shows standard error.

Nothing may change files, so options such as `ruff check --fix`, `--add-noqa` and `mypy --install-types` are rejected.

Run only the checks that matter for the task, and quote the failures that matter with file paths and line numbers. These are quick checks on a GitHub runner; the project's CI workflow stays the official result. If a command fails because a tool is missing, say that the checks could not run in this environment.
