#!/usr/bin/env bash
# Prepare a uv project before opencode runs, so that the agent can run the
# project's own checks (listed in checks.md). The workflow runs this step
# without secrets. If anything here fails, the step ends with a warning instead
# of an error: the run goes on, and the agent reports that the checks could not
# run.
#
# General for uv projects, except the line marked as specific below.
set -euo pipefail

# Specific to AI-NRE-System: the dependency groups that hold its lint and test
# tools, as its CI installs them. Change them for another project.
groups=(--group lint --group test)

# Ends this step with a warning, so that the run goes on without the checks.
skip_checks() {
  echo "::warning title=opencode-kix::$1, so the agent cannot run the project's checks in this run"
  exit 0
}

# GitHub's runners have pipx but not uv.
if ! command -v uv >/dev/null; then
  pipx install uv || skip_checks "installing uv failed"
fi
uv --version || skip_checks "uv does not run"

# Exactly what uv.lock pins, as in the project's CI.
uv sync --locked --no-default-groups "${groups[@]}" || skip_checks "uv sync failed"
