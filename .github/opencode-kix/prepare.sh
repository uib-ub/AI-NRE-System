#!/usr/bin/env bash
# Prepare a uv project before opencode runs, so that the agent can run the
# project's own checks (listed in checks.md). The workflow runs this step
# without secrets. If the install fails, the run goes on and the agent reports
# that the checks could not run.
#
# General for uv projects, except the line marked as specific below.
set -euo pipefail

# Specific to AI-NRE-System: the dependency groups that hold its lint and test
# tools, as its CI installs them. Change them for another project.
groups=(--group lint --group test)

if ! command -v uv >/dev/null; then
  pipx install uv
fi
uv --version

# Exactly what uv.lock pins, as in the project's CI.
if ! uv sync --locked --no-default-groups "${groups[@]}"; then
  echo "::warning title=opencode-kix::uv sync failed, so the agent cannot run the project's checks in this run"
fi
