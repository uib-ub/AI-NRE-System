#!/usr/bin/env bash
#
# Run the AI-NER test suite inside Docker as the host user.
#
# Why:
# - Running Docker as root can cause false permission-test failures.
# - This script gives Rui, Hermes, Claude Code, and CI-like local checks
#   one reproducible command for the project baseline.
# - This script must be run from the macOS host terminal, not from inside
#   the Hermes Docker backend. Hermes can read the generated result files
#   after Rui runs this script.
#
# Usage:
#   ./scripts/test-docker-host-user.sh
#
# Optional:
#   PYTEST_ARGS="-q" ./scripts/test-docker-host-user.sh
#   PYTEST_ARGS="tests/unit/prompt/test_builder.py -q" ./scripts/test-docker-host-user.sh
#   AI_NER_TEST_IMAGE="ai-ner-hermes:python3.11-uv-git" ./scripts/test-docker-host-user.sh
#
# Output files:
#   .test-results/latest-full-test.md
#   .test-results/latest-full-test.json
#   .test-results/logs/full-pytest-YYYYMMDD-HHMMSS.log
#   .test-results/logs/full-pytest-YYYYMMDD-HHMMSS.clean.log
#
# Important:
# - .test-results/ is generated local output and should not be committed.
# - The script exits with the original pytest exit code.

set -euo pipefail

IMAGE="${AI_NER_TEST_IMAGE:-ai-ner-hermes:python3.11-uv-git}"
WORKDIR_IN_CONTAINER="/workspace/ai-ner"
PYTEST_ARGS="${PYTEST_ARGS:-}"

RESULT_DIR=".test-results"
LOG_DIR="${RESULT_DIR}/logs"
TIMESTAMP="$(date -u +"%Y%m%d-%H%M%S")"
ISO_TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

LOG_FILE="${LOG_DIR}/full-pytest-${TIMESTAMP}.log"
CLEAN_LOG="${LOG_FILE%.log}.clean.log"
LATEST_MD="${RESULT_DIR}/latest-full-test.md"
LATEST_JSON="${RESULT_DIR}/latest-full-test.json"

mkdir -p "${LOG_DIR}"

json_escape() {
  # Escape a string for simple JSON values.
  # This is intentionally small and dependency-free.
  python3 - "$1" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1]))
PY
}

write_failure_result_files() {
  local reason="$1"
  local exit_code="$2"

  local branch
  local commit

  branch="$(git branch --show-current 2>/dev/null || echo "unknown")"
  commit="$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")"

  cat > "${LATEST_MD}" <<EOF
# AI-NER Full Test Result

## Summary

- Timestamp: ${ISO_TIMESTAMP}
- Status: failed_before_pytest
- Reason: ${reason}
- Exit code: ${exit_code}
- Test summary: not_run
- Coverage: not_run
- Coverage gate: not_run

## Environment

- Repository: $(pwd)
- Branch: ${branch}
- Commit: ${commit}
- Docker image: ${IMAGE}
- Host UID:GID: $(id -u):$(id -g)
- Workdir in container: ${WORKDIR_IN_CONTAINER}

## Command

\`\`\`bash
./scripts/test-docker-host-user.sh
\`\`\`

## Pytest arguments

\`\`\`text
${PYTEST_ARGS:-<none>}
\`\`\`

## Notes

This script must be run from the macOS host terminal, not from inside the Hermes Docker backend.

Hermes Docker backend does not have access to the macOS Docker CLI/daemon. We intentionally do not mount the host Docker socket into Hermes for security reasons.

EOF

  cat > "${LATEST_JSON}" <<EOF
{
  "timestamp": $(json_escape "${ISO_TIMESTAMP}"),
  "status": "failed_before_pytest",
  "reason": $(json_escape "${reason}"),
  "exit_code": ${exit_code},
  "repository": $(json_escape "$(pwd)"),
  "branch": $(json_escape "${branch}"),
  "commit": $(json_escape "${commit}"),
  "docker_image": $(json_escape "${IMAGE}"),
  "host_uid": $(json_escape "$(id -u)"),
  "host_gid": $(json_escape "$(id -g)"),
  "workdir_in_container": $(json_escape "${WORKDIR_IN_CONTAINER}"),
  "command": "./scripts/test-docker-host-user.sh",
  "pytest_args": $(json_escape "${PYTEST_ARGS}"),
  "log_file": null,
  "clean_log_file": null,
  "test_summary": "not_run",
  "coverage_percent": "not_run",
  "coverage_gate": "not_run"
}
EOF
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 command not found." >&2
  echo "python3 is needed to write JSON result files." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker command not found." >&2
  echo "This script must be run from the macOS host terminal, not from inside the Hermes Docker backend." >&2
  echo "Hermes should ask Rui to run this script manually and then read:" >&2
  echo "  ${LATEST_MD}" >&2
  echo "  ${LATEST_JSON}" >&2
  write_failure_result_files "docker command not found; script was probably run inside Hermes Docker backend or another environment without Docker CLI" 127
  exit 127
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon is not running or is not reachable." >&2
  echo "Please start Docker Desktop and run this script from the macOS host terminal." >&2
  write_failure_result_files "Docker daemon is not running or is not reachable" 125
  exit 125
fi

if [ ! -f "pyproject.toml" ] || [ ! -f "uv.lock" ]; then
  echo "ERROR: This script must be run from the repository root." >&2
  echo "Expected to find pyproject.toml and uv.lock in the current directory." >&2
  write_failure_result_files "script was not run from repository root" 2
  exit 2
fi

BRANCH="$(git branch --show-current 2>/dev/null || echo "unknown")"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")"
REPOSITORY="$(pwd)"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

echo "Running AI-NER tests in Docker"
echo "Image: ${IMAGE}"
echo "Host UID:GID: ${HOST_UID}:${HOST_GID}"
echo "Repository: ${REPOSITORY}"
echo "Branch: ${BRANCH}"
echo "Commit: ${COMMIT}"
echo "Pytest args: ${PYTEST_ARGS:-<none>}"
echo "Log file: ${LOG_FILE}"
echo

set +e

docker run --rm \
  --user "${HOST_UID}:${HOST_GID}" \
  -e HOME=/tmp \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -e UV_LINK_MODE=copy \
  -v "$PWD:${WORKDIR_IN_CONTAINER}:rw" \
  -w "${WORKDIR_IN_CONTAINER}" \
  "${IMAGE}" \
  bash -lc "uv sync --frozen && uv run pytest --color=no ${PYTEST_ARGS}" \
  2>&1 | tee "${LOG_FILE}"

PIPE_STATUS=("${PIPESTATUS[@]}")
PYTEST_EXIT_CODE="${PIPE_STATUS[0]}"

set -e

# Remove ANSI color/control codes before parsing.
python3 - "${LOG_FILE}" "${CLEAN_LOG}" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

text = src.read_text(errors="replace")
text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
dst.write_text(text)
PY

# First try to extract the final pytest summary line.
# Examples:
#   688 passed in 12.34s
#   688 passed, 1 warning in 12.34s
#   4 failed, 684 passed, 3 warnings in 11.89s
TEST_SUMMARY="$(
  grep -E "([0-9]+ (failed|passed|skipped|xfailed|xpassed|error|errors|warning|warnings))" "${CLEAN_LOG}" \
    | grep -E " in [0-9.]+s" \
    | tail -n 1 \
    | sed -E 's/^=+[[:space:]]*//; s/[[:space:]]*=+$//; s/^[[:space:]]+//; s/[[:space:]]+$//' \
    || true
)"

# Fallback: count per-test result lines from verbose pytest output.
# Examples:
#   tests/unit/foo.py::test_bar PASSED [ 12%]
#   tests/unit/foo.py::test_baz FAILED [ 13%]
if [ -z "${TEST_SUMMARY}" ]; then
  PASSED_COUNT="$(grep -E " (PASSED)[[:space:]]+\[[[:space:]]*[0-9]+%\]" "${CLEAN_LOG}" | wc -l | tr -d ' ')"
  FAILED_COUNT="$(grep -E " (FAILED)[[:space:]]+\[[[:space:]]*[0-9]+%\]" "${CLEAN_LOG}" | wc -l | tr -d ' ')"
  SKIPPED_COUNT="$(grep -E " (SKIPPED)[[:space:]]+\[[[:space:]]*[0-9]+%\]" "${CLEAN_LOG}" | wc -l | tr -d ' ')"
  XFAILED_COUNT="$(grep -E " (XFAIL|XFAILED)[[:space:]]+\[[[:space:]]*[0-9]+%\]" "${CLEAN_LOG}" | wc -l | tr -d ' ')"
  XPASSED_COUNT="$(grep -E " (XPASS|XPASSED)[[:space:]]+\[[[:space:]]*[0-9]+%\]" "${CLEAN_LOG}" | wc -l | tr -d ' ')"
  ERROR_COUNT="$(grep -E " (ERROR)[[:space:]]+\[[[:space:]]*[0-9]+%\]" "${CLEAN_LOG}" | wc -l | tr -d ' ')"

  SUMMARY_PARTS=()

  if [ "${FAILED_COUNT}" -gt 0 ]; then
    SUMMARY_PARTS+=("${FAILED_COUNT} failed")
  fi

  if [ "${ERROR_COUNT}" -gt 0 ]; then
    SUMMARY_PARTS+=("${ERROR_COUNT} errors")
  fi

  if [ "${PASSED_COUNT}" -gt 0 ]; then
    SUMMARY_PARTS+=("${PASSED_COUNT} passed")
  fi

  if [ "${SKIPPED_COUNT}" -gt 0 ]; then
    SUMMARY_PARTS+=("${SKIPPED_COUNT} skipped")
  fi

  if [ "${XFAILED_COUNT}" -gt 0 ]; then
    SUMMARY_PARTS+=("${XFAILED_COUNT} xfailed")
  fi

  if [ "${XPASSED_COUNT}" -gt 0 ]; then
    SUMMARY_PARTS+=("${XPASSED_COUNT} xpassed")
  fi

  if [ "${#SUMMARY_PARTS[@]}" -gt 0 ]; then
    TEST_SUMMARY="$(IFS=', '; echo "${SUMMARY_PARTS[*]}")"
  else
    TEST_SUMMARY="unknown"
  fi
fi

# Extract coverage percentage from pytest-cov output.
# Examples:
#   FAIL Required test coverage of 80% not reached. Total coverage: 77.57%
#   Required test coverage of 80% reached. Total coverage: 80.12%
COVERAGE_PERCENT="$(
  grep -E "Total coverage: [0-9.]+%" "${CLEAN_LOG}" \
    | tail -n 1 \
    | sed -E 's/.*Total coverage: ([0-9.]+)%.*/\1/' \
    || true
)"

if [ -z "${COVERAGE_PERCENT}" ]; then
  COVERAGE_PERCENT="unknown"
fi

if grep -q "FAIL Required test coverage" "${CLEAN_LOG}"; then
  COVERAGE_GATE="failed"
elif grep -q "Required test coverage.*reached" "${CLEAN_LOG}"; then
  COVERAGE_GATE="passed"
else
  COVERAGE_GATE="unknown"
fi

if [ "${PYTEST_EXIT_CODE}" -eq 0 ]; then
  OVERALL_STATUS="passed"
else
  if [ "${COVERAGE_GATE}" = "failed" ] \
    && echo "${TEST_SUMMARY}" | grep -q "passed" \
    && ! echo "${TEST_SUMMARY}" | grep -q "failed" \
    && ! echo "${TEST_SUMMARY}" | grep -q "errors"; then
    OVERALL_STATUS="failed_coverage_gate"
  else
    OVERALL_STATUS="failed"
  fi
fi

cat > "${LATEST_MD}" <<EOF
# AI-NER Full Test Result

## Summary

- Timestamp: ${ISO_TIMESTAMP}
- Status: ${OVERALL_STATUS}
- Pytest exit code: ${PYTEST_EXIT_CODE}
- Test summary: ${TEST_SUMMARY}
- Coverage: ${COVERAGE_PERCENT}%
- Coverage gate: ${COVERAGE_GATE}

## Environment

- Repository: ${REPOSITORY}
- Branch: ${BRANCH}
- Commit: ${COMMIT}
- Docker image: ${IMAGE}
- Host UID:GID: ${HOST_UID}:${HOST_GID}
- Workdir in container: ${WORKDIR_IN_CONTAINER}

## Command

\`\`\`bash
./scripts/test-docker-host-user.sh
\`\`\`

## Pytest arguments

\`\`\`text
${PYTEST_ARGS:-<none>}
\`\`\`

## Log files

Full log:

\`\`\`text
${LOG_FILE}
\`\`\`

Clean log:

\`\`\`text
${CLEAN_LOG}
\`\`\`

## Notes

This is the authoritative local full test result only when this script was run from the macOS host terminal.

Hermes should read this file or \`${LATEST_JSON}\` when updating \`testing-status.md\`.

Do not commit \`.test-results/\`.
EOF

cat > "${LATEST_JSON}" <<EOF
{
  "timestamp": $(json_escape "${ISO_TIMESTAMP}"),
  "status": $(json_escape "${OVERALL_STATUS}"),
  "pytest_exit_code": ${PYTEST_EXIT_CODE},
  "test_summary": $(json_escape "${TEST_SUMMARY}"),
  "coverage_percent": $(json_escape "${COVERAGE_PERCENT}"),
  "coverage_gate": $(json_escape "${COVERAGE_GATE}"),
  "repository": $(json_escape "${REPOSITORY}"),
  "branch": $(json_escape "${BRANCH}"),
  "commit": $(json_escape "${COMMIT}"),
  "docker_image": $(json_escape "${IMAGE}"),
  "host_uid": $(json_escape "${HOST_UID}"),
  "host_gid": $(json_escape "${HOST_GID}"),
  "workdir_in_container": $(json_escape "${WORKDIR_IN_CONTAINER}"),
  "command": "./scripts/test-docker-host-user.sh",
  "pytest_args": $(json_escape "${PYTEST_ARGS}"),
  "log_file": $(json_escape "${LOG_FILE}"),
  "clean_log_file": $(json_escape "${CLEAN_LOG}")
}
EOF

echo
echo "Test result summary written to:"
echo "  ${LATEST_MD}"
echo "  ${LATEST_JSON}"
echo "Full log written to:"
echo "  ${LOG_FILE}"
echo "Clean log written to:"
echo "  ${CLEAN_LOG}"
echo
echo "Status: ${OVERALL_STATUS}"
echo "Test summary: ${TEST_SUMMARY}"
echo "Coverage: ${COVERAGE_PERCENT}%"
echo "Coverage gate: ${COVERAGE_GATE}"
echo "Pytest exit code: ${PYTEST_EXIT_CODE}"

exit "${PYTEST_EXIT_CODE}"