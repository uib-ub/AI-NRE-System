#!/usr/bin/env bash
#
# Run the AI-NER test suite inside Docker as the host user.
#
# Usage:
#   ./scripts/test-docker-host-user.sh
#
# Optional:
#   AI_NER_TEST_IMAGE="ai-ner-hermes:python3.11-uv-git" ./scripts/test-docker-host-user.sh
#   PYTEST_ARGS="-q" ./scripts/test-docker-host-user.sh
#   PYTEST_ARGS="tests/unit/prompt/test_builder.py -q" ./scripts/test-docker-host-user.sh
#
# Important:
# - Run this from the macOS host terminal, not from inside the Hermes Docker backend.
# - Do not commit .test-results/.
# - PYTEST_ARGS is validated on the host and is passed into the container as an
#   environment variable, not interpolated into the shell command.

set -euo pipefail

IMAGE="${AI_NER_TEST_IMAGE:-ai-ner-hermes:python3.11-uv-git}"
WORKDIR_IN_CONTAINER="/workspace/ai-ner"
PYTEST_ARGS="${PYTEST_ARGS:-}"
COVERAGE_THRESHOLD="${AI_NER_COVERAGE_THRESHOLD:-80}"

RESULT_DIR=".test-results"
LOG_DIR="${RESULT_DIR}/logs"
COVERAGE_JSON="${RESULT_DIR}/coverage.json"

if [ -z "$(printf '%s' "${PYTEST_ARGS}" | tr -d '[:space:]')" ]; then
  RESULT_SCOPE="full_suite"
  RESULT_TITLE="Latest Full Test Result"
  RESULT_FILE_STEM="latest-full-test"
  LOG_PREFIX="full-pytest"
else
  RESULT_SCOPE="scoped_subset"
  RESULT_TITLE="Latest Scoped Test Result"
  RESULT_FILE_STEM="latest-scoped-test"
  LOG_PREFIX="scoped-pytest"
fi

LATEST_MD="${RESULT_DIR}/${RESULT_FILE_STEM}.md"
LATEST_JSON="${RESULT_DIR}/${RESULT_FILE_STEM}.json"

TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
STAMP="$(date -u +"%Y%m%d-%H%M%S")"
LOG_FILE="${LOG_DIR}/${LOG_PREFIX}-${STAMP}.log"
CLEAN_LOG_FILE="${LOG_DIR}/${LOG_PREFIX}-${STAMP}.clean.log"

mkdir -p "${LOG_DIR}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/test-docker-host-user.sh

Optional environment variables:
  AI_NER_TEST_IMAGE
      Docker image to use.
      Default: ai-ner-hermes:python3.11-uv-git

  PYTEST_ARGS
      Optional pytest arguments.
      Examples:
        PYTEST_ARGS="-q" ./scripts/test-docker-host-user.sh
        PYTEST_ARGS="tests/unit/prompt/test_builder.py -q" ./scripts/test-docker-host-user.sh
        PYTEST_ARGS="-k test_builder -q" ./scripts/test-docker-host-user.sh

      For safety, PYTEST_ARGS may not contain shell metacharacters:
        ; & | ` $ < > { } [ ] \

      Parentheses are also rejected to keep pytest argument handling simple
      and conservative for agent-assisted workflows.

  AI_NER_COVERAGE_THRESHOLD
      Coverage gate threshold used for structured result classification.
      Default: 80
EOF
}

write_failure_result() {
  local status="$1"
  local message="$2"
  local exit_code="$3"

  local repo branch commit
  repo="$(pwd 2>/dev/null || echo "unknown")"
  branch="unknown"
  commit="unknown"

  if command -v git >/dev/null 2>&1 && [ -d ".git" ]; then
    branch="$(git branch --show-current 2>/dev/null || echo "unknown")"
    commit="$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")"
  fi

  cat > "${LATEST_MD}" <<EOF
# AI-NER ${RESULT_TITLE}

## Summary

- Timestamp: ${TIMESTAMP}
- Status: ${status}
- Pytest exit code: ${exit_code}
- Test summary: not available
- Coverage: not available
- Coverage source: not available
- Coverage gate: not available
- Coverage threshold: ${COVERAGE_THRESHOLD}%

## Environment

- Repository: ${repo}
- Branch: ${branch}
- Commit: ${commit}
- Docker image: ${IMAGE}
- Host UID:GID: $(id -u):$(id -g)
- Workdir in container: ${WORKDIR_IN_CONTAINER}

## Command

\`\`\`bash
./scripts/test-docker-host-user.sh
\`\`\`

## Error

\`\`\`text
${message}
\`\`\`

## Interpretation

The command failed before pytest produced an authoritative test result.
The previous authoritative baseline remains unchanged.
EOF

  python3 - "$LATEST_JSON" \
    "$TIMESTAMP" "$status" "$exit_code" "$message" "$repo" "$branch" "$commit" \
    "$IMAGE" "$(id -u)" "$(id -g)" "$WORKDIR_IN_CONTAINER" "$PYTEST_ARGS" "$COVERAGE_THRESHOLD" <<'PY'
import json
import sys

(
    path,
    timestamp,
    status,
    exit_code,
    message,
    repo,
    branch,
    commit,
    image,
    host_uid,
    host_gid,
    workdir,
    pytest_args,
    threshold,
) = sys.argv[1:]

data = {
    "timestamp": timestamp,
    "status": status,
    "pytest_exit_code": int(exit_code),
    "test_summary": None,
    "coverage_percent": None,
    "coverage_source": None,
    "coverage_gate": None,
    "coverage_threshold": float(threshold),
    "repository": repo,
    "branch": branch,
    "commit": commit,
    "docker_image": image,
    "host_uid": int(host_uid),
    "host_gid": int(host_gid),
    "workdir_in_container": workdir,
    "command": "./scripts/test-docker-host-user.sh",
    "pytest_args": pytest_args,
    "log_file": None,
    "clean_log_file": None,
    "coverage_json_file": None,
    "error": message,
    "authoritative": False,
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
}

validate_pytest_args() {
  python3 - "$PYTEST_ARGS" <<'PY'
import sys

raw = sys.argv[1]

# This script no longer interpolates PYTEST_ARGS into the shell command, but
# keeping a strict validation layer is useful because PYTEST_ARGS may come from
# agent-influenced workflows.
bad_chars = set(";|&`$<>(){}[]\\")
found = sorted(ch for ch in bad_chars if ch in raw)

if found:
    print("ERROR: PYTEST_ARGS contains unsupported shell metacharacters.", file=sys.stderr)
    print("", file=sys.stderr)
    print("Unsupported characters found: " + " ".join(found), file=sys.stderr)
    print("", file=sys.stderr)
    print("For safety, PYTEST_ARGS may not contain:", file=sys.stderr)
    print("  ; & | ` $ < > ( ) { } [ ] \\", file=sys.stderr)
    print("", file=sys.stderr)
    print("Allowed examples:", file=sys.stderr)
    print('  PYTEST_ARGS="-q" ./scripts/test-docker-host-user.sh', file=sys.stderr)
    print('  PYTEST_ARGS="tests/unit/prompt/test_builder.py -q" ./scripts/test-docker-host-user.sh', file=sys.stderr)
    print('  PYTEST_ARGS="-k test_builder -q" ./scripts/test-docker-host-user.sh', file=sys.stderr)
    sys.exit(2)
PY
}

strip_ansi() {
  python3 - "$1" "$2" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text(encoding="utf-8", errors="replace")

ansi = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
dst.write_text(ansi.sub("", text), encoding="utf-8")
PY
}

parse_results() {
  python3 - \
    "$CLEAN_LOG_FILE" "$LATEST_JSON" "$LATEST_MD" "$COVERAGE_JSON" \
    "$TIMESTAMP" "$PYTEST_EXIT_CODE" "$REPO" "$BRANCH" "$COMMIT" "$IMAGE" \
    "$HOST_UID" "$HOST_GID" "$WORKDIR_IN_CONTAINER" "$PYTEST_ARGS" \
    "$LOG_FILE" "$CLEAN_LOG_FILE" "$COVERAGE_THRESHOLD" <<'PY'
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

(
    clean_log_path,
    json_path,
    md_path,
    coverage_json_path,
    timestamp,
    pytest_exit_code,
    repo,
    branch,
    commit,
    image,
    host_uid,
    host_gid,
    workdir,
    pytest_args,
    log_file,
    clean_log_file,
    coverage_threshold,
) = sys.argv[1:]

clean_log = Path(clean_log_path)
json_path = Path(json_path)
md_path = Path(md_path)
coverage_json_path_obj = Path(coverage_json_path)
pytest_exit_code = int(pytest_exit_code)
host_uid = int(host_uid)
host_gid = int(host_gid)
coverage_threshold = float(coverage_threshold)

text = clean_log.read_text(encoding="utf-8", errors="replace") if clean_log.exists() else ""

def find_test_summary(log_text: str) -> str:
    patterns = [
        r"=+\s*([^=\n]*\b(?:passed|failed|error|errors|skipped|xfailed|xpassed|deselected)\b[^=\n]*\s+in\s+[0-9.]+s)\s*=+",
        r"(^[^\n]*\b(?:passed|failed|error|errors|skipped|xfailed|xpassed|deselected)\b[^\n]*\s+in\s+[0-9.]+s$)",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, log_text, flags=re.MULTILINE | re.IGNORECASE))
    if matches:
        return matches[-1].strip()

    passed = len(re.findall(r"\bPASSED\b", log_text))
    failed = len(re.findall(r"\bFAILED\b", log_text))
    errors = len(re.findall(r"\bERROR\b", log_text))
    parts = []
    if failed:
        parts.append(f"{failed} failed")
    if errors:
        parts.append(f"{errors} errors")
    if passed:
        parts.append(f"{passed} passed")
    return ", ".join(parts) if parts else "unknown"

def parse_coverage_from_json():
    if not coverage_json_path_obj.exists():
        return None

    try:
        data = json.loads(coverage_json_path_obj.read_text(encoding="utf-8"))
    except Exception:
        return None

    totals = data.get("totals", {})

    display = totals.get("percent_covered_display")
    if display is not None:
        try:
            return round(float(str(display).rstrip("%")), 2)
        except ValueError:
            pass

    percent = totals.get("percent_covered")
    if percent is not None:
        try:
            return round(float(percent), 2)
        except ValueError:
            pass

    return None

def parse_coverage_from_console(log_text: str):
    matches = re.findall(r"^TOTAL\s+.*?\s+(\d+(?:\.\d+)?)%\s*$", log_text, flags=re.MULTILINE)
    if matches:
        return float(matches[-1])

    matches = re.findall(r"Total coverage:\s*([0-9]+(?:\.[0-9]+)?)%", log_text, flags=re.IGNORECASE)
    if matches:
        return float(matches[-1])

    return None

def parse_coverage_from_xml():
    coverage_xml = Path("coverage.xml")
    if not coverage_xml.exists():
        return None
    try:
        root = ET.parse(coverage_xml).getroot()
        line_rate = root.attrib.get("line-rate")
        if line_rate is None:
            return None
        return round(float(line_rate) * 100.0, 2)
    except Exception:
        return None

def functional_tests_passed(summary: str) -> bool:
    lowered = summary.lower()
    if summary == "unknown":
        return False
    if " failed" in lowered or lowered.startswith("failed"):
        return False
    if " error" in lowered or " errors" in lowered:
        return False
    return " passed" in lowered or lowered.startswith("passed") or "passed in" in lowered

test_summary = find_test_summary(text)

coverage_percent = parse_coverage_from_json()
coverage_source = "coverage.json" if coverage_percent is not None else None

if coverage_percent is None:
    coverage_percent = parse_coverage_from_console(text)
    coverage_source = "console" if coverage_percent is not None else None

if coverage_percent is None:
    coverage_percent = parse_coverage_from_xml()
    coverage_source = "coverage.xml" if coverage_percent is not None else None

if coverage_percent is None:
    coverage_gate = "unknown"
else:
    coverage_gate = "passed" if coverage_percent >= coverage_threshold else "failed"

func_passed = functional_tests_passed(test_summary)

if pytest_exit_code == 0 and func_passed and coverage_gate in {"passed", "unknown"}:
    status = "passed"
elif func_passed and coverage_gate == "failed":
    status = "failed_coverage_gate"
elif pytest_exit_code != 0:
    status = "failed"
else:
    status = "failed"

is_full_suite = not bool(pytest_args.strip())
result_scope = "full_suite" if is_full_suite else "scoped_subset"
result_title = "Latest Full Test Result" if is_full_suite else "Latest Scoped Test Result"
run_label = "full command" if is_full_suite else "scoped command"

data = {
    "timestamp": timestamp,
    "status": status,
    "pytest_exit_code": pytest_exit_code,
    "test_summary": test_summary,
    "coverage_percent": coverage_percent,
    "coverage_source": coverage_source,
    "coverage_gate": coverage_gate,
    "coverage_threshold": coverage_threshold,
    "functional_tests_passed": func_passed,
    "repository": repo,
    "branch": branch,
    "commit": commit,
    "docker_image": image,
    "host_uid": host_uid,
    "host_gid": host_gid,
    "workdir_in_container": workdir,
    "command": "./scripts/test-docker-host-user.sh",
    "pytest_args": pytest_args,
    "log_file": log_file,
    "clean_log_file": clean_log_file,
    "coverage_json_file": coverage_json_path,
    "authoritative": is_full_suite,
    "is_full_suite": is_full_suite,
    "result_scope": result_scope,
}

json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

coverage_text = "not available" if coverage_percent is None else f"{coverage_percent:.2f}%"
pytest_args_text = pytest_args if pytest_args else "<none>"

if status == "passed":
    interpretation = "Functional tests pass and the coverage gate passes."
elif status == "failed_coverage_gate":
    interpretation = f"Functional tests pass, but the {run_label} exits non-zero because the coverage gate fails."
elif status == "failed":
    interpretation = "One or more tests failed, or the result could not be classified as a coverage-only failure."
else:
    interpretation = "The result status is unknown. Inspect the logs before updating testing-status.md."

md = f"""# AI-NER {result_title}

## Summary

- Timestamp: {timestamp}
- Status: {status}
- Result scope: {result_scope}
- Authoritative full baseline: {str(is_full_suite).lower()}
- Pytest exit code: {pytest_exit_code}
- Test summary: {test_summary}
- Coverage: {coverage_text}
- Coverage source: {coverage_source or "not available"}
- Coverage gate: {coverage_gate}
- Coverage threshold: {coverage_threshold:.2f}%

## Environment

- Repository: {repo}
- Branch: {branch}
- Commit: {commit}
- Docker image: {image}
- Host UID:GID: {host_uid}:{host_gid}
- Workdir in container: {workdir}

## Command

```bash
./scripts/test-docker-host-user.sh
```

## Pytest arguments

```text
{pytest_args_text}
```

## Log files

- Full log: `{log_file}`
- Clean log: `{clean_log_file}`

## Structured coverage file

- Coverage JSON: `{coverage_json_path}`

## Interpretation

{interpretation}
"""
md_path.write_text(md, encoding="utf-8")
PY
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

if [ "${1:-}" != "" ]; then
  echo "ERROR: Unknown positional argument: ${1}" >&2
  usage >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  write_failure_result "failed_before_pytest" "python3 command not found on host." 1
  echo "ERROR: python3 command not found on host." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  write_failure_result "failed_before_pytest" "docker command not found." 1
  echo "ERROR: docker command not found." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  write_failure_result "failed_before_pytest" "Docker daemon is not running or is not reachable." 1
  echo "ERROR: Docker daemon is not running or is not reachable." >&2
  exit 1
fi

if [ ! -f "pyproject.toml" ] || [ ! -f "uv.lock" ]; then
  write_failure_result "failed_before_pytest" "This script must be run from the repository root. Expected pyproject.toml and uv.lock." 1
  echo "ERROR: This script must be run from the repository root." >&2
  echo "Expected to find pyproject.toml and uv.lock in the current directory." >&2
  exit 1
fi

validate_pytest_args

REPO="$(pwd)"
BRANCH="$(git branch --show-current 2>/dev/null || echo "unknown")"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

echo "Running AI-NER tests in Docker"
echo "Image: ${IMAGE}"
echo "Host UID:GID: ${HOST_UID}:${HOST_GID}"
echo "Repository: ${REPO}"
echo "Branch: ${BRANCH}"
echo "Commit: ${COMMIT}"
echo "Pytest args: ${PYTEST_ARGS:-<none>}"
echo "Result directory: ${RESULT_DIR}"
echo

set +e

docker run --rm \
  --user "${HOST_UID}:${HOST_GID}" \
  -e HOME=/tmp \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -e UV_LINK_MODE=copy \
  -e AI_NER_PYTEST_ARGS="${PYTEST_ARGS}" \
  -v "$PWD:${WORKDIR_IN_CONTAINER}:rw" \
  -w "${WORKDIR_IN_CONTAINER}" \
  "${IMAGE}" \
  bash -lc 'set -euo pipefail
uv sync --frozen
python3 - <<'"'"'PY'"'"'
import os
import shlex
import subprocess
import sys

raw_args = os.environ.get("AI_NER_PYTEST_ARGS", "")
try:
    extra_args = shlex.split(raw_args)
except ValueError as exc:
    print(f"ERROR: Could not parse PYTEST_ARGS: {exc}", file=sys.stderr)
    raise SystemExit(2)

cmd = [
    "uv",
    "run",
    "pytest",
    "--color=no",
    "--cov-report=xml",
    "--cov-report=json:.test-results/coverage.json",
    *extra_args,
]

print("Running pytest command:", " ".join(shlex.quote(part) for part in cmd), flush=True)
raise SystemExit(subprocess.run(cmd).returncode)
PY' \
  2>&1 | tee "${LOG_FILE}"

PYTEST_EXIT_CODE="${PIPESTATUS[0]}"

set -e

strip_ansi "${LOG_FILE}" "${CLEAN_LOG_FILE}"
parse_results

echo
echo "AI-NER test result written to:"
echo "  ${LATEST_MD}"
echo "  ${LATEST_JSON}"
echo
echo "Log files:"
echo "  ${LOG_FILE}"
echo "  ${CLEAN_LOG_FILE}"
echo
echo "Latest result summary:"
echo

sed -n '1,90p' "${LATEST_MD}" || cat "${LATEST_MD}"

exit "${PYTEST_EXIT_CODE}"
