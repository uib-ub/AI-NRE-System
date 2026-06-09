#!/usr/bin/env bash
#
# Run the AI-NER test suite inside Docker as the host user.
#
# Why:
# - Running Docker as root can cause false permission-test failures.
# - This script gives Rui, Hermes, Claude Code, and CI-like local checks
#   one reproducible command for the project baseline.
#
# Usage:
#   ./scripts/test-docker-host-user.sh
#
# Optional:
#   PYTEST_ARGS="-q" ./scripts/test-docker-host-user.sh
#   PYTEST_ARGS="tests/unit/prompt/test_builder.py -q" ./scripts/test-docker-host-user.sh

set -euo pipefail

IMAGE="${AI_NER_TEST_IMAGE:-ai-ner-hermes:python3.11-uv-git}"
WORKDIR_IN_CONTAINER="/workspace/ai-ner"
PYTEST_ARGS="${PYTEST_ARGS:-}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker command not found." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon is not running or is not reachable." >&2
  exit 1
fi

if [ ! -f "pyproject.toml" ] || [ ! -f "uv.lock" ]; then
  echo "ERROR: This script must be run from the repository root." >&2
  echo "Expected to find pyproject.toml and uv.lock in the current directory." >&2
  exit 1
fi

echo "Running AI-NER tests in Docker"
echo "Image: ${IMAGE}"
echo "Host UID:GID: $(id -u):$(id -g)"
echo "Repository: $(pwd)"
echo

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -e UV_LINK_MODE=copy \
  -v "$PWD:${WORKDIR_IN_CONTAINER}:rw" \
  -w "${WORKDIR_IN_CONTAINER}" \
  "${IMAGE}" \
  bash -lc "uv sync --frozen && uv run pytest ${PYTEST_ARGS}"
