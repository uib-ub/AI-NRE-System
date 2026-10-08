#!/usr/bin/env bash
#
# Clean local AI-NER test/build artifacts safely.
#
# This script removes only generated local artifacts that should not be
# committed to Git. It does not remove source files, tests, docs, Git data,
# secrets, or private data.
#
# Default behavior:
#
#   ./scripts/clean-test-artifacts.sh
#
# removes common Python/pytest/coverage artifacts, but keeps .test-results/
# because Hermes may need to read the latest generated test result.
#
# To also remove .test-results/:
#
#   ./scripts/clean-test-artifacts.sh --include-test-results
#
# Use that only when you intentionally want to delete the latest local test
# result files.
#
# Dry run:
#
#   ./scripts/clean-test-artifacts.sh --dry-run
#   ./scripts/clean-test-artifacts.sh --include-test-results --dry-run

set -euo pipefail

INCLUDE_TEST_RESULTS=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  ./scripts/clean-test-artifacts.sh [--dry-run] [--include-test-results] [--help]

Options:
  --dry-run
      Show what would be removed, but do not remove anything.

  --include-test-results
      Also remove .test-results/.
      By default .test-results/ is kept because Hermes may need to read:
        .test-results/latest-full-test.md
        .test-results/latest-full-test.json

  --help
      Show this help message.

This script removes only generated local artifacts.

Default removed paths:
  .venv/
  .pytest_cache/
  htmlcov/
  coverage.xml
  .coverage
  .coverage.*
  .cache/test-logs/
  __pycache__/ directories
  *.pyc files
  .DS_Store files

Optional removed path:
  .test-results/
EOF
}

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=true
      ;;
    --include-test-results)
      INCLUDE_TEST_RESULTS=true
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $arg" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ ! -f "pyproject.toml" ] || [ ! -d ".git" ]; then
  cat >&2 <<'EOF'
ERROR: This script should be run from the AI-NER repository root.

Expected to find:
  pyproject.toml
  .git/

Refusing to clean from this directory.
EOF
  exit 1
fi

remove_path() {
  local path="$1"

  if [ ! -e "$path" ]; then
    return 0
  fi

  if [ "$DRY_RUN" = true ]; then
    echo "Would remove: $path"
  else
    rm -rf "$path"
    echo "Removed: $path"
  fi
}

remove_file() {
  local path="$1"

  if [ ! -f "$path" ]; then
    return 0
  fi

  if [ "$DRY_RUN" = true ]; then
    echo "Would remove: $path"
  else
    rm -f "$path"
    echo "Removed: $path"
  fi
}

echo "AI-NER clean test artifacts"
echo "Repository: $(pwd)"
echo "Dry run: $DRY_RUN"
echo "Include .test-results/: $INCLUDE_TEST_RESULTS"
echo

# Common Python / pytest / coverage artifacts.
remove_path ".venv"
remove_path ".pytest_cache"
remove_path "htmlcov"
remove_file "coverage.xml"
remove_file ".coverage"

# Coverage files such as .coverage.hostname.pid.random
for path in .coverage.*; do
  if [ "$path" != ".coverage.*" ]; then
    remove_file "$path"
  fi
done

# Local backend test logs, if the fallback workflow created them.
remove_path ".cache/test-logs"

# Python bytecode/cache artifacts.
if [ "$DRY_RUN" = true ]; then
  find . \
    -path ./.git -prune -o \
    -path ./.test-results -prune -o \
    -type d -name "__pycache__" -print \
    | sed 's/^/Would remove: /'

  find . \
    -path ./.git -prune -o \
    -path ./.test-results -prune -o \
    -type f -name "*.pyc" -print \
    | sed 's/^/Would remove: /'

  find . \
    -path ./.git -prune -o \
    -path ./.test-results -prune -o \
    -type f -name ".DS_Store" -print \
    | sed 's/^/Would remove: /'
else
  find . \
    -path ./.git -prune -o \
    -path ./.test-results -prune -o \
    -type d -name "__pycache__" -print -exec rm -rf {} +

  find . \
    -path ./.git -prune -o \
    -path ./.test-results -prune -o \
    -type f -name "*.pyc" -print -delete

  find . \
    -path ./.git -prune -o \
    -path ./.test-results -prune -o \
    -type f -name ".DS_Store" -print -delete
fi

if [ "$INCLUDE_TEST_RESULTS" = true ]; then
  echo
  echo "Removing .test-results/ because --include-test-results was provided."
  remove_path ".test-results"
else
  echo
  echo "Keeping .test-results/ by default."
  echo "Reason: Hermes may need to read the latest authoritative local test result:"
  echo "  .test-results/latest-full-test.md"
  echo "  .test-results/latest-full-test.json"
fi

echo
echo "Clean complete."
echo

if [ "$DRY_RUN" = false ]; then
  echo "Current generated-artifact status check:"
  git status --short --ignored | grep -E '(^!! |^\\?\\? |^ M |^ A |^ D |^R  |^AM |^MM )' || true
  echo
  echo "Recommended next check:"
  echo "  git status --short --branch"
fi
