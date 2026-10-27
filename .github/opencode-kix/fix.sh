#!/usr/bin/env bash
# Proposed fixes as a pull request. Called by .github/workflows/opencode-kix-fix.yml:
#
#   fix.sh patch   first job, after GLM edited files: turn the edits into fix.patch,
#                  and write the step output changed=true or changed=false
#   fix.sh open    second job, which never has the KIX key: apply fix.patch, push it
#                  as a new branch, and open a draft pull request, or, where GitHub
#                  Actions may not open pull requests, give a link to open one
#
# Environment: RESULT_DIR; for patch also GITHUB_OUTPUT; for open also GH_TOKEN,
# GITHUB_REPOSITORY, BASE (the default branch), BRANCH, TITLE, REQUESTER, RUN_URL,
# and NUMBER (the issue, for fixes started from an issue).
set -euo pipefail

out=${RESULT_DIR:?RESULT_DIR is not set}
max_files=50
max_bytes=1000000

fail() {
  echo "opencode-kix: $*" >&2
  exit 1
}

# Fixes may not change the CI setup. Reads one path per line.
check_paths() {
  local bad
  bad=$(grep -E '^\.github/' || true)
  [ -z "$bad" ] || fail "the edits change files under .github/, which fixes may not change: $(printf '%s ' $bad)"
}

check_size() { # files, bytes
  [ "$1" -le "$max_files" ] || fail "the edits change $1 files; a fix may change at most $max_files"
  [ "$2" -le "$max_bytes" ] || fail "the patch has $2 bytes; a fix may have at most $max_bytes"
}

patch() {
  git add -A
  git diff --cached --name-only --no-renames | check_paths
  git diff --cached --binary --no-renames > "$out/fix.patch"
  if [ ! -s "$out/fix.patch" ]; then
    echo "changed=false" >> "${GITHUB_OUTPUT:?GITHUB_OUTPUT is not set}"
    echo "opencode-kix: GLM changed no files"
    return 0
  fi
  check_size "$(git diff --cached --name-only --no-renames | wc -l | tr -d ' ')" "$(wc -c < "$out/fix.patch" | tr -d ' ')"
  git diff --cached --stat --no-renames
  echo "changed=true" >> "${GITHUB_OUTPUT:?GITHUB_OUTPUT is not set}"
}

open_pr() {
  : "${GH_TOKEN:?}" "${BASE:?}" "${BRANCH:?}" "${TITLE:?}"
  local repo=${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is not set}
  local title=${TITLE%%$'\n'*}
  title=${title:0:72}
  [ -s "$out/fix.patch" ] || fail "no fix.patch from the first job"

  # This job can push, so it checks the patch again before using it.
  git apply --numstat "$out/fix.patch" | cut -f3 | check_paths
  check_size "$(git apply --numstat "$out/fix.patch" | wc -l | tr -d ' ')" "$(wc -c < "$out/fix.patch" | tr -d ' ')"
  git switch -q -c "$BRANCH"
  git apply --index "$out/fix.patch"
  git -c user.name="github-actions[bot]" -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    -c core.hooksPath=/dev/null commit -q -m "$title" \
    -m "Proposed by GLM-5.3 via UiB KIX for @${REQUESTER:-?}. Run: ${RUN_URL:-?}"
  local auth
  auth=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 | tr -d '\n')
  git -c http.extraheader="AUTHORIZATION: basic $auth" -c core.hooksPath=/dev/null \
    push -q origin "HEAD:refs/heads/$BRANCH"
  echo "opencode-kix: pushed the branch $BRANCH"

  {
    printf 'Proposed by GLM-5.3 via UiB KIX for @%s%s. Nothing in it has been run yet: review it like any other change.\n\n' \
      "${REQUESTER:-?}" "${NUMBER:+, for issue #$NUMBER}"
    if [ -s "$out/answer.md" ]; then
      cat "$out/answer.md"
    fi
    printf '\nChecks do not start by themselves on pull requests that GitHub Actions opens: close and reopen this pull request, or push a commit, to start them. Run: %s\n' "${RUN_URL:-?}"
  } > "$out/pr-body.md"

  local url message
  if url=$(gh pr create --repo "$repo" --draft --base "$BASE" --head "$BRANCH" \
    --title "$title" --body-file "$out/pr-body.md" 2> "$out/pr-create.err"); then
    message="draft pull request $url"
  else
    url="https://github.com/$repo/compare/$BASE...$BRANCH?expand=1&title=$(jq -rn --arg t "$title" '$t | @uri')"
    message="branch \`$BRANCH\`. GitHub Actions may not open pull requests in this repository, so open one here: $url (its description is on the run's summary page)"
    echo "opencode-kix: gh pr create: $(head -c 300 "$out/pr-create.err" | tr '\n' ' ')"
  fi
  echo "opencode-kix: proposed fix: $message"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
      printf 'Proposed fix: %s\n\n' "$message"
      cat "$out/pr-body.md"
    } >> "$GITHUB_STEP_SUMMARY"
  fi
  if [ -n "${NUMBER:-}" ]; then
    printf '@%s, proposed fix: %s\n' "${REQUESTER:-?}" "$message" > "$out/issue-comment.md"
    gh api "repos/$repo/issues/$NUMBER/comments" -F "body=@$out/issue-comment.md" > /dev/null
  fi
}

case "${1:-}" in
  patch) patch ;;
  open) open_pr ;;
  *) fail "usage: fix.sh patch|open" ;;
esac
