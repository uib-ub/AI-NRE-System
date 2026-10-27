#!/usr/bin/env bash
# Proposed fixes as a pull request. Called by .github/workflows/opencode-kix-fix.yml:
#
#   fix.sh patch   first job, after GLM edited files: turn the edits into fix.patch,
#                  and write the step output changed=true or changed=false
#   fix.sh open    second job, which never has the KIX key: apply fix.patch, push it
#                  as a new branch, and open a draft pull request, or, where GitHub
#                  Actions may not open pull requests, give a link to open one
#   fix.sh report  third job, also without the KIX key, for a fix from an issue that
#                  did not end in a pull request: say why on the issue
#
# Environment: RESULT_DIR; for patch also GITHUB_OUTPUT; for open also GH_TOKEN,
# GITHUB_REPOSITORY, BASE (the default branch), BRANCH, TITLE, REQUESTER, RUN_URL,
# and NUMBER (the issue, for fixes started from an issue); for report also
# GH_TOKEN, GITHUB_REPOSITORY, NUMBER, REQUESTER, RUN_URL, and the results of the
# other jobs: EDIT_RESULT, CHANGED and PR_RESULT.
set -euo pipefail

out=${RESULT_DIR:?RESULT_DIR is not set}
max_files=50
max_bytes=1000000

fail() {
  echo "opencode-kix: $*" >&2
  exit 1
}

# Fixes may not change the CI setup. Reads the paths of `git diff --name-only -z`,
# which ends each path with a NUL byte. Without -z, Git puts unusual paths, such
# as ones with non-ASCII letters, in quotes, so they would not start with .github/.
check_paths() {
  local path bad=""
  while IFS= read -r -d '' path; do
    case "$path" in
      .github/*) bad="$bad $path" ;;
    esac
  done
  [ -z "$bad" ] || fail "the edits change files under .github/, which fixes may not change:$bad"
}

check_size() { # files, bytes
  [ "$1" -le "$max_files" ] || fail "the edits change $1 files; a fix may change at most $max_files"
  [ "$2" -le "$max_bytes" ] || fail "the patch has $2 bytes; a fix may have at most $max_bytes"
}

patch() {
  git add -A
  git diff --cached --name-only --no-renames -z | check_paths
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

  # This job can push, so it checks the patch again before using it: it applies the
  # patch to a separate index, which changes no file, and checks every path that
  # differs from HEAD there. A rename counts as both of its paths.
  local index="$out/check.index"
  GIT_INDEX_FILE=$index git read-tree HEAD
  GIT_INDEX_FILE=$index git apply --cached "$out/fix.patch"
  GIT_INDEX_FILE=$index git diff --cached --name-only --no-renames -z | check_paths
  check_size "$(GIT_INDEX_FILE=$index git diff --cached --name-only --no-renames | wc -l | tr -d ' ')" "$(wc -c < "$out/fix.patch" | tr -d ' ')"
  rm -f "$index"
  git switch -q -c "$BRANCH"
  git apply --index "$out/fix.patch"
  git -c user.name="github-actions[bot]" -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    -c core.hooksPath=/dev/null commit -q -m "$title" \
    -m "Proposed by GLM-5.3 via UiB KIX for @${REQUESTER:-?}. Run: ${RUN_URL:-?}"
  # The token reaches git through its environment, not its arguments, which any
  # process on the runner can read; and the log masks it, like the plain token.
  local auth
  auth=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 | tr -d '\n')
  echo "::add-mask::$auth"
  GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.extraheader GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $auth" \
    git -c core.hooksPath=/dev/null push -q origin "HEAD:refs/heads/$BRANCH"
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
  # The branch, and perhaps the pull request, exist by now: a failed comment must
  # not fail this job, or the report job would tell the issue that none came of it.
  if [ -n "${NUMBER:-}" ]; then
    printf '@%s, proposed fix: %s\n' "${REQUESTER:-?}" "$message" > "$out/issue-comment.md"
    gh api "repos/$repo/issues/$NUMBER/comments" -F "body=@$out/issue-comment.md" > /dev/null ||
      echo "opencode-kix: could not comment on issue #$NUMBER; the run's summary page has the result"
  fi
}

# Why no pull request came of a fix from an issue, as a comment on the issue, with
# GLM's answer if it changed nothing.
report() {
  : "${GH_TOKEN:?}" "${NUMBER:?}"
  local repo=${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is not set} why
  # CHANGED is set once the patch step ran, even if a later step of the job failed.
  case "${EDIT_RESULT:-}:${CHANGED:-}:${PR_RESULT:-}" in
    *:false:*) why="GLM changed no files" ;;
    success:true:failure) why="making the pull request failed" ;;
    *cancelled*) why="the run was cancelled or ran out of time" ;;
    *:true:*) why="GLM made a change, but the run failed before the pull request" ;;
    *) why="the run failed before there was a change to propose" ;;
  esac
  mkdir -p "$out"
  {
    printf '@%s, no pull request came of this: %s. Run: %s\n' "${REQUESTER:-?}" "$why" "${RUN_URL:-?}"
    if [ "${CHANGED:-}" = false ] && [ -s "$out/answer.md" ]; then
      printf '\n%s\n\n' "GLM's answer:"
      cat "$out/answer.md"
    fi
  } > "$out/report.md"
  gh api "repos/$repo/issues/$NUMBER/comments" -F "body=@$out/report.md" > /dev/null
  echo "opencode-kix: told issue #$NUMBER: $why"
}

case "${1:-}" in
  patch) patch ;;
  open) open_pr ;;
  report) report ;;
  *) fail "usage: fix.sh patch|open|report" ;;
esac
