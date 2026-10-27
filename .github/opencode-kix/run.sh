#!/usr/bin/env bash
# Run OpenCode 2 with UiB KIX GLM-5.3 in GitHub Actions.
# Called by the workflows in .github/workflows/opencode-kix*.yml.
#
# Environment:
#   KIX_API_KEY    the KIX key, from the repository secret (required)
#   MODE           "review" (a pull request's diff) or "prompt" (a request)
#   TASK           for MODE=prompt: "custom" (use PROMPT, the default) or a prompt file in tasks/
#   TARGET         optional: the path, module or Git ref a task is about
#   PROMPT         the request, for TASK=custom
#   REQUEST        optional, for other tasks: a request added after the task's prompt
#   PREAMBLE       optional: text put before the request, such as who asked and where
#   CONTEXT_FILES  optional: files to attach, separated by spaces (for example an issue thread)
#   DIFF_FILE      optional, for MODE=review: the diff to review, instead of git diff
#   EXTRA_CONFIG   optional: settings files next to this script, separated by spaces, added
#                  after project.json (web.json for web lookups, fix.json for file edits)
#   EFFORT         reasoning effort: max (default), high or low
#   BASE_REF       the pull request's base branch, for MODE=review
#   RESULT_DIR     where request.md, answer.md, events.jsonl and usage.json are written
#   KIX_MODEL      optional: model and effort in one, for example kix/uib/glm-5.3#high
#   OPENCODE_BIN   optional, default opencode
#
# Files next to this script. General, the same in every repository:
#   opencode.json  the CI config: KIX provider, read-only rules, step limit
#   fix.json       settings for fix runs: file edits allowed, except in .github/ and .git/
#   tasks/*.md     the prompts for reviews, tasks, comments and fixes
# The repository's own, all optional:
#   project.json   settings added on top of opencode.json (allowed check commands,
#                  instruction files, step limit); passed as inline config
#   web.json       web lookups allowed for dependency pull requests
#   prepare.sh     run by the workflow before this script, without secrets
#   checks.md, rules.md, ...  instruction files that project.json lists
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
config="$here/opencode.json"
overlay="$here/project.json"
opencode_bin=${OPENCODE_BIN:-opencode}
mode=${MODE:-prompt}
task=${TASK:-custom}
target=${TARGET:-}
effort=${EFFORT:-max}
out=${RESULT_DIR:?RESULT_DIR is not set}
mkdir -p "$out"

fail() {
  echo "opencode-kix: $*" >&2
  exit 1
}

case "$effort" in
  max | high | low) ;;
  *) fail "EFFORT must be max, high or low, not '$effort'" ;;
esac
model=${KIX_MODEL:-"kix/uib/glm-5.3#$effort"}

[ -n "${KIX_API_KEY:-}" ] ||
  fail "KIX_API_KEY is empty: add the repository secret. Pull requests from forks get no secrets."

# GitHub tokens never belong in opencode's environment; the steps that post have their own.
unset GH_TOKEN GITHUB_TOKEN

# Repository settings on top of the CI config: project.json, then EXTRA_CONFIG.
configs=("$config")
overlays=()
[ ! -f "$overlay" ] || overlays+=("$overlay")
for name in ${EXTRA_CONFIG:-}; do
  case "$name" in
    */* | .*) fail "EXTRA_CONFIG names files next to run.sh, not '$name'" ;;
  esac
  [ -f "$here/$name" ] || fail "no settings file $name next to run.sh"
  overlays+=("$here/$name")
done
unset OPENCODE_CONFIG_CONTENT
if [ ${#overlays[@]} -gt 0 ]; then
  for file in "${overlays[@]}"; do
    jq empty "$file" 2> /dev/null || fail "$(basename "$file") is not valid JSON"
  done
  # Permissions and instructions are lists: keep all of them, in order, because
  # the last matching permission rule wins. Other settings are merged.
  OPENCODE_CONFIG_CONTENT=$(jq -s 'reduce .[] as $c ({permissions: [], instructions: []};
    . as $all
    | ($all * ($c | del(.permissions, .instructions)))
    | .permissions = $all.permissions + ($c.permissions // [])
    | .instructions = $all.instructions + ($c.instructions // []))' "${overlays[@]}")
  export OPENCODE_CONFIG_CONTENT
  configs+=("${overlays[@]}")
  names=$(for file in "${overlays[@]}"; do basename "$file"; done | paste -s -d , - | sed 's/,/, /g')
  echo "opencode-kix: repository settings from $names"
fi

# 1. Can this runner reach KIX? No key is sent, so HTTP 401 means reachable.
base_url=$(jq -r '.providers.kix.settings.baseURL' "$config")
code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$base_url/models") ||
  fail "KIX ($base_url) is not reachable from this runner. Use a self-hosted runner at UiB."
echo "opencode-kix: KIX reachable from this runner (HTTP $code without a key)"

# 2. The request: a prompt file from tasks/, plus the target or base branch.
prompt_file() {
  case "$1" in
    *[!a-z0-9-]* | '') fail "task names use only a-z, 0-9 and '-', not '$1'" ;;
  esac
  [ -f "$here/tasks/$1.md" ] || fail "no prompt file tasks/$1.md"
  cat "$here/tasks/$1.md"
}
files=()
for file in ${CONTEXT_FILES:-}; do
  [ -s "$file" ] || fail "no context file $file"
  files+=(--file "$file")
done
case "$mode" in
  review)
    [ -n "${BASE_REF:-}" ] || fail "BASE_REF is empty"
    if [ -n "${DIFF_FILE:-}" ]; then
      [ -f "$DIFF_FILE" ] || fail "no diff file $DIFF_FILE"
      [ "$DIFF_FILE" -ef "$out/pr.diff" ] || cp "$DIFF_FILE" "$out/pr.diff"
    else
      git diff "origin/$BASE_REF...HEAD" > "$out/pr.diff"
    fi
    [ -s "$out/pr.diff" ] || fail "no changes against origin/$BASE_REF"
    files+=(--file "$out/pr.diff")
    task=pr-review
    request="$(prompt_file pr-review)"$'\n\n'"Base branch: $BASE_REF"
    title="kix review ${GITHUB_RUN_ID:-local}"
    ;;
  prompt)
    if [ "$task" = custom ]; then
      request=${PROMPT:-}
      [ -n "$request" ] || fail "PROMPT is empty"
    else
      request=$(prompt_file "$task")
      if [ -n "${REQUEST:-}" ]; then
        request="$request"$'\n\n'"Request: $REQUEST"
      fi
    fi
    if [ -n "$target" ]; then
      request="$request"$'\n\n'"Target: $target"
    fi
    title="kix $task ${GITHUB_RUN_ID:-local}"
    ;;
  *)
    fail "MODE must be review or prompt, not '$mode'"
    ;;
esac
if [ -n "${PREAMBLE:-}" ]; then
  request="$PREAMBLE"$'\n\n'"$request"
fi
printf '%s\n' "$request" > "$out/request.md"

# Read-only, unless the settings allow file edits (fix.json).
access=$(jq -rs '[.[].permissions[]? | select(.action == "edit" and .resource == "*")] | last | if .effect == "allow" then "file edits allowed" else "read-only" end' "${configs[@]}")

# 3. Run opencode: private server, Build agent (the CI config gives it no edit
#    tools unless fix.json is added), JSON events. Requests that would need
#    approval are rejected. The request goes in on standard input, which opencode
#    passes on unchanged (a message argument with spaces gets wrapped in quotes).
# What went wrong, for the log: the last error, and how many model replies
# started and finished.
failure() {
  local summary
  summary=$(jq -rs '
    (map(select(.type == "error") | .error.message // .error.type // "unknown error") | last // "no error message") as $error
    | "\($error) (\(map(select(.type == "step_start")) | length) model replies started, \(map(select(.type == "step_finish")) | length) finished)"
  ' "$out/events.jsonl" 2> /dev/null) || summary="no events"
  case "$summary" in
    *"ended without finish_reason"*) summary="$summary. KIX ended its replies before they were complete; try again later" ;;
  esac
  fail "opencode run failed: $summary. events.jsonl is in the run's artifact"
}

version=$("$opencode_bin" --version 2> /dev/null | sed -n 1p)
echo "opencode-kix: $version, model $model, task $task, $access"
OPENCODE_CONFIG="$config" OPENCODE_DISABLE_CLAUDE_CODE=1 \
  "$opencode_bin" run --standalone --agent build --model "$model" --title "$title" --format json \
  ${files[@]+"${files[@]}"} < "$out/request.md" > "$out/events.jsonl" ||
  failure

errors=$(jq -r 'select(.type == "error") | .error.message // "unknown error"' "$out/events.jsonl")
[ -z "$errors" ] || fail "opencode reported: $errors"

# 4. The answer is the last text part.
sid=$(jq -rs 'map(select(.sessionID)) | first | .sessionID // empty' "$out/events.jsonl")
answer=$(jq -rs '[.[] | select(.type == "text") | .part.text] | last // empty' "$out/events.jsonl")
[ -n "$answer" ] || fail "no answer in the output; see $out/events.jsonl"

# 5. Usage, from the saved session (no model call, no key needed).
usage='{}'
if [ -n "$sid" ] &&
  env -u KIX_API_KEY "$opencode_bin" session export --standalone "$sid" < /dev/null > "$out/session.json" 2> /dev/null; then
  usage=$(jq -c '[.messages[] | select(.type == "assistant") | .tokens] | {model_requests: length, input: (map(.input) | add), output: (map(.output) | add), reasoning: (map(.reasoning) | add)}' "$out/session.json")
fi
rm -f "$out/session.json"
echo "$usage" > "$out/usage.json"
requests=$(jq -r '.model_requests // "?"' <<< "$usage")
input=$(jq -r '.input // "?"' <<< "$usage")
limit=$(jq -rs 'map(.agents.build.steps // empty) | last // empty' "${configs[@]}")
note=""
case "$requests:$limit" in
  *[!0-9:]* | :* | *:) ;;
  *) [ "$requests" -lt "$limit" ] || note=" The step limit was reached, so the answer may be incomplete." ;;
esac

# 6. answer.md, and the run summary page.
{
  printf '%s\n\n' "$answer"
  printf -- '---\n_GLM-5.3 via UiB KIX (%s), %s, task %s, effort %s: %s model requests (limit %s), %s input tokens.%s_\n' \
    "${version:-opencode}" "$access" "$task" "$effort" "$requests" "${limit:-none}" "$input" "$note"
} > "$out/answer.md"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat "$out/answer.md" >> "$GITHUB_STEP_SUMMARY"
fi
echo "opencode-kix: answer written to $out/answer.md ($requests model requests, $input input tokens)"
