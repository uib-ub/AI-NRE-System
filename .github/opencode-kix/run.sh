#!/usr/bin/env bash
# Run OpenCode 2 with UiB KIX GLM-5.3 in GitHub Actions, read-only.
# Called by .github/workflows/opencode-kix.yml.
#
# Environment:
#   KIX_API_KEY   the KIX key, from the repository secret (required)
#   MODE          "review" (pull request labelled kix-review) or "prompt" (manual run)
#   TASK          for MODE=prompt: "custom" (use PROMPT, the default) or a prompt file in tasks/
#   TARGET        optional: the path, module or Git ref a task is about
#   PROMPT        the request, for TASK=custom
#   EFFORT        reasoning effort: max (default), high or low
#   BASE_REF      the pull request's base branch, for MODE=review
#   RESULT_DIR    where request.md, answer.md, events.jsonl and usage.json are written
#   KIX_MODEL     optional: model and effort in one, for example kix/uib/glm-5.3#high
#   OPENCODE_BIN  optional, default opencode
#
# Files next to this script. General, the same in every repository:
#   opencode.json  the CI config: KIX provider, read-only rules, step limit
#   tasks/*.md     the prompts for the pull request review and the manual-run tasks
# The repository's own, all optional:
#   project.json   settings added on top of opencode.json (allowed check commands,
#                  instruction files, step limit); passed as inline config
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

# GitHub tokens never belong in opencode's environment; the comment step has its own.
unset GH_TOKEN GITHUB_TOKEN

# Repository settings on top of the CI config, if present.
configs=("$config")
unset OPENCODE_CONFIG_CONTENT
if [ -f "$overlay" ]; then
  jq empty "$overlay" 2>/dev/null || fail "project.json is not valid JSON"
  OPENCODE_CONFIG_CONTENT=$(cat "$overlay")
  export OPENCODE_CONFIG_CONTENT
  configs+=("$overlay")
  echo "opencode-kix: repository settings from project.json"
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
case "$mode" in
  review)
    [ -n "${BASE_REF:-}" ] || fail "BASE_REF is empty"
    git diff "origin/$BASE_REF...HEAD" >"$out/pr.diff"
    [ -s "$out/pr.diff" ] || fail "no changes against origin/$BASE_REF"
    files=(--file "$out/pr.diff")
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
printf '%s\n' "$request" >"$out/request.md"

# 3. Run opencode: private server, Build agent (the CI config gives it no edit
#    tools), JSON events. Requests that would need approval are rejected.
#    The request goes in on standard input, which opencode passes on unchanged
#    (a message argument with spaces gets wrapped in quotes).
version=$("$opencode_bin" --version 2>/dev/null | sed -n 1p)
echo "opencode-kix: $version, model $model, task $task"
OPENCODE_CONFIG="$config" OPENCODE_DISABLE_CLAUDE_CODE=1 \
  "$opencode_bin" run --standalone --agent build --model "$model" --title "$title" --format json \
  ${files[@]+"${files[@]}"} <"$out/request.md" >"$out/events.jsonl" ||
  fail "opencode run failed; see $out/events.jsonl"

errors=$(jq -r 'select(.type == "error") | .error.message // "unknown error"' "$out/events.jsonl")
[ -z "$errors" ] || fail "opencode reported: $errors"

# 4. The answer is the last text part.
sid=$(jq -rs 'map(select(.sessionID)) | first | .sessionID // empty' "$out/events.jsonl")
answer=$(jq -rs '[.[] | select(.type == "text") | .part.text] | last // empty' "$out/events.jsonl")
[ -n "$answer" ] || fail "no answer in the output; see $out/events.jsonl"

# 5. Usage, from the saved session (no model call, no key needed).
usage='{}'
if [ -n "$sid" ] &&
  env -u KIX_API_KEY "$opencode_bin" session export --standalone "$sid" </dev/null >"$out/session.json" 2>/dev/null; then
  usage=$(jq -c '[.messages[] | select(.type == "assistant") | .tokens] | {model_requests: length, input: (map(.input) | add), output: (map(.output) | add), reasoning: (map(.reasoning) | add)}' "$out/session.json")
fi
rm -f "$out/session.json"
echo "$usage" >"$out/usage.json"
requests=$(jq -r '.model_requests // "?"' <<<"$usage")
input=$(jq -r '.input // "?"' <<<"$usage")
limit=$(jq -rs 'map(.agents.build.steps // empty) | last // empty' "${configs[@]}")
note=""
case "$requests:$limit" in
  *[!0-9:]* | :* | *:) ;;
  *) [ "$requests" -lt "$limit" ] || note=" The step limit was reached, so the answer may be incomplete." ;;
esac

# 6. answer.md, and the run summary page.
{
  printf '%s\n\n' "$answer"
  printf -- '---\n_GLM-5.3 via UiB KIX (%s), read-only, task %s, effort %s: %s model requests (limit %s), %s input tokens.%s_\n' \
    "${version:-opencode}" "$task" "$effort" "$requests" "${limit:-none}" "$input" "$note"
} >"$out/answer.md"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat "$out/answer.md" >>"$GITHUB_STEP_SUMMARY"
fi
echo "opencode-kix: answer written to $out/answer.md ($requests model requests, $input input tokens)"
