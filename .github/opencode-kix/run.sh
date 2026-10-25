#!/usr/bin/env bash
# Run OpenCode 2 with UiB KIX GLM-5.3 in GitHub Actions, read-only.
# Called by .github/workflows/opencode-kix.yml.
#
# Environment:
#   KIX_API_KEY   the KIX key, from the repository secret (required)
#   MODE          "review" (pull request labelled kix-review) or "prompt" (manual run)
#   PROMPT        the request, for MODE=prompt
#   BASE_REF      the pull request's base branch, for MODE=review
#   RESULT_DIR    where answer.md, events.jsonl and usage.json are written
#   KIX_MODEL     optional, default kix/uib/glm-5.3 (for example kix/uib/glm-5.3#high)
#   OPENCODE_BIN  optional, default opencode
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
config="$here/opencode.json"
opencode_bin=${OPENCODE_BIN:-opencode}
model=${KIX_MODEL:-kix/uib/glm-5.3}
mode=${MODE:-prompt}
out=${RESULT_DIR:?RESULT_DIR is not set}
mkdir -p "$out"

fail() {
  echo "opencode-kix: $*" >&2
  exit 1
}

[ -n "${KIX_API_KEY:-}" ] ||
  fail "KIX_API_KEY is empty: add the repository secret. Pull requests from forks get no secrets."

# GitHub tokens never belong in opencode's environment; the comment step has its own.
unset GH_TOKEN GITHUB_TOKEN

# 1. Can this runner reach KIX? No key is sent, so HTTP 401 means reachable.
base_url=$(jq -r '.providers.kix.settings.baseURL' "$config")
code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$base_url/models") ||
  fail "KIX ($base_url) is not reachable from this runner. Use a self-hosted runner at UiB."
echo "opencode-kix: KIX reachable from this runner (HTTP $code without a key)"

# 2. The request.
files=()
case "$mode" in
review)
  [ -n "${BASE_REF:-}" ] || fail "BASE_REF is empty"
  git diff "origin/$BASE_REF...HEAD" >"$out/pr.diff"
  [ -s "$out/pr.diff" ] || fail "no changes against origin/$BASE_REF"
  files=(--file "$out/pr.diff")
  request="Review the attached diff of this pull request against $BASE_REF. Read files in the repository when you need more context, but do not modify anything. Reply in Markdown: one sentence on what the change does, then concrete problems with file paths and line numbers and why they matter, or say that you found none. Do not repeat the diff."
  title="kix review ${GITHUB_RUN_ID:-local}"
  ;;
prompt)
  request=${PROMPT:-}
  [ -n "$request" ] || fail "PROMPT is empty"
  title="kix prompt ${GITHUB_RUN_ID:-local}"
  ;;
*)
  fail "MODE must be review or prompt, not '$mode'"
  ;;
esac

# 3. Run opencode: private server, Build agent (the CI config gives it no edit
#    tools), JSON events. Requests that would need approval are rejected.
version=$("$opencode_bin" --version 2>/dev/null | sed -n 1p)
echo "opencode-kix: $version, model $model, mode $mode"
OPENCODE_CONFIG="$config" OPENCODE_DISABLE_CLAUDE_CODE=1 \
  "$opencode_bin" run --standalone --agent build --model "$model" --title "$title" --format json \
  ${files[@]+"${files[@]}"} "$request" </dev/null >"$out/events.jsonl" ||
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

# 6. answer.md, and the run summary page.
{
  printf '%s\n\n' "$answer"
  printf -- '---\n_GLM-5.3 via UiB KIX (%s), read-only: %s model requests, %s input tokens._\n' \
    "${version:-opencode}" "$requests" "$input"
} >"$out/answer.md"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat "$out/answer.md" >>"$GITHUB_STEP_SUMMARY"
fi
echo "opencode-kix: answer written to $out/answer.md ($requests model requests, $input input tokens)"
