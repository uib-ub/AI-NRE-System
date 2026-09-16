# Git hooks policy for the AI-NER sandbox

This repository uses a controlled Git hooks path for the Hermes / agent-assisted sandbox.

The local sandbox Git config should be:

    git config core.hooksPath .githooks

## Why this exists

The Hermes Docker backend has read-write access to the mounted sandbox repository.

The default Git hooks directory is:

    .git/hooks/

That directory is not visible in normal git diff, but Git hooks run on the macOS host when doing commits.

Therefore, using .git/hooks/ directly is a hidden host-side execution risk.

## Policy

- Do not use .git/hooks/ for this sandbox.
- Use this Git-tracked .githooks/ directory if hooks are needed.
- Any hook added here must be reviewed before commit.
- Hooks should be small, readable, and non-destructive.
- Hooks must not access secrets, private data, production systems, SSH keys, tokens, or UiB credentials.
- Hooks must not call external services unless explicitly approves it.
- Hooks must not modify tracked files silently.

## Current status

No active project hook is required by default.

This directory exists mainly to make the hook policy explicit and to prevent hidden .git/hooks/ behavior from becoming part of the agent-assisted workflow.

## Verification

Check the active hooks path:

    git config --local --get core.hooksPath

Expected output:

    .githooks

Check for hidden active hooks:

    find .git/hooks -maxdepth 1 -type f ! -name "*.sample" -print

Expected output should normally be empty.
