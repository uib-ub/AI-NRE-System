#!/bin/sh
set -eu

# use this to render all .mmd files in the current directory to .png using mermaid-cli
# to run: ./render-mermaid.sh file.mmd [file.png]

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 <input.mmd> [output.png]" >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
INPUT="$1"
CONFIG="$SCRIPT_DIR/puppeteer-config.json"
WIDTH="${MERMAID_WIDTH:-2200}"
HEIGHT="${MERMAID_HEIGHT:-1400}"
SCALE="${MERMAID_SCALE:-2}"

case "$INPUT" in
  /*) ;;
  *) INPUT="$(pwd)/$INPUT" ;;
esac

if [ ! -f "$INPUT" ]; then
  echo "error: input file not found: $INPUT" >&2
  exit 1
fi

if [ "${2-}" ]; then
  OUTPUT="$2"
  case "$OUTPUT" in
    /*) ;;
    *) OUTPUT="$(pwd)/$OUTPUT" ;;
  esac
else
  OUTPUT="${INPUT%.mmd}.png"
fi

if command -v mmdc >/dev/null 2>&1; then
  MMDC_BIN="mmdc"
elif command -v npx >/dev/null 2>&1; then
  MMDC_BIN="npx @mermaid-js/mermaid-cli"
else
  echo "error: neither 'mmdc' nor 'npx' is available" >&2
  exit 1
fi

echo "Rendering $(basename "$INPUT") -> $(basename "$OUTPUT")"
if [ -f "$CONFIG" ]; then
  # shellcheck disable=SC2086
  $MMDC_BIN -p "$CONFIG" -i "$INPUT" -o "$OUTPUT" -w "$WIDTH" -H "$HEIGHT" -s "$SCALE"
else
  # shellcheck disable=SC2086
  $MMDC_BIN -i "$INPUT" -o "$OUTPUT" -w "$WIDTH" -H "$HEIGHT" -s "$SCALE"
fi
