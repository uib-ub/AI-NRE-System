#!/usr/bin/env bash
# Utility: Sort records in input file by Bindnr and Brevid
# Usage: ./sort-records.sh

set -euo pipefail

# Change to project root directory
cd "$(dirname "$0")/../.." || exit 1

file=./examples/__DN__AI.txt
sorted_file=./examples/Brevid-DN-AI-sorted.txt

echo "Sorting records from ${file}..."

{
  head -n1 "${file}"
  tail -n +2 "${file}" \
   | sed $'s/\r$//' \
   | sed -e '/^[[:space:]]*$/d' -e '/^[[:space:]]*"[[:space:]]*$/d' \
   | sort -t ";" -k1.2,1n -k2.2,2n
} > "${sorted_file}"

echo "✅ Sorted records written to ${sorted_file}"
