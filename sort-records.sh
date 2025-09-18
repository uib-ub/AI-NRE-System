#!/usr/bin/env bash

set -euo pipefail

file=./examples/__DN__AI.txt
sorted_file=./examples/Brevid-DN-AI-sorted.txt

{
  head -n1 ${file}
  tail -n +2 ${file} \
   | sed $'s/\r$//' \
   | sed -e '/^[[:space:]]*$/d' -e '/^[[:space:]]*"[[:space:]]*$/d' \
   | sort -t ";" -k1.2,1n -k2.2,2n;
} > ${sorted_file}
