#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <tex-file>" >&2
  exit 2
fi

tex_file="$1"
pdf_file="${tex_file%.tex}.pdf"
build_dir="/tmp/ai-ner-beamer-build"

fingerprint() {
  {
    find . -maxdepth 1 -type f \( -name '*.tex' -o -name '*.sty' -o -name '*.cls' -o -name '*.bib' \) -exec stat -f '%m %N' {} \;
    find ../figs -type f \( -name '*.png' -o -name '*.pdf' -o -name '*.jpg' -o -name '*.jpeg' \) -exec stat -f '%m %N' {} \;
  } | sort | shasum | awk '{print $1}'
}

build() {
  mkdir -p "$build_dir"
  pdflatex -interaction=nonstopmode -output-directory="$build_dir" "$tex_file"
  pdflatex -interaction=nonstopmode -output-directory="$build_dir" "$tex_file"
  cp "$build_dir/$pdf_file" "$pdf_file"
}

build
open -a Preview "$pdf_file" >/dev/null 2>&1 || true

last_fp="$(fingerprint)"

while true; do
  sleep 1
  current_fp="$(fingerprint)"
  if [ "$current_fp" != "$last_fp" ]; then
    if build; then
      last_fp="$current_fp"
    fi
  fi
done
