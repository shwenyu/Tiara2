#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python -m unittest discover -s tests -v
tiara2-classify --verify
output_file="$(mktemp "${TMPDIR:-/tmp}/tiara2-smoke.XXXXXX.tsv")"
threshold_output="$(mktemp "${TMPDIR:-/tmp}/tiara2-smoke-threshold.XXXXXX.tsv")"
trap 'rm -f "$output_file" "$threshold_output"' EXIT
tiara2-classify --config config/default.json --device cpu -i examples/example.fasta -o "$output_file"
test "$(wc -l < "$output_file" | tr -d ' ')" -eq 2
head -1 "$output_file" | grep -q $'p_eukaryote\tp_prokaryote\tp_organelle$'
grep -q $'^example_contig\t' "$output_file"
tiara2-classify --config config/default.json --device cpu -i examples/example.fasta \
  -o "$threshold_output" --euk-threshold 0.50
test "$(wc -l < "$threshold_output" | tr -d ' ')" -eq 2
echo "Tiara2 smoke test passed"
