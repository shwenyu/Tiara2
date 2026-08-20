#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python -m unittest discover -s tests -v
tiara2-classify --verify
output_file="$(mktemp "${TMPDIR:-/tmp}/tiara2-smoke.XXXXXX.tsv")"
trap 'rm -f "$output_file"' EXIT
tiara2-classify --config config/default.json --device cpu -i examples/example.fasta -o "$output_file"
test "$(wc -l < "$output_file" | tr -d ' ')" -eq 2
grep -q $'^example_contig\t' "$output_file"
echo "Tiara2 smoke test passed"
