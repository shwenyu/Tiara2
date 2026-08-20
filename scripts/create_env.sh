#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
env_name="${1:-tiara2}"

if ! git lfs version >/dev/null 2>&1; then
  echo "Git LFS is required. Install it, then rerun this script." >&2
  exit 1
fi
git lfs install --local
git lfs pull

conda create --yes --override-channels \
  --name "$env_name" \
  --channel pytorch \
  --channel conda-forge \
  python=3.10 \
  pip \
  'setuptools<81' \
  numpy=1.26.4 \
  pytorch=2.4.1 \
  scikit-learn=1.5.1 \
  joblib=1.4.2 \
  numba=0.58.1 \
  llvmlite=0.41.1 \
  'scipy>=1.15,<1.16'

conda run --name "$env_name" python -m pip install --no-deps --editable "$repo_root"
echo "Created Conda environment: $env_name"
