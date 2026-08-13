#!/usr/bin/env bash
set -euo pipefail

cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

start_time=$(date +%s)
python code/src/train.py
elapsed_seconds=$(( $(date +%s) - start_time ))

printf '训练总耗时: %02d:%02d:%02d\n' \
  $((elapsed_seconds / 3600)) \
  $(((elapsed_seconds % 3600) / 60)) \
  $((elapsed_seconds % 60))
