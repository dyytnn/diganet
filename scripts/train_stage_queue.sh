#!/usr/bin/env bash
# Train a list of stage configs, 3 seeds each, sequentially (one stage = one run root).
# Usage: scripts/train_stage_queue.sh <run_root> <config_stem> [<config_stem> ...]
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
run_root="$1"; shift
for stem in "$@"; do
  cfg="configs/${stem}.yaml"
  [ -f "$cfg" ] || { echo "missing $cfg" >&2; exit 1; }
  mkdir -p "$run_root/$stem"
  PYTHONPATH=src python scripts/train_queue.py --config "$cfg" --seeds 0 1 2 --gpus 0 1 \
    --run-root "$run_root/$stem" --resume-existing
done
echo "QUEUE_DONE $*"
