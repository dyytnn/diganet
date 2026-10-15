#!/usr/bin/env bash
# Evaluate all completed literature-block queue runs on the fixed validation split.
set -euo pipefail

repo="${REPO_ROOT:-$(pwd)}"
run_root="${RUN_ROOT:-runs/literature_blocks_queue}"
out="${OUT_JSON:-results/literature_blocks_validation_no_tta.json}"
csv_out="${OUT_CSV:-results/literature_blocks_validation_no_tta.csv}"
cd "$repo"

configs=(configs/A1_dynamic_snake_conv_grouped_v1_metricfixed.yaml configs/L*.yaml)
if [ "${#configs[@]}" -ne 23 ]; then
  echo "expected 23 literature configs, found ${#configs[@]}" >&2
  exit 1
fi

args=()
for config in "${configs[@]}"; do
  stem="${config##*/}"
  stem="${stem%.yaml}"
  checkpoints=()
  for seed in 0 1 2; do
    checkpoint="$run_root/$stem/${stem}_seed${seed}/best.pt"
    if [ ! -s "$checkpoint" ]; then
      echo "missing checkpoint: $checkpoint" >&2
      exit 1
    fi
    checkpoints+=("$checkpoint")
  done
  args+=(--spec "$stem=$config:${checkpoints[0]},${checkpoints[1]},${checkpoints[2]}")
done

python scripts/eval_benchmark.py "${args[@]}" \
  --out "$out" --csv-out "$csv_out" --num-workers "${NUM_WORKERS:-2}"
