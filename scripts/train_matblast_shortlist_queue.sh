#!/usr/bin/env bash
# Train shortlisted MAT-BLAST variants one at a time; seeds run on GPUs 0/1.
set -euo pipefail

repo="${REPO_ROOT:-$(pwd)}"
run_root="${RUN_ROOT:-runs/matblast_shortlist_queue}"
resume_flag="${RESUME_EXISTING:-1}"
cd "$repo"

configs=(
  configs/S43_matblast_point_rend_grouped_v1.yaml
  configs/S44_matblast_mednext_grouped_v1.yaml
  configs/S45_matblast_dlka_grouped_v1.yaml
)

for config in "${configs[@]}"; do
  stem="${config##*/}"
  stem="${stem%.yaml}"
  args=(--config "$config" --seeds 0 1 2 --gpus 0 1 --run-root "$run_root/$stem")
  if [ "$resume_flag" = "1" ]; then
    args+=(--resume-existing)
  fi
  echo "[$(date --iso-8601=seconds)] START $stem"
  python scripts/train_queue.py "${args[@]}"
  echo "[$(date --iso-8601=seconds)] DONE $stem"
done

echo "[$(date --iso-8601=seconds)] ALL 3 MAT-BLAST SHORTLIST RUNS COMPLETED"
