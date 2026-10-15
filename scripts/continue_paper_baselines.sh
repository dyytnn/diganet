#!/usr/bin/env bash
# Continue the paper-faithful queue only after the preceding model succeeds.
set -u

wait_for_success() {
  local container="$1"
  local status
  status="$(docker wait "$container")"
  if [ "$status" != "0" ]; then
    echo "${container} failed with exit=${status}; stopping scheduler" >&2
    exit "$status"
  fi
}

launch_queue() {
  local container="$1"
  local config="$2"
  local run_root="$3"
  docker run -d --rm --name "$container" --gpus '"device=0,1"' \
    -v "$(cd "$(dirname "$0")/.." && pwd)":/workspace \
    -w /workspace \
    pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime \
    bash -lc "pip install -e . && python scripts/train_queue.py --config ${config} --seeds 0 1 2 --gpus 0 1 --run-root ${run_root}"
}

wait_for_success matblast-train-b3
launch_queue matblast-train-b4 configs/B4_i2cnet_grouped_v2_metricfixed.yaml runs/B4_i2cnet_grouped_v2_metricfixed
wait_for_success matblast-train-b4
launch_queue matblast-train-b5 configs/B5_mimics_net_grouped_v3_metricfixed.yaml runs/B5_mimics_net_grouped_v3_metricfixed
wait_for_success matblast-train-b5
echo "Paper-faithful baseline queue completed successfully."
