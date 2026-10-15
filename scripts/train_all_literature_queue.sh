#!/usr/bin/env bash
# Train the remaining literature-review blocks sequentially.
# Each config runs three seeds; train_queue.py schedules seeds across GPUs 0/1.
set -euo pipefail

repo="${REPO_ROOT:-$(pwd)}"
run_root=runs/literature_blocks_queue
resume_flag="${RESUME_EXISTING:-1}"
dry_run="${DRY_RUN:-0}"

configs=(
  configs/A1_dynamic_snake_conv_grouped_v1_metricfixed.yaml
  configs/L01_mednext_grouped_v1_metricfixed.yaml
  configs/L02_convnext_v2_grouped_v1_metricfixed.yaml
  configs/L03_msca_grouped_v1_metricfixed.yaml
  configs/L04_dlka_grouped_v1_metricfixed.yaml
  configs/L05_dcnv4_grouped_v1_metricfixed.yaml
  configs/L06_odconv_grouped_v1_metricfixed.yaml
  configs/L07_se_grouped_v1_metricfixed.yaml
  configs/L08_cbam_grouped_v1_metricfixed.yaml
  configs/L09_eca_grouped_v1_metricfixed.yaml
  configs/L10_coordinate_attention_grouped_v1_metricfixed.yaml
  configs/L11_mfmsa_grouped_v1_metricfixed.yaml
  configs/L12_cenet_grouped_v1_metricfixed.yaml
  configs/L13_ad_gbc_grouped_v1_metricfixed.yaml
  configs/L14_mambalite_grouped_v1_metricfixed.yaml
  configs/L15_dysample_grouped_v1_metricfixed.yaml
  configs/L16_carafe_grouped_v1_metricfixed.yaml
  configs/L17_pointrend_grouped_v1_metricfixed.yaml
  configs/L18_maxvit_grouped_v1_metricfixed.yaml
  configs/L19_biformer_grouped_v1_metricfixed.yaml
  configs/L20_tokkan_grouped_v1_metricfixed.yaml
  configs/L21_bilateral_graph_grouped_v1_metricfixed.yaml
  configs/L22_phase_aligned_grouped_v1_metricfixed.yaml
)

cd "$repo"
mkdir -p "$run_root"

for config in "${configs[@]}"; do
  name="${config##*/}"
  name="${name%.yaml}"
  echo "[$(date --iso-8601=seconds)] START $name"
  args=(
    --config "$config"
    --seeds 0 1 2
    --gpus 0 1
    --run-root "$run_root/$name"
  )
  if [ "$resume_flag" = "1" ]; then
    args+=(--resume-existing)
  fi
  if [ "$dry_run" = "1" ]; then
    printf 'python scripts/train_queue.py'
    printf ' %q' "${args[@]}"
    printf '\n'
    continue
  fi
  python scripts/train_queue.py "${args[@]}"
  echo "[$(date --iso-8601=seconds)] DONE $name"
done

echo "[$(date --iso-8601=seconds)] ALL 23 LITERATURE BLOCK CONFIGS COMPLETED"
