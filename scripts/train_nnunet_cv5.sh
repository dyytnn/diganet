#!/usr/bin/env bash
# Train the frozen nnU-Net recipe on the 5-fold patient-grouped split (one seed per fold),
# reproducing the original docker environment. Usage: scripts/train_nnunet_cv5.sh <gpu> <fold> [<fold> ...]
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
gpu="$1"; shift
for fold in "$@"; do
  out="$repo/runs/nnunet_cv5/fold${fold}"
  if [ -s "$out/Dataset601_MATBLAST/nnUNetTrainer__nnUNetPlans__2d/fold_${fold}/checkpoint_final.pth" ]; then echo "fold $fold done, skip"; continue; fi
  mkdir -p "$out"
  docker run --rm --name "matblast-nnunet-cv5-fold${fold}" --gpus "\"device=${gpu}\"" --shm-size=16g \
    -v "$repo:/workspace" -w /workspace \
    -e nnUNet_raw=/workspace/nnunet/raw -e nnUNet_preprocessed=/workspace/nnunet/preprocessed_cv5 \
    -e nnUNet_results="/workspace/runs/nnunet_cv5/fold${fold}" -e PYTHONPATH=/workspace/src \
    pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime \
    bash -lc "apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq git gcc g++ >/dev/null 2>&1; git config --global --add safe.directory /workspace; pip install -q -e third_party/nnUNet >/dev/null 2>&1; pip install -q pyyaml >/dev/null 2>&1; python scripts/train_nnunet_seed.py --seed 0 --fold ${fold}" \
    > "$out/train.log" 2>&1
  echo "fold $fold exit $?"
done
echo "NNUNET_CV5_DONE gpu=$gpu folds=$*"
