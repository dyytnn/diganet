#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
compiler=matblast-nnunet-compiler
generic=results/benchmark_validation_with_per_image_no_tta.json

while [ "$(docker inspect -f '{{.State.Running}}' "$compiler" 2>/dev/null || true)" = "true" ]; do
    sleep 60
done
compiler_exit=$(docker inspect -f '{{.State.ExitCode}}' "$compiler" 2>/dev/null || echo 125)
if [ "$compiler_exit" != "0" ]; then
    echo "nnU-Net compiler exited with $compiler_exit" >&2
    exit 1
fi
while [ ! -s "$repo/$generic" ]; do
    generic_running=$(docker inspect -f '{{.State.Running}}' matblast-eval-cpu-paired-retry 2>/dev/null || true)
    if [ "$generic_running" != "true" ]; then
        echo "generic evaluator ended without $generic" >&2
        exit 1
    fi
    sleep 60
done

docker run --rm --name matblast-eval-nnunet-after-queue --gpus all --shm-size=16g \
    -v "$repo:/workspace" -w /workspace \
    pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime bash -lc '
      set -euo pipefail
      apt-get update -qq && apt-get install -y -qq git >/dev/null
      pip install -e . >/dev/null
      pip install -e third_party/nnUNet >/dev/null
      python scripts/eval_nnunet.py \
        --model-folder 0=runs/nnunet/seed0/Dataset601_MATBLAST/nnUNetTrainer__nnUNetPlans__2d \
        --model-folder 1=runs/nnunet/seed1/Dataset601_MATBLAST/nnUNetTrainer__nnUNetPlans__2d \
        --model-folder 2=runs/nnunet/seed2/Dataset601_MATBLAST/nnUNetTrainer__nnUNetPlans__2d \
        --out results/nnunet_validation_no_tta.json \
        --csv-out results/nnunet_validation_no_tta.csv --device cuda
      python scripts/merge_benchmark_reports.py \
        --generic results/benchmark_validation_with_per_image_no_tta.json \
        --nnunet results/nnunet_validation_no_tta.json \
        --out results/benchmark_all_validation_no_tta.json \
        --csv-out results/benchmark_all_validation_no_tta.csv
      python scripts/pair_benchmark_against_s24.py \
        --benchmark results/benchmark_all_validation_no_tta.json \
        --out results/benchmark_all_paired_s24_no_tta.json \
        --csv-out results/benchmark_all_paired_s24_no_tta.csv \
        --bootstrap-samples 10000 --seed 0
      python scripts/render_benchmark_report.py \
        --benchmark results/benchmark_all_validation_no_tta.json \
        --paired results/benchmark_all_paired_s24_no_tta.json --out REPORT.md
    '
