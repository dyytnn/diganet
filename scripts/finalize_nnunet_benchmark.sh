#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
seed2_container=matblast-nnunet-seed2
mass_container=matblast-massnet-table1-queue
blast_container=matblast-blastnet-reconstruction-queue

seed2_exit=$(docker wait "$seed2_container")
if [ "$seed2_exit" != "0" ]; then
    echo "nnU-Net seed 2 exited with $seed2_exit" >&2
    exit 1
fi

test -s "$repo/runs/nnunet/seed0/Dataset601_MATBLAST/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
test -s "$repo/runs/nnunet/seed1/Dataset601_MATBLAST/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
test -s "$repo/runs/nnunet/seed2/Dataset601_MATBLAST/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth"

mass_exit=$(docker wait "$mass_container")
if [ "$mass_exit" != "0" ]; then
    echo "MASS-Net queue exited with $mass_exit" >&2
    exit 1
fi
for seed in 0 1 2; do
    test -s "$repo/runs/MASS_massnet_table1_grouped_v1_metricfixed_seed${seed}/best.pt"
done

blast_exit=$(docker wait "$blast_container")
if [ "$blast_exit" != "0" ]; then
    echo "BLAST-Net reconstruction queue exited with $blast_exit" >&2
    exit 1
fi
for seed in 0 1 2; do
    test -s "$repo/runs/B10_blastnet_documented_recon_grouped_v1_metricfixed_seed${seed}/best.pt"
done

docker run --rm --name matblast-eval-nnunet-after-seed2 --gpus all --shm-size=16g \
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
      python scripts/eval_benchmark.py \
        --spec B10_MASSNet=configs/MASS_massnet_grouped_v1_metricfixed.yaml:runs/MASS_massnet_table1_grouped_v1_metricfixed_seed0/best.pt,runs/MASS_massnet_table1_grouped_v1_metricfixed_seed1/best.pt,runs/MASS_massnet_table1_grouped_v1_metricfixed_seed2/best.pt \
        --out results/massnet_validation_no_tta.json \
        --csv-out results/massnet_validation_no_tta.csv --num-workers 0
      python scripts/eval_benchmark.py \
        --spec B11_BLASTNet_documented_recon=configs/B10_blastnet_documented_recon_grouped_v1_metricfixed.yaml:runs/B10_blastnet_documented_recon_grouped_v1_metricfixed_seed0/best.pt,runs/B10_blastnet_documented_recon_grouped_v1_metricfixed_seed1/best.pt,runs/B10_blastnet_documented_recon_grouped_v1_metricfixed_seed2/best.pt \
        --out results/blastnet_documented_recon_validation_no_tta.json \
        --csv-out results/blastnet_documented_recon_validation_no_tta.csv --num-workers 0
      python scripts/merge_benchmark_reports.py \
        --generic results/benchmark_validation_with_per_image_no_tta.json \
        --nnunet results/nnunet_validation_no_tta.json \
        --additional results/massnet_validation_no_tta.json \
        --additional results/blastnet_documented_recon_validation_no_tta.json \
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
