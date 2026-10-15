#!/usr/bin/env bash
# Evaluate the 5-fold CV runs of the frozen shortlist (S24, S43, S54 x 3 seeds; nnU-Net x 1 seed)
# on each fold's held-out images, then pool folds 1-4 ("clean") vs fold 0 ("selection fold").
# Usage: scripts/eval_cv5.sh [--skip-nnunet]
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH=src
skip_nnunet=0; [ "${1:-}" = "--skip-nnunet" ] && skip_nnunet=1
ckpts() { echo "$1_seed0/best.pt,$1_seed1/best.pt,$1_seed2/best.pt"; }
for k in 0 1 2 3 4; do
  specs=()
  for stem in S24_pyramid_context S43_matblast_point_rend S54_matblast_bmask_boundary; do
    cfg="CV5_${stem}_fold${k}"; run="runs/cv5_queue/${cfg}/${cfg}"
    for s in 0 1 2; do [ -f "${run}_seed$s/best.pt" ] || { echo "missing ${run}_seed$s/best.pt" >&2; exit 1; }; done
    specs+=(--spec "${cfg}=configs/${cfg}.yaml:$(ckpts "$run")")
  done
  [ -s "results/cv5/fold${k}_validation_no_tta.json" ] || python scripts/eval_benchmark.py "${specs[@]}" \
    --out "results/cv5/fold${k}_validation_no_tta.json" --csv-out "results/cv5/fold${k}_validation_no_tta.csv" --num-workers 2 >/dev/null
  if [ "$skip_nnunet" = 0 ] && [ ! -s "results/cv5/fold${k}_nnunet_no_tta.json" ]; then
    docker run --rm --name "matblast-eval-nnunet-cv5-fold${k}" --gpus all --shm-size=16g \
      -v "$PWD:/workspace" -w /workspace -e PYTHONPATH=/workspace/src \
      pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime bash -lc "
        pip install -q -e third_party/nnUNet >/dev/null 2>&1; pip install -q pyyaml scipy >/dev/null 2>&1
        python scripts/eval_nnunet.py --fold ${k} --split data/splits/source_grouped_5fold_v1.json \
          --model-folder 0=runs/nnunet_cv5/fold${k}/Dataset601_MATBLAST/nnUNetTrainer__nnUNetPlans__2d \
          --out results/cv5/fold${k}_nnunet_no_tta.json --csv-out results/cv5/fold${k}_nnunet_no_tta.csv --device cuda" >/dev/null
  fi
  echo "evaluated fold $k"
done
python scripts/paper/cv5_pooled_analysis.py --out-dir results/cv5 --doc docs/cv5_results.md
