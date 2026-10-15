#!/usr/bin/env bash
# Evaluate finished stage runs against S24 and S43 and re-render REPORT.md.
# Usage: scripts/eval_stage_queue.sh <run_root> <config_stem> [<config_stem> ...]
# Every stem needs <run_root>/<stem>/<stem>_seed{0,1,2}/best.pt. Produces per stem:
#   results/<stem>_validation_no_tta.json      (S24 + S43 + stem, per-image IoU)
#   results/<stem>_paired_s24_no_tta.json      results/<stem>_paired_s43_no_tta.json
#   results/<stem>_extended_no_tta.json
# and re-renders REPORT.md with every stage report found under results/.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH=src
run_root="$1"; shift
S24_CFG=configs/S24_pyramid_context_grouped_v1_metricfixed.yaml
S24_RUN=runs/S24_pyramid_context_grouped_v1_metricfixed/S24_pyramid_context_grouped_v1_metricfixed
S43_CFG=configs/S43_matblast_point_rend_grouped_v1.yaml
S43_RUN=runs/matblast_shortlist_queue/S43_matblast_point_rend_grouped_v1/S43_matblast_point_rend_grouped_v1
ckpts() { echo "$1_seed0/best.pt,$1_seed1/best.pt,$1_seed2/best.pt"; }
for stem in "$@"; do
  run="$run_root/$stem/$stem"
  for s in 0 1 2; do [ -f "${run}_seed$s/best.pt" ] || { echo "missing ${run}_seed$s/best.pt" >&2; exit 1; }; done
  python scripts/eval_benchmark.py \
    --spec "S24=$S24_CFG:$(ckpts "$S24_RUN")" --spec "S43=$S43_CFG:$(ckpts "$S43_RUN")" \
    --spec "$stem=configs/$stem.yaml:$(ckpts "$run")" \
    --out "results/${stem}_validation_no_tta.json" --csv-out "results/${stem}_validation_no_tta.csv" --num-workers 2 >/dev/null
  for ref in S24 S43; do
    python scripts/pair_benchmark_against_s24.py --benchmark "results/${stem}_validation_no_tta.json" --reference "$ref" \
      --out "results/${stem}_paired_${ref,,}_no_tta.json" --csv-out "results/${stem}_paired_${ref,,}_no_tta.csv" \
      --bootstrap-samples 10000 --seed 0 >/dev/null
  done
  python scripts/eval_extended_metrics.py --spec "$stem=configs/$stem.yaml:$(ckpts "$run")" \
    --out "results/${stem}_extended_no_tta.json" --csv-out "results/${stem}_extended_no_tta.csv" >/dev/null
  if [ "${PAIR_NNUNET:-0}" = "1" ]; then
    # nnU-Net pairing needs the stem merged into the full benchmark (per-image IoU of B9_nnunet lives there)
    python scripts/eval_benchmark.py --spec "$stem=configs/$stem.yaml:$(ckpts "$run")" \
      --out "results/${stem}_only_validation_no_tta.json" --csv-out "results/${stem}_only_validation_no_tta.csv" --num-workers 2 >/dev/null
    additional=()
    for f in s33_adaptive_skip_fusion s34_frequency_multiscale b12_blasto_net s36_convnext s37_residual_encoder s38_adaptive_receptive_field s40_antialiased_downsampling s41_cnn_ssm s42_token_moe; do additional+=(--additional "results/${f}_validation_no_tta.json"); done
    python scripts/merge_benchmark_reports.py --generic results/benchmark_validation_with_per_image_no_tta.json --nnunet results/nnunet_validation_no_tta.json \
      "${additional[@]}" --additional "results/${stem}_only_validation_no_tta.json" \
      --out "results/benchmark_all_plus_${stem}_validation_no_tta.json" --csv-out "results/benchmark_all_plus_${stem}_validation_no_tta.csv" >/dev/null
    python scripts/pair_benchmark_against_s24.py --benchmark "results/benchmark_all_plus_${stem}_validation_no_tta.json" --reference B9_nnunet \
      --out "results/${stem}_paired_b9_nnunet_no_tta.json" --csv-out "results/${stem}_paired_b9_nnunet_no_tta.csv" --bootstrap-samples 10000 --seed 0 >/dev/null
  fi
  echo "evaluated $stem"
done
# Re-render with every known section; stage reports are discovered by file pattern.
extended_args=(--shortlist-extended results/shortlist_extended_no_tta.json --shortlist-extended results/matblast_s47_extended_no_tta.json --shortlist-extended results/v2_0_extended_no_tta.json)
paired_args=(--custom-paired results/matblast_shortlist_paired_s24_no_tta.json --custom-paired results/matblast_s47_paired_s24_no_tta.json --custom-paired results/matblast_s47_paired_s43_no_tta.json --custom-paired results/v2_0_paired_s24_no_tta.json --custom-paired results/v2_0_paired_b9_nnunet_no_tta.json)
for f in results/S[45][0-9]_*_extended_no_tta.json results/V2_[1-9]_*_extended_no_tta.json; do [ -f "$f" ] && extended_args+=(--shortlist-extended "$f"); done
for f in results/S[45][0-9]_*_paired_s24_no_tta.json results/S[45][0-9]_*_paired_s43_no_tta.json results/V2_[1-9]_*_paired_*_no_tta.json; do [ -f "$f" ] && paired_args+=(--custom-paired "$f"); done
python scripts/render_benchmark_report.py \
  --benchmark results/benchmark_all_validation_no_tta.json --paired results/benchmark_all_paired_s24_no_tta.json \
  --extended results/benchmark_extended_metrics_no_tta.json \
  --literature results/literature_blocks_validation_no_tta.json --literature-paired results/literature_blocks_paired_s24_no_tta.json \
  "${extended_args[@]}" "${paired_args[@]}" --out REPORT.md
echo "REPORT.md rendered"
