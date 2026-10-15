#!/usr/bin/env bash
# Package the stored predicted label maps into a single compressed archive tracked in the repository.
#
# results/paper/preds holds the inputs of tests 4-5 of the diagnostic and of every qualitative figure:
# int8 arrays of shape (50, 256, 256) with class ids 0..4, one per model and seed, plus gt.npy and
# image_ids.json. Raw they are 118 MB, but label maps of five values compress to about 3.7 MB, so the
# archive is tracked in the repository and a fresh clone can run all five tests.
#
#   bash scripts/package_predictions.sh     # writes results/paper/preds.tar.zst
#   tar -xf results/paper/preds.tar.zst -C results/paper     # restore on a fresh clone
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
[ -d results/paper/preds ] || { echo "results/paper/preds not found" >&2; exit 1; }
n=$(ls results/paper/preds/*.npy | wc -l)
if command -v zstd > /dev/null; then
  out=results/paper/preds.tar.zst
  tar --zstd -cf "$out" -C results/paper preds
else
  out=results/paper/preds.tar.gz
  tar -czf "$out" -C results/paper preds
fi
printf '%s: %s arrays, %s\n' "$out" "$n" "$(du -h "$out" | cut -f1)"
cat > results/paper/PREDICTIONS.md <<'INNER'
# Predicted label maps

Restore them on a fresh clone with

    tar -xf results/paper/preds.tar.zst -C results/paper

Contents: `<model>_seed<k>.npy`, int8, shape (50, 256, 256), class ids 0=BG, 1=ZP, 2=TE, 3=BL, 4=ICM, in
the order of `image_ids.json`; `gt.npy` is the reference in the same layout. These are the inputs of
tests 4 and 5 of `scripts/paper/benchmark_diagnostic.py` and of the qualitative figures. They contain no
image data and no patient information.
INNER
echo "wrote results/paper/PREDICTIONS.md"
