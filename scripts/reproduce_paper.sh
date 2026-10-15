#!/usr/bin/env bash
# Single entry point: regenerate every table, figure and both PDFs of the paper from the released result files.
#
#   scripts/reproduce_paper.sh              # figures + tables + supplement + main.pdf + supplement.pdf
#   scripts/reproduce_paper.sh --figures    # stop after figures and tables
#   scripts/reproduce_paper.sh --with-gpu   # also re-run the steps that need checkpoints and a GPU
#   scripts/reproduce_paper.sh --nature     # the Nature Methods version in paper_nature/ (figures must exist)
#
# What is NOT re-run here: training (scripts/train_*_queue.sh, scripts/train_nnunet_cv5.sh) and the per-model
# evaluations (scripts/eval_*.sh, scripts/eval_cv5.sh). Those produce results/*.json from checkpoints and take
# GPU-days; every script below consumes only files already in results/.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH=src
mode="${1:-}"

if [ "$mode" = "--nature" ]; then
  echo "== Nature Methods version"
  [ -d paper_nature ] || { echo "paper_nature/ is not part of the public repository; figures and tables are built by the default mode"; exit 0; }
  python scripts/paper/nature_figures.py
  python scripts/paper/nature_table1.py
  python scripts/paper/nature_extended_data.py
  cd paper_nature
  latexmk -pdf -silent main.tex > /dev/null
  echo "main.pdf $(pdfinfo main.pdf | awk '/Pages/{print $2}') pages, $(grep -c '^!' main.log) LaTeX errors"
  cd ..
  python scripts/paper/nature_limits.py
  echo "== five-test diagnostic (must reproduce the Results values)"
  python scripts/paper/benchmark_diagnostic.py --scores results/paper/per_image_iou.csv --cls TE \
    --preds results/paper/preds --gt results/paper/preds/gt.npy --out results/paper/benchmark_diagnostic_TE.json > /dev/null
  python -c "import json;r=json.load(open('results/paper/benchmark_diagnostic_TE.json'));print('  tests indicating an annotation limit:',r['tests_indicating_annotation_limit'],'of 4 decidable')"
  cd paper_nature
  rm -f nature_submission.zip
  zip -qr nature_submission.zip main.tex refs.bib sn-jnl.cls sn-nature.bst tables figures
  echo "wrote paper_nature/nature_submission.zip"
  exit 0
fi

step() { echo "== $*"; }

# ---------------------------------------------------------------- per-image tables (feeds every figure)
step "per-image scores and model summary"
python scripts/paper/build_per_image_table.py      # -> results/paper/per_image_iou.csv, model_summary.csv

step "main-text figures"
python scripts/paper/ceiling_figure.py             # F1  ceiling vs published numbers
python scripts/paper/difficulty_and_agreement_tables.py  # F2  difficulty, oracle, variance decomposition
python scripts/paper/agreement_and_errors.py       # F3 agreement, F4 error location, F9 qualitative
python scripts/paper/label_sensitivity.py          # F6  GT perturbation
python scripts/paper/forest_and_stats.py           # F7  refinement forest + multiplicity
python scripts/paper/pareto_cost.py                # F8  accuracy vs cost
python scripts/paper/zoom_in_panel.py              # S3  per-class zoom (supplement)
python scripts/paper/recipe_ladder.py              # S1  recipe ladder
python scripts/paper/boundary_tolerant_metrics.py  # C2  trimmed IoU / boundary F

step "effect sizes"
python scripts/paper/effect_sizes.py > /dev/null   # -> results/paper/effect_sizes.json

step "tables"
python scripts/paper/topology_metrics.py > /dev/null   # Table 4 + results/paper/topology_summary.{csv,md}
python scripts/paper/full_table_tex.py                 # Table 7 -> paper/tables/full_models.tex

if [ "$mode" = "--with-gpu" ]; then
  step "GPU steps (need checkpoints under runs/)"
  python scripts/paper/robustness_sweep.py           # S2  corruption sweep
  python scripts/paper/gate_map.py                   # F10 gate maps of DiGA-Net+
else
  step "GPU steps skipped (--with-gpu to run); redrawing the robustness figure from the stored CSV"
  python scripts/paper/robustness_sweep.py --plot-only
fi

cp -f figures/paper/*.png paper/figures/ 2>/dev/null || true
[ "$mode" = "--figures" ] && { echo "figures and tables done"; exit 0; }

if [ ! -d paper ]; then
  echo "== manuscript sources are not part of the public repository; figures and tables are done"
  exit 0
fi

step "supplement source"
python scripts/paper/build_supplement.py           # -> paper/supplement.tex

step "PDFs"
cd paper
latexmk -pdf -silent S7_architecture.tex -cd figures_src/S7_architecture.tex >/dev/null 2>&1 || \
  (cd figures_src && latexmk -pdf -silent S7_architecture.tex >/dev/null 2>&1 && cp S7_architecture.pdf ../figures/ && cp S7_architecture.pdf ../../figures/paper/)
latexmk -pdf -silent main.tex > /dev/null
latexmk -pdf -silent supplement.tex > /dev/null
echo "main.pdf $(pdfinfo main.pdf | awk '/Pages/{print $2}') pages, supplement.pdf $(pdfinfo supplement.pdf | awk '/Pages/{print $2}') pages"

step "submission archive"
rm -f overleaf_upload.zip
zip -qr overleaf_upload.zip main.tex supplement.tex refs.bib cas-dc.cls cas-common.sty thumbnails \
  tables/full_models.tex figures/*.png figures/S7_architecture.pdf
echo "wrote paper/overleaf_upload.zip"
