# DiGA-Net — blastocyst segmentation at the annotation limit

Code, experiment configurations, frozen splits, evaluation tables and figures behind two manuscripts:

* *DiGA-Net: lightweight blastocyst segmentation at the limit set by label boundary ambiguity*
* *A diagnostic for annotation-limited segmentation benchmarks, applied to human blastocyst imaging*

Both report identical numbers from the result files released here. **65 architectures, three seeds each, one
embryo-grouped protocol, one statistical protocol.** The finding: every model lands within foreground mean
IoU 0.816–0.838, the per-image error is shared across architectures, independent models agree with each
other more than with the annotation, and displacing the annotation by two pixels reproduces the whole
published range on this dataset.

## For a reviewer in a hurry

Every figure and table of both manuscripts is regenerated from the released result files by one command,
with no GPU and no image data:

```bash
pip install -e .
bash scripts/reproduce_paper.sh --figures   # every figure into figures/paper/ and every table
```

The manuscript sources themselves are not in this repository, so the script stops after the figures and
tables rather than compiling a PDF.

The five-test diagnostic is a standalone tool. Run it on our score table, or on your own:

```bash
python scripts/paper/benchmark_diagnostic.py \
    --scores results/paper/per_image_iou.csv --cls TE \
    --preds results/paper/preds --gt results/paper/preds/gt.npy
```

It prints the five verdicts and reproduces the values quoted in the Results to the third decimal
(variance 0.784 image / 0.065 method, median Spearman 0.872, oracle 0.760 → 0.810, method–method
0.80–0.87 against method–reference 0.74–0.78, reference perturbation 0.892 / 0.797 / 0.713). Tests 1–3
need only the score table; tests 4–5 need the stored predictions, so extract them first:
`tar -xf results/paper/preds.tar.zst -C results/paper`.

## Where each claim's evidence lives

| Claim | Evidence file |
|---|---|
| Per-image, per-seed, per-class score of every model | `results/paper/per_image_iou.csv` |
| Model summary, cost, agreement, oracle, variance decomposition | `results/paper/*.csv` |
| 5-fold cross-validated shortlist and its paired statistics | `results/cv5/cv5_summary.json` |
| The eleven refinement methods against the pre-registered criterion | `results/paper/stage_deltas.csv` |
| Paired effect sizes and win rates for the headline contrasts | `results/paper/effect_sizes.json` |
| Topology and anatomical nesting, annotation included | `results/paper/topology_summary.csv` |
| Robustness under corruption | `results/paper/robustness.csv` |
| Reverse-lever self-supervised pretraining result | `results/reverse_lever/summary.json` |
| Every model's exact configuration | `configs/` (one YAML per run, 135 files) |
| Frozen splits | `data/splits/source_grouped_v1.json`, `source_grouped_5fold_v1.json` |

### The pre-registered criterion

The eleven boundary and sparse-refinement methods were run as a falsification test. The criterion below was
fixed in the project protocol before any of them was trained, and no method met it:

> A refinement method counts as an improvement only if the paired difference to the base model is positive
> for **both** trophectoderm and inner cell mass, with a 95 % hierarchical-bootstrap confidence interval
> excluding zero, over three seeds on the same images.

The protocol also fixed, before training, that aggregate scores including background would not be the
primary metric, that splits would be grouped by embryo, that per-class results would always be reported,
and that no hard anatomical-nesting constraint would be applied because hatching embryos violate it.
`results/paper/stage_deltas.csv` holds every one of those paired differences.

## Repository layout

```
src/matblast/        models, modules (one file per building block), losses, data, metrics, eval
configs/             one YAML per run; the file name is the run identity
scripts/             train.py, eval_benchmark.py, queue runners, and scripts/paper/* for every figure and table
scripts/paper/       benchmark_diagnostic.py is the standalone five-test tool
data/splits/         frozen splits, read-only
results/             released analysis outputs: score tables, JSON, and the compressed predictions
figures/paper/       every figure of both manuscripts, as generated
third_party/         vendored inference code for the reproduced baselines, with their licences
tests/               149 tests; `PYTHONPATH=src pytest tests/ -q`
```

The code follows three rules throughout: every hyperparameter is an explicit named config key so a typo
raises rather than silently changing a run; every feature module is registered and defaults to off, so a
configuration reproduces exactly one experiment; and a regression test pins the baseline model's weights
and seeded output, so no later feature can change it unnoticed.

## Data not in this repository

* **Images and expert annotations.** The 249 Hoffman-modulation-contrast images and their zona,
  trophectoderm and inner-cell-mass masks are the public dataset of Saeedi et al., *IEEE TBME* 64(12),
  2968–2978 (2017), doi:10.1109/TBME.2017.2759665. Obtain them from the original authors. Place them at
  `dataset/source/images` and `dataset/source/masks`; the derived blastocoel masks are then rebuilt by
  `python data/prepare.py`, and its quality-control statistics are written next to them.
* **Predicted label maps** are included, compressed: `results/paper/preds.tar.zst` (3.7 MB, 118 MB
  extracted). Restore them with `tar -xf results/paper/preds.tar.zst -C results/paper` before running
  tests 4–5 of the diagnostic or the qualitative figures. They contain no image data.
* **Checkpoints** (23 GB) and **nnU-Net preprocessed data** (381 MB) are not distributed. Training a
  compact model from scratch takes 13–21 minutes per seed on one GPU; `scripts/train_stage_queue.sh`
  reproduces any run from its config.
* **Time-lapse video and its manual annotations** are patient data from a partner clinic and are not
  released in any form: no frame, no mask, no filename and no patient or embryo identifier appears in this
  repository. The self-supervised pretraining experiment that used them appears only as the aggregate
  foreground IoU of its folds, scored on the public images.

## Reproducing the experiments themselves

```bash
PYTHONPATH=src python scripts/train.py --config configs/S43_matblast_point_rend_grouped_v1.yaml --seed 0
PYTHONPATH=src python scripts/eval_benchmark.py --spec "name=configs/<cfg>.yaml:runs/<run>_seed0/best.pt,..." \
    --out results/<name>_validation_no_tta.json --csv-out results/<name>_validation_no_tta.csv
bash scripts/eval_cv5.sh                 # the 5-fold cross-validated shortlist, nnU-Net inside docker
```

nnU-Net is run unmodified from `third_party/nnUNet` in the official v2 2D configuration, inside the
`pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` image; `scripts/train_nnunet_cv5.sh` has the exact
invocation.

## Licence and citation

Code: MIT (`LICENSE`). The dataset remains under the terms set by its original authors and was used here
for non-commercial academic research only. If you use this code or the diagnostic, please cite the relevant
manuscript above and the dataset paper of Saeedi et al.
