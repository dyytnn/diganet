#!/usr/bin/env python3
"""The five-test diagnostic for annotation-limited segmentation benchmarks (paper_nature Methods).

Standalone: nothing here is specific to blastocysts, to this metric or to this repository's models.

Tests 1-3 need only a per-image score table (CSV with columns model, seed, image, cls, iou -- extra columns
are ignored). Tests 4-5 additionally need the predicted label maps and the reference, because they compare
masks rather than scores; pass --preds and --gt for those.

  Test 1  variance decomposition        image vs method component of the score variance
  Test 2  rank agreement                median Spearman correlation of per-image scores across method pairs
  Test 3  oracle bound                  best-of-n selector over the method pool, against n
  Test 4  agreement asymmetry           method-method vs method-reference overlap        (needs masks)
  Test 5  reference perturbation        reference against itself, eroded/dilated/shifted (needs masks)

    python scripts/paper/benchmark_diagnostic.py --scores results/paper/per_image_iou.csv --cls TE \
        --preds results/paper/preds --gt results/paper/preds/gt.npy
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ tests 1-3: scores only
def variance_decomposition(w: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Two-way (image x method) decomposition of the seed-averaged score variance."""
    m = w.to_numpy(float)
    def parts(a: np.ndarray) -> tuple[float, float, float]:
        grand = np.nanmean(a)
        row, col = np.nanmean(a, 1), np.nanmean(a, 0)
        ss_img = a.shape[1] * np.nansum((row - grand) ** 2)
        ss_mod = a.shape[0] * np.nansum((col - grand) ** 2)
        tot = np.nansum((a - grand) ** 2)
        return ss_img / tot, ss_mod / tot, 1 - (ss_img + ss_mod) / tot
    point = parts(m)
    boot = np.array([parts(m[RNG.integers(0, m.shape[0], m.shape[0])][:, RNG.integers(0, m.shape[1], m.shape[1])]) for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {"image": point[0], "method": point[1], "residual": point[2],
            "image_ci": [lo[0], hi[0]], "method_ci": [lo[1], hi[1]],
            "verdict": "image-dominated" if point[0] > 10 * point[1] else "method still matters"}


def rank_agreement(w: pd.DataFrame) -> dict:
    cols = list(w.columns)
    rhos = [stats.spearmanr(w[a], w[b], nan_policy="omit").statistic for a, b in itertools.combinations(cols, 2)]
    rhos = [r for r in rhos if r == r]
    return {"median_spearman": float(np.median(rhos)), "iqr": [float(np.percentile(rhos, 25)), float(np.percentile(rhos, 75))],
            "n_pairs": len(rhos), "verdict": "methods fail on the same images" if np.median(rhos) > 0.7 else "failures are method-specific"}


def oracle_bound(w: pd.DataFrame, n_draws: int = 200) -> dict:
    m = w.to_numpy(float)
    ns = sorted({1, 2, 3, 5, 10, 20, 40, m.shape[1]} & set(range(1, m.shape[1] + 1)))
    curve = {}
    for n in ns:
        best = [np.nanmean(np.nanmax(m[:, RNG.choice(m.shape[1], n, replace=False)], axis=1)) for _ in range(n_draws)]
        curve[n] = float(np.mean(best))
    mean_of_methods = float(np.nanmean(m))
    gain = curve[ns[-1]] - curve[1]
    return {"curve": curve, "mean_over_methods": mean_of_methods, "oracle_gain": float(gain),
            "verdict": "oracle saturates far from 1.0" if curve[ns[-1]] < 0.95 else "selection still helps"}


# ------------------------------------------------------------------ tests 4-5: masks
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else np.nan


def agreement_asymmetry(preds: dict[str, np.ndarray], gt: np.ndarray, cls: int) -> dict:
    names = list(preds)
    mm = [np.nanmean([_iou(preds[a][i] == cls, preds[b][i] == cls) for i in range(len(gt))])
          for a, b in itertools.combinations(names, 2)]
    mr = [np.nanmean([_iou(preds[a][i] == cls, gt[i] == cls) for i in range(len(gt))]) for a in names]
    return {"method_method": [float(np.min(mm)), float(np.max(mm))], "method_reference": [float(np.min(mr)), float(np.max(mr))],
            "verdict": "methods resemble each other more than the reference" if np.median(mm) > np.median(mr) else "reference is the closest object"}


def reference_perturbation(gt: np.ndarray, cls: int, ks: tuple[int, ...] = (1, 2, 3)) -> dict:
    out = {}
    for k in ks:
        vals = []
        for g in gt:
            m = g == cls
            if not m.any():
                continue
            vals += [_iou(m, ndimage.binary_erosion(m, iterations=k)),
                     _iou(m, ndimage.binary_dilation(m, iterations=k)),
                     _iou(m, np.roll(m, k, axis=1))]
        out[k] = float(np.nanmean(vals))
    return {"self_overlap_by_shift_px": out,
            "verdict": "compare the published spread of the field with this band"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="CSV with columns model, seed, image, cls, iou")
    ap.add_argument("--cls", default="TE", help="class to analyse in the score table")
    ap.add_argument("--preds", help="directory of <model>_seed0.npy label maps (tests 4-5)")
    ap.add_argument("--gt", help="reference label map .npy (tests 4-5)")
    ap.add_argument("--gt-class-id", type=int, default=2, help="integer id of --cls in the label maps")
    ap.add_argument("--out", help="write the report as JSON here")
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    df = df[df.cls == args.cls]
    w = df.groupby(["image", "model"]).iou.mean().unstack()
    report = {"n_images": int(w.shape[0]), "n_methods": int(w.shape[1]), "class": args.cls,
              "test1_variance_decomposition": variance_decomposition(w),
              "test2_rank_agreement": rank_agreement(w),
              "test3_oracle_bound": oracle_bound(w)}

    if args.preds and args.gt:
        gt = np.load(args.gt)
        preds = {p.name.rsplit("_seed", 1)[0]: np.load(p) for p in sorted(Path(args.preds).glob("*_seed0.npy"))}
        report["test4_agreement_asymmetry"] = agreement_asymmetry(preds, gt, args.gt_class_id)
        report["test5_reference_perturbation"] = reference_perturbation(gt, args.gt_class_id)
    else:
        report["test4_agreement_asymmetry"] = "skipped: needs --preds and --gt"
        report["test5_reference_perturbation"] = "skipped: needs --gt"

    failed = sum(1 for k, v in report.items() if isinstance(v, dict) and "verdict" in v
                 and v["verdict"] in ("image-dominated", "methods fail on the same images",
                                      "oracle saturates far from 1.0", "methods resemble each other more than the reference"))
    report["tests_indicating_annotation_limit"] = failed
    print(json.dumps(report, indent=2, default=float))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
