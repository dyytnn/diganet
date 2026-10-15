#!/usr/bin/env python3
"""Standardised effect size and win rate for the paired contrasts quoted in both manuscripts.

A paired difference and its confidence interval say whether an effect is distinguishable from zero; they do
not say how large it is relative to the spread between images, nor how often one method actually wins. Two
scale-free companions are reported alongside every headline contrast:

  dz        Cohen's d for paired samples: mean(delta_i) / sd(delta_i) over the seed-averaged per-image
            differences. Conventional anchors are 0.2 small, 0.5 medium, 0.8 large.
  win rate  fraction of images on which the candidate's seed-averaged score exceeds the reference's.
            0.5 is a coin flip; it is insensitive to a few large per-image swings.

Both are computed on the seed-averaged per-image difference, the same quantity the Wilcoxon test uses, so
the three statistics describe one object.

    PYTHONPATH=src python scripts/paper/effect_sizes.py     ->  results/paper/effect_sizes.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

VALIDATION_PAIRS = [
    ("S24", "S43"),
    ("S24", "B9_nnunet"),
    ("S43", "B9_nnunet"),
    ("S24", "S54_matblast_bmask_boundary_grouped_v1"),
    ("B0", "S43"),
]
CV_PAIRS = [("S24", "S43"), ("S43", "B9_nnunet"), ("S24", "B9_nnunet")]
CV_SPEC = {"S24": "CV5_S24_pyramid_context_fold{k}", "S43": "CV5_S43_matblast_point_rend_fold{k}"}


def summarise(delta_per_image: np.ndarray) -> dict:
    d = delta_per_image[np.isfinite(delta_per_image)]
    sd = float(np.std(d, ddof=1))
    return {
        "n_images": int(d.size),
        "delta": float(np.mean(d)),
        "dz": float(np.mean(d) / sd) if sd > 0 else float("nan"),
        "win_rate": float(np.mean(d > 0)),
    }


def validation_table() -> dict:
    per = pd.read_csv(ROOT / "results/paper/per_image_iou.csv")
    per = per[per.cls != "BG"]

    def mat(model: str) -> np.ndarray:
        return per[per.model == model].groupby(["seed", "image"]).iou.mean().unstack().to_numpy()

    out = {}
    for ref, cand in VALIDATION_PAIRS:
        delta = np.nanmean(mat(cand), axis=0) - np.nanmean(mat(ref), axis=0)
        out[f"{style.display_name(cand)} - {style.display_name(ref)}"] = summarise(delta)
    return out


def cv_table() -> dict:
    def fold(path: Path, keys: list[str]) -> np.ndarray:
        """Seed-averaged mIoU-FG per image; the nnU-Net files key seeds as "0", the rest as "<spec>:seed<k>"."""
        j = json.load(open(path))
        a = np.array([j["per_image_iou"][k] for k in keys], dtype=float)
        return np.nanmean(np.nanmean(a[:, :, 1:], axis=2), axis=0)

    scores: dict[str, list[np.ndarray]] = {k: [] for k in ("S24", "S43", "B9_nnunet")}
    for k in (1, 2, 3, 4):  # clean folds only: fold 0 was the selection set
        j = ROOT / f"results/cv5/fold{k}_validation_no_tta.json"
        for key, spec in CV_SPEC.items():
            scores[key].append(fold(j, [f"{spec.format(k=k)}:seed{s}" for s in range(3)]))
        nn = ROOT / f"results/cv5/fold{k}_nnunet_no_tta.json"
        scores["B9_nnunet"].append(fold(nn, list(json.load(open(nn))["per_image_iou"])))
    pooled = {k: np.concatenate(v) for k, v in scores.items()}
    return {
        f"{style.display_name(cand)} - {style.display_name(ref)}": summarise(pooled[cand] - pooled[ref])
        for ref, cand in CV_PAIRS
    }


def main() -> None:
    report = {
        "definition": {
            "dz": "mean(delta_i) / sd(delta_i) over seed-averaged per-image differences (Cohen's d for paired samples)",
            "win_rate": "fraction of images on which the candidate's seed-averaged score exceeds the reference's",
        },
        "validation_50_images": validation_table(),
        "cross_validation_199_clean_images": cv_table(),
    }
    out = ROOT / "results/paper/effect_sizes.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for group in ("validation_50_images", "cross_validation_199_clean_images"):
        print(f"\n{group}")
        for name, r in report[group].items():
            print(f"  {name:<44s} delta {r['delta']:+.4f}   dz {r['dz']:+.2f}   win {100 * r['win_rate']:3.0f}%   n={r['n_images']}")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
