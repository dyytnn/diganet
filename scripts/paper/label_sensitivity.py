#!/usr/bin/env python3
"""F7: IoU between the ground truth and a k-pixel perturbed copy of itself, per class.

Perturbations: erosion, dilation, shift (x), for k = 1..3 px, at native resolution and at
256x256. The asymptote per class is the annotation-consistency ceiling used in F1.
Writes results/paper/label_sensitivity.csv and figures/paper/F7_label_sensitivity.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
import style  # noqa: E402

CLASSES = {"ZP": "zp", "TE": "te", "BL": "bl", "ICM": "icm"}
COLORS = {"ZP": style.VIOLET, "TE": style.ORANGE, "BL": style.AQUA, "ICM": style.RED}


def load(cls, name, size):
    m = Image.open(ROOT / "dataset/source/masks" / cls / name).convert("L")
    if size:
        m = m.resize((size, size), Image.NEAREST)
    return np.array(m) > 0


def iou(a, b):
    u = (a | b).sum()
    return np.nan if u == 0 else (a & b).sum() / u


def main():
    ids = json.load(open(ROOT / "results/benchmark_all_validation_no_tta.json"))["validation_image_ids"]
    rows = []
    for res_label, size in (("native", None), ("256", 256)):
        for cls, folder in CLASSES.items():
            for n in ids:
                g = load(folder, n, size)
                if g.sum() == 0:
                    continue
                for k in (1, 2, 3):
                    rows.append((res_label, cls, k, "erode", iou(g, ndimage.binary_erosion(g, iterations=k))))
                    rows.append((res_label, cls, k, "dilate", iou(g, ndimage.binary_dilation(g, iterations=k))))
                    rows.append((res_label, cls, k, "shift", iou(g, np.roll(g, k, axis=1))))
    df = pd.DataFrame(rows, columns=["resolution", "cls", "k_px", "perturbation", "iou"])
    df.to_csv(ROOT / "results/paper/label_sensitivity.csv", index=False)
    summary = df.groupby(["resolution", "cls", "k_px", "perturbation"])["iou"].mean().unstack("perturbation")
    print(summary.round(3).to_string())

    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, res_label in zip(axes, ("256", "native")):
        sub = df[df.resolution == res_label]
        for cls, color in COLORS.items():
            m = sub[sub.cls == cls].groupby("k_px")["iou"].mean()
            ax.plot(m.index, m.values, marker="o", color=color, label=cls)
            ax.annotate(cls, (3.05, m.values[-1]), color=style.INK2, fontsize=8, va="center")
        ax.set_xlabel(f"GT perturbation (px), {'256 px' if res_label == '256' else 'native resolution'}")
        ax.set_xticks([1, 2, 3]); ax.set_xlim(0.8, 3.4); ax.set_ylim(0.5, 1.0)
    axes[0].set_ylabel("IoU")
    axes[0].legend(loc="lower left", ncol=2)
    fig.savefig(ROOT / "figures/paper/F7_label_sensitivity.png")
    print("saved F7")


if __name__ == "__main__":
    main()
