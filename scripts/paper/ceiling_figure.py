#!/usr/bin/env python3
"""F1: every number on this dataset sits on one line. Inputs: results/paper/model_summary.csv,
results/paper/literature_numbers.csv, results/paper/label_sensitivity.csv, results/paper/multiple_comparisons.csv.
Output: figures/paper/F1_ceiling.png, results/paper/f1_points.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
import style  # noqa: E402

FAMILY_ORDER = ["nnU-Net", "DiGA-Net family", "Boundary stage", "Literature block", "Recipe rung", "Blastocyst paper", "Generic baseline"]


def main():
    summ = pd.read_csv(ROOT / "results/paper/model_summary.csv")
    lit = pd.read_csv(ROOT / "results/paper/literature_numbers.csv")
    sens = pd.read_csv(ROOT / "results/paper/label_sensitivity.csv")
    mc = pd.read_csv(ROOT / "results/paper/multiple_comparisons.csv")
    sig = mc[(mc.metric == "mIoU") & (mc.p_holm < 0.05) & (mc.delta > 0)].model.tolist()
    print("Holm-significant mIoU improvements over S24:", sig)
    # ceiling band from 1-2 px label perturbation of FG classes @256 (mean over ZP,TE,BL,ICM)
    band = sens[(sens.resolution == "256") & (sens.cls.isin(["ZP", "TE", "BL", "ICM"]))].groupby("k_px")["iou"].mean()
    bg = summ["BG"].mean()
    band5 = ((4 * band[3] + bg) / 5, (4 * band[1] + bg) / 5)
    style.apply()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    rng = np.random.default_rng(1)
    rows = []
    for fi, fam in enumerate(FAMILY_ORDER):
        sub = summ[summ.family == fam]
        y = fi + rng.uniform(-0.18, 0.18, len(sub))
        ax.scatter(sub.miou_5cls * 100, y, color=style.FAMILY_COLORS[fam], marker=style.FAMILY_MARKERS[fam], s=36 if fam != "nnU-Net" else 140,
                   edgecolor=style.SURFACE, linewidth=0.6, zorder=3, label=f"{fam} (n={len(sub)})")
        for (_, r), yy in zip(sub.iterrows(), y):
            rows.append((r.model, fam, r.miou_5cls * 100, r.miou_fg * 100, "ours"))
            if r.model in ("S43", "B9_nnunet", "S24", "B0", "S54_matblast_bmask_boundary_grouped_v1"):
                off = {"S24": (-6, -13), "S43": (6, 7)}.get(r.model, (0, 7))
                ha = {"S24": "right", "S43": "left"}.get(r.model, "center")
                ax.annotate(r.label, (r.miou_5cls * 100, yy), textcoords="offset points", xytext=off, ha=ha, fontsize=7, color=style.INK2)
    yl = len(FAMILY_ORDER)
    lit5 = lit[lit.metric == "jaccard_5cls"]
    offsets = {"I2C2Net": (-28, 9), "BTFormer": (28, -14), "MASS-Net": (0, 9)}
    for _, r in lit5.iterrows():
        variant = "235-image" in str(r.split)
        ax.scatter(r.value, yl, facecolor="none" if variant else style.VIOLET, color=style.VIOLET, marker="v", s=60, linewidth=1.2, edgecolor=style.VIOLET if variant else style.SURFACE, zorder=3)
        text = f"{r.paper} ({r.venue_year.split()[-1]})" + ("*" if variant else "")
        ax.annotate(text, (r.value, yl), textcoords="offset points", xytext=offsets.get(r.paper, (0, 9)), ha="center", fontsize=7, color=style.INK2)
        rows.append((r.paper, "published", r.value, np.nan, r.split))
    ax.annotate("* 235-image dataset variant (200/35 split),\n  not directly comparable", (0.02, 0.04), xycoords="axes fraction", ha="left", va="bottom", fontsize=7, color=style.MUTED)
    for k, alpha in ((1, 0.9), (2, 0.7), (3, 0.5)):
        fg = band[k]; x5 = (4 * fg + bg) / 5 * 100
        ax.axvline(x5, color=style.ORANGE, linewidth=1.2, linestyle="--", alpha=alpha, zorder=1)
        ax.annotate(f"GT vs GT shifted by {k} px", (x5, -0.45), fontsize=7, color=style.ORANGE, ha="center", va="bottom", rotation=0)
    ax.annotate("dashed: IoU of GT vs GT moved by k px\n(mean of erode / dilate / shift)", (0.99, 0.03), xycoords="axes fraction", ha="right", va="bottom", fontsize=7.5, color=style.INK2)
    ax.set_yticks(list(range(yl + 1))); ax.set_yticklabels(FAMILY_ORDER + ["Published numbers\n(image-level splits)"], fontsize=8)
    ax.set_ylim(-0.9, yl + 0.8); ax.invert_yaxis()
    ax.set_xlabel("mean Jaccard over 5 classes incl. background (%) — the convention used by published work"); ax.grid(True, axis="x"); ax.grid(False, axis="y")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    fig.savefig(ROOT / "figures/paper/F1_ceiling.png"); pd.DataFrame(rows, columns=["name", "family", "jaccard_5cls", "miou_fg", "source"]).to_csv(ROOT / "results/paper/f1_points.csv", index=False)
    print("saved F1 | band 5-cls:", [round(b * 100, 2) for b in band5], "| x-range ours:", round(summ.miou_5cls.min() * 100, 2), round(summ.miou_5cls.max() * 100, 2))


if __name__ == "__main__":
    main()
