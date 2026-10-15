#!/usr/bin/env python3
"""C2: boundary-tolerant metrics as an alternative primary metric for thin classes.
For each dumped model (seed-averaged over available seeds): Boundary-F at tolerance 1/2/3 px and
"trimmed IoU" (IoU computed after excluding a k-px band around the GT boundary, k=1,2,3) per class.
Inputs: results/paper/preds/*.npy. Outputs: results/paper/boundary_tolerant_metrics.csv, figures/paper/C2_boundary_tolerant.png.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper")); sys.path.insert(0, str(ROOT / "src"))
import style  # noqa: E402
_DN = style.display_name
from matblast.metrics import boundary_f  # noqa: E402

P = ROOT / "results/paper/preds"
CLS = {"ZP": 1, "TE": 2, "BL": 3, "ICM": 4}
NAMES = {m: style.display_name(m) for m in ['B9_nnunet', 'S43', 'S24', 'B0', 'B4', 'B3', 'B12_BLASTO_Net', 'S54_matblast_bmask_boundary_grouped_v1', 'L01_mednext_grouped_v1_metricfixed', 'S37_residual_encoder']}


def trimmed_iou(pred, gt, k, cls):
    g = gt == cls; p = pred == cls
    band = ndimage.binary_dilation(g, iterations=k) & ~ndimage.binary_erosion(g, iterations=k)
    keep = ~band
    inter = (g & p & keep).sum(); union = ((g | p) & keep).sum()
    return np.nan if union == 0 else inter / union


def main():
    gt = np.load(P / "gt.npy")
    rows = []
    for model in NAMES:
        files = sorted(glob.glob(str(P / f"{model}_seed*.npy")))
        if not files: continue
        for f in files:
            pred = np.load(f); seed = int(f[-5])
            for cls, k in CLS.items():
                for i in range(len(gt)):
                    if (gt[i] == k).sum() == 0: continue
                    rows.append((model, seed, cls, i, "iou", 0, trimmed_iou(pred[i], gt[i], 1, k) if False else (( (pred[i]==k)&(gt[i]==k)).sum()/max(((pred[i]==k)|(gt[i]==k)).sum(),1))))
                    for tol in (1, 2, 3):
                        rows.append((model, seed, cls, i, "boundary_f", tol, boundary_f(pred[i] == k, gt[i] == k, tolerance=tol)))
                        rows.append((model, seed, cls, i, "trimmed_iou", tol, trimmed_iou(pred[i], gt[i], tol, k)))
    df = pd.DataFrame(rows, columns=["model", "seed", "cls", "image", "metric", "tol", "value"])
    df.to_csv(ROOT / "results/paper/boundary_tolerant_metrics.csv", index=False)
    agg = df.groupby(["model", "cls", "metric", "tol"])["value"].mean().reset_index()
    print(agg[(agg.model.isin(["S43", "B9_nnunet"])) & (agg.cls == "TE")].pivot_table(index=["metric", "tol"], columns="model", values="value").round(3).to_string())

    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, cls in zip(axes, ("TE", "ICM")):
        for model, color, mk in (("S43", style.ORANGE, "o"), ("B9_nnunet", style.BLUE, "*"), ("B0", style.MUTED, "^"), ("B4", style.VIOLET, "D")):
            sub = agg[(agg.model == model) & (agg.cls == cls)]
            base = sub[sub.metric == "iou"].value.values[0]
            tr = sub[sub.metric == "trimmed_iou"].sort_values("tol")
            ax.plot([0] + tr.tol.tolist(), [base] + tr.value.tolist(), marker=mk, color=color, label=f"{_DN(model)} trimmed IoU")
            bf = sub[sub.metric == "boundary_f"].sort_values("tol")
            ax.plot(bf.tol, bf.value, marker=mk, color=color, linestyle="--", linewidth=1.2, alpha=0.8, label=f"{_DN(model)} Boundary-F")
        ax.set_title(f"({'a' if cls == 'TE' else 'b'}) {cls}", loc="left", fontsize=9)
        ax.set_xlabel("k (px)"); ax.set_xticks([0, 1, 2, 3]); ax.set_ylim(0.5, 1.0); ax.grid(True, axis="both")
    axes[0].set_ylabel("score"); handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(ROOT / "figures/paper/C2_boundary_tolerant.png"); print("saved C2")


if __name__ == "__main__":
    main()
