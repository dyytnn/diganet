#!/usr/bin/env python3
"""S1: recipe ladder V2_0..V2_6 — train/val loss and val mIoU curves per seed (over-fitting evidence).
Inputs: runs/v2_queue/*/*/history.json. Output: figures/paper/S1_recipe_ladder.png, results/paper/recipe_ladder_summary.csv.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
import style  # noqa: E402

RUNGS = [("V2_0_recipe_parity_grouped_v1", "recipe control"), ("V2_2_nnunet_augmentation_grouped_v1", "+ nnU-Net augmentation"),
         ("V2_3_ema_grouped_v1", "+ EMA"), ("V2_1_backbone_v2_grouped_v1", "+ six-stage residual backbone"),
         ("V2_4_icm_oversample_grouped_v1", "+ ICM oversampling"), ("V2_5_per_image_overlap_grouped_v1", "+ per-image Dice"), ("V2_6_pad_384x448_grouped_v1", "+ 384×448 padding")]


def main():
    style.apply()
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2), sharex=False)
    rows = []
    for ax, (stem, title) in zip(axes.flat, RUNGS):
        for s, hist in enumerate(sorted(glob.glob(str(ROOT / f"runs/v2_queue/{stem}/*_seed*/history.json")))):
            h = json.load(open(hist)); ep = [e["epoch"] for e in h]
            ax.plot(ep, [e["train_loss"] for e in h], color=style.MUTED, linewidth=1, alpha=0.8, label="train loss" if s == 0 else None)
            ax.plot(ep, [e["val_loss"] for e in h], color=style.ORANGE, linewidth=1.2, alpha=0.9, label="val loss" if s == 0 else None)
            best = max(h, key=lambda e: e["val_metrics"]["miou_fg"])
            ax.axvline(best["epoch"], color=style.BLUE, linewidth=0.7, linestyle=":")
            rows.append((stem, s, len(h), best["epoch"], best["val_metrics"]["miou_fg"], min(e["val_loss"] for e in h), np.mean([e["val_loss"] for e in h[-10:]])))
        ax.text(0.98, 0.95, title, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=style.INK2); ax.set_ylim(0, 2.0); ax.set_xlabel("epoch"); ax.grid(True, axis="y")
    axes[0, 0].set_ylabel("loss"); axes[1, 0].set_ylabel("loss"); axes[0, 0].legend(fontsize=7, loc="center right")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, 0.9, "Dotted blue line: epoch of best val mIoU-FG.\nVal loss rises after the best epoch in every\nrung (over-fitting on 199 images); nnU-Net\naugmentation and the six-stage residual\nbackbone delay it most.", fontsize=8, va="top", color=style.INK2)
    fig.tight_layout(); fig.savefig(ROOT / "figures/paper/S1_recipe_ladder.png")
    pd.DataFrame(rows, columns=["rung", "seed", "epochs", "best_epoch", "best_miou_fg", "min_val_loss", "val_loss_last10"]).to_csv(ROOT / "results/paper/recipe_ladder_summary.csv", index=False)
    print("saved S1")


if __name__ == "__main__":
    main()
