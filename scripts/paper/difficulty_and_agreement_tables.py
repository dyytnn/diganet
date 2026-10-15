#!/usr/bin/env python3
"""F3 + S6: per-image difficulty heatmap, oracle/ensemble curves, variance decomposition,
rank-correlation matrix. Input: results/paper/per_image_iou.csv.
Outputs: results/paper/{oracle_curve.csv, variance_decomposition.csv, rank_correlation_TE.csv},
figures/paper/F3_difficulty_oracle.png, figures/paper/S6_rank_correlation.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
import style  # noqa: E402

RNG = np.random.default_rng(0)


def seed_avg(df):
    return df.groupby(["model", "label", "family", "image", "cls"])["iou"].mean().reset_index()


def matrix(avg, cls):
    sub = avg[avg.cls == cls].pivot(index="model", columns="image", values="iou")
    return sub


def variance_decomposition(M, n_boot=1000):
    """Two-way (image x model) decomposition of variance; returns fractions and bootstrap CIs."""
    X = M.values
    X = X[:, ~np.isnan(X).any(0)]
    def decomp(Xb):
        gm = Xb.mean(); img = Xb.mean(0) - gm; mod = Xb.mean(1) - gm
        resid = Xb - Xb.mean(0, keepdims=True) - Xb.mean(1, keepdims=True) + gm
        tot = ((Xb - gm) ** 2).sum()
        return np.array([(img ** 2).sum() * Xb.shape[0], (mod ** 2).sum() * Xb.shape[1], (resid ** 2).sum()]) / tot
    point = decomp(X)
    boots = np.stack([decomp(X[RNG.integers(0, X.shape[0], X.shape[0])][:, RNG.integers(0, X.shape[1], X.shape[1])]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return point, lo, hi


def oracle_curves(M, n_rep=200):
    """Best-of-N (oracle) and mean-of-N (naive ensemble proxy) as a function of N."""
    X = M.values; models = X.shape[0]
    Ns = [1, 2, 3, 5, 8, 12, 20, 30, 45, models]
    rows = []
    for N in Ns:
        o, e = [], []
        for _ in range(n_rep if N < models else 1):
            idx = RNG.choice(models, N, replace=False)
            o.append(np.nanmean(np.nanmax(X[idx], 0)))
            e.append(np.nanmean(np.nanmean(X[idx], 0)))
        rows.append((N, np.mean(o), np.percentile(o, 5), np.percentile(o, 95), np.mean(e)))
    return pd.DataFrame(rows, columns=["N", "oracle", "oracle_p5", "oracle_p95", "mean_of_models"])


def main():
    df = pd.read_csv(ROOT / "results/paper/per_image_iou.csv")
    avg = seed_avg(df)
    fam = avg.drop_duplicates("model").set_index("model")["family"]
    label = avg.drop_duplicates("model").set_index("model")["label"]
    out_rows, var_rows = [], []
    for cls in ["ZP", "TE", "BL", "ICM"]:
        M = matrix(avg, cls)
        oc = oracle_curves(M); oc["cls"] = cls; out_rows.append(oc)
        p, lo, hi = variance_decomposition(M)
        for name, v, l, h in zip(["image", "model", "residual"], p, lo, hi):
            var_rows.append((cls, name, v, l, h))
    oracle = pd.concat(out_rows); oracle.to_csv(ROOT / "results/paper/oracle_curve.csv", index=False)
    var = pd.DataFrame(var_rows, columns=["cls", "component", "fraction", "ci_lo", "ci_hi"]); var.to_csv(ROOT / "results/paper/variance_decomposition.csv", index=False)
    print(var.pivot(index="cls", columns="component", values="fraction").round(3).to_string())
    print(oracle[oracle.cls == "TE"][["N", "oracle", "mean_of_models"]].round(4).to_string(index=False))

    # ---- F3 figure: (a) heatmap TE, (b) oracle curves TE & ICM, (c) variance decomposition
    style.apply()
    M = matrix(avg, "TE")
    img_order = M.mean(0).sort_values().index
    model_order = M.mean(1).sort_values(ascending=False).index
    H = M.loc[model_order, img_order].values
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1.3, 1.0], wspace=0.45, left=0.12)
    ax = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("seq", style.SEQ_BLUE)
    im = ax.imshow(H, aspect="auto", cmap=cmap, vmin=0.6, vmax=0.9, interpolation="nearest")
    ax.set_title("(a)", loc="left")
    ax.set_xlabel("images, sorted by mean difficulty →"); ax.set_ylabel("models, sorted by mean TE IoU →")
    ax.set_xticks([]); ax.set_yticks([])
    for name, color in (("B9_nnunet", style.BLUE), ("S43", style.ORANGE), ("B0", style.MUTED)):
        if name in model_order:
            y = list(model_order).index(name); ax.annotate(label[name], (-1.0, y), color=color, fontsize=8, va="center", ha="right", annotation_clip=False)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.12); cb.ax.set_title("TE IoU", fontsize=8, color=style.INK2, pad=4); cb.outline.set_visible(False)
    ax = fig.add_subplot(gs[1])
    for cls, color in (("TE", style.ORANGE), ("ICM", style.RED)):
        oc = oracle[oracle.cls == cls]
        ax.fill_between(oc.N, oc.oracle_p5, oc.oracle_p95, color=color, alpha=0.15, linewidth=0)
        ax.plot(oc.N, oc.oracle, color=color, marker="o", markersize=4, label=f"{cls} oracle (best-of-N)")
        ax.plot(oc.N, oc.mean_of_models, color=color, linestyle="--", linewidth=1.2, label=f"{cls} mean-of-N")
    sens = pd.read_csv(ROOT / "results/paper/label_sensitivity.csv")
    for cls, color in (("TE", style.ORANGE), ("ICM", style.RED)):
        band = sens[(sens.resolution == "256") & (sens.cls == cls) & (sens.k_px == 1)]["iou"].mean()
        ax.axhline(band, color=color, linewidth=0.8, linestyle=":")
        ax.annotate(f"{cls} 1-px label band", (65, band), color=color, fontsize=7, va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_xlabel("number of models N"); ax.set_ylabel("IoU (best-of-N per image)"); ax.set_title("(b)", loc="left")
    ax.set_xticks([1, 3, 10, 30, 65]); ax.set_xticklabels(["1", "3", "10", "30", "65"]); ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    ax = fig.add_subplot(gs[2])
    comp_colors = {"image": style.BLUE, "model": style.ORANGE, "residual": style.MUTED}
    classes = ["ZP", "TE", "BL", "ICM"]; bottom = np.zeros(4)
    for comp in ["image", "model", "residual"]:
        vals = var.set_index(["cls", "component"]).loc[[(c, comp) for c in classes], "fraction"].values
        ax.bar(classes, vals, bottom=bottom, color=comp_colors[comp], width=0.6, label=comp, edgecolor=style.SURFACE, linewidth=1.5)
        bottom += vals
    ax.set_ylim(0, 1); ax.set_ylabel("fraction of IoU variance"); ax.set_title("(c)", loc="left"); ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    fig.savefig(ROOT / "figures/paper/F3_difficulty_oracle.png")
    print("saved F3")

    # ---- S6: rank correlation matrix (TE), models ordered by family then score
    M = matrix(avg, "TE")
    order = sorted(M.index, key=lambda m: (["nnU-Net", "DiGA-Net family", "Boundary stage", "Literature block", "Recipe rung", "Blastocyst paper", "Generic baseline"].index(fam[m]), -M.loc[m].mean()))
    X = M.loc[order].values
    R = np.array([[spearmanr(X[i], X[j], nan_policy="omit")[0] for j in range(len(order))] for i in range(len(order))])
    pd.DataFrame(R, index=[label[m] for m in order], columns=[label[m] for m in order]).to_csv(ROOT / "results/paper/rank_correlation_TE.csv")
    fig, ax = plt.subplots(figsize=(7.5, 7))
    im = ax.imshow(R, cmap=cmap, vmin=0.5, vmax=1.0); ax.grid(False)
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels([label[m] for m in order], rotation=90, fontsize=5); ax.set_yticklabels([label[m] for m in order], fontsize=5)
    for i, m in enumerate(order):
        ax.get_yticklabels()[i].set_color(style.FAMILY_COLORS[fam[m]]); ax.get_xticklabels()[i].set_color(style.FAMILY_COLORS[fam[m]])
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.outline.set_visible(False)
    fig.savefig(ROOT / "figures/paper/S6_rank_correlation.png")
    print("saved S6; median off-diagonal rho =", round(float(np.median(R[np.triu_indices(len(order), 1)])), 3))


if __name__ == "__main__":
    main()
