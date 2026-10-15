#!/usr/bin/env python3
"""F5 + D2 + D3: forest plot of the eleven refinement stages (Δ vs S24 and vs S43 with hierarchical
95% CI), ΔTE-vs-ΔICM scatter, multiple-comparison control over the whole program, and a power estimate.
Inputs: results/*paired*_no_tta.json. Outputs: results/paper/{stage_deltas.csv, multiple_comparisons.csv,
power.json}, figures/paper/F5_forest.png.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
import style  # noqa: E402

STAGES = [("S46", "S46_matblast_pointrend_faithful", "PointRend (faithful)", "faithful"),
          ("S47", "S47", "BPR (proxy)", "proxy"),
          ("S48", "S48_matblast_deep_strip", "DeepStrip (inspired)", "inspired"),
          ("S49", "S49_matblast_refine_mask", "RefineMask-lite", "lite"),
          ("S50", "S50_matblast_mask_transfiner", "Mask Transfiner-lite", "lite"),
          ("S51", "S51_matblast_sharp_contour", "SharpContour (polar)", "inspired"),
          ("S52", "S52_matblast_segfix", "SegFix (post-proc.)", "inspired"),
          ("S53", "S53_matblast_cascade_psp", "CascadePSP-lite", "lite"),
          ("S54", "S54_matblast_bmask_boundary", "BMask boundary branch", "inspired"),
          ("S55", "S55_matblast_deep_snake", "Deep Snake (polar)", "inspired"),
          ("S56", "S56_matblast_poly_transform", "PolyTransform-lite", "lite")]
MARK = {"faithful": "o", "proxy": "s", "inspired": "D", "lite": "^"}


def load_comparison(stage_key, ref):
    for f in glob.glob(str(ROOT / "results" / "*paired*_no_tta.json")):
        d = json.load(open(f))
        if d.get("reference") != ref:
            continue
        for name, c in d["comparisons"].items():
            if name.startswith(stage_key):
                return c
    return None


def main():
    rows = []
    for sid, key, label, kind in STAGES:
        for ref in ("S24", "S43"):
            c = load_comparison(key, ref)
            if c is None:
                print("missing", sid, ref); continue
            for metric, s in (("mIoU", c["macro_fg_iou"]), ("TE", c["per_class_iou"]["TE"]), ("ICM", c["per_class_iou"]["ICM"])):
                rows.append((sid, label, kind, ref, metric, s["delta"], *s["ci95_hierarchical"], s["p_image_wilcoxon_greater"]))
    df = pd.DataFrame(rows, columns=["stage", "label", "kind", "ref", "metric", "delta", "lo", "hi", "p"])
    df.to_csv(ROOT / "results/paper/stage_deltas.csv", index=False)

    # D2: multiple comparisons over every model paired against S24 in the program (image-level one-sided p, mIoU/TE/ICM)
    allp = []
    for f in glob.glob(str(ROOT / "results" / "*paired*_no_tta.json")):
        d = json.load(open(f))
        if d.get("reference") != "S24":
            continue
        for name, c in d["comparisons"].items():
            for metric, s in (("mIoU", c["macro_fg_iou"]), ("TE", c["per_class_iou"]["TE"]), ("ICM", c["per_class_iou"]["ICM"])):
                allp.append((name, metric, s["delta"], s["p_image_wilcoxon_greater"]))
    ap = pd.DataFrame(allp, columns=["model", "metric", "delta", "p"]).drop_duplicates(["model", "metric"])
    for metric in ("mIoU", "TE", "ICM"):
        sub = ap[ap.metric == metric].copy()
        sub["holm"] = multipletests(sub.p, method="holm")[1]; sub["bh"] = multipletests(sub.p, method="fdr_bh")[1]
        ap.loc[sub.index, "p_holm"] = sub.holm; ap.loc[sub.index, "p_bh"] = sub.bh
    ap.to_csv(ROOT / "results/paper/multiple_comparisons.csv", index=False)
    summary = {m: dict(n=int((ap.metric == m).sum()), raw_sig=int(((ap.metric == m) & (ap.p < .05)).sum()),
                       holm_sig=int(((ap.metric == m) & (ap.p_holm < .05)).sum()), bh_sig=int(((ap.metric == m) & (ap.p_bh < .05)).sum()),
                       holm_sig_positive=int(((ap.metric == m) & (ap.p_holm < .05) & (ap.delta > 0)).sum()))
               for m in ("mIoU", "TE", "ICM")}
    print("multiple comparisons vs S24:", json.dumps(summary))

    # D3: power — detectable paired Δ at 80% power, alpha .05 one-sided, n=50 images, using observed SD of per-image paired deltas (S43 - S24)
    per = pd.read_csv(ROOT / "results/paper/per_image_iou.csv")
    avg = per.groupby(["model", "image", "cls"])["iou"].mean().unstack("cls")
    power = {}
    for cls in ("TE", "ICM"):
        d = (avg.loc["S43"][cls] - avg.loc["S24"][cls]).dropna()
        sd = d.std(ddof=1); n = len(d)
        mde = (norm.ppf(0.95) + norm.ppf(0.80)) * sd / np.sqrt(n)
        power[cls] = dict(n=int(n), sd_paired=float(sd), mde_80pct=float(mde), observed_S43_minus_S24=float(d.mean()))
    json.dump(power, open(ROOT / "results/paper/power.json", "w"), indent=2); print("power:", json.dumps(power))

    # ---- F5
    style.apply()
    fig = plt.figure(figsize=(11, 4.6)); gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.1], wspace=0.35)
    order = [s[0] for s in STAGES]
    y = np.arange(len(order))[::-1]
    for j, metric in enumerate(("mIoU", "TE", "ICM")):
        ax = fig.add_subplot(gs[j])
        for ref, color, dy in (("S24", style.ORANGE, 0.18), ("S43", style.MUTED, -0.18)):
            sub = df[(df.metric == metric) & (df.ref == ref)].set_index("stage").loc[order]
            ax.errorbar(sub.delta, y + dy, xerr=[sub.delta - sub.lo, sub.hi - sub.delta], fmt="none", ecolor=color, elinewidth=1.4, capsize=0)
            for yi, (sid, r) in zip(y + dy, sub.iterrows()):
                ax.plot(r.delta, yi, marker=MARK[r.kind], color=color, markersize=5, markeredgecolor=style.SURFACE, markeredgewidth=0.6)
        ax.axvline(0, color=style.INK2, linewidth=0.8)
        ax.axvspan(0, ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 0.03, color=style.GREEN, alpha=0.06, linewidth=0)
        ax.set_yticks(y); ax.set_yticklabels([s[2] for s in STAGES] if j == 0 else [""] * len(order))
        ax.set_xlabel(f"Δ {'mIoU-FG' if metric == 'mIoU' else metric + ' IoU'}"); ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="x"); ax.grid(False, axis="y")
    ax = fig.add_subplot(gs[3])
    sub = df[df.ref == "S24"].pivot(index="stage", columns="metric", values="delta").loc[order]
    kinds = dict(zip([s[0] for s in STAGES], [s[3] for s in STAGES])); labels = dict(zip([s[0] for s in STAGES], [s[2] for s in STAGES]))
    for sid, r in sub.iterrows():
        ax.plot(r.TE, r.ICM, marker=MARK[kinds[sid]], color=style.RED, markersize=7, markeredgecolor=style.SURFACE)
        ax.annotate(labels[sid].replace(" (faithful)", "").replace(" (polar)", "").replace(" (post-proc.)", "").replace(" (inspired)", "").replace(" (proxy)", ""), (r.TE, r.ICM), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color=style.INK2)
    ax.axhline(0, color=style.INK2, linewidth=0.8); ax.axvline(0, color=style.INK2, linewidth=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4)); ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f')); ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("Δ TE IoU vs DiGA-Net"); ax.set_ylabel("Δ ICM IoU vs DiGA-Net"); ax.grid(True, axis="both")
    handles = [plt.Line2D([], [], color=style.ORANGE, lw=2, label="vs DiGA-Net"), plt.Line2D([], [], color=style.MUTED, lw=2, label="vs DiGA-Net+ (best DiGA-Net+)")]
    handles += [plt.Line2D([], [], color=style.INK2, marker=m, lw=0, label=k) for k, m in MARK.items()]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.02), frameon=False, title="Δ = candidate − reference; error bars: 95% hierarchical bootstrap CI over seeds × images", title_fontsize=7.5)
    fig.savefig(ROOT / "figures/paper/F5_forest.png"); print("saved F5")


if __name__ == "__main__":
    main()
