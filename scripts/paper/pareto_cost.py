#!/usr/bin/env python3
"""F6: accuracy vs cost for every three-seed model. Params and GFLOPs are measured by building each
model from its recorded config (hook-based conv/linear FLOP count at the model's own input size);
training wall time per seed = meta.json mtime -> history.json mtime where the run directory exists.
nnU-Net cost comes from its checkpoint (47.6M unique params, 87.4 GFLOPs @384x448, 8.4 h / seed).
Outputs: results/paper/model_cost.csv, figures/paper/F6_pareto.png.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from matblast.config import load_config
from matblast.models import build_model

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
import style  # noqa: E402

NNUNET = dict(params_M=47.64, gflops=87.4, train_hours=8.4)


def collect_specs():
    specs = {}
    for path in glob.glob(str(ROOT / "results" / "*_validation_no_tta.json")):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        for s in d.get("specs", []):
            if "=" in s and ":" in s:
                name, rest = s.split("=", 1); cfg, ck = rest.split(":", 1); specs.setdefault(name, (cfg, ck.split(",")))
    return specs


def flops(model, x):
    tot = [0]
    def hook(m, i, o):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            tot[0] += 2 * m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups) * m.out_channels * o.shape[-1] * o.shape[-2]
        elif isinstance(m, nn.Linear):
            tot[0] += 2 * m.in_features * m.out_features * (o.numel() // o.shape[-1])
        elif isinstance(m, nn.Conv1d):
            tot[0] += 2 * m.kernel_size[0] * (m.in_channels // m.groups) * m.out_channels * o.shape[-1] * o.shape[0]
    hs = [m.register_forward_hook(hook) for m in model.modules()]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return tot[0] / 1e9


def train_hours(ckpts):
    hours = []
    for ck in ckpts:
        d = ROOT / Path(ck).parent
        if (d / "meta.json").exists() and (d / "history.json").exists():
            hours.append((os.path.getmtime(d / "history.json") - os.path.getmtime(d / "meta.json")) / 3600)
    return float(np.mean(hours)) if hours else np.nan


def main():
    summ = pd.read_csv(ROOT / "results/paper/model_summary.csv")
    specs = collect_specs()
    rows = []
    for _, r in summ.iterrows():
        name = r.model
        if name.startswith("B9"):
            rows.append((name, r.label, r.family, NNUNET["params_M"], NNUNET["gflops"], NNUNET["train_hours"], 384)); continue
        if name not in specs:
            print("no spec:", name); continue
        cfg_path, ckpts = specs[name]
        cfg = load_config(ROOT / cfg_path)
        try:
            m = build_model(cfg).eval()
        except Exception as e:
            print("build failed", name, repr(e)[:80]); continue
        size = cfg["data"].get("image_size", 256); size = size if isinstance(size, int) else max(size)
        in_ch = int(cfg["data"].get("in_channels", cfg["model"].get("in_channels", 3)))
        params = sum(p.numel() for p in m.parameters()) / 1e6
        try:
            g = flops(m, torch.randn(1, in_ch, size, size))
        except Exception as e:
            g = np.nan; print("flops failed", name, repr(e)[:60])
        rows.append((name, r.label, r.family, params, g, train_hours(ckpts), size))
    cost = pd.DataFrame(rows, columns=["model", "label", "family", "params_M", "gflops", "train_hours", "input"]).merge(summ[["model", "miou_fg", "TE", "ICM"]], on="model")
    cost.to_csv(ROOT / "results/paper/model_cost.csv", index=False)
    print(cost.sort_values("miou_fg", ascending=False)[["label", "params_M", "gflops", "train_hours", "miou_fg"]].head(6).round(3).to_string(index=False))

    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, (col, xl) in zip(axes, (("params_M", "parameters (M)"), ("gflops", "GFLOPs per image (own input size)"), ("train_hours", "training wall time per seed (h)"))):
        for fam in style.FAMILY_COLORS:
            sub = cost[(cost.family == fam) & cost[col].notna()]
            if len(sub) == 0: continue
            ax.scatter(sub[col], sub.miou_fg, color=style.FAMILY_COLORS[fam], marker=style.FAMILY_MARKERS[fam], s=110 if fam == "nnU-Net" else 30, edgecolor=style.SURFACE, linewidth=0.5, label=fam, zorder=3)
        for name in ("S43", "B9_nnunet", "S24", "B0", "S37_residual_encoder", "V2_1_backbone_v2_grouped_v1"):
            r = cost[cost.model == name]
            if len(r) and pd.notna(r[col].values[0]):
                off = {"S37_residual_encoder": (5, -9), "S24": (5, -9), "V2_1_backbone_v2_grouped_v1": (5, 9)}.get(name, (5, 3))
                ax.annotate(r.label.values[0], (r[col].values[0], r.miou_fg.values[0]), textcoords="offset points", xytext=off, fontsize=7, color=style.INK2)
        ax.set_xscale("log"); ax.set_xlabel(xl); ax.grid(True, axis="both")
    axes[0].set_ylabel("mIoU-FG (val, 3 seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(ROOT / "figures/paper/F6_pareto.png"); print("saved F6")


if __name__ == "__main__":
    main()
