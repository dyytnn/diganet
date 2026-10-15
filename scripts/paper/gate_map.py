#!/usr/bin/env python3
"""F10: what the uncertainty gate of DiGA-Net+ actually does.

The gated refinement block at the bottleneck computes u = sigmoid(conv1x1(x)) on the 32x32 map and adds
tanh(s) * (u * r(x)) to the features. This script records u for the 50 validation images and 3 seeds and asks
where the gate opens:
  (a) per-image panels: image + annotation contours, the seed-averaged gate upsampled to 256^2, and the gate
      overlaid with the pixels DiGA-Net+ gets wrong (seed-majority prediction != annotation);
  (b) mean gate value by anatomical region over all 50 images x 3 seeds: within 3 px of an annotated class
      boundary, TE interior, ICM interior, BL interior, ZP interior, background; error bars = across images;
  (c) the gate value against the fraction of stored models that disagree with the majority at each pixel
      (binned), i.e. does the gate open where the *models* are uncertain.
Outputs figures/paper/F10_gate_map.png and results/paper/gate_map_summary.csv.

    PYTHONPATH=src python scripts/paper/gate_map.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
from matblast.models import build_model  # noqa: E402

CFG = ROOT / "configs/S43_matblast_point_rend_grouped_v1.yaml"
CKPT = "runs/matblast_shortlist_queue/S43_matblast_point_rend_grouped_v1/S43_matblast_point_rend_grouped_v1_seed{s}/best.pt"
PREDS = ROOT / "results/paper/preds"
CLASS_NAMES = {1: "ZP", 2: "TE", 3: "BL", 4: "ICM"}
CLASS_COLOR = {1: style.YELLOW, 2: style.ORANGE, 3: style.AQUA, 4: style.RED}


def find_ckpt(s: int) -> Path:
    p = ROOT / CKPT.format(s=s)
    if p.exists():
        return p
    hits = list(ROOT.glob(f"runs/**/{Path(CKPT.format(s=s)).parent.name}/best.pt"))
    if not hits:
        raise FileNotFoundError(CKPT.format(s=s))
    return hits[0]


def legacy(x: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(x.astype(np.float32))[None]
    z = (t - t.mean()) / (t.std() + 1e-6)
    return torch.cat((t, z, t), 0)


def boundary_band(gt: np.ndarray, px: int = 3) -> np.ndarray:
    edge = np.zeros(gt.shape, bool)
    for k in range(1, 5):
        m = gt == k
        if m.any():
            edge |= m ^ ndimage.binary_erosion(m)
    return ndimage.binary_dilation(edge, iterations=px)


@torch.no_grad()
def record_gates(ids: list[str], base: list[np.ndarray], device) -> np.ndarray:
    cfg = yaml.safe_load(CFG.read_text())
    gates = []
    for s in range(3):
        model = build_model(cfg)
        state = torch.load(find_ckpt(s), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        model.to(device).eval()
        store = {}
        h = model.literature_block.uncertainty.register_forward_hook(lambda m, i, o: store.__setitem__("u", torch.sigmoid(o)))
        out = []
        xs = torch.stack([legacy(b) for b in base])
        for i in range(0, len(xs), 10):
            model(xs[i:i + 10].to(device))
            out.append(F.interpolate(store["u"], size=(256, 256), mode="bilinear", align_corners=False)[:, 0].cpu())
        h.remove()
        gates.append(torch.cat(out).numpy())
        print("seed", s, "gate mean", float(np.mean(gates[-1])), flush=True)
    return np.stack(gates)  # (3, 50, 256, 256)


def main() -> None:
    style.apply()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = json.load(open(PREDS / "image_ids.json"))
    gt = np.load(PREDS / "gt.npy").astype(np.int64)
    base = [np.array(Image.open(ROOT / "dataset/source/images" / n).convert("L").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0 for n in ids]
    gates = record_gates(ids, base, device)
    g = gates.mean(0)  # seed-averaged (50, 256, 256)

    # DiGA-Net+ majority prediction and error mask; disagreement across all stored models
    s43 = np.stack([np.load(PREDS / f"S43_seed{k}.npy") for k in range(3)]).astype(np.int64)
    maj = np.apply_along_axis(lambda v: np.bincount(v, minlength=5).argmax(), 0, s43)
    err = maj != gt
    models = sorted({p.name.rsplit("_seed", 1)[0] for p in PREDS.glob("*_seed0.npy")})
    stack = np.stack([np.load(PREDS / f"{m}_seed0.npy") for m in models]).astype(np.int64)
    allmaj = np.apply_along_axis(lambda v: np.bincount(v, minlength=5).argmax(), 0, stack)
    disagree = (stack != allmaj[None]).mean(0)  # fraction of models disagreeing, per pixel

    # (b) region statistics
    rows = []
    for i in range(len(ids)):
        band = boundary_band(gt[i])
        regions = {"boundary band (3 px)": band}
        for k, name in CLASS_NAMES.items():
            regions[f"{name} interior"] = (gt[i] == k) & ~band
        regions["background"] = (gt[i] == 0) & ~band
        for name, m in regions.items():
            if m.any():
                rows.append({"image": ids[i], "region": name, "gate": float(g[i][m].mean())})
    reg = pd.DataFrame(rows)
    reg.to_csv(ROOT / "results/paper/gate_map_summary.csv", index=False)

    # (c) gate vs disagreement bins
    bins = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0001])
    d = disagree.ravel(); gv = g.ravel()
    idx = np.digitize(d, bins) - 1
    curve = [(bins[b], bins[b + 1], gv[idx == b].mean(), gv[idx == b].std(), (idx == b).mean()) for b in range(len(bins) - 1) if (idx == b).any()]

    # ---- figure ----
    te = np.array([np.nan if not ((maj[i] == 2) | (gt[i] == 2)).any() else ((maj[i] == 2) & (gt[i] == 2)).sum() / ((maj[i] == 2) | (gt[i] == 2)).sum() for i in range(len(ids))])
    picks = [int(np.nanargmax(te)), int(np.argsort(te)[len(ids) // 2]), int(np.nanargmin(te))]
    labels = ["easiest TE", "median TE", "hardest TE"]
    fig = plt.figure(figsize=(12.5, 8.2))
    gs = fig.add_gridspec(len(picks) + 1, 4, height_ratios=[1, 1, 1, 0.95], hspace=0.12, wspace=0.05)
    heat = plt.get_cmap("magma")
    for r, i in enumerate(picks):
        rgb = np.stack([base[i]] * 3, -1)
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(rgb, vmin=0, vmax=1)
        for k in CLASS_NAMES:
            ax.contour(gt[i] == k, levels=[0.5], colors=[CLASS_COLOR[k]], linewidths=0.8)
        ax.axis("off"); ax.text(-0.04, 0.5, labels[r], transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=8, color=style.INK2)
        if r == 0: ax.set_title("(a) image + annotation contours", fontsize=8, loc="left", color=style.INK2)
        ax = fig.add_subplot(gs[r, 1]); im = ax.imshow(g[i], cmap=heat, vmin=0, vmax=1); ax.axis("off")
        if r == 0: ax.set_title("gate $u$ (seed mean, 32$^2$ upsampled)", fontsize=8, loc="left", color=style.INK2)
        ax = fig.add_subplot(gs[r, 2]); ax.imshow(rgb, vmin=0, vmax=1); ax.imshow(g[i], cmap=heat, alpha=0.55, vmin=0, vmax=1)
        for k in CLASS_NAMES:
            ax.contour(gt[i] == k, levels=[0.5], colors=["white"], linewidths=0.5, alpha=0.8)
        ax.axis("off")
        if r == 0: ax.set_title("gate over image", fontsize=8, loc="left", color=style.INK2)
        ax = fig.add_subplot(gs[r, 3]); ax.imshow(rgb, vmin=0, vmax=1)
        e = np.zeros((256, 256, 4)); e[err[i]] = (*plt.matplotlib.colors.to_rgb(style.RED), 0.9); ax.imshow(e)
        ax.contour(g[i], levels=[0.5], colors=[style.YELLOW], linewidths=0.8)
        ax.axis("off")
        if r == 0: ax.set_title("DiGA-Net+ errors (red) and gate $>0.5$ (yellow)", fontsize=8, loc="left", color=style.INK2)
    cax = fig.add_axes([0.905, 0.42, 0.012, 0.36]); fig.colorbar(im, cax=cax).set_label("gate value $u$", fontsize=8)
    # (b) bars
    order = ["boundary band (3 px)", "TE interior", "ICM interior", "ZP interior", "BL interior", "background"]
    m = reg.groupby("region").gate.agg(["mean", "std"]).loc[order]
    axb = fig.add_subplot(gs[3, :2])
    axb.bar(range(len(order)), m["mean"], yerr=m["std"], color=[style.INK2] + [CLASS_COLOR[2], CLASS_COLOR[4], CLASS_COLOR[1], CLASS_COLOR[3], style.MUTED], capsize=2, width=0.7)
    axb.set_xticks(range(len(order))); axb.set_xticklabels(order, rotation=15, fontsize=7.5); axb.set_ylabel("mean gate value $u$"); axb.set_ylim(0, 1); axb.grid(axis="y")
    axb.text(0.01, 0.95, "(b) by annotated region, 50 images x 3 seeds", transform=axb.transAxes, fontsize=8, va="top", color=style.INK2)
    # (c) gate vs disagreement
    axc = fig.add_subplot(gs[3, 2:])
    pos = axc.get_position(); axc.set_position([pos.x0 + 0.05, pos.y0, pos.width - 0.05, pos.height])
    xs = [(lo + hi) / 2 for lo, hi, *_ in curve]; ys = [mu for *_, mu, sd, w in curve]; sds = [sd for *_, mu, sd, w in curve]
    axc.errorbar(xs, ys, yerr=sds, marker="o", color=style.ORANGE, lw=1.6, capsize=2)
    for (lo, hi, mu, sd, w), x in zip(curve, xs):
        axc.annotate(f"{100 * w:.0f}% px", (x, mu), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=6.5, color=style.MUTED)
    axc.set_xlabel(f"fraction of the {len(models)} stored models disagreeing with the majority (binned)"); axc.set_ylabel("mean gate value $u$"); axc.set_ylim(0, 1); axc.grid(True)
    axc.text(0.01, 0.95, "(c) gate against model disagreement", transform=axc.transAxes, fontsize=8, va="top", color=style.INK2)
    fig.savefig(ROOT / "figures/paper/F10_gate_map.png", bbox_inches="tight")
    print(m.round(3)); print(pd.DataFrame(curve, columns=["lo", "hi", "gate", "sd", "px_frac"]).round(3))


if __name__ == "__main__":
    main()
