#!/usr/bin/env python3
"""F2 (agreement), F4 (error location vs GT boundary), F8 (qualitative + disagreement map).
Inputs: results/paper/preds/*.npy (argmax 256x256), gt.npy. Outputs: results/paper/agreement_matrix_{TE,ICM}.csv,
results/paper/error_distance_profile.csv, figures/paper/{F2_agreement,F4_error_location,F8_qualitative}.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
import style  # noqa: E402
_DN = style.display_name

P = ROOT / "results/paper/preds"
NAMES = {m: style.display_name(m) for m in ['B9_nnunet', 'S43', 'S24', 'S54_matblast_bmask_boundary_grouped_v1', 'S37_residual_encoder', 'L01_mednext_grouped_v1_metricfixed', 'S46_matblast_pointrend_faithful_grouped_v1', 'S49_matblast_refine_mask_grouped_v1', 'B12_BLASTO_Net', 'B3', 'B4', 'B0']}
CLS = {"ZP": 1, "TE": 2, "BL": 3, "ICM": 4}


def load(model, seed=0):
    return np.load(P / f"{model}_seed{seed}.npy")


def iou_per_image(a, b, k):
    A, B = a == k, b == k
    inter = (A & B).sum((1, 2)); union = (A | B).sum((1, 2))
    out = inter / np.maximum(union, 1); out[union == 0] = np.nan
    return out


def main():
    gt = np.load(P / "gt.npy"); ids = json.load(open(P / "image_ids.json"))
    models = [m for m in NAMES if (P / f"{m}_seed0.npy").exists()]
    preds = {m: np.mean([iou_per_image(load(m, s), gt, 2) for s in range(3)], 0) for m in models}  # unused placeholder
    style.apply()

    # ---------- F2: pairwise agreement (model-model and model-GT), seed 0 predictions, mean over images
    entities = ["GT"] + models
    stacks = {"GT": gt, **{m: load(m, 0) for m in models}}
    for cls in ("TE", "ICM"):
        k = CLS[cls]; n = len(entities); A = np.zeros((n, n))
        for i, a in enumerate(entities):
            for j, b in enumerate(entities):
                A[i, j] = np.nanmean(iou_per_image(stacks[a], stacks[b], k))
        pd.DataFrame(A, index=[_DN(e) for e in entities], columns=[_DN(e) for e in entities]).to_csv(ROOT / f"results/paper/agreement_matrix_{cls}.csv")
        if cls == "TE":
            ATE = A
    fig = plt.figure(figsize=(11, 4.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0]); cmap = LinearSegmentedColormap.from_list("seq", style.SEQ_BLUE)
    im = ax.imshow(ATE, cmap=cmap, vmin=0.7, vmax=1.0); ax.grid(False)
    lab = [_DN(e) for e in entities]
    ax.set_xticks(range(len(lab))); ax.set_yticks(range(len(lab))); ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=7); ax.set_yticklabels(lab, fontsize=7)
    for i in range(len(lab)):
        for j in range(len(lab)):
            if i != j: ax.text(j, i, f"{ATE[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color=style.INK if ATE[i, j] < 0.88 else style.SURFACE)
    ax.set_title("(a)", loc="left", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.outline.set_visible(False)
    # (b) per-image scatter: agreement with another model vs agreement with GT, for S43
    ax = fig.add_subplot(gs[1])
    s43 = stacks["S43"]
    x = iou_per_image(s43, gt, 2)
    for other, color, mk in (("L01_mednext_grouped_v1_metricfixed", style.AQUA, "s"), ("B9_nnunet", style.BLUE, "*"), ("B0", style.MUTED, "^")):
        if other not in stacks: continue
        yv = iou_per_image(s43, stacks[other], 2)
        ax.scatter(x, yv, s=22, color=color, marker=mk, alpha=0.85, edgecolor=style.SURFACE, linewidth=0.5, label=f"DiGA-Net+ vs {_DN(other)} (mean {np.nanmean(yv):.2f})")
    ax.plot([0.55, 1], [0.55, 1], color=style.INK2, linewidth=0.8, linestyle="--")
    ax.set_xlabel("TE IoU: DiGA-Net+ vs ground truth (per image)"); ax.set_ylabel("TE IoU: DiGA-Net+ vs another model")
    ax.set_xlim(0.6, 1); ax.set_ylim(0.6, 1); ax.set_title("(b)", loc="left", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False); ax.grid(True, axis="both")
    fig.savefig(ROOT / "figures/paper/F2_agreement.png"); print("saved F2")

    # ---------- F4: where do S43 (and nnU-Net) errors sit relative to the GT boundary of each class?
    rows = []
    for model in ("S43", "B9_nnunet"):
        pred = stacks[model]
        for cls, k in CLS.items():
            for i in range(len(ids)):
                g = gt[i] == k
                if g.sum() == 0: continue
                # signed distance to GT class boundary: negative inside the GT region
                dist_out = ndimage.distance_transform_edt(~g); dist_in = ndimage.distance_transform_edt(g)
                signed = np.where(g, -dist_in, dist_out)
                err = (pred[i] == k) != g
                d = signed[err]
                rows.append((model, cls, i, err.sum(), (np.abs(d) <= 1).sum(), (np.abs(d) <= 2).sum(), (np.abs(d) <= 3).sum(), (d < 0).sum()))
    er = pd.DataFrame(rows, columns=["model", "cls", "image", "n_err", "within1", "within2", "within3", "inside_gt"])
    er.to_csv(ROOT / "results/paper/error_distance_profile.csv", index=False)
    prof = er.groupby(["model", "cls"])[["n_err", "within1", "within2", "within3"]].sum()
    prof["f1"] = prof.within1 / prof.n_err; prof["f2"] = prof.within2 / prof.n_err; prof["f3"] = prof.within3 / prof.n_err
    print(prof[["f1", "f2", "f3"]].round(3).to_string())
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    classes = list(CLS); xs = np.arange(len(classes)); w = 0.36
    for off, model, color in ((-w / 2, "S43", style.ORANGE), (w / 2, "B9_nnunet", style.BLUE)):
        f1 = prof.loc[model].loc[classes, "f1"].values; f2 = prof.loc[model].loc[classes, "f2"].values - f1; f3 = prof.loc[model].loc[classes, "f3"].values - f1 - f2
        far = 1 - f1 - f2 - f3
        b = np.zeros(len(classes))
        for vals, alpha, tag in ((f1, 1.0, "≤1 px"), (f2, 0.7, "≤2 px"), (f3, 0.45, "≤3 px"), (far, 0.2, ">3 px")):
            ax.bar(xs + off, vals, bottom=b, width=w, color=color, alpha=alpha, edgecolor=style.SURFACE, linewidth=1.2, label=f"{_DN(model)} {tag}" if True else None); b += vals
    ax.set_xticks(xs); ax.set_xticklabels(classes); ax.set_ylabel("fraction of error pixels"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.savefig(ROOT / "figures/paper/F4_error_location.png"); print("saved F4")

    # ---------- F8: qualitative panel with disagreement map over all dumped models (seed 0)
    cols = np.array([[0, 0, 0], [74, 58, 167], [235, 104, 52], [27, 175, 122], [227, 73, 72]], np.uint8)  # BG, ZP, TE, BL, ICM
    all_preds = np.stack([stacks[m] for m in models])  # (M,50,H,W)
    disagreement = (all_preds != all_preds[0:1]).any(0) | (all_preds[:, None] != all_preds[None]).any((0, 1))  # any model differs
    frac_disagree = np.stack([(all_preds[:, i] != np.median(all_preds[:, i], 0)).mean(0) for i in range(len(ids))])
    te_iou_s43 = iou_per_image(stacks["S43"], gt, 2)
    pick = [int(np.nanargmax(te_iou_s43)), int(np.nanargmin(te_iou_s43))]  # easiest and hardest TE image
    fig, axes = plt.subplots(len(pick), 5, figsize=(11, 2.3 * len(pick)))
    for r, i in enumerate(pick):
        img = np.array(Image.open(ROOT / "dataset/source/images" / ids[i]).convert("L").resize((256, 256)))
        rgb = np.stack([img] * 3, -1)
        ov = lambda lab: (0.5 * rgb + 0.5 * cols[lab]).astype(np.uint8)
        panels = [rgb, ov(gt[i]), ov(stacks["S43"][i]), ov(stacks["B9_nnunet"][i])]
        titles = ["image", "ground truth", f"DiGA-Net+ (TE IoU {te_iou_s43[i]:.2f})", f"nnU-Net (TE IoU {iou_per_image(stacks['B9_nnunet'], gt, 2)[i]:.2f})", f"disagreement across {len(models)} models"]
        for c, pan in enumerate(panels):
            axes[r, c].imshow(pan); axes[r, c].axis("off")
        axes[r, 4].imshow(rgb); axes[r, 4].imshow(frac_disagree[i], cmap=LinearSegmentedColormap.from_list("d", [(1, 1, 1, 0), style.RED]), vmin=0, vmax=0.6); axes[r, 4].axis("off")
        for c, t in enumerate(titles):
            if r == 0: axes[r, c].set_title(t, fontsize=8, loc="left")
        axes[r, 0].text(-0.05, 0.5, ["easiest TE", "hardest TE"][r], transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8, color=style.INK2)
    fig.subplots_adjust(wspace=0.02, hspace=0.08)
    fig.savefig(ROOT / "figures/paper/F8_qualitative.png"); print("saved F8")


if __name__ == "__main__":
    main()
