#!/usr/bin/env python3
"""F9: per-class zoom-in panel on two well-segmented validation images (raw intensities).

Pick the two images with the highest S43 mean IoU over ZP/TE/ICM (seed-averaged, from
results/paper/per_image_iou.csv); for each class locate the largest connected
S43-vs-GT error component of that class, and show a square crop around it: image, crop,
GT contour, and the predictions of S43, nnU-Net, U-Net and MedNeXt (filled, with the GT
contour overlaid), plus the fraction of the twelve dumped models that disagree with
their majority vote. Predictions come from results/paper/preds (seed 0, 256x256).
Output: figures/paper/F9_zoom_hard_cases.png
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

PREDS = ROOT / "results" / "paper" / "preds"
OUT = ROOT / "figures" / "paper" / "F9_zoom_per_class.png"
CLASSES = {"ZP": 1, "TE": 2, "ICM": 4}
SHOW = [("S43", "DiGA-Net+"), ("B9_nnunet", "nnU-Net"), ("B0", "U-Net"),
        ("L01_mednext_grouped_v1_metricfixed", "base + MedNeXt")]
CROP = 96


def load(model: str) -> np.ndarray:
    return np.load(PREDS / f"{model}_seed0.npy")


def crop_box(mask_err: np.ndarray, mask_gt: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left corner of a size x size crop centred on the largest error component."""
    lab, n = ndimage.label(mask_err)
    if n == 0:
        cy, cx = ndimage.center_of_mass(mask_gt)
    else:
        sizes = ndimage.sum(mask_err, lab, range(1, n + 1))
        cy, cx = ndimage.center_of_mass(mask_err, lab, int(np.argmax(sizes)) + 1)
    h, w = mask_err.shape
    y0 = int(np.clip(round(cy - size / 2), 0, h - size))
    x0 = int(np.clip(round(cx - size / 2), 0, w - size))
    return y0, x0


def enhance(img: np.ndarray) -> np.ndarray:
    """Display-only contrast: CLAHE on the 8-bit image, then a 1-99 percentile stretch."""
    try:
        from skimage import exposure

        out = exposure.equalize_adapthist(img, clip_limit=0.015, kernel_size=32)
    except ImportError:  # pragma: no cover
        out = img.astype(np.float32) / 255.0
    lo, hi = np.percentile(out, [1, 99])
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)

CLASS_FILL = {1: style.YELLOW, 2: style.ORANGE, 3: style.AQUA, 4: style.RED}


def overlay(ax, img: np.ndarray, label: np.ndarray, alpha: float = 0.45) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    for k, col in CLASS_FILL.items():
        m = label == k
        if m.any():
            ax.imshow(np.ma.masked_where(~m, m), cmap=plt.matplotlib.colors.ListedColormap([col]), alpha=alpha, vmin=0, vmax=1)


def overall_row(axs, img, g, models, dis, miou) -> None:
    """Whole-image row: image, GT overlay, GT contours, model overlays with mIoU-FG, disagreement."""
    axs[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axs[0].set_ylabel(f"all classes\nDiGA-Net+ mIoU {miou:.2f}", fontsize=8, color=style.INK2)
    overlay(axs[1], img, g)
    axs[2].imshow(img, cmap="gray", vmin=0, vmax=255)
    for k, col in CLASS_FILL.items():
        if (g == k).any():
            axs[2].contour(g == k, levels=[0.5], colors=[col], linewidths=1.0)
    for c, (m, _label) in enumerate(SHOW, start=3):
        pm = models[m]
        pred = pm[overall_row.index]
        overlay(axs[c], img, pred)
        ious = []
        for k in (1, 2, 3, 4):
            gk, pk = g == k, pred == k
            if gk.any() or pk.any():
                ious.append(np.logical_and(gk, pk).sum() / np.logical_or(gk, pk).sum())
        axs[c].text(0.03, 0.03, f"mIoU {np.mean(ious):.2f}", transform=axs[c].transAxes, fontsize=7, color=style.INK,
                    bbox=dict(facecolor=style.SURFACE, edgecolor="none", alpha=0.8, pad=1.2), va="bottom")
    axs[-1].imshow(img, cmap="gray", vmin=0, vmax=255)
    axs[-1].imshow(dis, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("d", [(0, 0, 0, 0), style.RED]), vmin=0, vmax=0.5)


def main() -> None:
    style.apply()
    ids = json.load(open(PREDS / "image_ids.json"))
    gt = np.load(PREDS / "gt.npy")
    per = pd.read_csv(ROOT / "results" / "paper" / "per_image_iou.csv")
    s43 = per[per.model == "S43"].groupby(["image", "cls"]).iou.mean().unstack()
    models = {m: load(m) for m, _ in SHOW}
    all_models = sorted({p.name.rsplit("_seed", 1)[0] for p in PREDS.glob("*_seed0.npy")})
    stack = np.stack([load(m) for m in all_models])  # (M, 50, H, W)
    majority = np.apply_along_axis(lambda v: np.bincount(v, minlength=5).argmax(), 0, stack)
    disagree = (stack != majority[None]).mean(0)

    rows = []
    score = s43[list(CLASSES)].mean(axis=1).sort_values(ascending=False)
    showcase = list(score.index[:1])
    for img_id in showcase:
        rows.append(("ALL", None, ids.index(img_id), img_id, s43.loc[img_id, ["ZP", "TE", "BL", "ICM"]].mean()))
        for cname, k in CLASSES.items():
            rows.append((cname, k, ids.index(img_id), img_id, s43.loc[img_id, cname]))

    ncol = 3 + len(SHOW) + 1
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.55 * ncol, 1.62 * len(rows)))
    fill = {"ZP": style.YELLOW, "TE": style.ORANGE, "ICM": style.RED}
    for r, (cname, k, i, img_id, iou) in enumerate(rows):
        img = np.array(Image.open(ROOT / "dataset" / "source" / "images" / img_id).convert("L").resize((256, 256)))
        if k is None:  # overall row: whole image, every class filled
            overall_row.index = i
            overall_row(axes[r], img, gt[i], models, disagree[i], iou)
            continue
        g = gt[i] == k
        err = (models["S43"][i] == k) != g
        y0, x0 = crop_box(err, g, CROP)
        sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        ax = axes[r, 0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.add_patch(plt.Rectangle((x0, y0), CROP, CROP, fill=False, color=style.BLUE, lw=1.4))
        ax.set_ylabel(f"{cname}\nDiGA-Net+ IoU {iou:.2f}", fontsize=8, color=style.INK2)
        axes[r, 1].imshow(img[sl], cmap="gray", vmin=0, vmax=255)
        axes[r, 2].imshow(img[sl], cmap="gray", vmin=0, vmax=255)
        axes[r, 2].contour(g[sl], levels=[0.5], colors=[style.BLUE], linewidths=1.3)
        for c, (m, _label) in enumerate(SHOW, start=3):
            p = models[m][i] == k
            ax = axes[r, c]
            ax.imshow(img[sl], cmap="gray", vmin=0, vmax=255)
            ax.imshow(np.ma.masked_where(~p[sl], p[sl]), cmap=plt.matplotlib.colors.ListedColormap([fill[cname]]),
                      alpha=0.45, vmin=0, vmax=1)
            ax.contour(g[sl], levels=[0.5], colors=[style.BLUE], linewidths=1.1)
            pi = np.logical_and(p, g).sum() / max(np.logical_or(p, g).sum(), 1)
            ax.text(0.03, 0.03, f"IoU {pi:.2f}", transform=ax.transAxes, fontsize=7, color=style.INK,
                    bbox=dict(facecolor=style.SURFACE, edgecolor="none", alpha=0.8, pad=1.2), va="bottom")
        ax = axes[r, -1]
        ax.imshow(img[sl], cmap="gray", vmin=0, vmax=255)
        im = ax.imshow(disagree[i][sl], cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
            "d", [(0, 0, 0, 0), style.RED]), vmin=0, vmax=0.5)
        ax.contour(g[sl], levels=[0.5], colors=[style.BLUE], linewidths=1.1)
    heads = ["image", "crop / GT overlay", "GT contour"] + [lab for _, lab in SHOW] + [f"disagreement\n({len(all_models)} models)"]
    for c, h in enumerate(heads):
        axes[0, c].set_title(h, fontsize=8, color=style.INK2, pad=4)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    cax = fig.add_axes([0.905, 0.10, 0.008, 0.16])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("fraction of models\ndisagreeing", fontsize=7, color=style.INK2)
    cb.ax.tick_params(labelsize=6)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.05, right=0.9, top=0.95, bottom=0.02)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print("saved", OUT, "rows:", [(c, i, round(v, 3)) for c, _, _, i, v in rows])


if __name__ == "__main__":
    main()
