#!/usr/bin/env python3
"""C4+: topology beyond the Betti number — matched components and nesting (containment) violations.

Inputs: results/paper/preds/<model>_seed<k>.npy (int8, 50x256x256), gt.npy. For every model (seed-averaged) and
for the ground truth itself:
  * Betti-0 / Betti-1 error per class (|#components_pred - #components_gt|, |#holes_pred - #holes_gt|)
  * component matching (2-D analogue of Betti matching): a predicted component is *matched* when it overlaps a
    GT component of the same class; reports false components (unmatched predicted) and missed components
    (GT components without any overlapping prediction) per image
  * nesting violations for the anatomical order BG ⊃ ZP ⊃ TE ⊃ BL ⊃ ICM:
        TE, BL, ICM must lie inside the region enclosed by the ZP ring  (fill_holes(ZP))
        BL, ICM must lie inside the region enclosed by the TE ring      (fill_holes(TE))
    reported as the fraction of class pixels outside the enclosing region and as the image-level violation
    rate (> 2 % of the class's pixels outside). The ground truth is scored too: hatching / open rings make the
    GT itself violate nesting, which is why hard nesting constraints are avoided (AGENTS.md rule 6).
Outputs: results/paper/topology_metrics.csv, results/paper/topology_summary.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
_DN = style.display_name
PREDS = ROOT / "results/paper/preds"
CLASSES = {1: "ZP", 2: "TE", 3: "BL", 4: "ICM"}
LABELS = {}  # filled from style.display_name below


def components(m: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(m)


def holes(m: np.ndarray) -> int:
    filled = ndimage.binary_fill_holes(m)
    _, n = ndimage.label(filled & ~m)
    return int(n)


def enclosed(m: np.ndarray) -> np.ndarray:
    """Region enclosed by a ring-like mask: the filled mask (ring + everything inside)."""
    return ndimage.binary_fill_holes(m) if m.any() else np.zeros_like(m, dtype=bool)


def per_image(pred: np.ndarray, gt: np.ndarray) -> dict:
    out = {}
    for k, name in CLASSES.items():
        p, g = pred == k, gt == k
        lp, n_p = components(p)
        lg, n_g = components(g)
        out[f"betti0_err_{name}"] = abs(n_p - n_g)
        out[f"betti1_err_{name}"] = abs(holes(p) - holes(g))
        # matching by overlap
        matched_p = len({i for i in np.unique(lp[g]) if i > 0})
        matched_g = len({i for i in np.unique(lg[p]) if i > 0})
        out[f"false_components_{name}"] = n_p - matched_p
        out[f"missed_components_{name}"] = n_g - matched_g
    zp_in, te_in = enclosed(pred == 1), enclosed(pred == 2)
    for k, name in CLASSES.items():
        if k == 1:
            continue
        m = pred == k
        out[f"outside_zp_frac_{name}"] = float((m & ~zp_in).sum() / m.sum()) if m.any() else np.nan
        if k in (3, 4):
            out[f"outside_te_frac_{name}"] = float((m & ~te_in).sum() / m.sum()) if m.any() else np.nan
    return out


def main() -> None:
    ids = json.load(open(PREDS / "image_ids.json"))
    gt = np.load(PREDS / "gt.npy")
    models = sorted({p.name.rsplit("_seed", 1)[0] for p in PREDS.glob("*_seed0.npy")})
    rows = []
    for i in range(len(ids)):
        r = {"model": "GT", "seed": 0, "image": ids[i], **per_image(gt[i], gt[i])}
        rows.append(r)
    for m in models:
        for k in range(3):
            f = PREDS / f"{m}_seed{k}.npy"
            if not f.exists():
                continue
            pred = np.load(f)
            for i in range(len(ids)):
                rows.append({"model": m, "seed": k, "image": ids[i], **per_image(pred[i], gt[i])})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/paper/topology_metrics.csv", index=False)

    # summary: mean over seeds and images; image-level violation rate (>2 %)
    lines = ["# Topology and nesting summary (50 val images; models seed-averaged)", "",
             "| model | Betti-0 err TE | Betti-0 err ICM | Betti-1 err TE | false comp TE | missed comp TE | false comp ICM | ICM outside TE ring (% px / % images>2%) | BL outside TE ring (% px / % img) | TE outside ZP ring (% px / % img) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    order = ["GT"] + [m for m in ["B9_nnunet", "S43", "S24", "S54_matblast_bmask_boundary_grouped_v1", "L01_mednext_grouped_v1_metricfixed", "S37_residual_encoder", "B0", "B3", "B4", "B12_BLASTO_Net", "S46_matblast_pointrend_faithful_grouped_v1", "S49_matblast_refine_mask_grouped_v1"] if m in models]
    summary_rows = []
    for m in order:
        d = df[df.model == m]
        g = d.groupby("image").mean(numeric_only=True)  # seed-average per image
        def pct(col): return 100 * np.nanmean(g[col])
        def rate(col): return 100 * np.nanmean(g[col] > 0.02)
        row = {"model": _DN(m), "betti0_te": g["betti0_err_TE"].mean(), "betti0_icm": g["betti0_err_ICM"].mean(), "betti1_te": g["betti1_err_TE"].mean(),
               "false_te": g["false_components_TE"].mean(), "missed_te": g["missed_components_TE"].mean(), "false_icm": g["false_components_ICM"].mean(),
               "icm_out_te_px": pct("outside_te_frac_ICM"), "icm_out_te_img": rate("outside_te_frac_ICM"),
               "bl_out_te_px": pct("outside_te_frac_BL"), "bl_out_te_img": rate("outside_te_frac_BL"),
               "te_out_zp_px": pct("outside_zp_frac_TE"), "te_out_zp_img": rate("outside_zp_frac_TE")}
        summary_rows.append(row)
        lines.append(f"| {row['model']} | {row['betti0_te']:.2f} | {row['betti0_icm']:.2f} | {row['betti1_te']:.2f} | {row['false_te']:.2f} | {row['missed_te']:.2f} | {row['false_icm']:.2f} | "
                     f"{row['icm_out_te_px']:.1f} / {row['icm_out_te_img']:.0f} | {row['bl_out_te_px']:.1f} / {row['bl_out_te_img']:.0f} | {row['te_out_zp_px']:.1f} / {row['te_out_zp_img']:.0f} |")
    pd.DataFrame(summary_rows).to_csv(ROOT / "results/paper/topology_summary.csv", index=False)
    (ROOT / "results/paper/topology_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
