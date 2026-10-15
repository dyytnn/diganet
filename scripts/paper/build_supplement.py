#!/usr/bin/env python3
"""Generate paper/supplement.tex from stored results (never hand-edit the tables).

Contents: S1 recipe ladder, S6 rank correlation, C2 boundary-tolerant curves, the eleven refinement stages
paired against DiGA-Net, DiGA-Net+ and nnU-Net, the per-fold cross-validation table, and the blastocoel derivation rule
with its QC statistics.  Build: python scripts/paper/build_supplement.py && (cd paper && latexmk -pdf supplement.tex)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/supplement.tex"
STAGE_LABEL = {"S46": "PointRend (faithful)", "S47": "BPR proximity refiner (proxy)", "S48": "DeepStrip (inspired)", "S49": "RefineMask (lite)",
               "S50": "Mask Transfiner (lite)", "S51": "SharpContour (inspired)", "S52": "SegFix (faithful)", "S53": "CascadePSP (lite)",
               "S54": "BMask boundary branch (inspired)", "S55": "Deep Snake (inspired)", "S56": "PolyTransform (lite)"}


def esc(t: str) -> str:
    return t.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt_ci(d, lo, hi):
    return f"${d:+.4f}$ [{lo:+.4f}, {hi:+.4f}]"


def stage_table() -> str:
    d = pd.read_csv(ROOT / "results/paper/stage_deltas.csv")
    # nnU-Net reference: pair each stage's per-image IoU (3 seeds) with nnU-Net's (3 seeds) on the same 50 images
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    from matblast.eval.paired import multiseed_paired_summary

    nnj = json.load(open(ROOT / "results/nnunet_validation_no_tta.json"))
    nn_arr = np.array([nnj["per_image_iou"][k] for k in ("0", "1", "2")], dtype=float)
    nn_ids = nnj["validation_image_ids"]
    nn = {}
    for f in (ROOT / "results").glob("S[45][0-9]_*_validation_no_tta.json"):
        j = json.load(open(f))
        if j["validation_image_ids"] != nn_ids:
            continue
        stem = [m["model"] for m in j["models"] if m["model"] not in ("S24", "S43")]
        if not stem:
            continue
        arr = np.array([j["per_image_iou"][f"{stem[0]}:seed{k}"] for k in range(3)], dtype=float)
        r = multiseed_paired_summary(np.nanmean(nn_arr[:, :, 1:], 2), np.nanmean(arr[:, :, 1:], 2), 10000, 0)
        nn[stem[0][:3]] = (r["delta"], *r["ci95_hierarchical"])
    s47 = ROOT / "results/matblast_s47_validation_no_tta.json"
    if s47.exists():
        j = json.load(open(s47)); stem = [m["model"] for m in j["models"] if m["model"] not in ("S24", "S43")]
        if stem and j["validation_image_ids"] == nn_ids:
            arr = np.array([j["per_image_iou"][f"{stem[0]}:seed{k}"] for k in range(3)], dtype=float)
            r = multiseed_paired_summary(np.nanmean(nn_arr[:, :, 1:], 2), np.nanmean(arr[:, :, 1:], 2), 10000, 0)
            nn["S47"] = (r["delta"], *r["ci95_hierarchical"])
    lines = [r"\begin{table}[H]", r"\caption{The eleven refinement stages paired against DiGA-Net, DiGA-Net+ and nnU-Net (mIoU-FG, 95\,\% hierarchical-bootstrap CI; 3 seeds, 50 validation images). TE and ICM columns are the paired differences to DiGA-Net.}",
             r"\centering\footnotesize", r"\begin{tabular}{lllll}", r"\toprule",
             r"Method & $\Delta$ vs base & $\Delta$ vs DiGA-Net+ & $\Delta$ vs nnU-Net & $\Delta$TE / $\Delta$ICM vs base \\", r"\midrule"]
    for st in sorted(STAGE_LABEL):
        r24 = d[(d.stage == st) & (d.ref == "S24")].set_index("metric")
        r43 = d[(d.stage == st) & (d.ref == "S43")].set_index("metric")
        if r24.empty:
            continue
        m = r24.loc["mIoU"]
        c43 = fmt_ci(*r43.loc["mIoU"][["delta", "lo", "hi"]]) if not r43.empty else "---"
        cnn = fmt_ci(*nn[st]) if st in nn else "---"
        lines.append(f"{esc(STAGE_LABEL[st])} & {fmt_ci(m.delta, m.lo, m.hi)} & {c43} & {cnn} & ${r24.loc['TE'].delta:+.4f}$ / ${r24.loc['ICM'].delta:+.4f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def cv_table() -> str:
    s = json.load(open(ROOT / "results/cv5/cv5_summary.json"))
    j5 = json.load(open(ROOT / "results/cv5/cv5_jaccard5.json"))
    models = ["B9_nnunet", "S24", "S43", "S54"]
    names = {"B9_nnunet": "nnU-Net (1 seed/fold)", "S24": "DiGA-Net", "S43": "DiGA-Net+", "S54": "DiGA-Net + boundary branch"}
    lines = [r"\begin{table}[H]", r"\caption{Per-fold mIoU-FG of the frozen shortlist under 5-fold embryo-grouped cross-validation (fold 0 = historical validation set; DiGA-Net+ models 3 seeds per fold). The last two columns give the pooled clean-fold (1--4) mIoU-FG and 5-class Jaccard.}",
             r"\centering\footnotesize", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Model & fold 0 & fold 1 & fold 2 & fold 3 & fold 4 & clean mIoU-FG & clean Jac$_5$ \\", r"\midrule"]
    for m in models:
        vals = [s["per_fold"][str(k)][m]["miou_fg_mean"] for k in range(5)]
        clean = s["groups"]["clean_folds_1_4"]["models"][m]["miou_fg_mean"]
        key = "nnU-Net" if m == "B9_nnunet" else m
        jac = j5[key]["clean"]
        lines.append(f"{names[m]} & " + " & ".join(f"{v:.4f}" for v in vals) + f" & {clean:.4f} & {jac:.4f} \\\\")
    lines += [r"\midrule"]
    for cand, ref in [("S43", "S24"), ("S54", "S24"), ("B9_nnunet", "S24"), ("S43", "B9_nnunet"), ("S54", "B9_nnunet")]:
        c = s["groups"]["clean_folds_1_4"]["paired"][f"vs_{ref}"][cand]["macro_fg_iou"]
        lines.append(rf"\multicolumn{{8}}{{l}}{{{names[cand]} $-$ {names[ref]}, clean folds: {fmt_ci(c['delta'], *c['ci95_hierarchical'])}, image-level $p$ = {c['p_image_wilcoxon_greater']:.3g}}} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def bl_text() -> str:
    q = pd.read_csv(ROOT / "dataset/source/bl_derivation_qc.csv")
    ok = q[q.status == "derived_candidate"]
    n_te = int((ok.method == "filled_te_minus_te_icm").sum())
    n_zp = int((ok.method == "filled_zp_fallback_minus_zp_te_icm").sum())
    return (f"The public release provides ZP, TE and ICM masks. The blastocoel (BL) mask used in this study is derived deterministically: "
            f"the region enclosed by the TE ring is filled and TE and ICM are subtracted ({n_te} of {len(ok)} images); when the TE ring is open or "
            f"fragmented the region enclosed by the ZP ring is used instead and ZP, TE and ICM are subtracted ({n_zp} images); {int((q.status != 'derived_candidate').sum())} image had no source mask. "
            f"The derived BL occupies a median {ok.bl_fraction.median() * 100:.0f}\\,\\% of the image (range {ok.bl_fraction.min() * 100:.0f}--{ok.bl_fraction.max() * 100:.0f}\\,\\%). "
            "These masks are flagged as derived in the released data and are not expert ground truth; the main text excludes BL from every argument that carries a conclusion.")


def main() -> None:
    tex = r"""\documentclass[a4paper,11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,float}
\usepackage[hidelinks]{hyperref}
\graphicspath{{figures/}}
\title{Supplementary material for\\``DiGA-Net: lightweight blastocyst segmentation at the limit set by label boundary ambiguity''}
\author{}
\date{}
\begin{document}
\maketitle
\noindent All tables in this document are generated programmatically from the released result files; no number is hand-edited.

\section{Recipe ladder (Fig.\ S1)}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{S1_recipe_ladder.png}
\caption{Recipe rungs transferring nnU-Net ingredients to the DiGA-Net+ protocol, one change per rung (recipe control at 9.8\,M parameters and $384^2$ input; then six-stage residual InstanceNorm backbone, nnU-Net augmentation, EMA, ICM oversampling, per-image Dice, and padding to $384\times448$, each added alone). Validation mIoU-FG per rung with seed spread and training/validation loss curves showing the over-fitting of the unregularised control.}
\end{figure}

\section{Rank correlation of per-image scores (Fig.\ S6)}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{S6_rank_correlation.png}
\caption{Spearman rank correlation of seed-averaged per-image TE IoU between all 65 models on the 50 validation images (median 0.87).}
\end{figure}

\section{Per-class zoom-in view (Fig.\ S3)}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{F9_zoom_per_class.png}
\caption{Whole-frame and per-class zoom-in view of one well-segmented validation image (the highest seed-averaged DiGA-Net+ mean IoU over ZP, TE and ICM). The first row shows the whole frame with all four classes (ZP yellow, TE orange, BL aqua, ICM red) as annotation overlay, annotation contours, model overlays with foreground mIoU, and the disagreement map. The three class rows below show a $96\times96$-px crop: the crop location (blue box), the crop at working resolution, the annotated contour of the class (blue), the predictions of DiGA-Net+, nnU-Net, U-Net and the MedNeXt-block variant (filled; inset gives the class IoU on that image), and the fraction of twelve stored models that disagree with their pixel-wise majority. Each crop is centred on the largest connected DiGA-Net+ error component of that class. Raw intensities are shown.}
\end{figure}

\section{Boundary-tolerant metrics (Fig.\ C2)}
\begin{figure}[H]\centering\includegraphics[width=0.9\textwidth]{C2_boundary_tolerant.png}
\caption{Trimmed IoU after excluding a $k$-px band around the annotated class boundary and boundary F-measure at tolerance $k$, per class, for DiGA-Net+ and nnU-Net.}
\end{figure}

\section{Refinement stages paired against three references}
""" + stage_table() + r"""

\section{Per-fold cross-validation results}
""" + cv_table() + r"""

\section{Blastocoel label derivation}
""" + bl_text() + r"""

\end{document}
"""
    OUT.write_text(tex)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
