#!/usr/bin/env python3
"""Generate the Extended Data tables of the Nature Methods version (paper_nature/tables/ed*.tex).

The Elsevier version keeps these in its appendix and supplement; an Analysis moves them to Extended Data.
They are regenerated from the same result files, not copied from paper/main.tex, so the two versions cannot
drift. The CAS-specific \\tblwidth macro is replaced by a plain tabular sized for the Springer Nature text
block.

  ED Table 1  all 65 models on the validation set
  ED Table 2  reproduction protocol of the blastocyst-specific networks and nnU-Net
  ED Table 3  the eleven refinement methods against the pre-registered criterion
  ED Table 4  topology and anatomical nesting, including the annotation scored against itself
  ED Table 5  agreement between the released annotation and a second manual reading

    python scripts/paper/nature_extended_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_nature/tables"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

FAMILY_ORDER = {"nnU-Net": 0, "DiGA-Net family": 1, "Boundary stage": 2, "Literature block": 3,
                "Recipe rung": 4, "Blastocyst paper": 5, "Generic baseline": 6}
STAGE_LABEL = {"S46": "PointRend (faithful)", "S47": "BPR proximity refiner (proxy)", "S48": "DeepStrip (inspired)",
               "S49": "RefineMask (lite)", "S50": "Mask Transfiner (lite)", "S51": "SharpContour (inspired)",
               "S52": "SegFix (faithful)", "S53": "CascadePSP (lite)", "S54": "BMask boundary branch (inspired)",
               "S55": "Deep Snake (inspired)", "S56": "PolyTransform (lite)"}


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def ed_table1() -> None:
    per = pd.read_csv(ROOT / "results/paper/per_image_iou.csv")
    summ = pd.read_csv(ROOT / "results/paper/model_summary.csv")
    fg = per[per.cls != "BG"].groupby(["model", "seed", "image"]).iou.mean().groupby(["model", "seed"]).mean()
    mean, sd = fg.groupby("model").mean(), fg.groupby("model").std(ddof=1)
    summ["o"] = summ.family.map(FAMILY_ORDER)
    summ = summ.sort_values(["o", "miou_fg"], ascending=[True, False])
    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r"Model & Family & mIoU$_{\mathrm{FG}}$ $\pm$ sd & Jac$_5$ & ZP & TE & BL & ICM \\", r"\midrule"]
    last = None
    for r in summ.itertuples():
        if last is not None and r.family != last:
            lines.append(r"\addlinespace[2pt]")
        last = r.family
        s = sd.get(r.model, float("nan"))
        val = f"{mean[r.model]:.4f} $\\pm$ {s:.4f}" if s == s else f"{mean[r.model]:.4f}"
        lines.append(f"{esc(style.display_name(r.model))} & {esc(r.family)} & {val} & {r.miou_5cls:.4f} & {r.ZP:.3f} & {r.TE:.3f} & {r.BL:.3f} & {r.ICM:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "ed_table1.tex").write_text("\n".join(lines) + "\n")
    print(f"ed_table1.tex: {len(summ)} models")


def ed_table2() -> None:
    rows = [
        ("BTFormer", "paper architecture", "1 ch, $256^2$", "paper WSJA", "paper", r"Adam $8\times10^{-4}$, $\le$2000 ep., early stop", "---"),
        ("I2CNet", "paper architecture", "1 ch, $256^2$", "CE (paper)", "paper", r"Adam $1\times10^{-4}$, $\le$2000 ep., early stop", "paper reports I2C2Net; we reimple\\-mented its published I2CNet path"),
        ("MiMics-Net", "paper architecture", "1 ch, $400^2$", "genera\\-lised Dice", r"paper, $16\times$ repeats", r"Adam $1\times10^{-3}$, 64 ep., batch 4", "matched update budget"),
        ("BLAST-NET", "ResNet-50, CAPP at rates 1--16, dense progressive PixelShuffle decoder", "3 ch, $256^2$", "common", "common", "common", "undisclosed details made explicit in the config"),
        ("Blasto-Net", "EfficientNet-B3, CBAM, EAAM segmentation path", "3 ch, $256^2$", "common", "common", "common", "grading and implantation heads omitted (no labels)"),
        ("MASS-Net", "Table-1 architecture of the paper", "1 ch, global L2, $400^2$", "common", r"paper, $16\times$ repeats", r"Adam $1\times10^{-4}$, 40 ep., batch 20", "paper used a 235-image variant"),
        ("nnU-Net", "official v2, 2D configuration, unmodified", r"planner $384\times448$", "Dice + CE", "nnU-Net default", "SGD, 1000 epochs", "split converted to nnU-Net case ids; 47.6\\,M unique para\\-meters"),
    ]
    # widths sum to 15.6 cm, inside the 16.6 cm text block; L = ragged-right p column (see main.tex preamble)
    # widths sum to 15.7 cm, inside the 16.6 cm text block; L = ragged-right p column (see main.tex preamble).
    # Long header words carry discretionary hyphens so they break instead of overflowing into the next column.
    lines = [r"\begin{tabular}{@{}L{1.9cm}L{3.2cm}L{1.5cm}L{1.7cm}L{2.0cm}L{2.6cm}L{2.8cm}@{}}", r"\toprule",
             r"Model & Archi\-tecture as reimple\-mented & Input & Loss & Augmen\-tation & Optimiser, schedule & Notes \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "ed_table2.tex").write_text("\n".join(lines) + "\n")
    print("ed_table2.tex: 7 reproductions")


def ed_table3() -> None:
    d = pd.read_csv(ROOT / "results/paper/stage_deltas.csv")
    lines = [r"\begin{tabular}{@{}llrrrl@{}}", r"\toprule",
             r"Method & Fidelity & $\Delta$mIoU$_{\mathrm{FG}}$ [95\% CI] & $\Delta$TE & $\Delta$ICM & Criterion \\", r"\midrule"]
    rows = []
    for st, label in sorted(STAGE_LABEL.items()):
        r24 = d[(d.stage == st) & (d.ref == "S24")].set_index("metric")
        if r24.empty:
            continue
        m = r24.loc["mIoU"]
        fid = label.split("(")[-1].rstrip(")") if "(" in label else "---"
        name = label.split(" (")[0]
        passed = r24.loc["TE"].delta > 0 and r24.loc["ICM"].delta > 0
        rows.append((m.delta, f"{esc(name)} & {fid} & ${m.delta:+.4f}$ [{m.lo:+.4f}, {m.hi:+.4f}] & ${r24.loc['TE'].delta:+.4f}$ & ${r24.loc['ICM'].delta:+.4f}$ & {'---' if passed else 'fail'} \\\\"))
    lines += [r for _, r in sorted(rows, reverse=True)]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "ed_table3.tex").write_text("\n".join(lines) + "\n")
    print(f"ed_table3.tex: {len(rows)} refinement methods")


def ed_table4() -> None:
    t = pd.read_csv(ROOT / "results/paper/topology_summary.csv")
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Betti-0 error} & \multicolumn{2}{c}{False components} & \multicolumn{3}{c}{Nesting violation (\% px / \% images)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
             r"Model & TE & ICM & TE & ICM & ICM outside TE & BL outside TE & TE outside ZP \\", r"\midrule"]
    for r in t.itertuples():
        name = "Annotation (reference)" if r.model == "GT" else esc(r.model)
        lines.append(f"{name} & {r.betti0_te:.2f} & {r.betti0_icm:.2f} & {r.false_te:.2f} & {r.false_icm:.2f} & "
                     f"{r.icm_out_te_px:.1f} / {r.icm_out_te_img:.0f} & {r.bl_out_te_px:.1f} / {r.bl_out_te_img:.0f} & {r.te_out_zp_px:.1f} / {r.te_out_zp_img:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "ed_table4.tex").write_text("\n".join(lines) + "\n")
    print(f"ed_table4.tex: {len(t)} rows")


AGREEMENT_RUN = "results/annotation_agreement/A3_v5_technical_256_20260915"
CLASS_LABEL = {"ZP": "Zona pellucida", "TE": "Trophectoderm", "ICM": "Inner cell mass",
               "macro3_complete_cases": "Three-class mean"}


def ed_table5() -> None:
    src = ROOT / AGREEMENT_RUN / "summary.csv"
    if not src.exists():
        print(f"ed_table5.tex: skipped, {src.relative_to(ROOT)} not found")
        return
    s = pd.read_csv(src)
    s = s[s.cohort == "technical_visual_review"].set_index(["class", "metric"])
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Class & $n$ images / groups & IoU [95\% CI] & Dice & Boundary-F@2 & HD95 (px) \\", r"\midrule"]
    for cls, label in CLASS_LABEL.items():
        if (cls, "iou") not in s.index:
            continue
        r = s.loc[(cls, "iou")]
        lines.append(f"{label} & {int(r.n_images)} / {int(r.n_groups)} & "
                     f"{r['mean']:.3f} [{r.ci95_low:.3f}, {r.ci95_high:.3f}] & "
                     f"{s.loc[(cls, 'dice'), 'mean']:.3f} & {s.loc[(cls, 'boundary_f2'), 'mean']:.3f} & "
                     f"{s.loc[(cls, 'hd95'), 'mean']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "ed_table5.tex").write_text("\n".join(lines) + "\n")
    print(f"ed_table5.tex: {len(CLASS_LABEL)} rows from {AGREEMENT_RUN.split('/')[-1]}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ed_table1(); ed_table2(); ed_table3(); ed_table4(); ed_table5()
