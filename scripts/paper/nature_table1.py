#!/usr/bin/env python3
"""Table 1 of the Nature Methods version: validation and cross-validation in one table.

Merges what the Elsevier version splits over Table 1 (50-image validation set) and Table 5 (5-fold
embryo-grouped cross-validation), because an Analysis is limited to six display items. Every number comes
from the released result files; nothing is retyped.

    python scripts/paper/nature_table1.py   ->  paper_nature/tables/table1.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

# rows: internal code -> (display name, whether it is in the cross-validated shortlist)
ROWS = [
    ("B9_nnunet", True),
    ("S43", True),
    ("S54_matblast_bmask_boundary_grouped_v1", True),
    ("S24", True),
    ("B12_BLASTO_Net", False),
    ("B3", False),
    ("B4", False),
    ("B0", False),
    ("B5", False),
    ("B10_MASSNet", False),
]
CV_KEY = {"B9_nnunet": "B9_nnunet", "S43": "S43", "S54_matblast_bmask_boundary_grouped_v1": "S54", "S24": "S24"}


def main() -> None:
    per = pd.read_csv(ROOT / "results/paper/per_image_iou.csv")
    summ = pd.read_csv(ROOT / "results/paper/model_summary.csv").set_index("model")
    cost = pd.read_csv(ROOT / "results/paper/model_cost.csv").set_index("model")
    cv = json.load(open(ROOT / "results/cv5/cv5_summary.json"))
    clean = cv["groups"]["clean_folds_1_4"]

    fg = per[per.cls != "BG"].groupby(["model", "seed", "image"]).iou.mean().groupby(["model", "seed"]).mean()
    mean, sd = fg.groupby("model").mean(), fg.groupby("model").std(ddof=1)

    lines = [
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Validation set (50 images, 3 seeds)} & \multicolumn{2}{c}{Cross-validation (199 unseen images)} & \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
        r"Model & mIoU$_{\mathrm{FG}}$ & Jac$_5$ (\%) & TE & ICM & mIoU$_{\mathrm{FG}}$ & $\Delta$ vs nnU-Net [95\% CI] & Params (M) \\",
        r"\midrule",
    ]
    for code, in_cv in ROWS:
        name = style.display_name(code)
        m, s = mean[code], sd.get(code, float("nan"))
        r = summ.loc[code]
        val = f"{m:.4f} $\\pm$ {s:.4f}" if s == s else f"{m:.4f}"
        if in_cv:
            k = CV_KEY[code]
            cvm = f"{clean['models'][k]['miou_fg_mean']:.4f}"
            if k == "B9_nnunet":
                delta = "reference"
            else:
                d = clean["paired"]["vs_B9_nnunet"][k]["macro_fg_iou"]
                lo, hi = d["ci95_hierarchical"]
                delta = f"${d['delta']:+.4f}$ [{lo:+.4f}, {hi:+.4f}]"
        else:
            cvm, delta = "---", "---"
        params = f"{cost.loc[code].params_M:.2f}" if code in cost.index else "---"
        bold = r"\textbf{" if code in ("B9_nnunet", "S43") else ""
        end = "}" if bold else ""
        lines.append(f"{bold}{name}{end} & {val} & {r.miou_5cls * 100:.2f} & {r.TE:.3f} & {r.ICM:.3f} & {cvm} & {delta} & {params} \\\\")
        if code == "S24":
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = ROOT / "paper_nature/tables/table1.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out.relative_to(ROOT), f"({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
