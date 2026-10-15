#!/usr/bin/env python3
"""Appendix table: all 65 models (mIoU-FG mean +/- seed sd, per-class IoU) -> paper/tables/full_models.tex."""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
per = pd.read_csv(ROOT / "results/paper/per_image_iou.csv")
summ = pd.read_csv(ROOT / "results/paper/model_summary.csv")
fg = per[per.cls != "BG"].groupby(["model", "seed", "image"]).iou.mean().groupby(["model", "seed"]).mean()
sd = fg.groupby("model").std(ddof=1)
mean = fg.groupby("model").mean()
order = {"nnU-Net": 0, "DiGA-Net family": 1, "Boundary stage": 2, "Literature block": 3, "Recipe rung": 4, "Blastocyst paper": 5, "Generic baseline": 6}
summ["fam_order"] = summ.family.map(order)
summ = summ.sort_values(["fam_order", "miou_fg"], ascending=[True, False])
def esc(s): return s.replace("_", r"\_").replace("&", r"\&")
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from name_map_tex import config_for  # noqa: E402


def cfg_stem(model: str) -> str:
    c = config_for(model)
    return re.sub(r"(_grouped_v\d(_metricfixed)?)?\.yaml$", "", c)


lines = [r"\begin{tabular*}{\tblwidth}{@{}llrrrrrr@{}}", r"\toprule",
         r"Model & Family & \miou{} $\pm$ sd & Jac\textsubscript{5} & ZP & TE & BL & ICM \\", r"\midrule"]
last = None
for _, r in summ.iterrows():
    if last is not None and r.family != last:
        lines.append(r"\addlinespace[2pt]")
    last = r.family
    m = r.model
    lines.append(f"{esc(r.label)} & {esc(r.family)} & {mean[m]:.4f} $\\pm$ {sd[m]:.4f} & {r.miou_5cls:.4f} & {r.ZP:.3f} & {r.TE:.3f} & {r.BL:.3f} & {r.ICM:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular*}"]
out = ROOT / "paper/tables/full_models.tex"
out.write_text("\n".join(lines) + "\n")
print("wrote", out, len(summ), "rows")
