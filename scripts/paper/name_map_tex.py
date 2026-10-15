#!/usr/bin/env python3
"""Appendix table: reader-facing model name -> internal configuration code and config file (paper/tables/name_map.tex)."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402


def config_for(code: str) -> str:
    stem = code.split("_grouped")[0]
    hits = sorted(glob.glob(str(ROOT / "configs" / f"{code}*.yaml"))) or sorted(glob.glob(str(ROOT / "configs" / f"{stem}*.yaml")))
    if code == "B9_nnunet":
        hits = [str(ROOT / "configs/B9_nnunet_grouped_v1_metricfixed.yaml")]
    if code == "B10_MASSNet":
        hits = sorted(glob.glob(str(ROOT / "configs/MASS_massnet*.yaml")))
    if code == "B11_BLASTNet_documented_recon":
        hits = sorted(glob.glob(str(ROOT / "configs/B10_blastnet*.yaml")))
    return Path(hits[0]).name if hits else "---"


def esc(t: str) -> str:
    return t.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def main() -> None:
    summ = pd.read_csv(ROOT / "results/paper/model_summary.csv")
    order = {"nnU-Net": 0, "DiGA-Net family": 1, "Boundary stage": 2, "Literature block": 3, "Recipe rung": 4, "Blastocyst paper": 5, "Generic baseline": 6}
    summ["o"] = summ.family.map(order)
    summ = summ.sort_values(["o", "miou_fg"], ascending=[True, False])
    rows = [(style.display_name(r.model), config_for(r.model)) for r in summ.itertuples()]
    ncol = 3
    per = (len(rows) + ncol - 1) // ncol
    lines = [r"\begin{tabular*}{\tblwidth}{@{}" + "ll" * ncol + r"@{}}", r"\toprule",
             " & ".join([r"Name in paper & Configuration file (\texttt{configs/})"] * ncol) + r" \\", r"\midrule"]
    for i in range(per):
        cells = []
        for c in range(ncol):
            k = i + c * per
            cells += list(rows[k]) if k < len(rows) else ["", ""]
        lines.append(" & ".join(esc(x) for x in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    (ROOT / "paper/tables/name_map.tex").write_text("\n".join(lines) + "\n")
    print("wrote name_map.tex", len(rows), "rows")


if __name__ == "__main__":
    main()
