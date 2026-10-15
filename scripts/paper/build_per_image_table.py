#!/usr/bin/env python3
"""Assemble one long table of per-image, per-class IoU for every three-seed model.

Sources: every ``results/*_validation_no_tta.json`` written by eval_benchmark.py or the
nnU-Net evaluator (all share the same 50 validation image ids). Output:
``results/paper/per_image_iou.csv`` with columns model, family, seed, image, class, iou, and
``results/paper/model_summary.csv`` (seed-averaged per-class means, family, n_seeds).
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import style  # noqa: E402

CLASSES = ["BG", "ZP", "TE", "BL", "ICM"]
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

BLASTO_PAPERS = {"B3": "BTFormer", "B4": "I2CNet", "B5": "MiMics-Net", "B12_BLASTO_Net": "Blasto-Net",
                 "B11_BLASTNet_documented_recon": "BLAST-NET", "B10_MASSNet": "MASS-Net"}
GENERIC = {"B0": "U-Net", "B1": "Attention U-Net", "B6": "DeepLabV3+", "B7": "TransUNet", "B8": "SegFormer-B0"}


def family(model: str) -> str:
    if model.startswith("B9"):
        return "nnU-Net"
    if model in BLASTO_PAPERS:
        return "Blastocyst paper"
    if model in GENERIC:
        return "Generic baseline"
    if re.match(r"^(A1|L\d\d)_", model):
        return "Literature block"
    if re.match(r"^S(4[6-9]|5[0-6])", model) and model != "S47":
        return "Boundary stage"
    if model == "S47":
        return "Boundary stage"
    if model.startswith("V2_"):
        return "Recipe rung"
    return "DiGA-Net family"


def short(model: str) -> str:
    if model in BLASTO_PAPERS:
        return BLASTO_PAPERS[model]
    if model in GENERIC:
        return GENERIC[model]
    if model.startswith("B9"):
        return "nnU-Net"
    return style.display_name(model)


def main() -> None:
    rows: dict[tuple, list] = {}
    ids_ref = None
    for path in sorted(glob.glob(str(ROOT / "results" / "*_validation_no_tta.json"))):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if "per_image_iou" not in d or "validation_image_ids" not in d:
            continue
        ids = d["validation_image_ids"]
        if ids_ref is None:
            ids_ref = ids
        if ids != ids_ref:
            print("skip (different image order):", path)
            continue
        per = d["per_image_iou"]
        for key, table in per.items():
            model, seed = key.split(":") if ":" in key else (key, "seed0")
            seed = int(re.sub(r"\D", "", seed) or 0)
            if (model, seed) in rows:
                continue  # first file wins (same checkpoints, duplicated across reports)
            rows[(model, seed)] = table
    long = []
    for (model, seed), table in rows.items():
        for image, vals in zip(ids_ref, table):
            for c, v in zip(CLASSES, vals):
                long.append((model, short(model), family(model), seed, image, c, np.nan if v is None else float(v)))
    df = pd.DataFrame(long, columns=["model", "label", "family", "seed", "image", "cls", "iou"])
    n_seeds = df.groupby("model")["seed"].nunique()
    keep = n_seeds[n_seeds == 3].index
    dropped = sorted(set(df.model) - set(keep))
    df = df[df.model.isin(keep)]
    df.to_csv(OUT / "per_image_iou.csv", index=False)
    summ = (df.groupby(["model", "label", "family", "cls"])["iou"].mean().unstack("cls").reset_index())
    summ["miou_fg"] = summ[["ZP", "TE", "BL", "ICM"]].mean(1)
    summ["miou_5cls"] = summ[CLASSES].mean(1)
    summ = summ.sort_values("miou_fg", ascending=False)
    summ.to_csv(OUT / "model_summary.csv", index=False)
    print(f"models with 3 seeds: {len(keep)} | dropped (not 3 seeds): {dropped}")
    print(summ[["label", "family", "miou_fg", "miou_5cls", "TE", "ICM"]].head(8).to_string(index=False))
    print(summ.family.value_counts().to_string())


if __name__ == "__main__":
    main()
