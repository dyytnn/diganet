from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def grouped_split(records: list[dict], n_splits: int = 5, seed: int = 42) -> dict:
    """Deterministic patient-group split; records need image and patient_id."""
    groups = defaultdict(list)
    for r in records: groups[r.get("patient_id", r.get("embryo_id", r["image"]))].append(r["image"])
    keys = list(groups); np.random.default_rng(seed).shuffle(keys)
    folds = [{"val": [], "train": []} for _ in range(n_splits)]
    for i, key in enumerate(keys): folds[i % n_splits]["val"].extend(groups[key])
    all_ids = [r["image"] for r in records]
    for fold in folds: fold["train"] = [x for x in all_ids if x not in set(fold["val"])]
    return {"folds": {str(i): f for i, f in enumerate(folds)}, "train": folds[0]["train"], "val": folds[0]["val"], "seed": seed}


def save_split(records: list[dict], output: str, n_splits: int = 5, seed: int = 42):
    p = Path(output); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(grouped_split(records, n_splits, seed), indent=2))
