#!/usr/bin/env python3
"""Build the 5-fold patient-grouped split used for the frozen-shortlist cross-validation.

Fold 0 is exactly the historical validation set of ``source_grouped_v1.json`` (the images on
which every model-selection decision was made - the "selection fold"). Folds 1-4 partition the
historical training images by their embryo/source group with a fixed seed, so their scores were
never used for any decision. Output schema (read by ``build_dataloaders`` via ``data.fold``):
``folds[k] = {"train": [...], "val": [...]}`` plus ``fold_members``, ``groups``, provenance.
The output path must not exist (frozen splits are never overwritten).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/splits/source_grouped_v1.json")
    ap.add_argument("--out", default="data/splits/source_grouped_5fold_v1.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    base = json.load(open(args.base))
    groups: dict[str, list[str]] = base["groups"]
    train, val = set(base["train"]), set(base["val"])
    train_groups = sorted(g for g, imgs in groups.items() if set(imgs) <= train)
    val_groups = sorted(g for g, imgs in groups.items() if set(imgs) <= val)
    assert len(train_groups) + len(val_groups) == len(groups), "a group straddles train/val"
    rng = random.Random(args.seed)
    rng.shuffle(train_groups)
    # greedy balanced assignment of groups to 4 folds by image count
    members: dict[int, list[str]] = {k: [] for k in range(1, 5)}
    for g in sorted(train_groups, key=lambda g: (-len(groups[g]), g)):
        k = min(members, key=lambda k: (len(members[k]), k))
        members[k].extend(groups[g])
    members[0] = sorted(val)
    all_images = sorted(train | val)
    folds = {}
    for k in range(5):
        held = set(members[k])
        folds[str(k)] = {
            "train": sorted(i for i in all_images if i not in held),
            "val": sorted(held),
        }
    out = {
        "seed": args.seed,
        "split_type": "patient_grouped_5fold_v1",
        "grouping": base["grouping"],
        "base_split": args.base,
        "selection_fold": 0,
        "note": "fold 0 = historical validation set (used for all model selection); folds 1-4 partition the historical training set by group and were never used for any decision",
        "fold_members": {str(k): sorted(v) for k, v in members.items()},
        "folds": folds,
        "groups": groups,
    }
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(f"{path} exists; frozen splits are never overwritten")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print({k: len(v) for k, v in out["fold_members"].items()}, "->", path)


if __name__ == "__main__":
    main()
