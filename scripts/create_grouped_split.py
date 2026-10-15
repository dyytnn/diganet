#!/usr/bin/env python3
"""Create a conservative filename-grouped train/validation split.

The public source data has no patient or embryo metadata. Shared filename
stems are therefore only a sensitivity-analysis heuristic, not proof of subject
identity or leakage.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def embryo_group(filename: str) -> str:
    """Return a conservative source-name group for sensitivity analysis."""
    stem = Path(filename).stem
    match = re.match(r"^(Blast_PCRM_\d+)\s+D\d+", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.match(r"^(Blast_PCRM_R\d+-\d+)[a-zA-Z]$", stem)
    if match:
        return match.group(1)
    match = re.match(r"^(Blast_PCRM_d\d+\.\d+)\s+(?:left|right)$", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.match(r"^(Blast_PCRM_2 x 2AB d\d+)\s+(?:left|right)$", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="dataset/source/images")
    parser.add_argument("--output", default="data/splits/source_grouped_v1.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must be strictly between zero and one")

    image_dir = Path(args.image_dir)
    names = sorted(p.name for p in image_dir.iterdir() if p.is_file())
    if not names:
        raise ValueError(f"No source images found in {image_dir}")
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[embryo_group(name)].append(name)
    group_names = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(group_names)
    target_val = round(len(names) * args.val_fraction)
    val_groups: list[str] = []
    val_count = 0
    for group in group_names:
        if val_count < target_val:
            val_groups.append(group)
            val_count += len(groups[group])
    val_set = set(val_groups)
    val = sorted(name for group in val_groups for name in groups[group])
    train = sorted(name for group in group_names if group not in val_set for name in groups[group])
    payload = {
        "train": train,
        "val": val,
        "seed": args.seed,
        "split_type": "deterministic_filename_grouped_sensitivity_v1",
        "grouping": "filename heuristic only; paired a/b and left/right names stay together, subject identity unverified",
        "groups": {group: sorted(items) for group, items in sorted(groups.items())},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    train_groups = {embryo_group(name) for name in train}
    val_groups_check = {embryo_group(name) for name in val}
    if train_groups & val_groups_check:
        raise RuntimeError("group leakage detected after split creation")
    print(f"wrote {output}: {len(train)} train / {len(val)} val, {len(groups)} groups")


if __name__ == "__main__":
    main()
