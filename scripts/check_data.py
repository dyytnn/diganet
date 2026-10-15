#!/usr/bin/env python3
"""Audit a frozen MAT-BLAST split without modifying any data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_split(path: Path) -> dict[str, Any]:
    """Load and validate the minimal structure of a frozen split."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"split must be a JSON object: {path}")
    if not isinstance(payload.get("train"), list) or not isinstance(payload.get("val"), list):
        raise ValueError("split must contain list-valued train and val entries")
    return payload


def _group_map(payload: dict[str, Any], names: set[str]) -> dict[str, str]:
    """Return the frozen split's group assignment for every selected image."""
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("grouped split must contain a groups object")
    assignment: dict[str, str] = {}
    for group, members in groups.items():
        if not isinstance(group, str) or not isinstance(members, list):
            raise ValueError("groups must map string names to image-name lists")
        for name in members:
            if not isinstance(name, str):
                raise ValueError("group members must be strings")
            if name in assignment and assignment[name] != group:
                raise ValueError(f"image occurs in multiple groups: {name}")
            assignment[name] = group
    missing = sorted(name for name in names if name not in assignment)
    if missing:
        raise ValueError(f"selected images missing from groups (first 5): {missing[:5]}")
    return assignment


def _check_masks(
    mask_root: Path, image_names: set[str], class_names: tuple[str, ...]
) -> dict[str, int]:
    """Count available class masks and reject masks for unknown images."""
    counts: dict[str, int] = {}
    for class_name in class_names:
        directory = mask_root / class_name
        paths = (
            {path.name for path in directory.iterdir() if path.is_file()}
            if directory.exists()
            else set()
        )
        unknown = sorted(paths - image_names)
        if unknown:
            raise ValueError(
                f"{class_name} contains masks for images outside the split source (first 5): "
                f"{unknown[:5]}"
            )
        counts[class_name] = len(paths)
    return counts


def main() -> None:
    """Run split, source-image, group-disjointness, and mask audits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--image-dir", default="dataset/source/images")
    parser.add_argument("--mask-root", default="dataset/source/masks")
    args = parser.parse_args()

    split_path = Path(args.split)
    image_dir = Path(args.image_dir)
    mask_root = Path(args.mask_root)
    payload = _load_split(split_path)
    source_names = {path.name for path in image_dir.iterdir() if path.is_file()}
    if not source_names:
        raise ValueError(f"no source images found: {image_dir}")

    train = set(payload["train"])
    val = set(payload["val"])
    if len(train) != len(payload["train"]) or len(val) != len(payload["val"]):
        raise ValueError("duplicate image names in train or val")
    if train & val:
        raise ValueError(f"image overlap between train and val: {sorted(train & val)[:5]}")
    selected = train | val
    missing = sorted(selected - source_names)
    if missing:
        raise ValueError(f"split references missing source images (first 5): {missing[:5]}")
    unassigned = sorted(source_names - selected)
    if unassigned:
        raise ValueError(f"source images absent from split (first 5): {unassigned[:5]}")

    assignment = _group_map(payload, selected)
    train_groups = {assignment[name] for name in train}
    val_groups = {assignment[name] for name in val}
    overlap = train_groups & val_groups
    if overlap:
        raise ValueError(f"group overlap between train and val: {sorted(overlap)[:5]}")

    mask_counts = _check_masks(mask_root, source_names, ("zp", "te", "bl", "icm"))
    print(f"OK: {split_path}")
    print(
        f"images={len(source_names)} train={len(train)} val={len(val)} groups={len(set(assignment.values()))}"
    )
    print(f"mask_files={mask_counts}")
    print("group_disjoint=True; split files were not modified")


if __name__ == "__main__":
    main()
