"""Leakage/structure tests for the frozen 5-fold patient-grouped split."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIVE = ROOT / "data/splits/source_grouped_5fold_v1.json"
BASE = ROOT / "data/splits/source_grouped_v1.json"


@pytest.fixture(scope="module")
def split():
    if not FIVE.exists():
        pytest.skip("5-fold split not present")
    return json.loads(FIVE.read_text()), json.loads(BASE.read_text())


def test_folds_partition_all_images_exactly_once(split):
    five, base = split
    members = [set(v) for v in five["fold_members"].values()]
    assert len(members) == 5
    union = set().union(*members)
    assert union == set(base["train"]) | set(base["val"])
    assert sum(len(m) for m in members) == len(union)


def test_fold0_is_the_historical_validation_set_and_groups_never_straddle(split):
    five, base = split
    assert set(five["fold_members"]["0"]) == set(base["val"])
    for images in five["groups"].values():
        owners = {k for k, m in five["fold_members"].items() if set(images) & set(m)}
        assert len(owners) == 1, images


def test_each_fold_train_val_are_complementary_and_disjoint(split):
    five, _ = split
    for k, f in five["folds"].items():
        assert not set(f["train"]) & set(f["val"])
        assert set(f["train"]) | set(f["val"]) == set().union(
            *[set(v) for v in five["fold_members"].values()]
        )
        assert set(f["val"]) == set(five["fold_members"][k])
    sizes = [len(five["fold_members"][str(k)]) for k in range(1, 5)]
    assert max(sizes) - min(sizes) <= 3, sizes
