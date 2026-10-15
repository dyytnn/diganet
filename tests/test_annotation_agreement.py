"""Hand-calculated binary agreement and group-aware descriptive uncertainty."""

import numpy as np
import pytest

from matblast.eval.annotation_agreement import annotation_scores, grouped_descriptive_summary


def test_overlap_on_four_by_four_mask() -> None:
    a = np.zeros((4, 4), bool)
    b = a.copy()
    a[1:3, 0:2] = True
    b[1:3, 1:3] = True
    values = annotation_scores(a, b)
    assert values["iou"] == pytest.approx(1 / 3)
    assert values["dice"] == pytest.approx(0.5)
    assert values["boundary_f1"] == 1
    assert values["hd95"] == 1
    assert values == annotation_scores(b, a)


def test_identity_and_absent_reference() -> None:
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    values = annotation_scores(mask, mask)
    assert values["iou"] == values["dice"] == values["boundary_f2"] == 1
    assert values["hd95"] == 0
    assert all(np.isnan(x) for x in annotation_scores(mask, np.zeros_like(mask)).values())


def test_intentionally_empty_candidate_differs_from_missing_reference() -> None:
    truth = np.zeros((4, 4), bool)
    truth[1, 1] = True
    values = annotation_scores(np.zeros_like(truth), truth)
    assert values["dice"] == values["iou"] == values["boundary_f2"] == 0
    assert np.isinf(values["hd95"])


def test_group_bootstrap_uses_whole_groups_and_image_mean() -> None:
    xs = np.array([0.0, 0.0, 1.0])
    result = grouped_descriptive_summary(xs, ["a", "a", "b"], seed=7, bootstrap_samples=1000)
    assert result["mean"] == pytest.approx(1 / 3)
    assert result["std"] == pytest.approx(np.std(xs, ddof=1))
    assert result["n_groups"] == 2 and result["n_images"] == 3 and result["n_seeds"] == 0
    assert result["ci95_low"] == 0 and result["ci95_high"] == 1
    assert result == grouped_descriptive_summary(
        xs, ["a", "a", "b"], seed=7, bootstrap_samples=1000
    )


def test_nan_excluded_but_infinite_not_hidden() -> None:
    result = grouped_descriptive_summary(np.array([np.nan, 0.0, 1.0]), ["x", "a", "b"])
    assert result["n_nan"] == 1 and result["mean"] == 0.5
    result = grouped_descriptive_summary(np.array([np.inf, 1.0]), ["a", "b"])
    assert result["n_infinite"] == 1 and np.isinf(result["mean"])
    assert np.isnan(result["ci95_low"])
    result = grouped_descriptive_summary(np.array([0.2]), ["a"])
    assert np.isnan(result["std"]) and np.isnan(result["ci95_low"])


def test_invalid_input_rejected() -> None:
    with pytest.raises(ValueError):
        annotation_scores(np.zeros((4, 4)), np.zeros((4, 4)))
    with pytest.raises(ValueError):
        grouped_descriptive_summary(np.zeros(2), ["a"])
