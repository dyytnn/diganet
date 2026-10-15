"""Native-resolution annotation comparison and grouped descriptive uncertainty."""

from __future__ import annotations

import numpy as np

from matblast.metrics.segmentation import boundary_f, hd95


def annotation_scores(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compare independent Boolean (H, W) class masks; absent reference yields NaN.

    An unresolved extracted mask must be excluded by the caller, not interpreted
    as an intentional empty annotation. HD95 uses pooled symmetric boundary
    distances, Euclidean spacing (1, 1); Boundary-F tolerances are native pixels.
    """
    if candidate.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("Masks must have identical 2-D native shape")
    if candidate.dtype != bool or reference.dtype != bool:
        raise ValueError("Expected Boolean masks")
    names = ("iou", "dice", "boundary_f1", "boundary_f2", "boundary_f3", "hd95")
    if not reference.any():
        return {key: float("nan") for key in names}
    intersection = int((candidate & reference).sum())
    return {
        "iou": intersection / int((candidate | reference).sum()),
        "dice": 2 * intersection / int(candidate.sum() + reference.sum()),
        **{
            f"boundary_f{tol}": boundary_f(candidate, reference, tolerance=tol) for tol in (1, 2, 3)
        },
        "hd95": hd95(candidate, reference, spacing=(1.0, 1.0)),
    }


def grouped_descriptive_summary(
    values: np.ndarray,
    groups: list[str],
    *,
    seed: int = 7,
    bootstrap_samples: int = 5000,
) -> dict:
    """Summarize (N,) scores with whole-filename-group bootstrap of the image mean.

    SD describes between-image variation, not training-seed variation. NaNs
    represent unavailable reference classes. Infinities are counted explicitly
    and never silently discarded from the mean. Groups are frozen split IDs,
    not independently verified patient identities.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(groups) != len(values) or bootstrap_samples < 1:
        raise ValueError("Invalid values/groups/bootstrap count")
    valid = ~np.isnan(values)
    xs = values[valid]
    labels = np.asarray(groups, dtype=str)[valid]
    unique = np.unique(labels)
    result = {
        "n_images": int(xs.size),
        "n_groups": int(len(unique)),
        "n_seeds": 0,
        "n_nan": int(np.isnan(values).sum()),
        "n_infinite": int(np.isinf(xs).sum()),
        "mean": float(xs.mean()) if xs.size else float("nan"),
        "std": float(xs.std(ddof=1)) if xs.size > 1 and np.isfinite(xs).all() else float("nan"),
        "ci95_low": float("nan"),
        "ci95_high": float("nan"),
    }
    if len(unique) < 2 or not np.isfinite(xs).all():
        return result
    sums = np.array([xs[labels == group].sum() for group in unique])
    counts = np.array([(labels == group).sum() for group in unique])
    indices = np.random.default_rng(seed).integers(
        0, len(unique), size=(bootstrap_samples, len(unique))
    )
    means = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    result["ci95_low"], result["ci95_high"] = map(float, np.percentile(means, [2.5, 97.5]))
    return result
