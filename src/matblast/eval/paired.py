from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.stats import wilcoxon

from matblast.metrics import per_image_scores


def per_image_iou(logits: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Compute per-image class IoU with GT-absent classes set to NaN.

    Args:
        logits: (B, K, H, W) class logits.
        target: (B, H, W) integer masks.

    Returns:
        Array of shape (B, K). A class is NaN for an image when it is absent
        from that image's ground truth.
    """
    _, iou = per_image_scores(logits, target)
    return iou.cpu().numpy().astype(np.float64, copy=False)


def paired_summary(
    values_a: np.ndarray,
    values_b: np.ndarray,
    bootstrap_samples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize paired per-image metric differences.

    Args:
        values_a: (N,) metric values for the reference model.
        values_b: (N,) metric values for the candidate model.
        bootstrap_samples: Number of paired bootstrap resamples.
        seed: Bootstrap RNG seed.

    Returns:
        Mean values, mean delta (B - A), percentile bootstrap CI, and a
        one-sided Wilcoxon p-value testing whether B is better than A.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired metric arrays must both have shape (N,)")
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size == 0:
        return {
            "n": 0,
            "mean_a": float("nan"),
            "mean_b": float("nan"),
            "delta": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "p_greater": float("nan"),
        }

    delta = b - a
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(max(1, bootstrap_samples), delta.size))
    bootstrap_means = delta[indices].mean(axis=1)
    try:
        p_value = float(wilcoxon(delta, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        p_value = 1.0
    return {
        "n": int(delta.size),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta": float(delta.mean()),
        "ci95": [
            float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)),
        ],
        "p_greater": p_value,
    }


def multiseed_paired_summary(
    values_a: np.ndarray,
    values_b: np.ndarray,
    bootstrap_samples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize a paired metric over repeated training seeds and images.

    The outer axis is the independently-trained model seed and the inner axis
    is the fixed evaluation image. Confidence intervals resample both axes, so
    the result reflects data-set and initialization variability. The exact
    seed-level Wilcoxon p-value is reported separately; with only three seeds
    it cannot by itself reach 0.05 and must not be over-interpreted.

    Args:
        values_a: (S, N) reference scores, with NaN for absent GT classes.
        values_b: (S, N) candidate scores, aligned to ``values_a``.
        bootstrap_samples: Number of hierarchical bootstrap replicates.
        seed: Bootstrap RNG seed.

    Returns:
        Aggregate means, seed-level mean and standard deviation, hierarchical
        percentile CI for B - A, plus complementary one-sided Wilcoxon tests
        over seed means and per-image means. The latter averages predictions
        across independently trained seeds before testing fixed validation
        images.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("multi-seed arrays must both have shape (S, N)")

    paired = np.isfinite(a) & np.isfinite(b)
    delta = np.where(paired, b - a, np.nan)
    n_seeds, n_images = delta.shape
    seed_counts = np.isfinite(delta).sum(axis=1)
    seed_delta = np.divide(
        np.nansum(delta, axis=1),
        seed_counts,
        out=np.full(n_seeds, np.nan, dtype=np.float64),
        where=seed_counts > 0,
    )
    valid_seed_delta = seed_delta[np.isfinite(seed_delta)]
    if not paired.any():
        return {
            "n_seeds": int(n_seeds),
            "n_images": int(n_images),
            "n_paired_scores": 0,
            "mean_a": float("nan"),
            "mean_b": float("nan"),
            "delta": float("nan"),
            "seed_delta_mean": float("nan"),
            "seed_delta_std": float("nan"),
            "seed_deltas": [],
            "ci95_hierarchical": [float("nan"), float("nan")],
            "p_seed_wilcoxon_greater": float("nan"),
            "n_paired_images": 0,
            "p_image_wilcoxon_greater": float("nan"),
        }

    rng = np.random.default_rng(seed)
    samples = max(1, bootstrap_samples)
    sampled_seeds = rng.integers(0, n_seeds, size=(samples, n_seeds))
    sampled_images = rng.integers(0, n_images, size=(samples, n_images))
    boot_delta = delta[sampled_seeds[:, :, np.newaxis], sampled_images[:, np.newaxis, :]]
    bootstrap_counts = np.isfinite(boot_delta).sum(axis=(1, 2))
    bootstrap_means = np.divide(
        np.nansum(boot_delta, axis=(1, 2)),
        bootstrap_counts,
        out=np.full(samples, np.nan, dtype=np.float64),
        where=bootstrap_counts > 0,
    )
    bootstrap_means = bootstrap_means[np.isfinite(bootstrap_means)]
    try:
        p_value = float(
            wilcoxon(valid_seed_delta, alternative="greater", zero_method="wilcox").pvalue
        )
    except ValueError:
        p_value = 1.0
    image_counts = np.isfinite(delta).sum(axis=0)
    image_delta = np.divide(
        np.nansum(delta, axis=0),
        image_counts,
        out=np.full(n_images, np.nan, dtype=np.float64),
        where=image_counts > 0,
    )
    valid_image_delta = image_delta[np.isfinite(image_delta)]
    try:
        image_p_value = float(
            wilcoxon(valid_image_delta, alternative="greater", zero_method="wilcox").pvalue
        )
    except ValueError:
        image_p_value = 1.0

    return {
        "n_seeds": int(n_seeds),
        "n_images": int(n_images),
        "n_paired_scores": int(paired.sum()),
        "mean_a": float(np.nanmean(np.where(paired, a, np.nan))),
        "mean_b": float(np.nanmean(np.where(paired, b, np.nan))),
        "delta": float(np.nanmean(delta)),
        "seed_delta_mean": float(np.nanmean(seed_delta)),
        "seed_delta_std": float(np.nanstd(seed_delta, ddof=1))
        if valid_seed_delta.size > 1
        else 0.0,
        "seed_deltas": [float(value) for value in seed_delta],
        "ci95_hierarchical": [
            float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)),
        ],
        "p_seed_wilcoxon_greater": p_value,
        "n_paired_images": int(valid_image_delta.size),
        "p_image_wilcoxon_greater": image_p_value,
    }
