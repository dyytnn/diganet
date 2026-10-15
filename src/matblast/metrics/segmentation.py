"""Per-image segmentation metrics with absent-GT classes excluded by NaN."""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage
import torch


LOGGER = logging.getLogger(__name__)


def per_image_scores(
    logits: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Dice and IoU for every image and class.

    Args:
        logits: (B, K, H, W) class logits.
        target: (B, H, W) integer masks.

    Returns:
        Tuple of Dice and IoU tensors, each shaped (B, K). A class is NaN for
        an image when it is absent from that image's ground truth.
    """
    prediction = logits.argmax(1)
    shape = (target.shape[0], logits.shape[1])
    dice = torch.full(shape, float("nan"), dtype=logits.dtype, device=logits.device)
    iou = torch.full_like(dice, float("nan"))
    for row in range(target.shape[0]):
        for class_id in range(logits.shape[1]):
            truth = target[row] == class_id
            if not bool(truth.any()):
                continue
            predicted = prediction[row] == class_id
            intersection = torch.logical_and(predicted, truth).sum()
            dice[row, class_id] = 2.0 * intersection / (predicted.sum() + truth.sum())
            iou[row, class_id] = intersection / torch.logical_or(predicted, truth).sum()
    return dice, iou


def summarize_scores(
    dice: torch.Tensor, iou: torch.Tensor, include_background: bool = False
) -> dict[str, float]:
    """Aggregate per-image scores without giving unequal weight to batches.

    Args:
        dice: (N, K) per-image Dice values.
        iou: (N, K) per-image IoU values.
        include_background: Whether macro metrics include class 0.

    Returns:
        Per-class and macro metric dictionary. NaN classes remain excluded from
        the corresponding ``nanmean`` aggregation.
    """
    if dice.ndim != 2 or iou.shape != dice.shape:
        raise ValueError("dice and iou must have identical shape (N, K)")
    dice_per_class = torch.nanmean(dice, dim=0)
    iou_per_class = torch.nanmean(iou, dim=0)
    start = 0 if include_background else 1

    def _macro(values: torch.Tensor, start_index: int) -> float:
        per_image = torch.nanmean(values[:, start_index:], dim=1)
        return float(torch.nanmean(per_image).item())

    return {
        "dice_bg": float(dice_per_class[0].item()),
        "iou_bg": float(iou_per_class[0].item()),
        **{
            f"dice_class_{index}": float(value.item())
            for index, value in enumerate(dice_per_class[1:], 1)
        },
        **{
            f"iou_class_{index}": float(value.item())
            for index, value in enumerate(iou_per_class[1:], 1)
        },
        "mdice_fg": _macro(dice, 1),
        "miou_fg": _macro(iou, 1),
        "mdice": _macro(dice, start),
        "miou": _macro(iou, start),
    }


def multiclass_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> list[float]:
    """Return per-class Dice with classes absent from GT ignored by NaN."""
    del eps
    dice, _ = per_image_scores(logits, target)
    return [float(value.item()) for value in torch.nanmean(dice, dim=0)]


def segmentation_metrics(
    logits: torch.Tensor, target: torch.Tensor, include_background: bool = False
) -> dict[str, float]:
    """Compute NaN-safe, per-image macro Dice and IoU.

    Args:
        logits: (B, K, H, W) class logits.
        target: (B, H, W) integer masks.
        include_background: Whether macro metrics include class 0.

    Returns:
        Per-class and macro Dice/IoU metrics.
    """
    dice, iou = per_image_scores(logits, target)
    return summarize_scores(dice, iou, include_background)


def hd95(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple[float, float] = (1.0, 1.0)) -> float:
    """Compute the 95th percentile symmetric Hausdorff distance.

    Args:
        pred_mask: Boolean mask with shape ``(H, W)``.
        gt_mask: Boolean mask with shape ``(H, W)``.
        spacing: Pixel spacing ``(row, column)``.

    Returns:
        Symmetric HD95 in spacing units, or infinity when either mask is empty.
    """
    pred = np.asarray(pred_mask, dtype=bool)
    truth = np.asarray(gt_mask, dtype=bool)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("pred_mask and gt_mask must have identical 2-D shapes")
    if not pred.any() or not truth.any():
        LOGGER.warning("HD95 is infinite because one mask is empty")
        return float("inf")
    pred_border = pred ^ ndimage.binary_erosion(pred)
    truth_border = truth ^ ndimage.binary_erosion(truth)
    pred_distance = ndimage.distance_transform_edt(~pred_border, sampling=spacing)
    truth_distance = ndimage.distance_transform_edt(~truth_border, sampling=spacing)
    distances = np.concatenate((pred_distance[truth_border], truth_distance[pred_border]))
    return float(np.percentile(distances, 95))


def boundary_f(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: int = 2) -> float:
    """Compute boundary F1 with a pixel tolerance.

    Args:
        pred_mask: Boolean mask with shape ``(H, W)``.
        gt_mask: Boolean mask with shape ``(H, W)``.
        tolerance: Matching tolerance in pixels.

    Returns:
        Boundary F1 score. The tolerance is measured in pixels.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    pred = np.asarray(pred_mask, dtype=bool)
    truth = np.asarray(gt_mask, dtype=bool)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("pred_mask and gt_mask must have identical 2-D shapes")
    pred_border = pred ^ ndimage.binary_erosion(pred)
    truth_border = truth ^ ndimage.binary_erosion(truth)
    if not pred_border.any() and not truth_border.any():
        return 1.0
    if not pred_border.any() or not truth_border.any():
        return 0.0
    pred_distance = ndimage.distance_transform_edt(~pred_border)
    truth_distance = ndimage.distance_transform_edt(~truth_border)
    precision = float((pred_distance[truth_border] <= tolerance).mean())
    recall = float((truth_distance[pred_border] <= tolerance).mean())
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def betti_error(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, int]:
    """Report absolute connected-component and hole-count errors.

    Args:
        pred_mask: Boolean mask with shape ``(H, W)``.
        gt_mask: Boolean mask with shape ``(H, W)``.

    Returns:
        Dictionary containing component, hole, and total Betti errors.
    """
    pred = np.asarray(pred_mask, dtype=bool)
    truth = np.asarray(gt_mask, dtype=bool)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("pred_mask and gt_mask must have identical 2-D shapes")

    def counts(mask: np.ndarray) -> tuple[int, int]:
        components = int(ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))[1])
        background_components = int(
            ndimage.label(~mask, structure=np.ones((3, 3), dtype=np.uint8))[1]
        )
        holes = max(0, background_components - 1)
        return components, holes

    pred_components, pred_holes = counts(pred)
    truth_components, truth_holes = counts(truth)
    component_error = abs(pred_components - truth_components)
    hole_error = abs(pred_holes - truth_holes)
    return {"components": component_error, "holes": hole_error, "total": component_error + hole_error}


def morphometry(mask: np.ndarray) -> dict[str, float]:
    """Compute simple blastocyst morphology measures from a 5-class mask.

    Args:
        mask: Integer class mask with shape ``(H, W)`` and classes
            ``BG=0, ZP=1, TE=2, BL=3, ICM=4``.

    Returns:
        BL area, mean ZP thickness, ICM area, ICM/TE ratio, and TE continuity.
    """
    labels = np.asarray(mask)
    if labels.ndim != 2:
        raise ValueError("mask must have shape (H, W)")
    zp = labels == 1
    te = labels == 2
    bl = labels == 3
    icm = labels == 4
    zp_thickness = float(ndimage.distance_transform_edt(zp).mean() * 2.0) if zp.any() else float("nan")
    te_area = int(te.sum())
    largest_te = 0
    if te_area:
        component_sizes = np.bincount(ndimage.label(te, structure=np.ones((3, 3), dtype=np.uint8))[0].ravel())
        largest_te = int(component_sizes[1:].max()) if len(component_sizes) > 1 else 0
    return {
        "bl_area": float(bl.sum()),
        "zp_thickness": zp_thickness,
        "icm_area": float(icm.sum()),
        "icm_te_ratio": float(icm.sum() / te_area) if te_area else float("nan"),
        "te_continuity": float(largest_te / te_area) if te_area else float("nan"),
    }


def ece(probs: np.ndarray, gt: np.ndarray, bins: int = 15) -> float:
    """Compute expected calibration error from multiclass probabilities.

    Args:
        probs: Probabilities with shape ``(K, H, W)``.
        gt: Integer ground truth with shape ``(H, W)``.
        bins: Number of confidence bins.

    Returns:
        Expected calibration error in ``[0, 1]``.
    """
    probabilities = np.asarray(probs, dtype=np.float64)
    truth = np.asarray(gt)
    if probabilities.ndim != 3 or truth.shape != probabilities.shape[1:]:
        raise ValueError("probs must be (K, H, W) and gt must be (H, W)")
    if bins <= 0:
        raise ValueError("bins must be positive")
    confidence = probabilities.max(axis=0).ravel()
    prediction = probabilities.argmax(axis=0).ravel()
    correct = (prediction == truth.ravel()).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        selected = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if selected.any():
            error += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    return float(error)
