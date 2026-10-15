from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.point_rend import sample_point_labels
from ..modules.segfix import IGNORE_INDEX as SEGFIX_IGNORE_INDEX
from ..modules.segfix import segfix_targets
from ..modules.contour_refinement import contour_target_radii
from .relation import soft_anatomy_relation_loss


def foreground_boundary_target(target: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """Binary foreground-boundary target used by the ``boundary`` head.

    Args:
        target: (B, H, W) integer class labels.
        threshold: Minimum |avg_pool3(fg) - fg| for a pixel to count as boundary.

    Returns:
        (B, 1, H, W) float tensor in {0, 1}.
    """
    fg = (target > 0).float().unsqueeze(1)
    edge = torch.abs(F.avg_pool2d(fg, 3, 1, 1) - fg)
    return (edge > threshold).float()


def class_boundary_target(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Per-class boundary target: 3x3 morphological gradient of the one-hot labels.

    Args:
        target: (B, H, W) integer class labels.
        num_classes: K, number of classes including background.

    Returns:
        (B, K, H, W) float tensor in {0, 1}.
    """
    onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    local_max = F.max_pool2d(onehot, 3, 1, 1)
    local_min = -F.max_pool2d(-onehot, 3, 1, 1)
    return (local_max - local_min > 0).float()


def overlap_loss(
    logits,
    target,
    beta: float | None = None,
    class_weights: list[float] | None = None,
    eps: float = 1e-6,
    include_background: bool = False,
    metric: str = "dice",
    adaptive_weights: list[float] | None = None,
    reduction: str = "batch",
):
    """Soft Dice / Jaccard / Tversky loss.

    Args:
        logits: (B, K, H, W) raw scores.
        target: (B, H, W) integer labels.
        reduction: ``"batch"`` pools statistics over the whole batch before the
            ratio (the historical behaviour); ``"image"`` computes one score per
            image and averages them.
    """
    if reduction not in ("batch", "image"):
        raise ValueError(f"reduction must be 'batch' or 'image', got {reduction!r}")
    probabilities = logits.softmax(1)
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3) if reduction == "batch" else (2, 3)
    true_positive = (probabilities * onehot).sum(dims)
    false_positive = (probabilities * (1 - onehot)).sum(dims)
    false_negative = ((1 - probabilities) * onehot).sum(dims)
    if metric == "jaccard":
        score = (true_positive + eps) / (true_positive + false_positive + false_negative + eps)
    elif beta is None:
        score = (2 * true_positive + eps) / (
            2 * true_positive + false_positive + false_negative + eps
        )
    else:
        score = (true_positive + eps) / (
            true_positive + (1 - beta) * false_positive + beta * false_negative + eps
        )
    weights = torch.ones(logits.shape[1], device=logits.device)
    if class_weights is not None:
        weights = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)
    start = 0 if include_background else 1
    selected_score = score[..., start:]
    selected_weights = weights[start:]
    if metric == "jaccard" and adaptive_weights:
        ranking_source = selected_score if reduction == "batch" else selected_score.mean(0)
        ranking = torch.argsort(ranking_source.detach())
        for rank, extra in enumerate(adaptive_weights[: ranking.numel()]):
            selected_weights = selected_weights.clone()
            selected_weights[ranking[rank]] += extra
    weighted = (selected_score * selected_weights / selected_weights.sum()).sum(-1)
    return 1 - weighted.mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=1.0,
        boundary_weight=0.0,
        relation_weight=0.0,
        class_weights=None,
        tversky_beta=None,
        label_smoothing=0.0,
        class_boundary_weight=0.0,
        overlap_include_background=False,
        overlap_metric="dice",
        adaptive_weights=None,
        relation_pairs=None,
        relation_kernel_size=5,
        aux_loss_weight=0.15,
        aux_ce_weight=1.0,
        aux_overlap_weight=1.0,
        overlap_reduction="batch",
        deep_supervision_weights=None,
        deep_supervision_normalize=True,
        point_loss_weight=0.0,
        refine_loss_weight=0.0,
        segfix_loss_weight=0.0,
        segfix_offset_px=2,
        contour_loss_weight=0.0,
        contour_classes_of_interest=(1, 2, 4),
        contour_ray_steps=64,
        contour_max_radius=0.75,
    ):
        """Segmentation loss with optional boundary, relation and deep-supervision terms.

        Args:
            overlap_reduction: ``"batch"`` (default, historical) or ``"image"``, see
                :func:`overlap_loss`.
            deep_supervision_weights: Per-level weights for ``aux_logits_2, aux_logits_3,
                ...`` (index 0 = level 2, the finest auxiliary head). ``None`` keeps the
                legacy behaviour: the two heads ``aux_logits_3``/``aux_logits_2`` share the
                flat ``aux_loss_weight`` and any other level raises.
            deep_supervision_normalize: Divide the weights by their sum when a list is given.
            point_loss_weight: Weight of the PointRend point-wise cross-entropy, applied when
                the model output carries ``point_logits``/``point_coords``.
            refine_loss_weight: Weight of the averaged CE + overlap loss over intermediate
                ``refine_logits_*`` outputs (RefineMask-style multi-stage supervision).
            segfix_loss_weight: Weight of the SegFix boundary BCE + direction CE applied to
                ``segfix_boundary`` / ``segfix_direction`` outputs.
            segfix_offset_px: Offset used to derive the direction target (match the model).
            contour_loss_weight: Weight of the smooth-L1 loss between predicted and ray-cast
                ground-truth polar radii (``contour_outer``/``contour_inner`` outputs); the
                contour_* settings must match the model's contour refiner.
        """
        super().__init__()
        if overlap_reduction not in ("batch", "image"):
            raise ValueError(
                f"overlap_reduction must be 'batch' or 'image', got {overlap_reduction!r}"
            )
        self.overlap_reduction = overlap_reduction
        self.point_loss_weight = point_loss_weight
        self.refine_loss_weight = refine_loss_weight
        self.segfix_loss_weight = segfix_loss_weight
        self.segfix_offset_px = segfix_offset_px
        self.contour_loss_weight = contour_loss_weight
        self.contour_classes_of_interest = tuple(int(c) for c in contour_classes_of_interest)
        self.contour_ray_steps = contour_ray_steps
        self.contour_max_radius = contour_max_radius
        if deep_supervision_weights is not None:
            weights_list = [float(w) for w in deep_supervision_weights]
            if not weights_list or any(w < 0 for w in weights_list):
                raise ValueError("deep_supervision_weights must be a non-empty list of >= 0")
            if deep_supervision_normalize:
                total_weight = sum(weights_list)
                if total_weight <= 0:
                    raise ValueError("deep_supervision_weights must not all be zero")
                weights_list = [w / total_weight for w in weights_list]
            self.deep_supervision_weights: list[float] | None = weights_list
        else:
            self.deep_supervision_weights = None
        self.weights = (ce_weight, dice_weight, boundary_weight, relation_weight)
        self.class_weights = class_weights
        self.tversky_beta = tversky_beta
        self.label_smoothing = label_smoothing
        self.class_boundary_weight = class_boundary_weight
        self.overlap_include_background = overlap_include_background
        self.overlap_metric = overlap_metric
        self.adaptive_weights = adaptive_weights
        self.relation_pairs = tuple(tuple(pair) for pair in (relation_pairs or ()))
        self.relation_kernel_size = relation_kernel_size
        self.aux_loss_weight = aux_loss_weight
        self.aux_ce_weight = aux_ce_weight
        self.aux_overlap_weight = aux_overlap_weight

    def _overlap(self, logits, target):
        return overlap_loss(
            logits,
            target,
            self.tversky_beta,
            self.class_weights,
            include_background=self.overlap_include_background,
            metric=self.overlap_metric,
            adaptive_weights=self.adaptive_weights,
            reduction=self.overlap_reduction,
        )

    def _aux_terms(self, output: dict) -> list[tuple[str, float]]:
        """Resolve which ``aux_logits_*`` keys are supervised and with what weight."""
        levels = sorted(
            int(key.rsplit("_", 1)[1])
            for key in output
            if key.startswith("aux_logits_") and key.rsplit("_", 1)[1].isdigit()
        )
        if self.deep_supervision_weights is None:
            extra = [level for level in levels if level not in (2, 3)]
            if extra:
                raise ValueError(
                    "output has auxiliary levels "
                    f"{extra} but loss.deep_supervision_weights is not set; the legacy "
                    "default only supervises aux_logits_3 and aux_logits_2"
                )
            return [(f"aux_logits_{level}", 1.0) for level in (3, 2) if level in levels]
        expected = list(range(2, 2 + len(self.deep_supervision_weights)))
        if levels != expected:
            raise ValueError(
                f"deep_supervision_weights has {len(self.deep_supervision_weights)} entries "
                f"(levels {expected}) but output provides levels {levels}"
            )
        return [
            (f"aux_logits_{level}", weight)
            for level, weight in zip(expected, self.deep_supervision_weights)
        ]

    def forward(self, output, target):
        logits = output["logits"] if isinstance(output, dict) else output
        weights = (
            None
            if self.class_weights is None
            else torch.as_tensor(self.class_weights, dtype=logits.dtype, device=logits.device)
        )
        ce = F.cross_entropy(logits, target, weight=weights, label_smoothing=self.label_smoothing)
        d = self._overlap(logits, target)
        aux_loss = torch.zeros((), device=target.device)
        if isinstance(output, dict):
            for key, level_weight in self._aux_terms(output):
                aux_logits = output[key]
                aux_ce = F.cross_entropy(
                    aux_logits, target, weight=weights, label_smoothing=self.label_smoothing
                )
                aux_dice = self._overlap(aux_logits, target)
                aux_loss = aux_loss + self.aux_loss_weight * level_weight * (
                    self.aux_ce_weight * aux_ce + self.aux_overlap_weight * aux_dice
                )
        bd = torch.zeros((), device=target.device)
        rel = torch.zeros((), device=target.device)
        if self.weights[3]:
            rel = soft_anatomy_relation_loss(
                logits, target, self.relation_pairs, self.relation_kernel_size
            )
        if isinstance(output, dict) and "boundary" in output and self.weights[2]:
            bd = F.binary_cross_entropy_with_logits(
                output["boundary"], foreground_boundary_target(target)
            )
        class_bd = torch.zeros((), device=target.device)
        if isinstance(output, dict) and "class_boundary" in output and self.class_boundary_weight:
            class_bd = F.binary_cross_entropy_with_logits(
                output["class_boundary"], class_boundary_target(target, logits.shape[1])
            )
        point = torch.zeros((), device=target.device)
        if isinstance(output, dict) and "point_logits" in output and self.point_loss_weight:
            point_target = sample_point_labels(target, output["point_coords"])
            point = F.cross_entropy(
                output["point_logits"],
                point_target,
                weight=weights,
                label_smoothing=self.label_smoothing,
            )
        refine = torch.zeros((), device=target.device)
        if isinstance(output, dict) and self.refine_loss_weight:
            refine_keys = sorted(key for key in output if key.startswith("refine_logits_"))
            for key in refine_keys:
                stage_ce = F.cross_entropy(
                    output[key], target, weight=weights, label_smoothing=self.label_smoothing
                )
                refine = refine + (
                    self.aux_ce_weight * stage_ce
                    + self.aux_overlap_weight * self._overlap(output[key], target)
                )
            if refine_keys:
                refine = refine / len(refine_keys)
        segfix = torch.zeros((), device=target.device)
        if isinstance(output, dict) and "segfix_boundary" in output and self.segfix_loss_weight:
            boundary_target, direction_target = segfix_targets(target, self.segfix_offset_px)
            segfix = F.binary_cross_entropy_with_logits(output["segfix_boundary"], boundary_target)
            if (direction_target != SEGFIX_IGNORE_INDEX).any():
                segfix = segfix + F.cross_entropy(
                    output["segfix_direction"], direction_target, ignore_index=SEGFIX_IGNORE_INDEX
                )
        contour = torch.zeros((), device=target.device)
        if isinstance(output, dict) and "contour_outer" in output and self.contour_loss_weight:
            count = output["contour_outer"].shape[-1]
            angles = torch.arange(
                count, device=target.device, dtype=output["contour_outer"].dtype
            ) * (2 * math.pi / count)
            gt_outer, gt_inner, present = contour_target_radii(
                target,
                output["contour_centers"],
                self.contour_classes_of_interest,
                angles,
                self.contour_ray_steps,
                self.contour_max_radius,
            )
            if present.any():
                per_class = F.smooth_l1_loss(
                    output["contour_outer"], gt_outer, reduction="none"
                ).mean(-1) + F.smooth_l1_loss(
                    output["contour_inner"], gt_inner, reduction="none"
                ).mean(-1)
                contour = (per_class * present).sum() / present.sum()
        total = (
            self.weights[0] * ce
            + self.weights[1] * d
            + self.weights[2] * bd
            + self.weights[3] * rel
            + aux_loss
            + self.class_boundary_weight * class_bd
            + self.point_loss_weight * point
            + self.refine_loss_weight * refine
            + self.segfix_loss_weight * segfix
            + self.contour_loss_weight * contour
        )
        return {
            "total_loss": total,
            "ce": ce.detach(),
            "dice": d.detach(),
            "boundary": bd.detach(),
            "class_boundary": class_bd.detach(),
            "aux": aux_loss.detach(),
            "point": point.detach(),
            "refine": refine.detach(),
            "segfix": segfix.detach(),
            "contour": contour.detach(),
        }
