"""Decoder head container shared by the v2 backbone.

Reproduces the head ordering of :class:`~matblast.models.matblast_unet.MATBlastUNet`
(BMask branch -> boundary-guided refiner -> class context -> boundary head -> BPR ->
class-boundary head -> deep-supervision heads -> PointRend / segmentation head) for a
decoder with an arbitrary number of levels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.bmask_boundary import BMaskBoundaryBranch
from ..modules.boundary_patch_refinement import BoundaryProximityRefiner
from ..modules.boundary_refinement import BoundaryGuidedRefiner
from ..modules.point_rend import PointRendHead
from .matblast_unet import ClassContextRefiner


class DecoderHeads(nn.Module):
    """Segmentation, boundary, deep-supervision and PointRend heads for a decoder.

    Args:
        level_channels: Channels of the decoder features from finest (level 1,
            full resolution) to coarsest.
        num_classes: K.
        norm_layer: Normalisation constructor used by branches that need one.
        deep_supervision: Emit ``aux_logits_{l}`` for the selected levels.
        deep_supervision_levels: Levels (>= 2) to supervise; ``None`` = all coarser
            decoder levels ``2..len(level_channels)``.
        Remaining flags mirror the MATBlastUNet keyword arguments of the same name.

    Input:
        features: list of decoder tensors ``[d1, d2, ...]`` with ``d1`` at full
        resolution and each next level at half the previous resolution.

    Returns:
        dict with ``logits`` (B, K, H, W) plus auxiliary keys.
    """

    def __init__(
        self,
        level_channels: Sequence[int],
        num_classes: int,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
        *,
        deep_supervision: bool = False,
        deep_supervision_levels: Sequence[int] | None = None,
        class_boundary: bool = False,
        boundary_refinement: bool = False,
        boundary_hidden_channels: int | None = None,
        class_context: bool = False,
        bpr_refinement: bool = False,
        bpr_hidden_channels: int | None = None,
        bpr_proximity_kernel_size: int = 5,
        bmask_boundary_branch: bool = False,
        bmask_hidden_channels: int | None = None,
        bmask_fusion: str = "gate",
        pointrend: bool = False,
        pointrend_coarse_level: int = 2,
        pointrend_num_points: int = 2048,
        pointrend_oversample_ratio: float = 3.0,
        pointrend_importance_sample_ratio: float = 0.75,
        pointrend_mlp_hidden: int = 256,
        pointrend_mlp_layers: int = 3,
        pointrend_classes_of_interest: Sequence[int] = (1, 2, 4),
        pointrend_subdivision_steps: int = 2,
        pointrend_subdivision_num_points: int = 8192,
    ) -> None:
        super().__init__()
        channels = [int(c) for c in level_channels]
        if len(channels) < 2:
            raise ValueError("DecoderHeads needs at least two decoder levels")
        fine = channels[0]
        if bmask_boundary_branch and (class_boundary or boundary_refinement):
            raise ValueError(
                "bmask_boundary_branch is mutually exclusive with class_boundary and boundary_refinement"
            )
        self.bmask = (
            BMaskBoundaryBranch(
                fine, channels[1], num_classes, bmask_hidden_channels, bmask_fusion, norm_layer
            )
            if bmask_boundary_branch
            else None
        )
        self.boundary_refiner = (
            BoundaryGuidedRefiner(fine, num_classes, boundary_hidden_channels)
            if boundary_refinement
            else None
        )
        self.class_context = ClassContextRefiner(fine, num_classes) if class_context else None
        self.boundary_head = nn.Conv2d(fine, 1, kernel_size=1)
        self.bpr_refiner = (
            BoundaryProximityRefiner(fine, bpr_hidden_channels, bpr_proximity_kernel_size)
            if bpr_refinement
            else None
        )
        self.class_boundary_head = (
            nn.Conv2d(fine, num_classes, kernel_size=1)
            if class_boundary and not boundary_refinement
            else None
        )
        self.seg_head = nn.Conv2d(fine, num_classes, kernel_size=1)
        self.deep_supervision = deep_supervision
        levels = (
            list(range(2, len(channels) + 1))
            if deep_supervision_levels is None
            else [int(level) for level in deep_supervision_levels]
        )
        if deep_supervision and (
            not levels or any(level < 2 or level > len(channels) for level in levels)
        ):
            raise ValueError(
                f"deep_supervision_levels must be within 2..{len(channels)}, got {levels}"
            )
        self.deep_supervision_levels = levels if deep_supervision else []
        self.aux_heads = nn.ModuleDict(
            {
                str(level): nn.Conv2d(channels[level - 1], num_classes, 1)
                for level in self.deep_supervision_levels
            }
        )
        self.pointrend = None
        self.pointrend_coarse_level = int(pointrend_coarse_level)
        if pointrend:
            if not deep_supervision:
                raise ValueError("pointrend requires deep_supervision=True")
            if self.pointrend_coarse_level not in self.deep_supervision_levels:
                raise ValueError(
                    f"pointrend_coarse_level {pointrend_coarse_level} is not a supervised level "
                    f"{self.deep_supervision_levels}"
                )
            self.pointrend = PointRendHead(
                fine,
                num_classes,
                pointrend_num_points,
                pointrend_oversample_ratio,
                pointrend_importance_sample_ratio,
                pointrend_mlp_hidden,
                pointrend_mlp_layers,
                tuple(pointrend_classes_of_interest),
                pointrend_subdivision_steps,
                pointrend_subdivision_num_points,
            )

    def forward(self, features: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply all heads to decoder features ordered fine -> coarse."""
        d1 = features[0]
        aux: dict[str, torch.Tensor] = {}
        if self.bmask is not None:
            d1, aux["class_boundary"] = self.bmask(d1, features[1])
        if self.boundary_refiner is not None:
            d1, aux["class_boundary"] = self.boundary_refiner(d1)
        if self.class_context is not None:
            d1, aux["class_context"] = self.class_context(d1)
        aux["boundary"] = self.boundary_head(d1)
        if self.bpr_refiner is not None:
            d1, aux["bpr_proximity"] = self.bpr_refiner(d1, aux["boundary"])
        if self.class_boundary_head is not None:
            aux["class_boundary"] = self.class_boundary_head(d1)
        raw_aux: dict[int, torch.Tensor] = {}
        for level in self.deep_supervision_levels:
            raw = self.aux_heads[str(level)](features[level - 1])
            raw_aux[level] = raw
            aux[f"aux_logits_{level}"] = F.interpolate(
                raw, size=d1.shape[-2:], mode="bilinear", align_corners=False
            )
        if self.pointrend is not None:
            coarse = raw_aux[self.pointrend_coarse_level]
            if self.training:
                aux["point_logits"], aux["point_coords"] = self.pointrend.forward_train(d1, coarse)
                return {"logits": self.seg_head(d1), **aux}
            aux["dense_logits"] = self.seg_head(d1)
            refined = self.pointrend.forward_inference(d1, coarse, tuple(d1.shape[-2:]))
            return {"logits": refined, **aux}
        return {"logits": self.seg_head(d1), **aux}
