"""RefineMask-inspired multi-stage mask refinement head.

RefineMask (Zhang et al., 2021) refines instance masks over several stages, each
time upsampling the mask features and fusing fine-grained semantic features, with
a boundary-aware refinement that concentrates on uncertain pixels. This module is
a semantic-segmentation adaptation over the U-Net decoder pyramid (quarter, half
and full resolution): stage k upsamples the running mask features, fuses the
decoder features of the next finer level (optionally weighted by the uncertainty
of the previous prediction) and predicts logits. It is *inspired*, not a
reproduction of the instance-level head with its ROI features.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_module
from .point_rend import calculate_uncertainty


@register_module("refine_mask_head")
class RefineMaskHead(nn.Module):
    """Progressive coarse-to-fine mask refinement over decoder levels.

    Args:
        level_channels: Decoder channels from finest to coarsest, e.g. ``(c, 2c, 4c)``;
            the number of stages equals ``len(level_channels)``.
        num_classes: K.
        hidden_channels: Mask-feature width carried across stages.
        boundary_aware: Weight the fine-feature fusion by the previous stage's
            uncertainty (top-1/top-2 margin) so refinement focuses on boundaries.

    Input: decoder features ``[d1, d2, ..., dS]`` (fine to coarse).
    Returns:
        (final_logits (B, K, H, W), [stage logits at their native resolution, coarse to fine]).
    """

    def __init__(
        self,
        level_channels: Sequence[int],
        num_classes: int,
        hidden_channels: int = 32,
        boundary_aware: bool = True,
    ) -> None:
        super().__init__()
        channels = [int(c) for c in level_channels]
        if len(channels) < 2:
            raise ValueError("RefineMaskHead needs at least two decoder levels")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        self.boundary_aware = boundary_aware
        self.num_stages = len(channels)
        coarse = channels[-1]
        self.stem = nn.Sequential(
            nn.Conv2d(coarse, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=False),
        )
        self.fine_projections = nn.ModuleList(
            [
                nn.Conv2d(channels[i], hidden_channels, 1, bias=False)
                for i in range(self.num_stages - 2, -1, -1)
            ]
        )
        self.fusions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=False),
                )
                for _ in range(self.num_stages - 1)
            ]
        )
        self.predictors = nn.ModuleList(
            [nn.Conv2d(hidden_channels, num_classes, 1) for _ in range(self.num_stages)]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if len(features) != self.num_stages:
            raise ValueError(f"expected {self.num_stages} decoder levels, got {len(features)}")
        mask = self.stem(features[-1])
        stage_logits = [self.predictors[0](mask)]
        for step in range(self.num_stages - 1):
            fine = features[self.num_stages - 2 - step]
            mask = F.interpolate(mask, size=fine.shape[-2:], mode="bilinear", align_corners=False)
            fine_projected = self.fine_projections[step](fine)
            if self.boundary_aware:
                previous = F.interpolate(
                    stage_logits[-1], size=fine.shape[-2:], mode="bilinear", align_corners=False
                )
                # uncertainty is <= 0 (negative margin); map to a (0, 1] gate, 1 at the boundary
                gate = torch.exp(calculate_uncertainty(previous.detach()))
                fine_projected = fine_projected * gate
            mask = self.fusions[step](torch.cat((mask, fine_projected), dim=1))
            stage_logits.append(self.predictors[step + 1](mask))
        return stage_logits[-1], stage_logits
