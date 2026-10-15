from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_module

FUSION_MODES = ("gate", "concat")


@register_module("bmask_boundary_branch")
class BMaskBoundaryBranch(nn.Module):
    """BMask-style class-aware boundary branch with feature fusion.

    Inspired by BMask R-CNN (Cheng et al.): a dedicated boundary branch predicts
    per-class boundary logits from decoder features and its boundary features are
    fused back into the segmentation features. Here the branch reads the finest
    decoder level and the level below it; the returned class-boundary logits are
    supervised with the existing morphological-gradient target
    (``class_boundary_target``). This is a semantic-segmentation adaptation, not a
    reproduction of the instance-level BMask head.

    Args:
        fine_channels: Channels C1 of the finest decoder feature ``d1``.
        mid_channels: Channels C2 of the next-coarser decoder feature ``d2``.
        num_classes: K, number of classes including background.
        hidden_channels: Width of the boundary branch; defaults to ``max(C1 // 2, 8)``.
        fusion: ``"gate"`` (residual gated by boundary features) or ``"concat"``.
        norm_layer: Normalisation constructor taking a channel count.

    Input:
        fine: (B, C1, H, W) decoder features at full resolution.
        mid: (B, C2, H/2, W/2) decoder features one level coarser.

    Returns:
        A tuple of fused features (B, C1, H, W) and class-boundary logits (B, K, H, W).
    """

    def __init__(
        self,
        fine_channels: int,
        mid_channels: int,
        num_classes: int,
        hidden_channels: int | None = None,
        fusion: str = "gate",
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if fusion not in FUSION_MODES:
            raise ValueError(f"fusion must be one of {FUSION_MODES}, got {fusion!r}")
        hidden = hidden_channels or max(fine_channels // 2, 8)
        self.fusion = fusion
        self.mid_projection = nn.Conv2d(mid_channels, hidden, kernel_size=1, bias=False)
        self.fine_projection = nn.Conv2d(
            fine_channels, hidden, kernel_size=3, padding=1, bias=False
        )
        self.trunk = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            norm_layer(hidden),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            norm_layer(hidden),
            nn.ReLU(inplace=False),
        )
        self.boundary_logits = nn.Conv2d(hidden, num_classes, kernel_size=1)
        if fusion == "gate":
            self.feature_projection = nn.Conv2d(
                fine_channels, fine_channels, kernel_size=1, bias=False
            )
            self.gate = nn.Sequential(
                nn.Conv2d(hidden, fine_channels, kernel_size=1, bias=False), nn.Sigmoid()
            )
            self.residual_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.fuse = nn.Sequential(
                nn.Conv2d(fine_channels + hidden, fine_channels, kernel_size=1, bias=False),
                norm_layer(fine_channels),
                nn.ReLU(inplace=False),
            )

    def forward(self, fine: torch.Tensor, mid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict class boundaries and fuse boundary features into ``fine``."""
        mid_up = F.interpolate(
            self.mid_projection(mid), size=fine.shape[-2:], mode="bilinear", align_corners=False
        )
        boundary_features = self.trunk(self.fine_projection(fine) + mid_up)
        logits = self.boundary_logits(boundary_features)
        if self.fusion == "gate":
            update = self.feature_projection(fine) * self.gate(boundary_features)
            fused = fine + self.residual_scale.tanh() * update
        else:
            fused = self.fuse(torch.cat((fine, boundary_features), dim=1))
        return fused, logits
