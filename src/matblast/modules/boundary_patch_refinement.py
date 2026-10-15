from __future__ import annotations

import torch
import torch.nn as nn

from . import register_module


@register_module("bpr_refinement")
class BoundaryProximityRefiner(nn.Module):
    """BPR-inspired proxy refinement gated by predicted-boundary proximity.

    This is an *approximation* of BPR / "Look Closer To Segment Better"
    (Tang et al.), not a faithful reproduction. The original method crops
    high-resolution patches around predicted boundaries, refines each crop with
    a separate network, and pastes the result back. Here the decoder feature map
    is already at full input resolution, so instead of literal cropping the
    refinement is applied densely and gated by a learned boundary-proximity
    mask: refinement strength is high near predicted boundaries and decays to
    approximately zero in region interiors, which preserves the method's core
    behaviour ("spend capacity on boundaries, leave the interior alone") without
    ragged per-image crops that would break batching and determinism.

    Args:
        channels: Number of decoder feature channels C.
        hidden_channels: Width of the refinement branch. Defaults to
            ``max(channels // 2, 8)``.
        proximity_kernel_size: Odd kernel size of the learned dilation applied to
            the boundary map to grow it into a proximity band.
        boundary_channels: Number of channels in the incoming boundary map.

    Input:
        features: (B, C, H, W) decoder features.
        boundary_logits: (B, boundary_channels, H, W) predicted boundary logits.

    Returns:
        A tuple containing refined features (B, C, H, W) and the boundary
        proximity mask (B, 1, H, W) in the range (0, 1).
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        proximity_kernel_size: int = 5,
        boundary_channels: int = 1,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if proximity_kernel_size % 2 == 0:
            raise ValueError(f"proximity_kernel_size must be odd, got {proximity_kernel_size}")
        hidden = hidden_channels or max(channels // 2, 8)
        self.proximity = nn.Sequential(
            nn.Conv2d(
                boundary_channels,
                1,
                kernel_size=proximity_kernel_size,
                padding=proximity_kernel_size // 2,
            ),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, features: torch.Tensor, boundary_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine decoder features inside the predicted boundary band."""
        proximity = self.proximity(boundary_logits)
        update = self.refine(features) * proximity
        refined = features + self.residual_scale.tanh() * update
        return refined, proximity
