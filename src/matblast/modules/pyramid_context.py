from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidContext(nn.Module):
    """Fuse local, dilated, and global context at one feature resolution.

    Args:
        channels: Number of input and output channels.
        dilations: Dilation rates for depthwise local branches.
        reduction: Channel reduction used inside each branch.

    Input:
        x: (B, C, H, W) bottleneck feature map.

    Returns:
        (B, C, H, W) context-enhanced feature map.
    """

    def __init__(
        self,
        channels: int,
        dilations: Iterable[int] = (1, 2, 4, 8),
        reduction: int = 4,
    ) -> None:
        super().__init__()
        rates = tuple(int(rate) for rate in dilations)
        if not rates or any(rate < 1 for rate in rates):
            raise ValueError("dilations must contain positive integers")
        if reduction < 1:
            raise ValueError("reduction must be positive")

        inner = max(channels // reduction, 8)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, inner, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=False),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        inner,
                        inner,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        groups=inner,
                        bias=False,
                    ),
                    nn.BatchNorm2d(inner),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(inner, inner, kernel_size=1, bias=False),
                )
                for rate in rates
            ]
        )
        self.global_projection = nn.Conv2d(inner, inner, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(inner * (len(rates) + 1), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance bottleneck features without changing their spatial shape."""
        reduced = self.reduce(x)
        local = [branch(reduced) for branch in self.branches]
        pooled = F.adaptive_avg_pool2d(reduced, output_size=1)
        global_context = self.global_projection(pooled)
        global_context = F.interpolate(
            global_context, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        fused = self.fuse(torch.cat((*local, global_context), dim=1))
        return x + self.residual_scale.tanh() * fused
