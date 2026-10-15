"""Frequency-aware multi-scale feature refinement."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import register_module


@register_module("frequency_multiscale")
class FrequencyMultiScale(nn.Module):
    """Fuse low/high-frequency features over multiple receptive fields.

    Args:
        channels: Number of input and output channels.
        hidden_channels: Internal pointwise projection width.
        dilations: Depthwise convolution dilation rates for high-frequency maps.

    Shape:
        Input and output are ``(B, C, H, W)``.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        dilations: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if not dilations or any(rate < 1 for rate in dilations):
            raise ValueError("dilations must contain positive rates")
        hidden = hidden_channels or max(channels // 2, 8)
        self.low_pass = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.high_pass = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=rate,
                dilation=rate,
                groups=channels,
                bias=False,
            )
            for rate in dilations
        )
        branch_count = len(dilations) + 1
        self.project = nn.Sequential(
            nn.Conv2d(channels * branch_count, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        gate_hidden = max(channels // 4, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, gate_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, branch_count, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Refine one feature map.

        Args:
            x: ``(B, C, H, W)`` bottleneck feature map.

        Returns:
            ``(B, C, H, W)`` residual-refined feature map.
        """
        low = self.low_pass(x)
        high = [branch(x - low) for branch in self.high_pass]
        branches = [low, *high]
        weights = torch.softmax(self.gate(x), dim=1)
        weighted = torch.cat(
            [feature * weights[:, index : index + 1] for index, feature in enumerate(branches)],
            dim=1,
        )
        return x + self.project(weighted)
