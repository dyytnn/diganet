"""ConvNeXt-style local representation block."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import register_module


@register_module("convnext_block")
class ConvNeXtBlock(nn.Module):
    """Apply depthwise large-kernel mixing and channel-wise feature scaling.

    Args:
        channels: Number of input and output channels.
        expansion: Pointwise hidden-channel expansion ratio.
        kernel_size: Odd depthwise-convolution kernel size.
        layer_scale_init: Initial residual branch scale.
        dropout: Dropout probability in the residual branch.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        kernel_size: int = 7,
        layer_scale_init: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0 or expansion <= 0:
            raise ValueError("channels and expansion must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, hidden)
        self.project = nn.Linear(hidden, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((channels,), layer_scale_init))

    def forward(self, x: Tensor) -> Tensor:
        """Transform a bottleneck feature map.

        Args:
            x: ``(B, C, H, W)`` feature map.

        Returns:
            ``(B, C, H, W)`` residual feature map.
        """
        residual = x
        x = self.depthwise(x).permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.project(self.dropout(self.activation(self.expand(x))))
        x = self.dropout(x) * self.layer_scale
        return residual + x.permute(0, 3, 1, 2)
