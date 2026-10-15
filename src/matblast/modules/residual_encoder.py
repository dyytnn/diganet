"""Residual encoder feature blocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import register_module


@register_module("residual_encoder_block")
class ResidualEncoderBlock(nn.Module):
    """Pre-activation residual block for a 2-D encoder bottleneck.

    Args:
        channels: Number of input and output channels.
        expansion: Hidden-channel multiplier for the residual branch.
        dropout: Dropout probability in the residual branch.

    Shape:
        Input and output are ``(B, C, H, W)``.
    """

    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        if channels <= 0 or expansion <= 0:
            raise ValueError("channels and expansion must be positive")
        hidden = channels * expansion
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, channels, 3, padding=1, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual encoding.

        Args:
            x: Bottleneck feature map with shape ``(B, C, H, W)``.

        Returns:
            Feature map with shape ``(B, C, H, W)``.
        """
        residual = self.conv1(self.activation(self.norm1(x)))
        residual = self.dropout(self.conv2(self.activation(self.norm2(residual))))
        return x + self.residual_scale * residual
