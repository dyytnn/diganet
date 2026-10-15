"""Convolutional state-space-inspired feature mixing."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import register_module


@register_module("cnn_ssm_block")
class CNNSSMBlock(nn.Module):
    """Gated convolutional state-space-inspired bottleneck mixer.

    This compact 2-D variant uses depthwise local mixing and a pooled global
    state gate; it is an explicit ablation of the CNN--Mamba/SSM family, not a
    claim of reproducing a particular Mamba implementation.

    Shape:
        Input and output are ``(B, C, H, W)``.
    """

    def __init__(self, channels: int, state_ratio: int = 4) -> None:
        super().__init__()
        if channels <= 0 or state_ratio <= 0:
            raise ValueError("channels and state_ratio must be positive")
        state_channels = max(channels // state_ratio, 4)
        self.norm = nn.BatchNorm2d(channels)
        self.local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.state_in = nn.Conv2d(channels, state_channels, 1)
        self.state_out = nn.Conv2d(state_channels, channels, 1)
        self.gate = nn.Conv2d(channels, channels, 1)
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply local mixing modulated by a learned global state.

        Args:
            x: Bottleneck feature map with shape ``(B, C, H, W)``.

        Returns:
            Feature map with shape ``(B, C, H, W)``.
        """
        normalized = self.norm(x)
        local = self.local(normalized)
        state = self.state_out(self.state_in(normalized).mean(dim=(-2, -1), keepdim=True))
        gate = torch.sigmoid(self.gate(normalized) + state)
        return x + self.projection(torch.tanh(local) * gate)
