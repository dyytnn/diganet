"""Adaptive receptive-field feature blocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import register_module


@register_module("adaptive_receptive_field")
class AdaptiveReceptiveField(nn.Module):
    """Lightweight gated multi-dilation block inspired by D-LKA/DCN families.

    Shape:
        Input and output are ``(B, C, H, W)``.
    """

    def __init__(self, channels: int, dilations: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        if channels <= 0 or not dilations or any(d <= 0 for d in dilations):
            raise ValueError("channels and positive dilations are required")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 1, bias=False),
                )
                for dilation in dilations
            ]
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, len(dilations), 1),
            nn.Softmax(dim=1),
        )
        self.output_norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Mix dilation branches with an input-conditioned spatial gate.

        Args:
            x: Bottleneck feature map with shape ``(B, C, H, W)``.

        Returns:
            Adaptively aggregated feature map with shape ``(B, C, H, W)``.
        """
        weights = self.gate(x)
        features = torch.stack([branch(x) for branch in self.branches], dim=1)
        mixed = (features * weights.unsqueeze(2)).sum(dim=1)
        return x + self.output_norm(mixed)
