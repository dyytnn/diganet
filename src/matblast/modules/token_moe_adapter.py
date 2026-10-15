"""Token-level mixture-of-experts adapters."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import register_module


@register_module("token_moe_adapter")
class TokenMoEAdapter(nn.Module):
    """Spatial-token mixture of pointwise experts for feature adaptation.

    Shape:
        Input and output are ``(B, C, H, W)``; routing is computed per token.
    """

    def __init__(self, channels: int, experts: int = 4, expansion: int = 2) -> None:
        super().__init__()
        if channels <= 0 or experts <= 0 or expansion <= 0:
            raise ValueError("channels, experts, and expansion must be positive")
        hidden = channels * expansion
        self.norm = nn.BatchNorm2d(channels)
        self.router = nn.Conv2d(channels, experts, 1)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, hidden, 1),
                    nn.GELU(),
                    nn.Conv2d(hidden, channels, 1),
                )
                for _ in range(experts)
            ]
        )
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Route every spatial token through a learned expert mixture.

        Args:
            x: Feature map with shape ``(B, C, H, W)``.

        Returns:
            Adapted feature map with shape ``(B, C, H, W)``.
        """
        normalized = self.norm(x)
        weights = torch.softmax(self.router(normalized), dim=1)
        expert_features = torch.stack([expert(normalized) for expert in self.experts], dim=1)
        mixed = (expert_features * weights.unsqueeze(2)).sum(dim=1)
        return x + self.projection(mixed)
