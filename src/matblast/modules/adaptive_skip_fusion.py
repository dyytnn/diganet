"""Adaptive feature fusion for encoder-to-decoder skip connections."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import register_module


@register_module("adaptive_skip_fusion")
class AdaptiveSkipFusion(nn.Module):
    """Learn a spatial, channel-wise gate for an encoder skip feature."""

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        if gate_channels <= 0 or skip_channels <= 0:
            raise ValueError("gate_channels and skip_channels must be positive")
        hidden = hidden_channels or max(min(gate_channels, skip_channels) // 2, 4)
        if hidden <= 0:
            raise ValueError("hidden_channels must be positive")
        self.gate_projection = nn.Conv2d(gate_channels, hidden, 1, bias=False)
        self.skip_projection = nn.Conv2d(skip_channels, hidden, 1, bias=False)
        self.attention = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, skip_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, gate: Tensor, skip: Tensor) -> Tensor:
        """Fuse one decoder feature with its aligned encoder skip.

        Args:
            gate: (B, Cg, H, W) decoder feature at the target resolution.
            skip: (B, Cs, H, W) encoder skip feature at the same resolution.

        Returns:
            (B, Cg + Cs, H, W) concatenated decoder and gated skip features.
        """
        if gate.ndim != 4 or skip.ndim != 4:
            raise ValueError(f"expected 4D tensors, got {gate.shape} and {skip.shape}")
        if gate.shape[0] != skip.shape[0] or gate.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                f"gate and skip must be spatially aligned, got {gate.shape} and {skip.shape}"
            )
        weights = self.attention(self.gate_projection(gate) + self.skip_projection(skip))
        return torch.cat((gate, skip * weights), dim=1)
