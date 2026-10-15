"""2.5-D and token-level proxy blocks from the literature review."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from . import register_module


def _check_volume(x: Tensor, channels: int) -> None:
    if x.ndim != 5 or x.shape[2] != channels:
        raise ValueError(f"expected (B, T, {channels}, H, W), got {tuple(x.shape)}")


@register_module("csam")
class CSAM(nn.Module):
    """Cross-slice attention module for 2.5-D feature sequences.

    Args:
        channels: Feature channels per slice.
        heads: Number of attention heads.

    Shape:
        Input and output are ``(B, T, C, H, W)``. The temporal axis ``T``
        remains explicit and is never folded into the batch axis.
    """

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels <= 0 or heads <= 0 or channels % heads != 0:
            raise ValueError("channels must be positive and divisible by heads")
        self.channels = channels
        self.heads = heads
        self.norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.project = nn.Linear(channels, channels)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        """Exchange information between neighboring slices at each pixel."""
        _check_volume(x, self.channels)
        batch, time, channels, height, width = x.shape
        tokens = x.permute(0, 3, 4, 1, 2).reshape(batch * height * width, time, channels)
        tokens = self.norm(tokens)
        query = self.query(tokens).reshape(-1, time, self.heads, channels // self.heads).transpose(1, 2)
        key = self.key(tokens).reshape(-1, time, self.heads, channels // self.heads).transpose(1, 2)
        value = self.value(tokens).reshape(-1, time, self.heads, channels // self.heads).transpose(1, 2)
        attention = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(channels // self.heads), -1)
        mixed = torch.matmul(attention, value).transpose(1, 2).reshape(-1, time, channels)
        mixed = self.project(mixed).reshape(batch, height, width, time, channels)
        mixed = mixed.permute(0, 3, 4, 1, 2)
        return x + self.residual_scale.tanh() * mixed


@register_module("bio2vol_csdha")
class Bio2VolCSDHA(nn.Module):
    """Cross-slice dual-head attention proxy for Bio2Vol adaptation.

    One head models appearance/content and the other models edge-sensitive
    features. Both heads attend along ``T`` independently, then are fused.

    Shape:
        Input and output are ``(B, T, C, H, W)``.
    """

    def __init__(self, channels: int, heads: int = 2) -> None:
        super().__init__()
        if channels <= 0 or heads <= 0 or channels % heads != 0:
            raise ValueError("channels must be positive and divisible by heads")
        self.channels = channels
        self.heads = heads
        self.content = CSAM(channels, heads)
        self.edge = CSAM(channels, heads)
        self.edge_filter = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), groups=channels)
        self.fuse = nn.Conv3d(channels * 2, channels, 1)
        self.gate = nn.Conv3d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Fuse content and edge cross-slice attention without flattening T."""
        _check_volume(x, self.channels)
        content = self.content(x)
        edge_input = x.permute(0, 2, 1, 3, 4)
        edge_features = self.edge_filter(edge_input).permute(0, 2, 1, 3, 4)
        edge = self.edge(edge_features)
        fused = self.fuse(torch.cat((content, edge), dim=2).permute(0, 2, 1, 3, 4))
        fused = fused.permute(0, 2, 1, 3, 4)
        return x + torch.sigmoid(self.gate(fused.permute(0, 2, 1, 3, 4))).permute(0, 2, 1, 3, 4) * fused


@register_module("segmote")
class SegMoTE(nn.Module):
    """Token-level mixture-of-experts adapter for a mask-decoder interface.

    Args:
        channels: Token embedding dimension.
        experts: Number of expert MLPs.
        expansion: Expert hidden expansion ratio.

    Shape:
        Input and output are ``(B, N, D)`` token sequences.
    """

    def __init__(self, channels: int, experts: int = 4, expansion: int = 2) -> None:
        super().__init__()
        if channels <= 0 or experts <= 0 or expansion <= 0:
            raise ValueError("channels, experts, and expansion must be positive")
        hidden = channels * expansion
        self.channels = channels
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)) for _ in range(experts)]
        )
        self.router = nn.Linear(channels, experts)
        self.project = nn.Linear(channels, channels)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, tokens: Tensor) -> Tensor:
        """Route each mask-decoder token through a learned expert mixture."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.channels:
            raise ValueError(f"expected (B, N, {self.channels}), got {tuple(tokens.shape)}")
        weights = self.router(tokens).softmax(-1)
        mixed = sum(expert(tokens) * weights[..., index : index + 1] for index, expert in enumerate(self.experts))
        return tokens + self.residual_scale.tanh() * self.project(mixed)


__all__ = ["Bio2VolCSDHA", "CSAM", "SegMoTE"]
