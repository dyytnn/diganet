from __future__ import annotations

import torch
import torch.nn as nn


class BoundaryGuidedRefiner(nn.Module):
    """Use class-boundary evidence to refine decoder features.

    Args:
        channels: Number of decoder feature channels.
        classes: Number of semantic classes.
        hidden_channels: Channels in the boundary predictor.

    Input:
        features: (B, C, H, W) decoder features.

    Returns:
        A tuple containing refined features (B, C, H, W) and class-boundary
        logits (B, K, H, W).
    """

    def __init__(self, channels: int, classes: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = hidden_channels or max(channels // 2, 8)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, classes, kernel_size=1),
        )
        self.feature_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(classes, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine features around predicted class boundaries."""
        boundary_logits = self.boundary_head(features)
        gate = self.boundary_gate(boundary_logits)
        update = self.feature_projection(features) * gate
        refined = features + self.residual_scale.tanh() * update
        return refined, boundary_logits
