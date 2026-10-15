"""Attention U-Net baseline with decoder-to-skip attention gates."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .matblast_unet import ConvBlock


class AttentionGate(nn.Module):
    """Filter an encoder skip feature using a coarser decoder gating signal."""

    def __init__(self, gate_channels: int, skip_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.gate_projection = nn.Sequential(
            nn.Conv2d(gate_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )
        self.attention = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, gate: Tensor, skip: Tensor) -> Tensor:
        """Apply attention to a same-resolution skip map.

        Args:
            gate: (B, Cg, H, W) upsampled decoder feature.
            skip: (B, Cs, H, W) encoder skip feature.

        Returns:
            (B, Cs, H, W) attention-weighted skip feature.
        """
        return skip * self.attention(self.gate_projection(gate) + self.skip_projection(skip))


class AttentionUNet(nn.Module):
    """Canonical two-convolution Attention U-Net for 2D semantic segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        base_channels: int = 24,
        dropout: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )
        self.pool = nn.MaxPool2d(2)
        self.encoder1 = ConvBlock(channels, channels)
        self.encoder2 = ConvBlock(channels, channels * 2)
        self.encoder3 = ConvBlock(channels * 2, channels * 4)
        self.bottleneck = ConvBlock(channels * 4, channels * 8)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.up3 = nn.ConvTranspose2d(channels * 8, channels * 4, 2, stride=2)
        self.gate3 = AttentionGate(channels * 4, channels * 4, channels * 2)
        self.decoder3 = ConvBlock(channels * 8, channels * 4)
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
        self.gate2 = AttentionGate(channels * 2, channels * 2, channels)
        self.decoder2 = ConvBlock(channels * 4, channels * 2)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.gate1 = AttentionGate(channels, channels, max(channels // 2, 1))
        self.decoder1 = ConvBlock(channels * 2, channels)
        self.head = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Segment one image.

        Args:
            x: (B, C, H, W) input image.

        Returns:
            Dictionary with ``logits`` shaped (B, K, H, W).
        """
        x = self.stem(x)
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        bottleneck = self.dropout(self.bottleneck(self.pool(encoder3)))
        decoder3 = self.up3(bottleneck)
        decoder3 = self.decoder3(torch.cat((decoder3, self.gate3(decoder3, encoder3)), dim=1))
        decoder2 = self.up2(decoder3)
        decoder2 = self.decoder2(torch.cat((decoder2, self.gate2(decoder2, encoder2)), dim=1))
        decoder1 = self.up1(decoder2)
        decoder1 = self.decoder1(torch.cat((decoder1, self.gate1(decoder1, encoder1)), dim=1))
        return {"logits": self.head(decoder1)}
