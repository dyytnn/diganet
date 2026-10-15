"""Segmentation-only documented reconstruction of Blasto-Net.

The paper describes a multitask EfficientNet-B3/UNet model with CBAM and an
edge-aware attention module. This implementation keeps the segmentation path
only because this repository has no grading or implantation labels.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b3


class _CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, channels, 1),
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x: Tensor) -> Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        max_value = F.adaptive_max_pool2d(x, 1)
        channel_gate = torch.sigmoid(self.channel(avg) + self.channel(max_value))
        refined = x * channel_gate
        spatial = torch.cat((refined.mean(1, keepdim=True), refined.amax(1, keepdim=True)), dim=1)
        return refined * torch.sigmoid(self.spatial(spatial))


class _EAAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.edge = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.edge(x) + x


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, 1)
        self.block = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )
        self.cbam = _CBAM(out_channels)
        self.eaam = _EAAM(out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(self.projection(x), size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.eaam(self.cbam(self.block(torch.cat((x, skip), dim=1))))


class BlastoNet(nn.Module):
    """Blasto-Net segmentation-path reconstruction using an EfficientNet-B3 encoder.

    Args:
        x: (B, C, H, W) normalized image.

    Returns:
        Dictionary with ``logits`` of shape (B, K, H, W) and ``boundary`` of
        shape (B, 1, H, W). Grading and implantation heads are intentionally
        absent because their labels are outside the segmentation dataset.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 5, **_: object) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("Blasto-Net reconstruction expects three input channels")
        encoder = efficientnet_b3(weights=None)
        self.encoder = encoder.features
        self.decode3 = _DecoderBlock(1536, 136, 256)
        self.decode2 = _DecoderBlock(256, 48, 128)
        self.decode1 = _DecoderBlock(128, 32, 64)
        self.decode0 = _DecoderBlock(64, 24, 32)
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
        )
        self.seg_head = nn.Conv2d(16, num_classes, 1)
        self.boundary_head = nn.Conv2d(16, 1, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Segment one frame.

        Args:
            x: (B, 3, H, W) normalized image.

        Returns:
            ``logits``: (B, K, H, W); ``boundary``: (B, 1, H, W).
        """
        features: list[Tensor] = []
        for index, block in enumerate(self.encoder):
            x = block(x)
            if index in (1, 2, 3, 5, 8):
                features.append(x)
        e0, e1, e2, e3, bottleneck = features
        decoded = self.decode3(bottleneck, e3)
        decoded = self.decode2(decoded, e2)
        decoded = self.decode1(decoded, e1)
        decoded = self.decode0(decoded, e0)
        decoded = F.interpolate(self.final(decoded), scale_factor=2, mode="bilinear", align_corners=False)
        return {"logits": self.seg_head(decoded), "boundary": self.boundary_head(decoded)}

