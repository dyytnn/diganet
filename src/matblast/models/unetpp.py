"""U-Net++ baseline with nested dense skip connections."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _UpProject(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        return F.interpolate(self.projection(x), size=size, mode="bilinear", align_corners=False)


class UNetPlusPlus(nn.Module):
    """Compact U-Net++ segmentation baseline.

    Args:
        x: (B, C, H, W) normalized image.

    Returns:
        Dictionary with ``logits`` of shape (B, K, H, W) and ``boundary`` of
        shape (B, 1, H, W), where K is ``num_classes``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 24,
        num_classes: int = 5,
        dropout: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        c = base_channels
        self.pool = nn.MaxPool2d(2)
        self.x0_0 = _ConvBlock(in_channels, c)
        self.x1_0 = _ConvBlock(c, 2 * c)
        self.x2_0 = _ConvBlock(2 * c, 4 * c)
        self.x3_0 = _ConvBlock(4 * c, 8 * c)
        self.x4_0 = _ConvBlock(8 * c, 16 * c)
        self.up4_3 = _UpProject(16 * c, 8 * c)
        self.up3_2 = _UpProject(8 * c, 4 * c)
        self.up2_1 = _UpProject(4 * c, 2 * c)
        self.up1_0 = _UpProject(2 * c, c)
        self.x3_1 = _ConvBlock(16 * c, 8 * c)
        self.x2_1 = _ConvBlock(8 * c, 4 * c)
        self.x1_1 = _ConvBlock(4 * c, 2 * c)
        self.x0_1 = _ConvBlock(2 * c, c)
        self.x2_2 = _ConvBlock(12 * c, 4 * c)
        self.x1_2 = _ConvBlock(6 * c, 2 * c)
        self.x0_2 = _ConvBlock(3 * c, c)
        self.x1_3 = _ConvBlock(8 * c, 2 * c)
        self.x0_3 = _ConvBlock(4 * c, c)
        self.x0_4 = _ConvBlock(5 * c, c)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.seg_head = nn.Conv2d(c, num_classes, 1)
        self.boundary_head = nn.Conv2d(c, 1, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Segment one frame.

        Args:
            x: (B, C, H, W) normalized image.

        Returns:
            ``logits``: (B, K, H, W); ``boundary``: (B, 1, H, W).
        """
        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(self.pool(x0_0))
        x2_0 = self.x2_0(self.pool(x1_0))
        x3_0 = self.x3_0(self.pool(x2_0))
        x4_0 = self.x4_0(self.pool(x3_0))
        x3_1 = self.x3_1(torch.cat((x3_0, self.up4_3(x4_0, x3_0.shape[-2:])), dim=1))
        x2_1 = self.x2_1(torch.cat((x2_0, self.up3_2(x3_0, x2_0.shape[-2:])), dim=1))
        x1_1 = self.x1_1(torch.cat((x1_0, self.up2_1(x2_0, x1_0.shape[-2:])), dim=1))
        x0_1 = self.x0_1(torch.cat((x0_0, self.up1_0(x1_0, x0_0.shape[-2:])), dim=1))
        x2_2 = self.x2_2(
            torch.cat((x2_0, x2_1, self.up3_2(x3_1, x2_0.shape[-2:])), dim=1)
        )
        x1_2 = self.x1_2(
            torch.cat((x1_0, x1_1, self.up2_1(x2_1, x1_0.shape[-2:])), dim=1)
        )
        x0_2 = self.x0_2(torch.cat((x0_0, x0_1, self.up1_0(x1_1, x0_0.shape[-2:])), dim=1))
        x1_3 = self.x1_3(
            torch.cat((x1_0, x1_1, x1_2, self.up2_1(x2_2, x1_0.shape[-2:])), dim=1)
        )
        x0_3 = self.x0_3(torch.cat((x0_0, x0_1, x0_2, self.up1_0(x1_2, x0_0.shape[-2:])), dim=1))
        x0_4 = self.x0_4(
            torch.cat((x0_0, x0_1, x0_2, x0_3, self.up1_0(x1_3, x0_0.shape[-2:])), dim=1)
        )
        features = self.dropout(x0_4)
        return {"logits": self.seg_head(features), "boundary": self.boundary_head(features)}

