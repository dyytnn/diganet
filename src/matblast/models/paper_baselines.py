"""Faithful reimplementations of the three requested paper baselines.

The papers do not publish executable source or checkpoints.  These classes
therefore implement the architectural details that are specified in the
papers and expose every reconstruction choice through constructor arguments.
The associated configs document details that are not recoverable from print.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
    )


class _LearnableSinusoidalPosition(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(torch.linspace(1.0, 4.0, max(1, channels // 4)))
        self.projection = nn.Conv2d(4 * len(self.frequencies), channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add learnable sinusoidal coordinates to ``x``.

        Args:
            x: (B, C, H, W) feature map.

        Returns:
            (B, C, H, W) position-enriched feature map.
        """
        height, width = x.shape[-2:]
        device, dtype = x.device, x.dtype
        yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(1, 1, height, 1)
        xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, width)
        phase_x = xx * self.frequencies.to(dtype).view(1, -1, 1, 1) * math.pi
        phase_y = yy * self.frequencies.to(dtype).view(1, -1, 1, 1) * math.pi
        phase_x = phase_x.expand(-1, -1, height, -1)
        phase_y = phase_y.expand(-1, -1, -1, width)
        position = torch.cat((phase_x.sin(), phase_x.cos(), phase_y.sin(), phase_y.cos()), dim=1)
        return x + self.projection(position)


class _AxialFreeAttention(nn.Module):
    """Approximate the paper's angle-wise axial-free attention block.

    Args:
        channels: Feature channels.
        n_angles: Number of rotated attention views, reported as D in BTFormer.
        pos_encoding: Whether to use learnable sinusoidal relative coordinates.
    """

    def __init__(self, channels: int, n_angles: int = 4, pos_encoding: bool = True) -> None:
        super().__init__()
        if channels % 4:
            raise ValueError("BTFormer channels must be divisible by four")
        self.channels = channels
        self.n_angles = n_angles
        self.row_attention = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.column_attention = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.aggregate = nn.Conv2d(channels * n_angles, channels, 1, bias=False)
        self.angle_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, n_angles, 1))
        self.position = _LearnableSinusoidalPosition(channels) if pos_encoding else nn.Identity()
        self.norm = nn.BatchNorm2d(channels)

    def _axis_attention(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        rows = x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        rows, _ = self.row_attention(rows, rows, rows, need_weights=False)
        rows = rows.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        cols = rows.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        cols, _ = self.column_attention(cols, cols, cols, need_weights=False)
        return cols.reshape(batch, width, height, channels).permute(0, 3, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply D rotated row/column attention views.

        Args:
            x: (B, C, H, W) feature map.

        Returns:
            (B, C, H, W) attended feature map.
        """
        x = self.position(x)
        views = []
        for angle_index in range(self.n_angles):
            angle = angle_index * math.pi / max(1, self.n_angles)
            theta = (
                x.new_tensor(
                    [
                        [math.cos(angle), -math.sin(angle), 0.0],
                        [math.sin(angle), math.cos(angle), 0.0],
                    ]
                )
                .unsqueeze(0)
                .repeat(x.shape[0], 1, 1)
            )
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            rotated = F.grid_sample(
                x, grid, mode="bilinear", padding_mode="border", align_corners=False
            )
            attended = self._axis_attention(rotated)
            inverse = -angle
            inverse_theta = (
                x.new_tensor(
                    [
                        [math.cos(inverse), -math.sin(inverse), 0.0],
                        [math.sin(inverse), math.cos(inverse), 0.0],
                    ]
                )
                .unsqueeze(0)
                .repeat(x.shape[0], 1, 1)
            )
            inverse_grid = F.affine_grid(inverse_theta, x.shape, align_corners=False)
            views.append(
                F.grid_sample(
                    attended,
                    inverse_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
            )
        weights = torch.softmax(self.angle_gate(x), dim=1)
        fused = self.aggregate(torch.cat(views, dim=1))
        return self.norm(x + fused * weights.mean(dim=1, keepdim=True))


class BTFormer(nn.Module):
    """BTFormer-style encoder-decoder with angle-wise axial-free attention.

    Args:
        x: (B, C, H, W) grayscale input, normally C = 1.

    Returns:
        Dictionary containing ``logits`` with shape (B, K, H, W).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        base_channels: int = 16,
        n_angles: int = 4,
        pos_encoding: bool = True,
    ) -> None:
        super().__init__()
        c = base_channels
        self.stem = _conv_block(in_channels, c)
        self.enc2 = _conv_block(c, 2 * c)
        self.enc3 = nn.Sequential(
            _conv_block(2 * c, 4 * c), _AxialFreeAttention(4 * c, n_angles, pos_encoding)
        )
        self.enc4 = nn.Sequential(
            _conv_block(4 * c, 8 * c), _AxialFreeAttention(8 * c, n_angles, pos_encoding)
        )
        self.enc5 = nn.Sequential(
            _conv_block(8 * c, 8 * c), _AxialFreeAttention(8 * c, n_angles, pos_encoding)
        )
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(8 * c, 8 * c, 2, 2)
        self.dec4 = _conv_block(16 * c, 8 * c)
        self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, 2)
        self.dec3 = _conv_block(8 * c, 4 * c)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2)
        self.dec2 = _conv_block(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, 2)
        self.dec1 = _conv_block(2 * c, c)
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Segment one frame.

        Args:
            x: (B, C, H, W) normalized image.

        Returns:
            Dictionary with ``logits`` of shape (B, K, H, W).
        """
        e1 = self.stem(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        d4 = self.dec4(torch.cat((self.up4(e5), e4), dim=1))
        d3 = self.dec3(torch.cat((self.up3(d4), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return {"logits": self.head(d1)}


class _I2CContext(nn.Module):
    def __init__(self, channels: int, num_classes: int, relation_weights: Sequence[float]) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_features = nn.Conv2d(channels, channels * num_classes, 1, bias=False)
        self.intra_fuse = nn.Sequential(
            nn.Conv2d(channels * num_classes, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )
        self.relation_features = nn.Conv2d(channels, channels * (num_classes - 1), 1, bias=False)
        self.relation_fuse = nn.Sequential(
            nn.Conv2d(channels * (num_classes - 1), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )
        weights = torch.as_tensor(tuple(relation_weights), dtype=torch.float32)
        self.register_buffer("relation_weights", weights, persistent=False)

    def forward(self, features: Tensor, probabilities: Tensor) -> Tensor:
        """Apply IACCM and inside-out IRCCM context.

        Args:
            features: (B, C, H, W) high-level encoder features.
            probabilities: (B, K, H, W) class probabilities.

        Returns:
            (B, C, H, W) context-refined features.
        """
        batch, channels, height, width = features.shape
        class_features = self.class_features(features).reshape(
            batch, self.num_classes, channels, height, width
        )
        intra = (class_features * probabilities.unsqueeze(2)).reshape(batch, -1, height, width)
        intra = self.intra_fuse(intra)
        # Fixed ordering is anatomy-aware, but the relation is a soft feature gate.
        anatomy_order = [4, 3, 2, 1]
        relation_maps = []
        for relation_index in range(self.num_classes - 1):
            selected = anatomy_order[: relation_index + 1]
            cumulative = probabilities[:, selected].sum(dim=1, keepdim=True)
            relation_maps.append(cumulative * self.relation_weights[relation_index])
        relation = torch.cat(relation_maps, dim=1)
        relation_features = self.relation_features(features).reshape(
            batch, self.num_classes - 1, channels, height, width
        )
        relation_features = (relation_features * relation.unsqueeze(2)).reshape(
            batch, -1, height, width
        )
        return features + self.relation_fuse(relation_features) + intra


class I2CNet(nn.Module):
    """I2CNet-style intra/inter-class context network with deep supervision.

    Args:
        x: (B, C, H, W) normalized grayscale image.

    Returns:
        ``logits`` (B, K, H, W) and ``deep_logits`` at the five encoder scales.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        channels: Sequence[int] = (16, 24, 40, 64, 96),
        relation_weights: Sequence[float] = (1.0, 2.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError("I2CNet expects five encoder channel widths")
        self.blocks = nn.ModuleList()
        previous = in_channels
        for current in channels:
            self.blocks.append(_conv_block(previous, current))
            previous = current
        self.pool = nn.MaxPool2d(2)
        self.scale_heads = nn.ModuleList(nn.Conv2d(current, num_classes, 1) for current in channels)
        self.context = _I2CContext(channels[-1], num_classes, relation_weights)
        self.decode4 = _conv_block(channels[-1] + channels[-2], channels[-2])
        self.decode3 = _conv_block(channels[-2] + channels[-3], channels[-3])
        self.decode2 = _conv_block(channels[-3] + channels[-4], channels[-4])
        self.decode1 = _conv_block(channels[-4] + channels[-5], channels[-5])
        self.output_head = nn.Conv2d(channels[-5], num_classes, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor | list[Tensor]]:
        """Segment one frame with context and scale-wise supervision.

        Args:
            x: (B, C, H, W) normalized image.

        Returns:
            Dictionary with full-resolution ``logits`` and five ``deep_logits``.
        """
        features = []
        current = x
        for index, block in enumerate(self.blocks):
            current = block(current)
            features.append(current)
            if index < len(self.blocks) - 1:
                current = self.pool(current)
        probabilities = torch.softmax(self.scale_heads[-1](features[-1]), dim=1)
        context = self.context(features[-1], probabilities)
        d4 = self.decode4(
            torch.cat(
                (
                    F.interpolate(
                        context, size=features[-2].shape[-2:], mode="bilinear", align_corners=False
                    ),
                    features[-2],
                ),
                dim=1,
            )
        )
        d3 = self.decode3(
            torch.cat(
                (
                    F.interpolate(
                        d4, size=features[-3].shape[-2:], mode="bilinear", align_corners=False
                    ),
                    features[-3],
                ),
                dim=1,
            )
        )
        d2 = self.decode2(
            torch.cat(
                (
                    F.interpolate(
                        d3, size=features[-4].shape[-2:], mode="bilinear", align_corners=False
                    ),
                    features[-4],
                ),
                dim=1,
            )
        )
        d1 = self.decode1(
            torch.cat(
                (
                    F.interpolate(
                        d2, size=features[-5].shape[-2:], mode="bilinear", align_corners=False
                    ),
                    features[-5],
                ),
                dim=1,
            )
        )
        logits = self.output_head(d1)
        deep_logits = [self.scale_heads[index](feature) for index, feature in enumerate(features)]
        return {"logits": logits, "deep_logits": deep_logits}


def _gabor_kernel(size: int, sigma: float, wavelength: float, gamma: float, theta: float) -> Tensor:
    coordinate = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    x_theta = xx * math.cos(theta) + yy * math.sin(theta)
    y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
    gaussian = torch.exp(
        -(x_theta.square() + gamma * gamma * y_theta.square()) / (2 * sigma * sigma)
    )
    carrier = torch.cos(2 * math.pi * x_theta / wavelength)
    kernel = gaussian * carrier
    return kernel / kernel.abs().sum().clamp_min(1e-6)


class _MBIStem(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 9,
        gabor_sigma: float = 2.0,
        gabor_wavelength: float = 5.0,
        gabor_gamma: float = 0.5,
    ) -> None:
        super().__init__()
        kernels = torch.stack(
            (
                _gabor_kernel(kernel_size, gabor_sigma, gabor_wavelength, gabor_gamma, 0.0),
                _gabor_kernel(kernel_size, gabor_sigma, gabor_wavelength, gabor_gamma, math.pi / 2),
            )
        ).unsqueeze(1)
        self.register_buffer("gabor_kernels", kernels, persistent=False)
        self.projection = nn.Sequential(
            nn.Conv2d(4, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Construct intensity, LBP and two orientation-response channels.

        Args:
            x: (B, 1, H, W) normalized grayscale image.

        Returns:
            (B, C, H, W) fused MBI features.
        """
        intensity = x[:, :1]
        padded = F.pad(intensity, (1, 1, 1, 1), mode="reflect")
        center = padded[:, :, 1:-1, 1:-1]
        neighbors = [
            padded[:, :, dy : dy + intensity.shape[-2], dx : dx + intensity.shape[-1]]
            for dy, dx in ((0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (2, 1), (1, 0), (1, 2))
        ]
        lbp = torch.stack(
            [(neighbor >= center).to(intensity.dtype) for neighbor in neighbors], dim=0
        )
        lbp = (lbp * intensity.new_tensor([1, 2, 4, 8, 16, 32, 64, 128]).view(8, 1, 1, 1, 1)).sum(
            0
        ) / 255.0
        orientation = F.conv2d(
            intensity, self.gabor_kernels, padding=self.gabor_kernels.shape[-1] // 2
        )
        return self.projection(torch.cat((intensity, lbp, orientation), dim=1))


class _MiMicDPG(nn.Module):
    def __init__(self, channels: int, groups: int = 4) -> None:
        super().__init__()
        if channels % groups:
            raise ValueError("MiMics channels must be divisible by groups")
        self.path_a = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )
        self.path_b = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply two parallel grouped-convolution paths.

        Args:
            x: (B, C, H, W) feature map.

        Returns:
            (B, C, H, W) fused feature map.
        """
        return self.fuse(torch.cat((self.path_a(x), self.path_b(x)), dim=1))


class MiMicsNet(nn.Module):
    """MiMics-Net-style multimodal grouped encoder/refinement decoder.

    Args:
        x: (B, 1, H, W) normalized grayscale image.

    Returns:
        Dictionary containing ``logits`` with shape (B, K, H, W).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        base_channels: int = 16,
        groups: int = 4,
        gabor_kernel_size: int = 9,
        gabor_sigma: float = 2.0,
        gabor_wavelength: float = 5.0,
        gabor_gamma: float = 0.5,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("MiMics-Net MBI stem expects one grayscale input channel")
        c = base_channels
        self.stem = _MBIStem(c, gabor_kernel_size, gabor_sigma, gabor_wavelength, gabor_gamma)
        self.enc1 = _MiMicDPG(c, groups)
        self.down1 = nn.Conv2d(c, 2 * c, 3, stride=2, padding=1, bias=False)
        self.enc2 = _MiMicDPG(2 * c, groups)
        self.down2 = nn.Conv2d(2 * c, 3 * c, 3, stride=2, padding=1, bias=False)
        self.enc3 = _MiMicDPG(3 * c, groups)
        self.down3 = nn.Conv2d(3 * c, 4 * c, 3, stride=2, padding=1, bias=False)
        self.enc4 = _MiMicDPG(4 * c, groups)
        self.bottleneck_down = nn.Sequential(
            nn.Conv2d(4 * c, 5 * c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(5 * c),
            nn.ReLU(inplace=False),
        )
        self.bottleneck = _MiMicDPG(5 * c, groups)
        self.up4 = nn.ConvTranspose2d(5 * c, 4 * c, 2, 2)
        self.dec4 = _MiMicDPG(4 * c, groups)
        self.up3 = nn.ConvTranspose2d(4 * c, 3 * c, 2, 2)
        self.dec3 = _MiMicDPG(3 * c, groups)
        self.up2 = nn.ConvTranspose2d(3 * c, 2 * c, 2, 2)
        self.dec2 = _MiMicDPG(2 * c, groups)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, 2)
        self.dec1 = _MiMicDPG(c, groups)
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Segment one frame.

        Args:
            x: (B, 1, H, W) normalized grayscale image.

        Returns:
            Dictionary with ``logits`` of shape (B, K, H, W).
        """
        x = self.stem(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        bottleneck = self.bottleneck(self.bottleneck_down(e4))
        d4 = self.dec4(self.up4(bottleneck) + e4)
        d3 = self.dec3(self.up3(d4) + e3)
        d2 = self.dec2(self.up2(d3) + e2)
        d1 = self.dec1(self.up1(d2) + e1)
        return {"logits": self.head(d1)}
