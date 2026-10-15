"""Documented reconstruction of BLAST-Net's CAPP and DPSU design."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _ConvBNReLU(nn.Sequential):
    """Configurable convolution, batch normalization, and ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )


class _Bottleneck(nn.Module):
    """ResNet bottleneck used by the documented BLAST-Net encoder."""

    def __init__(
        self,
        in_channels: int,
        planes: int,
        expansion: int,
        stride: int,
        dilation: int,
    ) -> None:
        super().__init__()
        out_channels = planes * expansion
        self.reduce = _ConvBNReLU(in_channels, planes, kernel_size=1)
        self.spatial = _ConvBNReLU(planes, planes, kernel_size=3, stride=stride, dilation=dilation)
        self.expand = nn.Sequential(
            nn.Conv2d(planes, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply one residual bottleneck.

        Args:
            x: (B, C, H, W) encoder feature map.

        Returns:
            (B, expansion * planes, H / stride, W / stride) feature map.
        """
        return self.activation(self.expand(self.spatial(self.reduce(x))) + self.shortcut(x))


class _ResNetEncoder(nn.Module):
    """Configurable ResNet-50-style encoder with optional final dilation."""

    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        stem_kernel_size: int,
        stem_stride: int,
        pool_kernel_size: int,
        pool_stride: int,
        stage_planes: Sequence[int],
        stage_blocks: Sequence[int],
        stage_strides: Sequence[int],
        stage_dilations: Sequence[int],
        bottleneck_expansion: int,
    ) -> None:
        super().__init__()
        lengths = {
            len(stage_planes),
            len(stage_blocks),
            len(stage_strides),
            len(stage_dilations),
        }
        if len(lengths) != 1 or not stage_planes:
            raise ValueError("BLAST-Net encoder stage settings must have one nonzero common length")
        self.stem = _ConvBNReLU(
            in_channels, stem_channels, kernel_size=stem_kernel_size, stride=stem_stride
        )
        self.pool = nn.MaxPool2d(
            pool_kernel_size, stride=pool_stride, padding=pool_kernel_size // 2
        )
        current_channels = stem_channels
        stages: list[nn.Module] = []
        for planes, blocks, stride, dilation in zip(
            stage_planes, stage_blocks, stage_strides, stage_dilations, strict=True
        ):
            if blocks < 1:
                raise ValueError("BLAST-Net encoder requires at least one block per stage")
            layers: list[nn.Module] = [
                _Bottleneck(
                    current_channels,
                    planes,
                    bottleneck_expansion,
                    stride,
                    dilation,
                )
            ]
            current_channels = planes * bottleneck_expansion
            layers.extend(
                _Bottleneck(current_channels, planes, bottleneck_expansion, 1, dilation)
                for _ in range(blocks - 1)
            )
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.out_channels = current_channels

    def forward(self, x: Tensor) -> Tensor:
        """Encode an image.

        Args:
            x: (B, C, H, W) normalized input image.

        Returns:
            (B, C_enc, H / output_stride, W / output_stride) feature map.
        """
        return self.stages(self.pool(self.stem(x)))


class _CAPP(nn.Module):
    """Cascaded atrous pyramid pooling with dense final aggregation."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        rates: Sequence[int],
        kernel_size: int,
    ) -> None:
        super().__init__()
        if not rates or any(rate < 1 for rate in rates):
            raise ValueError("CAPP dilation rates must be positive")
        self.project = _ConvBNReLU(in_channels, channels, kernel_size=1)
        self.layers = nn.ModuleList(
            _ConvBNReLU(channels, channels, kernel_size=kernel_size, dilation=rate)
            for rate in rates
        )
        self.fuse = _ConvBNReLU(channels * (len(rates) + 1), channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Aggregate cascaded dilated context.

        Args:
            x: (B, C_enc, H, W) encoder feature map.

        Returns:
            (B, C_capp, H, W) multi-scale contextual feature map.
        """
        current = self.project(x)
        features = [current]
        for layer in self.layers:
            current = layer(current)
            features.append(current)
        return self.fuse(torch.cat(features, dim=1))


class _DensePixelShuffleDecoder(nn.Module):
    """Dense progressive sub-pixel upsampling decoder (DPSU reconstruction)."""

    def __init__(
        self,
        in_channels: int,
        stage_channels: Sequence[int],
        upsample_factor: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if not stage_channels:
            raise ValueError("DPSU needs at least one decoder stage")
        if upsample_factor < 2:
            raise ValueError("DPSU upsample_factor must be at least two")
        self.upsample_factor = upsample_factor
        blocks: list[nn.Module] = []
        prior_channels = in_channels
        for channels in stage_channels:
            blocks.append(
                _ConvBNReLU(
                    prior_channels,
                    channels * upsample_factor**2,
                    kernel_size=kernel_size,
                )
            )
            prior_channels += channels
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = stage_channels[-1]

    def forward(self, x: Tensor) -> Tensor:
        """Decode with dense, progressive PixelShuffle stages.

        Args:
            x: (B, C_capp, H, W) CAPP output.

        Returns:
            (B, C_last, H * scale**S, W * scale**S) decoder feature map.
        """
        features = [x]
        resolution = x.shape[-2:]
        for block in self.blocks:
            dense = torch.cat(
                [
                    feature
                    if feature.shape[-2:] == resolution
                    else F.interpolate(
                        feature, size=resolution, mode="bilinear", align_corners=False
                    )
                    for feature in features
                ],
                dim=1,
            )
            upsampled = F.pixel_shuffle(block(dense), self.upsample_factor)
            features.append(upsampled)
            resolution = upsampled.shape[-2:]
        return features[-1]


class BlastNet(nn.Module):
    """BLAST-Net documented reconstruction with ResNet, CAPP, and DPSU.

    This is a paper-derived reconstruction, not an official implementation. The
    original publication identifies a ResNet-50 encoder, CAPP, and DPSU but
    does not disclose every decoder width and optimization choice. Every such
    choice is supplied by the experiment config.

    Args:
        x: (B, C, H, W) normalized input image.

    Returns:
        Dictionary with ``logits`` of shape (B, K, H, W).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        stem_channels: int = 64,
        stem_kernel_size: int = 7,
        stem_stride: int = 2,
        pool_kernel_size: int = 3,
        pool_stride: int = 2,
        stage_planes: Sequence[int] = (64, 128, 256, 512),
        stage_blocks: Sequence[int] = (3, 4, 6, 3),
        stage_strides: Sequence[int] = (1, 2, 2, 1),
        stage_dilations: Sequence[int] = (1, 1, 1, 2),
        bottleneck_expansion: int = 4,
        capp_channels: int = 256,
        capp_rates: Sequence[int] = (1, 2, 4, 8, 16),
        capp_kernel_size: int = 3,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        upsample_factor: int = 2,
        decoder_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = _ResNetEncoder(
            in_channels,
            stem_channels,
            stem_kernel_size,
            stem_stride,
            pool_kernel_size,
            pool_stride,
            stage_planes,
            stage_blocks,
            stage_strides,
            stage_dilations,
            bottleneck_expansion,
        )
        self.capp = _CAPP(self.encoder.out_channels, capp_channels, capp_rates, capp_kernel_size)
        self.decoder = _DensePixelShuffleDecoder(
            capp_channels, decoder_channels, upsample_factor, decoder_kernel_size
        )
        self.head = nn.Conv2d(self.decoder.out_channels, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Segment one blastocyst frame.

        Args:
            x: (B, C, H, W) normalized image.

        Returns:
            Dictionary with ``logits`` of shape (B, K, H, W).
        """
        logits = self.head(self.decoder(self.capp(self.encoder(x))))
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits}
