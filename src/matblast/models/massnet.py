"""Table-1 reconstruction of the MASS-Net blastocyst segmentation model."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _ConvBNReLU(nn.Sequential):
    """Paper-specified 3x3 convolution followed by batch norm and ReLU."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )


class _DWSeparableConv(nn.Sequential):
    """Paper-specified 3x3 depthwise-separable convolution block."""

    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        out_channels = in_channels if out_channels is None else out_channels
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )


def _repeat_dwsc(channels: int, repetitions: int) -> nn.Sequential:
    """Build repeated DWSC blocks from the MASS-Net Table 1 specification."""
    if repetitions < 1:
        raise ValueError("DWSC repetitions must be positive")
    return nn.Sequential(*(_DWSeparableConv(channels) for _ in range(repetitions)))


class MASSNet(nn.Module):
    """MASS-Net reconstructed from the architecture table in Arsalan et al.

    The model uses an input block followed by Scale-8, Scale-2, Scale-4, and
    feature-booster branches. Their full-resolution outputs are channel-wise
    concatenated before a shallow progressive upsampling decoder.

    Args:
        x: (B, 1, H, W) globally L2-normalized grayscale image, normally
            H = W = 400 as in the paper.

    Returns:
        Dictionary with ``logits`` of shape (B, K, H, W), K = num_classes.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 5) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("MASS-Net expects one grayscale input channel")

        self.input_block = _ConvBNReLU(in_channels, 16, stride=2)

        self.s8_start = _ConvBNReLU(16, 32, stride=8, dilation=8)
        self.s8_dwsc = _repeat_dwsc(32, 8)
        self.s8_up_a = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, bias=False)
        self.s8_up_b = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, bias=False)
        self.s8_up_c = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, bias=False)

        # The paper table labels this stride as eight; its stated 100x100 output
        # and prose require stride two, which is used here.
        self.s2_start = _ConvBNReLU(16, 32, stride=2, dilation=2)
        self.s2_conv_a = _ConvBNReLU(32, 64)
        self.s2_conv_b = _ConvBNReLU(64, 64)
        self.s2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.s2_conv_c = _ConvBNReLU(64, 128)
        self.s2_conv_d = _ConvBNReLU(128, 256)
        self.s2_dwsc = _repeat_dwsc(256, 2)
        self.s2_up_a = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
        self.s2_up_b = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)

        self.s4_start = _ConvBNReLU(16, 32, stride=4, dilation=4)
        self.s4_conv_a = _ConvBNReLU(32, 64)
        self.s4_conv_b = _ConvBNReLU(64, 128)
        self.s4_dwsc_a = _repeat_dwsc(128, 2)
        self.s4_dwsc_b = nn.Sequential(_DWSeparableConv(128, 64), _DWSeparableConv(64))
        self.s4_up_a = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2, bias=False)
        self.s4_up_b = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)

        self.fbb = nn.Sequential(
            _ConvBNReLU(16, 32),
            _ConvBNReLU(32, 64),
            _ConvBNReLU(64, 64),
            _ConvBNReLU(64, 128),
        )
        self.decoder = nn.Sequential(
            _ConvBNReLU(288, 256),
            _ConvBNReLU(256, 128),
            _ConvBNReLU(128, 64),
        )
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=False)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Segment one frame with four-scale feature aggregation.

        Args:
            x: (B, 1, H, W) normalized grayscale image.

        Returns:
            Dictionary with ``logits`` of shape (B, K, H, W).
        """
        feature = self.input_block(x)

        s8 = self.s8_dwsc(self.s8_start(feature))
        s8 = self.s8_up_c(self.s8_up_b(self.s8_up_a(s8)))

        s2 = self.s2_conv_b(self.s2_conv_a(self.s2_start(feature)))
        s2 = self.s2_conv_d(self.s2_conv_c(self.s2_pool(s2)))
        s2 = self.s2_dwsc(s2)
        s2 = self.s2_up_b(self.s2_up_a(s2))

        s4 = self.s4_conv_b(self.s4_conv_a(self.s4_start(feature)))
        s4 = self.s4_dwsc_a(s4)
        s4 = self.s4_dwsc_b(s4)
        s4 = self.s4_up_b(self.s4_up_a(s4))

        aggregated = torch.cat((s8, s2, s4, self.fbb(feature)), dim=1)
        return {"logits": self.head(self.final_up(self.decoder(aggregated)))}
