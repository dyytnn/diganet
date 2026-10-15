"""Anti-aliased spatial downsampling."""

from __future__ import annotations

from torch import Tensor, nn

from . import register_module


@register_module("anti_aliased_downsampling")
class AntiAliasedDownsample(nn.Module):
    """Blur before stride-two downsampling to reduce phase-sensitive detail loss.

    Shape:
        Input ``(B, C, H, W)`` becomes ``(B, C, ceil(H/2), ceil(W/2))``.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        kernel = Tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        """Apply depthwise blur and stride-two sampling.

        Args:
            x: Feature map with shape ``(B, C, H, W)``.

        Returns:
            Downsampled feature map with shape ``(B, C, ceil(H/2), ceil(W/2))``.
        """
        kernel = self.kernel[:1].expand(x.shape[1], -1, -1, -1)
        return nn.functional.conv2d(x, kernel, stride=2, padding=1, groups=x.shape[1])
