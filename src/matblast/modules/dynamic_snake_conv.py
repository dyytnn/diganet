"""Dynamic Snake Convolution for thin and elongated structures."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import register_module


@register_module("dynamic_snake_conv")
class DynamicSnakeConv(nn.Module):
    """Apply input-conditioned, continuity-constrained axial sampling.

    This is a self-contained 2-D reimplementation of the Dynamic Snake
    Convolution idea.  It uses cumulative offsets along horizontal and
    vertical kernel paths, so neighboring samples move continuously rather
    than independently.  It is intentionally implemented with
    ``grid_sample`` to avoid a custom CUDA dependency.

    Args:
        channels: Number of input and output feature channels.
        kernel_size: Odd number of samples on each axial path.
        max_offset: Maximum normalized pixel displacement per path sample.

    Shape:
        Input and output are ``(B, C, H, W)``.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        max_offset: float = 2.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if max_offset <= 0:
            raise ValueError("max_offset must be positive")

        self.channels = channels
        self.kernel_size = kernel_size
        self.max_offset = float(max_offset)
        self.offset_head = nn.Conv2d(channels, 2 * kernel_size, 3, padding=1)
        self.projection = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _centered_path_offsets(delta: Tensor) -> Tensor:
        """Convert per-sample deltas to centered cumulative snake offsets."""
        cumulative = torch.cumsum(delta, dim=1)
        return cumulative - cumulative.mean(dim=1, keepdim=True)

    @staticmethod
    def _base_grid(height: int, width: int, x: Tensor) -> Tensor:
        """Create an ``align_corners=True`` grid with shape ``(H, W, 2)``."""
        y = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
        x_coords = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(y, x_coords, indexing="ij")
        return torch.stack((xx, yy), dim=-1)

    def _sample_axis(self, x: Tensor, offsets: Tensor, axis: int) -> Tensor:
        """Sample ``x`` along one axial path and aggregate kernel positions."""
        batch, _, height, width = x.shape
        base = self._base_grid(height, width, x).unsqueeze(0).expand(batch, -1, -1, -1)
        coordinates = base.unsqueeze(1).expand(-1, self.kernel_size, -1, -1, -1).clone()
        if axis == 0:  # horizontal path: move x coordinate
            scale = 2.0 / max(width - 1, 1)
            coordinates[..., 0] += offsets * scale
        else:  # vertical path: move y coordinate
            scale = 2.0 / max(height - 1, 1)
            coordinates[..., 1] += offsets * scale

        samples = F.grid_sample(
            x.unsqueeze(1).expand(-1, self.kernel_size, -1, -1, -1).reshape(
                batch * self.kernel_size, self.channels, height, width
            ),
            coordinates.reshape(batch * self.kernel_size, height, width, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        samples = samples.reshape(batch, self.kernel_size, self.channels, height, width)
        return samples.mean(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Refine a feature map with horizontal and vertical snake paths.

        Args:
            x: Feature map with shape ``(B, C, H, W)``.

        Returns:
            Residual feature map with shape ``(B, C, H, W)``.
        """
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"expected (B, {self.channels}, H, W), got {tuple(x.shape)}")
        raw_offsets = torch.tanh(self.offset_head(x)) * self.max_offset
        horizontal, vertical = raw_offsets.chunk(2, dim=1)
        horizontal = self._centered_path_offsets(horizontal)
        vertical = self._centered_path_offsets(vertical)
        sampled = 0.5 * self._sample_axis(x, horizontal, axis=0)
        sampled = sampled + 0.5 * self._sample_axis(x, vertical, axis=1)
        return x + self.residual_scale.tanh() * self.projection(sampled)
