"""SegFix-inspired boundary relabelling (model-agnostic post-processing baseline).

SegFix (Yuan et al., 2020) predicts, for every boundary pixel, the direction to a
reliable interior pixel and replaces the boundary label with the interior label.
This module attaches a boundary head and an 8-way direction head to the finest
decoder features, derives both targets from the ground truth (morphological
boundary; the first of eight 2-pixel offsets that lands on a non-boundary pixel of
the same class), and at inference copies the logits from the pixel reached by
walking ``offset_px`` along the predicted direction for ``steps`` iterations.
Inspired/proxy: the original trains a separate network on distance transforms;
here the heads share the segmentation decoder and directions are quantised to
eight neighbours.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_module

# 8 quantised directions as (dx, dy) pixel steps; axis-aligned first so that they win ties
# when deriving targets: E, W, N, S, NE, NW, SW, SE
DIRECTIONS = torch.tensor(
    [[1, 0], [-1, 0], [0, -1], [0, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]], dtype=torch.long
)
IGNORE_INDEX = -1


def _shift(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Return ``x`` sampled at (col + dx, row + dy) with edge replication; x is (B, ..., H, W)."""
    height, width = x.shape[-2:]
    rows = (torch.arange(height, device=x.device) + dy).clamp(0, height - 1)
    cols = (torch.arange(width, device=x.device) + dx).clamp(0, width - 1)
    return x[..., rows, :][..., :, cols]


def segfix_targets(target: torch.Tensor, offset_px: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive boundary and direction targets from integer labels.

    Args:
        target: (B, H, W) integer labels.
        offset_px: Step length used to look for a same-class interior pixel.

    Returns:
        boundary (B, 1, H, W) float in {0, 1}; direction (B, H, W) long in 0..7 for
        boundary pixels that have a valid interior neighbour, ``IGNORE_INDEX`` elsewhere.
    """
    labels = target.unsqueeze(1).float()
    local_max = F.max_pool2d(labels, 3, 1, 1)
    local_min = -F.max_pool2d(-labels, 3, 1, 1)
    boundary = (local_max - local_min > 0).float()
    direction = torch.full_like(target, IGNORE_INDEX)
    interior = boundary[:, 0] == 0
    for index, (dx, dy) in enumerate(DIRECTIONS.tolist()):
        shifted_label = _shift(target, dx * offset_px, dy * offset_px)
        shifted_interior = _shift(interior, dx * offset_px, dy * offset_px)
        valid = (
            (boundary[:, 0] > 0)
            & (direction == IGNORE_INDEX)
            & shifted_interior
            & (shifted_label == target)
        )
        direction = torch.where(valid, torch.full_like(direction, index), direction)
    return boundary, direction


@register_module("segfix_head")
class SegFixHead(nn.Module):
    """Boundary + 8-way direction heads with offset-based label relabelling.

    Args:
        in_channels: Channels of the finest decoder features.
        hidden_channels: Width of the shared branch.
        offset_px: Pixel step per relabelling iteration.
        steps: Number of relabelling iterations at inference.
        boundary_threshold: Sigmoid threshold selecting pixels to relabel.

    Input: features (B, C, H, W). ``forward`` returns boundary logits (B, 1, H, W) and
    direction logits (B, 8, H, W); ``refine`` rewrites dense logits (B, K, H, W).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        offset_px: int = 2,
        steps: int = 2,
        boundary_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if offset_px < 1 or steps < 1:
            raise ValueError("offset_px and steps must be >= 1")
        if not 0.0 < boundary_threshold < 1.0:
            raise ValueError("boundary_threshold must be in (0, 1)")
        self.offset_px = offset_px
        self.steps = steps
        self.boundary_threshold = boundary_threshold
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=False),
        )
        self.boundary_head = nn.Conv2d(hidden_channels, 1, 1)
        self.direction_head = nn.Conv2d(hidden_channels, len(DIRECTIONS), 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        branch = self.branch(features)
        return self.boundary_head(branch), self.direction_head(branch)

    @torch.no_grad()
    def refine(
        self, logits: torch.Tensor, boundary_logits: torch.Tensor, direction_logits: torch.Tensor
    ) -> torch.Tensor:
        """Copy logits from the interior pixel reached along the predicted direction."""
        batch, _, height, width = logits.shape
        device = logits.device
        rows = torch.arange(height, device=device).view(1, height, 1).expand(batch, height, width)
        cols = torch.arange(width, device=device).view(1, 1, width).expand(batch, height, width)
        is_boundary = torch.sigmoid(boundary_logits[:, 0]) > self.boundary_threshold
        direction = direction_logits.argmax(1)
        steps = DIRECTIONS.to(device)[direction] * self.offset_px  # (B, H, W, 2)
        src_rows, src_cols = rows.clone(), cols.clone()
        for _ in range(self.steps):
            candidate_cols = (src_cols + steps[..., 0]).clamp(0, width - 1)
            candidate_rows = (src_rows + steps[..., 1]).clamp(0, height - 1)
            still_boundary = is_boundary[
                torch.arange(batch, device=device).view(-1, 1, 1), src_rows, src_cols
            ]
            src_cols = torch.where(still_boundary, candidate_cols, src_cols)
            src_rows = torch.where(still_boundary, candidate_rows, src_rows)
        flat_index = (
            (src_rows * width + src_cols).view(batch, 1, -1).expand(-1, logits.shape[1], -1)
        )
        gathered = torch.gather(logits.reshape(batch, logits.shape[1], -1), 2, flat_index)
        gathered = gathered.view_as(logits)
        return torch.where(is_boundary.unsqueeze(1), gathered, logits)
