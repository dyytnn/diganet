"""CascadePSP-lite: global coarse refinement followed by local high-resolution patches.

CascadePSP (Cheng et al., 2020) refines a coarse mask with one network applied in a
cascade: a global pass at reduced resolution, then local passes on high-resolution
crops. This lite semantic adaptation shares one small residual refinement CNN
(input: image + softmax of the current logits) between a global pass at
``global_scale`` and a local pass on ``num_patches`` fixed-size crops centred on the
most uncertain pixels; patch outputs are pasted back. Memory is bounded by the
patch size and count. Inspired/proxy: no multi-level pyramid of ground-truth
perturbations and no separately trained refinement network.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_module
from .point_rend import calculate_uncertainty


@register_module("cascade_psp_lite")
class CascadePSPLite(nn.Module):
    """Shared refinement CNN applied globally then on uncertain local patches.

    Args:
        in_channels: Image channels.
        num_classes: K.
        hidden_channels: Width of the refinement CNN.
        global_scale: Resolution factor of the global pass (e.g. 0.5).
        patch_size: Side of the square local crops (pixels, full resolution).
        num_patches: Crops per image in the local pass.
        classes_of_interest: Channels used for the uncertainty that places crops.

    Input: image (B, C, H, W), logits (B, K, H, W).
    Returns: (final logits (B, K, H, W), global-pass logits (B, K, H, W)).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 32,
        global_scale: float = 0.5,
        patch_size: int = 64,
        num_patches: int = 8,
        classes_of_interest: Sequence[int] = (1, 2, 4),
    ) -> None:
        super().__init__()
        if not 0.0 < global_scale <= 1.0:
            raise ValueError("global_scale must be in (0, 1]")
        if patch_size < 8 or num_patches < 0:
            raise ValueError("patch_size must be >= 8 and num_patches >= 0")
        classes = tuple(int(c) for c in classes_of_interest)
        if len(classes) < 2 or any(c < 0 or c >= num_classes for c in classes):
            raise ValueError(f"classes_of_interest must hold >= 2 valid class ids, got {classes}")
        self.global_scale = global_scale
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.classes_of_interest = classes
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + num_classes, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, num_classes, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _refine(self, image: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        update = self.net(torch.cat((image, logits.softmax(1)), dim=1))
        return logits + self.residual_scale.tanh() * update

    def _patch_origins(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, M, 2) integer (row, col) top-left corners of the most uncertain patches."""
        batch, _, height, width = logits.shape
        size = min(self.patch_size, height, width)
        pooled = F.avg_pool2d(
            calculate_uncertainty(logits, self.classes_of_interest), size, stride=size // 2
        )
        count = min(self.num_patches, pooled.shape[-1] * pooled.shape[-2])
        flat = pooled.reshape(batch, -1)
        tie = torch.arange(flat.shape[1], device=flat.device, dtype=flat.dtype) * -1e-7
        indices = (flat + tie).topk(count, dim=1).indices
        rows = torch.div(indices, pooled.shape[-1], rounding_mode="floor") * (size // 2)
        cols = (indices % pooled.shape[-1]) * (size // 2)
        rows = rows.clamp(max=height - size)
        cols = cols.clamp(max=width - size)
        return torch.stack((rows, cols), dim=-1)

    def forward(
        self, image: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = logits.shape[-2:]
        if self.global_scale < 1.0:
            small = (
                max(8, int(round(height * self.global_scale))),
                max(8, int(round(width * self.global_scale))),
            )
            small_image = F.interpolate(image, size=small, mode="bilinear", align_corners=False)
            small_logits = F.interpolate(logits, size=small, mode="bilinear", align_corners=False)
            global_logits = F.interpolate(
                self._refine(small_image, small_logits),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        else:
            global_logits = self._refine(image, logits)
        if self.num_patches == 0:
            return global_logits, global_logits
        origins = self._patch_origins(global_logits.detach())
        size = min(self.patch_size, height, width)
        refined = global_logits.clone()
        for b in range(logits.shape[0]):
            for m in range(origins.shape[1]):
                r, c = int(origins[b, m, 0]), int(origins[b, m, 1])
                patch_image = image[b : b + 1, :, r : r + size, c : c + size]
                patch_logits = global_logits[b : b + 1, :, r : r + size, c : c + size]
                refined[b : b + 1, :, r : r + size, c : c + size] = self._refine(
                    patch_image, patch_logits
                )
        return refined, global_logits
