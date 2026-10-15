"""Gate variants motivated by the gate-map analysis of DiGA-Net+ (paper Section 4.10).

The bottleneck gate of ``pointrend_refiner`` (stride 8) cannot resolve a 2--4 px boundary band and learns a
trophectoderm-texture prior instead of an uncertainty map. Two follow-ups are implemented here:

* :class:`EntropyGatedRefiner` -- the same residual refinement, but applied at full decoder resolution (``d1``)
  and gated by the *actual* predictive uncertainty (normalised entropy of the coarse ``aux2`` logits) instead of a
  learned gate. Tests the hypothesis "a gate at the right resolution and driven by real uncertainty fixes boundaries".
* the multi-scale texture gate is the existing :class:`~matblast.modules.literature_blocks.PointRendRefiner`
  instantiated at the ``dec3`` and ``dec2`` widths as well (wired in :class:`~matblast.models.MATBlastUNet`);
  it exploits what the bottleneck gate demonstrably learned at more scales.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import register_module


@register_module("entropy_gated_refinement")
class EntropyGatedRefiner(nn.Module):
    """Residual refinement of full-resolution decoder features gated by coarse predictive entropy.

    Args:
        channels: Channels of the decoder feature map to refine (``d1``).
        num_classes: Number of classes in the coarse logits (used to normalise the entropy to [0, 1]).
        hidden_channels: Width of the refinement branch; defaults to ``channels``.

    Input:
        features: (B, C, H, W) decoder features.
        coarse_logits: (B, K, h, w) logits of a coarser head; bilinearly resized to (H, W).

    Returns:
        (refined features (B, C, H, W), gate (B, 1, H, W) in [0, 1]).
    """

    def __init__(self, channels: int, num_classes: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        if channels <= 0 or num_classes <= 1:
            raise ValueError("channels must be positive and num_classes > 1")
        hidden = int(hidden_channels or channels)
        self.num_classes = int(num_classes)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, channels, 1)
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def gate(self, coarse_logits: Tensor, size: tuple[int, int]) -> Tensor:
        """Normalised entropy of the coarse prediction, resized to ``size``."""
        if coarse_logits.shape[1] != self.num_classes:
            raise ValueError(
                f"coarse logits have {coarse_logits.shape[1]} classes, expected {self.num_classes}"
            )
        p = coarse_logits.softmax(1)
        entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(1, keepdim=True) / math.log(
            self.num_classes
        )
        return F.interpolate(entropy, size=size, mode="bilinear", align_corners=False).clamp(
            0.0, 1.0
        )

    def forward(self, features: Tensor, coarse_logits: Tensor) -> tuple[Tensor, Tensor]:
        u = self.gate(coarse_logits.detach(), features.shape[-2:])
        return features + self.residual_scale.tanh() * (u * self.refine(features)), u
