"""DeepStrip-inspired strip refinement of boundary regions.

DeepStrip (Zhou et al., 2020) unfolds the region around a predicted boundary into
high-resolution strips that a dedicated network refines. This module keeps that
idea for semantic segmentation at full resolution: it selects uncertain boundary
pixels, builds a short strip through each pixel along the local boundary normal
(estimated from the probability gradient), refines the strip with 1-D
convolutions and rewrites the label of the strip centre. It is an *inspired*
adaptation, not a reproduction of the original crop-based pipeline; strips are
sampled with ``grid_sample`` rather than cropped and re-rendered.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_module
from .point_rend import (
    calculate_uncertainty,
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = _SOBEL_X.t()


def boundary_normals(
    logits: torch.Tensor, coords: torch.Tensor, classes_of_interest: Sequence[int]
) -> torch.Tensor:
    """Unit normals of the class-probability field at ``coords``.

    Args:
        logits: (B, K, H, W) dense logits.
        coords: (B, N, 2) normalised (x, y) points.
        classes_of_interest: Channels whose summed probability defines the boundary.

    Returns:
        (B, N, 2) unit vectors in normalised image units (dx/W, dy/H), defaulting to
        (1, 0) where the gradient vanishes.
    """
    probability = logits.softmax(1)[:, list(classes_of_interest)].sum(1, keepdim=True)
    kernel = torch.stack((_SOBEL_X, _SOBEL_Y)).unsqueeze(1).to(logits.dtype).to(logits.device)
    gradient = F.conv2d(F.pad(probability, (1, 1, 1, 1), mode="replicate"), kernel)
    sampled = point_sample(gradient, coords).transpose(1, 2)  # (B, N, 2) in pixel units
    norm = sampled.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(sampled)
    fallback[..., 0] = 1.0
    unit = torch.where(norm > 1e-6, sampled / norm.clamp_min(1e-6), fallback)
    return unit


@register_module("deep_strip_refiner")
class DeepStripRefiner(nn.Module):
    """Refine uncertain boundary pixels through normal-aligned strips.

    Args:
        in_channels_fine: Channels of the full-resolution decoder features.
        num_classes: K.
        num_strips: Strips sampled per image during training.
        strip_length: Odd number of samples along each strip.
        strip_spacing_px: Distance between consecutive strip samples in pixels.
        hidden_channels: Width of the strip network.
        classes_of_interest: Channels used for uncertainty and normals (ZP, TE, ICM).
        oversample_ratio / importance_sample_ratio: PointRend-style biased sampling.
        inference_num_strips: Strips refined per image at inference.

    Input: fine (B, C, H, W) features and dense logits (B, K, H, W).
    Training output: point logits (B, K, N) and coordinates (B, N, 2) of strip centres.
    Inference output: refined logits (B, K, H, W).
    """

    def __init__(
        self,
        in_channels_fine: int,
        num_classes: int,
        num_strips: int = 1024,
        strip_length: int = 9,
        strip_spacing_px: float = 1.0,
        hidden_channels: int = 64,
        classes_of_interest: Sequence[int] = (1, 2, 4),
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        inference_num_strips: int = 4096,
    ) -> None:
        super().__init__()
        if strip_length < 3 or strip_length % 2 == 0:
            raise ValueError(f"strip_length must be an odd number >= 3, got {strip_length}")
        if strip_spacing_px <= 0:
            raise ValueError(f"strip_spacing_px must be positive, got {strip_spacing_px}")
        if num_strips < 1 or inference_num_strips < 1:
            raise ValueError("num_strips and inference_num_strips must be >= 1")
        if oversample_ratio < 1.0 or not 0.0 <= importance_sample_ratio <= 1.0:
            raise ValueError("invalid oversample_ratio / importance_sample_ratio")
        classes = tuple(int(c) for c in classes_of_interest)
        if len(classes) < 2 or any(c < 0 or c >= num_classes for c in classes):
            raise ValueError(f"classes_of_interest must hold >= 2 valid class ids, got {classes}")
        self.num_classes = num_classes
        self.num_strips = num_strips
        self.strip_length = strip_length
        self.strip_spacing_px = float(strip_spacing_px)
        self.classes_of_interest = classes
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.inference_num_strips = inference_num_strips
        in_channels = in_channels_fine + num_classes
        self.strip_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=False),
        )
        self.predictor = nn.Sequential(
            nn.Conv1d(hidden_channels * 2 + num_classes, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv1d(hidden_channels, num_classes, kernel_size=1),
        )
        offsets = torch.arange(strip_length, dtype=torch.float32) - (strip_length - 1) / 2.0
        self.register_buffer("offsets", offsets * self.strip_spacing_px, persistent=False)

    def _strip_coords(
        self, coords: torch.Tensor, normals: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        """(B, N, 2), (B, N, 2) -> (B, N * L, 2) strip sample coordinates in [0, 1]."""
        height, width = size
        scale = coords.new_tensor([1.0 / width, 1.0 / height])
        steps = self.offsets.to(coords.dtype).view(1, 1, -1, 1)  # (1, 1, L, 1)
        strip = coords.unsqueeze(2) + steps * (normals * scale).unsqueeze(2)
        return strip.clamp(0.0, 1.0).reshape(coords.shape[0], -1, 2)

    def _refine(
        self, fine: torch.Tensor, logits: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Refined logits (B, K, N) for strip centres at ``coords``."""
        batch, count = coords.shape[:2]
        normals = boundary_normals(logits.detach(), coords, self.classes_of_interest)
        strip_coords = self._strip_coords(coords, normals, tuple(logits.shape[-2:]))
        features = torch.cat(
            (point_sample(fine, strip_coords), point_sample(logits, strip_coords)), dim=1
        )
        features = features.view(batch, features.shape[1], count, self.strip_length)
        encoded = self.strip_net(features)  # (B, hidden, N, L)
        centre = encoded[..., self.strip_length // 2]
        pooled = encoded.amax(dim=-1)
        centre_logits = point_sample(logits, coords)
        return self.predictor(torch.cat((centre, pooled, centre_logits), dim=1))

    def forward_train(
        self, fine: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = get_uncertain_point_coords_with_randomness(
            logits.detach(),
            self.num_strips,
            self.oversample_ratio,
            self.importance_sample_ratio,
            self.classes_of_interest,
        )
        return self._refine(fine, logits, coords), coords

    def forward_inference(self, fine: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        uncertainty = calculate_uncertainty(logits, self.classes_of_interest)
        indices, coords = get_uncertain_point_coords_on_grid(uncertainty, self.inference_num_strips)
        refined_points = self._refine(fine, logits, coords)
        batch, classes, height, width = logits.shape
        flat = logits.reshape(batch, classes, -1).clone()
        flat.scatter_(2, indices.unsqueeze(1).expand(-1, classes, -1), refined_points)
        return flat.view(batch, classes, height, width)
