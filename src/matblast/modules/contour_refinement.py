"""Polar contour refinement family: Deep Snake, SharpContour and PolyTransform variants.

All three stages share one representation: every class of interest is described,
per image, by an outer and an inner radius profile around the centroid of its
predicted region (N angles), i.e. a star-shaped ring/blob. Contour vertices are
initialised from the dense prediction by ray casting, features are sampled at the
vertices, and a variant-specific network deforms the radii:

* ``snake``  - Deep Snake style: circular 1-D convolutions over the vertex sequence
  predict radial offsets, applied for several iterations.
* ``sharp``  - SharpContour style: a point classifier looks just inside/outside each
  vertex and moves it towards the boundary in small steps.
* ``poly``   - PolyTransform style: a small transformer over the vertex sequence
  predicts one geometric deformation of the polygon.

The refined contours are rasterised and used to bias the dense logits. Training
supervises the radii against ray-cast ground-truth radii (smooth L1).

LIMITATION (stated per AGENTS.md rule 14): the polar representation assumes each
class region is star-shaped around its centroid. Hatching or zona-ablated embryos,
fragmented TE and nested ICM inside TE violate this; these stages are therefore
geometry-oriented *baselines*, inspired by the cited methods, not faithful
reproductions of their instance-level pipelines.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from . import register_module
from .point_rend import point_sample

VARIANTS = ("snake", "sharp", "poly")


def region_centroids(probability: torch.Tensor) -> torch.Tensor:
    """Probability-weighted centroids: (B, C, H, W) -> (B, C, 2) normalised (x, y)."""
    batch, classes, height, width = probability.shape
    ys = (torch.arange(height, device=probability.device, dtype=probability.dtype) + 0.5) / height
    xs = (torch.arange(width, device=probability.device, dtype=probability.dtype) + 0.5) / width
    mass = probability.sum((2, 3)).clamp_min(1e-6)
    cx = (probability.sum(2) * xs).sum(-1) / mass
    cy = (probability.sum(3) * ys).sum(-1) / mass
    return torch.stack((cx, cy), dim=-1)


def ray_coords(centers: torch.Tensor, angles: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    """(B, C, 2), (N,), (B, C, N, R) radii -> (B, C, N, R, 2) normalised coordinates."""
    direction = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)  # (N, 2)
    offsets = radii.unsqueeze(-1) * direction.view(1, 1, -1, 1, 2)
    return (centers.view(*centers.shape[:2], 1, 1, 2) + offsets).clamp(0.0, 1.0)


def cast_radii(
    mask: torch.Tensor, centers: torch.Tensor, angles: torch.Tensor, steps: int, max_radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outer and inner radius profiles of ``mask`` (B, C, H, W) in {0,1} around ``centers``.

    Returns:
        (outer, inner) each (B, C, N) in normalised units; both 0 where the ray never hits.
    """
    batch, classes = mask.shape[:2]
    radii = torch.linspace(0.0, max_radius, steps, device=mask.device, dtype=mask.dtype)
    radii = radii.view(1, 1, 1, steps).expand(batch, classes, angles.numel(), steps)
    coords = ray_coords(centers, angles, radii).reshape(batch, -1, 2)
    hits = torch.stack(
        [
            point_sample(
                mask[:, c : c + 1],
                coords[:, c * angles.numel() * steps : (c + 1) * angles.numel() * steps],
                mode="nearest",
            )[:, 0]
            for c in range(classes)
        ],
        dim=1,
    ).view(batch, classes, angles.numel(), steps)
    hit = hits > 0.5
    any_hit = hit.any(-1)
    outer_index = torch.where(hit, radii, torch.full_like(radii, -1.0)).amax(-1)
    inner_index = torch.where(hit, radii, torch.full_like(radii, float("inf"))).amin(-1)
    outer = torch.where(any_hit, outer_index, torch.zeros_like(outer_index))
    inner = torch.where(any_hit, inner_index, torch.zeros_like(inner_index))
    return outer, inner


def rasterise(
    centers: torch.Tensor,
    angles: torch.Tensor,
    outer: torch.Tensor,
    inner: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """Ring masks (B, C, H, W) with ``inner(theta) <= r < outer(theta)``."""
    height, width = size
    batch, classes, count = outer.shape
    ys = (torch.arange(height, device=outer.device, dtype=outer.dtype) + 0.5) / height
    xs = (torch.arange(width, device=outer.device, dtype=outer.dtype) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    dx = grid_x.view(1, 1, height, width) - centers[..., 0].view(batch, classes, 1, 1)
    dy = grid_y.view(1, 1, height, width) - centers[..., 1].view(batch, classes, 1, 1)
    radius = torch.sqrt(dx * dx + dy * dy)
    theta = torch.atan2(dy, dx) % (2 * math.pi)
    index = (theta / (2 * math.pi) * count).long().clamp(0, count - 1)
    outer_at = torch.gather(
        outer.view(batch, classes, count, 1).expand(-1, -1, -1, height * width),
        2,
        index.view(batch, classes, 1, -1),
    ).view(batch, classes, height, width)
    inner_at = torch.gather(
        inner.view(batch, classes, count, 1).expand(-1, -1, -1, height * width),
        2,
        index.view(batch, classes, 1, -1),
    ).view(batch, classes, height, width)
    return ((radius >= inner_at) & (radius < outer_at)).to(outer.dtype)


class _CircConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B*, C, N) circular
        return self.conv(torch.cat((x[..., -self.pad :], x, x[..., : self.pad]), dim=-1))


@register_module("polar_contour_refiner")
class PolarContourRefiner(nn.Module):
    """Deform per-class polar contours with a Deep Snake / SharpContour / PolyTransform head.

    Args:
        in_channels_fine: Channels of the full-resolution decoder features.
        num_classes: K.
        variant: ``"snake"``, ``"sharp"`` or ``"poly"``.
        classes_of_interest: Classes represented by contours (ZP, TE, ICM by default).
        num_vertices: Angles N per contour.
        iterations: Deformation iterations (snake / sharp).
        hidden_channels: Width of the vertex network.
        step_px: Radial step per iteration in pixels (sharp) / max offset scale (snake, poly).
        ray_steps / max_radius: Ray-casting resolution and reach (normalised units).
        logit_boost: Additive logit bias applied inside/outside the rasterised contours.

    Input: fine (B, C, H, W), logits (B, K, H, W).
    Returns: dict with ``centers`` (B, C_k, 2), ``outer``/``inner`` (B, C_k, N) refined radii,
    ``init_outer``/``init_inner`` and ``logits`` (B, K, H, W) biased by the rasterised contours.
    """

    def __init__(
        self,
        in_channels_fine: int,
        num_classes: int,
        variant: str = "snake",
        classes_of_interest: Sequence[int] = (1, 2, 4),
        num_vertices: int = 64,
        iterations: int = 3,
        hidden_channels: int = 64,
        step_px: float = 2.0,
        ray_steps: int = 64,
        max_radius: float = 0.75,
        logit_boost: float = 2.0,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        classes = tuple(int(c) for c in classes_of_interest)
        if not classes or any(c < 0 or c >= num_classes for c in classes):
            raise ValueError(f"classes_of_interest must hold valid class ids, got {classes}")
        if num_vertices < 8 or iterations < 1 or ray_steps < 8 or not 0 < max_radius <= 1.5:
            raise ValueError("invalid num_vertices / iterations / ray_steps / max_radius")
        self.variant = variant
        self.classes = classes
        self.num_vertices = num_vertices
        self.iterations = iterations if variant != "poly" else 1
        self.step_px = float(step_px)
        self.ray_steps = ray_steps
        self.max_radius = float(max_radius)
        self.logit_boost = float(logit_boost)
        self.register_buffer(
            "angles",
            torch.arange(num_vertices, dtype=torch.float32) * (2 * math.pi / num_vertices),
            persistent=False,
        )
        in_channels = in_channels_fine + num_classes + 2  # features, logits, (radius, is_inner)
        self.embed = nn.Conv1d(in_channels, hidden_channels, 1)
        if variant == "poly":
            layer = nn.TransformerEncoderLayer(
                hidden_channels, 4, hidden_channels * 2, 0.0, activation="gelu", batch_first=True
            )
            self.body: nn.Module = nn.TransformerEncoder(layer, 2)
        else:
            self.body = nn.Sequential(
                _CircConv(hidden_channels),
                nn.ReLU(inplace=False),
                _CircConv(hidden_channels),
                nn.ReLU(inplace=False),
            )
        self.offset = nn.Conv1d(hidden_channels, 1, 1)

    def _deform(
        self,
        fine: torch.Tensor,
        logits: torch.Tensor,
        centers: torch.Tensor,
        radii: torch.Tensor,
        is_inner: bool,
    ) -> torch.Tensor:
        """One deformation step of a (B, C, N) radius profile."""
        batch, classes, count = radii.shape
        coords = ray_coords(centers, self.angles, radii.unsqueeze(-1)).reshape(batch, -1, 2)
        feats = torch.cat(
            (point_sample(fine, coords), point_sample(logits, coords)), dim=1
        )  # (B, C_f+K, C*N)
        extra = torch.stack(
            (radii.reshape(batch, -1), torch.full_like(radii.reshape(batch, -1), float(is_inner))),
            dim=1,
        )
        tokens = self.embed(torch.cat((feats, extra), dim=1))  # (B, hidden, C*N)
        tokens = (
            tokens.view(batch, -1, classes, count)
            .permute(0, 2, 1, 3)
            .reshape(batch * classes, -1, count)
        )
        if self.variant == "poly":
            hidden = self.body(tokens.transpose(1, 2)).transpose(1, 2)
        else:
            hidden = self.body(tokens)
        delta = self.offset(hidden).view(batch, classes, count)
        height, width = logits.shape[-2:]
        scale = self.step_px / min(height, width)
        if self.variant == "sharp":
            return (radii + scale * torch.tanh(delta)).clamp(0.0, self.max_radius)
        return (radii + scale * delta).clamp(0.0, self.max_radius)

    def forward(self, fine: torch.Tensor, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        probability = logits.softmax(1)[:, list(self.classes)]
        with torch.no_grad():
            centers = region_centroids(probability.detach())
            predicted = (
                logits.argmax(1, keepdim=True)
                == torch.tensor(self.classes, device=logits.device).view(1, -1, 1, 1)
            ).to(logits.dtype)
            init_outer, init_inner = cast_radii(
                predicted, centers, self.angles, self.ray_steps, self.max_radius
            )
        outer, inner = init_outer, init_inner
        for _ in range(self.iterations):
            outer = self._deform(fine, logits, centers, outer, is_inner=False)
            inner = self._deform(fine, logits, centers, inner, is_inner=True)
        inner = torch.minimum(inner, outer)
        masks = rasterise(
            centers, self.angles, outer.detach(), inner.detach(), tuple(logits.shape[-2:])
        )
        biased = logits.clone()
        for index, class_id in enumerate(self.classes):
            biased[:, class_id] = biased[:, class_id] + self.logit_boost * (
                2.0 * masks[:, index] - 1.0
            )
        return {
            "logits": biased,
            "centers": centers,
            "outer": outer,
            "inner": inner,
            "init_outer": init_outer,
            "init_inner": init_inner,
        }


def contour_target_radii(
    target: torch.Tensor,
    centers: torch.Tensor,
    classes: Sequence[int],
    angles: torch.Tensor,
    ray_steps: int,
    max_radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ground-truth (outer, inner) radii and a validity mask (class present) for the loss."""
    gt = torch.stack([(target == c) for c in classes], dim=1).to(centers.dtype)
    outer, inner = cast_radii(gt, centers, angles, ray_steps, max_radius)
    present = gt.flatten(2).sum(-1) > 0  # (B, C)
    return outer, inner, present
