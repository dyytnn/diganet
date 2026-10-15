"""Faithful PointRend head (Kirillov et al., 2020) for semantic segmentation.

Training: sample points biased towards uncertain locations of a coarse prediction,
gather fine-grained features and coarse logits at those points, and classify them
with a small point-wise MLP. Inference: iteratively upsample the coarse prediction
and replace the ``N`` most uncertain points with MLP predictions.

Uncertainty is the negative margin between the two highest scores; it can be
restricted to a subset of classes (here ZP/TE/ICM) so that sampling focuses on the
anatomically difficult boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_module


def point_sample(
    features: torch.Tensor, coords: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    """Sample ``features`` at normalised point coordinates.

    Args:
        features: (B, C, H, W) feature map.
        coords: (B, P, 2) points as (x, y) in ``[0, 1]``.
        mode: ``grid_sample`` interpolation mode.

    Returns:
        (B, C, P) sampled values.
    """
    grid = (2.0 * coords - 1.0).unsqueeze(2)  # (B, P, 1, 2)
    sampled = F.grid_sample(features, grid, mode=mode, padding_mode="border", align_corners=False)
    return sampled.squeeze(3)


def calculate_uncertainty(
    logits: torch.Tensor, classes_of_interest: Sequence[int] | None = None
) -> torch.Tensor:
    """Negative top-1/top-2 margin; larger values mean more uncertain.

    Args:
        logits: (B, K, ...) class scores.
        classes_of_interest: Channel indices to consider; ``None`` uses all classes.

    Returns:
        (B, 1, ...) uncertainty map.
    """
    selected = logits if classes_of_interest is None else logits[:, list(classes_of_interest)]
    if selected.shape[1] < 2:
        raise ValueError("uncertainty needs at least two class channels")
    top2 = selected.topk(2, dim=1).values
    return top2[:, 1:2] - top2[:, 0:1]


def get_uncertain_point_coords_with_randomness(
    coarse_logits: torch.Tensor,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
    classes_of_interest: Sequence[int] | None = None,
) -> torch.Tensor:
    """PointRend training-time sampling (biased towards uncertain points).

    Returns:
        (B, num_points, 2) coordinates in ``[0, 1]``.
    """
    batch = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    candidates = torch.rand(batch, num_sampled, 2, device=coarse_logits.device)
    with torch.no_grad():
        candidate_logits = point_sample(coarse_logits, candidates)
        uncertainty = calculate_uncertainty(candidate_logits, classes_of_interest)[:, 0]
    num_uncertain = int(importance_sample_ratio * num_points)
    num_random = num_points - num_uncertain
    top_index = uncertainty.topk(num_uncertain, dim=1).indices  # (B, num_uncertain)
    chosen = torch.gather(candidates, 1, top_index.unsqueeze(-1).expand(-1, -1, 2))
    if num_random > 0:
        chosen = torch.cat((chosen, torch.rand(batch, num_random, 2, device=chosen.device)), dim=1)
    return chosen


def get_uncertain_point_coords_on_grid(
    uncertainty: torch.Tensor, num_points: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the ``num_points`` most uncertain pixels of a dense map.

    Args:
        uncertainty: (B, 1, H, W).
        num_points: Points per image (capped at H*W).

    Returns:
        Flat indices (B, N) and coordinates (B, N, 2) of the pixel centres.
    """
    batch, _, height, width = uncertainty.shape
    count = min(height * width, num_points)
    flat = uncertainty.reshape(batch, -1)
    # Deterministic tie-break: earlier pixels win when scores are equal.
    tie_break = torch.arange(flat.shape[1], device=flat.device, dtype=flat.dtype) * -1e-7
    indices = (flat + tie_break).topk(count, dim=1).indices
    xs = (indices % width).to(flat.dtype) + 0.5
    ys = torch.div(indices, width, rounding_mode="floor").to(flat.dtype) + 0.5
    coords = torch.stack((xs / width, ys / height), dim=-1)
    return indices, coords


def sample_point_labels(target: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour labels at ``coords``: (B, H, W) long -> (B, P) long."""
    return point_sample(target.unsqueeze(1).float(), coords, mode="nearest").squeeze(1).long()


@register_module("point_rend_head")
class PointRendHead(nn.Module):
    """Point-wise refinement MLP with PointRend sampling and subdivision inference.

    Args:
        in_channels_fine: Channels of the fine-grained feature map.
        num_classes: K.
        num_points: Points sampled per image during training.
        oversample_ratio: Candidate multiplier for importance sampling (>= 1).
        importance_sample_ratio: Fraction of ``num_points`` taken from the most
            uncertain candidates; the rest is uniform.
        mlp_hidden: Hidden width of the point MLP.
        mlp_layers: Number of hidden layers; coarse logits are re-concatenated
            before every layer as in the reference implementation.
        classes_of_interest: Channels used for uncertainty (default ZP, TE, ICM).
        num_subdivision_steps: Inference refinement iterations.
        subdivision_num_points: Points refined per inference step.

    Input (training): fine (B, C, H, W), coarse_logits (B, K, h, w).
    Output (training): point logits (B, K, P) and coordinates (B, P, 2).
    Output (inference): refined logits (B, K, H, W).
    """

    def __init__(
        self,
        in_channels_fine: int,
        num_classes: int,
        num_points: int = 2048,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        mlp_hidden: int = 256,
        mlp_layers: int = 3,
        classes_of_interest: Sequence[int] = (1, 2, 4),
        num_subdivision_steps: int = 2,
        subdivision_num_points: int = 8192,
    ) -> None:
        super().__init__()
        if oversample_ratio < 1.0:
            raise ValueError(f"oversample_ratio must be >= 1, got {oversample_ratio}")
        if not 0.0 <= importance_sample_ratio <= 1.0:
            raise ValueError(
                f"importance_sample_ratio must be in [0, 1], got {importance_sample_ratio}"
            )
        if (
            mlp_layers < 1
            or num_points < 1
            or subdivision_num_points < 1
            or num_subdivision_steps < 0
        ):
            raise ValueError(
                "num_points, mlp_layers, subdivision_num_points must be >= 1 and steps >= 0"
            )
        classes = tuple(int(c) for c in classes_of_interest)
        if len(classes) < 2 or any(c < 0 or c >= num_classes for c in classes):
            raise ValueError(f"classes_of_interest must hold >= 2 valid class ids, got {classes}")
        self.num_classes = num_classes
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.classes_of_interest = classes
        self.num_subdivision_steps = num_subdivision_steps
        self.subdivision_num_points = subdivision_num_points
        layers = []
        in_channels = in_channels_fine + num_classes
        for _ in range(mlp_layers):
            layers.append(nn.Conv1d(in_channels, mlp_hidden, kernel_size=1))
            in_channels = mlp_hidden + num_classes
        self.hidden_layers = nn.ModuleList(layers)
        self.predictor = nn.Conv1d(mlp_hidden, num_classes, kernel_size=1)

    def _mlp(self, fine_points: torch.Tensor, coarse_points: torch.Tensor) -> torch.Tensor:
        x = torch.cat((fine_points, coarse_points), dim=1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x), inplace=False)
            x = torch.cat((x, coarse_points), dim=1)
        return self.predictor(x[:, : -self.num_classes])

    def forward_train(
        self, fine: torch.Tensor, coarse_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample uncertain points and classify them (training path)."""
        coords = get_uncertain_point_coords_with_randomness(
            coarse_logits,
            self.num_points,
            self.oversample_ratio,
            self.importance_sample_ratio,
            self.classes_of_interest,
        )
        point_logits = self._mlp(point_sample(fine, coords), point_sample(coarse_logits, coords))
        return point_logits, coords

    def forward_inference(
        self, fine: torch.Tensor, coarse_logits: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        """Iterative subdivision refinement to ``output_size`` (inference path)."""
        logits = coarse_logits
        for _ in range(self.num_subdivision_steps):
            if tuple(logits.shape[-2:]) != tuple(output_size):
                target = (
                    min(logits.shape[-2] * 2, output_size[0]),
                    min(logits.shape[-1] * 2, output_size[1]),
                )
                logits = F.interpolate(logits, size=target, mode="bilinear", align_corners=False)
            uncertainty = calculate_uncertainty(logits, self.classes_of_interest)
            indices, coords = get_uncertain_point_coords_on_grid(
                uncertainty, self.subdivision_num_points
            )
            point_logits = self._mlp(point_sample(fine, coords), point_sample(logits, coords))
            batch, classes, height, width = logits.shape
            flat = logits.reshape(batch, classes, -1).clone()
            flat.scatter_(2, indices.unsqueeze(1).expand(-1, classes, -1), point_logits)
            logits = flat.view(batch, classes, height, width)
        if tuple(logits.shape[-2:]) != tuple(output_size):
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits
